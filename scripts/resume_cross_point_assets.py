"""Resume pinned asset downloads with fresh connections and bounded recovery."""
import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import time
import zipfile
from pathlib import Path

import requests

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def completed(asset):
    path = Path(asset['path'])
    if not path.exists() or Path(str(path)+'.aria2').exists():
        return False
    if asset.get('bytes') and path.stat().st_size != asset['bytes']:
        return False
    if asset.get('sha256'):
        return digest(path) == asset['sha256']
    if asset['kind'] == 'source':
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    # Existing files originate from this task's pinned aria2 queue. Without an
    # LFS publisher hash, retain the weaker size/control-file completion evidence.
    if path.suffix == '.json':
        try:
            json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            return False
    return path.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--stop-pids', type=int, nargs=2)
    args = ap.parse_args()
    base = args.base.resolve()
    manifest = json.loads((base/'logs/asset-manifest.json').read_text())
    assets = manifest['assets']
    for asset in assets:
        if not Path(asset['path']).resolve().is_relative_to(base):
            raise ValueError('Asset path outside task directory')
    if args.stop_pids:
        coordinator, downloader = args.stop_pids
        checks = [(coordinator, ['prepare_cross_point_probe_assets.py', 'resume_cross_point_assets.py']),
                  (downloader, ['model-aria2-queue.private.txt', 'model-resume-queue.private.txt'])]
        for pid, expected in checks:
            command = Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode()
            if str(base) not in command or not any(name in command for name in expected):
                raise ValueError('Process identity mismatch')
        os.kill(downloader, signal.SIGINT)  # aria2 persists its resume control files.
        deadline = time.monotonic()+20
        while Path('/proc/'+str(downloader)).exists() and time.monotonic()<deadline:
            time.sleep(.2)
        if Path('/proc/'+str(downloader)).exists():
            raise RuntimeError('Original downloader did not exit; refusing second writer')
        if Path('/proc/'+str(coordinator)).exists():
            os.kill(coordinator, signal.SIGTERM)
        print('ORIGINAL_DOWNLOAD_STOPPED_WITH_RESUME_PRESERVED', flush=True)
    expected = sum(a.get('bytes') or 0 for a in assets)
    allocated = sum(Path(a['path']).stat().st_blocks*512 for a in assets if Path(a['path']).exists())
    if shutil.disk_usage(base).free < max(0, expected-allocated)+3*1024**3:
        raise ValueError('Insufficient data disk reserve')
    env = os.environ.copy()
    for name in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
        env.pop(name, None)
    pending = [a for a in assets if not completed(a)]
    queue = base/'logs/model-resume-queue.private.txt'
    recovery = []
    rpc_secret = secrets.token_hex(32)
    with socket.socket() as socket_probe:
        socket_probe.bind(('127.0.0.1', 0))
        rpc_port = socket_probe.getsockname()[1]
    rpc_config = base/'logs/aria2-rpc.private.conf'
    rpc_config.write_text('enable-rpc=true\nrpc-listen-all=false\nrpc-listen-port='+str(rpc_port)
                          +'\nrpc-secret='+rpc_secret+'\n')
    rpc_config.chmod(0o600)
    rpc_session = requests.Session()
    rpc_session.trust_env = False
    reconnect_events = []

    def rpc(method, *parameters):
        response = rpc_session.post('http://127.0.0.1:'+str(rpc_port)+'/jsonrpc',
            json={'jsonrpc': '2.0', 'id': 'download-monitor', 'method': 'aria2.'+method,
                  'params': ['token:'+rpc_secret, *parameters]}, timeout=5)
        response.raise_for_status()
        body = response.json()
        if 'error' in body:
            raise RuntimeError('aria2 RPC method failed: '+method)
        return body['result']

    for attempt in range(3):
        if not pending:
            break
        blocks = []
        url_by_path = {}
        for asset in pending:
            path = Path(asset['path'])
            if asset['kind'] == 'source':
                url = 'https://codeload.github.com/brains-bots-n-behavior/motionforesight/zip/'+manifest['repo_revision']
            else:
                repo = asset['kind']
                if repo == 'trackcraft3r/checkpoint':
                    filename = 'model.safetensors'
                elif repo == 'Wan-AI/Wan2.1-T2V-1.3B':
                    filename = str(path.relative_to(base/'weights'/repo))
                elif repo == 'depth-anything/DA3-BASE':
                    filename = path.name
                else:
                    raise ValueError('Unknown pinned repository')
                url = 'https://hf-mirror.com/'+repo+'/resolve/'+asset['revision']+'/'+filename
                url += '?download=true&route_refresh='+str(time.time_ns())
            block = url+'\n  dir='+str(path.parent)+'\n  out='+path.name+'\n'
            if asset.get('sha256'):
                block += '  checksum=sha-256='+asset['sha256']+'\n'
            blocks.append(block)
            url_by_path[str(path)] = url
        queue.write_text(''.join(blocks))
        queue.chmod(0o600)
        with (base/'logs/models-aria2.log').open('a') as output:
            output.write('\nRESUME_ATTEMPT_'+str(attempt+1)+'\n')
            output.flush()
            command = ['aria2c', '--conf-path='+str(rpc_config), '-c', '-j', '3', '-x', '16', '-s', '16', '--min-split-size=1M',
                       '--file-allocation=none', '--console-log-level=error', '--summary-interval=15',
                       '--max-tries=12', '--retry-wait=3', '--connect-timeout=15', '--timeout=60',
                       '--auto-save-interval=15', '--auto-file-renaming=false',
                       '--allow-overwrite=false', '-i', str(queue)]
            worker = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            (base/'logs/resume-process.json').write_text(json.dumps({
                'coordinator_pid': os.getpid(), 'aria2_pid': worker.pid,
                'attempt': attempt+1, 'pending_count': len(pending),
                'connections_per_file': 16, 'gpu_used': False}))
            low_since = {}
            reconnect_counts = {}
            while worker.poll() is None:
                time.sleep(10)
                try:
                    totals = rpc('getGlobalStat')
                    if (int(totals['numActive']) == 0 and int(totals['numWaiting']) == 0
                            and int(totals['numStoppedTotal']) >= len(pending)):
                        rpc('shutdown')  # RPC mode otherwise stays alive after downloads finish.
                        continue
                    active = rpc('tellActive', ['gid', 'connections', 'downloadSpeed', 'files'])
                    for download in active:
                        path = download['files'][0]['path']
                        gid = download['gid']
                        slow = int(download['connections']) <= 8 and int(download['downloadSpeed']) < 512*1024
                        if not slow:
                            low_since.pop(gid, None)
                            continue
                        low_since.setdefault(gid, time.monotonic())
                        if time.monotonic()-low_since[gid] < 90 or reconnect_counts.get(path, 0) >= 4:
                            continue
                        rpc('forcePause', gid)
                        paused = False
                        for _ in range(20):
                            if rpc('tellStatus', gid, ['status'])['status'] == 'paused':
                                paused = True
                                break
                            time.sleep(.1)
                        if not paused:
                            raise RuntimeError('aria2 did not pause before URI refresh')
                        original = url_by_path[path].split('?')[0]
                        fresh_url = original+'?download=true&route_refresh='+str(time.time_ns())
                        old_uris = [row['uri'] for row in download['files'][0]['uris']]
                        rpc('changeUri', gid, 1, old_uris, [fresh_url])
                        rpc('unpause', gid)
                        reconnect_counts[path] = reconnect_counts.get(path, 0)+1
                        low_since.pop(gid, None)
                        reconnect_events.append({'filename': Path(path).name, 'reconnect_number': reconnect_counts[path],
                                                 'previous_connections': int(download['connections']),
                                                 'previous_bytes_s': int(download['downloadSpeed'])})
                        (base/'logs/connection-recoveries.json').write_text(json.dumps(reconnect_events, indent=2))
                        print('REFRESHED_DOWNLOAD_CONNECTIONS_'+Path(path).name, flush=True)
                except (requests.RequestException, RuntimeError, KeyError) as error:
                    # Never echo RPC secrets, proxy strings, or signed URLs.
                    print('DOWNLOAD_MONITOR_'+type(error).__name__, flush=True)
            status = worker.returncode
        recovery.append({'attempt': attempt+1, 'exit_code': status})
        pending = [a for a in assets if not completed(a)]
        (base/'logs/resume-history.json').write_text(json.dumps(recovery, indent=2))
        if pending:
            print('DOWNLOAD_ATTEMPT_ENDED_PENDING_'+str(len(pending)), flush=True)
            time.sleep(3)
    if pending:
        raise RuntimeError('Bounded retries exhausted; remaining assets: '+str(len(pending)))
    rows = []
    for asset in assets:
        path = Path(asset['path'])
        sha = digest(path)
        if asset.get('sha256') and sha != asset['sha256']:
            raise ValueError('Publisher SHA256 mismatch')
        if asset.get('bytes') and path.stat().st_size != asset['bytes']:
            raise ValueError('Publisher size mismatch')
        rows.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha,
                     'official_sha256_available': bool(asset.get('sha256'))})
    destination = base/'source'
    with zipfile.ZipFile(destination/'motionforesight.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('Source ZIP CRC failure')
        for item in archive.infolist():
            if not (destination/item.filename).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive member')
        archive.extractall(destination)
    fine = json.loads((base/'logs/fine-checkpoint-complete.json').read_text())
    fine_ready = bool(fine['fine_checkpoint_files_ready'] and fine['local_remote_sha256_match'] and fine['zip_crc_pass'])
    for name in fine['extracted_files']:
        fine_ready = fine_ready and (base/'weights/motionforesight'/name).is_file()
    result = {'assets_ready': fine_ready, 'source': str(destination/('motionforesight-'+manifest['repo_revision'])),
              'files': rows, 'missing_assets': [] if fine_ready else ['MotionForesight_finetuned'],
              'source_zip_crc_pass': True, 'fine_archive_crc_pass': bool(fine['zip_crc_pass']),
              'model_loaded': False, 'forward_called': False, 'gpu_used': False}
    (base/'logs/assets-complete.json').write_text(json.dumps(result, indent=2))
    print('ASSETS_VERIFIED_NO_MODEL_FORWARD', flush=True)


if __name__ == '__main__':
    main()
