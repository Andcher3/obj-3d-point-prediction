"""Aggregate CPU evidence only; this status authorizes no GPU action."""
import argparse
import datetime
import json
import shutil
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--publisher-metadata', type=Path)
    ap.add_argument('--fetch-publisher-metadata', type=Path)
    args = ap.parse_args()
    if args.fetch_publisher_metadata:
        url = 'https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B/revision/37ec512624d61f7aa208f7ea8140a131f93afc9a?blobs=true'
        with urllib.request.urlopen(url, timeout=30) as response:
            metadata = json.load(response)
        args.fetch_publisher_metadata.write_text(json.dumps(metadata, indent=2))
        print('PUBLIC_SMALL_PUBLISHER_METADATA_SAVED')
        return
    base = args.base.resolve()
    logs = base / 'logs'
    read = lambda name: json.loads((logs / name).read_text())
    assets = read('assets-complete.json')
    fine = read('fine-checkpoint-complete.json')
    data = read('preparation-complete.json')
    env = read('cpu-environment-interface-check.json')
    inputs = read('cpu-input-interface-audit.json')
    da3 = read('da3-source-complete.json')
    tests = read('cpu-preparation-tests.json')
    local = read('query-previews-local-verification.json')
    if not (assets['assets_ready'] and not assets['missing_assets']
            and assets['source_zip_crc_pass'] and assets['fine_archive_crc_pass']
            and fine['fine_checkpoint_files_ready'] and fine['zip_crc_pass']
            and fine['local_remote_sha256_match'] and data['cpu_data_ready']
            and data['publisher_md5_all_pass'] and data['gt_audit_all_pass']
            and env['cpu_environment_pass'] and env['compiled_sm_120']
            and inputs['all_inputs_pass'] and da3['source_ready'] and da3['zip_crc_pass']
            and local['archive_sha256_pass'] and local['archive_crc_pass']
            and local['all_file_hashes_pass'] and local['scenes'] == 160):
        raise ValueError('Incomplete CPU preparation evidence')
    if any(row['returncode'] != 0 for row in tests[:2]) or not tests[2]['guard_pass']:
        raise ValueError('CPU tests or unauthorized-model guard failed')
    for row in assets['files']:
        p = Path(row['path'])
        if not p.is_file() or p.stat().st_size != row['bytes'] or Path(str(p)+'.aria2').exists():
            raise ValueError('Prepared asset missing, changed, or incomplete')
    # Persist the publisher comparison previously performed at download completion.
    wan = read('wan-dit-publisher-audit.json')
    endpoint = wan['publisher_metadata_endpoint']
    if args.publisher_metadata:
        metadata = json.loads(args.publisher_metadata.read_text())
        endpoint = 'https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B/revision/' + wan['revision'] + '?blobs=true'
    else:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(endpoint, timeout=30) as response:
            metadata = json.load(response)
    if metadata['sha'] != wan['revision']:
        raise ValueError('Publisher revision changed')
    published = {x['rfilename']: x for x in metadata['siblings']}
    checks = []
    for row in assets['files']:
        prefix = '/weights/Wan-AI/Wan2.1-T2V-1.3B/'
        if prefix not in row['path']:
            continue
        name = row['path'].split(prefix, 1)[1]
        expected = published[name]
        lfs = expected.get('lfs')
        size = lfs['size'] if lfs else expected['size']
        if size != row['bytes'] or (lfs and lfs['sha256'] != row['sha256']):
            raise ValueError('Wan publisher metadata mismatch: ' + name)
        checks.append({'filename': name, 'publisher_size_match': True,
                       'publisher_sha256_available': bool(lfs),
                       'publisher_sha256_match': True if lfs else None,
                       'computed_sha256': row['sha256']})
    (logs / 'wan-all-publisher-audit.json').write_text(json.dumps({
        'revision': wan['revision'], 'endpoint': endpoint, 'files': checks,
        'model_loaded': False, 'forward_called': False}, indent=2))
    status = {
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cpu_preparation_pass': True, 'assets_ready': True, 'cpu_environment_pass': True,
        'all_160_inputs_and_preprocessing_pass': True, 'query_previews_delivered': True,
        'query_user_review': 'pending', 'counts': inputs['counts'],
        'query_count_per_scene': 48, 'observed_frames': 7, 'future_frames': 15,
        'native_fps': 12, 'query_frame': 0, 'reference_camera_frame': 6,
        'model_loaded': False, 'forward_called': False, 'training_started': False,
        'gpu_used': False, 'gpu_forward_authorized': False,
        'observed_model_cache_ready': False, 'training_coordinate_audit_pass': False,
        'actual_weight_loading_tested': False,
        'gpu_stage_pending': ['actual offline model loading', 'observed-only geometry and MF caches',
                              'prediction/GT coordinate and scale audit', 'independent-reader training'],
        'data_disk_free_gib': shutil.disk_usage(base).free / 1024**3,
    }
    (logs / 'cpu-preparation-complete.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
