"""Launch an authorized stage with exit-code, elapsed-time and timeout evidence."""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--stage', choices=['geometry', 'forecast', 'all'], required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--timeout-seconds', type=int, default=1800)
    ap.add_argument('--allow-model-forward', action='store_true')
    args = ap.parse_args()
    if not args.allow_model_forward:
        raise PermissionError('GPU stage requires user authorization')
    base = args.base.resolve()
    status_file = base / 'logs' / (args.run_name + '-status.json')
    log_file = base / 'logs' / (args.run_name + '.log')
    if status_file.exists() or log_file.exists():
        raise FileExistsError('Preserve previous execution evidence')
    command = [os.sys.executable, '-u', str(base / 'code/generate_cross_point_model_cache.py'),
               '--base', str(base), '--allow-model-forward', '--stage', args.stage,
               '--limit', str(args.limit)]
    start = time.monotonic()
    status = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': command, 'state': 'running', 'hard_timeout_seconds': args.timeout_seconds,
              'stage': args.stage, 'limit': args.limit, 'controller_pid': os.getpid()}
    with log_file.open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status['child_pid'] = child.pid
        status_file.write_text(json.dumps(status, indent=2))
        try:
            rc = child.wait(timeout=args.timeout_seconds)
            status.update(returncode=rc, state='completed' if rc == 0 else 'crashed')
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            status.update(returncode=child.returncode, state='hard_timeout')
    status['elapsed_seconds'] = time.monotonic() - start
    status['ended_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status_file.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)
    if status['returncode']:
        raise SystemExit(status['returncode'])


if __name__ == '__main__':
    main()
