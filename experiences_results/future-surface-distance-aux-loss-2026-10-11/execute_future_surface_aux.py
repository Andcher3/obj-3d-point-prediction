"""Bounded GPU controller: preserve outputs, report failures, never auto-retry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--allow-training',action='store_true')
    args=ap.parse_args()
    if not args.allow_training:raise PermissionError('Experiment authorization required')
    if args.run_dir.exists():raise FileExistsError('Preserve existing run')
    args.run_dir.mkdir(parents=True)
    cfg=json.loads(args.config.read_text())
    started=time.monotonic()
    status={'controller_pid':os.getpid(),'state':'running','GPU_used':True,
            'started_at_utc':datetime.now(timezone.utc).isoformat(),
            'hard_timeout_seconds':cfg['hard_timeout_seconds']}
    env=os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='8',
               OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    command=[sys.executable,'-u',str(args.base/'code/run_future_surface_aux.py'),
             '--base',str(args.base),'--config',str(args.config),
             '--output',str(args.run_dir/'results'),'--device','cuda','--allow-training']
    with (args.run_dir/'training.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        status['child_pid']=process.pid
        while process.poll() is None:
            status.update(elapsed_seconds=time.monotonic()-started,
                          updated_at_utc=datetime.now(timezone.utc).isoformat())
            (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
            if status['elapsed_seconds']>cfg['hard_timeout_seconds']:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                status.update(state='timeout',returncode=process.returncode)
                (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
                raise TimeoutError('Future surface auxiliary experiment hard timeout')
            time.sleep(2)
    status.update(state='completed' if process.returncode==0 else 'failed',
                  returncode=process.returncode,elapsed_seconds=time.monotonic()-started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
    if process.returncode:raise RuntimeError('Training failed; no automatic retry')
    print('FUTURE_SURFACE_AUX_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
