"""Finite CPU-only experiment controller with explicit output and hard timeout."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--allow-training',action='store_true');args=ap.parse_args()
    if not args.allow_training:raise PermissionError('User experiment authorization required')
    if args.run_dir.exists():raise FileExistsError('Preserve existing run directory')
    args.run_dir.mkdir(parents=True)
    started=time.monotonic();status={'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'controller_pid':os.getpid(),'state':'running','GPU_used':False,'hard_timeout_seconds':900}
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                                  CUDA_VISIBLE_DEVICES='')
    stages=[('surface_prepare',[sys.executable,'-u',str(args.base/'code/prepare_g2_visible_surface.py'),
             '--base',str(args.base),'--config',str(args.config),'--output',str(args.run_dir/'surfaces')]),
            ('condition_train',[sys.executable,'-u',str(args.base/'code/run_g2_visible_conditions.py'),
             '--base',str(args.base),'--surface-dir',str(args.run_dir/'surfaces'),'--config',str(args.config),
             '--output',str(args.run_dir/'results'),'--allow-training'])]
    codes={}
    for name,command in stages:
        status.update(stage=name,command=command)
        with (args.run_dir/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            status['child_pid']=process.pid
            while process.poll() is None:
                status.update(elapsed_seconds=time.monotonic()-started,
                              updated_at_utc=datetime.now(timezone.utc).isoformat())
                (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
                if status['elapsed_seconds']>900:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    status.update(state='timeout',returncode=process.returncode)
                    (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
                    raise TimeoutError('G2 CPU pipeline hard timeout')
                time.sleep(2)
            codes[name]=process.returncode
            if process.returncode:
                status.update(state='failed',returncode=process.returncode,stage_codes=codes)
                (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
                raise RuntimeError('G2 stage failed: '+name)
    status.update(state='completed',returncode=0,stage_codes=codes,
                  elapsed_seconds=time.monotonic()-started,
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    (args.run_dir/'status.json').write_text(json.dumps(status,indent=2))
    print('G2_CPU_PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':main()
