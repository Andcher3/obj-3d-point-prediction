"""Wait for the fixed GPU caches, then run approved calibration and readers once."""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--allow-training',action='store_true')
    ap.add_argument('--allow-observed-gt-calibration',action='store_true')
    args=ap.parse_args()
    if not args.allow_training or not args.allow_observed_gt_calibration:
        raise PermissionError('Both model training and observed label calibration require user approval')
    base=args.base.resolve()
    path=base/'logs/reader-stage-status.json'
    if path.exists():raise FileExistsError('Preserve previous reader execution')
    status={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'state':'waiting_for_forecast','controller_pid':os.getpid(),
            'hard_timeout_seconds':1800,'training_started':False}
    save=lambda:path.write_text(json.dumps(status,indent=2))
    save();start=time.monotonic()
    try:
        while True:
            forecast=json.loads((base/'logs/gpu-forecast-all-status.json').read_text())
            if forecast['state']=='completed':break
            if forecast['state'] in ('crashed','hard_timeout'):
                raise RuntimeError('Forecast failed; no calibration/training attempted')
            if time.monotonic()-start>1800:raise TimeoutError('Forecast wait hard timeout')
            time.sleep(5)
        if len(list((base/'model_cache_raw').glob('*.npz')))!=160:
            raise ValueError('Incomplete scene caches')
        commands=[('calibrating_supervision',[os.sys.executable,'-u',str(base/'code/calibrate_cross_point_supervision.py'),
            '--base',str(base),'--allow-observed-gt-calibration']),
            ('training',[os.sys.executable,'-u',str(base/'code/run_cross_point_probe.py'),
            '--base',str(base),'--cache-dir',str(base/'model_cache_audited'),
            '--config',str(base/'logs/calibrated-protocol.json'),
            '--output',str(base/'reader-results-g1.1'),'--device','cuda','--allow-training'])]
        for state,command in commands:
            status.update(state=state,command=command,training_started=state=='training')
            save()
            with (base/'logs'/('reader-'+state+'.log')).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status['child_pid']=child.pid;save()
                try:rc=child.wait(timeout=max(1,1800-(time.monotonic()-start)))
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise TimeoutError('Reader stage hard timeout')
            if rc:raise RuntimeError(state+' failed, exit='+str(rc))
        status.update(state='completed',returncode=0)
    except Exception as error:
        status.update(state='hard_timeout' if isinstance(error,TimeoutError) else 'crashed',
                      error=str(error),returncode=1)
        raise
    finally:
        status['elapsed_seconds']=time.monotonic()-start
        status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__=='__main__':main()
