#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=[]
    for stage,n in [('smoke12',12),('smoke16',16),('full22',22)]:
        log=a.dir/f'{stage}.log'; text=log.read_text(errors='replace') if log.exists() else ''
        row={'stage':stage,'expected_shape':[n,48,3],'log':str(log),'log_exists':log.exists()}
        m=re.search(r'\[END\] '+stage+r' rc=(\d+)',text); row['returncode']=int(m.group(1)) if m else None
        out=a.dir/f'{stage}.npz'; row['output_exists']=out.exists()
        if out.exists():
            z=np.load(out,allow_pickle=True); x=z['trackcraft3r_tracks']; row.update(shape=list(x.shape),finite_fraction=float(np.isfinite(x).mean()),shape_ok=bool(x.shape==(n,48,3)))
        else:
            row['shape_ok']=False
            oom='OutOfMemoryError' in text
            row['failure_stage']='predictor.predict/model_fn_wan_video' if oom else 'unknown_or_process_exit'
            row['cuda_oom']=oom
        rows.append(row)
    payload={'protocol':'query-only with fixed t0-to-frame0 query mapping','stages':rows,'overall_smoke_pass':all(r.get('shape_ok') for r in rows[:2]),'full22_dense_output_ready':bool(rows[2].get('shape_ok'))}
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
