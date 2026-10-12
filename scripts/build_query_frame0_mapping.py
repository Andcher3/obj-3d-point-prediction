import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',type=Path,required=True); ap.add_argument('--input-h',type=int,required=True); ap.add_argument('--input-w',type=int,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    z=np.load(a.cache,allow_pickle=True); q=np.asarray(z['query_uv_frame0_dense'],dtype=float); q=q*np.array([a.input_w/832.0,a.input_h/480.0]);
    a.output.write_text(json.dumps({'query_sets':{'64':q.tolist()},'source':'verified_query_uv_frame0_dense_converted_to_input_resolution','input_resolution_hw':[a.input_h,a.input_w]},indent=2)+'\n')
    print(json.dumps({'count':len(q),'range_xy':[q.min(0).tolist(),q.max(0).tolist()],'output':str(a.output)}))
if __name__=='__main__': main()
