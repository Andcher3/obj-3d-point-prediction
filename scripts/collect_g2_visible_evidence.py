"""Compact evidence manifest and pre-existing-input diagnostics, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_g2_visible_conditions import neighbor_indices


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    audit=json.loads((args.run_dir/'surfaces/surface-audit.json').read_text())
    cfg=json.loads((args.base/'surface_g2_visible_conditions_2026-10-11.json').read_text())
    diagnostics=[]
    for row in audit['rows']:
        query=json.loads((args.base/'inputs'/row['scene']/'queries.json').read_text())
        uv=np.asarray(query['query_uv_source_pixels']);n=len(uv);k=cfg['neighbor_count']
        with np.load(args.run_dir/'surfaces'/(row['scene']+'.npz'),allow_pickle=False) as surface:
            for prefix in ['estimated','oracle']:
                path=surface[prefix+'_surface_path'];straight=surface[prefix+'_straight_distance']
                image,_=neighbor_indices(path,uv,k,False)
                chosen,fallback=neighbor_indices(path,uv,k,True)
                overlap=np.mean([len(set(a)&set(b))/k for a,b in zip(image,chosen)])
                ids=np.arange(n)[:,None];valid=np.isfinite(path[ids,image])&(straight[ids,image]>1e-8)
                ratio=path[ids,image][valid]/straight[ids,image][valid]
                diagnostics.append({'scene':row['scene'],'split':row['split'],'source':prefix,
                    'surface_vs_image_neighbor_set_overlap':float(overlap),
                    'surface_neighbor_fallback_fraction':fallback/(n*k),
                    'image_neighbor_path_to_chord_ratio_median':float(np.median(ratio)),
                    'image_neighbor_path_to_chord_ratio_p90':float(np.quantile(ratio,.9)),
                    **row[prefix]})
    summaries={}
    for prefix in ['estimated','oracle']:
        rows=[r for r in diagnostics if r['source']==prefix]
        summaries[prefix]={
            'query_graph_connected_scenes':sum(r['query_connected_components']==1 for r in rows),
            'total_scenes':len(rows),
            'mean_neighbor_set_overlap':float(np.mean([r['surface_vs_image_neighbor_set_overlap'] for r in rows])),
            'mean_fallback_neighbor_fraction':float(np.mean([r['surface_neighbor_fallback_fraction'] for r in rows])),
            'mean_reachable_pair_fraction':float(np.mean([r['reachable_query_pair_fraction'] for r in rows])),
            'median_scene_path_to_chord_ratio':float(np.median([r['image_neighbor_path_to_chord_ratio_median'] for r in rows]))}
    out={'all_scene_summaries':summaries,'scene_rows':diagnostics,'new_models_fitted':False}
    (args.run_dir/'input-diagnostics.json').write_text(json.dumps(out,indent=2))
    paths=[('status.json',args.run_dir/'status.json'),
           ('surface-audit.json',args.run_dir/'surfaces/surface-audit.json'),
           ('surface_prepare.log',args.run_dir/'surface_prepare.log'),
           ('condition_train.log',args.run_dir/'condition_train.log'),
           ('input-diagnostics.json',args.run_dir/'input-diagnostics.json')]
    paths.extend((name,args.run_dir/'results'/name) for name in
                 ['results.json','comparisons.json','feature-audit.json','complete.json'])
    paths.extend([( 'readiness.json',args.base/'surface-g2-preflight-2026-10-11/readiness.json'),
                  ('surface_g2_visible_conditions_2026-10-11.json',args.base/'surface_g2_visible_conditions_2026-10-11.json')])
    manifest={'files':[{'local_name':name,'remote_path':str(path),'bytes':path.stat().st_size,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths],
              'model_weights_included':False}
    (args.run_dir/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'diagnostics':summaries,'file_count':len(paths),
                      'evidence_bytes':sum(r['bytes'] for r in manifest['files'])},indent=2))


if __name__=='__main__':main()
