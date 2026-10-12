"""Collect read-only diagnostics and hashes after the fixed experiment ends."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    status=json.loads((args.run_dir/'status.json').read_text())
    if status['state']!='completed':raise RuntimeError('Training not completed')
    results=args.run_dir/'results'
    report=json.loads((results/'results.json').read_text())
    cfg=report['protocol']
    assert len(report['candidates'])==21 and len(report['selected_results'])==9
    assert len({r['parameter_count'] for r in report['candidates']})==1
    from run_future_surface_aux import prepare
    splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    assert len(set(sum(splits.values(),[])))==sum(map(len,splits.values()))
    pairs=np.array(np.triu_indices(48,1)).T
    diagnostics=[]
    for split,scenes in splits.items():
        rows,_=prepare(args.base,scenes,cfg,pairs)
        for row in rows:
            mask=row['aux_mask'];chord=row['chord_target'][mask];surface=row['surface_target'][mask]
            ratio=surface/np.maximum(chord,1e-8)
            diagnostics.append({'scene':row['scene'],'split':split,
                'surface_to_chord_ratio_median':float(np.median(ratio)),
                'surface_to_chord_ratio_p90':float(np.quantile(ratio,.9)),
                'aux_pair_fraction':float(mask.mean()),
                'maximum_future_surface_target_change':float(np.max(np.ptp(surface,axis=-1))),
                'physical_target_surface_mean':float(surface.mean()*row['scale']*row['evaluation_scale'])})
    summary={'scenes':diagnostics,'parameter_count':report['candidates'][0]['parameter_count'],
             'scene_split_disjoint':True,'baseline_macro_matches_prior_G1':
             bool(np.isclose(report['baseline_test']['scene_macro']['ade'],3.548819891,atol=1e-8))}
    (results/'target-diagnostics.json').write_text(json.dumps(summary,indent=2))
    files=[args.run_dir/'status.json',args.run_dir/'training.log',
           args.base/'logs/surface-future-aux-controller.log',
           args.base/'logs/surface-future-aux-protocol.json']
    files+=list(results.glob('*.json'))
    files+=[args.base/'code'/name for name in ['future_surface_aux_model.py','test_future_surface_aux_model.py',
                                              'run_future_surface_aux.py','execute_future_surface_aux.py',
                                              'collect_future_surface_aux_evidence.py']]
    manifest={'files':[{'remote':str(p),'name':p.name,'bytes':p.stat().st_size,
                       'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (args.run_dir/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'evidence_files':len(files),'bytes':sum(r['bytes'] for r in manifest['files'])}))


if __name__=='__main__':main()
