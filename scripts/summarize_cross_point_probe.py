"""Paired scene-level summaries; seeds/points are not independent samples."""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS=['ade','fde','relative_vector_error','pair_length_error','future_change_error']


def paired_bootstrap(a,b,replications=10000):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a+b).all() or a.mean()<=0:
        raise ValueError('Invalid paired scene errors')
    indices=np.random.default_rng(10).integers(0,len(a),(replications,len(a)))
    gains=(a[indices].mean(1)-b[indices].mean(1))/a[indices].mean(1)
    return {'relative_gain':float((a.mean()-b.mean())/a.mean()),
            'descriptive_95pct_interval':np.quantile(gains,[.025,.975]).tolist(),
            'two_reader_family_97_5pct_interval':np.quantile(gains,[.0125,.9875]).tolist(),
            'scene_win_fraction':float(np.mean(b<a)),
            'mean_paired_error_reduction':float((a-b).mean()),'bootstrap_replicates':replications}


def compare(data,reader):
    selected=[x for x in data['results'] if x['reader']==reader]
    groups={key:sorted([x for x in selected if x['condition']==key],key=lambda x:x.get('seed',-1))
            for key in ['self_shared_no_neighbors','self_shared_true_neighbors']}
    expected=1 if reader=='ridge' else 3
    if any(len(x)!=expected for x in groups.values()):raise ValueError('Incomplete reader results')
    seed_rows=[];arrays={};scenes=None
    for group,runs in groups.items():
        errors=[]
        for run in runs:
            rows=sorted(run['test']['scenes'],key=lambda x:x['scene'])
            ids=[x['scene'] for x in rows]
            if scenes is None:scenes=ids
            if ids!=scenes or len(ids)!=40:raise ValueError('Unmatched test scene identities')
            errors.append(np.array([[r[m] for m in METRICS] for r in rows]))
            seed_rows.append({'condition':group,'seed':run.get('seed'),
                              'selected_epoch':run.get('selected_epoch'),
                              'test_scene_macro':run['test']['scene_macro']})
        # Average across the same three fitted seeds BEFORE scene resampling.
        arrays[group]=np.mean(errors,axis=0)
    a=arrays['self_shared_no_neighbors'];b=arrays['self_shared_true_neighbors']
    metrics={m:paired_bootstrap(a[:,i],b[:,i]) for i,m in enumerate(METRICS)}
    interval=metrics['ade']['two_reader_family_97_5pct_interval']
    if interval[0]>=.05:decision='supports_at_least_5pct_gain_for_this_reader'
    elif interval[1]<.05:decision='does_not_meet_5pct_practical_target_for_this_reader'
    else:decision='5pct_practical_gain_undetermined'
    all_zero_initialization=(reader!='ridge' and
                             all(run.get('selected_epoch') == -1
                                 for runs in groups.values() for run in runs))
    if all_zero_initialization:
        decision='validation_retained_zero_correction_information_value_unresolved'
    return {'reader':reader,'independent_test_scenes':40,'fitted_seeds':expected,
            'all_runs_selected_zero_correction_initialization':all_zero_initialization,
            'metrics':metrics,'decision':decision,'scene_ids':scenes,'seed_results':seed_rows,
            'no_neighbor_macro':{m:float(a[:,i].mean()) for i,m in enumerate(METRICS)},
            'true_neighbor_macro':{m:float(b[:,i].mean()) for i,m in enumerate(METRICS)},
            'scene_rows':[{'scene':s,'no_neighbors_ade':float(a[i,0]),
                          'neighbors_ade':float(b[i,0])} for i,s in enumerate(scenes)]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous statistical summary')
    data=json.loads(args.results.read_text())
    if len(data['results'])!=8:raise ValueError('Expected two ridge and six MLP fitted outputs')
    ridges=[r for r in data['results'] if r['reader']=='ridge']
    if ridges[0]['baseline_test']!=ridges[1]['baseline_test']:
        raise ValueError('Baseline changed between input conditions')
    summary={'readers':[compare(data,r) for r in ['ridge','mlp_128_128']],
        'baseline':ridges[0]['baseline_test']['scene_macro'],
        'coordinate_protocol':data['coordinate_protocol'],
        'GT_assisted_label_evaluation_calibration':data['GT_assisted_label_evaluation_calibration'],
        'bootstrap_unit':'independent scene; paired conditions; three MLP seeds averaged within scene',
        'primary_metric':'scene-macro ADE','secondary_intervals':'descriptive, not family-corrected',
        'test_used_for_tuning':False,'family_size_primary_readers':2,
        'minimum_practical_gain':.05,'no_claim_about_all_relation_architectures':True}
    args.output.write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='readers'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['scene_rows','scene_ids','seed_results']}
                      for r in summary['readers']],indent=2))


if __name__=='__main__':main()
