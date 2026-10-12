"""Matched CPU ridge probes of observed visible-surface conditions.

No future surface geometry, model forward, or new teacher labels are used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cross_point_probe_features import features, ridge_fit, standardize, training_standardizer
from run_cross_point_probe import load_rows, evaluate
from summarize_cross_point_probe import METRICS, paired_bootstrap


def neighbor_indices(path,uv,k,surface):
    image=np.linalg.norm(uv[:,None]-uv[None,:],axis=-1)
    np.fill_diagonal(image,np.inf)
    image_order=np.argsort(image,axis=1,kind='stable')
    if not surface:return image_order[:,:k],0
    path=np.asarray(path).copy();np.fill_diagonal(path,np.inf)
    order=np.argsort(path,axis=1,kind='stable');result=[];fallback=0
    for point,row in enumerate(order):
        chosen=[int(i) for i in row if np.isfinite(path[point,i])][:k]
        for other in image_order[point]:
            if len(chosen)==k:break
            if int(other) not in chosen and int(other)!=point:
                chosen.append(int(other));fallback+=1
        if len(chosen)!=k:raise ValueError('Insufficient distinct neighbor queries')
        result.append(chosen)
    return np.asarray(result),fallback


def condition_features(cache,surface,condition,k):
    original,info=features(cache['observed_xyz'],cache['baseline_future'],
                           cache['latent_future'],cache['query_uv'],True,k)
    n=len(original)
    prefix='oracle' if condition=='oracle_visible_surface' else 'estimated'
    path=surface[prefix+'_surface_path'];straight=surface[prefix+'_straight_distance']
    if path.shape!=(n,n) or straight.shape!=(n,n):raise ValueError('Surface identity shape mismatch')
    use_topology=condition in ['surface_neighbors_straight','surface_neighbors_surface','oracle_visible_surface']
    use_distance=condition in ['image_neighbors_surface','surface_neighbors_surface','oracle_visible_surface']
    index,fallback=neighbor_indices(path,cache['query_uv'],k,use_topology)
    own=(np.asarray(cache['observed_xyz'])-info['origin'])/info['scale']
    neighbor=(own[:,index]-own[-1][None,:,None,:]).transpose(1,2,0,3).reshape(n,-1)
    neighbor=np.c_[neighbor,np.ones(n)]
    ids=np.arange(n)[:,None];selected_path=path[ids,index];reachable=np.isfinite(selected_path)
    distance=selected_path if use_distance else straight[ids,index]
    distance=np.where(np.isfinite(distance),distance,0.)
    # Reachability flags accompany both straight/path choices for a given graph:
    # the distance-only ablation does not introduce a second missingness signal.
    relation=np.c_[distance,reachable.astype(np.float64)]
    x=np.c_[original[:,:info['self_feature_width']],neighbor,relation]
    if not np.isfinite(x).all():raise ValueError('Nonfinite condition features')
    return x,{'fallback_neighbor_slots':fallback,'total_neighbor_slots':n*k,
              'input_width':x.shape[1],'surface_query_frame':0,
              'future_surface_used':False,'oracle_GT_input':prefix=='oracle'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--surface-dir',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-training',action='store_true')
    args=ap.parse_args()
    if not args.allow_training:raise PermissionError('Explicit experiment training authorization required')
    if args.output.exists():raise FileExistsError('Preserve old test results')
    args.output.mkdir(parents=True)
    cfg=json.loads(args.config.read_text())
    inherited=json.loads((args.base/'logs/calibrated-protocol.json').read_text())
    splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    report={'experiment':cfg,'results':[],'all_scenes_retained':True,
            'model_forward_performed':False,'new_GT_coordinate_fit':False,
            'test_previously_evaluated_in_G1':True,'test_used_for_G2_tuning':False,
            'reader':'ridge','future_geometry_as_condition':False}
    audit_rows=[]
    def prepare_rows(split,condition):
        rows=load_rows(args.base,args.base/'model_cache_audited',split,
                       'self_shared_true_neighbors',inherited)
        for row in rows:
            with np.load(args.base/'model_cache_audited'/(row['scene']+'.npz'),allow_pickle=False) as cache:
                with np.load(args.surface_dir/(row['scene']+'.npz'),allow_pickle=False) as surface:
                    if bool(surface['future_geometry_used']) or str(surface['query_sha256'])!=str(cache['query_sha256']):
                        raise ValueError('Query identity/future surface mismatch')
                    row['x'],audit=condition_features(cache,surface,condition,cfg['neighbor_count'])
            audit_rows.append({'scene':row['scene'],'condition':condition,**audit})
        return rows
    for condition in cfg['conditions']:
        train=prepare_rows(splits['train'],condition);valid=prepare_rows(splits['validation'],condition)
        mean,std=training_standardizer([r['x'] for r in train])
        x=np.concatenate([standardize(r['x'],mean,std) for r in train]);y=np.concatenate([r['y'] for r in train])
        best=None;candidates=[]
        for alpha in cfg['ridge_alpha_candidates']:
            w,b=ridge_fit(x,y,alpha)
            score=evaluate(valid,lambda z:z@w+b,mean,std)['scene_macro']['ade']
            candidates.append({'alpha':alpha,'validation_ade':score})
            print('SURFACE_RIDGE_VALIDATION',condition,alpha,score,flush=True)
            if best is None or score<best[0]:best=(score,alpha,w,b)
        score,alpha,w,b=best
        # Preserve G1 ridge selection: alpha is chosen within fitted candidates.
        # Zero correction is a diagnostic reference, not a new selection policy.
        zero_score=evaluate(valid,lambda z:np.zeros((len(z),y.shape[1])),mean,std)['scene_macro']['ade']
        # Surface-specific test labels are first opened after this selection.
        test=prepare_rows(splits['test'],condition)
        predict=lambda z:z@w+b
        result={'condition':condition,'alpha':alpha,'validation_candidates':candidates,
                'zero_validation_ade':zero_score,'retained_zero_correction':False,
                'best_fitted_validation_ade':score,'input_width':x.shape[1],
                'parameter_count':int(w.size+b.size),'test':evaluate(test,predict,mean,std)}
        report['results'].append(result)
        np.savez_compressed(args.output/(condition+'-ridge.npz'),weight=w,bias=b,mean=mean,std=std,
                            selected_zero_correction=False)
        (args.output/'results.json').write_text(json.dumps(report,indent=2))
        print('SURFACE_RIDGE_SELECTED',condition,alpha,flush=True)
    reports={r['condition']:r for r in report['results']}
    control=cfg['primary_comparison'][0]
    comparisons=[]
    for condition in cfg['conditions']:
        if condition==control:continue
        a=sorted(reports[control]['test']['scenes'],key=lambda x:x['scene'])
        b=sorted(reports[condition]['test']['scenes'],key=lambda x:x['scene'])
        if [r['scene'] for r in a]!=[r['scene'] for r in b]:raise ValueError('Unmatched evaluation scenes')
        comparisons.append({'condition':condition,'control':control,
                            'primary':condition==cfg['primary_comparison'][1],
                            'oracle_only':condition=='oracle_visible_surface',
                            'metrics':{m:paired_bootstrap([r[m] for r in a],[r[m] for r in b]) for m in METRICS}})
    (args.output/'comparisons.json').write_text(json.dumps({'comparisons':comparisons,
        'one_prespecified_primary_comparison_95pct_interval':True,
        'other_intervals_descriptive_only':True,
        'all_final_models_chose_zero':all(r['retained_zero_correction'] for r in report['results'])},indent=2))
    (args.output/'feature-audit.json').write_text(json.dumps(audit_rows,indent=2))
    (args.output/'complete.json').write_text(json.dumps({'completed':True,'conditions':len(report['results']),
        'counts':{k:len(v) for k,v in splits.items()},'GPU_used':False,'future_geometry_used':False}))
    print('G2_VISIBLE_CONDITION_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
