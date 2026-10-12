"""Matched supervised auxiliary-task experiment; no frozen model forward."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from cross_point_probe_features import training_standardizer,standardize
from future_surface_aux_model import FutureSurfaceAuxModel,supervised_losses
from run_cross_point_probe import load_rows,scene_metrics
from summarize_cross_point_probe import METRICS,paired_bootstrap


def prepare(base,split,cfg,pairs):
    inherited=json.loads((base/'logs/calibrated-protocol.json').read_text())
    rows=load_rows(base,base/'model_cache_audited',split,'self_shared_true_neighbors',inherited)
    audit=[];i,j=pairs.T
    for row in rows:
        scene=row['scene']
        with np.load(base/'inputs'/scene/'gt_tracks.npz',allow_pickle=False) as gt:
            corners=gt['target_bboxes_world'].copy()
        ci,cj=np.triu_indices(8,1)
        length=np.linalg.norm(corners[:,ci]-corners[:,cj],axis=-1)
        rigid_error=float(np.max(np.abs(length-length[0])/np.maximum(length[0],1e-8)))
        if rigid_error>cfg['rigid_label_isometry_max_relative_error']:
            raise ValueError('Rigid transported-surface label invalid: '+scene)
        with np.load(base/'surface-g2-visible-run-2026-10-11/surfaces'/(scene+'.npz'),allow_pickle=False) as surface:
            path=surface['oracle_surface_path'][i,j]*float(surface['oracle_radius'])
            if bool(surface['future_geometry_used']):raise ValueError('Unexpected future surface input')
        with np.load(base/'model_cache_audited'/(scene+'.npz'),allow_pickle=False) as cache:
            cache_hash=str(cache['query_sha256'])
        if cache_hash!=hashlib.sha256((base/'inputs'/scene/'queries.json').read_bytes()).hexdigest():
            raise ValueError('Query identities changed: '+scene)
        mask=np.isfinite(path)
        normalized_surface=path/(row['evaluation_scale']*row['scale'])
        chord=np.linalg.norm(row['true'][:,i]-row['true'][:,j],axis=-1).T/row['scale']
        if np.any(chord[mask]>normalized_surface[mask,None]+1e-5):
            raise ValueError('Surface target shorter than future chord: '+scene)
        if not mask.any():raise ValueError('No valid supervised pairs: '+scene)
        row['aux_mask']=mask
        row['chord_target']=chord
        row['surface_target']=np.broadcast_to(np.where(mask,normalized_surface,0.)[:,None],chord.shape).copy()
        audit.append({'scene':scene,'maximum_bbox_isometry_relative_error':rigid_error,
                      'valid_aux_pairs':int(mask.sum()),'all_pairs':len(mask),
                      'geometry_future_used_as_prediction_input':False,
                      'surface_target':'rigid transport of GT frame0 visible material patch',
                      'full_mesh_geodesic':False})
    return rows,audit


def tensor_rows(rows,mean,std,device):
    def tensor(values):return torch.as_tensor(np.stack(values),dtype=torch.float32,device=device)
    return {'x':tensor([standardize(r['x'],mean,std) for r in rows]),
            'y':tensor([r['y'] for r in rows]),
            'chord':tensor([r['chord_target'] for r in rows]),
            'surface':tensor([r['surface_target'] for r in rows]),
            'mask':torch.as_tensor(np.stack([r['aux_mask'] for r in rows]),dtype=torch.bool,device=device)}


def evaluate_model(model,rows,data,pairs,batch_size=4):
    model.eval();scene_rows=[]
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            correction,distance=model(data['x'][start:start+batch_size],pairs)
            corrections=correction.cpu().numpy();distances=distance.cpu().numpy()
            for offset,(p,d) in enumerate(zip(corrections,distances)):
                row=rows[start+offset]
                xyz=row['baseline']+row['scale']*p.reshape(len(p),-1,3).transpose(1,0,2)
                metrics={k:v*row['evaluation_scale'] for k,v in scene_metrics(xyz,row['true']).items()}
                mask=row['aux_mask'];conversion=row['scale']*row['evaluation_scale']
                metrics['aux_chord_mae']=float(np.abs(d[mask]-row['chord_target'][mask]).mean()*conversion)
                metrics['aux_surface_mae']=float(np.abs(d[mask]-row['surface_target'][mask]).mean()*conversion)
                scene_rows.append({'scene':row['scene'],**metrics})
    return {'scene_macro':{m:float(np.mean([r[m] for r in scene_rows])) for m in scene_rows[0] if m!='scene'},
            'scenes':scene_rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],required=True);ap.add_argument('--allow-training',action='store_true')
    args=ap.parse_args()
    if not args.allow_training:raise PermissionError('Explicit user experiment authorization required')
    if args.output.exists():raise FileExistsError('Preserve existing results')
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Authorized GPU unavailable')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg=json.loads(args.config.read_text());splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    pair_array=np.array(np.triu_indices(48,1)).T
    pairs=torch.as_tensor(pair_array,dtype=torch.long,device=args.device)
    train,train_audit=prepare(args.base,splits['train'],cfg,pair_array)
    valid,valid_audit=prepare(args.base,splits['validation'],cfg,pair_array)
    mean,std=training_standardizer([r['x'] for r in train])
    training=tensor_rows(train,mean,std,args.device);validation=tensor_rows(valid,mean,std,args.device)
    point_moment=float(training['y'].square().mean())
    aux_moment=float(((training['chord'].square().mean(-1)*training['mask']).sum(-1)/training['mask'].sum(-1)).mean())
    if min(point_moment,aux_moment)<1e-10:raise ValueError('Degenerate training loss normalization')
    report={'protocol':cfg,'candidates':[],'selected_results':[],
            'point_loss_training_moment':point_moment,'aux_loss_training_chord_moment':aux_moment,
            'device':args.device,'frozen_model_forward':False,'GT_input_supplied':False,
            'auxiliary_future_geometry_only_in_loss':True,
            'distance_branch_shares_encoder':True,
            'distance_head_does_not_enforce_coordinate_geodesic_consistency':True}
    selected=[];started=time.monotonic()
    for seed in cfg['seeds']:
        for condition in cfg['conditions']:
            candidates=[]
            weights=[0.] if condition=='coordinate_only' else cfg['auxiliary_weight_candidates']
            for weight in weights:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=FutureSurfaceAuxModel(training['x'].shape[-1],cfg['shared_encoder_hidden_width'],
                    cfg['aux_pair_hidden_width'],cfg['forecast_steps']).to(args.device)
                parameters=sum(p.numel() for p in model.parameters())
                baseline_val=evaluate_model(model,valid,validation,pairs)['scene_macro']['ade']
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['learning_rate'],
                                            weight_decay=cfg['optimizer']['weight_decay'])
                history=[];best_score=float('inf');best_state=None;selected_epoch=None;stale=0
                target_key='surface' if condition=='future_surface_aux' else 'chord'
                batch_size=cfg['optimizer']['scene_batch_size']
                for epoch in range(cfg['max_epochs']):
                    model.train();point_values=[];aux_values=[]
                    order=rng.permutation(len(train))
                    for start in range(0,len(train),batch_size):
                        ids=torch.as_tensor(order[start:start+batch_size],device=args.device)
                        p,d=model(training['x'][ids],pairs)
                        point,aux=supervised_losses(p,training['y'][ids],d,training[target_key][ids],
                            training['mask'][ids],point_moment,aux_moment)
                        loss=point+weight*aux
                        if not torch.isfinite(loss):raise ValueError('Nonfinite supervised loss')
                        optimizer.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
                        optimizer.step();point_values.append(float(point.detach()));aux_values.append(float(aux.detach()))
                    score=evaluate_model(model,valid,validation,pairs)['scene_macro']['ade']
                    history.append({'epoch':epoch,'point_loss':float(np.mean(point_values)),
                                    'aux_loss':float(np.mean(aux_values)),'validation_ade':score})
                    if score<best_score:
                        best_score=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        selected_epoch=epoch;stale=0
                    else:stale+=1
                    if epoch%10==0:print('AUX_VALIDATION',seed,condition,weight,epoch,score,flush=True)
                    (args.output/'progress.json').write_text(json.dumps({'seed':seed,'condition':condition,
                        'weight':weight,'epoch':epoch,'elapsed_seconds':time.monotonic()-started}))
                    if stale>=cfg['early_stopping_patience']:break
                name=f'{condition}-seed{seed}-weight{weight}.pt'
                torch.save({'state':best_state,'mean':mean,'std':std},args.output/name)
                record={'seed':seed,'condition':condition,'weight':weight,'selected_epoch':selected_epoch,
                        'best_trained_validation_ade':best_score,'baseline_validation_ade':baseline_val,
                        'parameter_count':parameters,'history':history,'checkpoint':name}
                candidates.append(record);report['candidates'].append(record)
                (args.output/'results.json').write_text(json.dumps(report,indent=2))
            best=min(candidates,key=lambda r:r['best_trained_validation_ade'])
            selected.append(best)
            print('AUX_CONDITION_SELECTED',seed,condition,best['weight'],best['selected_epoch'],flush=True)
    # No test scores or labels are loaded into the fitting/selection routine above.
    test,test_audit=prepare(args.base,splits['test'],cfg,pair_array)
    testing=tensor_rows(test,mean,std,args.device)
    (args.output/'label-audit.json').write_text(json.dumps({'rows':train_audit+valid_audit+test_audit,
        'all_scenes_retained':True,'future_GT_in_prediction_features':False},indent=2))
    for best in selected:
        model=FutureSurfaceAuxModel(training['x'].shape[-1],cfg['shared_encoder_hidden_width'],
            cfg['aux_pair_hidden_width'],cfg['forecast_steps']).to(args.device)
        checkpoint=torch.load(args.output/best['checkpoint'],map_location=args.device,weights_only=False)
        model.load_state_dict(checkpoint['state'])
        score=evaluate_model(model,test,testing,pairs)
        retain_zero=best['baseline_validation_ade']<best['best_trained_validation_ade']
        row={k:v for k,v in best.items() if k!='history'}
        row.update(test_trained=score,deployment_validation_retains_zero=retain_zero)
        report['selected_results'].append(row)
    # Exact original-model reference, no calibration refit or conditional gate.
    baseline=FutureSurfaceAuxModel(training['x'].shape[-1],cfg['shared_encoder_hidden_width'],
            cfg['aux_pair_hidden_width'],cfg['forecast_steps']).to(args.device)
    report['baseline_test']=evaluate_model(baseline,test,testing,pairs)
    report['elapsed_seconds']=time.monotonic()-started
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    comparisons=[]
    groups={condition:sorted([r for r in report['selected_results'] if r['condition']==condition],key=lambda r:r['seed'])
            for condition in cfg['conditions']}
    for control in ['coordinate_only','future_chord_aux']:
        arrays={}
        for condition in [control,'future_surface_aux']:
            arrays[condition]=np.mean([np.array([[row[m] for m in METRICS]
                for row in sorted(r['test_trained']['scenes'],key=lambda s:s['scene'])])
                for r in groups[condition]],axis=0)
        comparisons.append({'control':control,'condition':'future_surface_aux',
            'metrics':{m:paired_bootstrap(arrays[control][:,i],arrays['future_surface_aux'][:,i])
                       for i,m in enumerate(METRICS)}})
    macros={c:{m:float(np.mean([r['test_trained']['scene_macro'][m] for r in groups[c]])) for m in METRICS}
            for c in cfg['conditions']}
    summary={'comparisons':comparisons,'seed_averaged_trained_macros':macros,
             'baseline_macro':report['baseline_test']['scene_macro'],
             'test_reused_exploratory':True,'independent_test_scenes':40,
             'all_selected_trained_models_deployment_retain_zero':all(r['deployment_validation_retains_zero'] for r in report['selected_results'])}
    (args.output/'paired-summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'complete.json').write_text(json.dumps({'complete':True,'fits':len(report['candidates']),
        'selected_models':len(selected),'counts':{k:len(v) for k,v in splits.items()},'device':args.device,
        'future_GT_input':False,'label_domain':cfg['surface_domain']}))
    print('FUTURE_SURFACE_AUX_COMPLETE',report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
