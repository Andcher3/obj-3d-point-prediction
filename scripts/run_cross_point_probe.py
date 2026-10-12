"""Frozen-input information probe. Never loads MotionForesight or a teacher.

Requires audited observed-only model caches. Training is opt-in; this script is
prepared now but must not run until the user authorizes the GPU stage.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from cross_point_probe_features import features,check_cache,training_standardizer,standardize,ridge_fit


def scene_metrics(pred,true):
    n=true.shape[1]; i,j=np.triu_indices(n,1)
    return {'ade':float(np.linalg.norm(pred-true,axis=-1).mean()),
        'fde':float(np.linalg.norm(pred[-1]-true[-1],axis=-1).mean()),
        'future_change_error':float(np.linalg.norm((pred[1:]-pred[0])-(true[1:]-true[0]),axis=-1).mean()),
        'relative_vector_error':float(np.linalg.norm((pred[:,j]-pred[:,i])-(true[:,j]-true[:,i]),axis=-1).mean()),
        'pair_length_error':float(np.abs(np.linalg.norm(pred[:,j]-pred[:,i],axis=-1)-np.linalg.norm(true[:,j]-true[:,i],axis=-1)).mean())}


def load_rows(base,cache_dir,split,condition,cfg):
    rows=[]
    for scene in split:
        with np.load(cache_dir/(scene+'.npz'),allow_pickle=False) as cache:
            check_cache(cache,cfg['frame_protocol']['observed_frames'],cfg['frame_protocol']['frames'])
            # A shared camera-frame name does not establish shared physical units.
            # Do not silently train against simulator coordinates using unverified
            # DA3-scale predictions. This gate is separate from CPU readiness.
            if ('training_coordinate_audit_pass' not in cache
                    or not bool(cache['training_coordinate_audit_pass'])):
                raise ValueError('Model/GT coordinate and scale audit pending: '+scene)
            x,info=features(cache['observed_xyz'],cache['baseline_future'],cache['latent_future'],cache['query_uv'],
                neighbors=condition=='self_shared_true_neighbors',k=cfg['neighbor_count'])
            baseline=np.asarray(cache['baseline_future']).copy()
            ts=np.asarray(cache['timestamps_s']).copy()
            calibration_source=str(cache['raw_cache_sha256']) if 'raw_cache_sha256' in cache else None
        with np.load(base/cfg.get('labels_directory','cpu_cache')/(scene+'-labels.npz'),allow_pickle=False) as labels:
            true=labels['future_xyz'].copy()
            evaluation_scale=float(labels['evaluation_to_gt_scale']) if 'evaluation_to_gt_scale' in labels else 1.
            if calibration_source is not None:
                if str(labels['raw_cache_sha256'])!=calibration_source or bool(labels['future_used_to_fit']):
                    raise ValueError('Label/model calibration mismatch: '+scene)
            if not np.allclose(ts[cfg['frame_protocol']['observed_frames']:],labels['timestamps_s'],atol=1e-8):
                raise ValueError('GT/model timestamp mismatch: '+scene)
            if not labels['validity'].all(): raise ValueError('Explicit partial-label loss needed')
        if true.shape!=baseline.shape: raise ValueError('GT/model shape mismatch: '+scene)
        y=((true-baseline)/info['scale']).transpose(1,0,2).reshape(len(x),-1)
        rows.append({'scene':scene,'x':x,'y':y,'baseline':baseline,'true':true,'scale':info['scale'],
                     'evaluation_scale':evaluation_scale})
    return rows


def evaluate(rows,predict,mean,std):
    result=[]
    for r in rows:
        correction=predict(standardize(r['x'],mean,std)).reshape(len(r['x']),-1,3).transpose(1,0,2)
        pred=r['baseline']+r['scale']*correction
        metrics=scene_metrics(pred,r['true'])
        metrics={k:v*r.get('evaluation_scale',1.) for k,v in metrics.items()}
        result.append({'scene':r['scene'],**metrics})
    return {'scene_macro':{k:float(np.mean([r[k] for r in result])) for k in ['ade','fde','relative_vector_error','pair_length_error','future_change_error']},'scenes':result}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--allow-training',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],required=True)
    args=ap.parse_args()
    if not args.allow_training: raise PermissionError('Training requires explicit --allow-training after GPU/user authorization')
    if args.output.exists(): raise FileExistsError('Preserve old test results; choose fresh output')
    cfg=json.loads(args.config.read_text()); splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    if not (args.base/'logs/preparation-complete.json').exists(): raise ValueError('CPU data not ready')
    # Verify disjointness before any fit; no query-wise train/test split.
    assert len(set(sum(splits.values(),[])))==sum(map(len,splits.values()))
    import torch
    from torch import nn
    torch.set_num_threads(8)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
    args.output.mkdir(parents=True)
    output={'results':[],'training_data':'simulator GT','frozen_model_modified':False,'teacher_used':False,
        'conditions':cfg['conditions'],'test_reused_for_tuning':False,'seeds':cfg['seeds'],
        'coordinate_protocol':cfg.get('supervision_coordinate_protocol','unmodified'),
        'GT_assisted_label_evaluation_calibration':cfg.get('labels_directory')=='calibrated_labels'}
    for condition in cfg['conditions']:
        train=load_rows(args.base,args.cache_dir,splits['train'],condition,cfg)
        valid=load_rows(args.base,args.cache_dir,splits['validation'],condition,cfg)
        # Test labels are not loaded until the validation selection is complete.
        mean,std=training_standardizer([r['x'] for r in train])
        x=np.concatenate([standardize(r['x'],mean,std) for r in train]); y=np.concatenate([r['y'] for r in train])
        best=None; candidates=[]
        for alpha in cfg['ridge_alpha_candidates']:
            w,b=ridge_fit(x,y,alpha)
            score=evaluate(valid,lambda z:z@w+b,mean,std)
            candidates.append({'alpha':alpha,'validation_ade':score['scene_macro']['ade']})
            print('RIDGE_VALIDATION',condition,alpha,score['scene_macro']['ade'],flush=True)
            if best is None or score['scene_macro']['ade']<best[0]: best=(score['scene_macro']['ade'],alpha,w,b)
        _,alpha,w,b=best
        test=load_rows(args.base,args.cache_dir,splits['test'],condition,cfg)
        output['results'].append({'condition':condition,'reader':'ridge','alpha':alpha,'validation_candidates':candidates,
            'test':evaluate(test,lambda z:z@w+b,mean,std),'baseline_test':evaluate(test,lambda z:np.zeros((len(z),y.shape[1])),mean,std)})
        np.savez_compressed(args.output/(condition+'-ridge.npz'),weight=w,bias=b,mean=mean,std=std)
        # Fixed MLP family; validation early stopping, independent paired seeds.
        for seed in cfg['seeds']:
            torch.manual_seed(seed); rng=np.random.default_rng(seed)
            params=cfg['mlp']; width=params['hidden_width']
            model=nn.Sequential(nn.Linear(x.shape[1],width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,y.shape[1])).to(args.device)
            nn.init.zeros_(model[-1].weight); nn.init.zeros_(model[-1].bias)
            optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
            xt=torch.as_tensor(x,dtype=torch.float32,device=args.device); yt=torch.as_tensor(y,dtype=torch.float32,device=args.device)
            best_score=float('inf'); best_state=None; stale=0; history=[]
            def predict(z):
                model.eval()
                with torch.no_grad(): return model(torch.as_tensor(z,dtype=torch.float32,device=args.device)).cpu().numpy()
            # The initialized model is a legitimate early-stopping checkpoint.
            # This permits validation selection to retain zero correction when
            # training gives no improvement; no test-dependent output gate.
            best_score=evaluate(valid,predict,mean,std)['scene_macro']['ade']
            best_state=copy.deepcopy(model.state_dict()); selected_epoch=-1
            history.append({'epoch':-1,'training_mse':None,'validation_ade':best_score})
            for epoch in range(params['max_epochs']):
                model.train(); losses=[]
                for start in range(0,len(x),params['batch_size']):
                    if start==0: order=rng.permutation(len(x))
                    ids=torch.as_tensor(order[start:start+params['batch_size']],device=args.device)
                    loss=(model(xt[ids])-yt[ids]).square().mean()
                    if not torch.isfinite(loss): raise ValueError('Nonfinite readout loss')
                    optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                score=evaluate(valid,predict,mean,std)['scene_macro']['ade']
                history.append({'epoch':epoch,'training_mse':float(np.mean(losses)),'validation_ade':score})
                if score<best_score: best_score=score; best_state=copy.deepcopy(model.state_dict()); stale=0; selected_epoch=epoch
                else: stale+=1
                if epoch%10==0: print('MLP_VALIDATION',condition,seed,epoch,score,flush=True)
                if stale>=params['early_stopping_patience']: break
            model.load_state_dict(best_state)
            output['results'].append({'condition':condition,'reader':'mlp_128_128','seed':seed,
                'parameter_count':sum(p.numel() for p in model.parameters()),'history':history,'selected_epoch':selected_epoch,
                'test':evaluate(test,predict,mean,std)})
            print('MLP_COMPLETE',condition,seed,'selected_epoch',selected_epoch,flush=True)
            torch.save({'state':{k:v.cpu() for k,v in best_state.items()},'mean':mean,'std':std},args.output/(condition+f'-mlp-seed{seed}.pt'))
        (args.output/'results.json').write_text(json.dumps(output,indent=2))
    print('PROBE_TRAINING_COMPLETE',flush=True)


if __name__=='__main__': main()
