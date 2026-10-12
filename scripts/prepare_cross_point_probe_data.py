"""CPU-only GT preparation; remote aria2 downloads, no model imports/forward."""
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from prepare_uniquery4r_movif_gt import (hash_file, prepare_scene, records,
                                       verify_shard, write_json, write_input_hashes)


def public_metadata(name):
    session = requests.Session()
    session.trust_env = False
    url = 'https://storage.googleapis.com/storage/v1/b/kubric-public/o/' + urllib.parse.quote(name, safe='')
    response = session.get(url, timeout=(15, 45))
    response.raise_for_status()
    info = response.json()
    return {'name':name.rsplit('/',1)[1], 'object_name':name, 'bytes':int(info['size']),
            'md5_base64':info['md5Hash'], 'generation':info['generation']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    args = ap.parse_args()
    base, cfg = args.base, json.loads(args.config.read_text())
    for part in ['data','logs','inputs','cpu_cache']:
        (base/part).mkdir(parents=True,exist_ok=True)
    if (base/'logs/preparation-complete.json').exists():
        raise FileExistsError('Preparation already completed; do not overwrite fixed split')
    session = requests.Session(); session.trust_env=False
    r=session.get(cfg['source_base']+'/dataset_info.json',timeout=(15,45)); r.raise_for_status()
    dataset_info=r.json(); write_json(base/'logs/dataset_info.json',dataset_info)
    splits={s['name']:s for s in dataset_info['splits']}
    names=[]
    for split, start, count in [('train',0,cfg['download']['train_shards']),
                              ('validation',cfg['download']['validation_shard_start'],cfg['download']['validation_shards'])]:
        total=len(splits[split]['shardLengths'])
        for index in range(start,start+count):
            names.append((split,index,f'tfds/movi_f/256x256/1.0.0/movi_f-{split}.tfrecord-{index:05d}-of-{total:05d}'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        entries=list(pool.map(public_metadata,[n[2] for n in names]))
    for entry,(split,index,_) in zip(entries,names):
        entry.update(split=split,index=index,scene_count=int(splits[split]['shardLengths'][index]))
    total=sum(e['bytes'] for e in entries)
    if total > cfg['download']['max_source_gib']*1024**3:
        raise ValueError('Source download exceeds fixed size budget')
    if shutil.disk_usage(base).free < total+8*1024**3:
        raise ValueError('Insufficient data-disk space with 8GiB reserve')
    write_json(base/'logs/source-manifest.json',{'entries':entries,'total_bytes':total,'gpu_used':False})
    queue=[]
    for e in entries:
        p=base/'data'/e['name']
        if verify_shard(p,e): continue
        queue.append(cfg['source_base']+'/'+e['name']+'?generation='+e['generation']+'\n  dir='+str(base/'data')+'\n  out='+e['name']+'\n  checksum=md5='+base64.b64decode(e['md5_base64']).hex()+'\n')
    queue_path=base/'logs/aria2-queue.txt'; queue_path.write_text(''.join(queue))
    env=os.environ.copy()
    for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
        env.pop(key,None)
    if queue:
        with (base/'logs/aria2.log').open('a') as log:
            subprocess.run(['aria2c','-c','-j','3','-x','16','-s','16','--min-split-size=1M',
                '--file-allocation=none','--summary-interval=15','--console-log-level=warn',
                '--max-tries=3','--connect-timeout=15','--timeout=30','--allow-overwrite=true',
                '--auto-file-renaming=false','-i',str(queue_path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # The CPU preparation protocol may be corrected during a long download;
    # reread it before any scene is created, and preserve the effective bytes.
    current_cfg=json.loads(args.config.read_text())
    if current_cfg['download']!=cfg['download']: raise ValueError('Download scope changed during download')
    cfg=current_cfg
    write_json(base/'logs/effective-protocol.json',cfg)
    chosen={'train':[],'validation':[],'test':[]}; seen=set(cfg['old_development_scenes'])
    all_rows=[]; source_rows=[]
    for e in entries:
        path=base/'data'/e['name']
        if not verify_shard(path,e): raise ValueError('Publisher MD5 mismatch: '+e['name'])
        source_rows.append({**e,'publisher_md5_matches':True,'sha256':hash_file(path).hex()})
        record_count=0
        for i,annot in enumerate(records(path)):
            record_count+=1
            scene='scene_'+annot['metadata/video_name'][0].decode().replace('/','_')
            if e['split']=='train':
                dest='train' if len(chosen['train'])<cfg['quotas']['train'] else 'validation'
            else: dest='test'
            if len(chosen[dest])>=cfg['quotas'][dest]: continue
            if scene in seen:
                all_rows.append({'scene_id':scene,'status':'EXCLUDED','reason':'already_seen_scene','source':e['name']})
                continue
            seen.add(scene)
            row=prepare_scene(base,cfg,annot,e['name'],i)
            row['research_split']=dest; all_rows.append(row)
            if row['status']=='READY':
                folder=base/'inputs'/scene
                with np.load(folder/'gt_tracks.npz',allow_pickle=False) as data:
                    xyz=data['tracks_last_observed_camera']; uv=data['tracks_2d']
                    t0=cfg['frame_protocol']['observed_frames']-1; points=xyz.shape[1]
                    reference=cfg['frame_protocol']['query_frame']
                    distance=np.linalg.norm(uv[reference,:,None]-uv[reference,None,:],axis=-1)
                    np.fill_diagonal(distance,np.inf)
                    neighbors=np.argsort(distance,axis=1,kind='stable')[:,:cfg['neighbor_count']]
                    # Simulator past is an explicitly labelled diagnostic, kept
                    # separate from the future-only loss targets and RGB inputs.
                    np.savez_compressed(base/'cpu_cache'/(scene+'-ideal-past.npz'),
                        observed_xyz=xyz[:t0+1],observed_uv=uv[:t0+1],
                        observed_visibility=data['visibility'][:t0+1],neighbors=neighbors,
                        timestamps_s=data['timestamps_s'][:t0+1])
                    np.savez_compressed(base/'cpu_cache'/(scene+'-labels.npz'),
                        future_xyz=xyz[t0+1:],validity=data['validity'][t0+1:],
                        visibility=data['visibility'][t0+1:],timestamps_s=data['timestamps_s'][t0+1:])
                write_input_hashes(base,folder)
                chosen[dest].append(scene)
            write_json(base/'logs/data-progress.json',{'counts':{k:len(v) for k,v in chosen.items()},'scenes':all_rows,'model_forward_called':False})
            print(json.dumps(row),flush=True)
        if record_count!=e['scene_count']: raise ValueError('TFRecord scene count mismatch')
    counts={k:len(v) for k,v in chosen.items()}
    if counts!=cfg['quotas']: raise ValueError('Scene quotas not met: '+str(counts))
    assert len(set(sum(chosen.values(),[])))==sum(counts.values())
    write_json(base/'split-manifest.json',{'splits':chosen,'counts':counts,'old_scenes_excluded':cfg['old_development_scenes'],
        'selection':'fixed official shard/record order, observed-only target rule; no prediction-result selection',
        'source_manifest_sha256':hash_file(base/'logs/source-manifest.json').hex()})
    write_json(base/'logs/source-audit.json',{'sources':source_rows,'all_publisher_md5_match':True})
    for split,scenes in chosen.items():
        tiles=[]
        for scene in scenes:
            im=cv2.imread(str(base/'inputs'/scene/'query_preview.png'))
            tiles.append(cv2.resize(im,(256,270)))
        rows=[]
        for i in range(0,len(tiles),5):
            items=tiles[i:i+5]; items+= [np.zeros_like(items[0])]*(5-len(items))
            rows.append(np.hstack(items))
        cv2.imwrite(str(base/(split+'-query-contact.jpg')),np.vstack(rows))
    status={'cpu_data_ready':True,'counts':counts,'publisher_md5_all_pass':True,
        'gt_audit_all_pass':True,'splits_disjoint':True,'future_gt_input':False,
        'model_forward_called':False,'training_started':False,'gpu_used':False,
        'observed_model_cache_ready':False,'model_assets_checked':False,
        'next_gpu_stage':'generate observed-only model caches after user enables GPU'}
    write_json(base/'logs/preparation-complete.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__': main()
