"""CPU-only inventory for a surface-distance ablation, without model or training.

Downloads only the public GSO JSON catalog with aria2, not any mesh archive.
Existing scene data and forecast caches are read-only.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.parse

import numpy as np
import requests

CATALOG_OBJECT='assets/GSO/GSO.json'


def rigidity_diagnostic(tracks):
    tracks=np.asarray(tracks,dtype=np.float64)
    if tracks.ndim!=3 or tracks.shape[-1]!=3 or not np.isfinite(tracks).all():
        raise ValueError('Invalid GT trajectory array')
    i,j=np.triu_indices(tracks.shape[1],1)
    lengths=np.linalg.norm(tracks[:,i]-tracks[:,j],axis=-1)
    reference=lengths[0]
    usable=reference>1e-6
    drift=np.abs(lengths[:,usable]-reference[usable])/reference[usable]
    return {'pair_length_drift_relative_p90':float(np.quantile(drift,.9)),
            'pair_length_drift_relative_max':float(drift.max()),
            'near_zero_pairs_excluded':int((~usable).sum()),
            'meaning':'diagnostic of this simulated object, not a required method assumption'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior readiness audit')
    args.output.mkdir(parents=True)
    splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    session=requests.Session();session.trust_env=False
    endpoint='https://storage.googleapis.com/storage/v1/b/kubric-public/o/'+urllib.parse.quote(CATALOG_OBJECT,safe='')
    response=session.get(endpoint,timeout=(15,45));response.raise_for_status()
    meta=response.json()
    if int(meta['size'])>8*1024**2:
        raise ValueError('Catalog exceeds fixed 8MiB download budget')
    expected=base64.b64decode(meta['md5Hash']).hex()
    env=os.environ.copy()
    for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
        env.pop(key,None)
    with (args.output/'catalog-aria2.log').open('w') as log:
        subprocess.run(['aria2c','-c','-x','16','-s','16','--min-split-size=1M',
                        '--file-allocation=none','--max-tries=1','--connect-timeout=15','--timeout=30',
                        '--auto-file-renaming=false','--checksum=md5='+expected,
                        '--dir='+str(args.output),'--out=GSO.json',
                        'https://storage.googleapis.com/kubric-public/'+CATALOG_OBJECT+'?generation='+meta['generation']],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    raw=(args.output/'GSO.json').read_bytes()
    if len(raw)!=int(meta['size']) or hashlib.md5(raw).hexdigest()!=expected:
        raise ValueError('Catalog publisher size/MD5 mismatch')
    catalog=json.loads(raw);rows=[]
    for split,scenes in splits.items():
        for scene in scenes:
            folder=args.base/'inputs'/scene
            query=json.loads((folder/'queries.json').read_text())
            asset=query['target_asset_id']
            with np.load(folder/'gt_tracks.npz',allow_pickle=False) as gt:
                required=['tracks_world','object_coordinates_t0_uint16','target_bboxes_world']
                if not all(k in gt for k in required):
                    raise ValueError('Missing material-point fields: '+scene)
                diagnostic=rigidity_diagnostic(gt['tracks_world'])
                rows.append({'scene':scene,'split':split,'asset_id':asset,
                             'query_count':int(gt['tracks_world'].shape[1]),
                             'material_point_ids_available':True,
                             'asset_bbox_transform_available':True,
                             'mesh_vertices_in_scene_cache':False,
                             'mesh_faces_in_scene_cache':False,
                             'asset_present_in_official_catalog':asset in catalog['assets'],
                             **diagnostic})
    ids=sorted({r['asset_id'] for r in rows})
    selected={asset:catalog['assets'][asset] for asset in ids if asset in catalog['assets']}
    (args.output/'selected-assets.json').write_text(json.dumps(
        {'data_dir':catalog.get('data_dir','gs://kubric-public/assets/GSO'),
         'assets':selected},indent=2))
    report={'stage':'CPU surface groundtruth readiness, no experiment outcome',
            'scenes':rows,'counts':{k:len(v) for k,v in splits.items()},
            'unique_target_assets':len(ids),'catalog_matches':len(selected),
            'catalog_publisher_md5_matches':True,'catalog_bytes':len(raw),
            'catalog_sha256':hashlib.sha256(raw).hexdigest(),
            'data_disk_free_bytes':shutil.disk_usage(args.base).free,
            'mesh_archives_downloaded':False,'surface_distances_computed':False,
            'model_loaded':False,'GPU_used':False,'training_started':False,
            'assets_first_entry_example':next(iter(selected.values())) if selected else None,
            'remaining':['download selected visual mesh archives if design approved',
                         'verify material coordinates against rendered visual mesh',
                         'audit connectivity and path distance approximation',
                         'define differentiable supervision separately from observed surface features']}
    (args.output/'readiness.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['scenes','assets_first_entry_example']},indent=2))


if __name__=='__main__':main()
