"""CPU-only visible surface graph preparation; reads observed frame 0 only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from prepare_uniquery4r_movif_gt import decode_images, material_tracks, records, hash_file


def surface_relations(xyz,mask,query_xy,depth,cfg):
    xyz=np.asarray(xyz,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    query_xy=np.asarray(query_xy,dtype=int)
    if xyz.shape!=(*mask.shape,3) or depth.shape!=mask.shape:
        raise ValueError('Surface raster dimensions mismatch')
    if not np.isfinite(xyz[mask]).all() or not np.isfinite(depth[mask]).all():
        raise ValueError('Nonfinite target surface geometry')
    h,w=mask.shape
    idx=np.full((h,w),-1,dtype=np.int32);idx[mask]=np.arange(mask.sum())
    query_nodes=idx[query_xy[:,1],query_xy[:,0]]
    if (query_nodes<0).any() or len(np.unique(query_nodes))!=len(query_nodes):
        raise ValueError('Query not a distinct visible target pixel')
    points=xyz[mask];queries=points[query_nodes]
    radius=float(np.sqrt(np.mean(np.sum((queries-queries.mean(0))**2,axis=-1))))
    if radius<1e-10:raise ValueError('Degenerate surface query geometry')
    yy,xx=np.nonzero(mask);src=[];dst=[]
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        target_y=yy+dy;target_x=xx+dx
        keep=(target_y>=0)&(target_y<h)&(target_x>=0)&(target_x<w)
        sy,sx=yy[keep],xx[keep];ty,tx=target_y[keep],target_x[keep]
        keep=mask[ty,tx]
        if dy and dx:keep &= mask[sy,tx]&mask[ty,sx]
        src.append(idx[sy[keep],sx[keep]]);dst.append(idx[ty[keep],tx[keep]])
    src=np.concatenate(src);dst=np.concatenate(dst)
    length=np.linalg.norm(points[src]-points[dst],axis=-1)
    radial=depth[mask]
    jump=np.abs(radial[src]-radial[dst])/np.maximum(np.minimum(radial[src],radial[dst]),1e-8)
    positive=length>1e-12
    if not positive.any():raise ValueError('No nondegenerate visible surface edges')
    median=float(np.median(length[positive]))
    accepted=positive&(jump<=cfg['mesh_edge_depth_ratio_limit'])&(length<=cfg['mesh_edge_length_median_multiplier']*median)
    rows=np.r_[src[accepted],dst[accepted]];cols=np.r_[dst[accepted],src[accepted]]
    weights=np.tile(length[accepted]/radius,2)
    graph=coo_matrix((weights,(rows,cols)),shape=(len(points),len(points))).tocsr()
    path=dijkstra(graph,directed=False,indices=query_nodes)[:,query_nodes]
    straight=np.linalg.norm(queries[:,None]-queries[None,:],axis=-1)/radius
    finite=np.isfinite(path)
    if np.any(path[finite]<straight[finite]-1e-7):
        raise ValueError('Path shorter than Euclidean chord: graph construction error')
    _,components=connected_components(graph,directed=False)
    q_components=components[query_nodes]
    upper=np.triu_indices(len(queries),1)
    return {'surface_path':path,'straight_distance':straight,
            'query_surface_components':q_components,
            'query_xyz':queries,'radius':radius}, {
                'surface_vertex_count':len(points),'accepted_undirected_edges':int(accepted.sum()),
                'rejected_undirected_edges':int((~accepted).sum()),
                'query_connected_components':len(np.unique(q_components)),
                'reachable_query_pair_fraction':float(finite[upper].mean()),
                'distance_kind':'discrete graph shortest path on visible pixel surface',
                'complete_surface_geodesic':False}


def estimated_surface(geometry,mask):
    h,w=mask.shape;depth=geometry['depth'][0];intr=geometry['intrinsics'][0]
    u,v=np.meshgrid(np.arange(w)*depth.shape[1]/w,np.arange(h)*depth.shape[0]/h)
    z=cv2.remap(depth.astype(np.float32),u.astype(np.float32),v.astype(np.float32),
                cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    xyz=np.stack([(u-intr[0,2])/intr[0,0]*z,(v-intr[1,2])/intr[1,1]*z,z],-1)
    radial=np.linalg.norm(xyz,axis=-1)
    return xyz,radial


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text())
    if args.output.exists():raise FileExistsError('Preserve old surface caches')
    args.output.mkdir(parents=True)
    splits=json.loads((args.base/'split-manifest.json').read_text())['splits']
    audits={};by_shard={}
    for split,scenes in splits.items():
        for scene in scenes:
            folder=args.base/'inputs'/scene
            audit=json.loads((folder/'gt_audit.json').read_text())
            by_shard.setdefault(audit['shard'],{})[audit['record_index']]=(scene,split)
    rows=[]
    for shard,selected in by_shard.items():
        for record_index,annot in enumerate(records(args.base/'data'/shard)):
            if record_index not in selected:continue
            scene,split=selected[record_index];folder=args.base/'inputs'/scene
            query=json.loads((folder/'queries.json').read_text())
            if annot['metadata/video_name'][0].decode()!=scene.removeprefix('scene_'):
                raise ValueError('Shard scene identity mismatch')
            xy=np.asarray(query['query_pixel_sample_xy'],dtype=int)
            # Only frame0 is decoded: no future geometry enters either condition.
            local=decode_images([annot['object_coordinates'][0]],channels=3)[0]
            seg=decode_images([annot['segmentations'][0]])[0]
            instance=query['target_instance_mask_id'];mask=seg==instance
            original_mask=cv2.imread(str(folder/'target_mask_t0.png'),cv2.IMREAD_GRAYSCALE)>0
            np.testing.assert_array_equal(mask,original_mask)
            with np.load(folder/'gt_tracks.npz',allow_pickle=False) as gt:
                box=gt['target_bboxes_world'][0].copy()
                np.testing.assert_array_equal(local[xy[:,1],xy[:,0]],gt['object_coordinates_t0_uint16'])
                K=gt['intrinsic_gt'][0].copy();pose=gt['extrinsic_w2c_gt'][0].copy()
                gt_queries=gt['tracks_world'][0].copy()
            coords=np.zeros((*mask.shape,3),dtype=np.float64)
            coords[mask]=material_tracks(local[mask],box[None])[0]
            np.testing.assert_allclose(coords[xy[:,1],xy[:,0]],gt_queries,atol=1e-6)
            # Radial depth is used only to suppress discontinuity edges.
            camera=coords@pose[:3,:3].T+pose[:3,3]
            gt_depth=np.linalg.norm(camera,axis=-1)
            oracle,oracle_audit=surface_relations(coords,mask,xy,gt_depth,cfg)
            with np.load(args.base/'observed_geometry'/(scene+'.npz'),allow_pickle=False) as geometry:
                xyz,depth=estimated_surface(geometry,mask)
            estimated,estimated_audit=surface_relations(xyz,mask,xy,depth,cfg)
            values={prefix+'_'+key:value for prefix,data in [('estimated',estimated),('oracle',oracle)]
                    for key,value in data.items()}
            np.savez_compressed(args.output/(scene+'.npz'),**values,
                                query_sha256=hash_file(folder/'queries.json').hex(),
                                future_geometry_used=False)
            row={'scene':scene,'split':split,'estimated':estimated_audit,'oracle':oracle_audit}
            rows.append(row);audits[scene]=row
            print('SURFACE_CACHE',scene,estimated_audit['query_connected_components'],
                  oracle_audit['query_connected_components'],flush=True)
            (args.output/'progress.json').write_text(json.dumps({'completed':len(rows),'expected':160}))
    if set(audits)!=set(sum(splits.values(),[])):
        raise ValueError('Incomplete surface preparation')
    result={'rows':rows,'counts':{k:len(v) for k,v in splits.items()},
            'surface_domain':cfg['surface_domain'],'future_geometry_used':False,
            'training_started':False,'GPU_used':False,'all_scenes_retained':True,
            'full_mesh_archives_used':False}
    (args.output/'surface-audit.json').write_text(json.dumps(result,indent=2))
    print('VISIBLE_SURFACE_PREPARATION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
