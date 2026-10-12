"""Observed-only frozen model caches. Run only after explicit GPU authorization.

No simulator arrays are opened. Raw predictions remain in estimated camera/DA3
units and are NOT marked training-ready before the separate coordinate audit.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--allow-model-forward', action='store_true')
    ap.add_argument('--stage', choices=['geometry', 'forecast', 'all'], default='all')
    ap.add_argument('--limit', type=int, default=0, help='First scenes only for the GPU interface check')
    args = ap.parse_args()
    if not args.allow_model_forward:
        raise PermissionError('Model construction/forward requires explicit GPU authorization')
    base = args.base.resolve()
    ready = json.loads((base / 'logs/cpu-preparation-complete.json').read_text())
    if not ready['cpu_preparation_pass']:
        raise ValueError('CPU preparation incomplete')
    assets = json.loads((base / 'logs/assets-complete.json').read_text())
    cfg = json.loads((base / 'logs/effective-protocol.json').read_text())
    split = json.loads((base / 'split-manifest.json').read_text())['splits']
    scenes = sum(split.values(), [])
    if args.limit:
        scenes = scenes[:args.limit]
    sys.path.insert(0, assets['source'])
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'MODELSCOPE_OFFLINE']:
        os.environ[key] = '1'
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing CPU model loading')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    geometry_dir = base / 'observed_geometry'
    output_dir = base / 'model_cache_raw'
    geometry_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    def image_inputs(scene):
        folder = base / 'inputs' / scene
        paths = [folder / 'rgb' / f'{i:03d}.png' for i in range(7)]
        return folder, paths, [digest(p) for p in paths]

    if args.stage in ('geometry', 'all'):
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.io.input_processor import InputProcessor

        class SerialProcessor(InputProcessor):
            def __call__(self, *pos, **kw):
                kw.update(num_workers=1, sequential=True)
                return super().__call__(*pos, **kw)

        da3 = DepthAnything3.from_pretrained(str(base / 'weights/DA3-BASE')).to('cuda').eval()
        da3.requires_grad_(False)
        da3.input_processor = SerialProcessor()
        for scene in scenes:
            folder, paths, hashes = image_inputs(scene)
            target = geometry_dir / (scene + '.npz')
            if target.exists():
                with np.load(target, allow_pickle=False) as old:
                    if old['observed_rgb_sha256'].tolist() != hashes:
                        raise ValueError('Existing geometry input changed: ' + scene)
                continue
            pred = da3.inference([str(p) for p in paths], process_res=392,
                                 infer_gs=False, export_dir=None)
            depth = np.asarray(pred.depth)
            ext = np.asarray(pred.extrinsics)
            intr = np.asarray(pred.intrinsics)
            if depth.shape != (7, 392, 392) or intr.shape != (7, 3, 3):
                raise ValueError('DA3 output interface differs from audited preprocessing')
            if ext.shape == (7, 3, 4):
                full = np.tile(np.eye(4), (7, 1, 1))
                full[:, :3] = ext
                ext = full
            if ext.shape != (7, 4, 4) or not all(np.isfinite(x).all() for x in (depth, ext, intr)):
                raise ValueError('Invalid observed geometry')
            np.savez_compressed(target, depth=depth, extrinsics_w2c=ext,
                                intrinsics=intr, observed_rgb_sha256=hashes,
                                observed_frames=np.arange(7), future_rgb_supplied=False,
                                gt_input_supplied=False, process_res=392)
            print('OBSERVED_GEOMETRY_COMPLETE', scene, flush=True)
        del da3
        torch.cuda.empty_cache()

    if args.stage in ('forecast', 'all'):
        from cross_point_offline_runtime import install_offline_runtime
        from scripts.inference_model import build_model_from_ckpt
        from models_pretrained.future_scene_flow.sparse_dataset import (
            _apply_rigid, _compute_pj_norm, _unproject_world, _resize_rgb)
        restore = install_offline_runtime(base, enable_model_loading=True)
        model, state = build_model_from_ckpt(
            base / 'weights/motionforesight/trainedon_ss100k/best.pt',
            base / 'weights/trackcraft3r/model.safetensors')
        model.requires_grad_(False)
        model.eval()
        mc = model.config
        if (mc.obs_frames, mc.total_frames) != (7, 22):
            raise ValueError('Checkpoint time slots differ from protocol')

        def sample(values, uv):
            grid = torch.as_tensor(uv, dtype=torch.float32, device=values.device).clone()
            grid[:, 0] = grid[:, 0] * 2 / (mc.width - 1) - 1
            grid[:, 1] = grid[:, 1] * 2 / (mc.height - 1) - 1
            per_time = values[0].permute(1, 0, 2, 3)
            return F.grid_sample(per_time, grid.view(1, 1, -1, 2).expand(
                per_time.shape[0], -1, -1, -1), align_corners=True
            )[:, :, 0].permute(0, 2, 1).cpu().numpy()

        for scene in scenes:
            folder, paths, hashes = image_inputs(scene)
            target = output_dir / (scene + '.npz')
            query_path = folder / 'queries.json'
            query_hash = digest(query_path)
            if target.exists():
                with np.load(target, allow_pickle=False) as old:
                    if old['observed_rgb_sha256'].tolist() != hashes or str(old['query_sha256']) != query_hash:
                        raise ValueError('Existing model cache input changed: ' + scene)
                continue
            with np.load(geometry_dir / (scene + '.npz'), allow_pickle=False) as geometry:
                if geometry['observed_rgb_sha256'].tolist() != hashes:
                    raise ValueError('Observed geometry mismatch')
                depth = geometry['depth'].copy()
                ext = geometry['extrinsics_w2c'].copy()
                intr = geometry['intrinsics'].copy()
            query = json.loads(query_path.read_text())
            uv = np.asarray(query['query_uv_source_pixels'], dtype=np.float32)
            if query['query_source_frame'] != 0 or uv.shape != (48, 2):
                raise ValueError('Unexpected query interface')
            rgb = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])
            h, w = rgb.shape[1:3]
            pj = []
            for i in range(7):
                k = intr[i].copy()
                k[0] *= mc.width / depth.shape[2]
                k[1] *= mc.height / depth.shape[1]
                d = cv2.resize(depth[i], (mc.width, mc.height), interpolation=cv2.INTER_NEAREST)
                pj.append(_unproject_world(d, [k[0,0], k[1,1], k[0,2], k[1,2]], ext[i]))
            pj = _apply_rigid(ext[-1], np.stack(pj))
            norm = _compute_pj_norm(pj, 80., 2., 98.)
            normalized = ((pj - norm.mean) / norm.scale).astype(np.float32)
            rgb_t = torch.from_numpy(_resize_rgb(rgb, mc.height, mc.width)).unsqueeze(0).to(model.device)
            pj_t = torch.from_numpy(normalized.transpose(0, 3, 1, 2)).unsqueeze(0).to(model.device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                latent = model(rgb_t, pj_t)
                xyz_latent, _ = model.split_latent(latent)
                delta = model.decode_xyz(xyz_latent).float()
                tracks = model.reconstruct(delta, pj_t[:, 0].float())
                tracks = model.denormalize(tracks,
                    torch.from_numpy(norm.mean).unsqueeze(0).to(model.device),
                    torch.tensor([norm.scale], device=model.device))
                model_uv = uv * np.array([mc.width / w, mc.height / h], dtype=np.float32)
                points = sample(tracks, model_uv)
                hidden = sample(xyz_latent.float(), model_uv)
            if points.shape != (22, 48, 3) or hidden.shape != (22, 48, 16):
                raise ValueError('Decoded/latent temporal shape mismatch')
            if not np.isfinite(points).all() or not np.isfinite(hidden).all():
                raise ValueError('Nonfinite model cache')
            np.savez_compressed(target, observed_xyz=points[:7], baseline_future=points[7:],
                latent_future=hidden[7:], query_uv=uv, timestamps_s=np.arange(22) / 12,
                coordinate_frame='last_observed_camera', coordinate_units='DA3_estimated_units',
                camera_reference='DA3 estimated observed frame 6', training_coordinate_audit_pass=False,
                future_rgb_supplied=False, gt_input_supplied=False, teacher_used=False,
                observed_rgb_sha256=hashes, query_sha256=query_hash,
                pj_mean=norm.mean, pj_scale=norm.scale, source_depth_resolution=392)
            print('RAW_MODEL_CACHE_COMPLETE', scene, flush=True)
        restore()
    print('REQUESTED_GPU_CACHE_STAGE_COMPLETE; TRAINING_COORDINATE_AUDIT_STILL_REQUIRED', flush=True)


if __name__ == '__main__':
    main()
