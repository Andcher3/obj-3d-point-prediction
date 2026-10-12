"""CPU visualization from saved predictions and independently reconstructed GT."""
import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(rgb, uv, frame, title, visible=None):
    scale = 2
    height, width = rgb.shape[:2]
    canvas = cv2.copyMakeBorder(cv2.resize(rgb, (width*scale, height*scale)),
                               48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
    cv2.putText(canvas, title, (12,22), cv2.FONT_HERSHEY_SIMPLEX, .52, (245,245,245), 1)
    cv2.putText(canvas, f'frame={frame} | t0=6 | same colors = same query',
                (12,40), cv2.FONT_HERSHEY_SIMPLEX, .43, (200,200,200), 1)
    for q in range(uv.shape[1]):
        color = tuple(round(v*255) for v in colorsys.hsv_to_rgb(q/uv.shape[1],.8,1)[::-1])
        def inside(p):
            return np.isfinite(p).all() and 0 <= p[0] < width and 0 <= p[1] < height
        def screen(p):
            return tuple(np.rint((p+.5)*scale-.5+[0,48]).astype(int))
        for t in range(max(1,frame-3),frame+1):
            if inside(uv[t-1,q]) and inside(uv[t,q]):
                cv2.line(canvas, screen(uv[t-1,q]), screen(uv[t,q]), color, 1, cv2.LINE_AA)
        if inside(uv[frame,q]):
            point=screen(uv[frame,q])
            cv2.circle(canvas, point, 5, (0,0,0), -1, cv2.LINE_AA)
            hidden=visible is not None and not visible[frame,q]
            cv2.circle(canvas, point, 3, color, 1 if hidden else -1, cv2.LINE_AA)
    return canvas


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    args=ap.parse_args()
    status=json.loads((args.base/'logs/gt-preparation-status.json').read_text())
    records=[]
    for scene in status['scenes']:
        if scene['status']!='READY':
            continue
        name=scene['scene_id']
        directory=args.base/'inputs'/name
        output=args.base/'outputs'/name
        raw_path=output/f'{name}_22frames_raw.npz'
        raw_sha=sha(raw_path)
        manifest=json.loads((directory/'queries.json').read_text())
        with np.load(raw_path, allow_pickle=False) as raw, np.load(directory/'gt_tracks.npz', allow_pickle=False) as gt:
            tracks=[raw['tracks_2d_source_pixels'].copy(),
                    raw['tracks_3d_projected_source_pixels'].copy(),gt['tracks_2d'].copy()]
            visibility=gt['visibility'].copy()
        if any(t.shape!=(22,48,2) for t in tracks):
            raise ValueError('Unexpected track shape: '+name)
        video=output/f'{name}_gt_review.mp4'
        if video.exists():
            raise FileExistsError(video)
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),8,(1536,560))
        if not writer.isOpened():
            raise RuntimeError('Video writer failed')
        samples=[]
        try:
            for frame in range(22):
                rgb=cv2.imread(str(directory/'rgb'/f'{frame:03d}.png'))
                if rgb is None or rgb.shape[:2]!=(256,256):
                    raise ValueError('Bad source RGB')
                panels=[panel(rgb,tracks[0],frame,'Model: direct 2D correspondence'),
                        panel(rgb,tracks[1],frame,'Model: 3D + predicted camera'),
                        panel(rgb,tracks[2],frame,'Simulator GT (hollow = hidden)',visibility)]
                canvas=np.hstack(panels)
                writer.write(canvas)
                if frame==6:
                    assert cv2.imwrite(str(output/f'{name}_gt_review_t0.png'),canvas)
                if frame in (0,6,13,21):
                    samples.append(cv2.resize(canvas,(1152,420)))
        finally:
            writer.release()
        capture=cv2.VideoCapture(str(video))
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count!=22:
            raise RuntimeError('Saved video verification failed')
        capture.release()
        assert cv2.imwrite(str(output/f'{name}_gt_review_contact.jpg'),np.vstack(samples))
        if sha(raw_path)!=raw_sha:
            raise ValueError('Raw predictions were modified')
        record={'scene_id':name,'frames':count,'panels':['raw_2d','raw_3d_projection','simulator_gt'],
                'raw_sha256':raw_sha,'review_video_sha256':sha(video),
                'gt_source_sha256':sha(directory/'gt_tracks.npz'),
                'gt_visibility':'approximate instance+depth test; all finite GT retained',
                'native_fps':12,'playback_fps':8,'pointwise_alignment':False}
        records.append(record)
        print(json.dumps(record),flush=True)
    result={'render_completed':True,'scene_count':len(records),'scenes':records,
            'raw_outputs_unchanged':True,'model_forward_called':False}
    (args.base/'logs/gt-visual-render-audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
