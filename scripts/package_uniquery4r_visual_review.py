"""Local review packaging of saved real outputs; no model or quality metrics."""
import argparse
import ast
import colorsys
import hashlib
import html
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--renderer', type=Path, required=True)
    args = parser.parse_args()
    tree = ast.parse(args.renderer.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if getattr(node, 'name', '') == 'draw_panel']
    assert len(selected) == 1
    namespace = {'cv2': cv2, 'np': np, 'colorsys': colorsys}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.renderer), 'exec'), namespace)
    sections, summary = [], []
    for video in ('car-turn', 'drift-turn', 'boat', 'soccerball'):
        directory = args.base / 'outputs' / video
        manifest = json.loads((args.base / 'inputs' / video / 'queries.json').read_text())
        t0 = manifest['query_model_frame_index']
        rgb = cv2.imread(str(args.base / 'inputs' / video / 'rgb' /
                            f"{manifest['query_source_frame_index']:05d}.jpg"))
        assert rgb is not None
        panels = [namespace['draw_panel'](rgb, np.asarray(manifest['query_uv_source_pixels'])[None],
                                         0, f'{video} | approved input query | t0')]
        items, videos = [], []
        for count in (22, 7):
            name = f'{video}_{count}frames'
            raw_path = directory / f'{name}_raw.npz'
            raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            with np.load(raw_path) as raw:
                assert raw['tracks_3d'].shape == (count, 48, 3)
                assert raw['tracks_3d_projected_source_pixels'].shape == (count, 48, 2)
                panels.append(namespace['draw_panel'](
                    rgb, raw['tracks_3d_projected_source_pixels'][t0:t0+1], 0,
                    f'{count} frames | raw 3D projection at t0 | no alignment'))
            source = directory / f'{name}.mp4'
            output = directory / f'{name}_browser.mp4'
            if output.exists():
                raise FileExistsError(output)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(source),
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', '-an', str(output)], check=True)
            capture = cv2.VideoCapture(str(output))
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == count
            assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1708
            assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 520
            capture.release()
            assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_hash
            items.append({'context_frames': count, 'raw_sha256': raw_hash,
                          'browser_video': output.name, 'decoded_frames': count})
            title = '22帧完整视频跟踪（教师候选）' if count == 22 else '7帧观测视频跟踪'
            videos.append(f'<h3>{title}</h3><video controls preload="metadata" '
                          f'src="outputs/{video}/{output.name}"></video>')
        comparison = directory / f'{video}_t0_teacher_observed_compare.png'
        assert cv2.imwrite(str(comparison), np.hstack(panels))
        # Sample decoded frames for visual inspection, not statistical scoring.
        capture = cv2.VideoCapture(str(directory / f'{video}_22frames_browser.mp4'))
        samples = []
        for frame in (0, 6, 13, 21):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            assert ok
            samples.append(cv2.resize(image, (1024, 312)))
        capture.release()
        assert cv2.imwrite(str(directory / f'{video}_22frames_contact_sheet.png'), np.vstack(samples))
        summary.append({'video': video, 'contexts': items, 't0_comparison': comparison.name})
        sections.append(f'<section><h2>{html.escape(video)}</h2>' + ''.join(videos) +
                        f'<h3>t0：批准输入点 / 22帧三维投影 / 7帧三维投影</h3>'
                        f'<img src="outputs/{video}/{comparison.name}"></section>')
    (args.base / 'visual-review.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UniQuery4R 轨迹审阅</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#171717;color:#eee}'
        'video,img{max-width:100%;display:block}section{margin:36px 0}h3{font-size:17px}</style>'
        '<h1>UniQuery4R 四物体视频轨迹审阅</h1><p>视频左侧为二维对应位置，右侧为三维投影。'
        '颜色表示相同query编号；轨迹为原始输出，未强制对齐。8fps仅为播放速度。'
        '22帧跟踪使用完整视频，属于离线教师候选，并非未来预测。</p>' + ''.join(sections) + '</html>',
        encoding='utf-8')
    result = {'packaging_checks_passed': True, 'model_forward_called': False,
              'numerical_quality_evaluation_started': False, 'raw_outputs_unchanged': True,
              'videos': summary}
    (args.base / 'visual-packaging-checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
