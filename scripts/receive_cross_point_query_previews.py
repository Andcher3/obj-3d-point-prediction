"""Verify the preview-only package and build a local, offline query gallery."""
import argparse
import hashlib
import html
import json
import zipfile
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', type=Path, required=True)
    args = ap.parse_args()
    folder = args.folder.resolve()
    archive = folder / 'query-previews-cpu.zip'
    package = json.loads((folder / 'query-preview-package.json').read_text())
    if sha(archive) != package['sha256'] or archive.stat().st_size != package['bytes']:
        raise ValueError('Preview transfer checksum failed')
    root = folder / 'query-review-2026-10-11'
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Preview CRC failed')
        for name in z.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe archive path')
        z.extractall(root)
    files = json.loads((root / 'logs/query-preview-files.json').read_text())
    for name, record in files.items():
        p = root / name
        if sha(p) != record['sha256'] or p.stat().st_size != record['bytes']:
            raise ValueError('Preview file hash mismatch: ' + name)
    split = json.loads((folder / 'split-manifest.json').read_text())['splits']
    chunks = ['<!doctype html><meta charset="utf-8"><title>跨点实验 query 审阅</title>',
              '<style>body{background:#182026;color:#eef2f4;font:16px Microsoft YaHei,sans-serif;margin:28px}',
              'a{color:#a9d7ff}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:18px}',
              'figure{margin:0;background:#28333b;padding:10px}img{width:100%;height:auto}h2{margin-top:36px}</style>',
              '<h1>跨点信息实验：160 个场景的 query 预览</h1>',
              '<p>每场景 48 点；采样帧为第 0 帧，观测为 0–6 帧。点均位于目标 mask 收缩 2 像素后的区域。点击图片查看原图。</p>',
              '<p>预览包含采样帧，不含未来视频。自动检查通过不等于人工确认。</p>']
    labels = {'train': '训练', 'validation': '验证', 'test': '测试'}
    for group, scenes in split.items():
        chunks.append(f'<h2>{labels[group]}：{len(scenes)} 场景</h2><div class="grid">')
        for scene in scenes:
            p = f'inputs/{scene}/query_preview.png'
            chunks.append(f'<figure><a href="{p}" target="_blank"><img loading="lazy" src="{p}" alt="{html.escape(scene)}"></a><figcaption>{html.escape(scene)}</figcaption></figure>')
        chunks.append('</div>')
    (root / 'query-review.html').write_text('\n'.join(chunks), encoding='utf-8')
    (root / 'local-verification.json').write_text(json.dumps({
        'archive_sha256_pass': True, 'archive_crc_pass': True,
        'all_file_hashes_pass': True, 'files': len(files), 'scenes': 160,
        'query_user_review': 'pending', 'future_rgb_included': False}, indent=2))
    print('ALL_PREVIEWS_TRANSFERRED_AND_VERIFIED', root / 'query-review.html')


if __name__ == '__main__':
    main()
