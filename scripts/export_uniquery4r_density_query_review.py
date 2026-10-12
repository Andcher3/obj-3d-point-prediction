"""Export only small approved-sampling review assets; no raw videos or model."""
import argparse
import html
import json
import zipfile
from pathlib import Path

from prepare_uniquery4r_movif_gt import hash_file, write_json


def export(base):
    status = json.loads((base / "preparation.json").read_text())
    rows = status["scenes"]
    content = []
    files = ["preparation.json", "query_contact_sheet.jpg", "logs/inference-input-audit.json", "logs/official-preprocessing-cpu.json"]
    for r in rows:
        directory = "inputs/" + r["scene_id"]
        files.extend(directory + "/" + f for f in ("query_preview.png", "queries.json", "sampling.json", "gt_audit.json"))
        content.append(f'<article><h2>{html.escape(r["scene_id"])} · {r["query_count"]}点</h2><p>{"沿用旧48点" if r["fallback_byte_exact"] else "固定间距网格采样"} · 第6帧query</p><a href="{directory}/query_preview.png"><img src="{directory}/query_preview.png" alt="query preview"></a></article>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第6帧稀疏query预览</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;background:#10151c;color:#eee;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}article{background:#1b2532;padding:14px;border-radius:8px}img{width:100%;height:auto}h2{font-size:20px}p{line-height:1.6}</style><h1>第6帧query · 22帧输入</h1><p>256×256原图名义间隔8.095px；整幅图约1000点的固定网格，取目标物体区域。有效网格点不足48时精确沿用旧48点。仅CPU采样准备；未运行模型，未产生新预测。以下图片显示已知query位置，不表示模型结果。</p><main>' + "".join(content) + "</main></html>"
    (base / "query-review.html").write_text(page, encoding="utf-8")
    files.append("query-review.html")
    hashes = {file: hash_file(base / file).hex() for file in files}
    write_json(base / "review-sha256.json", hashes)
    archive = base / "query-review.zip"
    if archive.exists():
        raise FileExistsError("Preserve prior archive")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file in files + ["review-sha256.json"]:
            z.write(base / file, file)
    print(json.dumps({"archive_sha256": hash_file(archive).hex(), "archive_bytes": archive.stat().st_size, "review_files": len(files), "scenes": len(rows)}))


def verify(base):
    hashes = json.loads((base / "review-sha256.json").read_text())
    for file, expected in hashes.items():
        target = (base / file).resolve()
        if not target.is_relative_to(base.resolve()) or hash_file(target).hex() != expected:
            raise ValueError("Unsafe path or hash mismatch: " + file)
    print(json.dumps({"review_file_sha256_passed": True, "verified_files": len(hashes), "model_forward_called": False}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    (verify if args.verify else export)(args.base)
