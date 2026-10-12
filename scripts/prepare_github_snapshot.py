"""Prepare a GitHub snapshot without altering local datasets or credentials.

Run without --write to inspect the plan; --write adds ignore rules and placeholders.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "github_upload_manifest.json"
PLACEHOLDER = "GITHUB_PLACEHOLDERS.md"
BINARY_DATA = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5", ".hdf5", ".pkl", ".pickle", ".tfrecord", ".parquet"}
ARCHIVES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
CACHE_PARTS = {".cache", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv", "venv", "_tmp_contact"}
DATA_PARTS = {"UCI HAR Dataset", "DAVIS", "JPEGImages", "Annotations", "runtime_inputs"}
SECRET_PATTERNS = {
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "provider_token": re.compile(r"\b(?:hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{25,}|rpa_[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"https?://[^\s/:]+:[^\s/@]{8,}@"),
    "literal_credential": re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?key)["']?\s*[:=]\s*["']([A-Za-z0-9_+/.=-]{20,})["']'''),
}


def reason_for(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(part in CACHE_PARTS for part in parts):
        return "cache_or_temporary_output"
    if any(part in DATA_PARTS for part in parts) or ("pointodyssey-interface-preflight-2026-09-26" in parts and "sample" in parts):
        return "local_dataset_or_runtime_input"
    if path.suffix.lower() in BINARY_DATA:
        return "dataset_array_or_model_weight"
    if path.suffix.lower() in ARCHIVES:
        return "download_or_runtime_archive"
    if path.name.startswith(".env") and path.name not in {".env.example", ".env.template"}:
        return "local_secret_configuration"
    if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
        return "private_credential_file"
    if path.stat().st_size >= 10 * 1024**2:
        return "large_file_at_least_10_MiB"
    if path.name.lower() in {"nul", "con", "prn", "aux"} or (len(parts) == 1 and path.name.isdecimal()):
        return "nonportable_or_temporary_filename"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    omitted, included, secret_hits = [], [], []
    generated = {".gitignore", "github_upload_manifest.json", "GITHUB_UPLOAD.md"}
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if ".git" in path.relative_to(ROOT).parts or not path.is_file():
            continue
        if path.name == PLACEHOLDER or rel in generated:
            continue
        size = path.stat().st_size
        reason = reason_for(path)
        hits = []
        if not reason:
            raw = path.read_bytes()
            texts = []
            if path.suffix.lower() in {".docx", ".pptx", ".xlsx"} and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as document:
                    texts = [(name, document.read(name).decode("utf-8", errors="replace")) for name in document.namelist() if name.endswith(".xml")]
            elif b"\x00" not in raw[:8192]:
                try:
                    content = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    content = ""
                texts = [("", content)]
            for member, content in texts:
                for label, pattern in SECRET_PATTERNS.items():
                    for match in pattern.finditer(content):
                        # Output only locations and categories, never matched credentials.
                        hits.append({"category": label, "member": member, "line": content.count("\n", 0, match.start()) + 1})
            if hits:
                reason = "potential_embedded_credential"
                secret_hits.append({"path": rel, "hits": hits})
        if reason:
            omitted.append({"path": rel, "size_bytes": size, "reason": reason})
        else:
            included.append({"path": rel, "size_bytes": size})
    summary = {
        "included_files": len(included),
        "included_MiB": round(sum(x["size_bytes"] for x in included) / 1024**2, 2),
        "placeholder_files": len(omitted),
        "omitted_MiB": round(sum(x["size_bytes"] for x in omitted) / 1024**2, 2),
        "reasons": dict(Counter(x["reason"] for x in omitted)),
        "credential_locations": secret_hits,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.write:
        return
    manifest = {"date": "2026-10-12", "policy": "Local originals retained; GitHub receives directory placeholders for datasets, weights, caches, archives and credential-bearing files.", "summary": summary, "omitted_files": omitted, "included_files": included}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Local originals are preserved; see github_upload_manifest.json", "/nul", "/con", "/prn", "/aux", "# Exact paths below correspond to placeholder entries."]
    for row in omitted:
        # Escape Git ignore pattern metacharacters and spaces.
        escaped = re.sub(r"([\\*?\[\] #!])", r"\\\1", row["path"])
        lines.append("/" + escaped)
    lines.extend(["", "# Future cache / credential files", "**/__pycache__/*", "!**/__pycache__/GITHUB_PLACEHOLDERS.md", "*.py[cod]", "**/.pytest_cache/*", "!**/.pytest_cache/GITHUB_PLACEHOLDERS.md", "**/.venv/", "**/node_modules/", ".env", ".env.*", "!.env.example", "!.env.template", "*.pem", "*.key", ".cache/**/*", "!.cache/**/", "!.cache/**/GITHUB_PLACEHOLDERS.md"])
    (ROOT / ".gitignore").write_text("\n".join(lines) + "\n", encoding="utf-8")
    grouped = defaultdict(list)
    for row in omitted:
        parent = Path(row["path"]).parent
        # Cache directories are represented by one root placeholder.
        for i, part in enumerate(parent.parts):
            if part in CACHE_PARTS:
                parent = Path(*parent.parts[:i + 1])
                break
        grouped[parent].append(row)
    for parent, rows in grouped.items():
        output = ROOT / parent / PLACEHOLDER
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("# 本地文件占位说明\n\n原文件保留在本地，未上传至 GitHub。此文件不包含数据或权重。\n\n详细清单见仓库根目录 `github_upload_manifest.json`。\n\n| 原路径（相对仓库根目录） | 字节数 | 原因 |\n|---|---:|---|\n" + "".join(f'| `{r["path"]}` | {r["size_bytes"]} | {r["reason"]} |\n' for r in rows), encoding="utf-8")
    (ROOT / "GITHUB_UPLOAD.md").write_text("# GitHub 上传说明\n\n日期：2026-10-12。仓库设为私有。\n\n代码、研究文档、配置、汇报材料与适量实验可视化按原目录上传。数据集、数组、模型权重、下载压缩包、缓存及过大的文件通过各目录的 `GITHUB_PLACEHOLDERS.md` 表示；包含疑似凭据的文件也只上传占位说明。原文件保留在本地。\n\n完整原路径、大小与原因记录于 `github_upload_manifest.json`，占位说明不能用于恢复文件内容。运行相关实验时，需要按项目记录另行取得数据和权重。\n\n活动研究范围以 `AGENTS.md`、`research_roadmap.md` 及最新研究记录为准。`archive/imu/` 仅为历史归档，本次上传没有执行其中的程序。\n\n可用 `python scripts/prepare_github_snapshot.py` 重新检查目录；`--write` 更新占位文件及排除清单。\n", encoding="utf-8")
    print(json.dumps({"placeholder_directories": len(grouped), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
