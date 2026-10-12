#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$ROOT/run_tracker3d_molmo_probe.py}"
cp "$ROOT/baseline/run_text_type_probe.py" "$TARGET"
echo "RESTORED=$TARGET"
sha256sum "$TARGET"
