#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/projects/MolmoMotion-tracker3d-retest
TARGET=${1:-$ROOT/run_tracker3d_molmo_retest.py}
cp "$ROOT/baseline/run_tracker3d_molmo_probe.py" "$TARGET"
chmod +x "$TARGET"
sha256sum "$TARGET"
