#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: ROLLBACK.sh ROOT" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -W)"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$SCRIPT_DIR/ROLLBACK.ps1" -Root "$1"
