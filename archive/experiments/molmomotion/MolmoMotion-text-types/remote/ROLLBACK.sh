#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/projects/molmo-motion-text-test
SOURCE="$ROOT/baseline/01_quickstart.py"
PROTECTED="$ROOT/run_text_type_probe.py"
TARGET="${1:-}"
EXPECTED=32f7159be902e896f07659bb428f7d346febc754bf5bfbb6334ffaca0dc19bf0
if [[ -z "$TARGET" ]]; then
  echo 'ROLLBACK_RESULT=missing-target'
  exit 2
fi
if [[ "$(readlink -f "$TARGET")" == "$(readlink -f "$PROTECTED")" ]]; then
  echo 'ROLLBACK_RESULT=refused-protected-modified-file'
  exit 3
fi
mkdir -p "$(dirname "$TARGET")"
cp -f "$SOURCE" "$TARGET"
ACTUAL=$(sha256sum "$TARGET" | awk '{print $1}')
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo 'ROLLBACK_RESULT=hash-mismatch'
  echo "RESTORED_SHA256=$ACTUAL"
  exit 4
fi
echo 'ROLLBACK_RESULT=restored'
echo "RESTORED_SHA256=$ACTUAL"
echo 'ROLLBACK_EXIT=0'
