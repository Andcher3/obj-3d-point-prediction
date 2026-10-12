#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
TARGET=${1:-}
if [[ -z "$TARGET" ]]; then
  echo "usage: $0 TARGET_COPY_DIR" >&2
  exit 64
fi
TARGET=$(cd -- "$TARGET" && pwd)
if [[ "$TARGET" == "/" || "$TARGET" == "$HERE" ]]; then
  echo "refusing unsafe rollback target: $TARGET" >&2
  exit 65
fi
for name in complete_scheme_v2.py run_complete_search_v2.py future_l1.py; do
  test -f "$TARGET/$name"
  cp -- "$HERE/.baseline/$name" "$TARGET/$name"
done
echo "ROLLBACK_RESULT=RESTORED_BASELINE"
