#!/usr/bin/env sh
set -eu
target="${1:?usage: ROLLBACK.sh TARGET_JSON}"
printf '%s\n' '{"decision":"pending","experiment":"trajloom_event_text_probe","status":"planned"}' > "$target"
