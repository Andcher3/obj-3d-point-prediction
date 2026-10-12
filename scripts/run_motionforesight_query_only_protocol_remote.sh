#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/projects
DATA=/root/autodl-tmp/ego4d-formal-e2
TC=$ROOT/TrackCraft3r
MF=$ROOT/motionforesight
ENV=/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python
CKPT=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
TOK=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl
VID=9069bd26-0139-4e88-9957-9f9489b35a91
QMAN=/root/autodl-tmp/ego4d-formal-e2/query_manifests/${VID}_query_manifest.json
WORK=$MF/outputs/query_only_protocol_2026-09-24/${VID}
mkdir -p "$WORK"
export TRACKCRAFT_TOKENIZER_DIR="$TOK"
STATUS="$WORK/protocol_status.json"
echo '{"protocol":"query-only 12/16/22 smoke","stages":[]}' > "$STATUS"

record() {
  "$ENV" - "$STATUS" "$1" "$2" "$3" <<'PY'
import json,sys
p,stage,rc,log=sys.argv[1:]; x=json.load(open(p)); x['stages'].append({'stage':stage,'returncode':int(rc),'log':log}); json.dump(x,open(p,'w'),indent=2)
PY
}
run_stage() {
  stage="$1"; input="$2"; n="$3"; qjson="$4"; output="$WORK/${stage}.npz"; log="$WORK/${stage}.log"
  echo "[START] $stage input=$input frames=$n" | tee "$log"
  set +e
  "$ENV" -u "$TC/run_trackcraft_query_inference.py" --repo "$TC" --checkpoint "$CKPT" --input-npz "$input" --query-json "$qjson" --output-npz "$output" --height 240 --width 416 --num-frames "$n" --frame-stride 1 >> "$log" 2>&1
  rc=$?
  set -e
  "$ENV" - "$output" "$stage" "$n" "$rc" "$log" "$STATUS" <<'PY'
import json,sys,os,numpy as np
out,stage,n,rc,log,status=sys.argv[1:]; row={'stage':stage,'returncode':int(rc),'log':log,'output_exists':os.path.exists(out)}
if row['output_exists']:
 try:
  z=np.load(out,allow_pickle=True); a=z['trackcraft3r_tracks']; row['shape']=list(a.shape); row['finite_fraction']=float(np.isfinite(a).mean()); row['shape_ok']=bool(a.shape==(int(n),48,3))
 except Exception as e: row['load_error']=repr(e)
 x=json.load(open(status)); x['stages'].append(row); json.dump(x,open(status,'w'),indent=2)
PY
  echo "[END] $stage rc=$rc" | tee -a "$log"
}
run_stage smoke12 "$DATA/trackcraft_inputs/$VID/input.npz" 12 "$WORK/query_frame0_9069_12.json"
run_stage smoke16 "$MF/outputs/e3_formal/356ed369-7023-4660-904b-be73823a21d3/consistent16_user.npz" 16 "$WORK/query_frame0_356_16.json"
run_stage full22 "$MF/outputs/e3_22/$VID/full22_user.npz" 22 "$WORK/query_frame0_9069_22.json"
cat "$STATUS"
