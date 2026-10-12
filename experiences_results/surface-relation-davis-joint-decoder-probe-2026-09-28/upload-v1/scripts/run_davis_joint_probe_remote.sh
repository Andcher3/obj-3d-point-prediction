#!/usr/bin/env bash
set -euo pipefail

MF=/root/autodl-tmp/projects/motionforesight
TC=/root/autodl-tmp/projects/TrackCraft3r
DATA=/root/autodl-tmp/davis-joint-probe-inputs
OUT=$MF/outputs/davis_joint_probe_2026-09-28
PY=/root/autodl-tmp/conda_envs/motionforesight-e3/bin/python
TCPY=/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python
BASE=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
CHECKPOINT=$MF/checkpoints/trainedon_ss100k/best.pt
TOKENIZER=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl

export TRACKCRAFT_TOKENIZER_DIR="$TOKENIZER"
export MOTIONFORESIGHT_TOKENIZER_DIR="$TOKENIZER"
export TRACKCRAFT_ATTN_Q_CHUNK=512
export MODELSCOPE_CACHE=$MF/assets/models_pretrained/wan_models
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
sequences=(car-turn drift-turn boat soccerball)

stage() {
  local video=$1 name=$2
  shift 2
  local dir="$OUT/$video"
  mkdir -p "$dir"
  echo "[$(date -u +%FT%TZ)] START $video $name" | tee -a "$dir/pipeline.log"
  set +e
  "$@" >"$dir/${name}.log" 2>&1
  local rc=$?
  set -e
  echo "[$(date -u +%FT%TZ)] END $video $name rc=$rc" | tee -a "$dir/pipeline.log"
  if [[ $rc -ne 0 ]]; then
    tail -80 "$dir/${name}.log" | tee -a "$dir/pipeline.log"
    "$PY" - "$dir/failure.json" "$video" "$name" "$rc" <<'PY'
import json,sys
path,video,stage,rc=sys.argv[1:]
json.dump({'status':'FAIL','video_id':video,'failed_stage':stage,'returncode':int(rc)},open(path,'w'),indent=2)
PY
    exit "$rc"
  fi
}

for video in "${sequences[@]}"; do
  dir="$OUT/$video"
  mkdir -p "$dir"
  stage "$video" prepare_inputs env CUDA_VISIBLE_DEVICES=0 "$PY" "$MF/prepare_davis_joint_probe_inputs.py" \
    --sequence "$video" --frames-dir "$DATA/$video/JPEGImages" --masks-dir "$DATA/$video/Annotations" \
    --output-dir "$dir" --start-frame 14 --t0-frame 20 --end-frame 35 --mask-id 1 --process-res 392

  stage "$video" observed_dense bash -lc "cd '$TC' && CUDA_VISIBLE_DEVICES=0 '$PY' run_trackcraft_release_inference.py --checkpoint_path '$BASE' --input_npz '$dir/observed7_user.npz' --output_npz '$dir/observed7_trackcraft_dense.npz' --num_frames 7 --frame_stride 1"

  stage "$video" query_mapping "$PY" "$MF/build_davis_t0_frame0_mapping.py" \
    --dense-npz "$dir/observed7_trackcraft_dense.npz" --camera-npz "$dir/observed7_user.npz" \
    --query-t0-json "$dir/query_t0_manifest.json" --output-json "$dir/query_frame0_mapping.json"

  echo "[$(date -u +%FT%TZ)] START $video full22_trackcraft+motionforesight" | tee -a "$dir/pipeline.log"
  set +e
  CUDA_VISIBLE_DEVICES=1 "$TCPY" -u "$TC/run_trackcraft_query_inference.py" \
    --repo "$TC" --checkpoint "$BASE" --input-npz "$dir/full22_user.npz" \
    --query-json "$dir/query_frame0_mapping.json" --query-count 48 \
    --output-npz "$dir/full22_query_tracks.npz" --height 240 --width 416 --num-frames 22 --frame-stride 1 \
    >"$dir/full22_trackcraft.log" 2>&1 &
  tc_pid=$!
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$MF/surface_relation_e3_motionforesight_preflight.py" \
    --repo "$MF" --checkpoint "$CHECKPOINT" --base-checkpoint "$BASE" \
    --user-npz "$dir/observed7_user.npz" --query-json "$dir/query_frame0_mapping.json" --query-count 48 \
    --output "$dir/motionforesight_observed_only.json" --array-output "$dir/motionforesight_observed_only_arrays.npz" \
    --attn-q-chunk 128 >"$dir/motionforesight.log" 2>&1 &
  mf_pid=$!
  wait "$tc_pid"; tc_rc=$?
  wait "$mf_pid"; mf_rc=$?
  set -e
  echo "[$(date -u +%FT%TZ)] END $video full22_trackcraft rc=$tc_rc motionforesight rc=$mf_rc" | tee -a "$dir/pipeline.log"
  if [[ $tc_rc -ne 0 || $mf_rc -ne 0 ]]; then
    tail -80 "$dir/full22_trackcraft.log" | tee -a "$dir/pipeline.log"
    tail -80 "$dir/motionforesight.log" | tee -a "$dir/pipeline.log"
    "$PY" - "$dir/failure.json" "$video" "$tc_rc" "$mf_rc" <<'PY'
import json,sys
path,video,tc,mf=sys.argv[1:]
json.dump({'status':'FAIL','video_id':video,'failed_stage':'parallel_model_inference','trackcraft_returncode':int(tc),'motionforesight_returncode':int(mf)},open(path,'w'),indent=2)
PY
    exit 1
  fi

  stage "$video" build_bundle "$PY" "$MF/build_davis_joint_probe_bundle.py" \
    --observed-user "$dir/observed7_user.npz" --full-user "$dir/full22_user.npz" \
    --observed-dense "$dir/observed7_trackcraft_dense.npz" --full-query-track "$dir/full22_query_tracks.npz" \
    --motionforesight-arrays "$dir/motionforesight_observed_only_arrays.npz" \
    --query-t0-json "$dir/query_t0_manifest.json" --mapping-json "$dir/query_frame0_mapping.json" \
    --cache-output "$dir/p0_h_cache_aligned.npz" --audit-output "$dir/quality_audit.json"
done

"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); videos=['car-turn','drift-turn','boat','soccerball']
audits={v:json.load(open(root/v/'quality_audit.json')) for v in videos}
passed=all(a['status']=='PASS' for a in audits.values())
json.dump({'status':'PASS' if passed else 'FAIL','all_video_quality_gate_passed':passed,'videos':audits},open(root/'quality_summary.json','w'),indent=2)
if not passed: raise SystemExit(3)
PY

cd "$MF"
"$PY" surface_relation_joint_decoder_probe.py \
  --protocol "$MF/configs/surface_relation_davis_joint_decoder_probe_2026-09-28.json" \
  --output-dir "$OUT/training" >"$OUT/training.log" 2>&1
echo "[$(date -u +%FT%TZ)] COMPLETED DAVIS joint decoder probe" | tee "$OUT/COMPLETED"
