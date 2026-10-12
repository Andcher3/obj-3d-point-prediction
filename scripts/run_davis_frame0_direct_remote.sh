#!/usr/bin/env bash
set -euo pipefail

MF=/root/autodl-tmp/projects/motionforesight
SRC=/root/autodl-tmp/projects/motionforesight_src
TC=/root/autodl-tmp/projects/TrackCraft3r
DATA=/root/autodl-tmp/davis-joint-probe-inputs
OLD=$MF/outputs/davis_joint_probe_2026-09-28
OUT=$MF/outputs/davis_frame0_direct_2026-09-28
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
stage() {
    local video=$1 name=$2
    shift 2
    local dir="$OUT/$video"
    echo "[$(date -u +%FT%TZ)] START $video $name" | tee -a "$dir/pipeline.log"
    set +e
    "$@" >"$dir/$name.log" 2>&1
    local rc=$?
    set -e
    echo "[$(date -u +%FT%TZ)] END $video $name rc=$rc" | tee -a "$dir/pipeline.log"
    if [[ $rc -ne 0 ]]; then
        tail -80 "$dir/$name.log" | tee -a "$dir/pipeline.log"
        "$PY" - "$dir/failure.json" "$video" "$name" "$rc" <<'PYCODE'
import json,sys
p,v,s,r=sys.argv[1:]
json.dump({'status':'FAIL','video_id':v,'failed_stage':s,'returncode':int(r)},open(p,'w'),indent=2)
PYCODE
        exit "$rc"
    fi
}

for video in car-turn drift-turn boat soccerball; do
    dir="$OUT/$video"
    mkdir -p "$dir"
    for input in observed7_user.npz full22_user.npz; do
        source="$OLD/$video/$input"
        if [[ ! -s "$source" ]]; then echo "Missing reusable DA3 input: $source" >&2; exit 2; fi
        if [[ -e "$dir/$input" && ! -L "$dir/$input" ]]; then
            echo "Refusing to replace existing input: $dir/$input" >&2; exit 2
        fi
        ln -sfn "$source" "$dir/$input"
    done
    stage "$video" frame0_queries "$PY" "$MF/prepare_davis_frame0_queries.py" \
        --video-id "$video" --mask "$DATA/$video/Annotations/00014.png" \
        --observed-user "$dir/observed7_user.npz" --full-user "$dir/full22_user.npz" \
        --output "$dir/query_frame0_manifest.json"

    stage "$video" observed7_query env CUDA_VISIBLE_DEVICES=0 "$TCPY" -u "$TC/run_trackcraft_query_inference.py" \
        --repo "$TC" --checkpoint "$BASE" --input-npz "$dir/observed7_user.npz" \
        --query-json "$dir/query_frame0_manifest.json" --query-count 48 \
        --output-npz "$dir/observed7_query_tracks.npz" --height 240 --width 416 --num-frames 7

    echo "[$(date -u +%FT%TZ)] START $video teacher22+motionforesight" | tee -a "$dir/pipeline.log"
    set +e
    CUDA_VISIBLE_DEVICES=1 "$TCPY" -u "$TC/run_trackcraft_query_inference.py" \
        --repo "$TC" --checkpoint "$BASE" --input-npz "$dir/full22_user.npz" \
        --query-json "$dir/query_frame0_manifest.json" --query-count 48 \
        --output-npz "$dir/full22_query_tracks.npz" --height 240 --width 416 --num-frames 22 \
        >"$dir/full22_trackcraft.log" 2>&1 &
    teacher_pid=$!
    CUDA_VISIBLE_DEVICES=0 "$PY" -u "$MF/surface_relation_e3_motionforesight_preflight.py" \
        --repo "$SRC" --checkpoint "$CHECKPOINT" --base-checkpoint "$BASE" \
        --user-npz "$dir/observed7_user.npz" --query-json "$dir/query_frame0_manifest.json" --query-count 48 \
        --output "$dir/motionforesight_observed_only.json" \
        --array-output "$dir/motionforesight_observed_only_arrays.npz" --attn-q-chunk 128 \
        >"$dir/motionforesight.log" 2>&1 &
    model_pid=$!
    wait "$teacher_pid"; teacher_rc=$?
    wait "$model_pid"; model_rc=$?
    set -e
    echo "[$(date -u +%FT%TZ)] END $video teacher22 rc=$teacher_rc motionforesight rc=$model_rc" | tee -a "$dir/pipeline.log"
    if [[ $teacher_rc -ne 0 || $model_rc -ne 0 ]]; then
        tail -80 "$dir/full22_trackcraft.log" | tee -a "$dir/pipeline.log"
        tail -80 "$dir/motionforesight.log" | tee -a "$dir/pipeline.log"
        "$PY" - "$dir/failure.json" "$video" "$teacher_rc" "$model_rc" <<'PYCODE'
import json,sys
p,v,t,m=sys.argv[1:]
json.dump({'status':'FAIL','video_id':v,'failed_stage':'teacher22_and_motionforesight','teacher_returncode':int(t),'motionforesight_returncode':int(m)},open(p,'w'),indent=2)
PYCODE
        exit 3
    fi
    stage "$video" quality_audit "$PY" "$MF/audit_davis_frame0_direct_bundle.py" --video-dir "$dir"
done

"$PY" - "$OUT" <<'PYCODE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); videos=['car-turn','drift-turn','boat','soccerball']
audits={v:json.load(open(root/v/'quality_audit.json')) for v in videos}
passed=[v for v in videos if audits[v]['status']=='PASS']
result={'status':'QUALITY_PASS' if len(passed)==len(videos) else 'QUALITY_FAIL_NO_TRAINING',
        'protocol':'frame0_direct_query_2026-09-28','passed_video_ids':passed,
        'pass_count':len(passed),'training_gate_all_four':len(passed)==len(videos),
        'videos':audits,'preview_mp4_fps_used':False}
(root/'quality_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'passed_video_ids':passed},indent=2))
PYCODE
