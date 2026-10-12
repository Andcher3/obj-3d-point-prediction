#!/usr/bin/env bash
set -euo pipefail

MF=/root/autodl-tmp/projects/motionforesight
TC=/root/autodl-tmp/projects/TrackCraft3r
DATA=/root/autodl-tmp/davis-joint-probe-inputs
OLD=$MF/outputs/davis_frame0_direct_2026-09-28
OUT=$MF/outputs/davis_official_interleaved_pj0_2026-09-28
PY=/root/autodl-tmp/conda_envs/motionforesight-e3/bin/python
TCPY=/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python
BASE=/root/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
TOKENIZER=/root/trackcraft-assets/Wan2.1-T2V-1.3B/google/umt5-xxl

export TRACKCRAFT_TOKENIZER_DIR="$TOKENIZER"
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
        tail -100 "$dir/$name.log" | tee -a "$dir/pipeline.log"
        "$PY" - "$dir/failure.json" "$video" "$name" "$rc" <<'PY'
import json,sys
p,v,s,r=sys.argv[1:]
json.dump({'status':'FAIL','video_id':v,'failed_stage':s,'returncode':int(r)},open(p,'w'),indent=2)
PY
        return "$rc"
    fi
}

run_video() {
    local video=$1 gpu=$2
    local dir="$OUT/$video"
    mkdir -p "$dir"
    : >"$dir/pipeline.log"
    cp -f "$OLD/$video/query_frame0_manifest.json" "$dir/query_frame0_manifest.json"
    ln -sfn "$OLD/$video/observed7_user.npz" "$dir/observed7_user.npz"
    ln -sfn "$OLD/$video/motionforesight_observed_only_arrays.npz" "$dir/motionforesight_observed_only_arrays.npz"
    cp -f "$OLD/$video/quality_audit.json" "$dir/legacy_direct22_quality_audit.json"

    stage "$video" observed7_pj0_anchor "$PY" "$MF/anchor_davis_trackcraft_frame0_pj.py" \
        --source-dir "$OLD/$video" --output-dir "$dir" --contexts observed7 || return $?

    stage "$video" full23_da3 env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$MF/prepare_davis_full23_da3_input.py" \
        --sequence "$video" --frames-dir "$DATA/$video/JPEGImages" \
        --output "$dir/full23_user.npz" --process-res 392 || return $?

    stage "$video" official_interleaved_trackcraft env CUDA_VISIBLE_DEVICES="$gpu" "$TCPY" -u "$MF/run_trackcraft_query_interleaved23.py" \
        --repo "$TC" --checkpoint "$BASE" --input-npz "$dir/full23_user.npz" \
        --query-json "$dir/query_frame0_manifest.json" --output-npz "$dir/full22_query_tracks.npz" \
        --height 240 --width 416 || return $?

    stage "$video" quality_audit "$PY" "$MF/audit_davis_official_interleaved_pj0_bundle.py" --video-dir "$dir" || return $?
    stage "$video" projection "$PY" "$MF/build_davis_official_interleaved_projection.py" \
        --video-dir "$dir" --output "$dir/visualization_projection.npz" || return $?
    stage "$video" comparison "$PY" "$MF/audit_davis_pj0_anchor_comparison.py" \
        --source-dir "$OLD/$video" --anchored-dir "$dir" --mask-dir "$DATA/$video/Annotations" || return $?
}

run_pair() {
    local video0=$1 video1=$2
    run_video "$video0" 0 >"$OUT/$video0-driver.log" 2>&1 & local p0=$!
    run_video "$video1" 1 >"$OUT/$video1-driver.log" 2>&1 & local p1=$!
    set +e
    wait "$p0"; local r0=$?
    wait "$p1"; local r1=$?
    set -e
    cat "$OUT/$video0-driver.log"
    cat "$OUT/$video1-driver.log"
    [[ $r0 -eq 0 && $r1 -eq 0 ]]
}

if [[ $# -eq 0 ]]; then
    run_pair car-turn drift-turn
    run_pair boat soccerball
elif [[ $# -eq 2 ]]; then
    run_pair "$1" "$2"
else
    echo "usage: $0 [video_for_gpu0 video_for_gpu1]" >&2
    exit 2
fi

"$PY" - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); videos=['car-turn','drift-turn','boat','soccerball']
audits={v:json.load(open(root/v/'quality_audit.json')) for v in videos}
comparisons={v:json.load(open(root/v/'pj0_anchor_comparison.json')) for v in videos}
passed=[v for v in videos if audits[v]['status']=='PASS']
summary={'status':'QUALITY_PASS_NO_TRAINING_STARTED' if len(passed)==4 else 'QUALITY_FAIL_NO_TRAINING',
         'protocol':'official_interleaved23_two_12frame_runs_pj0_anchor_2026-09-28',
         'passed_video_ids':passed,'pass_count':len(passed),'training_started':False,
         'observed7_protocol':'nonofficial 7-frame diagnostic; no future-frame padding',
         'legacy_direct22_protocol_valid':False,'videos':audits}
(root/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={}
for p in sorted(root.rglob('*')):
    if p.is_file() and not p.is_symlink() and p.name!='sha256_manifest.json':
        manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'passed_video_ids':passed},indent=2))
PY
