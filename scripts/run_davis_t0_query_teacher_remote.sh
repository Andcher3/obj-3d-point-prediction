#!/usr/bin/env bash
set -euo pipefail

MF=/root/autodl-tmp/projects/motionforesight
TC=/root/autodl-tmp/projects/TrackCraft3r
OUT=$MF/outputs/davis_t0_query_teacher_2026-09-29
DROOT=$OUT/DAVIS
PY=/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python
CKPT=/root/autodl-tmp/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
VIDEOS=(car-turn drift-turn boat soccerball)

if [[ -z "${TRACKCRAFT_TOKENIZER_DIR:-}" ]]; then
    TRACKCRAFT_TOKENIZER_DIR=$(find /root/autodl-tmp/projects/motionforesight/assets /root/autodl-tmp/trackcraft-assets \
        -type d -path '*/google/umt5-xxl' -print -quit 2>/dev/null || true)
fi
export TRACKCRAFT_TOKENIZER_DIR
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
        tail -120 "$dir/$name.log" | tee -a "$dir/pipeline.log"
        "$PY" - "$dir/failure.json" "$video" "$name" "$rc" <<'PY'
import json,sys
p,v,s,r=sys.argv[1:]
json.dump({'status':'FAIL','video_id':v,'failed_stage':s,'returncode':int(r)},open(p,'w'),indent=2)
PY
        return "$rc"
    fi
}

validate_environment() {
    [[ -x "$PY" ]]
    [[ -f "$CKPT" ]]
    [[ -n "$TRACKCRAFT_TOKENIZER_DIR" && -d "$TRACKCRAFT_TOKENIZER_DIR" ]]
    "$PY" - <<'PY'
import json,torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                  'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory}))
PY
}

run_video() {
    local video=$1
    local dir="$OUT/$video"
    mkdir -p "$dir"
    : >"$dir/pipeline.log"
    rm -f "$dir/failure.json"

    stage "$video" environment validate_environment || return $?
    stage "$video" full23_da3 "$PY" -u "$MF/scripts/prepare_davis_full23_da3_input.py" \
        --sequence "$video" --frames-dir "$DROOT/JPEGImages/480p/$video" \
        --output "$dir/full23_user.npz" --start-index 20 --process-res 392 || return $?
    stage "$video" official_interleaved_trackcraft "$PY" -u "$MF/scripts/run_trackcraft_query_interleaved23.py" \
        --repo "$TC" --checkpoint "$CKPT" --input-npz "$dir/full23_user.npz" \
        --query-json "$dir/query_t0_manifest.json" --output-npz "$dir/full22_query_tracks.npz" \
        --height 240 --width 416 || return $?
    stage "$video" quality_audit "$PY" "$MF/scripts/audit_davis_t0_query_teacher_bundle.py" \
        --video-dir "$dir" --mask-root "$DROOT/Annotations/480p" || return $?
}

if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    SELECTED=("${VIDEOS[@]}")
fi

for video in "${SELECTED[@]}"; do
    run_video "$video" | tee "$OUT/$video-driver.log"
done

"$PY" - "$OUT" "$MF/configs/surface_relation_davis_joint_decoder_t0_teacher_2026-09-29.json" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]); protocol=Path(sys.argv[2]); videos=['car-turn','drift-turn','boat','soccerball']
audits={}
for video in videos:
    p=root/video/'quality_audit_t0_teacher.json'
    if p.exists(): audits[video]=json.load(open(p))
passed=[v for v,a in audits.items() if a['status']=='PASS']
all_complete=len(audits)==4
summary={'status':'QUALITY_PASS_READY_FOR_TRAINING' if all_complete and len(passed)==4 else 'QUALITY_FAIL_NO_TRAINING',
         'protocol':'davis_t0_query_official_interleaved23_2026-09-29',
         'completed_video_ids':sorted(audits),'passed_video_ids':passed,'pass_count':len(passed),
         'training_started':False,'videos':audits}
(root/'quality_summary_t0_teacher.json').write_text(json.dumps(summary,indent=2)+'\n')
if all_complete and len(passed)==4:
    for video in videos:
        shutil.copy2(root/video/'p0_h_cache_t0_teacher.npz',root/video/'p0_h_cache_aligned.npz')
    spec={
      'protocol_status':'FROZEN_BEFORE_DAVIS_T0_QUERY_TRAINING','date':'2026-09-29',
      'scope':'four-video DAVIS leave-one-video-out t0-query teacher comparison',
      'data_root':str(root),'video_ids':videos,'target':'t0-query TrackCraft3R future aligned to observed-only t0',
      'graph':'observed t0 symmetric 2D kNN k=3','models':['joint_relational','joint_node_control'],
      'train_slots':list(range(7)),'validation_slots':[7,8],'held_out_test_slots':list(range(9)),
      'learning_rate':0.001,'training_steps':400,'checkpoint_every_steps':10,
      'seeds':[9000,9001,9002,9003,9004],'hidden_width':16,'anchor_threshold_px_for_loss_only':5.0,
      'primary_metrics':['centered_error_normalized','edge_vector_error_normalized'],'safety_metric':'raw_ade_m',
      'known_limits':['DAVIS has no true 3D trajectory ground truth','TrackCraft3R is an offline pseudo-label',
                      'source capture timestamps unavailable; consecutive source frames define the grid',
                      'MotionForesight is frozen and receives seven observed frames only','four independent videos']
    }
    protocol.parent.mkdir(parents=True,exist_ok=True);protocol.write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY

if "$PY" - "$OUT/quality_summary_t0_teacher.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['status']=='QUALITY_PASS_READY_FOR_TRAINING' else 1)
PY
then
    mkdir -p "$OUT/training"
    "$PY" -u "$MF/scripts/surface_relation_joint_decoder_probe.py" \
        --protocol "$MF/configs/surface_relation_davis_joint_decoder_t0_teacher_2026-09-29.json" \
        --output-dir "$OUT/training" | tee "$OUT/training.log"
else
    echo "QUALITY GATE BLOCKED TRAINING"
fi

"$PY" - "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); manifest={}
for p in sorted(root.rglob('*')):
    if p.is_file() and not p.is_symlink() and p.name!='sha256_manifest.json':
        manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'sha256_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
