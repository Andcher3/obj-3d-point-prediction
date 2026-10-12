#!/usr/bin/env bash
set -euo pipefail

MF=/root/autodl-tmp/projects/motionforesight
TC=/root/autodl-tmp/projects/TrackCraft3r
BASE=$MF/outputs/davis_ab_teacher_2026-09-29
COMMON=$BASE/common
DROOT=$BASE/DAVIS
ROUTE_A=$BASE/route_a
ROUTE_B=$BASE/route_b
PY=${TRACKCRAFT_PYTHON:-/root/autodl-tmp/conda_envs/trackcraft-e2/bin/python}
CKPT=/root/autodl-tmp/trackcraft-assets/checkpoints/trackcraft3r/model.safetensors
VIDEOS=(car-turn drift-turn boat soccerball)

if [[ -z "${TRACKCRAFT_TOKENIZER_DIR:-}" ]]; then
    TRACKCRAFT_TOKENIZER_DIR=$(find /root/autodl-tmp/trackcraft-assets /root/autodl-tmp/projects/motionforesight/assets \
        -type d -path '*/google/umt5-xxl' -print -quit 2>/dev/null || true)
fi
export TRACKCRAFT_TOKENIZER_DIR
export TRACKCRAFT_ATTN_Q_CHUNK=512
export MODELSCOPE_CACHE=$MF/assets/models_pretrained/wan_models
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$COMMON" "$ROUTE_A" "$ROUTE_B"

stage() {
    local video=$1 name=$2
    shift 2
    local dir="$COMMON/$video"
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
    [[ -x "$PY" && -f "$CKPT" ]]
    [[ -n "$TRACKCRAFT_TOKENIZER_DIR" && -d "$TRACKCRAFT_TOKENIZER_DIR" ]]
    "$PY" - <<'PY'
import json,torch
assert torch.cuda.is_available(), 'CUDA unavailable'
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                  'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory}))
PY
}

run_video() {
    local video=$1 dir="$COMMON/$1"
    mkdir -p "$dir" "$ROUTE_A/$video" "$ROUTE_B/$video"
    : >"$dir/pipeline.log"
    rm -f "$dir/failure.json"
    stage "$video" environment validate_environment || return $?
    stage "$video" route_a_full23_da3 "$PY" -u "$MF/scripts/prepare_davis_full23_da3_input.py" \
        --sequence "$video" --frames-dir "$DROOT/JPEGImages/480p/$video" \
        --output "$dir/route_a_full23_user.npz" --start-index 20 --process-res 392 || return $?
    stage "$video" route_a_trackcraft "$PY" -u "$MF/scripts/run_trackcraft_query_interleaved23.py" \
        --repo "$TC" --checkpoint "$CKPT" --input-npz "$dir/route_a_full23_user.npz" \
        --query-json "$dir/query_t0_manifest.json" --output-npz "$dir/route_a_full22_query_tracks.npz" \
        --height 240 --width 416 || return $?
    stage "$video" ab_quality_audit "$PY" "$MF/scripts/audit_davis_ab_teacher_bundle.py" \
        --video-dir "$dir" --mask-root "$DROOT/Annotations/480p" \
        --route-a-output "$ROUTE_A/$video" --route-b-output "$ROUTE_B/$video" || return $?
}

if [[ $# -gt 0 ]]; then SELECTED=("$@"); else SELECTED=("${VIDEOS[@]}"); fi
for video in "${SELECTED[@]}"; do
    run_video "$video" | tee "$BASE/$video-driver.log"
done

"$PY" - "$BASE" <<'PY'
import json,sys
from pathlib import Path
base=Path(sys.argv[1]);videos=['car-turn','drift-turn','boat','soccerball']
summary={'protocol':'DAVIS A/B t0-first versus matched full-context 2026-09-29','training_started':False,'routes':{}}
for key in ['route_a','route_b']:
    audits={}
    for video in videos:
        p=base/key/video/'quality_audit.json'
        if p.exists():audits[video]=json.load(open(p))
    passed=[v for v,a in audits.items() if a['status']=='PASS']
    summary['routes'][key]={'completed_video_ids':sorted(audits),'passed_video_ids':passed,
                            'pass_count':len(passed),'training_eligible':len(passed)>=3,'videos':audits}
(base/'quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY

train_route() {
    local route=$1
    local protocol="$BASE/${route}_protocol.json"
    local output="$BASE/${route}_training"
    "$PY" - "$BASE/quality_summary.json" "$BASE/$route" "$protocol" "$route" <<'PY'
import json,sys
from pathlib import Path
summary=Path(sys.argv[1]);root=Path(sys.argv[2]);out=Path(sys.argv[3]);route=sys.argv[4]
x=json.load(open(summary));ids=x['routes'][route]['passed_video_ids']
if len(ids)<3:raise SystemExit(3)
spec={'protocol_status':'FROZEN_AB_SMALL_SCALE','date':'2026-09-29','scope':route+' DAVIS passed-video LOO',
      'data_root':str(root),'video_ids':ids,'target':route+' audited TrackCraft3R alternative supervision',
      'graph':'observed t0 symmetric 2D kNN k=3','models':['joint_relational','joint_node_control'],
      'train_slots':list(range(7)),'validation_slots':[7,8],'held_out_test_slots':list(range(9)),
      'learning_rate':0.001,'training_steps':400,'checkpoint_every_steps':10,
      'seeds':[9000,9001,9002,9003,9004],'hidden_width':16,'anchor_threshold_px_for_loss_only':5.0,
      'primary_metrics':['centered_error_normalized','edge_vector_error_normalized'],'safety_metric':'raw_ade_m',
      'known_limits':['alternative TrackCraft3R supervision, not true 3D ground truth',
                      'only quality-passing videos are used; failed videos are not substituted',
                      'source capture timestamps unavailable; consecutive frames define the grid',
                      'MotionForesight frozen; seven observed frames only']}
out.write_text(json.dumps(spec,indent=2)+'\n')
PY
    local rc=$?
    if [[ $rc -eq 3 ]]; then echo "$route has fewer than three passing videos; no training"; return 0; fi
    mkdir -p "$output"
    "$PY" -u "$MF/scripts/surface_relation_joint_decoder_probe.py" --protocol "$protocol" --output-dir "$output" \
        | tee "$BASE/${route}_training.log"
}

set +e
train_route route_a; rc_a=$?
train_route route_b; rc_b=$?
set -e
"$PY" - "$BASE/quality_summary.json" "$rc_a" "$rc_b" <<'PY'
import json,sys
p=sys.argv[1];x=json.load(open(p));x['training_returncodes']={'route_a':int(sys.argv[2]),'route_b':int(sys.argv[3])}
x['training_started']=any((__import__('pathlib').Path(p).parent/f'{r}_training/results.json').exists() for r in ['route_a','route_b'])
open(p,'w').write(json.dumps(x,indent=2)+'\n')
PY
