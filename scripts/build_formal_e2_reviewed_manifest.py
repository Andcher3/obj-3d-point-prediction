from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / 'experiences_results' / 'remote-pointmaps-history'
TS = ROOT / 'experiences_results' / 'remote-timestamps'
OUT = ROOT / 'configs' / 'surface_relation_formal_e2_reviewed_manifest_2026-09-23.json'

items = [
    dict(video_id='cb78dba8-6647-4ec7-875c-6ba1548f8312', source_video_uid='25d0feb6-f66f-499f-a6b0-0ceb9df56896', motion_category='articulated', target_object='sewing scissors', clip_start_s=1492.0, t0_seconds=1496.5, polygon_xy=[[402,245],[433,236],[473,258],[501,310],[493,360],[470,388],[438,365],[414,330]], source_status='reviewed_keep'),
    dict(video_id='356ed369-7023-4660-904b-be73823a21d3', source_video_uid='5944e682-eb74-4068-8c1f-b33a32a773c2', motion_category='rigid', target_object='watering can', clip_start_s=1492.0, t0_seconds=1496.5, polygon_xy=[[82,322],[125,316],[164,345],[178,408],[166,492],[133,516],[95,502],[72,438]], source_status='reviewed_keep'),
    dict(video_id='65165142-b15e-4bba-a9de-d5174ec46597', source_video_uid='374832bf-f977-4e8b-b0e0-2f2ea1e38b5d', motion_category='articulated', target_object='blue woodworking tool', clip_start_s=1492.0, t0_seconds=1498.810514, polygon_xy=[[296,54],[350,52],[374,90],[365,160],[331,177],[300,145]], source_status='reviewed_keep'),
    dict(video_id='9069bd26-0139-4e88-9957-9f9489b35a91', source_video_uid='60b0ccb6-49f7-4a44-a70d-bf319217af50', motion_category='rigid', target_object='orange cooking pot', clip_start_s=0.0210286, t0_seconds=5.791029, polygon_xy=[[78,171],[156,160],[225,181],[245,245],[215,302],[133,316],[76,275]], source_status='reviewed_keep'),
    dict(video_id='4d053322-1eef-4a69-8201-932c7b82fb03', source_video_uid='8762cf76-5526-43a2-b825-a0c396494c87', motion_category='rigid', target_object='clothes iron', clip_start_s=892.0, t0_seconds=898.360514, polygon_xy=[[295,174],[382,161],[454,194],[474,247],[436,286],[353,278],[311,239]], source_status='reviewed_keep'),
]
videos=[]
for x in items:
    vid=x['video_id']
    rel_ts= json.loads((TS/f'{vid}.json').read_text())
    abs_ts=[float(x['clip_start_s'])+float(t) for t in rel_ts]
    ts_path=ROOT/'experiences_results'/'formal_e2_absolute_timestamps'/f'{vid}.json'
    ts_path.parent.mkdir(parents=True,exist_ok=True); ts_path.write_text(json.dumps(abs_ts))
    videos.append({**x, 'clip_end_s': x['clip_start_s']+float(rel_ts[-1]), 'clip_path':f'clips/{vid}.mp4', 'polygon_xy':x['polygon_xy'], 'pointmap_history_npy':str(PM/f'{vid}_pointmap_history.npy'), 'timestamps_json':str(ts_path), 'erosion_pixels':2})
OUT.write_text(json.dumps({'protocol':'formal_e2_2026-09-22','source':'Ego4D','review_date':'2026-09-23','selection_rule':'fixed metadata order; manual t0 polygon review; one candidate excluded for insufficient thin-object area','excluded_video':{'video_id':'8de01b2c-0fc8-478e-a54d-24bee1cfde49','reason':'thin blue pole/object boundary cropped; not stable for 48-query sampling'},'videos':videos},indent=2,ensure_ascii=False)+'\n')
print(OUT)
