"""Create a local static query-review page from small transferred CPU outputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--directory",type=Path,required=True)
    args=ap.parse_args()
    directory=args.directory
    status=json.loads((directory/"gt-preparation-status.json").read_text())
    source=json.loads((directory/"source-audit.json").read_text())
    if not status["complete"] or not source["publisher_checksums_match"]:
        raise ValueError("Preparation or publisher checksums incomplete")
    cards=[]
    verified=[]
    for scene in status["scenes"]:
        if scene["status"]!="READY":
            continue
        name=scene["scene_id"]
        hashes=json.loads((directory/f"{name}_sha256.json").read_text())
        for remote_name, local_name in [("query_preview.png",f"{name}_query_preview.png"),
                                       ("queries.json",f"{name}_queries.json"),
                                       ("gt_tracks.npz",f"{name}_gt_tracks.npz")]:
            expected=hashes[f"inputs/{name}/{remote_name}"]
            if hashlib.sha256((directory/local_name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Local transfer checksum failed: "+local_name)
        verified.append(name)
        label=html.escape(scene["asset_id"].replace("_"," "))
        cards.append(f'<article><h2>{html.escape(name)}</h2><a href="{name}_query_preview.png"><img src="{name}_query_preview.png" alt="{label}"></a>'
                     f'<p>{label}</p><p>目标实例 {scene["target_instance_mask_id"]} · 48点 · t0=6</p>'
                     f'<p>真值构造投影误差最大 {scene["gt_t0_pixel_anchor_error_max"]:.3f}px</p></article>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-F query 预览与真值准备</title><style>
body{margin:28px;background:#10151d;color:#e8edf5;font-family:"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto}
h1{font-size:24px}h2{font-size:17px}p{line-height:1.65;font-size:14px;color:#c3cddd}a{color:#80beff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}article{padding:12px;background:#1b2431;border-radius:10px}
img{width:100%;display:block}article p{margin:8px 0}section{margin:24px 0}
</style><main><h1>MOVi-F 物体查询点预览</h1>
<p>官方验证集小子集；绿色轮廓为目标物体，彩色点为48个query。采样帧为第6帧（最后观测帧），点击图片可查看原图。</p>
<p>RGB原始256×256，官方预处理后252×252；原生12fps，7帧观测及15帧后续。当前仅完成CPU真值准备，尚未调用UniQuery4R或训练关系模块。</p>
<p>真值由仿真物体表面坐标和每帧位姿计算。其投影误差表示构造过程与采样像素的差异，不表示教师模型的误差。</p>
<section class="grid">'''+"\n".join(cards)+'''</section><p>等待用户确认query预览并开启GPU，之后执行实验1：教师轨迹对比已知真值。</p></main></html>'''
    (directory/"query-review.html").write_text(document,encoding="utf-8")
    (directory/"local-transfer-audit.json").write_text(json.dumps({"passed":True,"scenes":verified,
        "verified_files_per_scene":["query_preview.png","queries.json","gt_tracks.npz"],
        "source_rgb_and_shards_downloaded_to_local":False},indent=2)+"\n")
    print(json.dumps({"page":str(directory/"query-review.html"),"local_verified_scenes":len(verified)}))


if __name__=="__main__":
    main()
