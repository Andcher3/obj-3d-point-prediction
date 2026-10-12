"""Read-only audit of supplied L1 results; no model inference or source changes."""
from pathlib import Path
import csv, hashlib, io, json, re, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiences_results/L1-perturbation"
OUT = Path(__file__).resolve().parent
FFMPEG = Path("D:/ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe")
FFPROBE = FFMPEG.with_name("ffprobe.exe")

def digest(b):
    return hashlib.sha256(b).hexdigest()

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def vectors(path):
    with np.load(path, allow_pickle=False) as z:
        return z["latents"].copy()

def distances(base, arr):
    a, b = base.astype(np.float64), arr.astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    per_cos = (a*b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1), 1e-15)
    return {
        "mse": float(np.mean((a-b)**2)),
        "mean_cosine": float(per_cos.mean()),
        "relative_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-15)),
        "per_vector_cosine": per_cos.tolist(),
    }

manifest = {p.relative_to(SRC).as_posix(): {"bytes":p.stat().st_size, "sha256":digest(p.read_bytes())}
            for p in sorted(SRC.rglob("*")) if p.is_file()}
write_json("source_manifest.json", manifest)
rows, samples, issues = [], {}, []
for sample in ("flower", "bunny", "sintel"):
    folder = SRC / "public-short" / sample
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    summ = json.loads((folder / "summary.json").read_text(encoding="utf-8-sig"))
    runs = [json.loads(p.read_text(encoding="utf-8-sig"))
            for p in sorted(folder.glob("[0-9][0-9]_*.json"))]
    baseline = next(r for r in runs if r["sigma"] == 0)
    base_prefix = f'{baseline["run_index"]:02d}_{baseline["name"]}'
    base_arr = vectors(folder / f"{base_prefix}_latents.npz")
    groups = {}
    complete_answers = []
    for r in runs:
        name = f'{r["run_index"]:02d}_{r["name"]}'
        raw = (folder / f"{name}_raw.txt").read_text(encoding="utf-8")
        arr = vectors(folder / f"{name}_latents.npz")
        dist = distances(base_arr, arr)
        complete = re.search(r"<answer>(.*?)</answer>", raw, re.S)
        parsed_answer = complete.group(1).strip() if complete else None
        reasons = re.findall(r"<reason>(.*?)</reason>", raw, re.S)
        noise_steps = r.get("noise_stats", {}).get("steps", [])
        sr = next(t for t in summ["results"] if t["name"] == r["name"])
        mismatched_fields = [k for k,v in r.items() if sr.get(k) != v]
        if mismatched_fields:
            issues.append({"sample":sample,"run":r["name"],"summary_mismatch":mismatched_fields})
        groups.setdefault(raw, []).append(r["name"])
        if parsed_answer is not None:
            complete_answers.append(parsed_answer)
        row = {
            "sample":sample,"run":r["name"],"sigma":r["sigma"],"seed":r["seed"],
            "generated_tokens":r["generated_tokens"],"latent_vectors":len(arr),"latent_dim":arr.shape[-1],
            "finite":bool(np.isfinite(arr).all()),"nonzero_vs_baseline":bool(np.any(arr != base_arr)),
            "raw_file_matches_json":raw==r["raw"],
            "raw_hash_matches":digest(raw.encode("utf-8"))==r["raw_sha256"],
            "latent_array_hash_matches":digest(arr.astype(np.float32).tobytes())==r["latent_sha256"],
            "complete_answer":parsed_answer is not None,
            "answer_tag_flag_matches":bool(complete)==r["has_answer_tag"],
            "parsed_answer_matches_json":parsed_answer==r["answer"] if complete else None,
            "raw_changed":raw!=baseline["raw"],
            "complete_answer_changed":parsed_answer!=baseline["answer"] if complete else None,
            "at_token_cap":r["generated_tokens"]>=meta["max_new_tokens"],
            "im_end_present":"<|im_end|>" in raw,
            "perturbed_steps":r.get("noise_stats", {}).get("perturbed_step_count",0),
            "mean_relative_noise_rms":float(np.mean([s["relative_noise_rms"] for s in noise_steps])) if noise_steps else 0.0,
            "latent_mse":dist["mse"],"latent_mean_cosine":dist["mean_cosine"],"latent_relative_l2":dist["relative_l2"],
            "elapsed_seconds":r["elapsed_seconds"],
            "baseline_metrics_match":all(np.isclose(dist[k],sr["vs_baseline"][k],rtol=1e-5,atol=1e-9)
                for k in ("mse","mean_cosine","relative_l2")) if "vs_baseline" in sr else None,
            "answer":parsed_answer or "",
            "first_reason_same_as_baseline":reasons[0]==re.findall(r"<reason>(.*?)</reason>",baseline["raw"],re.S)[0],
        }
        rows.append(row)
    nz = [r for r in rows if r["sample"]==sample and r["sigma"]>0]
    obs = SRC / "public-action-samples" / f"{sample}_observation.mp4"
    samples[sample] = {
        "reported_aggregate":summ["aggregate"],
        "recomputed":{
            "runs":len(runs),"baseline_count":sum(r["sigma"]==0 for r in runs),
            "nonzero_runs":len(nz),"nonzero_raw_changed":sum(r["raw_changed"] for r in nz),
            "nonzero_complete_answer_changed":sum(r["complete_answer_changed"] is True for r in nz),
            "nonzero_incomplete_answers":sum(not r["complete_answer"] for r in nz),
            "unique_complete_answers_all_runs":len(set(complete_answers)),
            "unique_raw_strings_all_runs":len(groups),
            "all_finite":all(r["finite"] for r in rows if r["sample"]==sample),
            "video_hash_matches_metadata":digest(obs.read_bytes())==meta["video_sha256"],
        },
        "unique_outputs":[{"runs":run_names,"raw":raw} for raw,run_names in groups.items()],
    }

with (OUT/"run_audit.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
write_json("run_audit.json",rows)

font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",20)
video_info={}
for p in sorted((SRC/"public-action-samples").glob("*.mp4")):
    info=json.loads(subprocess.check_output([str(FFPROBE),"-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,r_frame_rate,nb_frames,duration:format=duration","-of","json",str(p)],text=True))
    video_info[p.name]=info

def contact_sheet(path,times,name,title):
    w,h=400,225
    canvas=Image.new("RGB",(4*w,55+2*(h+30)), "#171b23")
    draw=ImageDraw.Draw(canvas); draw.text((12,15),title,font=font,fill="white")
    for i,t in enumerate(times):
        content=subprocess.check_output([str(FFMPEG),"-v","error","-ss",str(float(t)),"-i",str(path),
            "-frames:v","1","-vf",f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            "-f","image2pipe","-vcodec","png","pipe:1"])
        im=Image.open(io.BytesIO(content)).convert("RGB")
        x=(i%4)*w; y=55+(i//4)*(h+30)
        canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
        draw.text((x+8,y+h+3),f"{float(t):.3f} s",font=font,fill="white")
    canvas.save(OUT/name)
    return {"path":str((OUT/name).resolve()),"source":str(path.resolve()),"timestamps_seconds":[float(t) for t in times]}

sheets=[]
for sample,original in [("flower","flower.mp4"),("bunny","bunny_trailer.mp4"),("sintel","sintel_trailer.mp4")]:
    obs=SRC/"public-action-samples"/f"{sample}_observation.mp4"
    dur=float(video_info[obs.name]["streams"][0]["duration"])
    sheets.append(contact_sheet(obs,np.linspace(0,max(0,dur-0.08),8),
        f"{sample}_observation.jpg",f"{sample} | observation review | uniform timestamps, not verified model sampling"))
    full=SRC/"public-action-samples"/original
    end=float(video_info[original]["streams"][0]["duration"])
    sheets.append(contact_sheet(full,np.linspace(dur,min(end-0.08,dur+8),8),
        f"{sample}_recorded_future.jpg",f"{sample} | recorded continuation | review only, not model input"))

after={p.relative_to(SRC).as_posix():digest(p.read_bytes()) for p in SRC.rglob("*") if p.is_file()}
unchanged=set(after)==set(manifest) and all(after[k]==v["sha256"] for k,v in manifest.items())
result={"status":"ANALYZED_NOT_MODEL_RERUN","source_files":len(manifest),
        "source_files_unchanged":unchanged,"samples":samples,"consistency_issues":issues,
        "videos":video_info,"review_sheets":sheets,
        "limits":["One baseline per video; repeated zero-noise determinism not established.",
                  "29 recorded latent SHA256 values match float32 round-trips of saved float16 arrays; 4 do not. Original pre-save precision is unavailable.",
                  "Float16 NPZ distances are recomputed from saved arrays; original logs may use different precision.",
                  "Review contact sheets are not verified exact qwen_vl_utils selected frames.",
                  "No experiment source or model-forward logs are included in supplied bundle.",
                  "Text event judgments require manual review; no statistical population inference."]}
write_json("audit_summary.json",result)
print(json.dumps({s:v["recomputed"] for s,v in samples.items()},ensure_ascii=False,indent=2))
print("SOURCE_FILES_UNCHANGED",unchanged)
print("CSV_ROWS",len(rows),"CONSISTENCY_ISSUES",len(issues))
print("CHECK_FLAGS", {k:sum(r[k] is True for r in rows) for k in
      ["raw_file_matches_json","raw_hash_matches","latent_array_hash_matches","answer_tag_flag_matches",
       "parsed_answer_matches_json","first_reason_same_as_baseline"]})
print("SHEETS",len(sheets))
