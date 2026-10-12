#!/usr/bin/env python
"""Verify and summarize the round-3 Future-L1 branch experiment.

The script only reads experiment artifacts.  It verifies recorded hashes,
latent finiteness, deterministic controls, branch-state separation, and the
stored event-aware selection result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BRANCH_RE = re.compile(
    r"^r(?P<round>\d+)_(?P<spec>\d+)_b(?P<branch>\d+)_sigma_(?P<sigma>[^_]+)_seed_(?P<seed>-?\d+)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-summary", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.code_dir))
    from complete_scheme_v2 import (  # type: ignore
        BranchCandidate,
        EventComparison,
        select_unique_branches,
    )

    reports = sorted(args.results_root.glob("cut_*/complete_search_report.json"))
    if len(reports) != 3:
        raise SystemExit(f"expected 3 reports, found {len(reports)} under {args.results_root}")

    rows: list[dict] = []
    videos: list[dict] = []
    source_hashes: dict[str, str] = {}
    for report_path in reports:
        video_id = report_path.parent.name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = report["metadata"]
        candidates = report["candidates"]
        by_id = {c["branch_id"]: c for c in candidates}
        retained_ids = {c["branch_id"] for c in report["retained"]}
        round_groups: dict[int, list[dict]] = defaultdict(list)
        spec_groups: dict[tuple, list[dict]] = defaultdict(list)
        raw_hash_ok = latent_hash_ok = latent_finite = True
        all_latents_nonempty = True
        for candidate in candidates:
            match = BRANCH_RE.match(candidate["branch_id"])
            if not match:
                raise AssertionError(candidate["branch_id"])
            round_index = int(match.group("round"))
            spec_index = int(match.group("spec"))
            branch_index = int(match.group("branch"))
            sigma = float(match.group("sigma"))
            seed = int(match.group("seed"))
            round_groups[round_index].append(candidate)
            spec_groups[(round_index, spec_index, sigma, seed)].append(candidate)
            raw_path = report_path.parent / f"{candidate['branch_id']}_raw.txt"
            latent_path = report_path.parent / f"{candidate['branch_id']}_latents.npz"
            raw_hash_ok &= raw_path.is_file() and sha256(raw_path) == candidate["metadata"]["raw_sha256"]
            latent_hash_ok &= latent_path.is_file() and sha256(latent_path) == candidate["metadata"]["latent_sha256_saved"]
            if latent_path.is_file():
                latent = np.load(latent_path)["latents"]
                latent_finite &= bool(np.isfinite(latent).all())
                all_latents_nonempty &= bool(latent.size)
            else:
                latent_finite = all_latents_nonempty = False
            rows.append(
                {
                    "video": video_id,
                    "branch_id": candidate["branch_id"],
                    "round": round_index,
                    "spec": spec_index,
                    "branch": branch_index,
                    "sigma": sigma,
                    "seed": seed,
                    "answer": candidate["answer"],
                    "valid": candidate["valid_format"],
                    "target_coverage": candidate["target_coverage"],
                    "subject_targeted": candidate["subject_targeted"],
                    "motion_specific": candidate["motion_specific"],
                    "quality": candidate["quality"],
                    "retained": candidate["branch_id"] in retained_ids,
                    "latent_sha256": candidate["metadata"]["latent_sha256_float32"],
                }
            )

        comparisons = [EventComparison(**item) for item in report["comparisons"]]
        objects = [
            BranchCandidate(
                c["branch_id"], c["answer"], c["sigma"], c["seed"], c["round_index"],
                raw=c["raw"], valid_format=c["valid_format"], target_coverage=c["target_coverage"],
                subject_targeted=c["subject_targeted"], motion_specific=c["motion_specific"],
                quality=c["quality"], metadata=c["metadata"],
            )
            for c in candidates
        ]
        reproduced = select_unique_branches(
            objects,
            comparisons=comparisons,
            branch_count=metadata["branch_count_max"],
            min_quality=metadata["min_quality"],
            min_target_coverage=metadata["min_target_coverage"],
        )
        reproduced_ids = [c.branch_id for c in reproduced]

        nonzero_specs = [group for key, group in spec_groups.items() if key[2] > 0]
        noisy_state_split = [len({c["metadata"]["latent_sha256_float32"] for c in group}) for group in nonzero_specs]
        noisy_answer_split = [len({c["answer"] for c in group}) for group in nonzero_specs]
        zero_specs = [group for key, group in spec_groups.items() if key[2] == 0]
        zero_branch_equal = all(
            len({c["metadata"]["latent_sha256_float32"] for c in group}) == 1
            and len({c["answer"] for c in group}) == 1
            for group in zero_specs
        )

        round_stats = []
        for round_index, group in sorted(round_groups.items()):
            accepted = [
                c for c in group
                if c["valid_format"] and c["quality"] >= metadata["min_quality"]
                and c["target_coverage"] >= metadata["min_target_coverage"]
                and c["subject_targeted"] and c["motion_specific"]
            ]
            round_stats.append(
                {
                    "round": round_index,
                    "candidates": len(group),
                    "exact_unique_answers": len({c["answer"] for c in group}),
                    "accepted_structure": len(accepted),
                    "accepted_exact_unique_answers": len({c["answer"] for c in accepted}),
                    "retained_after_round": len(report["rounds"][round_index]["retained"]),
                }
            )

        videos.append(
            {
                "video": video_id,
                "video_sha256": metadata["video_sha256"],
                "target": metadata["target_object"],
                "candidate_count": len(candidates),
                "valid_count": sum(bool(c["valid_format"]) for c in candidates),
                "subject_count": sum(bool(c["subject_targeted"]) for c in candidates),
                "accepted_structure_count": report["accepted_structure_count"],
                "exact_unique_answers": len({c["answer"] for c in candidates}),
                "retained_count": report["retained_count"],
                "retained_answers": [c["answer"] for c in report["retained"]],
                "baseline_raw_equal": report["baseline_raw_equal"],
                "baseline_latent_equal": report["baseline_latent_equal"],
                "zero_branch_equal": zero_branch_equal,
                "nonzero_spec_count": len(nonzero_specs),
                "nonzero_specs_all_latent_distinct": all(n == metadata["branch_count_max"] for n in noisy_state_split),
                "nonzero_specs_with_multiple_answers": sum(n > 1 for n in noisy_answer_split),
                "nonzero_answer_split_histogram": dict(sorted(Counter(noisy_answer_split).items())),
                "raw_hash_ok": raw_hash_ok,
                "latent_hash_ok": latent_hash_ok,
                "all_latents_nonempty": all_latents_nonempty,
                "all_latents_finite": latent_finite,
                "selection_reproduced": reproduced_ids == [c["branch_id"] for c in report["retained"]],
                "rounds": round_stats,
            }
        )
        source_hashes.update(metadata["code_sha256"])

    old = None
    if args.old_summary and args.old_summary.is_file():
        old_data = json.loads(args.old_summary.read_text(encoding="utf-8"))
        old = {
            item["video"]: {
                "candidate_count": item["candidate_count"],
                "unique_retained_answers": item["unique_retained_answers"],
            }
            for item in old_data["samples"]
        }

    summary = {
        "verification_status": "VERIFIED" if all(
            v["raw_hash_ok"] and v["latent_hash_ok"] and v["all_latents_nonempty"]
            and v["all_latents_finite"] and v["selection_reproduced"]
            and v["baseline_raw_equal"] and v["baseline_latent_equal"] and v["zero_branch_equal"]
            for v in videos
        ) else "FAILED",
        "result_root": str(args.results_root.resolve()),
        "source_hashes": source_hashes,
        "candidate_total": sum(v["candidate_count"] for v in videos),
        "retained_total": sum(v["retained_count"] for v in videos),
        "videos": videos,
        "old_comparison": old,
        "interpretation_boundary": "Event candidates only; no 3D trajectories and no calibrated probabilities.",
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "candidate_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["verification_status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
