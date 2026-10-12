"""Unit-test our pair/pixel helpers using a stub base, never loading the model."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    script = Path(__file__).with_name("run_uniquery4r_davis_visual_review.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    names = {"SourceFrameUniQuery4R", "processed_to_source", "correspondence_to_source", "project_to_source"}
    selected = [node for node in tree.body if getattr(node, "name", "") in names]
    namespace = {"np": np, "UniQuery4R": object}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), "exec"), namespace)
    base = script.parents[1] / "experiences_results" / "uniquery4r-davis-visual-review-2026-10-09"
    results = []
    for count in (7, 22):
        pairs = namespace["SourceFrameUniQuery4R"]._build_pair_idx(SimpleNamespace(query_source=6), count)
        assert len(pairs) == len(set(pairs)) == 2 * count - 1
        assert all((6, t) in pairs and (t, t) in pairs for t in range(count))
    for video in ("car-turn", "drift-turn", "boat", "soccerball"):
        manifest = json.loads((base / "inputs" / video / "queries.json").read_text())
        normalized = np.array(manifest["query_uv_normalized_01"])
        restored = namespace["correspondence_to_source"](normalized * 2 - 1, manifest)
        np.testing.assert_allclose(restored, manifest["query_uv_source_pixels"], atol=1e-10)
        h, w = manifest["processed_resolution_hw"]
        xyz = np.concatenate((normalized * [w-1, h-1], np.ones((48, 1))), axis=-1)[None]
        projected = namespace["project_to_source"](xyz, np.eye(4)[None, :3], np.eye(3)[None], manifest)[0]
        np.testing.assert_allclose(projected, restored, atol=1e-10)
        behind = xyz.copy()
        behind[..., 2] = -1
        assert np.isnan(namespace["project_to_source"](behind, np.eye(4)[None, :3], np.eye(3)[None], manifest)).all()
        results.append({"video": video, "pixel_roundtrip_max_error": float(np.max(np.abs(restored - manifest["query_uv_source_pixels"])))})
    result = {"helper_unit_tests_passed": True, "base_class": "explicit object stub, not model",
              "pair_selection_7_and_22": True, "model_loaded": False, "model_forward_called": False,
              "videos": results}
    (base / "helper-unit-tests.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
