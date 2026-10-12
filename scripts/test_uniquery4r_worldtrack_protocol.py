"""CPU-only protocol checks, including differential tests against pinned source."""
import ast
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from prepare_uniquery4r_worldtrack import prepare_one
from run_uniquery4r_worldtrack import audit_inputs, select_prediction_slots
from uniquery4r_worldtrack_protocol import (aggregate, dynamic_mask, first_camera_tracks,
                                           load_sequence, metric, query_normalization,
                                           processed_shape, score_tracks, select_source_queries)


def reference_metrics():
    path = Path(__file__).parents[1] / "reports/reference_code/worldtrack-st4rtrack-0f9a3f4/track_eval_util.py.txt"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_compute_scale_factor_global", "get_pointwise_threshold_multiplier",
             "compute_average_pts_within_thresh"}
    nodes = [n for n in tree.body if getattr(n, "name", "") in names]
    class NoTensor:
        pass
    namespace = {"np": np, "torch": SimpleNamespace(Tensor=NoTensor),
                 "PIXEL_TO_FIXED_METRIC_THRESH": {1: .1, 2: .3, 4: .5, 8: 1.}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["compute_average_pts_within_thresh"]


def fixture(path, frames=64, enriched=False):
    rgb = np.zeros((28, 42, 3), np.uint8)
    ok, jpeg = cv2.imencode(".jpg", rgb)
    assert ok
    point0 = np.array([[0., 0., 4.], [.1, .2, 4.], [.2, -.1, 4.], [.3, 0., 4.]])
    tracks = np.tile(point0, (frames, 1, 1))
    tracks[:, 0, 0] += np.arange(frames) * .005
    visible = np.ones((frames, 4), bool)
    visible[0, 3] = False
    visible[2:, 1] = False
    kwargs = dict(images_jpeg_bytes=np.array([jpeg.tobytes()] * frames),
                  tracks_XYZ=tracks, visibility=visible, fx_fy_cx_cy=[20., 20., 21., 14.],
                  extrinsics_w2c=np.tile(np.eye(4), (frames, 1, 1)))
    if enriched:
        kwargs["extrinsics_w2c_gt"] = kwargs["extrinsics_w2c"].copy()
        kwargs["extrinsics_w2c"][:, 0, 3] = 99  # Must never be used as GT.
    np.savez_compressed(path, **kwargs)


class ProtocolChecks(unittest.TestCase):
    def test_exact_scale(self):
        gt = np.array([[[1., 2., 4.], [2., 3., 5.]]] * 64)
        result = metric(gt, gt / 7)
        self.assertAlmostEqual(result["scale"], 7)
        self.assertLess(result["epe"], 1e-14)
        self.assertEqual(result["apd_percent"], 100)

    def test_pinned_reference_randomized_120_cases(self):
        ref = reference_metrics()
        rng = np.random.default_rng(10)
        for case in range(120):
            gt = rng.normal(size=(64, 3 + case % 16, 3)) + [0, 0, 5]
            prediction = (gt + rng.normal(size=gt.shape) * .3) / (case % 7 + 2)
            official = ref(gt, prediction, scaling="global", intrinsics_params=np.array([200., 210., 128., 128.]))
            result = metric(gt, prediction)
            self.assertAlmostEqual(result["scale"], official[3][0], places=12)
            self.assertAlmostEqual(result["epe"], official[4], places=12)
            self.assertAlmostEqual(result["apd_percent"], official[0] * 100, places=12)
            np.testing.assert_allclose(result["threshold_hit_percent"], np.array(list(official[2].values())) * 100)

    def test_threshold_inclusive_reference(self):
        # Exact-boundary comparison through official pred_aligned branch.
        ref = reference_metrics()
        gt = np.array([[[0., 0., 5.]]])
        pred = gt + [.1, 0, 0]
        expected = ref(gt, pred, intrinsics_params=np.array([200., 200., 128., 128.]), pred_aligned=pred)
        self.assertEqual(expected[2][1], 1.)

    def test_dynamic_uses_path_length(self):
        gt = np.zeros((64, 3, 3))
        gt[:, :, 2] = 5
        gt[1, 0, 0] = .006  # Out-and-back path .012; final displacement zero.
        gt[1:, 1, 0] = .01  # Exact threshold excluded.
        np.testing.assert_array_equal(dynamic_mask(gt), [True, False, False])

    def test_dynamic_scale_recomputed(self):
        gt = np.tile(np.array([[0., 0., 5.], [0., 0., 10.], [0., 0., 15.]]), (64, 1, 1))
        gt[:, 0, 0] = np.linspace(0, 1, 64)
        pred = gt.copy()
        pred[:, 0] /= 2
        pred[:, 1:] /= 7
        score = score_tracks(gt, pred)
        self.assertEqual(score["dynamic_query_count"], 1)
        self.assertAlmostEqual(score["dynamic_points"]["scale"], 2)
        self.assertNotAlmostEqual(score["all_points"]["scale"], 2)

    def test_empty_dynamic_explicit(self):
        gt = np.array([[[0., 0., 5.]]] * 64)
        self.assertIsNone(score_tracks(gt, gt)["dynamic_points"])

    def test_nonfinite_predictions_rejected(self):
        gt = np.array([[[0., 0., 5.]]] * 64)
        pred = gt.copy(); pred[3, 0] = np.nan
        with self.assertRaises(ValueError):
            metric(gt, pred)

    def test_source_visible_in_bounds_all_points(self):
        cam = np.array([[[0., 0., 5.], [10., 0., 5.], [0., 0., 5.], [1., 0., 5.]]] * 64)
        visibility = np.ones((64, 4), bool)
        visibility[0, 2] = False
        visibility[1:, 3] = False
        ids, _, selection = select_source_queries(cam, visibility, [100., 100., 50., 50.], 100, 100)
        np.testing.assert_array_equal(ids, [0, 3])
        self.assertEqual(selection["selected_points"], 2)

    def test_occluded_future_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"; fixture(path)
            sample = load_sequence(path)
            self.assertEqual(sample["gt"].shape, (64, 3, 3))
            self.assertFalse(sample["visibility"][3, 1])
            self.assertEqual(metric(sample["gt"], sample["gt"])["point_time_count"], 192)

    def test_real_64_frames_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"; fixture(path, 24)
            with self.assertRaisesRegex(ValueError, "64 real frames"):
                load_sequence(path)

    def test_first_64_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"; fixture(path, 80)
            self.assertEqual(load_sequence(path)["gt"].shape[0], 64)

    def test_explicit_native24_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / "native24.npz"; fixture(source, 24)
            folder = base / "inputs/movif_object_diagnostic/native24"
            row = prepare_one(source, folder, "movif_object_diagnostic", "original-worldtrack", False,
                              "open_d4rt_endpoint", frame_count=24)
            (base / "preparation.json").write_text(json.dumps({"rows": [row], "all_attempted_inputs_prepared": True}))
            self.assertEqual(audit_inputs(base, 24)[0][1]["frames"], 24)
            with self.assertRaises(ValueError):
                audit_inputs(base)  # Never silently relabel as64.
            pairs = [(0, t) for t in range(24)]
            self.assertEqual(select_prediction_slots(pairs, 24), list(range(24)))

    def test_enriched_camera_never_predicted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"; fixture(path, enriched=True)
            with self.assertRaisesRegex(ValueError, "Enriched"):
                load_sequence(path)
            sample = load_sequence(path, camera_kind="trackcraft-enriched")
            self.assertEqual(sample["camera_key"], "extrinsics_w2c_gt")
            self.assertLess(abs(sample["gt"][0, 0, 0]), 1)

    def test_camera_reference_conversion(self):
        world = np.tile([[1., 2., 4.]], (64, 1, 1))
        e = np.tile(np.eye(4), (64, 1, 1))
        e[:, 0, 3] = 5 + np.arange(64) * .01
        cam = world + e[:, None, :3, 3]
        result = first_camera_tracks(cam, e)
        np.testing.assert_allclose(result, world + [5, 0, 0], atol=1e-12)

    def test_incorrect_camera_rotation_rejected(self):
        tracks = np.array([[[0., 0., 5.]]] * 64)
        e = np.tile(np.eye(4), (64, 1, 1)); e[1, 0, 0] = 2
        with self.assertRaises(ValueError):
            first_camera_tracks(tracks, e)

    def test_normalization_modes_explicit(self):
        uv = np.array([[100., 90.]])
        end = query_normalization(uv, [256, 256], [252, 252], "open_d4rt_endpoint")
        center = query_normalization(uv, [256, 256], [252, 252], "pixel_center_resize")
        np.testing.assert_allclose(end, uv / 255)
        np.testing.assert_allclose((center * 251 + .5) * 256 / 252 - .5, uv)
        self.assertGreater(np.max(abs(end - center)), 0)

    def test_official_resize_shape_rounding(self):
        self.assertEqual(processed_shape(256, 256), [252, 252])
        self.assertEqual(processed_shape(360, 640), [280, 504])
        self.assertEqual(processed_shape(640, 360), [504, 280])
        self.assertEqual(processed_shape(480, 640), [378, 504])

    def test_pair_source0_and_order(self):
        pairs = [(0, t) for t in range(1, 64)] + [(t, t) for t in range(64)]
        slots = select_prediction_slots(pairs)
        self.assertEqual([pairs[i] for i in slots], [(0, t) for t in range(64)])

    def test_macro_weights_datasets_equally(self):
        rows = []
        for index, subset in enumerate(["pstudio_mini", "po_mini", "ds_mini", "adt_mini"]):
            for n in range(index + 1):
                rows.append({"subset": subset, "metrics": {"dynamic_points": {"epe": index, "apd_percent": 10 * index}}})
        result = aggregate(rows)
        self.assertEqual(result["four_dataset_macro"]["epe"], 1.5)
        self.assertFalse(result["full_200_sequence_coverage"])
        with self.assertRaises(ValueError):
            aggregate(rows, True)

    def test_missing_subset_not_four_dataset_macro(self):
        result = aggregate([{"subset": "po_mini", "metrics": {"dynamic_points": {"epe": .2, "apd_percent": 70}}}])
        self.assertIsNone(result["four_dataset_macro"])

    def test_movif_diagnostic_aggregation_separate(self):
        rows = [{"subset": "movif_object_diagnostic", "metrics": {"dynamic_points": {"epe": e, "apd_percent": p}}}
                for e, p in [(1., 20.), (3., 40.)]]
        result = aggregate(rows)
        self.assertEqual(result["other_dataset_diagnostics"]["movif_object_diagnostic"]["dynamic_epe"], 2.)
        self.assertEqual(result["other_dataset_diagnostics"]["movif_object_diagnostic"]["dynamic_apd_percent"], 30.)
        self.assertIsNone(result["four_dataset_macro"])
        self.assertFalse(result["full_200_sequence_coverage"])

    def test_prepare_audit_end_to_end_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); source = base / "fixture.npz"; fixture(source)
            folder = base / "inputs/po_mini/fixture"
            row = prepare_one(source, folder, "po_mini", "original-worldtrack", False, "open_d4rt_endpoint")
            (base / "preparation.json").write_text(json.dumps({"rows": [row], "all_attempted_inputs_prepared": True}))
            checked = audit_inputs(base)
            self.assertEqual(len(checked), 1)
            self.assertEqual(checked[0][1]["frames"], 64)
            self.assertFalse(checked[0][1]["model_forward_called"])
            self.assertNotIn("torch", sys.modules)
            command = [sys.executable, str(Path(__file__).with_name("run_uniquery4r_worldtrack.py")), "--base", str(base)]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(json.loads(proc.stdout)["model_forward_called"])
            denied = subprocess.run(command + ["--run-inference"], capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("model not imported", denied.stderr)
            # Mutation detection must fail before any model import.
            (folder / "rgb/001.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_inputs(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolChecks))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"tests_run": result.testsRun,
            "errors": len(result.errors), "failures": len(result.failures),
            "passed": result.wasSuccessful(), "differential_reference_cases": 120,
            "model_loaded": False, "model_forward_called": False,
            "actual_benchmark_inference": False}, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
