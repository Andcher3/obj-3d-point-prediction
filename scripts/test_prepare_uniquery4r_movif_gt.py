"""CPU unit checks; synthetic fixtures do not certify real dataset quality."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_uniquery4r_movif_gt import camera_matrices, fields, material_tracks, parse_example, prepare_scene, project, quaternion_matrices, target_interior, write_input_hashes
from run_uniquery4r_movif_experiment1 import input_audit
from evaluate_uniquery4r_movif_gt import compare, stats, scene_metrics
from prepare_uniquery4r_movif_protocol_sync import full_target_tracks
from prepare_uniquery4r_movif_density_query6 import prepare_scene as prepare_density_scene


def vint(x):
    result = bytearray()
    while x >= 128:
        result.append((x & 127) | 128)
        x >>= 7
    result.append(x)
    return bytes(result)


def part(number, data):
    return vint(number*8+2) + vint(len(data)) + data


class PreparationChecks(unittest.TestCase):
    def test_tf_example_wire_parser(self):
        floats = np.array([35, .5], dtype="<f4")
        entries = b""
        for key, feature in [
            ("camera/focal_length", part(2, part(1, floats.tobytes()))),
            ("metadata/num_frames", part(3, part(1, vint(24)))),
            ("metadata/video_name", part(1, part(1, b"synthetic"))),
            ("unused/huge_feature", part(1, part(1, b"ignored")))
        ]:
            entries += part(1, part(1, key.encode()) + part(2, feature))
        parsed = parse_example(part(1, entries))
        np.testing.assert_array_equal(parsed["camera/focal_length"], floats)
        np.testing.assert_array_equal(parsed["metadata/num_frames"], [24])
        self.assertEqual(parsed["metadata/video_name"], [b"synthetic"])
        self.assertNotIn("unused/huge_feature", parsed)

    def test_camera_conversion(self):
        data = {"camera/positions": np.zeros((2, 3)), "camera/quaternions": np.tile([1.,0,0,0], (2,1)),
                "camera/focal_length": np.array([35.]), "camera/sensor_width": np.array([32.])}
        k, e = camera_matrices(data, 2, 256, 256)
        point = np.array([[[0., 0., -5.], [1., 1., -5.]]]*2)
        uv, cam = project(point, k, e)
        np.testing.assert_allclose(uv[:,0], [[128,128]]*2)
        np.testing.assert_allclose(uv[:,1], [[184,72]]*2)
        np.testing.assert_allclose(cam[:,0,2], 5)

    def test_material_identity_and_motion(self):
        corners = np.array(list(itertools.product([-.5,.5], repeat=3)))
        theta = np.linspace(0, .7, 22)
        rotation = quaternion_matrices(np.c_[np.cos(theta/2), np.zeros((22,2)), np.sin(theta/2)])
        translation = np.c_[np.arange(22)*.1, np.zeros((22,2))]
        boxes = np.einsum("tij,bj->tbi", rotation, corners) + translation[:,None]
        local = np.array([[10000,20000,30000], [60000,30000,15000]], np.uint16)
        tracks = material_tracks(local, boxes)
        expected = np.einsum("tij,qj->tqi", rotation, local/65535-.5) + translation[:,None]
        np.testing.assert_allclose(tracks, expected, atol=1e-14)
        distances = np.linalg.norm(tracks[:,0]-tracks[:,1], axis=-1)
        np.testing.assert_allclose(distances, distances[0], atol=1e-14)

    def test_truncated_proto_rejected(self):
        with self.assertRaises(ValueError):
            list(fields(b"\x0a\x05\x00"))

    def test_erosion_image_boundary(self):
        interior=target_interior(np.ones((256,256),np.uint8),2)
        self.assertFalse(interior[:2].any())
        self.assertFalse(interior[-2:].any())
        self.assertFalse(interior[:,:2].any())
        self.assertFalse(interior[:,-2:].any())
        self.assertEqual(int(interior.sum()),252*252)

    def test_metrics_oracle_and_fixed_scale(self):
        rng = np.random.default_rng(123)
        truth = rng.normal(size=(22,48,3))
        groups, _ = compare(truth/3.0, truth, 6, 5.0, 3.0)
        for group in groups.values():
            for metric in group.values():
                self.assertLess(metric["mean"], 1e-14)
        distorted = truth.copy()
        distorted[7:] *= 2
        groups, _ = compare(distorted, truth, 6, 5.0, 1.0)
        self.assertEqual(groups["t0"]["relative_pair_length_error_fraction"]["mean"], 0)
        self.assertAlmostEqual(groups["future"]["relative_pair_length_error_fraction"]["mean"], 1)
        self.assertEqual(stats([])["count"], 0)

    def test_synthetic_scene_preparation(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"configs"/"uniquery4r_movif_experiment1_2026-10-09.json").read_text())
        y,x = np.mgrid[:256,:256]
        f=280.
        xw,yw=(x+.5-128)*5/f,-(y+.5-128)*5/f
        distance=np.sqrt(xw*xw+yw*yw+25)
        encoded_depth=np.rint((distance-4)/4*65535).astype(np.uint16)
        coords=np.stack([(xw+1.5-.18)/3,(yw+1.5)/3,np.full_like(xw,.5)],axis=-1)
        coords=np.rint(np.clip(coords,0,1)*65535).astype(np.uint16)
        mask=np.zeros((256,256),np.uint8)
        mask[70:186,70:186]=1
        rgb=np.zeros((256,256,3),np.uint8)
        def blobs(image, rgb_order=False):
            if rgb_order:
                image=image[...,::-1]
            ok,encoded=cv2.imencode(".png",image)
            assert ok
            return [encoded.tobytes()]*24
        corners=np.array(list(itertools.product([-.5,.5],repeat=3)))*[3,3,.4]+[0,0,-5]
        positions=np.c_[np.arange(24)*.03,np.zeros(24),np.full(24,-5.)]
        boxes=corners[None]+np.c_[np.arange(24)*.03,np.zeros((24,2))][:,None]
        annot={"video":blobs(rgb,True),"segmentations":blobs(mask),
               "object_coordinates":blobs(coords,True),"depth":blobs(encoded_depth),
               "metadata/video_name":[b"synthetic_fixture"],"metadata/num_frames":np.array([24]),
               "metadata/num_instances":np.array([1]),"metadata/height":np.array([256]),"metadata/width":np.array([256]),
               "metadata/depth_range":np.array([4.,8.]),"instances/bboxes_3d":boxes.reshape(-1),
               "instances/positions":positions.reshape(-1),"instances/quaternions":np.tile([1.,0,0,0],24),
               "instances/asset_id":[b"SyntheticPhysicalObject"],"camera/positions":np.zeros(72),
               "camera/quaternions":np.tile([1.,0,0,0],24),"camera/focal_length":np.array([35.]),"camera/sensor_width":np.array([32.])}
        source0_annot = dict(annot)
        source0_coords = np.stack([(xw+1.5)/3, (yw+1.5)/3, np.full_like(xw,.5)], axis=-1)
        source0_annot["object_coordinates"] = blobs(np.rint(np.clip(source0_coords,0,1)*65535).astype(np.uint16), True)
        camera, visible, _, _, full_mask, xy, audit = full_target_tracks(source0_annot, 1)
        self.assertEqual(camera.shape, (24, 116*116, 3))
        self.assertEqual(len(xy), int(full_mask.sum()))
        self.assertEqual(int(visible[0].sum()), len(xy))
        self.assertFalse(audit["erosion"])
        self.assertLess(audit["source_anchor_max_pixels"], .01)
        np.testing.assert_allclose(camera[-1, :, 0] - camera[0, :, 0], .69, atol=1e-10)
        with tempfile.TemporaryDirectory(prefix="movif_gt_unit_") as temp:
            result=prepare_scene(Path(temp),cfg,annot,"synthetic",0)
            self.assertEqual(result["status"],"READY")
            self.assertEqual(result["query_count"],48)
            self.assertLess(result["gt_t0_pixel_anchor_error_max"],.01)
            self.assertTrue((Path(temp)/"inputs"/"scene_synthetic_fixture"/"gt_tracks.npz").exists())
            folder=Path(temp)/"inputs"/"scene_synthetic_fixture"
            a=write_input_hashes(Path(temp),folder)
            b=write_input_hashes(Path(temp),folder)
            self.assertEqual(a,b)
            self.assertFalse(any(k.endswith("sha256.json") for k in b))
            (Path(temp)/"logs").mkdir()
            (Path(temp)/"logs"/"gt-preparation-status.json").write_text(json.dumps({
                "complete":True,"gt_coordinate_checks_passed":True,"scenes":[result]}))
            self.assertEqual(len(input_audit(Path(temp))),1)
            density_cfg = {"whole_image_nominal_query_count":1000, "cpu_audit":cfg["cpu_audit"]}
            dense_output = Path(temp)/"density"
            new = prepare_density_scene(Path(temp),dense_output,density_cfg,annot,result)
            self.assertEqual(new["status"],"READY")
            self.assertGreater(new["query_count"],48)
            self.assertFalse(new["fallback_byte_exact"])
            self.assertEqual(new["gt_tracks_shape"],[22,new["query_count"],3])
            self.assertTrue(new["all_query_visible_at_t0"])
            (dense_output/"logs").mkdir()
            (dense_output/"logs/gt-preparation-status.json").write_text(json.dumps({
                "complete":True,"gt_coordinate_checks_passed":True,"scenes":[new]}))
            self.assertEqual(len(input_audit(dense_output)),1)
            density_folder = dense_output/"inputs/scene_synthetic_fixture"
            density_manifest = json.loads((density_folder/"queries.json").read_text())
            with np.load(density_folder/"gt_tracks.npz",allow_pickle=False) as gt:
                raw = {"tracks_3d":gt["tracks_first_camera"]/3,
                    "source_frame_indices":gt["source_frame_indices"],
                    "query_uv_normalized_01":np.array(density_manifest["query_uv_normalized_01"]),
                    "tracks_2d_source_pixels":gt["tracks_2d"], "tracks_3d_projected_source_pixels":gt["tracks_2d"]}
                score = scene_metrics(raw,gt,density_manifest)
            self.assertEqual(score["query_count"],new["query_count"])
            self.assertEqual(score["pair_count"],new["query_count"]*(new["query_count"]-1)//2)
            self.assertLess(score["scale_adjusted"]["future"]["relative_vector_error_bbox_fraction"]["mean"],1e-14)
            fallback_output = Path(temp)/"fallback"
            fallback_cfg = {**density_cfg,"whole_image_nominal_query_count":1}
            reuse = prepare_density_scene(Path(temp),fallback_output,fallback_cfg,annot,result)
            self.assertTrue(reuse["fallback_byte_exact"])
            self.assertEqual(reuse["query_count"],48)
            for file in ("queries.json","gt_tracks.npz"):
                self.assertEqual((folder/file).read_bytes(),(fallback_output/"inputs/scene_synthetic_fixture"/file).read_bytes())


if __name__ == "__main__":
    unittest.main()
