"""Exercise the actual visual renderer with synthetic coordinates, no model imports."""
import argparse
import ast
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--renderer", type=Path, required=True)
    args = ap.parse_args()
    tree = ast.parse(args.renderer.read_text(encoding="utf-8"))
    names = {"processed_to_source", "correspondence_to_source", "project_to_source", "draw_panel", "render"}
    selected = [node for node in tree.body if getattr(node, "name", "") in names]
    assert {node.name for node in selected} == names
    namespace = {"np": np, "cv2": cv2, "colorsys": colorsys}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.renderer), "exec"), namespace)
    output = args.base / "render-smoke-synthetic-NOT-MODEL"
    output.mkdir(exist_ok=True)
    results = []
    for video in ("car-turn", "drift-turn", "boat", "soccerball"):
        directory = args.base / "inputs" / video
        manifest = json.loads((directory / "queries.json").read_text())
        paths = [directory / "rgb" / f"{frame:05d}.jpg" for frame in manifest["source_frame_indices"]]
        anchor = np.array(manifest["query_uv_source_pixels"], dtype=np.float64)
        normalized = np.array(manifest["query_uv_normalized_01"])
        h, w = manifest["processed_resolution_hw"]
        # General rotated/translated camera with non-identity intrinsics.
        angle = 0.23
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        translation = np.array([0.12, -0.05, 0.17])
        intrinsic = np.array([[430, 0, w / 2], [0, 410, h / 2], [0, 0, 1.]])
        pixel = normalized * [w - 1, h - 1]
        rays = np.concatenate((pixel, np.ones((48, 1))), axis=1) @ np.linalg.inv(intrinsic).T
        depth = np.linspace(1.2, 2.4, 48)[:, None]
        camera_xyz = rays * depth
        world_xyz = (camera_xyz - translation) @ rotation
        extrinsic = np.concatenate((rotation, translation[:, None]), axis=1)
        projected = namespace["project_to_source"](world_xyz[None], extrinsic[None],
                                                    intrinsic[None], manifest)[0]
        np.testing.assert_allclose(projected, anchor, atol=1e-10)
        camera_behind = camera_xyz.copy()
        camera_behind[:, 2] *= -1
        behind_world = (camera_behind - translation) @ rotation
        assert np.isnan(namespace["project_to_source"](behind_world[None], extrinsic[None],
                                                       intrinsic[None], manifest)).all()
        for count in (7, 22):
            offset = np.arange(count)[:, None, None] - 6
            uv2d = anchor[None] + offset * np.array([[[0.7, 0.2]]])
            # Deliberately offset the right panel at t0: renderer must NOT align it.
            uv3d = uv2d + np.array([[[6., -3.]]])
            uv2d[0, 0] = np.nan
            uv3d[1, 1] = [-1000, -1000]
            original_2d, original_3d = uv2d.copy(), uv3d.copy()
            name = f"SYNTHETIC-NOT-MODEL_{video}_{count}frames"
            namespace["render"](directory, paths[:count], uv2d, uv3d, manifest, output, name)
            np.testing.assert_array_equal(uv2d, original_2d)
            np.testing.assert_array_equal(uv3d, original_3d)
            capture = cv2.VideoCapture(str(output / (name + ".mp4")))
            frames = 0
            oh, ow = manifest["source_resolution_hw"]
            try:
                assert capture.isOpened()
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    assert frame.shape[:2] == (oh + 40, ow * 2)
                    frames += 1
            finally:
                capture.release()
            assert frames == count
            t0_image = cv2.imread(str(output / (name + "_t0_raw.png")))
            assert t0_image is not None and t0_image.shape[:2] == (oh + 40, ow * 2)
            # Independently inspect the RGB colors at each visible t0 dot center.
            for panel_index, points in enumerate((uv2d[6], uv3d[6])):
                for point_index, (u, v) in enumerate(points):
                    x, y = round(float(u)), round(float(v))
                    if 0 <= x < ow and 0 <= y < oh:
                        expected = np.array([round(c * 255) for c in
                                             colorsys.hsv_to_rgb(point_index / 48, 0.8, 1)[::-1]])
                        # Later dots can overlap; assert at least one exact dot color.
                        region = t0_image[max(40, y + 38):min(oh + 40, y + 43),
                                          panel_index * ow + max(0, x - 2):panel_index * ow + min(ow, x + 3)]
                        if point_index == 47:
                            assert np.any(np.all(region == expected, axis=2))
            results.append({"video": video, "frame_count": frames,
                            "output_resolution_hw": [oh + 40, ow * 2],
                            "t0_image_written": True, "coordinates_unmodified": True,
                            "file": name + ".mp4"})
    status = {"render_smoke_passed": True, "synthetic_only": True, "model_imported": False,
              "model_loaded": False, "model_forward_called": False,
              "real_tracking_quality_evaluated": False,
              "nonidentity_camera_projection_passed": True,
              "invalid_and_offscreen_points_handled": True, "outputs": results}
    (output / "render-smoke-summary.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
