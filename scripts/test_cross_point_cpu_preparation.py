"""CPU fixture tests; never load weights, train or invoke model forward."""
import ast
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cross_point_offline_runtime import safetensors_header, text_encoder_layout, local_asset


class CpuPreparationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, header):
        p = self.root / "weights/trackcraft3r/model.safetensors"
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(header).encode()
        p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4)
        return p

    def test_header_only_and_prefix(self):
        p = self.checkpoint({"pipe.text_encoder.a": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 4]},
                             "pipe.dit.a": {"shape": [0], "dtype": "BF16", "data_offsets": [4, 4]}})
        result = text_encoder_layout(p)
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["bytes"], 4)
        self.assertEqual(result["shapes"], {"a": [2]})

    def test_missing_text_encoder_rejected(self):
        with self.assertRaises(ValueError):
            text_encoder_layout(self.checkpoint({"other": {}}))

    def test_truncated_header_rejected(self):
        p = self.root / "truncated.safetensors"
        p.write_bytes(b"\1\0")
        with self.assertRaises(ValueError):
            safetensors_header(p)

    def test_invalid_length_rejected(self):
        p = self.root / "invalid.safetensors"
        p.write_bytes(struct.pack("<Q", 100))
        with self.assertRaises(ValueError):
            safetensors_header(p)

    def test_offline_unknown_pattern_rejected(self):
        with self.assertRaises(ValueError):
            local_asset(self.root, "anything-not-prepared")

    def test_offline_missing_asset_rejected(self):
        with self.assertRaises(FileNotFoundError):
            local_asset(self.root, "diffusion_pytorch_model*.safetensors")

    def test_embedded_t5_route_without_extracting(self):
        self.checkpoint({"pipe.text_encoder.a": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 4]}})
        route = local_asset(self.root, "models_t5_umt5-xxl-enc-bf16.pth")
        self.assertTrue(route.endswith("::embedded_text_encoder"))
        self.assertEqual(len(list(self.root.rglob("*.safetensors"))), 1)

    def test_query_resize_roundtrip(self):
        xy = np.array([[.2, .7], [250.1, 253.4]])
        factor = np.array([576 / 256, 320 / 256])
        np.testing.assert_allclose(xy * factor / factor, xy, atol=1e-12)

    def test_native_times_not_relabelled_15fps(self):
        ts = np.arange(22) / 12
        self.assertEqual(ts[6], .5)
        self.assertEqual(ts[7], 7 / 12)
        self.assertNotEqual(ts[-1], 21 / 15)


if __name__ == "__main__":
    unittest.main()
