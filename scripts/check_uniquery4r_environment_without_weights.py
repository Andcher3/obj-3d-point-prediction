"""Check installed runtime/build and official imports without model construction."""
import argparse
import importlib.metadata
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import torchvision
    import cv2
    import numpy as np
    from uniquery4r.models.uniquery4r import UniQuery4R
    from uniquery4r.models.query_bank import BaseQuery

    assert torch.__version__.split("+")[0] == "2.8.0"
    assert torchvision.__version__.split("+")[0] == "0.23.0"
    assert torch.version.cuda == "12.8"
    architectures = (torch._C._cuda_getArchFlags() or "").split()
    assert "sm_120" in architectures, architectures
    assert np.__version__ == "1.26.4"
    uv = torch.tensor([[0., 0.], [0.5, 0.5], [1., 1.]], dtype=torch.float32)
    query = BaseQuery(uv=uv, full_uv=False, width=504, height=280)
    assert query.query_nums == 3 and torch.equal(query.uv, uv)
    assert callable(UniQuery4R.load_checkpoint)
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "torchvision", "numpy", "opencv-python", "pillow", "einops",
                 "trimesh", "matplotlib", "scipy", "huggingface_hub", "viser")}
    result = {"environment_without_weights_passed": True, "torch_cuda_build": torch.version.cuda,
              "compiled_cuda_architectures": architectures, "sm_120_build_supported": True,
              "official_model_class_imported": True, "official_sparse_query_container_passed": True,
              "model_constructed": False, "checkpoint_loaded": False, "model_forward_called": False,
              "gpu_runtime_tested": False, "versions": versions}
    (args.base / "logs" / "environment-without-weights.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
