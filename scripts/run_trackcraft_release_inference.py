#!/usr/bin/env python3
"""Run TrackCraft user-video inference through the self-contained release loader."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo))

    import trackcraft_release_adapter as adapter
    import evaluation.wan_scene_flow_predictor as predictor_module

    adapter.install_release_loader()
    predictor_module.WanSceneFlowPredictor = adapter.ReleaseCheckpointPredictor
    runpy.run_path(str(repo / "scripts" / "inference_user_video.py"), run_name="__main__")


if __name__ == "__main__":
    main()
