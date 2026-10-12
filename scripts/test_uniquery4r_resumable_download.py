"""Offline retry/control tests; no external connection or real download."""
import importlib.util
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    path = Path(__file__).with_name("fetch_uniquery4r_assets.py")
    spec = importlib.util.spec_from_file_location("fetch_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.GIT_REVISION) == len(module.HF_REVISION) == 40
    assert len(module.OFFICIAL_WEIGHT_SHA256) == 64
    with tempfile.TemporaryDirectory(prefix="uq-download-test-") as directory:
        destination = Path(directory) / "checkpoint.pth"
        with patch.dict(os.environ, {"https_proxy": "http://test.invalid"}):
            with patch.object(module.subprocess, "run", side_effect=[SimpleNamespace(returncode=c) for c in (29, 2, 0)]) as run:
                with patch.object(module.time, "sleep") as sleep:
                    module.download("https://test.invalid/file", destination, direct=True)
                    assert run.call_count == 3
                    assert [c.args[0] for c in sleep.call_args_list] == [30, 60]
                    for call in run.call_args_list:
                        assert "https_proxy" not in call.kwargs["env"]
                        assert "--continue=true" in call.args[0]
                        assert "--max-connection-per-server=4" in call.args[0]
                        assert "--auto-save-interval=10" in call.args[0]
                        assert "--allow-overwrite=false" in call.args[0]
            assert os.environ["https_proxy"] == "http://test.invalid"
        with patch.object(module.subprocess, "run", return_value=SimpleNamespace(returncode=32)) as run:
            try:
                module.download("https://test.invalid/file", destination)
                raise AssertionError("Checksum/configuration failures must stop")
            except RuntimeError:
                assert run.call_count == 1
    print("PASS: transient retry/backoff, lower concurrency, proxy isolation, nontransport failures stop")


if __name__ == "__main__":
    main()
