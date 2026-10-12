from __future__ import annotations

import difflib
from pathlib import Path

here = Path(__file__).resolve().parent
pairs = [
    (here / ".baseline" / "complete_scheme_v2.py", here / "complete_scheme_v2.py", "complete_scheme_v2.py"),
    (here / ".baseline" / "run_complete_search_v2.py", here / "run_complete_search_v2.py", "run_complete_search_v2.py"),
    (here / ".baseline" / "future_l1.py", here / "remote_src" / "future_l1.py", "src/model/future_l1.py"),
]
chunks: list[str] = []
for before, after, label in pairs:
    chunks.extend(
        difflib.unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        )
    )
(here / "DIFF_FILE.patch").write_text("".join(chunks), encoding="utf-8")
