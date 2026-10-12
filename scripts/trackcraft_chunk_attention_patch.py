#!/usr/bin/env python3
"""Create a remote-safe patch for TrackCraft3R's SDPA fallback."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """        x = F.scaled_dot_product_attention(q, k, v)\n        x = rearrange(x, \"b n s d -> b s (n d)\", n=num_heads)"""
NEW = """        # V100 math-SDPA can materialize an O(S^2) score tensor. Chunking\n        # only the query axis is exact: keys/values remain complete, so this\n        # changes memory use without changing the attention definition.\n        chunk = int(os.environ.get(\"TRACKCRAFT_ATTN_Q_CHUNK\", \"0\"))\n        if chunk > 0 and q.shape[2] > chunk:\n            x = torch.cat([\n                F.scaled_dot_product_attention(q[:, :, start:start + chunk], k, v)\n                for start in range(0, q.shape[2], chunk)\n            ], dim=2)\n        else:\n            x = F.scaled_dot_product_attention(q, k, v)\n        x = rearrange(x, \"b n s d -> b s (n d)\", n=num_heads)"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8")
    if "import os\n" not in text:
        text = text.replace("import math\n", "import math\nimport os\n", 1)
    count = text.count(OLD)
    if count < 1:
        raise SystemExit("SDPA fallback block not found")
    before, match, after = text.rpartition(OLD)
    args.path.write_text(before + NEW + after, encoding="utf-8")
    print(args.path)


if __name__ == "__main__":
    main()
