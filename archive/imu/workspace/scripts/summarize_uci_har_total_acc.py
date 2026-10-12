from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "uci_har"
ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def main() -> int:
    X = np.fromfile(DATA_ROOT / "X_train_raw_float32.bin", dtype="<f4").reshape(7352, 128, 9)
    with (DATA_ROOT / "train_metadata.csv").open(encoding="utf-8", newline="") as f:
        metadata = list(csv.DictReader(f))

    print(
        "activity,n,mean_x,mean_y,mean_z,std_x,std_y,std_z,"
        "avg_window_std,mean_norm,std_norm"
    )
    for activity in ACTIVITIES:
        indices = np.array(
            [idx for idx, row in enumerate(metadata) if row["activity"] == activity],
            dtype=int,
        )
        data = X[indices, :, 0:3]
        flat = data.reshape(-1, 3)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        avg_window_std = data.std(axis=1).mean()
        norm = np.linalg.norm(data, axis=2)
        print(
            f"{activity},{len(indices)},"
            f"{means[0]:.3f},{means[1]:.3f},{means[2]:.3f},"
            f"{stds[0]:.3f},{stds[1]:.3f},{stds[2]:.3f},"
            f"{avg_window_std:.3f},{norm.mean():.3f},{norm.std():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
