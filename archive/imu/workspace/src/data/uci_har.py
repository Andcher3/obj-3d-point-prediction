from __future__ import annotations

import csv
import json
import struct
from pathlib import Path
from typing import Iterable


CHANNELS = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ACTIVITY_ORDER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def dataset_root(project_root: str | Path = ".") -> Path:
    return Path(project_root) / "HAR" / "UCI HAR Dataset"


def load_activity_labels(root: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with (Path(root) / "activity_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                label_id, activity = line.split()
                labels[int(label_id)] = activity
    return labels


def read_int_vector(path: str | Path) -> list[int]:
    return [int(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_metadata(root: str | Path, split: str = "train") -> list[dict[str, int | str]]:
    root = Path(root)
    labels = read_int_vector(root / split / f"y_{split}.txt")
    subjects = read_int_vector(root / split / f"subject_{split}.txt")
    activity_labels = load_activity_labels(root)
    return [
        {
            "sample_index": i,
            "subject_id": subjects[i],
            "label_id": labels[i],
            "activity": activity_labels[labels[i]],
        }
        for i in range(len(labels))
    ]


def signal_paths(root: str | Path, split: str = "train") -> list[Path]:
    signal_dir = Path(root) / split / "Inertial Signals"
    return [signal_dir / f"{channel}_{split}.txt" for channel in CHANNELS]


def parse_signal_line(line: str) -> list[float]:
    return [float(value) for value in line.split()]


def iter_samples(root: str | Path, split: str = "train") -> Iterable[list[list[float]]]:
    files = [path.open("r", encoding="utf-8") for path in signal_paths(root, split)]
    try:
        while True:
            lines = [f.readline() for f in files]
            if not lines[0]:
                break
            if any(not line for line in lines):
                raise ValueError("Signal files have different row counts.")
            channel_rows = [parse_signal_line(line) for line in lines]
            lengths = {len(row) for row in channel_rows}
            if lengths != {128}:
                raise ValueError(f"Expected 128 readings per row, got {sorted(lengths)}")

            sample = []
            for t in range(128):
                sample.append([channel_rows[channel_idx][t] for channel_idx in range(len(CHANNELS))])
            yield sample
    finally:
        for f in files:
            f.close()


def load_sample(root: str | Path, split: str, sample_index: int) -> list[list[float]]:
    for idx, sample in enumerate(iter_samples(root, split)):
        if idx == sample_index:
            return sample
    raise IndexError(f"sample_index out of range: {sample_index}")


def write_csv(path: Path, rows: list[dict[str, int | str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_processed_train(project_root: str | Path = ".") -> dict[str, Path]:
    project_root = Path(project_root)
    root = dataset_root(project_root)
    output_dir = project_root / "data" / "processed" / "uci_har"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(root, "train")
    labels = load_activity_labels(root)
    sample_count = len(metadata)
    shape = [sample_count, 128, len(CHANNELS)]

    paths = {
        "X_train_raw": output_dir / "X_train_raw_float32.bin",
        "shape": output_dir / "shape.json",
        "metadata": output_dir / "train_metadata.csv",
        "y_train": output_dir / "y_train.csv",
        "subject_train": output_dir / "subject_train.csv",
        "channels": output_dir / "channel_names.txt",
        "activity_labels": output_dir / "activity_labels.csv",
        "readme": output_dir / "README.md",
    }

    with paths["X_train_raw"].open("wb") as f:
        for sample in iter_samples(root, "train"):
            for time_step in sample:
                f.write(struct.pack("<9f", *time_step))

    paths["shape"].write_text(
        json.dumps(
            {
                "file": paths["X_train_raw"].name,
                "dtype": "float32",
                "endianness": "little",
                "shape": shape,
                "layout": "sample, time_step, channel",
                "sampling_rate_hz": 50,
                "window_seconds": 2.56,
                "channels": CHANNELS,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(paths["metadata"], metadata, ["sample_index", "subject_id", "label_id", "activity"])
    write_csv(paths["y_train"], [{"sample_index": row["sample_index"], "label_id": row["label_id"], "activity": row["activity"]} for row in metadata], ["sample_index", "label_id", "activity"])
    write_csv(paths["subject_train"], [{"sample_index": row["sample_index"], "subject_id": row["subject_id"]} for row in metadata], ["sample_index", "subject_id"])
    paths["channels"].write_text("\n".join(CHANNELS) + "\n", encoding="utf-8")
    write_csv(paths["activity_labels"], [{"label_id": label_id, "activity": activity} for label_id, activity in labels.items()], ["label_id", "activity"])
    paths["readme"].write_text(
        "# Processed UCI HAR Train Split\n\n"
        "`X_train_raw_float32.bin` stores the training split as a contiguous float32 array.\n\n"
        "Shape:\n\n"
        "```text\n"
        "7352 x 128 x 9\n"
        "```\n\n"
        "Dimension meaning:\n\n"
        "- 7352: training window samples\n"
        "- 128: time readings inside each 2.56 second window\n"
        "- 9: IMU channels listed in `channel_names.txt`\n\n"
        "Later, if NumPy is available, it can be loaded with:\n\n"
        "```python\n"
        "import numpy as np\n"
        "X = np.fromfile('X_train_raw_float32.bin', dtype='<f4').reshape(7352, 128, 9)\n"
        "```\n",
        encoding="utf-8",
    )
    return paths


if __name__ == "__main__":
    saved = save_processed_train(Path.cwd())
    for name, path in saved.items():
        print(f"{name}: {path}")
