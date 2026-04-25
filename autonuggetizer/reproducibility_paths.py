"""Path anchors: package lives under the project root; data and results resolve from there."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

AUTONUGGETIZER_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = AUTONUGGETIZER_DIR.parent


def resolve_reproducibility_output_dir(output_dir: str) -> str:
    p = Path(output_dir).expanduser()
    if p.is_absolute():
        return str(p)
    return str(EXPERIMENTS_ROOT / p)


def pickle_fingerprint(relative_path: str) -> dict:
    p = EXPERIMENTS_ROOT / relative_path
    if not p.is_file():
        return {"path": relative_path, "exists": False}
    st = p.stat()
    return {
        "path": relative_path,
        "exists": True,
        "bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
    }


def is_experiment_timestamp_dir(name: str) -> bool:
    return len(name) == 15 and "_" in name and name.replace("_", "").isdigit()


def latest_experiment_id(output_base: str, dataset_key: str, fallback_exp_id: str) -> str:
    root = os.path.join(output_base, dataset_key)
    if not os.path.isdir(root):
        return fallback_exp_id
    candidates = [n for n in os.listdir(root) if is_experiment_timestamp_dir(n)]
    if not candidates:
        return fallback_exp_id
    return max(candidates)


def next_non_clobber_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    directory, basename = os.path.split(path)
    stem, ext = os.path.splitext(basename)
    i = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def auto_assignment_filename(dataset_name: str) -> str:
    return f"{dataset_name}_auto_nuggetization_auto_assignment.json"
