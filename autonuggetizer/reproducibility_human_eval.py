"""Human-eval task JSON generation from pipeline or assignment outputs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from autonuggetizer.reproducibility_data import load_data, prepare_samples
from autonuggetizer.reproducibility_datasets import DATASETS, dataset_data_relpath
from autonuggetizer.reproducibility_paths import EXPERIMENTS_ROOT
from autonuggetizer.reproducibility_results import load_reproducibility_samples

logger = logging.getLogger(__name__)


def generate_human_eval_files(
    dataset_name: str,
    output_dir: str,
    results: list[dict],
    sample_lookup: dict[str, dict],
) -> dict[str, str]:
    assignment_tasks = []
    manual_nuggetization_tasks = []
    edited_nuggetization_tasks = []
    aid = 0
    mnid = 0
    enid = 0

    for row in results:
        if not isinstance(row, dict) or "error" in row:
            continue

        qid = row.get("qid")
        nuggets = row.get("nuggets", [])
        importance = row.get("importance", [])
        assignment = row.get("assignment", [])
        source = sample_lookup.get(qid, {})

        question = row.get("question") or source.get("question", "")
        passage = row.get("passage") or source.get("passage", "")
        context_segments = row.get("context_segments") or source.get("context_segments", [])

        assignment_tasks.append({
            "task_id": aid,
            "qid": qid,
            "question": question,
            "passage": passage,
            "human_assignment": [],
        })
        aid += 1

        manual_nuggetization_tasks.append({
            "task_id": mnid,
            "qid": qid,
            "question": question,
            "context_segments": context_segments,
            "manual_created_nuggets": [],
            "manual_created_importance": [],
        })
        mnid += 1

        edited_nuggetization_tasks.append({
            "task_id": enid,
            "qid": qid,
            "question": question,
            "context_segments": context_segments,
            "auto_nuggets": nuggets,
            "auto_importance": importance,
            "manual_edited_nuggets": [],
            "manual_edited_importance": [],
        })
        enid += 1

    base = os.path.join(output_dir, dataset_name)
    assignment_path = f"{base}_manual_assignment.json"
    manual_nuggetization_path = f"{base}_manual_nuggetization.json"
    edited_nuggetization_path = f"{base}_edited_nuggetization.json"

    assignment_payload = {
        "schema_version": "2.0",
        "task_type": "human_assignment_question_level",
        "label_set": ["support", "partial_support", "not_support"],
        "tasks": assignment_tasks,
    }
    manual_nuggetization_payload = {
        "schema_version": "2.0",
        "task_type": "human_manual_nuggetization",
        "label_set": {"importance": ["vital", "okay"]},
        "tasks": manual_nuggetization_tasks,
    }
    edited_nuggetization_payload = {
        "schema_version": "2.0",
        "task_type": "human_edited_nuggetization",
        "label_set": {"importance": ["vital", "okay"]},
        "tasks": edited_nuggetization_tasks,
    }

    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(assignment_payload, f, indent=2, ensure_ascii=False)
    with open(manual_nuggetization_path, "w", encoding="utf-8") as f:
        json.dump(manual_nuggetization_payload, f, indent=2, ensure_ascii=False)
    with open(edited_nuggetization_path, "w", encoding="utf-8") as f:
        json.dump(edited_nuggetization_payload, f, indent=2, ensure_ascii=False)

    return {
        "assignment": assignment_path,
        "nuggetization": manual_nuggetization_path,
        "edited_nuggetization": edited_nuggetization_path,
    }


def export_human_eval_from_auto_assignment(
    auto_json_path: str,
    dataset_name: str,
    output_dir: str | None = None,
) -> dict[str, str]:
    auto_json_path = str(Path(auto_json_path).resolve())
    if not os.path.isfile(auto_json_path):
        raise FileNotFoundError(f"Not a file: {auto_json_path}")
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    out_dir = output_dir or os.path.dirname(auto_json_path)
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    results = load_reproducibility_samples(auto_json_path)
    if not results:
        raise ValueError(f"No samples in {auto_json_path}")

    cfg = DATASETS[dataset_name]
    data_path = EXPERIMENTS_ROOT / dataset_data_relpath(cfg)
    raw_data = load_data(data_path)

    samples = prepare_samples(raw_data, cfg)
    sample_lookup = {s["qid"]: s for s in samples}

    paths = generate_human_eval_files(dataset_name, out_dir, results, sample_lookup)
    logger.info("Human eval templates from %s", auto_json_path)
    return paths
