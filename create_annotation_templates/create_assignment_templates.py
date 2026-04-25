"""Create assignment annotation templates from a filled nuggetization file + dataset.

The assignment template lets annotators label whether each nugget is supported,
partially supported, or not supported by the generated answer passage.

Inputs:
  --nuggetization   Filled nuggetization JSON (schema 2.0 template or 3.0 result).
                    Must contain completed nuggets + importance per question.
  --dataset         Dataset key whose *passages* (generated answers) will be used,
                    e.g. Qampari_RAGW4C.  The nuggetization QIDs must overlap with
                    this dataset so each question gets a passage.

Usage:
  # Use a human-labeled nuggetization to evaluate a specific RAG system
  python create_assignment_templates.py \\
    --nuggetization ../results/human_labels/Qampari/nuggets/Qampari_edited_nuggetization.json \\
    --dataset Qampari_RAGW4C

  # Custom output directory
  python create_assignment_templates.py \\
    --nuggetization ../results/human_labels/Qampari/nuggets/Qampari_edited_nuggetization.json \\
    --dataset Qampari_RAGW4C \\
    --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonuggetizer.reproducibility_data import load_data, prepare_samples
from autonuggetizer.reproducibility_datasets import DATASETS, dataset_data_relpath
from autonuggetizer.reproducibility_paths import EXPERIMENTS_ROOT


# ---------------------------------------------------------------------------
# Nugget extraction (handles schema 2.0 template and schema 3.0 result)
# ---------------------------------------------------------------------------

def _extract_nuggets(task: dict) -> list[str]:
    """Return the best available nugget list from a task, regardless of schema."""
    for key in ("nuggets", "manual_edited_nuggets", "manual_created_nuggets", "auto_nuggets"):
        val = task.get(key)
        if val:
            return val
    return []


def _extract_importance(task: dict) -> list[str]:
    """Return the best available importance list from a task, regardless of schema."""
    for key in ("importance", "manual_edited_importance", "manual_created_importance", "auto_importance"):
        val = task.get(key)
        if val:
            return val
    return []


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_assignment_template(
    nug_data: dict,
    sample_lookup: dict[str, dict],
    dataset_name: str,
) -> dict:
    tasks = []
    missing_passages = []

    for task in nug_data.get("tasks", []):
        qid = task["qid"]
        source = sample_lookup.get(qid, {})

        question = task.get("question") or source.get("question", "")
        passage = source.get("passage", "")
        context_segments = task.get("context_segments") or source.get("context_segments", [])

        nuggets = _extract_nuggets(task)
        importance = _extract_importance(task)

        if not passage:
            missing_passages.append(qid)

        tasks.append({
            "qid": qid,
            "question": question,
            "passage": passage,
            "nuggets": nuggets,
            "importance": importance,
            "human_assignment": [],
        })

    if missing_passages:
        print(
            f"  WARNING: {len(missing_passages)}/{len(tasks)} QIDs have no passage "
            f"in dataset '{dataset_name}':"
        )
        for qid in missing_passages[:5]:
            print(f"    {qid}")
        if len(missing_passages) > 5:
            print(f"    ... and {len(missing_passages) - 5} more")

    return {
        "schema_version": "2.0",
        "task_type": "human_assignment_question_level",
        "label_set": ["support", "partial_support", "not_support"],
        "nugget_source": nug_data.get("nugget_source", "unknown"),
        "passage_source": dataset_name,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Output path derivation
# ---------------------------------------------------------------------------

def _nugget_type(nug_path: Path, nug_data: dict) -> str:
    """Return 'edited', 'manual', or 'auto' from the nuggetization file metadata."""
    source = nug_data.get("nugget_source", "")
    if source in ("edited", "manual", "auto"):
        return source
    stem = nug_path.stem.lower()
    for tag in ("edited", "manual", "auto"):
        if tag in stem:
            return tag
    return "edited"


def _default_output_dir(base_dataset: str, dataset_name: str) -> Path:
    """templates/<base_dataset>/<dataset_name>/"""
    return SCRIPT_DIR / "templates" / base_dataset / dataset_name


def _output_filename(dataset_name: str, nug_type: str) -> str:
    return f"{dataset_name}_{nug_type}_nuggetization_human_assignment.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create assignment templates from a filled nuggetization file + dataset passages"
    )
    parser.add_argument(
        "--nuggetization", required=True,
        help="Path to filled nuggetization JSON (schema 2.0 template or 3.0 result)",
    )
    parser.add_argument(
        "--dataset", required=True,
        help=(
            f"Dataset key whose generated passages to use, e.g. Qampari_RAGW4C. "
            f"Available: {list(DATASETS.keys())}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory "
            "(default: templates/<base_dataset>/<dataset>/  next to this script)"
        ),
    )
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        print(f"ERROR: Unknown dataset '{args.dataset}'.")
        print(f"  Available: {list(DATASETS.keys())}")
        sys.exit(1)

    nug_path = Path(args.nuggetization).resolve()
    if not nug_path.exists():
        print(f"ERROR: Nuggetization file not found: {nug_path}")
        sys.exit(1)

    cfg = DATASETS[args.dataset]
    data_path = EXPERIMENTS_ROOT / dataset_data_relpath(cfg)
    if not data_path.exists():
        print(f"ERROR: Dataset file not found: {data_path}")
        sys.exit(1)

    print(f"Loading nuggetization from {nug_path} ...")
    with open(nug_path, "r", encoding="utf-8") as f:
        nug_data = json.load(f)
    n_tasks = len(nug_data.get("tasks", []))
    schema = nug_data.get("schema_version", "?")
    nug_type = _nugget_type(nug_path, nug_data)
    print(f"  {n_tasks} tasks | schema {schema} | nugget_type: {nug_type}")

    print(f"Loading dataset passages from {data_path} ...")
    raw_data = load_data(data_path)
    samples = prepare_samples(raw_data, cfg)
    sample_lookup = {s["qid"]: s for s in samples}
    print(f"  {len(samples)} dataset samples loaded")

    nuggetization_qids = {t["qid"] for t in nug_data.get("tasks", [])}
    overlap = nuggetization_qids & sample_lookup.keys()
    if len(overlap) < len(nuggetization_qids):
        print(
            f"  NOTE: {len(overlap)}/{len(nuggetization_qids)} nuggetization QIDs "
            f"found in dataset '{args.dataset}'"
        )

    payload = build_assignment_template(nug_data, sample_lookup, args.dataset)
    n = len(payload["tasks"])

    base_dataset = args.dataset.split("_")[0]  # e.g. "Qampari"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _default_output_dir(base_dataset, args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_name = _output_filename(args.dataset, nug_type)
    out_path = output_dir / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved assignment template ({n} tasks): {out_path}")


if __name__ == "__main__":
    main()
