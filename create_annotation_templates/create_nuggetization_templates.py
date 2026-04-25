"""Create nuggetization annotation templates directly from dataset JSON files.

Generates manual and edited nuggetization templates for human annotation,
reading directly from dataset files — no pipeline run required.

For the edited template, pass --auto-nuggets pointing to a pipeline results file
(e.g. *_auto_nuggetization_auto_assignment.json) to pre-populate auto_nuggets and
auto_importance so annotators can edit them rather than starting from scratch.

Usage:
  # All QIDs, no pre-filled auto nuggets
  python create_nuggetization_templates.py --dataset Qampari_RAGW4C

  # With auto nuggets pre-filled for the edited template
  python create_nuggetization_templates.py --dataset Qampari_RAGW4C \\
    --auto-nuggets ../diagrams/all_results_files/Qampari_RAGW4C/Qampari_RAGW4C_auto_nuggetization_auto_assignment.json

  # Stratified sample (30 QIDs, 40% simple / 60% complex)
  python create_nuggetization_templates.py --dataset Qampari_RAGW4C --total 30 --simple-ratio 0.4 \\
    --auto-nuggets ../diagrams/all_results_files/Qampari_RAGW4C/Qampari_RAGW4C_auto_nuggetization_auto_assignment.json

  # Custom output directory
  python create_nuggetization_templates.py --dataset Qampari_RAGW4C --total 30 --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonuggetizer.reproducibility_data import load_data, prepare_samples
from autonuggetizer.reproducibility_datasets import DATASETS, dataset_data_relpath
from autonuggetizer.reproducibility_paths import EXPERIMENTS_ROOT


# ---------------------------------------------------------------------------
# QID sampling
# ---------------------------------------------------------------------------

def _is_simple(qid: str) -> bool:
    return "simple" in qid.lower()


def _is_complex(qid: str) -> bool:
    qid_lower = qid.lower()
    return any(p in qid_lower for p in ("composition", "comp", "intersection", "inter"))


def sample_qids(
    all_qids: list[str],
    total: int,
    simple_ratio: float,
    seed: int = 42,
) -> list[str]:
    simple_pool = [q for q in all_qids if _is_simple(q)]
    complex_pool = [q for q in all_qids if _is_complex(q)]

    n_simple = round(total * simple_ratio)
    n_complex = total - n_simple

    if n_simple > len(simple_pool):
        raise ValueError(
            f"Requested {n_simple} simple QIDs but only {len(simple_pool)} available"
        )
    if n_complex > len(complex_pool):
        raise ValueError(
            f"Requested {n_complex} complex QIDs but only {len(complex_pool)} available"
        )

    rng = random.Random(seed)
    combined = rng.sample(simple_pool, n_simple) + rng.sample(complex_pool, n_complex)
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def build_manual_nuggetization(samples: list[dict], keep_qids: set[str] | None) -> dict:
    tasks = []
    for s in samples:
        if keep_qids is not None and s["qid"] not in keep_qids:
            continue
        tasks.append({
            "qid": s["qid"],
            "question": s["question"],
            "context_segments": s["context_segments"],
            "nuggets": [],
            "importance": [],
        })
    return {
        "schema_version": "2.0",
        "task_type": "human_manual_nuggetization",
        "label_set": {"importance": ["vital", "okay"]},
        "tasks": tasks,
    }


def load_auto_nuggets(path: Path) -> dict[str, dict]:
    """Load a pipeline results file and return a per-QID lookup of nuggets/importance.

    Expects a top-level ``samples`` array where each entry has ``qid``,
    ``nuggets``, and ``importance`` (e.g. *_auto_nuggetization_auto_assignment.json).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[str, dict] = {}
    for item in data.get("samples", []):
        qid = item.get("qid")
        if qid:
            lookup[qid] = {
                "nuggets": item.get("nuggets", []),
                "importance": item.get("importance", []),
            }
    return lookup


def build_edited_nuggetization(
    samples: list[dict],
    keep_qids: set[str] | None,
    auto_lookup: dict[str, dict] | None = None,
) -> dict:
    """Edited template: annotators edit pre-filled auto nuggets.

    Pass *auto_lookup* (from :func:`load_auto_nuggets`) to pre-populate
    ``auto_nuggets`` / ``auto_importance`` per question. Without it those
    fields remain empty.
    """
    tasks = []
    for s in samples:
        if keep_qids is not None and s["qid"] not in keep_qids:
            continue
        auto = (auto_lookup or {}).get(s["qid"], {})
        tasks.append({
            "qid": s["qid"],
            "question": s["question"],
            "context_segments": s["context_segments"],
            "auto_nuggets": auto.get("nuggets", []),
            "auto_importance": auto.get("importance", []),
            "nuggets": [],
            "importance": [],
            "edit_meta": {
                "add_nugget": False,
                "remove_nugget": False,
                "major_edit": False,
                "minor_edit": False,
            }
        })
    return {
        "schema_version": "2.0",
        "task_type": "human_edited_nuggetization",
        "label_set": {"importance": ["vital", "okay"]},
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create nuggetization annotation templates from dataset files"
    )
    parser.add_argument(
        "--dataset", required=True,
        help=f"Dataset key, e.g. Qampari_RAGW4C. Available: {list(DATASETS.keys())}",
    )
    parser.add_argument("--total", type=int, help="Total QIDs to sample (default: all)")
    parser.add_argument(
        "--simple-ratio", type=float, default=0.4,
        help="Fraction of simple QIDs when sampling (default: 0.4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--auto-nuggets",
        help=(
            "Path to a pipeline results file (e.g. *_auto_nuggetization_auto_assignment.json) "
            "whose nuggets/importance will pre-populate the edited nuggetization template."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: templates/<base_dataset>/  next to this script)",
    )
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        print(f"ERROR: Unknown dataset '{args.dataset}'.")
        print(f"  Available: {list(DATASETS.keys())}")
        sys.exit(1)

    cfg = DATASETS[args.dataset]
    data_path = EXPERIMENTS_ROOT / dataset_data_relpath(cfg)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path} ...")
    raw_data = load_data(data_path)
    samples = prepare_samples(raw_data, cfg)
    print(f"  {len(samples)} samples loaded")

    keep_qids: set[str] | None = None
    if args.total is not None:
        all_qids = [s["qid"] for s in samples]
        sampled = sample_qids(all_qids, args.total, args.simple_ratio, args.seed)
        keep_qids = set(sampled)
        n_simple = sum(1 for q in sampled if _is_simple(q))
        print(f"  Sampled {n_simple} simple + {args.total - n_simple} complex = {args.total} QIDs")

    auto_lookup: dict[str, dict] | None = None
    if args.auto_nuggets:
        auto_path = Path(args.auto_nuggets).resolve()
        if not auto_path.exists():
            print(f"ERROR: Auto-nuggets file not found: {auto_path}")
            sys.exit(1)
        auto_lookup = load_auto_nuggets(auto_path)
        print(f"Loaded auto nuggets for {len(auto_lookup)} QIDs from {auto_path.name}")

    manual_payload = build_manual_nuggetization(samples, keep_qids)
    edited_payload = build_edited_nuggetization(samples, keep_qids, auto_lookup)
    n = len(manual_payload["tasks"])
    print(f"  Built templates for {n} questions")

    base_dataset = args.dataset.split("_")[0]  # e.g. "Qampari" from "Qampari_RAGW4C"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "templates" / base_dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filenames use base dataset name only — nuggets are shared across RAG variants
    manual_path = output_dir / f"{base_dataset}_manual_nuggetization_template.json"
    edited_path = output_dir / f"{base_dataset}_edited_nuggetization_template.json"

    with open(manual_path, "w", encoding="utf-8") as f:
        json.dump(manual_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved manual nuggetization template ({n} tasks): {manual_path}")

    with open(edited_path, "w", encoding="utf-8") as f:
        json.dump(edited_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved edited nuggetization template ({n} tasks): {edited_path}")

    if keep_qids is not None:
        qid_list_path = output_dir / "sampled_qids.txt"
        qid_list_path.write_text("\n".join(sorted(keep_qids)), encoding="utf-8")
        print(f"Saved QID list: {qid_list_path}")


if __name__ == "__main__":
    main()
