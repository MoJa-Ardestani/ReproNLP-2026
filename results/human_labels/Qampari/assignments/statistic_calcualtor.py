#!/usr/bin/env python3
"""
Recompute per-task and top-level statistics in JSON files.

Supported layouts:
1) Files with top-level "samples" and aggregates:
   - score_aggregate
2) Files with top-level "tasks" and assignment statistics:
   - per-task  "statistics" block (injected when missing)
   - top-level "assignment_statistics"

Usage examples:
  # Process a named dataset directory beside this script
  python statistic_calcualtor.py --dataset Qampari_RAGW4C

  # Process explicit file(s) or folder(s)
  python statistic_calcualtor.py path/to/file.json another/folder/
"""

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_NAME_TOKENS = ("edited_nuggetization", "manual_nuggetization")
ASSIGNMENT_LABELS = ("support", "partial_support", "not_support")

# Directory that contains this script — used to resolve --dataset
SCRIPT_DIR = Path(__file__).resolve().parent


def has_allowed_name(path: Path) -> bool:
    return any(token in path.name for token in ALLOWED_NAME_TOKENS)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_okay(importance: list[Any]) -> int:
    # Some files use typo "okey"; count it as okay.
    count = 0
    for item in importance:
        if item in ("okay", "okey"):
            count += 1
    return count


def compute_task_statistics(task: dict[str, Any]) -> dict[str, Any]:
    """Return a per-task statistics dict computed from its nuggets and assignments."""
    nuggets = task.get("nuggets") or task.get("auto_nugget", [])
    importance = task.get("importance") or task.get("auto_importance", [])
    auto_assignment = task.get("auto_assignment", [])
    human_assignment = task.get("human_assignment", [])

    n_nuggets = len(nuggets)
    n_vital = importance.count("vital")
    n_okay = _count_okay(importance)

    auto_counts: dict[str, int] = {lbl: 0 for lbl in ASSIGNMENT_LABELS}
    for lbl in auto_assignment:
        if lbl in ASSIGNMENT_LABELS:
            auto_counts[lbl] += 1

    human_counts: dict[str, int] = {lbl: 0 for lbl in ASSIGNMENT_LABELS}
    has_human = bool(human_assignment)
    for lbl in human_assignment:
        if lbl in ASSIGNMENT_LABELS:
            human_counts[lbl] += 1

    agreement: dict[str, Any] = {}
    if has_human and auto_assignment and len(auto_assignment) == len(human_assignment):
        matches = sum(a == h for a, h in zip(auto_assignment, human_assignment))
        compared = len(auto_assignment)
        agreement = {
            "n_nuggets_compared": compared,
            "n_exact_match": matches,
            "n_mismatch": compared - matches,
            "agreement_rate": round(matches / compared, 6),
        }

    stats: dict[str, Any] = {
        "n_nuggets": n_nuggets,
        "n_vital": n_vital,
        "n_okay": n_okay,
        "auto_assignment_counts": auto_counts,
    }
    if has_human:
        stats["human_assignment_counts"] = human_counts
        stats["primary_assignment_source"] = "human"
        stats["assignment_counts"] = human_counts.copy()
    else:
        stats["primary_assignment_source"] = "auto"
        stats["assignment_counts"] = auto_counts.copy()

    if agreement:
        stats["agreement"] = agreement

    return stats


def inject_task_statistics(tasks: list[dict[str, Any]]) -> int:
    """Add a ``statistics`` block to every task that is missing one.

    Returns the number of tasks that were updated.
    """
    updated = 0
    for task in tasks:
        if "statistics" not in task:
            task["statistics"] = compute_task_statistics(task)
            updated += 1
    return updated


def build_score_aggregate_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    vstrict_values: list[float] = []
    astrict_values: list[float] = []

    total_counts = {
        "nuggets_total": 0,
        "nuggets_vital": 0,
        "nuggets_okay": 0,
        "support_count": 0,
        "partial_support_count": 0,
        "not_support_count": 0,
    }

    for sample in samples:
        sample_scores = sample.get("scores") if isinstance(sample.get("scores"), dict) else {}
        importance = sample.get("importance", [])
        assignment = sample.get("assignment", [])

        vstrict_value = sample_scores.get("vstrict", sample.get("vstrict", 0.0))
        astrict_value = sample_scores.get("astrict", sample.get("astrict", 0.0))
        vstrict_values.append(_as_float(vstrict_value))
        astrict_values.append(_as_float(astrict_value))

        total_counts["nuggets_total"] += int(
            sample_scores.get("nuggets_total", len(sample.get("nuggets", [])))
        )
        total_counts["nuggets_vital"] += int(
            sample_scores.get("nuggets_vital", importance.count("vital"))
        )
        total_counts["nuggets_okay"] += int(
            sample_scores.get("nuggets_okay", _count_okay(importance))
        )

        total_counts["support_count"] += int(
            sample_scores.get("support_count", assignment.count("support"))
        )
        total_counts["partial_support_count"] += int(
            sample_scores.get("partial_support_count", assignment.count("partial_support"))
        )
        total_counts["not_support_count"] += int(
            sample_scores.get("not_support_count", assignment.count("not_support"))
        )

    return {
        "n_valid_samples": len(samples),
        "vstrict": {
            "mean": round(_safe_mean(vstrict_values), 6),
            "min": round(min(vstrict_values), 6) if vstrict_values else 0.0,
            "max": round(max(vstrict_values), 6) if vstrict_values else 0.0,
        },
        "astrict": {
            "mean": round(_safe_mean(astrict_values), 6),
            "min": round(min(astrict_values), 6) if astrict_values else 0.0,
            "max": round(max(astrict_values), 6) if astrict_values else 0.0,
        },
        "total_counts": total_counts,
    }


def compute_assignment_statistics(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        "n_tasks": len(tasks),
        "n_nuggets_total": 0,
        "n_nuggets_vital": 0,
        "n_nuggets_okay": 0,
        "assignment_counts": {"support": 0, "partial_support": 0, "not_support": 0},
        "auto_assignment_counts": {"support": 0, "partial_support": 0, "not_support": 0},
        "human_assignment_counts": {"support": 0, "partial_support": 0, "not_support": 0},
        "n_tasks_with_human_assignment": 0,
        "n_tasks_without_human_assignment": 0,
        "agreement_stats": {
            "n_nuggets_compared": 0,
            "n_exact_match": 0,
            "n_mismatch": 0,
            "agreement_rate": 0.0,
        },
    }

    has_human_assignments = any(
        isinstance(task.get("human_assignment"), list) and len(task.get("human_assignment")) > 0
        for task in tasks
    )

    for task in tasks:
        nuggets = task.get("nuggets") or task.get("auto_nugget", [])
        importance = task.get("importance") or task.get("auto_importance", [])

        stats["n_nuggets_total"] += len(nuggets)
        stats["n_nuggets_vital"] += importance.count("vital")
        stats["n_nuggets_okay"] += _count_okay(importance)

        auto_assignment = task.get("auto_assignment", [])
        human_assignment = task.get("human_assignment", [])

        for label in auto_assignment:
            if label in ASSIGNMENT_LABELS:
                stats["auto_assignment_counts"][label] += 1

        if human_assignment:
            stats["n_tasks_with_human_assignment"] += 1
            for label in human_assignment:
                if label in ASSIGNMENT_LABELS:
                    stats["human_assignment_counts"][label] += 1

            if auto_assignment and len(auto_assignment) == len(human_assignment):
                for auto_label, human_label in zip(auto_assignment, human_assignment):
                    stats["agreement_stats"]["n_nuggets_compared"] += 1
                    if auto_label == human_label:
                        stats["agreement_stats"]["n_exact_match"] += 1
                    else:
                        stats["agreement_stats"]["n_mismatch"] += 1
        else:
            stats["n_tasks_without_human_assignment"] += 1

    compared = stats["agreement_stats"]["n_nuggets_compared"]
    if compared > 0:
        stats["agreement_stats"]["agreement_rate"] = stats["agreement_stats"]["n_exact_match"] / compared

    if has_human_assignments and stats["n_tasks_with_human_assignment"] > 0:
        stats["assignment_counts"] = stats["human_assignment_counts"].copy()
        stats["primary_assignment_source"] = "human"
    else:
        stats["assignment_counts"] = stats["auto_assignment_counts"].copy()
        stats["primary_assignment_source"] = "auto"

    return stats


def update_file(path: Path, dry_run: bool) -> tuple[bool, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return False, "top-level is not an object"

    changed = False
    notes: list[str] = []

    if isinstance(data.get("samples"), list):
        samples = data["samples"]
        data["score_aggregate"] = build_score_aggregate_from_samples(samples)
        changed = True
        notes.append(f"recomputed score_aggregate from {len(samples)} samples")

    if isinstance(data.get("tasks"), list):
        tasks = data["tasks"]
        n_injected = inject_task_statistics(tasks)
        if n_injected:
            notes.append(f"added per-task statistics to {n_injected}/{len(tasks)} tasks")
        data["assignment_statistics"] = compute_assignment_statistics(tasks)
        changed = True
        notes.append(f"recomputed assignment_statistics from {len(tasks)} tasks")

    if not changed:
        return False, "no supported statistics block found (samples/tasks missing)"

    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return True, "; ".join(notes)


def collect_json_files(paths: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            candidates.append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject per-task statistics and update top-level statistics in JSON files. "
            "Supply either --dataset <name> to target a directory beside this script, "
            "or pass explicit file/folder paths as positional arguments."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="File(s) or folder(s) to process (optional when --dataset is used)",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET_DIR",
        help=(
            "Name of a dataset directory located beside this script "
            "(e.g. Qampari_RAGW4C). Its JSON files are added to the processing list."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing files",
    )
    parser.add_argument(
        "--include-all-names",
        action="store_true",
        help="Process JSON files regardless of filename. By default, only files with edited/manual nuggetization in the name are updated.",
    )
    args = parser.parse_args()

    raw_paths: list[str] = list(args.paths)

    if args.dataset:
        dataset_dir = SCRIPT_DIR / args.dataset
        if not dataset_dir.is_dir():
            parser.error(
                f"--dataset '{args.dataset}' not found as a directory beside this script "
                f"(looked in {SCRIPT_DIR})"
            )
        raw_paths.append(str(dataset_dir))

    if not raw_paths:
        parser.error("Provide at least one path argument or use --dataset <name>.")

    files = collect_json_files(raw_paths)
    if not files:
        print("No JSON files found from provided paths.")
        return

    updated = 0
    skipped_name = 0
    skipped_unsupported = 0
    errors = 0

    for file_path in files:
        if not args.include_all_names and not has_allowed_name(file_path):
            skipped_name += 1
            continue

        try:
            did_update, note = update_file(file_path, dry_run=args.dry_run)
            if did_update:
                updated += 1
                mode = "WOULD UPDATE" if args.dry_run else "UPDATED"
                print(f"[{mode}] {file_path} -> {note}")
            else:
                skipped_unsupported += 1
                print(f"[SKIP] {file_path} -> {note}")
        except Exception as exc:  # keep batch processing resilient
            errors += 1
            print(f"[ERROR] {file_path} -> {exc}")

    print(
        f"Done. Updated: {updated}, "
        f"Skipped(name): {skipped_name}, "
        f"Skipped(unsupported): {skipped_unsupported}, "
        f"Errors: {errors}"
    )


if __name__ == "__main__":
    main()
