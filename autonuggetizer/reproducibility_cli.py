"""CLI entry for reproducibility runs (`main_run_reproducibility.py` is a thin shim here)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from autonuggetizer.pipeline import enable_tracing
from autonuggetizer.reproducibility_datasets import DATASETS
from autonuggetizer.reproducibility_experiment import (
    finalize_experiment_run,
    make_experiment_dir,
    write_assignment_only_run_config,
)
from autonuggetizer.reproducibility_paths import (
    EXPERIMENTS_ROOT,
    latest_experiment_id,
    next_non_clobber_path,
    resolve_reproducibility_output_dir,
)
from autonuggetizer.reproducibility_run import run, run_assignment_only_from_nuggetization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AutoNuggetizer reproducibility run")
    parser.add_argument("--dataset", default="all", choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["openai", "gemini", "azure", "qwen", "llama", "claude"],
    )
    parser.add_argument("--model", default="default")
    parser.add_argument(
        "--output-dir",
        default="results/reproducibility",
        help="Reproducibility root; relative paths are resolved under experiments/",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N samples (0=all)")
    parser.add_argument("--start-from", type=int, default=0, help="Resume from this index")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without calling LLM")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Save per-sample trace files with all intermediate pipeline data",
    )
    parser.add_argument(
        "--resume-experiment",
        default=None,
        help="Resume a previous experiment by its directory name (e.g. 20260405_183012)",
    )
    parser.add_argument(
        "--assignment-only-from-nuggetization",
        default=None,
        help=(
            "Path to a nuggetization JSON file, or a directory containing one (then use "
            "--nuggetization-json). Runs assignment only (Stage 3). "
            "Default output: {output-dir}/{dataset_key}/latest_YYYYMMDD_HHMMSS/ "
            "(new timestamp if none). Existing files are not overwritten (_1, _2, ...)."
        ),
    )
    parser.add_argument(
        "--nuggetization-json",
        default=None,
        metavar="FILENAME",
        help=(
            "When --assignment-only-from-nuggetization points to a directory, basename of the "
            "JSON inside it (e.g. Qampari_edited_nuggetization.json)."
        ),
    )
    parser.add_argument(
        "--assignment-output",
        default=None,
        metavar="PATH",
        help=(
            "Assignment-only: output .json path (overrides default under --output-dir). "
            "Relative paths are under experiments/. If the file exists, _1/_2 suffix is used."
        ),
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Assignment-only: read existing output file, skip qids already present, "
            "and append only new results.  Use this after adding samples to a nugget "
            "file to avoid re-processing already-completed entries.  Requires "
            "--assignment-only-from-nuggetization and --assignment-output pointing at "
            "the file you want to update in place."
        ),
    )
    args = parser.parse_args()
    args.output_dir = resolve_reproducibility_output_dir(args.output_dir)

    if args.verbose:
        enable_tracing(True)

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ds in datasets:
        if args.assignment_only_from_nuggetization:
            raw_in = args.assignment_only_from_nuggetization
            in_path = Path(raw_in).expanduser()
            if not in_path.exists():
                logger.error("Nuggetization path does not exist: %s", in_path)
                sys.exit(1)
            if in_path.is_dir():
                if not args.nuggetization_json:
                    logger.error(
                        "When --assignment-only-from-nuggetization is a directory, set "
                        "--nuggetization-json (e.g. Qampari_edited_nuggetization.json)."
                    )
                    sys.exit(1)
                nug_path = str((in_path / args.nuggetization_json).resolve())
                if not os.path.isfile(nug_path):
                    logger.error("Not a file: %s", nug_path)
                    sys.exit(1)
            else:
                nug_path = str(in_path.resolve())
                if args.nuggetization_json:
                    logger.warning(
                        "Ignoring --nuggetization-json because input path is a file: %s",
                        nug_path,
                    )

            nug_filename = os.path.basename(nug_path)
            nug_type = "auto"
            if "_manual_nuggetization" in nug_filename:
                nug_type = "manual"
            elif "_edited_nuggetization" in nug_filename:
                nug_type = "edited"
            elif "_auto_nuggetization" in nug_filename:
                nug_type = "auto"

            if args.assignment_output:
                out_raw = Path(os.path.expanduser(args.assignment_output))
                if out_raw.is_absolute():
                    out_path = str(out_raw.resolve())
                else:
                    out_path = str((EXPERIMENTS_ROOT / out_raw).resolve())
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            else:
                exp_subdir = latest_experiment_id(args.output_dir, ds, exp_id)
                out_dir = os.path.join(args.output_dir, ds, exp_subdir)
                os.makedirs(out_dir, exist_ok=True)
                out_filename = f"{ds}_{nug_type}_nuggetization_auto_assignment.json"
                out_path = os.path.join(out_dir, out_filename)

            if not args.skip_done:
                out_path = next_non_clobber_path(out_path)
            logger.info("Assignment-only mode: %s nuggetization -> %s", nug_type, out_path)

            started_at = datetime.now().isoformat()
            run_assignment_only_from_nuggetization(
                dataset_name=ds,
                provider=args.provider,
                model=args.model,
                nuggetization_file=nug_path,
                output_path=out_path,
                limit=args.limit,
                start_from=args.start_from,
                skip_done=args.skip_done,
            )
            rc_path = write_assignment_only_run_config(
                out_dir=os.path.dirname(out_path),
                dataset_name=ds,
                nug_type=nug_type,
                nuggetization_file=nug_path,
                assignment_output_file=out_path,
                args=args,
                started_at=started_at,
                finished_at=datetime.now().isoformat(),
            )
            logger.info("  Run config: %s", rc_path)
            continue

        if args.resume_experiment:
            exp_dir = os.path.join(args.output_dir, ds, args.resume_experiment)
            if not os.path.isdir(exp_dir):
                logger.error("Experiment directory not found: %s", exp_dir)
                sys.exit(1)
            logger.info("Resuming experiment: %s", exp_dir)
        else:
            exp_dir = make_experiment_dir(args.output_dir, ds, exp_id, args)

        run(
            ds,
            args.provider,
            args.model,
            exp_dir,
            args.limit,
            args.start_from,
            args.dry_run,
            save_traces=args.verbose,
        )

        finalize_experiment_run(exp_dir, ds)


if __name__ == "__main__":
    main()
