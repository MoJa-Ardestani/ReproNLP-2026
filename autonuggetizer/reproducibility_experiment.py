"""Experiment directories, run_config.json, and provenance helpers."""

from __future__ import annotations

import inspect
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

from autonuggetizer.pipeline import BATCH_SIZE, run_autonuggetizer
from autonuggetizer.reproducibility_datasets import data_sources_manifest
from autonuggetizer.reproducibility_paths import EXPERIMENTS_ROOT, auto_assignment_filename

logger = logging.getLogger(__name__)


def git_info() -> dict:
    repo_root = EXPERIMENTS_ROOT.parent
    info: dict = {"commit": None, "dirty": None}
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0:
            info["commit"] = r.stdout.strip()
        r2 = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r2.returncode == 0:
            info["dirty"] = len(r2.stdout.strip()) > 0
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def effective_model(provider: str, model_arg: str) -> str:
    from autonuggetizer.llm import DEFAULT_MODELS, _load_credentials

    if model_arg and model_arg != "default":
        return model_arg
    creds = _load_credentials()
    cred_model = (creds.get(provider) or {}).get("model", "")
    return cred_model if cred_model else DEFAULT_MODELS.get(provider, "gpt-4o")


def n_top_nuggets_default() -> int:
    return inspect.signature(run_autonuggetizer).parameters["n_top_nuggets"].default


def make_experiment_dir(base_dir: str, dataset_name: str, exp_id: str, args) -> str:
    exp_dir = os.path.join(base_dir, dataset_name, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    manifest = {
        "experiment_id": exp_id,
        "started_at": datetime.now().isoformat(),
        "dataset": dataset_name,
        "provider": args.provider,
        "model": args.model,
        "effective_model_resolved": effective_model(args.provider, args.model),
        "limit": args.limit,
        "start_from": args.start_from,
        "verbose": args.verbose,
        "resume_experiment": args.resume_experiment,
        "data_sources": data_sources_manifest(),
        "pipeline_constants": {
            "BATCH_SIZE": BATCH_SIZE,
            "n_top_nuggets": n_top_nuggets_default(),
        },
        "chat_completion_defaults": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "seed": 42,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "git": git_info(),
        "finished_at": None,
    }
    with open(os.path.join(exp_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Experiment dir: %s", exp_dir)
    logger.info("Effective model for API: %s", manifest["effective_model_resolved"])
    return exp_dir


def finalize_experiment_run(exp_dir: str, dataset_name: str) -> None:
    path = os.path.join(exp_dir, "run_config.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["finished_at"] = datetime.now().isoformat()
    cfg["result_file"] = os.path.join(exp_dir, auto_assignment_filename(dataset_name))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def assignment_only_run_config_filename(nug_type: str) -> str:
    safe = (nug_type or "auto").strip().lower()
    return f"run_config_{safe}.json"


def write_assignment_only_run_config(
    out_dir: str,
    dataset_name: str,
    nug_type: str,
    nuggetization_file: str,
    assignment_output_file: str,
    args,
    started_at: str,
    finished_at: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        "mode": "assignment_only_from_nuggetization",
        "nuggetization_type": nug_type,
        "experiment_id": os.path.basename(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset": dataset_name,
        "provider": args.provider,
        "model": args.model,
        "effective_model_resolved": effective_model(args.provider, args.model),
        "limit": args.limit,
        "start_from": args.start_from,
        "verbose": args.verbose,
        "input_nuggetization_file": os.path.abspath(nuggetization_file),
        "result_file": os.path.abspath(assignment_output_file),
        "data_sources": data_sources_manifest(),
        "pipeline_constants": {
            "BATCH_SIZE": BATCH_SIZE,
            "n_top_nuggets": n_top_nuggets_default(),
        },
        "chat_completion_defaults": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "seed": 42,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "git": git_info(),
    }

    path = os.path.join(out_dir, assignment_only_run_config_filename(nug_type))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path
