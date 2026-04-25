"""Core reproducibility pipeline loop and assignment-only mode."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from tqdm import tqdm

from autonuggetizer.pipeline import assign_nuggets, run_autonuggetizer
from autonuggetizer.scoring import compute_scores
from autonuggetizer.reproducibility_data import load_data, prepare_samples
from autonuggetizer.reproducibility_datasets import DATASETS, dataset_data_relpath
from autonuggetizer.reproducibility_paths import EXPERIMENTS_ROOT, auto_assignment_filename
from autonuggetizer.reproducibility_results import (
    extract_nuggets_from_record,
    load_reproducibility_samples,
    save_assignment_json,
    save_reproducibility_json,
    compute_llm_aggregate,
)

logger = logging.getLogger(__name__)


def save_trace(trace: dict, trace_dir: str, idx: int, qid: str):
    os.makedirs(trace_dir, exist_ok=True)
    safe_qid = qid.replace("/", "_").replace(" ", "_")[:80]
    filename = f"{idx + 1:03d}_{safe_qid}.json"
    path = os.path.join(trace_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    return path


def run(
    dataset_name: str,
    provider: str,
    model: str,
    output_dir: str,
    limit: int = 0,
    start_from: int = 0,
    dry_run: bool = False,
    save_traces: bool = False,
):
    cfg = DATASETS[dataset_name]
    data_path = dataset_data_relpath(cfg)
    logger.info("Loading %s from %s ...", dataset_name, data_path)
    raw_data = load_data(EXPERIMENTS_ROOT / data_path)

    samples = prepare_samples(raw_data, cfg)
    logger.info("%s: %d valid samples (from %d raw)", dataset_name, len(samples), len(raw_data))

    if limit > 0:
        samples = samples[:limit]
        logger.info("Limiting to first %d samples", limit)

    if dry_run:
        logger.info("DRY RUN — showing %d samples without calling LLM", len(samples))
        for i, s in enumerate(samples[:5]):
            L = s["_length_stats"]
            logger.info(
                "  [%d] qid=%s, question=%s..., %d segments, passage=%d chars, "
                "gt_len=%s gen_len=%s total=%s length_tag=%s",
                i,
                s["qid"],
                s["question"][:60],
                len(s["context_segments"]),
                len(s["passage"]),
                L["gt_len"],
                L["gen_len"],
                L["total"],
                L["length_tag"],
            )
        if len(samples) > 5:
            logger.info("  ... +%d more", len(samples) - 5)
        logger.info("  Data is ready. Remove --dry-run to execute.")

        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, auto_assignment_filename(dataset_name))
    trace_dir = os.path.join(output_dir, "traces", dataset_name) if save_traces else None

    existing = []
    if start_from > 0 and os.path.exists(output_path):
        existing = load_reproducibility_samples(output_path)
        logger.info("Resuming: %d existing results", len(existing))

    results = existing[:]

    sample_lookup = {s["qid"]: s for s in samples}

    for i, sample in enumerate(
        tqdm(
            samples[start_from:],
            desc=dataset_name,
            initial=start_from,
            total=len(samples),
        )
    ):
        idx = i + start_from
        try:
            pipeline_out = run_autonuggetizer(
                query=sample["question"],
                context_segments=sample["context_segments"],
                passage=sample["passage"],
                model=model,
                provider=provider,
            )
            scores = compute_scores(pipeline_out["importance"], pipeline_out["assignment"])

            result_entry = {
                "qid": sample["qid"],
                "dataset": dataset_name,
                "question": sample["question"],
                "passage": sample["passage"],
                "gt_len": sample["_length_stats"]["gt_len"],
                "gen_len": sample["_length_stats"]["gen_len"],
                "total": sample["_length_stats"]["total"],
                "length_tag": sample["_length_stats"]["length_tag"],
                "n_context_segments": len(sample["context_segments"]),
                "nuggets": pipeline_out["nuggets"],
                "importance": pipeline_out["importance"],
                "assignment": pipeline_out["assignment"],
                "vstrict": scores["vstrict"],
                "astrict": scores["astrict"],
                "scores": scores,
                "llm_usage": pipeline_out["llm_usage"],
            }
            results.append(result_entry)

            if save_traces and "trace" in pipeline_out:
                trace_data = {
                    "sample": {
                        "qid": sample["qid"],
                        "question": sample["question"],
                        "gt_len": sample["_length_stats"]["gt_len"],
                        "gen_len": sample["_length_stats"]["gen_len"],
                        "total": sample["_length_stats"]["total"],
                        "length_tag": sample["_length_stats"]["length_tag"],
                        "answer_list": sample["_answer_list"],
                        "generated_passage_key": sample["_raw_key"],
                        "generated_passage": sample["passage"],
                    },
                    **pipeline_out["trace"],
                    "scores": scores,
                }
                trace_path = save_trace(trace_data, trace_dir, idx, sample["qid"])
                logger.info("  Trace saved: %s", trace_path)

        except Exception as e:
            logger.error("Failed on %s: %s", sample["qid"], e)
            results.append({
                "qid": sample["qid"],
                "dataset": dataset_name,
                "question": sample["question"],
                "passage": sample["passage"],
                "gt_len": sample["_length_stats"]["gt_len"],
                "gen_len": sample["_length_stats"]["gen_len"],
                "total": sample["_length_stats"]["total"],
                "length_tag": sample["_length_stats"]["length_tag"],
                "error": str(e),
            })

        if (idx + 1) % 5 == 0:
            save_reproducibility_json(output_path, results)

    save_reproducibility_json(output_path, results)
    logger.info("Saved %d results to %s", len(results), output_path)

    valid = [r for r in results if "error" not in r]
    if valid:
        vs = [r["vstrict"] for r in valid]
        as_ = [r["astrict"] for r in valid]
        logger.info(
            "  Vstrict: mean=%.4f, min=%.4f, max=%.4f",
            sum(vs) / len(vs),
            min(vs),
            max(vs),
        )
        logger.info(
            "  Astrict: mean=%.4f, min=%.4f, max=%.4f",
            sum(as_) / len(as_),
            min(as_),
            max(as_),
        )

    agg = compute_llm_aggregate(results)
    if agg["n_samples_with_llm_usage"]:
        logger.info(
            "  LLM (dataset totals): calls=%d, in_tokens=%d, out_tokens=%d, "
            "pipeline_s=%.2f, api_s=%.2f",
            agg["total_llm_calls"],
            agg["total_input_tokens"],
            agg["total_output_tokens"],
            agg["total_pipeline_seconds"],
            agg["total_llm_latency_seconds"],
        )

    if save_traces:
        logger.info("  Traces saved to: %s", trace_dir)


def run_assignment_only_from_nuggetization(
    dataset_name: str,
    provider: str,
    model: str,
    nuggetization_file: str,
    output_path: str,
    limit: int = 0,
    start_from: int = 0,
    skip_done: bool = False,
) -> None:
    cfg = DATASETS[dataset_name]
    raw_data = load_data(EXPERIMENTS_ROOT / dataset_data_relpath(cfg))
    sample_lookup = {s["qid"]: s for s in prepare_samples(raw_data, cfg)}

    nug_rows = load_reproducibility_samples(nuggetization_file)
    if not nug_rows:
        logger.error("No rows found in nuggetization file: %s", nuggetization_file)
        return

    if limit > 0:
        nug_rows = nug_rows[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    existing = []
    if skip_done and os.path.exists(output_path):
        existing = load_reproducibility_samples(output_path)
        done_qids: set[str] = {r["qid"] for r in existing if isinstance(r, dict) and r.get("qid")}
        before = len(nug_rows)
        nug_rows = [r for r in nug_rows if r.get("qid") not in done_qids]
        logger.info(
            "skip-done: %d already processed, %d new to run (was %d total)",
            len(done_qids),
            len(nug_rows),
            before,
        )
        start_from = 0
    elif start_from > 0 and os.path.exists(output_path):
        existing = load_reproducibility_samples(output_path)
        logger.info("Resuming assignment-only run: %d existing rows", len(existing))

    results = existing[:]
    rows_to_process = nug_rows[start_from:]
    for i, row in enumerate(
        tqdm(
            rows_to_process,
            desc=f"{dataset_name} assignment-only",
            initial=start_from,
            total=len(nug_rows) + start_from,
        )
    ):
        idx = i + start_from
        qid = row.get("qid")
        sample = sample_lookup.get(qid, {})
        question = row.get("question", sample.get("question", ""))
        passage = row.get("passage", sample.get("passage", ""))
        nuggets, importance = extract_nuggets_from_record(row)

        if not qid or not question or not passage or not nuggets:
            logger.warning("Skipping malformed row at idx=%d qid=%s", idx, qid)
            results.append({
                "qid": qid,
                "dataset": dataset_name,
                "question": question,
                "passage": passage,
                "nuggets": nuggets,
                "importance": importance,
                "error": "missing required fields for assignment-only run",
            })
            continue

        try:
            usage_start = datetime.now()
            assignment = assign_nuggets(
                query=question,
                passage=passage,
                nuggets=nuggets,
                model=model,
                provider=provider,
            )
            elapsed = (datetime.now() - usage_start).total_seconds()
            scores = compute_scores(importance, assignment)
            results.append({
                "qid": qid,
                "dataset": dataset_name,
                "question": question,
                "passage": passage,
                "nuggets": nuggets,
                "importance": importance,
                "assignment": assignment,
                "vstrict": scores["vstrict"],
                "astrict": scores["astrict"],
                "scores": scores,
                "llm_usage": {
                    "n_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "llm_latency_seconds": 0.0,
                    "pipeline_seconds": round(elapsed, 6),
                },
            })
        except Exception as e:
            logger.error("Assignment-only failed on %s: %s", qid, e)
            results.append({
                "qid": qid,
                "dataset": dataset_name,
                "question": question,
                "passage": passage,
                "nuggets": nuggets,
                "importance": importance,
                "error": str(e),
            })

        if (idx + 1) % 5 == 0:
            save_assignment_json(output_path, results)

    save_assignment_json(output_path, results)
    logger.info("Saved %d assignment-only rows to %s", len(results), output_path)
