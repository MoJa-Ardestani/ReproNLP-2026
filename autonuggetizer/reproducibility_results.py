"""Results I/O, aggregates, and nuggetization-row helpers."""

from __future__ import annotations

import json


def load_reproducibility_samples(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
        return raw["samples"]
    if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        return raw["tasks"]
    return []


def extract_nuggets_from_record(record: dict) -> tuple[list[str], list[str]]:
    nuggets = record.get("nuggets")
    importance = record.get("importance")
    if isinstance(nuggets, list) and isinstance(importance, list) and len(nuggets) == len(importance):
        return nuggets, importance

    auto_nuggets = record.get("auto_nuggets")
    auto_importance = record.get("auto_importance")
    if (
        isinstance(auto_nuggets, list)
        and isinstance(auto_importance, list)
        and len(auto_nuggets) == len(auto_importance)
    ):
        return auto_nuggets, auto_importance

    return [], []


def compute_llm_aggregate(samples: list) -> dict:
    usages = [
        s["llm_usage"]
        for s in samples
        if isinstance(s, dict) and "llm_usage" in s and "error" not in s
    ]
    if not usages:
        return {
            "n_samples_with_llm_usage": 0,
            "total_llm_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_pipeline_seconds": 0.0,
            "total_llm_latency_seconds": 0.0,
            "mean_pipeline_seconds_per_sample": None,
            "mean_llm_latency_seconds_per_sample": None,
        }

    def _sum(key: str) -> float:
        return float(sum(u[key] for u in usages))

    n = len(usages)
    tp = _sum("pipeline_seconds")
    tl = _sum("llm_latency_seconds")
    return {
        "n_samples_with_llm_usage": n,
        "total_llm_calls": int(sum(u["n_calls"] for u in usages)),
        "total_input_tokens": int(sum(u["input_tokens"] for u in usages)),
        "total_output_tokens": int(sum(u["output_tokens"] for u in usages)),
        "total_pipeline_seconds": round(tp, 6),
        "total_llm_latency_seconds": round(tl, 6),
        "mean_pipeline_seconds_per_sample": round(tp / n, 6),
        "mean_llm_latency_seconds_per_sample": round(tl / n, 6),
    }


def compute_score_aggregate(samples: list) -> dict:
    valid = [s for s in samples if isinstance(s, dict) and "error" not in s and "scores" in s]
    if not valid:
        return {
            "n_valid_samples": 0,
            "vstrict": {"mean": None, "min": None, "max": None},
            "astrict": {"mean": None, "min": None, "max": None},
            "total_counts": {
                "nuggets_total": 0,
                "nuggets_vital": 0,
                "nuggets_okay": 0,
                "support_count": 0,
                "partial_support_count": 0,
                "not_support_count": 0,
            },
        }

    vs = [float(s["vstrict"]) for s in valid]
    a_s = [float(s["astrict"]) for s in valid]
    totals = {
        "nuggets_total": 0,
        "nuggets_vital": 0,
        "nuggets_okay": 0,
        "support_count": 0,
        "partial_support_count": 0,
        "not_support_count": 0,
    }
    for s in valid:
        sc = s.get("scores", {})
        for k in totals:
            totals[k] += int(sc.get(k, 0))

    return {
        "n_valid_samples": len(valid),
        "vstrict": {
            "mean": round(sum(vs) / len(vs), 6),
            "min": round(min(vs), 6),
            "max": round(max(vs), 6),
        },
        "astrict": {
            "mean": round(sum(a_s) / len(a_s), 6),
            "min": round(min(a_s), 6),
            "max": round(max(a_s), 6),
        },
        "total_counts": totals,
    }


def assignment_output_payload(samples: list[dict]) -> dict:
    return {
        "score_aggregate": compute_score_aggregate(samples),
        "llm_aggregate": compute_llm_aggregate(samples),
        "samples": samples,
    }


def save_assignment_json(path: str, samples: list[dict]) -> None:
    payload = assignment_output_payload(samples)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_reproducibility_json(path: str, samples: list) -> None:
    payload = {
        "score_aggregate": compute_score_aggregate(samples),
        "llm_aggregate": compute_llm_aggregate(samples),
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
