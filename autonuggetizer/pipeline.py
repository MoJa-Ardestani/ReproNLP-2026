"""
AutoNuggetizer three-stage pipeline.

Reproduces the methodology from Pradeep et al. (SIGIR 2025):
  Stage 1 — Iterative Nugget Creation from context segments
  Stage 2 — Nugget Importance Scoring (vital / okay)
  Stage 3 — Nugget Assignment against a system passage (support / partial / not)
"""

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .prompts import (
    NUGGET_CREATION_SYSTEM, NUGGET_CREATION_USER,
    NUGGET_SCORING_SYSTEM, NUGGET_SCORING_USER,
    NUGGET_ASSIGNMENT_SYSTEM, NUGGET_ASSIGNMENT_USER,
)
from .llm import ChatCompletionResult, chat_completion

logger = logging.getLogger(__name__)


@dataclass
class PipelineUsage:
    """Aggregates LLM usage across create / score / assign (API time + token counts)."""

    log_calls: bool = False
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_sec: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, res: ChatCompletionResult, meta: Optional[dict[str, Any]] = None) -> None:
        self.n_calls += 1
        pt = res.prompt_tokens if res.prompt_tokens is not None else 0
        ct = res.completion_tokens if res.completion_tokens is not None else 0
        self.input_tokens += pt
        self.output_tokens += ct
        self.llm_latency_sec += res.latency_sec
        if self.log_calls:
            row: dict[str, Any] = {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "latency_sec": res.latency_sec,
            }
            if meta:
                row.update(meta)
            self.calls.append(row)

    def summary(self, pipeline_seconds: float) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_latency_seconds": round(self.llm_latency_sec, 6),
            "pipeline_seconds": round(pipeline_seconds, 6),
        }

BATCH_SIZE = 10  # paper uses at most 10 nuggets/segments per LLM call

# Toggle for detailed intermediate tracing. Set via enable_tracing().
_tracing = False
_current_trace: Optional[dict] = None

def enable_tracing(on: bool = True):
    """Turn pipeline tracing on/off. When on, run_autonuggetizer()
    returns an extra 'trace' key with all intermediate data."""
    global _tracing
    _tracing = on

# Keep backward-compatible alias
enable_verbose_logging = enable_tracing

def _trace_append(stage: str, key: str, value):
    """Record intermediate data into the current trace."""
    if not _tracing or _current_trace is None:
        return
    if stage not in _current_trace:
        _current_trace[stage] = {}
    _current_trace[stage][key] = value


def _parse_list(raw: str) -> list[str]:
    """Robustly parse a Python-style list from LLM output."""
    raw = raw.strip()
    # Try direct parse first
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except (ValueError, SyntaxError):
        pass

    # Fallback: extract the first [...] block
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(match.group())
            if isinstance(result, list):
                return [str(x) for x in result]
        except (ValueError, SyntaxError):
            pass

    logger.warning("Could not parse list from LLM output: %s", raw[:200])
    return []


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Iterative Nugget Creation
# ──────────────────────────────────────────────────────────────────────

def create_nuggets(
    query: str,
    context_segments: list[str],
    model: str = "gpt-4o",
    provider: str = "openai",
    usage: Optional[PipelineUsage] = None,
) -> list[str]:
    """
    Iteratively build a nugget list by feeding batches of 10 context segments.

    Returns the final list of up to 30 atomic nuggets.
    """
    nuggets: list[str] = []
    creation_batches = []

    for batch_start in range(0, len(context_segments), BATCH_SIZE):
        batch = context_segments[batch_start:batch_start + BATCH_SIZE]
        context_block = "\n".join(
            f"[{i + 1}] {seg}" for i, seg in enumerate(batch)
        )
        nugget_list_str = json.dumps(nuggets) if nuggets else "[]"

        user_msg = NUGGET_CREATION_USER.format(
            query=query,
            context_block=context_block,
            nugget_list=nugget_list_str,
            nugget_list_length=len(nuggets),
        )
        batch_range = f"{batch_start}–{batch_start + len(batch)}"
        res = chat_completion(
            system=NUGGET_CREATION_SYSTEM,
            user=user_msg,
            model=model,
            provider=provider,
        )
        if usage is not None:
            usage.record(res, {"stage": "stage1_creation", "batch_range": batch_range})
        nuggets = _parse_list(res.text)
        logger.info("After batch %d–%d: %d nuggets",
                     batch_start, batch_start + len(batch), len(nuggets))

        creation_batches.append({
            "batch_range": batch_range,
            "segments_in_batch": batch,
            "llm_raw_output": res.text,
            "nuggets_after": list(nuggets),
        })

    _trace_append("stage1_creation", "batches", creation_batches)
    _trace_append("stage1_creation", "final_nuggets", list(nuggets))

    return nuggets


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Nugget Importance Scoring
# ──────────────────────────────────────────────────────────────────────

def score_nuggets(
    query: str,
    nuggets: list[str],
    model: str = "gpt-4o",
    provider: str = "openai",
    usage: Optional[PipelineUsage] = None,
) -> list[str]:
    """
    Label each nugget as 'vital' or 'okay'.
    Processes in batches of 10 (as per the paper).

    Returns a list of labels parallel to the nugget list.
    """
    all_labels: list[str] = []
    scoring_batches = []

    for batch_start in range(0, len(nuggets), BATCH_SIZE):
        batch = nuggets[batch_start:batch_start + BATCH_SIZE]
        user_msg = NUGGET_SCORING_USER.format(
            query=query,
            num_nuggets=len(batch),
            nugget_list=json.dumps(batch),
        )

        batch_range = f"{batch_start}–{batch_start + len(batch)}"
        res = chat_completion(
            system=NUGGET_SCORING_SYSTEM,
            user=user_msg,
            model=model,
            provider=provider,
        )
        if usage is not None:
            usage.record(res, {"stage": "stage2_scoring", "batch_range": batch_range})
        labels = _parse_list(res.text)

        if len(labels) != len(batch):
            logger.warning("Scoring returned %d labels for %d nuggets; padding.",
                           len(labels), len(batch))
            labels += ["okay"] * (len(batch) - len(labels))
            labels = labels[:len(batch)]

        all_labels.extend(labels)
        scoring_batches.append({
            "batch_range": batch_range,
            "llm_raw_output": res.text,
            "labels": list(labels),
        })

    scored = [{"nugget": n, "importance": l} for n, l in zip(nuggets, all_labels)]
    _trace_append("stage2_scoring", "batches", scoring_batches)
    _trace_append("stage2_scoring", "results", scored)

    return all_labels


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Nugget Assignment
# ──────────────────────────────────────────────────────────────────────

def assign_nuggets(
    query: str,
    passage: str,
    nuggets: list[str],
    model: str = "gpt-4o",
    provider: str = "openai",
    usage: Optional[PipelineUsage] = None,
) -> list[str]:
    """
    For each nugget, check if it is 'support', 'partial_support', or
    'not_support' given a system passage.
    Processes in batches of 10 (as per the paper).

    Returns a list of assignment labels parallel to the nugget list.
    """
    all_labels: list[str] = []
    assign_batches = []

    for batch_start in range(0, len(nuggets), BATCH_SIZE):
        batch = nuggets[batch_start:batch_start + BATCH_SIZE]
        user_msg = NUGGET_ASSIGNMENT_USER.format(
            query=query,
            passage=passage,
            num_nuggets=len(batch),
            nugget_list=json.dumps(batch),
        )

        batch_range = f"{batch_start}–{batch_start + len(batch)}"
        res = chat_completion(
            system=NUGGET_ASSIGNMENT_SYSTEM,
            user=user_msg,
            model=model,
            provider=provider,
        )
        if usage is not None:
            usage.record(res, {"stage": "stage3_assignment", "batch_range": batch_range})
        labels = _parse_list(res.text)

        if len(labels) != len(batch):
            logger.warning("Assignment returned %d labels for %d nuggets; padding.",
                           len(labels), len(batch))
            labels += ["not_support"] * (len(batch) - len(labels))
            labels = labels[:len(batch)]

        all_labels.extend(labels)
        assign_batches.append({
            "batch_range": batch_range,
            "llm_raw_output": res.text,
            "labels": list(labels),
        })

    assigned = [{"nugget": n, "assignment": l} for n, l in zip(nuggets, all_labels)]
    _trace_append("stage3_assignment", "batches", assign_batches)
    _trace_append("stage3_assignment", "results", assigned)

    return all_labels


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────

def _sort_and_truncate(
    nuggets: list[str],
    importance: list[str],
    n_top: int = 20,
) -> tuple[list[str], list[str]]:
    """
    Sort nuggets so vital comes before okay, then keep the top n_top.

    The paper (Section 3.1): "we sort the nuggets in descending order of
    importance and select the first 20 nuggets. This approach usually trims
    a few 'okay' nuggets."
    """
    pairs = list(zip(nuggets, importance))
    pairs.sort(key=lambda x: x[1] == "okay")  # vital first
    pairs = pairs[:n_top]
    if not pairs:
        return [], []
    sorted_nuggets, sorted_importance = zip(*pairs)
    return list(sorted_nuggets), list(sorted_importance)


def run_autonuggetizer(
    query: str,
    context_segments: list[str],
    passage: str,
    model: str = "gpt-4o",
    provider: str = "openai",
    precomputed_nuggets: Optional[list[str]] = None,
    n_top_nuggets: int = 20,
) -> dict:
    """
    Run the full 3-stage pipeline on a single sample.

    If precomputed_nuggets is provided, skip Stage 1 (nugget creation)
    and use them directly — useful when reusing facts from LongRecall's
    fact extraction as nuggets.

    After Stage 2, nuggets are sorted (vital first) and truncated to
    the top n_top_nuggets (default 20), matching the paper's methodology.

    Returns a dict with:
      - nuggets:     list[str]
      - importance:  list[str]   ("vital" / "okay")
      - assignment:  list[str]   ("support" / "partial_support" / "not_support")
      - llm_usage:   counts/timings for this sample (excludes trace file I/O)
    """
    global _current_trace
    usage = PipelineUsage(log_calls=_tracing)

    if _tracing:
        _current_trace = {
            "input": {
                "query": query,
                "context_segments": context_segments,
                "passage": passage,
                "passage_length": len(passage),
                "model": model,
                "provider": provider,
            }
        }

    t_pipeline = time.perf_counter()

    if precomputed_nuggets is not None:
        nuggets = precomputed_nuggets
        logger.info("Using %d precomputed nuggets (skipping creation).", len(nuggets))
    else:
        nuggets = create_nuggets(query, context_segments, model, provider, usage=usage)
        logger.info("Created %d nuggets.", len(nuggets))

    importance = score_nuggets(query, nuggets, model, provider, usage=usage)

    pre_count = len(nuggets)
    nuggets, importance = _sort_and_truncate(nuggets, importance, n_top_nuggets)
    logger.info("After sort+truncate: %d nuggets (%d vital).",
                len(nuggets), sum(1 for i in importance if i == "vital"))

    _trace_append("sort_truncate", "before_count", pre_count)
    _trace_append("sort_truncate", "after_count", len(nuggets))
    _trace_append("sort_truncate", "kept", [
        {"nugget": n, "importance": imp} for n, imp in zip(nuggets, importance)
    ])

    assignment = assign_nuggets(query, passage, nuggets, model, provider, usage=usage)

    pipeline_seconds = time.perf_counter() - t_pipeline
    llm_usage = usage.summary(pipeline_seconds)

    result = {
        "nuggets": nuggets,
        "importance": importance,
        "assignment": assignment,
        "llm_usage": llm_usage,
    }

    if _tracing and _current_trace is not None:
        _current_trace["llm_usage"] = dict(llm_usage)
        _current_trace["llm_calls"] = list(usage.calls)
        _current_trace["final"] = [
            {"nugget": n, "importance": imp, "assignment": asg}
            for n, imp, asg in zip(nuggets, importance, assignment)
        ]
        result["trace"] = _current_trace
        _current_trace = None

    return result
