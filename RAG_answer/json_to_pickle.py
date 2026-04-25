#!/usr/bin/env python3
"""
Convert a RAG answers JSON (produced by run_rag_answers.py) to a pickle file
compatible with the reproducibility pipeline.

For each question the output record has:
  {
    'qid':           str,
    'question_text': str,
    'answer_list':   dict  (GT entities → proof passages),
    '<retrieval>_<model>_self_contained_paragraph_answer_raw': str,
    'gt_len':        int   (word count of answer_list flattened to one string),
    'gen_len':       int   (word count of the RAG answer),
  }

The answer key name is derived automatically from the JSON's 'backend' and 'model'
fields, e.g.:
  backend=wikipedia+openai, model=gpt-4o
    → wikipedia_gpt-4o_self_contained_paragraph_answer_raw
  backend=perplexity, model=sonar
    → perplexity_sonar_self_contained_paragraph_answer_raw

If 'answer_list' is missing from a JSON record (older runs pre-date the field),
the script looks it up from the source reproducibility pickle by qid.

Usage (from RAG_answer/):
  python3 json_to_pickle.py --generation-type RAGW4G --run-id 20260414_104900
  python3 json_to_pickle.py --generation-type RAGP --run-id 20260414_104900 --source-pickle ../data/reproducibility_Qampari.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
_RESULTS_ROOT = _HERE / "results" / "rag_answers"
_DATA_DIR = _EXPERIMENTS / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _answer_list_word_count(answer_list: dict) -> int:
    """Flatten all proof strings in answer_list to one big string, count words."""
    tokens: list[str] = []
    for proofs in answer_list.values():
        if isinstance(proofs, list):
            for p in proofs:
                tokens.extend(str(p).split())
        else:
            tokens.extend(str(proofs).split())
    return len(tokens)


def _answer_key(backend: str, model: str) -> str:
    """
    Build the field name for the generated answer.
    wikipedia+openai / gpt-4o  →  wikipedia_gpt-4o_self_contained_paragraph_answer_raw
    perplexity       / sonar   →  perplexity_sonar_self_contained_paragraph_answer_raw
    """
    retrieval = backend.split("+")[0]   # "wikipedia" or "perplexity"
    return f"{retrieval}_{model}_self_contained_paragraph_answer_raw"


def _load_source_pickle(path: Path) -> dict[str, dict]:
    """Load a reproducibility file (JSON or pickle) and return a {qid -> record} lookup."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        with open(path, "rb") as f:
            records = pickle.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected list, got {type(records)}")
    return {str(r.get("qid", r.get("question_id", ""))): r for r in records}


def _generation_type_from_key(answer_key: str) -> str:
    """Derive generation type tag from the answer field name.

    wikipedia_*  → RAGW4G
    perplexity_* → RAGP
    """
    if answer_key.startswith("wikipedia_"):
        return "RAGW4G"
    if answer_key.startswith("perplexity_"):
        return "RAGP"
    return "RAGW4G"  # safe fallback


def _auto_source_file(base_dataset: str) -> Path | None:
    """Find the baseline file for a base dataset name (e.g. 'Qampari') in data/."""
    for candidate in [
        _DATA_DIR / f"reproducibility_{base_dataset}.json",
        _DATA_DIR / f"reproducibility_{base_dataset}_PARAM.json",
    ]:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_json_to_pickle(
    json_path: Path,
    source_lookup: dict[str, dict] | None,
    verbose: bool = True,
) -> tuple[list[dict], str]:
    """
    Convert one RAG answers JSON file to a list of output records.

    Returns (records, answer_key_name).
    """
    with open(json_path, encoding="utf-8") as f:
        entries: list[dict] = json.load(f)

    if not entries:
        raise ValueError(f"Empty JSON: {json_path}")

    # Derive the answer key from the first non-error entry
    first = next((e for e in entries if "rag_answer" in e), None)
    if first is None:
        raise ValueError(f"No successful entries in {json_path}")

    key = _answer_key(first["backend"], first["model"])
    if verbose:
        print(f"  Answer field: '{key}'")

    records: list[dict] = []
    missing_answer_list = 0
    missing_rag = 0

    for entry in entries:
        qid = entry.get("qid", "")
        question_text = entry.get("question_text", "")
        rag_answer = entry.get("rag_answer", "")

        if not rag_answer:
            missing_rag += 1
            if verbose:
                print(f"  [SKIP] qid={qid} — no rag_answer (error entry)", flush=True)
            continue

        # --- answer_list ---
        answer_list = entry.get("answer_list")
        if not answer_list and source_lookup is not None:
            src = source_lookup.get(qid)
            if src:
                answer_list = src.get("answer_list", {})
            else:
                missing_answer_list += 1
                if verbose:
                    print(f"  [WARN] qid={qid} — not found in source pickle", flush=True)
        if answer_list is None:
            answer_list = {}
            missing_answer_list += 1

        gt_len = _answer_list_word_count(answer_list)
        gen_len = _word_count(rag_answer)

        records.append({
            "qid": qid,
            "question_text": question_text,
            "answer_list": answer_list,
            key: rag_answer,
            "gt_len": gt_len,
            "gen_len": gen_len,
            "total": gt_len + gen_len
        })

    if verbose:
        print(f"  Converted {len(records)} records "
              f"({missing_rag} skipped — no answer, "
              f"{missing_answer_list} missing answer_list)")

    return records, key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert RAG answers JSON to reproducibility-compatible JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 json_to_pickle.py --generation-type RAGW4G --run-id 20260414_104900
  python3 json_to_pickle.py --generation-type RAGP   --run-id 20260414_104900
        """,
    )
    parser.add_argument(
        "--generation-type",
        required=True,
        metavar="TAG",
        help=(
            "Folder name under results/rag_answers/ for this run "
            "(e.g. RAGW4G, RAGW8G, RAGW4M from run_rag_answers.py, or RAGP for Perplexity)."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Timestamp folder name (e.g. 20260414_104900)",
    )
    parser.add_argument(
        "--source-pickle",
        type=Path,
        default=None,
        help=(
            "Reproducibility file to look up missing answer_list fields. "
            "If omitted, auto-detected from dataset name."
        ),
    )
    args = parser.parse_args()

    run_dir = _RESULTS_ROOT / args.generation_type / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    json_files = sorted(run_dir.glob("*_rag_answers.json"))
    if not json_files:
        print(f"ERROR: no *_rag_answers.json files in {run_dir}", file=sys.stderr)
        return 1

    overall_ok = True

    for json_path in json_files:
        # dataset_key: "Qampari_RAGW4G_rag_answers.json" → "Qampari_RAGW4G"
        dataset_key = json_path.stem.replace("_rag_answers", "")
        # base dataset name for source file lookup: "Qampari_RAGW4G" → "Qampari"
        base_dataset = dataset_key
        for tag in ("_RAGW4G", "_RAGP", "_PARAM", "_HUMAN"):
            base_dataset = base_dataset.replace(tag, "")
        base_dataset = re.sub(r"_RAGW\d+[GM]$", "", base_dataset)
        print(f"\n=== {dataset_key} ({json_path.name}) ===")

        # --- Resolve source file for answer_list lookup ---
        source_lookup: dict[str, dict] | None = None
        if args.source_pickle:
            sp = args.source_pickle
            if not sp.is_absolute():
                sp = _HERE / sp
            if sp.is_file():
                source_lookup = _load_source_pickle(sp)
                print(f"  Source file: {sp.name} ({len(source_lookup)} records)")
            else:
                print(f"  [WARN] source file not found: {sp}", file=sys.stderr)
        else:
            auto = _auto_source_file(base_dataset)
            if auto:
                source_lookup = _load_source_pickle(auto)
                print(f"  Source file (auto): {auto.name} ({len(source_lookup)} records)")
            else:
                print(f"  [WARN] no source file found for '{base_dataset}' — answer_list will be empty if missing from JSON")

        # --- Peek at first record for generation_type before conversion ---
        with open(json_path, encoding="utf-8") as f:
            raw_entries: list[dict] = json.load(f)
        first_entry = next((e for e in raw_entries if "rag_answer" in e), {})

        # --- Convert ---
        try:
            records, key = convert_json_to_pickle(json_path, source_lookup)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            overall_ok = False
            continue

        if not records:
            print("  No records to write, skipping.")
            continue

        # --- Derive generation type from stamped field, fall back to answer key ---
        generation_type = (
            first_entry.get("generation_type")    # stamped by run_rag_answers.py
            or _generation_type_from_key(key)     # fallback for older files
        )
        out_path = _DATA_DIR / f"reproducibility_{base_dataset}_{generation_type}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # --- Summary ---
        gt_lens = [r["gt_len"] for r in records]
        gen_lens = [r["gen_len"] for r in records]
        print(f"  Wrote {len(records)} records → {out_path.name}")
        print(f"  gt_len  : min={min(gt_lens)}, max={max(gt_lens)}, avg={sum(gt_lens)/len(gt_lens):.0f} words")
        print(f"  gen_len : min={min(gen_lens)}, max={max(gen_lens)}, avg={sum(gen_lens)/len(gen_lens):.0f} words")
        print(f"  Answer field: '{key}'")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
