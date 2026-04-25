#!/usr/bin/env python3
"""
Generate RAG-based self-contained paragraph answers for QA datasets.

Loads questions from a reproducibility pickle, retrieves relevant Wikipedia
passages (free), then calls an LLM to generate a comprehensive answer that
covers all entities in the question. Designed to produce output compatible
with the existing `gpt-4o_self_contained_paragraph_answer_raw` field format.

Backends
--------
  wikipedia+openai   Free Wikipedia retrieval + OpenAI generation.
                     Cheapest option: use --model gpt-4o-mini (~$0.001/question)
  wikipedia+gemini   Free Wikipedia retrieval + Gemini generation.
                     Has a free tier; use --model gemini-2.0-flash
  wikipedia+claude   Free Wikipedia retrieval + Anthropic Claude generation.
                     Use --model claude-haiku-4-5-20251001 (cheapest) or claude-opus-4-5.
  perplexity         All-in-one: Perplexity sonar searches the web and generates.
                     No separate retrieval step. ~$0.01-0.03/question.

Output
------
  results/rag_answers/<RAG tag>/<timestamp>/<dataset>_rag_answers.json
  RAG tag: RAGW{n}G (Wikipedia + n passages + GPT/OpenAI), RAGW{n}M (Wikipedia + Gemini),
             RAGW{n}C (Wikipedia + Claude), RAGP (Perplexity, no separate retrieval).
  Each record: { qid, question_text, rag_answer, backend, model,
                 retrieved_passages (wiki only), usage, num_passages (wiki only) }

Usage (from the project root)
-----------
  # Smoke test: 3 questions, cheapest backend
  python3 run_rag_answers.py --limit 3 --backend wikipedia+openai --model gpt-4o-mini

  # Wikipedia RAG with 8 passages (paths include RAGW8G for GPT)
  python3 run_rag_answers.py --num-passages 8 --backend wikipedia+openai --model gpt-4o-mini

  # Full Qampari run with Gemini (free tier; paths include RAGW4M by default)
  python3 run_rag_answers.py --backend wikipedia+gemini --model gemini-2.0-flash

  # Full Qampari run with Claude (paths include RAGW4C by default)
  python3 run_rag_answers.py --backend wikipedia+claude --model claude-opus-4-5

  # Perplexity all-in-one
  python3 run_rag_answers.py --limit 10 --backend perplexity --model sonar

  # Custom input file
  python3 run_rag_answers.py --input data/reproducibility_Qampari.json --limit 5

  # Skip first 499 rows; process from row 500 (1-based index in full file)
  python3 run_rag_answers.py --start-from 500
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent   # project root — one level up from RAG_answer/
_CREDS_PATH = _EXPERIMENTS / "credentials.yaml"
_RESULTS_ROOT = _HERE / "results" / "rag_answers"
_DEFAULT_INPUT = _EXPERIMENTS / "data" / "reproducibility_Qampari.json"

def _generation_type_tag(backend: str, num_passages: int) -> str:
    """
    Short tag for output paths and dataset_key suffix.
    RAG + W (Wikipedia) + {n} passages + G (GPT/OpenAI), M (Gemini), or C (Claude); Perplexity → RAGP.
    """
    if backend == "wikipedia+openai":
        return f"RAGW{num_passages}G"
    if backend == "wikipedia+gemini":
        return f"RAGW{num_passages}M"
    if backend == "wikipedia+claude":
        return f"RAGW{num_passages}C"
    if backend == "perplexity":
        return "RAGP"
    raise ValueError(f"Unknown backend: {backend!r}")

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

_creds_cache: dict | None = None


def _load_creds() -> dict:
    global _creds_cache
    if _creds_cache is not None:
        return _creds_cache
    if _CREDS_PATH.is_file():
        with open(_CREDS_PATH) as f:
            _creds_cache = yaml.safe_load(f) or {}
    else:
        _creds_cache = {}
    return _creds_cache


def _get_key(provider: str, env_var: str) -> str:
    creds = _load_creds()
    val = (creds.get(provider) or {}).get("api_key", "")
    if val and not val.startswith("sk-...") and not val.startswith("AIza...") and not val.startswith("pplx-..."):
        return val
    return os.environ.get(env_var, "")


def _get_default_model(provider: str) -> str:
    creds = _load_creds()
    return (creds.get(provider) or {}).get("model", "")


# ---------------------------------------------------------------------------
# Wikipedia retrieval (free, no API key required)
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_PASSAGE_MAX_CHARS = 1200   # chars per Wikipedia page extract (~200 words)
_DEFAULT_NUM_PASSAGES = 4   # default top-N passages (override with --num-passages)
# Wikipedia requires a descriptive User-Agent — requests with the default
# python-requests/x.x.x agent are blocked with 403 Forbidden.
_WIKI_HEADERS = {"User-Agent": "RAGAnswerGenerator/1.0 (research project; contact: researcher@example.com)"}


def _wiki_search(query: str, srlimit: int) -> list[str]:
    """Return page titles matching the query."""
    try:
        resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": srlimit,
                "format": "json",
            },
            headers=_WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        return [h["title"] for h in hits]
    except Exception as e:
        print(f"  [Wikipedia search error] {e}", flush=True)
        return []


def _wiki_extract(title: str) -> str:
    """Return plain-text intro extract for a Wikipedia page title."""
    try:
        resp = requests.get(
            _WIKI_API,
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            },
            headers=_WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            return text[:_PASSAGE_MAX_CHARS].strip()
    except Exception as e:
        print(f"  [Wikipedia extract error for '{title}'] {e}", flush=True)
    return ""


def retrieve_wikipedia(question: str, max_passages: int) -> list[dict[str, str]]:
    """
    Search Wikipedia for the question, return list of {title, text} passages.
    Two searches: full question + main noun phrase (subject of the question).
    Returns at most max_passages passages.
    """
    if max_passages < 1:
        raise ValueError("max_passages must be >= 1")
    # Request up to max_passages titles per query; merge + dedupe then cap.
    srlimit = min(max_passages, 50)

    titles_seen: set[str] = set()
    passages: list[dict[str, str]] = []

    # Search 1: full question
    for title in _wiki_search(question, srlimit):
        if title not in titles_seen:
            titles_seen.add(title)
            text = _wiki_extract(title)
            if text:
                passages.append({"title": title, "text": text})
                if len(passages) >= max_passages:
                    return passages[:max_passages]

    # Search 2: main subject — strip leading wh-words to get the entity
    subject = _extract_subject(question)
    if subject and subject.lower() != question.lower():
        for title in _wiki_search(subject, srlimit):
            if title not in titles_seen:
                titles_seen.add(title)
                text = _wiki_extract(title)
                if text:
                    passages.append({"title": title, "text": text})
                    if len(passages) >= max_passages:
                        break

    return passages[:max_passages]


def _extract_subject(question: str) -> str:
    """
    Heuristic: strip leading wh-word + auxiliary to get the main subject entity.
    e.g. "Martin Rackin produced software directed by whom?"
      -> "Martin Rackin"
    """
    q = question.strip().rstrip("?")
    # Remove leading wh-words
    q = re.sub(r"^(?:who|what|which|where|when|how|whom)\s+", "", q, flags=re.IGNORECASE)
    # Take first 3-4 words as likely subject
    words = q.split()
    return " ".join(words[:4]) if words else q


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with encyclopedic knowledge. "
    "Answer questions with factual accuracy and comprehensive coverage in the style of Wikipedia."
)

# Core task instructions — matches the original parametric-knowledge prompt exactly.
_TASK_INSTRUCTIONS = """\
Your objectives are:

1. **Provide an Exhaustive List of All Relevant Answer Entities:**
   - Identify the key elements and roles within the question to ensure a clear understanding of what is being asked (e.g., persons, roles, context).
   - Include ALL relevant entities that answer the question. Ensure no pertinent details are omitted, and strive to cover every possible correct entity extensively.

2. **Provide Standalone, Coherent Explanations:**
   - For each answer entity, write a standalone, coherent paragraph that is easily understandable.
   - Each paragraph should be informative, written in a style similar to Wikipedia, and explain the entity's relevance to the question.
   - Incorporate relevant information related to both the answer entity and the question.

3. **Ensure Diversity in Output:**
   - Provide diverse examples that span different genres, periods, or categories relevant to the question.
   - Do not omit any relevant entities.

4. **Prioritize Quality and Completeness Over Brevity:**
   - Ensure each paragraph offers significant insight through detailed and thorough explanations.
   - Avoid brief or templated responses.

5. **Avoid Hallucinations and Fabrications:**
   - Include only information that is verifiable on Wikipedia or other trustworthy resources.
   - Do not create or assume facts not present in reliable sources.

6. **Output Structure:**
   - Begin with the name of each entity, clearly marked in bold.
   - Immediately follow each entity's name with its corresponding explanation in a new paragraph.\
"""


def _build_user_prompt(question: str, passages: list[dict[str, str]]) -> str:
    """
    Build the user prompt.
    - If passages were retrieved: prepend them as primary context, ask LLM to
      supplement with parametric knowledge for any entities not covered.
    - If no passages (retrieval failed): fall back to pure parametric knowledge
      using the original prompt verbatim.
    """
    if passages:
        context_parts = []
        for p in passages:
            context_parts.append(f"[Wikipedia: {p['title']}]\n{p['text']}")
        context = "\n\n".join(context_parts)
        return (
            f"Using the Wikipedia passages below as your PRIMARY source of information, "
            f"generate a comprehensive and accurate response to the following question. "
            f"For any relevant answer entities NOT covered in the passages, supplement "
            f"with your own encyclopedic knowledge.\n\n"
            f"{_TASK_INSTRUCTIONS}\n\n"
            f"---\nWikipedia Context:\n\n{context}\n---\n\n"
            f"*Question: '{question}'*"
        )
    else:
        # Parametric fallback — original prompt, no retrieval context
        return (
            f"Using your expansive knowledge, generate a comprehensive and accurate response "
            f"to the following question. {_TASK_INSTRUCTIONS}\n\n"
            f"*Question: '{question}'*"
        )


# ---------------------------------------------------------------------------
# LLM generation — Wikipedia + OpenAI/Gemini backends
# ---------------------------------------------------------------------------

MAX_RETRIES = 4
BACKOFF_BASE = 2.0


def _call_openai(model: str, user_prompt: str) -> dict[str, Any]:
    from openai import OpenAI, RateLimitError

    api_key = _get_key("openai", "OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set openai.api_key in credentials.yaml "
            "or export OPENAI_API_KEY."
        )
    client = OpenAI(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            u = resp.usage
            return {
                "text": text,
                "prompt_tokens": u.prompt_tokens if u else None,
                "completion_tokens": u.completion_tokens if u else None,
                "latency_sec": round(latency, 2),
            }
        except RateLimitError:
            wait = BACKOFF_BASE ** attempt
            print(f"  [OpenAI rate limit, attempt {attempt + 1}/{MAX_RETRIES}, wait {wait:.0f}s]", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"OpenAI: failed after {MAX_RETRIES} retries.")


def _call_gemini(model: str, user_prompt: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    api_key = _get_key("gemini", "GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set gemini.api_key in credentials.yaml "
            "or export GEMINI_API_KEY."
        )
    client = genai.Client(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            latency = time.perf_counter() - t0
            text = (resp.text or "").strip()
            um = getattr(resp, "usage_metadata", None)
            return {
                "text": text,
                "prompt_tokens": getattr(um, "prompt_token_count", None) if um else None,
                "completion_tokens": getattr(um, "candidates_token_count", None) if um else None,
                "latency_sec": round(latency, 2),
            }
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = BACKOFF_BASE ** attempt
                print(f"  [Gemini rate limit, attempt {attempt + 1}/{MAX_RETRIES}, wait {wait:.0f}s]", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini: failed after {MAX_RETRIES} retries.")


def _call_claude(model: str, user_prompt: str) -> dict[str, Any]:
    import anthropic

    api_key = _get_key("claude", "ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. Set claude.api_key in credentials.yaml "
            "or export ANTHROPIC_API_KEY."
        )
    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            latency = time.perf_counter() - t0
            text = (resp.content[0].text if resp.content else "").strip()
            u = resp.usage
            return {
                "text": text,
                "prompt_tokens": getattr(u, "input_tokens", None) if u else None,
                "completion_tokens": getattr(u, "output_tokens", None) if u else None,
                "latency_sec": round(latency, 2),
            }
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or "overloaded" in str(e).lower():
                wait = BACKOFF_BASE ** attempt
                print(f"  [Claude rate limit, attempt {attempt + 1}/{MAX_RETRIES}, wait {wait:.0f}s]", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Claude: failed after {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Perplexity backend (search + generation in one call)
# ---------------------------------------------------------------------------

def _call_perplexity(model: str, question: str) -> dict[str, Any]:
    """
    Single API call to Perplexity: it searches the web and generates an answer.
    Returns same shape as _call_openai / _call_gemini, plus citations.
    """
    from openai import OpenAI, RateLimitError

    api_key = _get_key("perplexity", "PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Perplexity API key not found. Set perplexity.api_key in credentials.yaml "
            "or export PERPLEXITY_API_KEY."
        )
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    user_msg = (
        f"**LIST QUESTION:** {question}\n\n"
        "This question expects MULTIPLE entities as answers (typically 5-20+), not just one. "
        "Do NOT settle for a single answer — list ALL entities (people, films, places, etc.) "
        "that answer this question.\n\n"
        "For each entity, provide a BRIEF 2-3 sentence explanation with supporting context "
        "(year, genre, co-collaborators). Keep explanations concise — prioritize listing "
        "MORE entities over detailed explanations for each."
    )

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            u = resp.usage
            # Perplexity returns citations in the response object
            citations = getattr(resp, "citations", None) or []
            return {
                "text": text,
                "prompt_tokens": u.prompt_tokens if u else None,
                "completion_tokens": u.completion_tokens if u else None,
                "latency_sec": round(latency, 2),
                "citations": citations,
            }
        except RateLimitError:
            wait = BACKOFF_BASE ** attempt
            print(f"  [Perplexity rate limit, attempt {attempt + 1}/{MAX_RETRIES}, wait {wait:.0f}s]", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Perplexity: failed after {MAX_RETRIES} retries.")


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------

def process_question(
    record: dict,
    backend: str,
    model: str,
    num_passages: int = _DEFAULT_NUM_PASSAGES,
) -> dict[str, Any]:
    """
    Run the full RAG pipeline for one question record.
    Returns a result dict ready for JSON serialisation.
    """
    qid = record.get("qid", record.get("question_id", "unknown"))
    question = record.get("question_text", "")

    result: dict[str, Any] = {
        "qid": qid,
        "question_text": question,
        "answer_list": record.get("answer_list", {}),
        "backend": backend,
        "model": model,
    }

    if backend.startswith("wikipedia+"):
        gen_provider = backend.split("+", 1)[1]   # "openai" or "gemini"
        result["num_passages"] = num_passages

        # 1. Retrieve
        passages = retrieve_wikipedia(question, max_passages=num_passages)
        result["retrieved_passages"] = [
            {"title": p["title"], "text": p["text"][:300] + "…" if len(p["text"]) > 300 else p["text"]}
            for p in passages
        ]

        if not passages:
            print("  [No Wikipedia passages retrieved — falling back to parametric knowledge]", flush=True)
            result["retrieval_used"] = False
        else:
            result["retrieval_used"] = True

        # 2. Generate (passes empty list → parametric fallback prompt when retrieval failed)
        user_prompt = _build_user_prompt(question, passages)
        if gen_provider == "openai":
            gen = _call_openai(model, user_prompt)
        elif gen_provider == "gemini":
            gen = _call_gemini(model, user_prompt)
        elif gen_provider == "claude":
            gen = _call_claude(model, user_prompt)
        else:
            raise ValueError(f"Unknown generation provider in backend: {gen_provider}")

        result["rag_answer"] = gen["text"]
        result["usage"] = {
            k: gen[k] for k in ("prompt_tokens", "completion_tokens", "latency_sec") if k in gen
        }

    elif backend == "perplexity":
        gen = _call_perplexity(model, question)
        result["rag_answer"] = gen["text"]
        result["citations"] = gen.get("citations", [])
        result["usage"] = {
            k: gen[k] for k in ("prompt_tokens", "completion_tokens", "latency_sec") if k in gen
        }

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_BACKEND_MODEL_DEFAULTS = {
    "wikipedia+openai": "gpt-4o-mini",
    "wikipedia+gemini": "gemini-2.0-flash",
    "wikipedia+claude": "claude-haiku-4-5-20251001",
    "perplexity": "sonar",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate RAG-based answers for Qampari QA dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (3 questions, cheapest option)
  python3 run_rag_answers.py --limit 3 --backend wikipedia+openai --model gpt-4o-mini

  # More Wikipedia context (folder/file names use RAGW8G)
  python3 run_rag_answers.py --num-passages 8 --limit 3 --backend wikipedia+openai

  # Full run with free Gemini tier
  python3 run_rag_answers.py --backend wikipedia+gemini

  # Full run with Claude (folder names include RAGW4C)
  python3 run_rag_answers.py --backend wikipedia+claude --model claude-opus-4-5

  # Perplexity (search + generation combined)
  python3 run_rag_answers.py --limit 10 --backend perplexity

  # Custom input file (JSON or pickle)
  python3 run_rag_answers.py --input data/reproducibility_Qampari.json --limit 5

  # Skip first 200 questions (1-based index); continue from #201
  python3 run_rag_answers.py --start-from 201 --backend wikipedia+openai
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        dest="input_file",
        help=f"Path to input JSON or pickle (default: {_DEFAULT_INPUT.relative_to(_EXPERIMENTS)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process at most N questions after --start-from (if set). "
            "Smoke test: e.g. --start-from 10 --limit 2 = two questions starting at row 10."
        ),
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="I",
        help=(
            "1-based index of the first question in the full loaded file. Applied before "
            "--limit. Default 1. Use to skip earlier rows without --resume."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["wikipedia+openai", "wikipedia+gemini", "wikipedia+claude", "perplexity"],
        default="wikipedia+openai",
        help="Retrieval+generation backend (default: wikipedia+openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use. Defaults: gpt-4o-mini / gemini-2.0-flash / claude-haiku-4-5-20251001 / sonar",
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=_DEFAULT_NUM_PASSAGES,
        metavar="N",
        help=(
            "For Wikipedia backends: maximum number of passages to retrieve and feed "
            f"to the model (default: {_DEFAULT_NUM_PASSAGES}). Ignored for perplexity."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: results/rag_answers/<RAG tag>/<timestamp>/<dataset>_rag_answers.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output file exists, skip already-processed qids and append new results",
    )
    args = parser.parse_args()

    if args.backend.startswith("wikipedia+"):
        if args.num_passages < 1 or args.num_passages > 50:
            print(
                "ERROR: --num-passages must be between 1 and 50 for Wikipedia backends.",
                file=sys.stderr,
            )
            return 1

    # ---- Resolve input file ----
    input_path = args.input_file or _DEFAULT_INPUT
    if not input_path.is_absolute():
        input_path = _EXPERIMENTS / input_path
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    # ---- Resolve model ----
    model = args.model
    if not model:
        provider_key = args.backend.split("+")[-1]   # openai / gemini / perplexity
        model = _get_default_model(provider_key) or _BACKEND_MODEL_DEFAULTS[args.backend]

    # ---- Derive dataset key: {BaseDataset}_{GENERATION_TYPE} ----
    # Strip known prefixes/suffixes so the base name is always e.g. "Qampari"
    base_name = input_path.stem
    for prefix in ("reproducibility_", "toy_subset_"):
        base_name = base_name.replace(prefix, "")
    for tag in ("_PARAM", "_RAGW4G", "_RAGP", "_HUMAN"):
        base_name = base_name.replace(tag, "")
    # Strip dynamic tags like Qampari_RAGW8G, Qampari_RAGW4M
    base_name = re.sub(r"_RAGW\d+[GM]$", "", base_name)
    generation_type = _generation_type_tag(args.backend, args.num_passages)
    dataset_key = f"{base_name}_{generation_type}"   # e.g. "Qampari_RAGW4G"

    # ---- Resolve output path ----
    if args.output:
        out_path = args.output
        if not out_path.is_absolute():
            out_path = _HERE / out_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _RESULTS_ROOT / generation_type / ts
        out_path = out_dir / f"{dataset_key}_rag_answers.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load input (JSON or pickle) ----
    if input_path.suffix == ".json":
        import json as _json
        with open(input_path, encoding="utf-8") as f:
            records: list[dict] = _json.load(f)
    else:
        with open(input_path, "rb") as f:
            records: list[dict] = pickle.load(f)
    if not isinstance(records, list):
        print(f"ERROR: expected list, got {type(records)}", file=sys.stderr)
        return 1

    n_loaded = len(records)
    if args.start_from < 1:
        print("ERROR: --start-from must be >= 1", file=sys.stderr)
        return 1
    if args.start_from > n_loaded:
        print(
            f"ERROR: --start-from ({args.start_from}) is past end of list ({n_loaded} questions)",
            file=sys.stderr,
        )
        return 1

    start_idx = args.start_from - 1
    records = records[start_idx:]
    if args.limit is not None:
        if args.limit < 1:
            print("ERROR: --limit must be >= 1 when set", file=sys.stderr)
            return 1
        records = records[: args.limit]

    total = len(records)
    if total == 0:
        print("ERROR: no questions to process (check --start-from and --limit)", file=sys.stderr)
        return 1

    total_full = n_loaded
    print(f"Loaded {n_loaded} questions from {input_path.name}", end="")
    if start_idx or args.limit is not None:
        end_idx = args.start_from + total - 1
        print(
            f" (this run: {total} question(s), indices {args.start_from}–{end_idx} of {n_loaded})",
            end="",
        )
    print()
    print(f"Backend: {args.backend} | Model: {model}")
    if args.backend.startswith("wikipedia+"):
        print(f"Wikipedia passages (max): {args.num_passages}")
    print(f"Output:  {out_path}\n")

    # ---- Resume: load already-done qids ----
    done_qids: set[str] = set()
    existing_results: list[dict] = []
    if args.resume and out_path.is_file():
        with open(out_path) as f:
            existing_results = json.load(f)
        done_qids = {r["qid"] for r in existing_results}
        print(f"Resuming: {len(done_qids)} questions already done, skipping them.\n")

    # ---- Process ----
    results = list(existing_results)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = 0

    for offset, record in enumerate(records):
        i = start_idx + offset + 1
        qid = record.get("qid", record.get("question_id", f"unknown_{i}"))
        if qid in done_qids:
            continue

        question_short = record.get("question_text", "")[:80]
        print(f"[{i:3d}/{total_full}] qid={qid} | {question_short}", flush=True)

        try:
            result = process_question(
                record, args.backend, model, num_passages=args.num_passages
            )
            result["generation_type"] = generation_type  # e.g. "RAGW4G", "RAGW8M", "RAGP"
            results.append(result)

            usage = result.get("usage", {})
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            total_prompt_tokens += pt
            total_completion_tokens += ct
            latency = usage.get("latency_sec", 0)

            n_passages = len(result.get("retrieved_passages", []))
            preview = (result.get("rag_answer") or "")[:120].replace("\n", " ")
            if args.backend.startswith("wikipedia"):
                print(f"         passages={n_passages} | tokens={pt}+{ct} | {latency:.1f}s")
            else:
                cits = len(result.get("citations", []))
                print(f"         citations={cits} | tokens={pt}+{ct} | {latency:.1f}s")
            print(f"         answer: {preview}…\n", flush=True)

        except Exception as e:
            print(f"  ERROR on qid={qid}: {e}", flush=True)
            results.append({"qid": qid, "question_text": record.get("question_text", ""), "error": str(e)})
            errors += 1

        # Save incrementally after each question (safe against interruption)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("=" * 60)
    print(f"Done. {len(results) - errors} succeeded, {errors} errors.")
    print(f"Total tokens used: {total_prompt_tokens} prompt + {total_completion_tokens} completion")
    print(f"Results saved to: {out_path}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
