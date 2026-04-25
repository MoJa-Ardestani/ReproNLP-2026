"""Load reproducibility JSON/pickle and build pipeline-ready samples."""

from __future__ import annotations

import json
import pickle
from pathlib import Path


def load_pickle(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return load_pickle(str(path))


def length_fields_from_item(item: dict) -> dict:
    return {
        "gt_len": item.get("gt_len"),
        "gen_len": item.get("gen_len"),
        "total": item.get("total"),
        "length_tag": item.get("length_tag"),
    }


def extract_context_segments(answer_list: dict, ds_type: str) -> list[str]:
    segments = []
    for entity_key, proofs in answer_list.items():
        entity_name = entity_key.strip("[]'\"")
        if not isinstance(proofs, list):
            proofs = [proofs]
        if ds_type == "split_span" and entity_name:
            combined_proofs = " | ".join(p.strip() for p in proofs if p.strip())
            segments.append(f"{entity_name}: {combined_proofs}")
        else:
            for proof in proofs:
                segments.append(proof)
    return segments


def prepare_samples(data: list[dict], cfg: dict) -> list[dict]:
    samples = []
    for item in data:
        question = item.get("question_text", "")
        passage = item.get(cfg["raw_key"], "")
        answer_list = item.get("answer_list", {})

        if not question or not passage.strip() or not answer_list:
            continue

        context_segments = extract_context_segments(answer_list, cfg["type"])
        if not context_segments:
            continue

        qid = item["qid"]

        samples.append({
            "qid": qid,
            "question": question,
            "context_segments": context_segments,
            "passage": passage,
            "_answer_list": answer_list,
            "_raw_key": cfg["raw_key"],
            "_length_stats": length_fields_from_item(item),
        })
    return samples
