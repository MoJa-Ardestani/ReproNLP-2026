"""Reproducibility dataset registry and data-source manifest for run_config."""

from autonuggetizer.reproducibility_paths import pickle_fingerprint

DATASETS = {
    "Qampari_PARAM": {
        "dataset": "Qampari",
        "generation_type": "PARAM",
        "raw_key": "gpt-4o_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW4G": {
        "dataset": "Qampari",
        "generation_type": "RAGW4G",
        "raw_key": "wikipedia_gpt-4o_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW8G": {
        "dataset": "Qampari",
        "generation_type": "RAGW8G",
        "raw_key": "wikipedia_gpt-4o_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW12G": {
        "dataset": "Qampari",
        "generation_type": "RAGW12G",
        "raw_key": "wikipedia_gpt-4o_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW4M": {
        "dataset": "Qampari",
        "generation_type": "RAGW4M",
        "raw_key": "wikipedia_gemini-2.5-flash_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW8M": {
        "dataset": "Qampari",
        "generation_type": "RAGW8M",
        "raw_key": "wikipedia_gemini-2.5-flash_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW4C": {
        "dataset": "Qampari",
        "generation_type": "RAGW4C",
        "raw_key": "wikipedia_claude-sonnet-4-5_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
    "Qampari_RAGW8C": {
        "dataset": "Qampari",
        "generation_type": "RAGW8C",
        "raw_key": "wikipedia_claude-sonnet-4-5_self_contained_paragraph_answer_raw",
        "type": "split_span",
    },
}


def dataset_data_relpath(cfg: dict) -> str:
    """Path relative to the project root: canonical JSON next to autonuggetizer package."""
    return f"data/reproducibility_{cfg['dataset']}_{cfg['generation_type']}.json"


def data_sources_manifest() -> dict:
    out = {}
    for name, cfg in DATASETS.items():
        data_path = dataset_data_relpath(cfg)
        out[name] = {
            "data_file": data_path,
            "data_file_info": pickle_fingerprint(data_path),
            "generation_type": cfg["generation_type"],
            "raw_key": cfg["raw_key"],
            "context_type": cfg["type"],
        }
    return out
