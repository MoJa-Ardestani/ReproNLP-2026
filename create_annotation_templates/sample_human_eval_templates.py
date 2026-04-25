"""Sample from generated human-eval templates by stratified QID selection."""

import argparse
import json
import random
import re
from pathlib import Path


def _is_simple(qid: str) -> bool:
    return "simple" in qid.lower()


def _is_complex(qid: str) -> bool:
    qid_lower = qid.lower()
    return any(p in qid_lower for p in ("composition", "comp", "intersection", "inter"))


def sample_qids(
    all_qids: list[str],
    total: int,
    simple_ratio: float,
    seed: int = 42,
) -> list[str]:
    simple_pool = [q for q in all_qids if _is_simple(q)]
    complex_pool = [q for q in all_qids if _is_complex(q)]

    n_simple = round(total * simple_ratio)
    n_complex = total - n_simple

    if n_simple > len(simple_pool):
        raise ValueError(f"Requested {n_simple} simple QIDs but only {len(simple_pool)} available")
    if n_complex > len(complex_pool):
        raise ValueError(f"Requested {n_complex} complex QIDs but only {len(complex_pool)} available")

    rng = random.Random(seed)
    simple_sample = rng.sample(simple_pool, n_simple)
    complex_sample = rng.sample(complex_pool, n_complex)

    combined = simple_sample + complex_sample
    rng.shuffle(combined)
    return combined


def filter_template(template_path: Path, sampled_qids: set[str]) -> dict:
    with open(template_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_tasks = [t for t in data["tasks"] if t["qid"] in sampled_qids]
    data["tasks"] = filtered_tasks
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Sample nuggetization templates using stratified QID selection"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment directory, e.g. ../results/reproducibility/Qampari_RAGW4C/20260421_114136",
    )
    parser.add_argument("total", type=int, help="Total number of QIDs to sample")
    parser.add_argument("simple_ratio", type=float, help="Fraction of simple QIDs (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: results/human_labels/{dataset}/templates/)",
    )

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    dataset_tag = exp_dir.parent.name  # e.g. "Qampari_RAGW4C"

    manual_nug_path = exp_dir / f"{dataset_tag}_manual_nuggetization.json"
    edited_nug_path = exp_dir / f"{dataset_tag}_edited_nuggetization.json"

    for p in (manual_nug_path, edited_nug_path):
        if not p.exists():
            raise FileNotFoundError(f"Template not found: {p}")

    with open(manual_nug_path, "r", encoding="utf-8") as f:
        nug_data = json.load(f)

    all_qids = [t["qid"] for t in nug_data["tasks"]]

    sampled = sample_qids(all_qids, args.total, args.simple_ratio, args.seed)
    sampled_set = set(sampled)

    n_simple = sum(1 for q in sampled if _is_simple(q))
    n_complex = len(sampled) - n_simple
    print(f"Sampled {n_simple} simple + {n_complex} complex = {len(sampled)} total QIDs")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dataset = re.sub(r"_RAGW[A-Za-z0-9]+", "", dataset_tag)  # e.g. "Qampari"
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "results" / "human_labels" / base_dataset / "templates"

    output_dir.mkdir(parents=True, exist_ok=True)

    for src_path in (manual_nug_path, edited_nug_path):
        filtered = filter_template(src_path, sampled_set)
        stem = re.sub(r"_RAGW[A-Za-z0-9]+", "", src_path.stem) + "_template"
        out_path = output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")

    qid_list_path = output_dir / "sampled_qids.txt"
    qid_list_path.write_text("\n".join(sampled), encoding="utf-8")
    print(f"Saved QID list: {qid_list_path}")


if __name__ == "__main__":
    main()
