#!/usr/bin/env python3
"""
QRA++ Type I assessment for the Qampari reproducibility study.

Reads:
  - original_qra_spec.json  : original paper's Kendall τ values
  - reproduced_qra.json     : reproduction's Kendall τ values

Writes:
  - outputs/qra_results/qra_type_i_by_qc.csv    : per-unit CV* for all 24 pairs
  - outputs/qra_results/qra_type_i_summary.csv  : mean CV* per QC block (6 rows)

Usage:
  python3 qra_plus_plus.py \
    --original-spec original_qra_spec.json \
    --reproduced    reproduced_qra.json \
    --output-dir    outputs/qra_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


QC_ORDER = [
    ("Vstrict", "run"),
    ("Astrict", "run"),
    ("Vstrict", "topic_avg"),
    ("Astrict", "topic_avg"),
    ("Vstrict", "all_topic_run"),
    ("Astrict", "all_topic_run"),
]

COMPARISON_UNITS = {
    "C1": ("AA", "E_M"),
    "C2": ("AA", "M_M"),
    "C3": ("E_A", "E_M"),
    "C4": ("M_A", "M_M"),
}


def cv_star_belz(set_of_measurements: list[float]) -> float:
    """
    Small-sample adjusted CV* (Belz 2022; github.com/asbelz/coeff-var).

    For n=2: unbiased s* via c4 correction, then CV* = (1 + 1/(4n)) * (s*/mean) * 100.
    Returns NaN if n < 2, mean <= 0, or any input is non-finite.
    """
    if len(set_of_measurements) < 2:
        return float("nan")
    if not all(math.isfinite(x) for x in set_of_measurements):
        return float("nan")
    n = len(set_of_measurements)
    mean = statistics.fmean(set_of_measurements)
    if mean <= 0:
        return float("nan")
    df = n - 1
    sum_sq = sum((mean - x) ** 2 for x in set_of_measurements)
    s = math.sqrt(sum_sq / df)
    c4 = math.sqrt(2.0 / df) * math.gamma(n / 2.0) / math.gamma(df / 2.0)
    if c4 == 0:
        return float("nan")
    s_star = s / c4
    cv = (s_star / mean) * 100.0
    return (1.0 + 1.0 / (4.0 * n)) * cv


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.fmean(finite) if finite else float("nan")


def _load_kendall_cube(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cube = data.get("reproduced_kendall") or data.get("original_kendall", {})
    if not isinstance(cube, dict):
        raise ValueError(f"No Kendall cube found in {path}")
    return cube


def compute_type_i(
    original_spec_path: Path,
    reproduced_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    original_kendall: dict[str, Any] = json.loads(
        original_spec_path.read_text(encoding="utf-8")
    ).get("original_kendall", {})

    repr_cube = _load_kendall_cube(reproduced_path)

    by_qc_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for metric, agg in QC_ORDER:
        qc_name = f"{metric}_{agg}"
        cv_values: list[float] = []

        for cid in ("C1", "C2", "C3", "C4"):
            orig_val = (original_kendall.get(cid) or {}).get(metric, {}).get(agg)
            repr_val = ((repr_cube.get(cid) or {}).get(metric) or {}).get(agg)

            if orig_val is None or repr_val is None:
                cv = float("nan")
            else:
                try:
                    cv = cv_star_belz([float(orig_val), float(repr_val)])
                except (TypeError, ValueError):
                    cv = float("nan")
                if math.isfinite(cv):
                    cv_values.append(cv)

            by_qc_rows.append({
                "qc": qc_name,
                "comparison_unit": cid,
                "metric": metric,
                "agg_level": agg,
                "original_value": orig_val,
                "reproduction_value": repr_val,
                "cv_star": cv if math.isfinite(cv) else "",
            })

        summary_rows.append({
            "qc": qc_name,
            "metric": metric,
            "agg_level": agg,
            "n_pairs": len(cv_values),
            "mean_cv_star": _safe_mean(cv_values) if cv_values else "",
        })

    by_qc_csv = output_dir / "qra_type_i_by_qc.csv"
    with by_qc_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(by_qc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_qc_rows)

    summary_csv = output_dir / "qra_type_i_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Written: {by_qc_csv}")
    print(f"Written: {summary_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QRA++ Type I (CV*) assessment.")
    parser.add_argument(
        "--original-spec",
        default="original_qra_spec.json",
        help="Path to original paper Kendall τ values JSON.",
    )
    parser.add_argument(
        "--reproduced",
        default="reproduced_qra.json",
        help="Path to reproduced Kendall τ values JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qra_results",
        help="Directory for output CSVs.",
    )
    args = parser.parse_args()
    compute_type_i(
        original_spec_path=Path(args.original_spec),
        reproduced_path=Path(args.reproduced),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
