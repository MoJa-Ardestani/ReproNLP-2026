#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau


_HERE = Path(__file__).resolve().parent          # <project_root>/diagrams_analysis/figures/
ROOT  = _HERE.parents[1]                          # <project_root>/

REPRO_ROOT  = ROOT / "results" / "reproducibility"
HUMAN_ROOT  = ROOT / "results" / "human_labels" / "Qampari" / "assignments"
OUT         = _HERE
KENDALL_OUT = _HERE / "reproduced_kendal_values.json"
OUT.mkdir(parents=True, exist_ok=True)


METHOD_FILE = {
    "auto_auto": "auto_nuggetization_auto_assignment",
    "edited_auto": "edited_nuggetization_auto_assignment",
    "edited_human": "edited_nuggetization_human_assignment",
    "manual_auto": "manual_nuggetization_auto_assignment",
    "manual_human": "manual_nuggetization_human_assignment",
}

LABELS = ["not_support", "partial_support", "support"]


def load_condition(method_key: str, metric: str):
    suffix = METHOD_FILE[method_key]
    is_human = method_key.endswith("_human")
    base_root = HUMAN_ROOT if is_human else REPRO_ROOT
    out = {}
    for run_dir in sorted(p for p in base_root.iterdir() if p.is_dir() and p.name.startswith("Qampari_")):
        run = run_dir.name
        # human assignments live directly in run_dir; auto assignments are under ID_1/
        f = (run_dir / f"{run}_{suffix}.json") if is_human else (run_dir / "ID_1" / f"{run}_{suffix}.json")
        if not f.is_file():
            continue
        data = json.loads(f.read_text())
        mean = ((data.get("score_aggregate") or {}).get(metric) or {}).get("mean")
        if not isinstance(mean, (int, float)):
            continue
        entries = data.get("samples") or data.get("tasks") or []
        topics = {}
        for e in entries:
            qid = e.get("qid")
            if not qid:
                continue
            v = e.get(metric)
            if not isinstance(v, (int, float)):
                v = (e.get("scores") or {}).get(metric)
            if isinstance(v, (int, float)):
                topics[qid] = float(v)
        out[run] = {"run": float(mean), "topics": topics}
    return out


def compute_triplet(x_map, y_map):
    runs = sorted(set(x_map.keys()) & set(y_map.keys()))
    run_x = [x_map[r]["run"] for r in runs]
    run_y = [y_map[r]["run"] for r in runs]
    run_tau = float(kendalltau(run_x, run_y).statistic)

    flat_x = []
    flat_y = []
    by_topic = {}
    for r in runs:
        common = sorted(set(x_map[r]["topics"]) & set(y_map[r]["topics"]))
        for qid in common:
            xv = x_map[r]["topics"][qid]
            yv = y_map[r]["topics"][qid]
            flat_x.append(xv)
            flat_y.append(yv)
            by_topic.setdefault(qid, []).append((xv, yv))

    pt = []
    for qid, pairs in by_topic.items():
        if len(pairs) < 2:
            continue
        t = kendalltau([a for a, _ in pairs], [b for _, b in pairs]).statistic
        if t == t:
            pt.append(float(t))
    per_topic_avg = float(np.mean(pt)) if pt else float("nan")
    all_tau = float(kendalltau(flat_x, flat_y).statistic)
    return run_x, run_y, flat_x, flat_y, run_tau, per_topic_avg, all_tau


CONDITION_LABELS = {
    "auto_auto": "AutoNuggets / AutoAssign",
    "edited_auto": "AutoNuggets+Edits / AutoAssign",
    "edited_human": "AutoNuggets+Edits / ManualAssign",
    "manual_auto": "ManualNuggets / AutoAssign",
    "manual_human": "ManualNuggets / ManualAssign",
}

METRIC_LABELS = {
    "vstrict": "$V_{\\mathrm{strict}}$",
    "astrict": "$A_{\\mathrm{strict}}$",
}

COMPARISON_UNITS = {
    "C1": ("auto_auto", "edited_human"),
    "C2": ("auto_auto", "manual_human"),
    "C3": ("edited_auto", "edited_human"),
    "C4": ("manual_auto", "manual_human"),
}

CONDITION_SHORT = {
    "auto_auto": "AA",
    "edited_auto": "E_A",
    "edited_human": "E_M",
    "manual_auto": "M_A",
    "manual_human": "M_M",
}

METRIC_JSON_KEYS = {
    "vstrict": "Vstrict",
    "astrict": "Astrict",
}


def _draw_panel(ax, metric: str, x_key: str, y_key: str, subtitle: str):
        x_map = load_condition(x_key, metric)
        y_map = load_condition(y_key, metric)
        run_x, run_y, flat_x, flat_y, rt, pt, at = compute_triplet(x_map, y_map)

        ax.scatter(flat_x, flat_y, s=14, alpha=0.35, c="#1f77b4", label="All topic/run pairs")
        ax.scatter(run_x, run_y, s=90, marker="*", c="#d62728", edgecolors="black", linewidths=0.5, label="Run-level means")
        lo = min(min(flat_x), min(flat_y), min(run_x), min(run_y))
        hi = max(max(flat_x), max(flat_y), max(run_x), max(run_y))
        pad = 0.03
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        
        x_label = CONDITION_LABELS.get(x_key, x_key)
        y_label = CONDITION_LABELS.get(y_key, y_key)
        metric_label = METRIC_LABELS.get(metric, metric)
        ax.set_xlabel(f"{x_label} ({metric_label})", fontsize=9)
        ax.set_ylabel(f"{y_label} ({metric_label})", fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        text = f"run-level τ = {rt:.3f}\nper-topic avg τ = {pt:.3f}\nall topic/run τ = {at:.3f}"
        ax.text(
            0.98,
            0.02,
            text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#999999"),
        )
        return rt, pt, at


def write_kendall_values():
    payload = {
        "notes": "Template aligned to 4 comparison units x 2 metrics x 3 aggregation levels.",
        "type_iv_tolerance_abs": 0.1,
        "comparison_units": {
            cid: [CONDITION_SHORT[x_key], CONDITION_SHORT[y_key]]
            for cid, (x_key, y_key) in COMPARISON_UNITS.items()
        },
        "original_kendall": {},
    }

    for cid, (x_key, y_key) in COMPARISON_UNITS.items():
        payload["original_kendall"][cid] = {}
        for metric in ("vstrict", "astrict"):
            x_map = load_condition(x_key, metric)
            y_map = load_condition(y_key, metric)
            _, _, _, _, run_tau, topic_avg_tau, all_tau = compute_triplet(x_map, y_map)
            payload["original_kendall"][cid][METRIC_JSON_KEYS[metric]] = {
                "run": round(run_tau, 3),
                "topic_avg": round(topic_avg_tau, 3),
                "all_topic_run": round(all_tau, 3),
            }

    KENDALL_OUT.write_text(json.dumps(payload, indent=2))


def draw_scatter_grid(metric: str, out_name: str, comps, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0))
    for idx, (x_key, y_key, subtitle) in enumerate(comps):
        ax = axes[idx]
        _draw_panel(ax, metric, x_key, y_key, subtitle)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_combined_2x2(out_name: str, top_pair, bottom_pair, title: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    
    panels = [
        (0, 0, "vstrict", *top_pair),
        (0, 1, "astrict", *top_pair),
        (1, 0, "vstrict", *bottom_pair),
        (1, 1, "astrict", *bottom_pair),
    ]
    for r, c, metric, x_key, y_key, row_label in panels:
        ax = axes[r, c]
        metric_label = "$V_{\\mathrm{strict}}$" if metric == "vstrict" else "$A_{\\mathrm{strict}}$"
        subtitle = f"{row_label} — {metric_label}"
        _draw_panel(ax, metric, x_key, y_key, subtitle)
    
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


LABEL_ORDER = ["not_support", "partial_support", "support"]


def compute_confusion(nugget_type: str) -> np.ndarray:
    """Build a 3×3 confusion matrix (rows=auto, cols=human) from result files.

    nugget_type must be 'edited' or 'manual'.
    Iterates over all RAG-tag run dirs and pairs the auto-assignment with the
    human-assignment for the same qid, accumulating per-nugget label counts.
    """
    auto_suffix  = f"{nugget_type}_nuggetization_auto_assignment"
    human_suffix = f"{nugget_type}_nuggetization_human_assignment"
    label_idx = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
    matrix = np.zeros((3, 3), dtype=int)

    for run_dir in sorted(p for p in REPRO_ROOT.iterdir() if p.is_dir() and p.name.startswith("Qampari_")):
        tag = run_dir.name
        auto_file  = run_dir / "ID_1" / f"{tag}_{auto_suffix}.json"
        human_file = HUMAN_ROOT / tag / f"{tag}_{human_suffix}.json"
        if not auto_file.is_file() or not human_file.is_file():
            continue

        auto_data  = json.loads(auto_file.read_text())
        human_data = json.loads(human_file.read_text())

        # Index human tasks by qid for fast lookup
        human_by_qid = {t["qid"]: t["human_assignment"] for t in (human_data.get("tasks") or [])}

        for sample in (auto_data.get("samples") or []):
            qid = sample.get("qid")
            auto_labels  = sample.get("assignment") or []
            human_labels = human_by_qid.get(qid) or []
            for a_lbl, h_lbl in zip(auto_labels, human_labels):
                ai = label_idx.get(a_lbl)
                hi = label_idx.get(h_lbl)
                if ai is not None and hi is not None:
                    matrix[ai, hi] += 1

    return matrix


def draw_confusion():
    label_names = ["No Support", "Partial Support", "Support"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    for ax, nugget_type, ttl in [
        (axes[0], "edited", "AutoNuggets+Edits"),
        (axes[1], "manual", "ManualNuggets"),
    ]:
        # rows=auto, cols=human → transpose so x-axis=AutoAssign, y-axis=ManualAssign
        counts = compute_confusion(nugget_type).T
        total  = counts.sum()
        pct    = 100.0 * counts / total if total else counts.astype(float)

        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=max(55, float(np.max(pct))))

        ax.set_xticks(range(3))
        ax.set_xticklabels(label_names)
        ax.set_yticks(range(3))
        ax.set_yticklabels(label_names)

        ax.set_xlabel("AutoAssign", fontsize=11)
        ax.set_ylabel("ManualAssign", fontsize=11)
        ax.set_title(ttl, fontsize=12, fontweight="bold")

        for i in range(3):
            for j in range(3):
                val   = pct[i, j]
                count = counts[i, j]
                color = "white" if val > 20 else "black"
                ax.text(j, i, f"{val:.1f}%\n({count})", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Percentage (%)", fontsize=10, rotation=270, labelpad=15)

    fig.tight_layout()
    fig.savefig(OUT / "figure6_confusion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    write_kendall_values()
    draw_combined_2x2(
        out_name="figure4_combined.png",
        top_pair=("auto_auto", "edited_human", "AutoNuggets+Edits / ManualAssign"),
        bottom_pair=("auto_auto", "manual_human", "ManualNuggets / ManualAssign"),
        # title="Figure 4 Analogue: AutoNuggets / AutoAssign vs Manual Variants",
    )
    draw_combined_2x2(
        out_name="figure5_combined.png",
        top_pair=("edited_auto", "edited_human", "AutoNuggets+Edits"),
        bottom_pair=("manual_auto", "manual_human", "ManualNuggets"),
        # title="Figure 5 Analogue: AutoAssign vs ManualAssign (Fixed Nuggets)",
    )
    draw_scatter_grid(
        metric="vstrict",
        out_name="figure4_vstrict.png",
        comps=[
            ("auto_auto", "edited_human", "Figure 4 (top analogue): AutoNuggets/AutoAssign vs AutoNuggets+Edits/ManualAssign"),
            ("auto_auto", "manual_human", "Figure 4 (bottom analogue): AutoNuggets/AutoAssign vs ManualNuggets/ManualAssign"),
        ],
        title="Figure 4 analogue on Qampari runs (Vstrict)",
    )
    draw_scatter_grid(
        metric="astrict",
        out_name="figure4_astrict.png",
        comps=[
            ("auto_auto", "edited_human", "Figure 4 (top analogue): AutoNuggets/AutoAssign vs AutoNuggets+Edits/ManualAssign"),
            ("auto_auto", "manual_human", "Figure 4 (bottom analogue): AutoNuggets/AutoAssign vs ManualNuggets/ManualAssign"),
        ],
        title="Figure 4 analogue on Qampari runs (Astrict)",
    )
    draw_scatter_grid(
        metric="vstrict",
        out_name="figure5_vstrict.png",
        comps=[
            ("edited_auto", "edited_human", "Figure 5 (top analogue): AutoNuggets+Edits/AutoAssign vs AutoNuggets+Edits/ManualAssign"),
            ("manual_auto", "manual_human", "Figure 5 (bottom analogue): ManualNuggets/AutoAssign vs ManualNuggets/ManualAssign"),
        ],
        title="Figure 5 analogue on Qampari runs (Vstrict)",
    )
    draw_scatter_grid(
        metric="astrict",
        out_name="figure5_astrict.png",
        comps=[
            ("edited_auto", "edited_human", "Figure 5 (top analogue): AutoNuggets+Edits/AutoAssign vs AutoNuggets+Edits/ManualAssign"),
            ("manual_auto", "manual_human", "Figure 5 (bottom analogue): ManualNuggets/AutoAssign vs ManualNuggets/ManualAssign"),
        ],
        title="Figure 5 analogue on Qampari runs (Astrict)",
    )
    draw_confusion()
    print("Wrote figures to", OUT)
    print("Wrote Kendall values to", KENDALL_OUT)


if __name__ == "__main__":
    main()
