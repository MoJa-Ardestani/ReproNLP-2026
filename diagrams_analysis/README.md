# diagrams_analysis

Two scripts that produce all figures and QRA++ Type I results for the report.

## Directory layout

```
diagrams_analysis/
├── figures/
│   ├── make_report_figures.py      # generates Figures 4–6 and Kendall τ JSON
│   └── reproduced_kendal_values.json  # written by the script above
└── RQA++/
    ├── qra_plus_plus.py            # QRA++ Type I (CV*) assessment
    ├── original_qra_spec.json      # original paper's Kendall τ values (input)
    ├── reproduced_qra.json         # reproduced Kendall τ values (input)
    └── outputs/qra_results/        # CSV outputs written by qra_plus_plus.py
```

## Prerequisites

```bash
pip install matplotlib numpy scipy
```

## Step 1 — Generate report figures

Run from inside `figures/`:

```bash
cd diagrams_analysis/figures
python make_report_figures.py
```

**Reads** (relative to the project root):
- `results/reproducibility/Qampari_*/ID_1/` — auto-assignment result JSONs
- `results/human_labels/Qampari/assignments/Qampari_*/` — human-assignment result JSONs

**Writes** (into `figures/`):
- `figure4_combined.png`, `figure4_vstrict.png`, `figure4_astrict.png`
- `figure5_combined.png`, `figure5_vstrict.png`, `figure5_astrict.png`
- `figure6_confusion.png`
- `reproduced_kendal_values.json` — Kendall τ values needed by Step 2

## Step 2 — Run QRA++ Type I assessment

Run from inside `RQA++/`:

```bash
cd diagrams_analysis/RQA++
python qra_plus_plus.py \
    --original-spec original_qra_spec.json \
    --reproduced    reproduced_qra.json \
    --output-dir    outputs/qra_results
```

`reproduced_qra.json` must exist before this step (it is produced in Step 1 and copied/placed here manually if needed).

**Writes** (into `outputs/qra_results/`):
- `qra_type_i_by_qc.csv` — per-comparison-unit CV* for all 24 pairs
- `qra_type_i_summary.csv` — mean CV* per QC block (6 rows)
