# ReproNLP 2026 — Experiments

**Paper:** *"Do AutoNuggetizer's Agreement Patterns Generalize to List-QA?"*


---

## Quick start

```bash
# 1. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp credentials.yaml.example credentials.yaml   # or set env vars directly

# 3. Run the full pipeline on a dataset
python3 main_run_reproducibility.py \
    --dataset <DATASET_KEY> \
    --provider <PROVIDER>   # openai | gemini | claude
```

See [`autonuggetizer/README.md`](autonuggetizer/README.md) for the complete CLI reference.

---

## Repository layout

```
<project_root>/
├── main_run_reproducibility.py   # top-level entry point (delegates to autonuggetizer/)
├── config.yaml                   # default LLM / path settings
├── credentials.yaml              # API keys (git-ignored)
├── requirements.txt              # all Python dependencies
│
├── data/                         # input data files (one JSON per system condition)
├── autonuggetizer/               # core 3-stage pipeline + CLI
├── RAG_answer/                   # RAG answer generation scripts
├── create_annotation_templates/  # human-annotation template builders
├── results/                      # all outputs (auto + human labels)
└── diagrams_analysis/            # figures and QRA++ reproducibility assessment
```

> Each subfolder has its **own `README.md`** with detailed commands and argument
> references.  The descriptions below are summaries only.

---

## Folder descriptions

### `data/`

Static JSON input files — one file per system condition:

```
data/reproducibility_<DATASET>.json             # base dataset (no RAG)
data/reproducibility_<DATASET>_<RAG_TAG>.json   # RAG variant
```

These are the inputs consumed by the pipeline and the annotation template
scripts.  No `README` lives here because the files are produced by
[`RAG_answer/`](RAG_answer/README.md) and are otherwise read-only inputs.

---

### `autonuggetizer/`  → [`README`](autonuggetizer/README.md)

Core evaluation pipeline: a re-implementation of the AutoNuggetizer framework
(Pradeep et al., SIGIR 2025).

**Three-stage pipeline:**

1. **Nugget Creation** — extract atomic facts from reference passages
2. **Nugget Scoring** — label each nugget `vital` / `okay`
3. **Nugget Assignment** — determine whether each nugget is
   `support` / `partial_support` / `not_support` in the system answer

Entry point (run from the project root):

```bash
python3 main_run_reproducibility.py \
    --dataset <DATASET_KEY> \
    --provider <PROVIDER> \
    [--limit N] [--dry-run] [--verbose]
```

Outputs land in `results/reproducibility/<DATASET_KEY>/<YYYYMMDD_HHMMSS>/`.

---

### `RAG_answer/`  → [`README`](RAG_answer/README.md)

Scripts that call a retrieval-augmented generation backend to produce the
system answers that are later evaluated by the pipeline.

```bash
# Step 1 — generate answers
cd RAG_answer
python3 run_rag_answers.py \
    --backend <BACKEND>       \  # wikipedia+openai | wikipedia+gemini | wikipedia+claude
    --num-passages <N>        \
    --input  data/reproducibility_<DATASET>.json

# Step 2 — convert to reproducibility JSON consumed by autonuggetizer
python3 json_to_pickle.py \
    --generation-type <RAG_TAG> \
    --run-id <TIMESTAMP>
```

Outputs land in `results/rag_answers/<RAG_TAG>/<TIMESTAMP>/` and
`data/reproducibility_<DATASET>_<RAG_TAG>.json`.

---

### `create_annotation_templates/`  → [`README`](create_annotation_templates/README.md)

Builds the blank JSON templates that human annotators fill in.  Two
independent steps:

```bash
cd create_annotation_templates

# Step 1 — nuggetization template (annotator writes / edits nuggets)
python3 create_nuggetization_templates.py \
    --dataset <DATASET_KEY> \
    --total <N_QUESTIONS> \
    --simple-ratio <RATIO>

# Step 2 — assignment template (annotator labels each nugget in a system answer)
python3 create_assignment_templates.py \
    --nuggetization templates/<BASE>/<BASE>_<NUGGET_TYPE>_nuggetization_template.json \
    --dataset <DATASET_KEY>
```

Outputs land in `create_annotation_templates/templates/<BASE>/<DATASET_KEY>/`.

---

### `results/`

All pipeline and human-annotation outputs.  Two sub-trees:

```
results/
├── reproducibility/                            # auto-pipeline outputs
│   └── <DATASET_KEY>/
│       └── <YYYYMMDD_HHMMSS>/
│           ├── <DATASET_KEY>_auto_nuggetization_auto_assignment.json
│           └── run_config.json
│
└── human_labels/                               # completed human annotations
    └── <BASE_DATASET>/
        ├── nuggets/
        │   ├── <BASE_DATASET>_auto_nuggetization.json
        │   ├── <BASE_DATASET>_edited_nuggetization.json
        │   └── <BASE_DATASET>_manual_nuggetization.json
        └── assignments/
            └── <DATASET_KEY>/
                ├── <DATASET_KEY>_edited_nuggetization_human_assignment.json
                └── <DATASET_KEY>_manual_nuggetization_human_assignment.json
```

There is no separate README here; file naming conventions are documented in
[`autonuggetizer/README.md`](autonuggetizer/README.md).

---

### `diagrams_analysis/`  → [`README`](diagrams_analysis/README.md)

Generates all report figures and runs the QRA++ Type I reproducibility
assessment.

```bash
# Step 1 — report figures (Figures 4–6) + Kendall τ JSON
cd diagrams_analysis/figures
python make_report_figures.py

# Step 2 — QRA++ Type I assessment (CV* per quality criterion)
cd diagrams_analysis/RQA++
python qra_plus_plus.py \
    --original-spec original_qra_spec.json \
    --reproduced    reproduced_qra.json \
    --output-dir    outputs/qra_results
```

---

## Credentials

API keys are **never committed**.  Provide them in either of two ways:

**Option A — `credentials.yaml`** (git-ignored):

```yaml
openai_api_key:     "sk-..."
gemini_api_key:     "AIza..."
anthropic_api_key:  "sk-ant-..."
```

**Option B — environment variables**:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

