# Annotation Template Generation

Two scripts generate the human annotation templates independently of the full
pipeline.  Run both from the `create_annotation_templates/` directory.

```
create_annotation_templates/
├── create_nuggetization_templates.py   # Step 1 — nuggets
├── create_assignment_templates.py      # Step 2 — assignment
└── templates/
    └── Qampari/
        ├── Qampari_manual_nuggetization_template.json
        ├── Qampari_edited_nuggetization_template.json
        ├── sampled_qids.txt                          (only when --total is used)
        └── Qampari_{RAG_TAG}/
            ├── Qampari_{RAG_TAG}_edited_nuggetization_human_assignment.json
            └── Qampari_{RAG_TAG}_manual_nuggetization_human_assignment.json
```

---

## Step 1 — Nuggetization templates

Reads directly from `data/reproducibility_<DATASET>.json`.  The nugget set is
shared across all RAG variants so the output filename uses only the base dataset
name (`Qampari`), not the full variant tag.

### All QIDs (no sampling)

```bash
cd create_annotation_templates

python3 create_nuggetization_templates.py \
  --dataset Qampari_{RAG_TAG}
```

### Random sample (recommended for human annotation)

```bash
python3 create_nuggetization_templates.py \
  --dataset Qampari_{RAG_TAG} \
  --total 30 \
  --simple-ratio 0.4 \
  --seed 42
```

`--total 30 --simple-ratio 0.4` → 12 simple + 18 complex QIDs.

### With auto nuggets pre-filled (edited template only)

Pass the auto-nuggetization results file so annotators can *edit* LLM-generated
nuggets instead of writing them from scratch.

```bash
python3 create_nuggetization_templates.py \
  --dataset Qampari_{RAG_TAG} \
  --total 30 \
  --simple-ratio 0.4 \
  --auto-nuggets ../results/reproducibility/Qampari_{RAG_TAG}/{Experiment_ID}/Qampari_{RAG_TAG}_auto_nuggetization_auto_assignment.json
```

**Output** (`templates/Qampari/`):
- `Qampari_manual_nuggetization_template.json` — annotator creates nuggets from scratch
- `Qampari_edited_nuggetization_template.json` — annotator edits pre-filled auto nuggets
- `sampled_qids.txt` — list of sampled QIDs (only when `--total` is used)

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | Dataset key, e.g. `Qampari_{RAG_TAG}` |
| `--total` | no | all QIDs | Total number of QIDs to sample |
| `--simple-ratio` | no | `0.4` | Fraction of simple QIDs when sampling |
| `--seed` | no | `42` | Random seed |
| `--auto-nuggets` | no | — | Path to `*_auto_nuggetization_auto_assignment.json` |
| `--output-dir` | no | `templates/<base>/` | Override output directory |

---

## Step 2 — Assignment templates

Takes a **filled** nuggetization file (completed by a human annotator in Step 1)
and a dataset key to pull the generated answer passages from.  The nuggetization
QIDs and the dataset do not need to be the same variant — you can use base
`Qampari` nuggets to evaluate any RAG system's passages.

### Edited nuggetization → assignment

```bash
python3 create_assignment_templates.py \
  --nuggetization ../results/human_labels/Qampari/nuggets/Qampari_edited_nuggetization.json \
  --dataset Qampari_{RAG_TAG}
```

### Manual nuggetization → assignment

```bash
python3 create_assignment_templates.py \
  --nuggetization templates/Qampari/Qampari_manual_nuggetization_template.json \
  --dataset Qampari_{RAG_TAG}
```

### Evaluate multiple RAG systems from the same nuggets

```bash
for DATASET in Qampari_{RAG_TAG} Qampari_RAGW8C Qampari_RAGW4M Qampari_RAGW8M Qampari_RAGW12G; do
  python3 create_assignment_templates.py \
    --nuggetization templates/Qampari/Qampari_edited_nuggetization_template.json \
    --dataset $DATASET
done
```

**Output** (`templates/Qampari/<DATASET>/`):
- `<DATASET>_edited_nuggetization_human_assignment.json`
- `<DATASET>_manual_nuggetization_human_assignment.json`

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--nuggetization` | yes | — | Path to filled nuggetization JSON |
| `--dataset` | yes | — | Dataset key whose passages to use |
| `--output-dir` | no | `templates/<base>/<dataset>/` | Override output directory |

---

## Available datasets

| Key | Passage model |
|---|---|
| `Qampari_PARAM` | GPT-4o (parametric) |
| `Qampari_RAGW4G` | GPT-4o + Wikipedia RAG |
| `Qampari_RAGW8G` | GPT-4o + Wikipedia RAG (8 docs) |
| `Qampari_RAGW12G` | GPT-4o + Wikipedia RAG (12 docs) |
| `Qampari_RAGW4M` | Gemini 2.5 Flash + Wikipedia RAG (4 docs) |
| `Qampari_RAGW8M` | Gemini 2.5 Flash + Wikipedia RAG (8 docs) |
| `Qampari_{RAG_TAG}` | Claude Sonnet 4.5 + Wikipedia RAG (4 docs) |
| `Qampari_RAGW8C` | Claude Sonnet 4.5 + Wikipedia RAG (8 docs) |
