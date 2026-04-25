# AutoNuggetizer

Re-implementation of the AutoNuggetizer framework from:
> Pradeep et al. (2025). "The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models." SIGIR 2025. DOI: 10.1145/3726302.3730090

The package runs a three-stage pipeline over a dataset of questions + generated answers:

1. **Nugget Creation** — extract atomic facts from reference context segments
2. **Nugget Scoring** — label each nugget as `vital` or `okay`
3. **Nugget Assignment** — check whether each nugget is `support` / `partial_support` / `not_support` in the generated answer

---

## Prerequisites

**Working directory for all commands below:** `experiments/`

```bash
cd experiments
```

**Credentials** — set in `experiments/credentials.yaml` or as environment variables:

| Provider | Env var |
|----------|---------|
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Claude | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| Qwen | `DASHSCOPE_API_KEY` |

**Data files** — each dataset needs a JSON file under `experiments/data/`:

```
data/reproducibility_Qampari_PARAM.json
data/reproducibility_Qampari_RAGWG.json
data/reproducibility_Qampari_RAGW8G.json
... (one file per generation type)
```

---

## Registered datasets

| Key | Base | Generation |
|-----|------|------------|
| `Qampari_PARAM` | Qampari | PARAM (GPT-4o, no retrieval) |
| `Qampari_RAGWG` | Qampari | RAG Wikipedia + GPT-4o (4 passages) |
| `Qampari_RAGW8G` | Qampari | RAG Wikipedia + GPT-4o (8 passages) |
| `Qampari_RAGW12G` | Qampari | RAG Wikipedia + GPT-4o (12 passages) |
| `Qampari_RAGW4M` | Qampari | RAG Wikipedia + Gemini-2.5-flash (4 passages) |
| `Qampari_RAGW8M` | Qampari | RAG Wikipedia + Gemini-2.5-flash (8 passages) |
| `Qampari_RAGW4C` | Qampari | RAG Wikipedia + Claude-sonnet (4 passages) |
| `Qampari_RAGW8C` | Qampari | RAG Wikipedia + Claude-sonnet (8 passages) |

To add a new entry, see [**Adding a new dataset / generation type**](#adding-a-new-dataset--generation-type) below.

---

## Commands

### Smoke test — validate data without calling an LLM

```bash
python3 main_run_reproducibility.py --dataset Qampari_PARAM --provider openai --limit 2 --dry-run
```

### Full 3-stage pipeline (stages 1 + 2 + 3)

Run one dataset:

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW8G --provider gemini
```

Run all registered datasets:

```bash
python3 main_run_reproducibility.py --dataset all --provider gemini
```

Specify a non-default model:

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW8G --provider openai --model gpt-4o-mini
```

Process only the first N samples (useful for testing):

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW8G --provider gemini --limit 10
```

Save per-sample trace files with all intermediate data:

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW8G --provider gemini --verbose
```

Resume a previous run from sample index N:

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW8G --provider gemini \
  --resume-experiment 20260417_203540 --start-from 25
```

**Output** lands in:
```
experiments/results/reproducibility/<Dataset_KEY>/<YYYYMMDD_HHMMSS>/
  <Dataset_KEY>_auto_nuggetization_auto_assignment.json
  run_config.json
  traces/   (only with --verbose)
```

---

### Assignment only — Stage 3 on pre-existing nuggets

Use this when nuggets already exist (from a full run, or from human annotation) and you only want to re-run the assignment step.


#### From a human nuggetization file

```bash
python3 main_run_reproducibility.py \
  --dataset Qampari_RAGW8G \
  --provider openai \
  --assignment-only-from-nuggetization \
  results/human_labels/Qampari/nuggets/Qampari_manual_nuggetization.json
```

Other nugget sources (same pattern):
- `Qampari_edited_nuggetization.json` — human-edited auto nuggets
- `Qampari_auto_nuggetization.json` — auto nuggets 

#### Point to a directory + specify the filename inside it

```bash
python3 main_run_reproducibility.py \
  --dataset Qampari_RAGW8G \
  --provider gemini \
  --assignment-only-from-nuggetization results/human_labels/Qampari/nuggets/ \
  --nuggetization-json Qampari_edited_nuggetization.json
```

#### Custom output path

```bash
python3 main_run_reproducibility.py \
  --dataset Qampari_RAGW8G \
  --provider gemini \
  --assignment-only-from-nuggetization results/human_labels/Qampari/nuggets/Qampari_manual_nuggetization.json \
  --assignment-output results/my_run/Qampari_RAGW8G_manual_nuggetization_auto_assignment.json
```

#### Skip already-processed QIDs (append-only update)

Use `--skip-done` together with `--assignment-output` pointing at the existing file to avoid re-running QIDs that are already present:

```bash
python3 main_run_reproducibility.py \
  --dataset Qampari_RAGW8G \
  --provider gemini \
  --assignment-only-from-nuggetization results/human_labels/Qampari/nuggets/Qampari_manual_nuggetization.json \
  --assignment-output results/my_run/Qampari_RAGW8G_manual_nuggetization_auto_assignment.json \
  --skip-done
```

---

## Adding a new dataset / generation type

1. Place the data file at `experiments/data/reproducibility_<Base>_<TAG>.json`.
2. Add an entry to `DATASETS` in `autonuggetizer/reproducibility_datasets.py`:

```python
"Qampari_RAGW16G": {
    "dataset": "Qampari",
    "generation_type": "RAGW16G",
    "raw_key": "wikipedia_gpt-4o_self_contained_paragraph_answer_raw",
    "type": "split_span",
},
```

The `raw_key` is the answer field name printed by `RAG_answer/json_to_pickle.py` when you convert the raw RAG output.

3. Run the pipeline with the new key:

```bash
python3 main_run_reproducibility.py --dataset Qampari_RAGW16G --provider gemini
```

---

## Module overview

| Module | Role |
|--------|------|
| `pipeline.py` | Core 3-stage AutoNuggetizer (`run_autonuggetizer`, `assign_nuggets`) |
| `llm.py` | Unified LLM client (OpenAI, Gemini, Azure, Claude, Qwen, LLaMA) |
| `prompts.py` | Prompt strings for all three stages |
| `scoring.py` | `vstrict` / `astrict` metric computation |
| `reproducibility_cli.py` | Argparse CLI — thin layer over `reproducibility_run` |
| `reproducibility_run.py` | Main loop: full pipeline and assignment-only mode |
| `reproducibility_data.py` | Load and prepare samples from JSON data files |
| `reproducibility_datasets.py` | `DATASETS` registry and path helpers |
| `reproducibility_experiment.py` | Experiment directory creation and `run_config.json` |
| `reproducibility_paths.py` | Path constants (`EXPERIMENTS_ROOT`) and utilities |
| `reproducibility_results.py` | Load/save result JSON, aggregate LLM usage stats |
| `reproducibility_human_eval.py` | Generate human-annotation template JSONs from pipeline output |
