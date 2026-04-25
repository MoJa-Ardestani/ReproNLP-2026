# RAG Answer Generation

All commands run from `RAG_answer/`.

## 1. Generate RAG answers

```
python3 run_rag_answers.py \
  [--backend wikipedia+openai | wikipedia+gemini | wikipedia+claude | perplexity] \
  [--model MODEL] \
  [--num-passages N]        # Wikipedia backends only (default: 4)
  [--input PATH]            # default: data/reproducibility_Qampari.json
  [--limit N]               # process only first N questions
  [--start-from I]          # 1-based index to start from
  [--resume]                # skip already-processed qids in existing output
  [--output PATH]           # override default output path
```

Output: `results/rag_answers/<TAG>/<timestamp>/<dataset>_rag_answers.json`

Tags: `RAGW{n}G` (OpenAI), `RAGW{n}M` (Gemini), `RAGW{n}C` (Claude), `RAGP` (Perplexity)

## 2. Convert to reproducibility JSON

```
python3 json_to_pickle.py \
  --generation-type TAG \   # e.g. RAGW4C, RAGW8G, RAGP
  --run-id TIMESTAMP \      # e.g. 20260421_103355
  [--source-pickle PATH]    # override auto-detected baseline (default: data/reproducibility_Qampari.json)
```

Output: `data/reproducibility_Qampari_<TAG>.json`
