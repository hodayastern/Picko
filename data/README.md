# data/

| Path | What |
|---|---|
| `picko_training_pool.jsonl` | Main training pool — all 75 tools, 120 examples each + negatives (~26 MB). **Not in git**; upload to `MyDrive/picko/`. The notebooks filter the 40 focus tools out of it in code. |
| `scientific_pools.json` | The 40 focus tools grouped by family (pubmed/arxiv/wikipedia/huggingface) — the tool-cluster input the Gemini generator (`notebooks/01_generate_data.ipynb`) synthesizes examples from. |
| `semantic/cs.json` | nb3 source-free computer-science queries (→ arxiv). |
| `semantic/medicine.json` | nb3 source-free medical queries (→ pubmed). |
| `semantic/probe.jsonl` | nb3 prebuilt  dataset. |
| `semantic/train.jsonl` | Built by nb3 at runtime — not in git. |
