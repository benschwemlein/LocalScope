# Test Suite

Integration and semantic evaluation tests for the Local RAG LLM indexer and query engine.
All tests run against **AngularAndSpringSampleApp** (a medium-sized Java + Angular crypto trading app).

---

## Quick start

```bash
# From the repo root
cd /path/to/code_llm

# Run the full standard suite (indexes once, reuses for all tests)
python3.13 -m pytest test_suite/ -v -s

# Run the chunking/overlap eval tests (slow — builds many indexes)
python3.13 -m pytest test_suite/test_05_chunking_strategies.py -m chunking_eval -v -s
python3.13 -m pytest test_suite/test_06_overlap_tuning.py -m chunking_eval -v -s
```

The standard suite indexes the sample app **once per session** and reuses the index for all tests.
Mutation tests (incremental update, delete, modify) get a cheap `shutil.copytree` copy — no re-embedding.

---

## Test files

| File | What it tests |
|------|---------------|
| `test_01_index_full.py` | Full index completes, collection created, chunks present, metadata valid |
| `test_02_index_incremental.py` | Incremental indexer detects adds, updates, deletes; skips unchanged files |
| `test_03_query.py` | Query engine returns results with correct structure and source metadata |
| `test_04_semantic_eval.py` | Semantic recall/precision/MRR against 10 ground-truth queries |
| `test_05_chunking_strategies.py` | Compares AST vs text chunking at 3 chunk sizes — report only, always passes |
| `test_06_overlap_tuning.py` | Compares text overlap values (0–600) at fixed 800-char chunks — report only |
| `test_07_answer_quality.py` | LLM answer faithfulness + reference keyword overlap for 10 queries |
| `test_08_model_comparison.py` | Compares chat models on faithfulness + reference overlap — report only |
| `test_09_embed_model_comparison.py` | Compares embedding models on p@5, r@10, MRR — report only |

---

## Semantic evaluation thresholds (test_04)

| Metric | Threshold |
|--------|-----------|
| Mean Precision@5 | ≥ 0.35 |
| Mean Recall@10 | ≥ 0.50 |
| Mean MRR | ≥ 0.40 |
| Per-query Recall@10 | ≥ 0.25 (floor) |

---

## Chunking strategy results (test_05)

Tested on AngularAndSpringSampleApp with overlap=200.

```
Strategy          p@5   r@10    MRR
----------------------------------------------------
  ast_1200       0.34   0.47   0.70
  ast_800        0.43   0.56   0.68
  ast_600        0.38   0.50   0.62
  text_1200      0.38   0.50   0.64
  text_800        0.40   0.51   0.80  ← best MRR
  text_400       0.38   0.46   0.62
----------------------------------------------------
```

**Winner: text_800** — best MRR (0.80). AST chunking wins recall at 800 chars (0.56) but
text chunking's MRR lead indicates it ranks the right file higher. Current default is text_800.

---

## Overlap tuning results (test_06)

Fixed chunk_size=800, text chunking, varied overlap.

```
Variant           p@5   r@10    MRR
----------------------------------------------------
  text_800_ov0   0.38   0.46   0.70
  text_800_ov100 0.36   0.46   0.52
  text_800_ov200 0.38   0.49   0.75  ← best MRR & recall
  text_800_ov400 0.38   0.48   0.70
  text_800_ov600 0.36   0.46   0.50
----------------------------------------------------
```

**Winner: overlap=200** — best on both MRR (0.75) and recall (0.49). Current default confirmed optimal.

---

## Chat model comparison results (test_08)

```
Model                   faith   ref_ovlp
--------------------------------------------
  llama3.1               0.41     0.38
  deepseek-coder:6.7b    0.42     0.38
  qwen2.5:7b             0.56     0.41  ← current default
--------------------------------------------
```

`qwen2.5:7b` wins on both metrics. Faithfulness +37% over llama3.1 — it cites class names
rather than paraphrasing. `deepseek-coder` strong on some queries but inconsistent.

Run with:
```bash
python3.13 -m pytest test_suite/test_08_model_comparison.py -m chunking_eval -v -s
```

---

## Current defaults

| Parameter | Value | Set in |
|-----------|-------|--------|
| `chars_per_chunk` | 800 | `indexer.py`, `incremental_indexer.py`, `config.json` |
| `chunk_overlap` | 200 | `indexer.py`, `incremental_indexer.py` |
| `use_ast_chunking` | False (text) | GUI default |
| `max_file_bytes` | 500,000 | `indexer.py` |
| `embed_model` | mxbai-embed-large | `config.json` |
| `chat_model` | qwen2.5:7b | `config.json` |

---

## Answer quality results (test_07, llama3.1 baseline)

Faithfulness = fraction of retrieved class names mentioned in answer.
Reference overlap = keyword match against hand-written reference answer.

```
Query                       faithful  must_kw  ref_ovlp
------------------------------------------------------------
jwt_authentication            0.40     PASS      0.25
kafka_events                  0.30     PASS      0.27
mongodb_configuration         0.20     PASS      0.51
user_management               0.50     PASS      0.43
scheduled_tasks               0.20     xfail     0.23   ← weak query
angular_routing               0.60     PASS      0.62
statistics                    0.50     xfail     0.34   ← weak query
exception_handling            0.30     PASS      0.45
quote_data_model              0.57     PASS      0.37
angular_exchange_services     0.50     PASS      0.36
------------------------------------------------------------
Thresholds: faithfulness ≥ 0.20, reference overlap ≥ 0.20
```

`scheduled_tasks` and `statistics` are marked `xfail` — llama3.1 paraphrases these
answers without citing specific class names. Will flip to xpass with a better model.

---

## Embedding model comparison results (test_09)

```
Model                  p@5   r@10    MRR
----------------------------------------------------
  nomic-embed-text    0.40   0.56   0.62
  mxbai-embed-large   0.66   0.82   0.91  ← current default
----------------------------------------------------

Query                        nomic   mxbai  (r@10)
----------------------------------------------------
  jwt_authentication          0.60    0.60
  kafka_events                0.60    0.80
  mongodb_configuration       0.50    1.00
  user_management             0.83    1.00
  scheduled_tasks             0.75    0.75
  angular_routing             1.00    1.00
  statistics                  0.60    0.80
  exception_handling          0.75    0.75
  quote_data_model            0.00    0.80  ← was broken
  angular_exchange_services   0.00    0.75  ← was broken
----------------------------------------------------
```

`mxbai-embed-large` fixed the two previously 0.00-recall queries (thin POJOs and
cross-language TypeScript). Existing indexes must be fully reindexed after switching.

Run with:
```bash
python3.13 -m pytest test_suite/test_09_embed_model_comparison.py -m chunking_eval -v -s
```

---

## Known gaps

- No remaining 0.00-recall queries with `mxbai-embed-large` — previously broken queries
  (`quote_data_model`, `angular_exchange_services`) now score 0.80 and 0.75 respectively.
- Answer quality (`test_07`) thresholds calibrated to `llama3.1`; re-run `test_08` if chat
  model changes to verify improvement.
