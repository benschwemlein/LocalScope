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

## Current defaults

| Parameter | Value | Set in |
|-----------|-------|--------|
| `chars_per_chunk` | 800 | `indexer.py`, `incremental_indexer.py`, `config.json` |
| `chunk_overlap` | 200 | `indexer.py`, `incremental_indexer.py` |
| `use_ast_chunking` | False (text) | GUI default |
| `max_file_bytes` | 500,000 | `indexer.py` |

---

## Known gaps

- `quote_data_model` and `angular_exchange_services` score 0.00 recall — thin POJO files and
  cross-language TypeScript queries are not well handled by `nomic-embed-text`. A reranker or
  better embedding model would be needed to close these.
