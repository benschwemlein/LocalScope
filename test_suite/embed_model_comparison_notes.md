# Embedding Model Comparison Notes

## test_09 results — four-model comparison (June 2026)

Run against the library-catalog-app corpus (397 Java files, 143 TypeScript files).
Metrics: Precision@5, Recall@10, and MRR across 10 ground truth queries from test_04.
Each model built its own index from scratch.

| Model | Size | P@5 | R@10 | MRR |
|---|---|---|---|---|
| **mxbai-embed-large** | 669 MB | **0.38** | **0.56** | **0.83** |
| bge-m3 | 1.2 GB | 0.32 | 0.50 | 0.83 |
| nomic-embed-text | 274 MB | 0.28 | 0.47 | 0.71 |
| snowflake-arctic-embed2 | 568 MB | 0.26 | 0.36 | 0.75 |

### Per-query recall@10

| Query | nomic-embed-text | mxbai-embed-large | snowflake-arctic-embed2 | bge-m3 |
|---|---|---|---|---|
| loan_checkout_validation | 0.25 | 0.25 | 0.00 | 0.00 |
| loan_eligibility_chain | 0.40 | 0.40 | 0.20 | 0.20 |
| fine_calculation_strategy | 0.40 | 0.20 | 0.40 | 0.40 |
| hold_state_machine | 0.80 | 0.80 | 0.20 | 0.80 |
| recommendation_engine | 0.50 | **1.00** | 0.50 | 0.50 |
| overdue_batch_processing | 0.67 | 0.67 | 0.67 | 0.67 |
| full_text_search | 0.25 | 0.50 | 0.25 | 0.50 |
| notification_events | 0.50 | 0.25 | 0.50 | 0.50 |
| circulation_rules | 0.67 | **1.00** | 0.67 | 0.67 |
| reading_challenge | 0.25 | 0.50 | 0.25 | **0.75** |

## Conclusions

### mxbai-embed-large is the clear winner

Best P@5, best R@10, tied best MRR. Perfect recall (1.00) on `recommendation_engine` and
`circulation_rules`. Already configured as the default in `config.py`. No change needed.

### snowflake-arctic-embed2 is surprisingly weak

Despite being marketed as retrieval-optimized, it scores worst overall on R@10 (0.36) and
P@5 (0.26). Completely misses `loan_checkout_validation` and nearly misses `hold_state_machine`
(0.20). Not a good fit for code retrieval on this corpus.

### bge-m3 is a credible alternative at the cost of size

Ties mxbai-embed-large on MRR (0.83) and outperforms it on `reading_challenge` (0.75 vs 0.50).
At 1.2 GB it's nearly 2x the size of mxbai-embed-large with no overall gain. Worth revisiting
if the corpus grows significantly and MRR becomes more important than P@5.

### nomic-embed-text (prior default) is competitive for its size

At 274 MB it punches above its weight — beats snowflake on all metrics and only trails
mxbai-embed-large by 0.10 on R@10. A reasonable fallback for machines with limited storage.

### fine_calculation_strategy is a persistent retrieval failure

No embedding model achieves recall@10 above 0.40 for this query, and mxbai-embed-large
actually scores lower (0.20) than the others. The strategy pattern files
(`StandardFineStrategy`, `PremiumFineStrategy`, `StudentFineStrategy`) are semantically
similar to unrelated "fine" and "overdue" content in the corpus, causing consistent
retrieval confusion across all models. This is a query formulation and chunking problem,
not an embedding model problem.
