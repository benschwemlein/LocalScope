# Model Comparison Notes

## test_08 results — three-model comparison (June 2026)

Run against the library-catalog-app corpus (397 Java files, 143 TypeScript files).
Metrics: faithfulness score and reference overlap score across 10 ground truth queries.
qwen2.5:7b and qwen3.6:27b ran together; gemma4:12b ran separately against the same corpus.

| Model | Faithfulness | Ref Overlap | Runtime |
|---|---|---|---|
| **qwen2.5:7b** | **0.51** | 0.48 | ~18 min |
| qwen3.6:27b | 0.47 | **0.50** | ~36 min |
| gemma4:12b | 0.44 | 0.40 | ~11 min |

### Per-query breakdown — faithfulness

| Query | qwen2.5:7b | qwen3.6:27b | gemma4:12b |
|---|---|---|---|
| loan_checkout_validation | 0.30 | 0.50 | 0.20 |
| loan_eligibility_chain | 0.43 | 0.43 | 0.43 |
| fine_calculation_strategy | 0.40 | 0.50 | 0.10 |
| hold_state_machine | 0.43 | 0.57 | **0.86** |
| recommendation_engine | 0.56 | 0.44 | 0.33 |
| overdue_batch_processing | 0.60 | 0.60 | 0.60 |
| full_text_search | 0.80 | 0.80 | 0.80 |
| notification_events | 0.40 | 0.50 | 0.40 |
| circulation_rules | 0.50 | 0.33 | 0.33 |
| reading_challenge | 0.67 | **0.00** | 0.33 |

### Per-query breakdown — reference overlap

| Query | qwen2.5:7b | qwen3.6:27b | gemma4:12b |
|---|---|---|---|
| loan_checkout_validation | 0.38 | 0.50 | 0.41 |
| loan_eligibility_chain | 0.53 | 0.42 | 0.37 |
| fine_calculation_strategy | 0.53 | 0.47 | 0.34 |
| hold_state_machine | 0.55 | 0.60 | 0.54 |
| recommendation_engine | 0.48 | 0.53 | 0.45 |
| overdue_batch_processing | 0.45 | 0.55 | 0.40 |
| full_text_search | 0.47 | 0.49 | 0.34 |
| notification_events | 0.52 | 0.59 | 0.39 |
| circulation_rules | 0.45 | 0.48 | 0.35 |
| reading_challenge | 0.39 | 0.38 | 0.41 |

## Conclusions

### qwen2.5:7b is the best fit for RAG-based code Q&A

It leads on faithfulness (0.51) and is competitive on reference overlap (0.48). It produces
clean output, runs in about half the time of qwen3.6:27b, and its scores are consistent
across query types with no format-related outliers.

### gemma4:12b trails on both metrics

Despite being the fastest (11 min) and mid-sized (12B), gemma4:12b scores lowest overall.
Notable exception: `hold_state_machine` faithfulness of 0.86 — best single score across all
three models. But `fine_calculation_strategy` at 0.10 and `loan_checkout_validation` at 0.20
drag the aggregate down. Not the right fit for this use case.

### The qwen3.6:27b reading_challenge 0.00 is a format bug, not a quality failure

qwen3.6:27b uses chain-of-thought thinking mode by default, emitting `<think>...</think>`
blocks before its answer. The scoring function compares raw output against the reference,
so the thinking tokens dilute the overlap score. gemma4:12b scored 0.33 on the same query
with no thinking mode, which supports this theory. Stripping `<think>` blocks before scoring
would validate it.

### Why Zack's recommendation does not transfer here

Zack Lowery's June 2026 article evaluated qwen3.6:27b on **agentic tool-calling tasks**
(Project Euler via web access + code execution, SVG file generation). qwen3's thinking mode
is an asset there — deeper reasoning improves sequential tool use and complex data structure
handling. For RAG-based code Q&A (our use case), thinking mode is a liability: the tokens
appear in the answer and harm scoring without improving the actual response content.

### Current recommendation

Keep **qwen2.5:7b** as the default chat model (`config.py: CHAT_MODEL`).

Revisit if: (1) `<think>` stripping is added to the answer pipeline and qwen3.6:27b
re-scores significantly higher, or (2) the use case shifts toward agentic workflows where
reasoning depth matters more than answer conciseness.
