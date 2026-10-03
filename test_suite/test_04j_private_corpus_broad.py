"""
Broad questions on the private benchmark corpus.

The main bank (test_04b and friends) asks mostly targeted questions: where is
X decided, which callers use Y. Broad questions ("how does a hold work end to
end", "where do the settings come from") need many files, name no classes, and
push the search agent to lean on the semantic index. On one such question the
agent read 18 files but could hand only 10 to the answer step, which then
guessed about the parts it never saw. This test measures that.

For each file budget (LCQ_BROAD_FILE_BUDGETS, default "10,15") it runs the
default answer path's search (the trust-seeded agent, querying/agent_answer.py)
and reports, per question and on average:

    found       share of the question's evidence files the agent returned
    in context  share that also fit into the whole-file context the answer
                model reads (LCQ_AGENT_CONTEXT_CHARS, default 70,000)
    MRR         rank of the first evidence file
    calls, index queries, seconds

Broad questions list 8 to 15 evidence files, so "found" can't reach 1.0 with a
budget below the evidence count.

    LOCALSCOPE_CORPUS=/path/to/corpus \\
    LOCALSCOPE_BROAD_BANK=/path/to/broad-question-bank \\
    python3 -m pytest test_suite/test_04j_private_corpus_broad.py -v -s

Optional:
    LCQ_BROAD_FILE_BUDGETS  comma-separated file budgets to compare ("10,15")
    LCQ_BROAD_INDEX_DIR     reuse or keep this index directory
    LCQ_AGENT_QUESTIONS     comma-separated question ids to run
    LCQ_BROAD_OUT           write per-question results (files, trace counts,
                            and answers if LCQ_BROAD_ANSWER=1) to this JSON file
    LCQ_BROAD_ANSWER=1      also answer each question from the whole files, for
                            grading against the bank's `expected` by a judge

The answer key never reaches the agent or the answer model; `evidence_files`
is only read here, for scoring.
"""

import json
import os
import time

import pytest
import yaml

from test_04b_private_corpus_eval import build_index, mrr

ANSWER_TEMPLATE = (
    "You are a senior engineer answering a question about the codebase.\n\n"
    "Question:\n<<BUG_TEXT>>\n\n"
    "Source files:\n<<SNIPPETS>>\n\n"
    "Answer in one to three paragraphs of plain prose, naming the specific "
    "classes, methods and file paths that matter. Be concrete and direct."
)


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def load_broad_questions(bank_dir: str) -> list[dict]:
    questions: list[dict] = []
    for name in sorted(os.listdir(bank_dir)):
        if name.endswith((".yaml", ".yml")):
            with open(os.path.join(bank_dir, name), encoding="utf-8") as fh:
                questions += [q for q in (yaml.safe_load(fh) or []) if q.get("category") == "broad"]
    return questions


def _covered(evidence: list[str], files: list[str]) -> float:
    if not evidence:
        return 1.0
    return sum(1 for e in evidence if any(e in f for f in files)) / len(evidence)


def test_broad_questions(tmp_path_factory):
    corpus = os.path.expanduser(os.environ.get("LOCALSCOPE_CORPUS", ""))
    bank = os.path.expanduser(os.environ.get("LOCALSCOPE_BROAD_BANK", ""))
    if not corpus or not bank:
        pytest.skip("LOCALSCOPE_CORPUS and LOCALSCOPE_BROAD_BANK must both be set")

    import chromadb
    from chromadb.config import Settings

    import config
    from querying.agent_answer import find_files, whole_files
    from querying.query_engine import _chat_with_context

    config.RERANK_ENABLED, config.SECOND_ROUND, config.HYDE_MODE = False, "off", "off"
    questions = load_broad_questions(bank)
    subset = set(_env_list("LCQ_AGENT_QUESTIONS"))
    if subset:
        questions = [q for q in questions if q["id"] in subset]
    assert questions, f"no broad questions found in {bank}"
    budgets = [int(b) for b in _env_list("LCQ_BROAD_FILE_BUDGETS")] or [10, 15]
    answer = os.environ.get("LCQ_BROAD_ANSWER") == "1"

    index_dir = os.path.expanduser(os.environ.get("LCQ_BROAD_INDEX_DIR", "")) \
        or str(tmp_path_factory.mktemp("broad_index"))
    if not (os.path.isdir(index_dir) and os.listdir(index_dir)):
        build_index(corpus, index_dir)
    client = chromadb.PersistentClient(path=index_dir, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(client.list_collections()[0].name)

    out_path = os.environ.get("LCQ_BROAD_OUT")
    raw: dict = {"budgets": budgets, "context_chars": config.AGENT_CONTEXT_CHARS, "runs": {}}
    rows = []
    saved_files = config.AGENT_FILES
    try:
        for budget in budgets:
            config.AGENT_FILES = budget
            found, in_ctx, ranks, calls, index_q, secs = [], [], [], [], [], []
            for q in questions:
                run = find_files(q["question"], collection, corpus, log=lambda _: None)
                docs, metas, skipped = whole_files(corpus, run.files, config.AGENT_CONTEXT_CHARS)
                context = [m["source"] for m in metas]
                ev = q.get("evidence_files") or []
                found.append(_covered(ev, run.files))
                in_ctx.append(_covered(ev, context))
                ranks.append(mrr(run.files, set(ev)))
                calls.append(run.tool_calls)
                index_q.append(sum(1 for e in run.trace if e.get("event") == "index"))
                secs.append(run.seconds)
                entry = {"files": run.files, "context_files": context, "skipped": skipped,
                         "found": found[-1], "in_context": in_ctx[-1], "tool_calls": run.tool_calls,
                         "submitted": run.submitted, "fallback": run.fallback, "error": run.error,
                         "index_queries": index_q[-1], "seconds": round(run.seconds, 1)}
                if answer:
                    t0 = time.monotonic()
                    entry["answer"] = _chat_with_context(q["question"], docs, metas, ANSWER_TEMPLATE,
                                                         lambda _: None)
                    entry["answer_seconds"] = round(time.monotonic() - t0, 1)
                raw["runs"].setdefault(str(budget), {})[q["id"]] = entry
                print(f"    budget {budget} {q['id']}: found {found[-1]:.2f}, in context "
                      f"{in_ctx[-1]:.2f}, {run.tool_calls} calls, {index_q[-1]} index queries, "
                      f"{run.seconds:.0f}s", flush=True)
                if out_path:
                    with open(os.path.expanduser(out_path), "w", encoding="utf-8") as fh:
                        json.dump(raw, fh, indent=1)
            n = len(questions)
            rows.append((budget, sum(found) / n, sum(in_ctx) / n, sum(ranks) / n,
                         sum(calls) / n, sum(index_q) / n, sum(secs) / n))
    finally:
        config.AGENT_FILES = saved_files

    print("\n\n=== Private Corpus Broad Questions (trust agent + whole files) ===\n")
    print(f"  n={len(questions)} broad questions, whole-file context "
          f"{config.AGENT_CONTEXT_CHARS:,} characters\n")
    print(f"{'files':>6} {'found':>7} {'in context':>11} {'MRR':>6} {'calls/q':>8} "
          f"{'index/q':>8} {'s/q':>6}")
    print("-" * 60)
    for budget, f, c, r, cl, iq, s in rows:
        print(f"{budget:>6} {f:>7.2f} {c:>11.2f} {r:>6.2f} {cl:>8.1f} {iq:>8.1f} {s:>6.0f}")
    print("=" * 60)
