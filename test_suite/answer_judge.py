"""
LLM-judged answer quality, as opposed to retrieval hit rate.

Every other measurement in this suite scores whether the right files reached
the model's context. That is a proxy, and a leaky one: hit rate rises
mechanically with top_k because more slots mean more chances, so a retriever
can "improve" by simply returning more files. It also cannot see chunk
precision at all — a file split into seven chunks scores a full hit whether
the useful chunk or a useless one was retrieved.

This module scores the thing actually wanted: given the retrieved context,
does the tool answer correctly?

Design decisions worth stating:

- **The judge is a different model from the generator.** Self-evaluation
  inflates scores. The generator is the configured CHAT_MODEL; the judge is
  set separately and should be the strongest local model available.
- **The judge sees the reference answer and the scoring guidance**, both of
  which name the plausible-but-wrong response a naive tool gives. Those come
  from the private bank at runtime and are never written to disk or logged.
- **Reasoning is emitted before the verdict.** A model that states a score
  first then justifies it rationalizes; ordering the fields the other way
  keeps the judgment open while it reads the evidence.
- **Absolute scores from a small local judge are not trustworthy.** Arm-to-arm
  comparison is, because both arms face the same judge, same prompt, and same
  questions. Report deltas, not absolutes.
"""

import json
import os
import re

import requests

_JUDGE_PROMPT = """You are grading a code-navigation tool's answer to a question about a codebase.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{expected}

SCORING GUIDANCE (note especially any plausible-but-wrong answer named here):
{scoring_notes}

THE TOOL'S ANSWER:
{answer}

Grade how well the tool's answer matches the reference. Judge substance, not
wording or length. An answer that reaches the right conclusion by naming the
right code earns full credit even if phrased differently. An answer that gives
the plausible-but-wrong response named in the scoring guidance earns zero.

Respond with ONLY a JSON object, no other text, with these fields IN THIS ORDER:
{{"reasoning": "<one or two sentences comparing the answer to the reference>",
  "score": <0, 1, or 2>}}

score 2 = substantially correct, covers the reference answer's key points
score 1 = partially correct, right area but missing or muddling key points
score 0 = wrong, or the plausible-but-wrong answer, or non-committal
"""

_CHAT_TEMPLATE = """You are a senior engineer answering a question about a codebase.
Use ONLY the code snippets provided. If they are insufficient, say so plainly
rather than guessing.

Question:
<<BUG_TEXT>>

Relevant code snippets:
<<SNIPPETS>>

Answer concisely and name the specific files, classes, and methods involved."""


def _strip_fences(text: str) -> str:
    """Small local models wrap JSON in markdown fences despite instructions."""
    return re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.M)


def judge_answer(
    question: str,
    expected: str,
    scoring_notes: str,
    answer: str,
    judge_model: str,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 300,
) -> tuple[int, str]:
    """Return (score 0-2, reasoning). Returns (-1, reason) if judging failed,
    so callers can exclude rather than silently score it zero."""
    if not answer.strip():
        return 0, "empty answer"

    prompt = _JUDGE_PROMPT.format(
        question=question, expected=expected,
        scoring_notes=scoring_notes or "(none given)", answer=answer,
    )
    try:
        resp = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=timeout,
        )
        if not resp.ok:
            return -1, f"judge HTTP {resp.status_code}"
        content = resp.json()["message"]["content"]
    except Exception as e:
        return -1, f"judge call failed: {type(e).__name__}"

    raw = _strip_fences(content)
    try:
        data = json.loads(raw)
    except ValueError:
        m = re.search(r'"score"\s*:\s*([0-2])', raw)
        if m:
            return int(m.group(1)), "recovered from malformed JSON"
        return -1, "unparseable judge output"

    score = data.get("score")
    if not isinstance(score, int) or score not in (0, 1, 2):
        return -1, "judge returned no valid score"
    return score, str(data.get("reasoning", ""))[:200]


def generate_answer(question: str, index_dir: str, top_k: int) -> str:
    """Run the full pipeline including generation, using the configured arm."""
    from querying.query_engine import run_query

    result = run_query(
        bug_text=question,
        index_dir=index_dir,
        top_k=top_k,
        chat_template=_CHAT_TEMPLATE,
        log=lambda _: None,
    )
    return result.get("answer", "")
