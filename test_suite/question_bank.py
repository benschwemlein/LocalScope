"""
Loader for an external evaluation question bank.

LocalScope is a PUBLIC repository. The benchmark corpus and its question
bank live in PRIVATE repositories, and the whole point of keeping them
private is that a corpus a model has seen is a corpus that no longer
measures anything. So nothing from either may ever be committed here: no
question text, no expected answers, no evidence-file paths, no corpus
class names.

This module is plumbing only. It reads the bank at runtime from a path
given by an environment variable and returns just the two fields a
retrieval evaluation needs — the question, and which files must show up
in the tool's context. `expected` and `scoring_notes` are deliberately
NOT surfaced: those are judge-time answer key, and retrieval scoring
doesn't need them.

Both locations come from environment variables with no defaults. A default
would have to name the private repositories, and naming them here would
advertise from a public codebase exactly what is meant to stay unlisted.
Unset variables mean the eval skips, which is the correct behaviour for
every checkout that isn't the corpus author's.

Environment:
    LOCALSCOPE_QUESTION_BANK  directory holding the q-*.yaml question files
    LOCALSCOPE_CORPUS         checkout of the corpus under test
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def question_bank_dir() -> Path | None:
    raw = os.environ.get("LOCALSCOPE_QUESTION_BANK")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def corpus_dir() -> Path | None:
    raw = os.environ.get("LOCALSCOPE_CORPUS")
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


@dataclass
class Question:
    id: str
    category: str          # nav | live | bug | impact | hist | cross
    language: str          # java | ts | cross
    arm_prediction: str    # graph | tie | vector
    difficulty: str        # easy | medium | hard
    question: str
    evidence_files: list[str] = field(default_factory=list)
    ledger: list[str] = field(default_factory=list)

    @property
    def is_neutral(self) -> bool:
        """True when no planted flaw sits behind this question."""
        return not self.ledger


def load_questions() -> list[Question]:
    """
    Load every question in the bank. Returns [] if the bank or PyYAML is
    unavailable, so callers can skip rather than fail.
    """
    bank = question_bank_dir()
    if bank is None:
        return []

    try:
        import yaml
    except ImportError:
        return []

    questions: list[Question] = []
    for path in sorted(bank.glob("q-*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries = yaml.safe_load(fh) or []
        except Exception:
            continue
        for e in entries:
            if not isinstance(e, dict) or "id" not in e:
                continue
            questions.append(
                Question(
                    id=e["id"],
                    category=e.get("category", ""),
                    language=e.get("language", ""),
                    arm_prediction=e.get("arm_prediction", ""),
                    difficulty=e.get("difficulty", ""),
                    question=(e.get("question") or "").strip(),
                    evidence_files=list(e.get("evidence_files") or []),
                    ledger=list(e.get("ledger") or []),
                )
            )
    return questions


def load_judge_fields() -> dict[str, dict[str, str]]:
    """
    Load the answer-key fields, keyed by question id: the reference answer and
    the scoring guidance naming the plausible-but-wrong response.

    Kept out of Question deliberately. These exist to be handed to a judge
    model at scoring time and nowhere else — not printed, not logged, not
    summarized. Loading them separately keeps the ordinary retrieval path from
    ever touching them by accident.
    """
    bank = question_bank_dir()
    if bank is None:
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    out: dict[str, dict[str, str]] = {}
    for path in sorted(bank.glob("q-*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entries = yaml.safe_load(fh) or []
        except Exception:
            continue
        for e in entries:
            if isinstance(e, dict) and "id" in e:
                out[e["id"]] = {
                    "expected": (e.get("expected") or "").strip(),
                    "scoring_notes": (e.get("scoring_notes") or "").strip(),
                }
    return out


def resolvable_at(questions: list[Question], corpus: Path) -> list[Question]:
    """
    Keep only questions whose every evidence file exists in this checkout.

    Some questions deliberately anchor to a historical tag or branch; their
    evidence was deleted from the tip. Evaluating those against HEAD would
    score a correct tool as wrong, so they're excluded unless the harness
    checks out the ref they name.
    """
    return [
        q for q in questions
        if q.evidence_files
        and all((corpus / p).exists() for p in q.evidence_files)
    ]


def retrieval_eligible(questions: list[Question]) -> list[Question]:
    """
    Drop questions a retrieval-only evaluation can't fairly score.

    `hist` questions reason over git history. LocalScope indexes a working
    tree and has no history access at all, so these are N/A rather than
    failures — scoring them as zero would understate every arm equally and
    add noise to the comparison.
    """
    return [q for q in questions if q.category != "hist"]
