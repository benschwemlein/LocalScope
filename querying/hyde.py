"""
HyDE: Hypothetical Document Embeddings, adapted to code search.

A plain-English question and the code that answers it share few words, so
their embeddings can sit far apart. HyDE (Gao et al., ACL 2023) closes that
gap before searching: a language model writes a made-up answer, and the
made-up answer is embedded instead of the question. For code the made-up
answer is a short, plausible snippet: the class, method and field names the
real code would probably use. It can be wrong in every detail; it only has
to look like the right neighbourhood, so the search compares code with code.

Modes (config.HYDE_MODE):
    off    embed the question (default)
    doc    embed only the hypothetical snippet
    both   embed the question followed by the snippet, so the question's
           own wording anchors the search while the snippet adds code-like
           vocabulary
"""

import requests

import config

PROMPT = """You are helping search a codebase. Write a short code snippet that
would plausibly appear in the code that answers the question below: the
class, method and field declarations, annotations and key calls a developer
would expect to find there. Use realistic names in the style of the
codebase. Output only code, at most about 30 lines, no explanation.
{hint}
Question: {question}"""


def hypothetical_snippet(question: str, log=print) -> str:
    """The model's guess at the code that answers `question`, or "" on failure."""
    hint = f"The codebase is: {config.HYDE_HINT}\n" if config.HYDE_HINT else ""
    try:
        resp = requests.post(
            f"{config.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": config.HYDE_MODEL,
                "messages": [{"role": "user",
                              "content": PROMPT.format(hint=hint, question=question)}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 400},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as e:
        log(f"[hyde] snippet generation failed, embedding the question instead: {e}")
        return ""
    # Strip a surrounding markdown fence if the model added one.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def text_to_embed(question: str, mode: str | None = None, snippet: str | None = None,
                  log=print) -> str:
    """What to embed for `question` under `mode` (default config.HYDE_MODE).

    Pass `snippet` to reuse an already generated guess.
    """
    mode = mode or config.HYDE_MODE
    if mode == "off":
        return question
    if snippet is None:
        snippet = hypothetical_snippet(question, log)
    if not snippet:
        return question
    return snippet if mode == "doc" else f"{question}\n\n{snippet}"
