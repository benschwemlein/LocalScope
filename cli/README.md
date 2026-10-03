# rag_query CLI

Command-line tool for querying a local RAG index. Uses the same model and index settings as the GUI (`~/.local-rag-llm/config.json`).

## Usage

```bash
python3 cli/rag_query.py --index /path/to/index "your question"
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--index`, `-i` | from GUI settings | Path to ChromaDB index directory |
| `--top-k`, `-k` | 12 | Number of results to return |
| `--max-chars` | 4000 | Summarize query if longer than this many chars |
| `--no-snippets` | off | Print file paths and scores only, no snippet text |
| `--json` | off | Output results as JSON |

## Examples

```bash
# Basic query
python3 cli/rag_query.py -i ~/dev/indexes/commerce "how does JWT auth work"

# More results, no snippet text
python3 cli/rag_query.py -i ~/dev/indexes/commerce -k 20 --no-snippets "payment flow"

# JSON output
python3 cli/rag_query.py -i ~/dev/indexes/commerce --json "order status updates"
```

## Output formats

**Default** — ranked snippets with scores:
```
=== RAG Results: 12 files ===

[01]  95.0%  src/main/java/com/example/auth/JwtTokenService.java  (chunk 2)
------------------------------------------------------------
// JwtTokenService.java
public String validateToken(String token) { ...
```

**`--no-snippets`** — file list only:
```
[01]  95.0%  src/main/java/com/example/auth/JwtTokenService.java  (chunk 2)
[02]  87.3%  src/main/java/com/example/auth/JwtTokenFilter.java  (chunk 0)
```

**`--json`** — machine-readable, useful for feeding results to Claude:
```json
[
  {
    "source": "src/main/java/com/example/auth/JwtTokenService.java",
    "score": 95.0,
    "distance": 0.1234,
    "snippet": "..."
  }
]
```

## Settings

Model and Ollama settings are read from `~/.local-rag-llm/config.json` — the same file the GUI writes. Changing models in the GUI Settings tab will automatically apply here too.
