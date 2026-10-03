

# LocalScope: Local AI code intelligence for any codebase

A desktop application for asking questions about any source code repository using only local models. You ask in plain English (a question, a bug report or a stack trace), a local search agent finds the relevant files with help from a semantic index, and a local LLM reads those files in full and explains how the code works or why a problem occurs. **Everything runs on your machine, no cloud calls, no code leaves your laptop.**

### Why I built it

Ramping onto a large, unfamiliar codebase is slow, and sending proprietary source to a cloud LLM is often off the table. I built this to ask plain-English questions about a codebase ("where is the credit-card application flow handled?", "what calls this service?") and get back the exact files plus an explanation, entirely offline. I used it to onboard onto a large commercial codebase faster than the rest of the team.

### Technical highlights

* **Agent search seeded by the index.** A local model searches the repository with grep, file search, file reads and a semantic search tool, starting from the index's best matches, until it can name the files that answer the question.
* **Answers from whole files.** The answer model reads the chosen files in full instead of short snippets, which is the single biggest accuracy gain measured.
* **AST aware chunking.** The index splits code at logical boundaries (functions, classes, methods) instead of arbitrary character offsets, using the `astchunk` library with a Python `ast` fallback.
* **Incremental indexing.** Only changed files are indexed again instead of rebuilding the whole vector store.
* **Fully local and private.** ChromaDB for vector search and Ollama for embeddings and chat, with no API keys and no external services.
* **Measured, not guessed.** A pytest benchmark suite scores retrieval and answer accuracy against an answer key, so every default here was chosen by measurement.
* **GUI and CLI.** A Tkinter desktop app for interactive use and a CLI for scripting.

---

# Results

Answer accuracy on two codebases, graded against a written answer key (correct = 1, partly right = 0.5, wrong = 0) by a blind Claude Opus judge that didn't know which tool wrote each answer:

| Tool | Private codebase (49 questions) | ThingsBoard (40 questions) |
|---|---|---|
| **LocalScope** (local qwen3.6:35b-a3b) | **0.86** | **0.81** |
| Claude Sonnet 5 agent (cloud) | 0.90 | 1.00 |
| Continue CLI (same local model) | 0.70 | 0.34 |

* The private codebase is a synthetic Java/Spring and Angular application with planted defects and hidden wiring that no model has seen in training. On it LocalScope and Sonnet 5 are statistically tied.
* ThingsBoard is a public IoT platform of about 10,000 files. Sonnet 5 leads there, and a claim by claim check showed its answers came from code it read, not from memorizing the repository.
* With the same local model, LocalScope beats Continue on both codebases, by the widest margin on the large one.
* Answering from index snippets alone, the approach LocalScope used before, scored 0.61 on the private codebase.
* LocalScope ran fully offline on a 48 GB Apple silicon Mac; only the grading used a cloud model. A question takes about 1.5 to 2 minutes. One run per question with a single judge model, so treat differences of a few points as noise.

---

# Quick Start

1. Install **Python 3.10+**

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install **Ollama** and make sure it is running  
   https://ollama.com/download

4. Clone this repo:

   ```bash
   git clone https://github.com/benschwemlein/LocalScope.git
   cd LocalScope
   pip install -r requirements.txt
   ```

5. Pull the models the results above were measured with:

   ```bash
   ollama pull qwen3.6:35b-a3b
   ollama pull mxbai-embed-large
   ```

6. Start the application:

   ```bash
   python app.py
   ```

7. Open the **Settings** tab  
   Confirm Ollama shows a **green status icon** and select the embedding and chat models.

8. Open the **Index** tab and index your repository.

9. On the **Query** tab, set the **Repository root** to the same repository. The search agent reads the code from there.


You are ready to query code.

---

# Overview

LocalScope provides two workflows:

### Indexing

The indexer scans a repository, chunks supported files, embeds the chunks using a local embedding model, and stores vectors in a ChromaDB index.

### Querying

The user provides a question, bug report, log output, or general investigation text. A local search agent starts from the index's best matching files, confirms them and fills gaps with grep, file search, file reads and semantic search, then picks the ten most relevant files. The chat model reads those files in full and explains the answer.

---

# Features

* Fully local RAG pipeline
* No cloud calls
* Vector search powered by ChromaDB
* Embedding and chat models served by Ollama
* Interactive GUI with tabs for **Query**, **Indexing**, **Settings**, and **Prompts**
* Clear model status indicator (green/red)
* Download new models directly via dropdown
* Selectable file types and excluded directories for indexing
* Click result to open file in your OS
* Editable prompts stored as JSON

---

# Requirements

### Python

* Python 3.10+
* Tkinter (included in most Python distributions)

### Python packages

```bash
pip install chromadb requests
```

### Ollama

1. Install Ollama
   [https://ollama.com/download](https://ollama.com/download)

2. Install at least one embedding model and one chat model:

```bash
ollama pull mxbai-embed-large
ollama pull qwen3.6:35b-a3b
```

qwen3.6:35b-a3b is a mixture of experts model (about 3B active parameters), fast enough for the agent loop on a laptop. It needs about 24 GB of memory.

3. Ensure Ollama is running (shows green in Settings tab):

```
http://localhost:11434
```

---

# Installation

Clone the project:

```bash
git clone https://github.com/benschwemlein/LocalScope.git
cd LocalScope
pip install chromadb requests
```

Run the application:

```bash
python app.py
```

This launches the full Tkinter GUI.

---

# First Time Setup

Open the **Settings** tab:

### 1. Check Ollama status

A green ● indicator means Ollama is reachable.
A red ● means it is not running or the URL is wrong.

### 2. Select models

Use the dropdowns to choose:

* Embedding model
* Chat model

You may also download models using the **Download Model** dropdown button.

### 3. Save changes

Click **Apply Settings**.

---

# Indexing a Repository

Open the **Index** tab.

### Parameters include:

* Repository root directory
* Index output directory
* Collection name
* Chunk size and overlap
* Maximum file size
* **Selectable file types** grouped by language ecosystem
* **Excluded directories**, editable by the user

### Steps

1. Choose the repository you want to index
2. Select the file types you care about
3. Adjust excluded directories if needed
4. Click **Index**

The indexer:

* Walks the repository
* Skips excluded directories such as:
  `.git`, `node_modules`, `build`, `dist`, `target`, `.gradle`, virtual envs, caches
* Reads only file types you selected
* Splits files into overlapping chunks
* Embeds each chunk with the local embedding model
* Stores chunks and metadata in ChromaDB

Re-index whenever you switch to a different embedding model.

---

# Running Queries

Open the **Query** tab.

You can configure:

* Index directory
* Repository root (the repository the agent searches and reads)
* Number of snippets to retrieve (used when answering from snippets)
* Maximum characters for summarization
* The question / bug report text
* Output display mode

### Steps

1. Type or paste your investigation text
2. Click **Run Query**

The engine:

1. Summarizes long input using your Summarizer prompt
2. Runs the search agent, seeded with the index's top matching files
3. Reads the agent's chosen files in full, up to 70,000 characters
4. Injects the files and your question into the Chat prompt
5. Runs your local LLM to produce an answer
6. Displays the result with clickable file paths

Without a repository root, or if the agent finds nothing, the engine answers from the index's top snippets instead. That mode is much faster but much less accurate; to use it on purpose, set `LCQ_ANSWER_MODE=snippets`.

---

# Prompts

Two prompt templates drive the workflow:

### Summarizer Prompt

Condenses long text into a search query.
Must contain:

```
<<BUG_TEXT>>
```

### Chat Prompt

Produces the final explanation.
Must contain:

```
<<BUG_TEXT>>
<<SNIPPETS>>
```

Prompts are editable, savable, loadable, and validated before running queries.

---

# Settings

The **Settings** tab includes:

* Ollama URL
* Model status indicator (green/red)
* Embedding model dropdown
* Chat model dropdown
* **Download Model** button with a curated dropdown list
* Refresh models
* Apply settings

---

# Opening Files

After a query, results show:

* File path
* Snippet
* Relevance score

Double-clicking a result opens the file using:

* macOS → `open`
* Windows → `start`
* Linux → `xdg-open`

When a repository root is set, paths resolve relative to it.

---

# Tips for Best Results

* Include logs, stack traces, symptoms, and environment details
* Increase `number of results` for complex issues
* Keep prompts explicit and stable
* Re-index after changing embedding models
* Ensure Ollama is running before starting queries or indexing

---

# Troubleshooting

### Ollama shows red

Ollama is not running or the URL is wrong.

### No embedding model available

Use the **Download Model** dropdown or run:

```bash
ollama pull nomic-embed-text
```

### Query produces no results

Index directory is incorrect or has not been built.

### File paths fail to open

Repository root is incorrect or files moved.

### Query fails to start

Summarizer or Chat prompt is empty.

---

# Project Structure

```
LocalScope/
  app.py                      # Tkinter GUI entry point
  config.py / settings_*.py   # settings + persisted config (~/.local-rag-llm/config.json)

  gui/                        # GUI tabs: Query, Index, Settings, Prompts
    query_tab.py
    index_tab.py
    settings_tab.py
    prompts_tab.py

  indexing/                   # repository indexing
    indexer.py                # full index
    incremental_indexer.py    # re-index only changed files
    ast_chunker.py            # AST-aware code chunking

  querying/
    query_engine.py           # run_query: agent answer path, snippet fallback
    agent_search.py           # local search agent (grep, find, read, semantic search)
    agent_answer.py           # index seeded agent search + whole file context
    reranker.py, second_round.py, hyde.py   # optional retrieval experiments, off by default

  ollama_manager/             # local model management + downloads
    download_manager.py

  cli/
    rag_query.py              # command-line query interface

  test_suite/                 # pytest benchmarks: embedding models, chunking,
                              # overlap, top-k, answer quality, model comparison
```

---

# License

[PolyForm Noncommercial License 1.0.0](LICENSE.md): free for noncommercial use; commercial use requires a separate license from the author. Contact: benschwemlein@gmail.com


