

# **Local RAG LLM**

A desktop application for running Retrieval Augmented Generation against any source code repository using only local models. It indexes a repository into a vector database, retrieves the most relevant code for a natural language question, bug report, or stack trace, and uses a local LLM to explain how the code works or why a problem occurs. **Everything runs on your machine, no cloud calls, no code leaves your laptop.**

### Why I built it

Ramping onto a large, unfamiliar codebase is slow, and sending proprietary source to a cloud LLM is often off the table. I built this to ask plain-English questions about a codebase ("where is the credit-card application flow handled?", "what calls this service?") and get back the exact files plus an explanation, entirely offline. I used it to onboard onto a large commercial codebase faster than the rest of the team.

### Technical highlights

* **AST-aware chunking** — splits code at logical boundaries (functions, classes, methods) instead of arbitrary character offsets, so retrieved snippets are coherent units of code. Uses the `astchunk` library for multi-language support with a Python-`ast` fallback.
* **Incremental indexing** — re-indexes only changed files instead of rebuilding the whole vector store.
* **Fully local and private** — ChromaDB for vector search, Ollama for embeddings and chat. No API keys, no external services.
* **Benchmarking test suite** — a pytest suite that compares embedding models, chunking strategies, overlap tuning, top-k, and answer quality, so model and chunking choices are measured, not guessed.
* **GUI and CLI** — a Tkinter desktop app for interactive use and a CLI for scripting.

---

# Quick Start

1. Install **Python 3.10+**

2. Install Python dependencies:

   ```bash
   pip install chromadb requests
   ```

3. Install **Ollama** and make sure it is running  
   https://ollama.com/download

4. Clone this repo:

   ```bash
   git clone https://github.com/benschwemlein/code_llm.git
   cd code_llm
   pip install chromadb requests
   ```

5. Start the application:

   ```bash
   python app.py
   ```

6. Open the **Settings** tab  
   Confirm Ollama shows a **green status icon**, select embedding and chat models, or download one from the dropdown.

7. Open the **Index** tab and index your repository.


You are ready to query code.

---

# Overview

Local RAG LLM provides two workflows:

### Indexing

The indexer scans a repository, chunks supported files, embeds the chunks using a local embedding model, and stores vectors in a ChromaDB index.

### Querying

The user provides a question, bug report, log output, or general investigation text. The system embeds the text, retrieves the most relevant code chunks, and sends them along with the prompt to a local chat model for explanation.

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
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

3. Ensure Ollama is running (shows green in Settings tab):

```
http://localhost:11434
```

---

# Installation

Clone the project:

```bash
git clone https://github.com/benschwemlein/code_llm.git
cd code_llm
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
* Repository root (optional, for file opening)
* Number of snippets to retrieve
* Maximum characters for summarization
* The question / bug report text
* Output display mode

### Steps

1. Type or paste your investigation text
2. Click **Run Query**

The engine:

1. Summarizes long input using your Summarizer prompt
2. Embeds the summarized text
3. Retrieves the top matching code chunks
4. Injects snippets and your question into the Chat prompt
5. Runs your local LLM to produce an answer
6. Displays the result with clickable file paths

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
code_llm/
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
    query_engine.py           # embed -> retrieve -> summarize -> LLM answer

  ollama_manager/             # local model management + downloads
    download_manager.py

  cli/
    rag_query.py              # command-line query interface

  test_suite/                 # pytest benchmarks: embedding models, chunking,
                              # overlap, top-k, answer quality, model comparison
```

---

# License

See [LICENSE_AGREEMENT.md](LICENSE_AGREEMENT.md).


