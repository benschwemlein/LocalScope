

---

# **Local RAG LLM**

Local RAG LLM is a desktop application for running Retrieval Augmented Generation against any source code repository using only local models. The application indexes a repository into a vector database, retrieves relevant code snippets for any natural language question, and uses a local LLM to explain how the code works or why a problem occurs. All processing remains on your machine.

---

# Overview

Local RAG LLM provides two workflows:

### Indexing

The application scans a repository, chunks each file, embeds the chunks using a local embedding model, and stores them in a ChromaDB index.

### Querying

A user provides a question, bug report, log output, or general investigation text. The application embeds the text, retrieves the most relevant code chunks, and sends them along with the question to a local chat model for explanation.

---

# Features

* Fully local RAG pipeline
* Vector search using ChromaDB
* Embedding and chat models provided by Ollama
* Semantic retrieval across code and documentation
* Interactive GUI with tabs for Query, Indexing, Settings, and Prompts
* Click-to-open file results
* Editable prompt templates stored in JSON
* Clear error handling and help text
* No network communication beyond Ollama running locally

---

# Requirements

### Python

* Python 3.10 or later
* Tkinter

### Python packages

```bash
pip install chromadb requests
```

### Ollama

Download Ollama:
[https://ollama.com/download](https://ollama.com/download)

Install at least one embedding model and one chat model:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

Ollama must be running at:

```
http://localhost:11434
```

unless changed in the Settings tab.

---

# Installation

Clone the project and install dependencies:

```bash
git clone <your repo>
cd local-rag-llm
pip install -r requirements.txt
```

If requirements.txt is not used:

```bash
pip install chromadb requests
```

Run the application:

```bash
python app.py
```

---

# First Time Setup

1. Confirm Ollama is installed:

```bash
ollama list
```

2. Open the Settings tab and select:

   * Ollama server URL
   * Embedding model
   * Chat model

3. Open the Index tab:

   * Select a repository root
   * Select or accept an index directory
   * Click “Index Repository”

Once indexing completes, the repository is ready for semantic search.

---

# Indexing a Repository

The indexer walks the repository, skipping directories such as:

* .git
* node_modules
* build
* target
* dist

Supported file types include:

* Java and Kotlin
* JavaScript and TypeScript
* HTML and CSS
* JSON and YAML
* Markdown and text files

Large files are skipped for speed.
Each file is split into overlapping text chunks, embedded, and stored with metadata including path and chunk index.

Changing the embedding model requires re-indexing the repository.

---

# Running Queries

The Query tab includes settings for:

* Index directory
* Repository root
* Number of results to retrieve
* Maximum characters before summarization
* Input text (bug report, logs, question)
* Response output area

Steps:

1. Enter or paste your text
2. Click “Run Query”

The application:

* Summarizes long text using the Summarizer prompt
* Embeds the summarized text
* Retrieves the top results from the vector index
* Sends the question and snippets to the LLM
* Displays the explanation in the Response section

---

# Prompts

Prompts control how the models interpret your input.

Two prompts are used:

### Summarizer Prompt

Reduces long investigation text into a short semantic query. Must contain `<<BUG_TEXT>>`.

### Chat Prompt

Explains retrieved code, describes system behavior, identifies possible root causes, and separates evidence from hypothesis. Must contain both `<<BUG_TEXT>>` and `<<SNIPPETS>>`.

Prompts can be edited, saved, or loaded from JSON.
If either prompt is empty, the application will not run queries.

---

# Settings

The Settings tab controls:

* Ollama server URL
* Embedding model name (dropdown)
* Chat model name (dropdown)

Each setting includes a help button explaining its purpose and linking to the Ollama download page if models are missing.

The embedding model determines vector index compatibility.
The chat model affects the quality and depth of explanations.

---

# Opening Files

After a query, the application lists files grouped by relevance.
Double-clicking a file opens it using:

* `open` on macOS
* `start` on Windows
* `xdg-open` on Linux

If a repository root is specified, paths resolve relative to it.

---

# Tips for Effective Use

* Provide detailed input text for more accurate retrieval
* Keep prompts explicit about what information matters
* Retrieve 8–16 results for debugging
* Retrieve 20–30 results when exploring unfamiliar code
* Re-index if you change the embedding model
* Ensure Ollama is running before launching the application

---

# Troubleshooting

### No models appear in Settings

Ollama is not running or the server URL is incorrect.

### Embeddings fail

The embedding model is missing or named incorrectly.

### No results in query

The index directory is incorrect or the repository has not been indexed.

### File fails to open

The repository root is wrong or files have moved.

### Query does not run

Summarizer or Chat prompt is empty.

---

# Project Structure

```
local-rag-llm/
  app.py
  config.py

  gui/
    query_tab.py
    prompts_tab.py
    index_tab.py
    settings_tab.py

  querying/
    query_engine.py

  indexing/
    indexer.py

  README.md
```

---

