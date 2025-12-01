#!/usr/bin/env python3
import os
import sys
import textwrap
import json
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import requests
import chromadb
from chromadb.config import Settings

# Models
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

# Defaults
DEFAULT_INDEX_DIR = os.path.expanduser("~/dev/indexes/commerce_customer_aeo")
DEFAULT_REPO_ROOT = os.path.expanduser("~/dev/repos/commerce-customer-aeo")
DEFAULT_TOP_K = 16
DEFAULT_MAX_DIRECT_EMBED_CHARS = 4000


def embed_text(text: str):
    """
    Ask Ollama for an embedding of a single text chunk or query.

    Returns the embedding list or None on error.
    """
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        print(f"[embed_text] Error calling Ollama: {e}", file=sys.stderr)
        return None

    if not resp.ok:
        print(f"[embed_text] Ollama returned {resp.status_code}", file=sys.stderr)
        try:
            print(f"[embed_text] Body (first 400 chars): {resp.text[:400]!r}", file=sys.stderr)
        except Exception:
            pass
        return None

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[embed_text] Could not parse JSON from Ollama: {e}", file=sys.stderr)
        return None

    embedding = data.get("embedding")
    if embedding is None:
        print(f"[embed_text] No 'embedding' field in response: {data}")
        return None

    return embedding


def summarize_query(long_text: str, template: str):
    """
    Use the chat LLM to rewrite a long bug or log into a compact semantic query.

    The template can contain the placeholder <<BUG_TEXT>> which will be replaced
    with the full bug text.

    If there is an error, we fall back to the original long_text.
    """
    if "<<BUG_TEXT>>" in template:
        user_content = template.replace("<<BUG_TEXT>>", long_text)
    else:
        user_content = template + "\n\nBug text:\n" + long_text

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        resp = requests.post(url, json=payload)
    except requests.RequestException as e:
        print(f"[summarize_query] Error calling Ollama: {e}", file=sys.stderr)
        return long_text

    if not resp.ok:
        print(f"[summarize_query] Ollama returned {resp.status_code}", file=sys.stderr)
        print(f"[summarize_query] Body (first 400 chars): {resp.text[:400]!r}", file=sys.stderr)
        return long_text

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[summarize_query] Could not parse JSON: {e}", file=sys.stderr)
        return long_text

    summary = data["message"]["content"].strip()
    print(f"[gui_query] Summarized question to {len(summary)} chars for embedding.", file=sys.stderr)
    return summary


def chat_with_context(question: str, docs, metas, template: str):
    """
    Call the chat model with the bug description and retrieved snippets.

    The template can contain the placeholders:
        <<BUG_TEXT>>   replaced with the original question or bug text
        <<SNIPPETS>>   replaced with the formatted snippets
    """
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        path = meta.get("path", "<unknown>")
        chunk_idx = meta.get("chunk_index", "?")
        header = f"[Snippet {i} from {path} chunk {chunk_idx}]"
        context_parts.append(header + "\n" + doc)

    snippets_text = "\n\n".join(context_parts)

    prompt = template
    if "<<BUG_TEXT>>" in prompt:
        prompt = prompt.replace("<<BUG_TEXT>>", question)
    else:
        prompt = prompt + "\n\nBug description:\n" + question

    if "<<SNIPPETS>>" in prompt:
        prompt = prompt.replace("<<SNIPPETS>>", snippets_text)
    else:
        prompt = prompt + "\n\nRelevant snippets:\n" + snippets_text

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    resp = requests.post(url, json=payload)
    if not resp.ok:
        print(f"[chat_with_context] Ollama returned {resp.status_code}", file=sys.stderr)
        print(f"[chat_with_context] Body (first 400 chars): {resp.text[:400]!r}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()["message"]["content"]


class CodeSearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local Code Query")

        # list of {"meta": meta, "distance": dist}
        self.last_results = None
        self.last_index_dir = None
        self.last_repo_root = None

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.tab_query = ttk.Frame(notebook)
        self.tab_prompts = ttk.Frame(notebook)

        notebook.add(self.tab_query, text="Query")
        notebook.add(self.tab_prompts, text="Prompts")

        self._build_query_tab()
        self._build_prompts_tab()

        self._set_default_prompts()

    def _build_query_tab(self):
        params_frame = ttk.LabelFrame(self.tab_query, text="Parameters")
        params_frame.pack(fill="x", padx=8, pady=8)

        # Index dir
        ttk.Label(params_frame, text="Index directory:").grid(row=0, column=0, sticky="w")
        self.index_dir_var = tk.StringVar(value=DEFAULT_INDEX_DIR)
        self.index_dir_entry = ttk.Entry(params_frame, textvariable=self.index_dir_var, width=60)
        self.index_dir_entry.grid(row=0, column=1, sticky="we", padx=4)
        browse_idx_btn = ttk.Button(params_frame, text="Browse", command=self.browse_index_dir)
        browse_idx_btn.grid(row=0, column=2, padx=4)

        # Repo root
        ttk.Label(params_frame, text="Repo root:").grid(row=1, column=0, sticky="w")
        self.repo_root_var = tk.StringVar(value=DEFAULT_REPO_ROOT)
        self.repo_root_entry = ttk.Entry(params_frame, textvariable=self.repo_root_var, width=60)
        self.repo_root_entry.grid(row=1, column=1, sticky="we", padx=4)
        browse_repo_btn = ttk.Button(params_frame, text="Browse", command=self.browse_repo_root)
        browse_repo_btn.grid(row=1, column=2, padx=4)

        # Top K
        ttk.Label(params_frame, text="Top K:").grid(row=2, column=0, sticky="w")
        self.top_k_var = tk.StringVar(value=str(DEFAULT_TOP_K))
        self.top_k_entry = ttk.Entry(params_frame, textvariable=self.top_k_var, width=10)
        self.top_k_entry.grid(row=2, column=1, sticky="w", padx=4)

        # Max chars before summarizing
        ttk.Label(params_frame, text="Max chars before summarize:").grid(row=3, column=0, sticky="w")
        self.max_chars_var = tk.StringVar(value=str(DEFAULT_MAX_DIRECT_EMBED_CHARS))
        self.max_chars_entry = ttk.Entry(params_frame, textvariable=self.max_chars_var, width=10)
        self.max_chars_entry.grid(row=3, column=1, sticky="w", padx=4)

        params_frame.columnconfigure(1, weight=1)

        # Bug text frame
        bug_frame = ttk.LabelFrame(self.tab_query, text="Bug or Question")
        bug_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.bug_text = ScrolledText(bug_frame, wrap="word", height=10)
        self.bug_text.pack(fill="both", expand=True, padx=4, pady=4)

        btn_frame = ttk.Frame(bug_frame)
        btn_frame.pack(fill="x", padx=4, pady=4)

        load_btn = ttk.Button(btn_frame, text="Load from file", command=self.load_bug_from_file)
        load_btn.pack(side="left")

        run_btn = ttk.Button(btn_frame, text="Run query", command=self.run_query)
        run_btn.pack(side="right")

        # Output frame
        output_frame = ttk.LabelFrame(self.tab_query, text="Output")
        output_frame.pack(fill="both", expand=True, padx=8, pady=4)

        output_btn_frame = ttk.Frame(output_frame)
        output_btn_frame.pack(fill="x", padx=4, pady=2)

        save_output_btn = ttk.Button(output_btn_frame, text="Save log to file", command=self.save_output_to_file)
        save_output_btn.pack(side="left")

        view_files_btn = ttk.Button(output_btn_frame, text="View result files", command=self.show_files_window)
        view_files_btn.pack(side="right")

        # Summary / Answer box
        summary_frame = ttk.LabelFrame(output_frame, text="Summary / Answer")
        summary_frame.pack(fill="both", expand=True, padx=4, pady=(4, 2))

        self.summary_text = ScrolledText(summary_frame, wrap="word", height=8, state="normal")
        self.summary_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Log box
        log_frame = ttk.LabelFrame(output_frame, text="Log")
        log_frame.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        self.output_text = ScrolledText(log_frame, wrap="word", height=8, state="normal")
        self.output_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_prompts_tab(self):
        ctl_frame = ttk.Frame(self.tab_prompts)
        ctl_frame.pack(fill="x", padx=8, pady=4)

        load_prompts_btn = ttk.Button(ctl_frame, text="Load prompts from file", command=self.load_prompts_from_file)
        load_prompts_btn.pack(side="left", padx=4)

        save_prompts_btn = ttk.Button(ctl_frame, text="Save prompts to file", command=self.save_prompts_to_file)
        save_prompts_btn.pack(side="left", padx=4)

        sum_frame = ttk.LabelFrame(self.tab_prompts, text="Summarizer prompt template")
        sum_frame.pack(fill="both", expand=True, padx=8, pady=4)

        sum_label = ttk.Label(
            sum_frame,
            text="Use <<BUG_TEXT>> where the full bug description should go."
        )
        sum_label.pack(anchor="w", padx=4, pady=2)

        self.summarizer_text = ScrolledText(sum_frame, wrap="word", height=10)
        self.summarizer_text.pack(fill="both", expand=True, padx=4, pady=4)

        chat_frame = ttk.LabelFrame(self.tab_prompts, text="Answer prompt template")
        chat_frame.pack(fill="both", expand=True, padx=8, pady=4)

        chat_label = ttk.Label(
            chat_frame,
            text="Use <<BUG_TEXT>> for bug description and <<SNIPPETS>> for retrieved code snippets."
        )
        chat_label.pack(anchor="w", padx=4, pady=2)

        self.chat_prompt_text = ScrolledText(chat_frame, wrap="word", height=12)
        self.chat_prompt_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _set_default_prompts(self):
        summarizer_default = textwrap.dedent("""
        You are a senior engineer helping with code search.

        You will be given a long bug description, Jira ticket text, logs, and sometimes SQL queries.
        Your job is to rewrite it as a shorter query that preserves all important technical details
        needed to search the codebase.

        Long bug text:

        <<BUG_TEXT>>

        Task:
        - Rewrite this as a concise search query for a codebase.
        - Keep all important technical details: class names, endpoints, error messages,
          stack trace fragments, config keys, error codes, i18n string keys, table and column names,
          and SQL fragments.
        - Never remove or alter i18n keys, table or column names, or any text that looks like SQL.
        - Remove obvious noise.
        - Length target: a few sentences or a short paragraph.
        - Do not invent new information.
        """).strip()

        chat_default = textwrap.dedent("""
        You are a senior engineer working on a large Java and Angular based retail point of sale system.

        You will be given:
        - A bug description, which may include SQL, database results, UI behavior, and business context.
        - A set of code snippets retrieved as relevant to that bug.

        Your job:
        - Explain the likely root cause of the bug using the snippets as evidence.
        - Point to specific files, classes, and methods involved.
        - Explain what needs to be changed and where in order to fix the bug.
        - When possible, explain how to make the fix data driven so that future changes only require updating
          database or configuration entries rather than code.

        Bug description:
        <<BUG_TEXT>>

        Relevant code and documentation snippets:
        <<SNIPPETS>>

        Answer directly in terms of the bug.
        Separate clearly:
        - Facts that are directly supported by the snippets.
        - Hypotheses or guesses that go beyond the snippets.
        """).strip()

        self.summarizer_text.insert("1.0", summarizer_default)
        self.chat_prompt_text.insert("1.0", chat_default)

    def browse_index_dir(self):
        path = filedialog.askdirectory(initialdir=self.index_dir_var.get() or os.path.expanduser("~"))
        if path:
            self.index_dir_var.set(path)

    def browse_repo_root(self):
        path = filedialog.askdirectory(initialdir=self.repo_root_var.get() or os.path.expanduser("~"))
        if path:
            self.repo_root_var.set(path)

    def load_bug_from_file(self):
        path = filedialog.askopenfilename(
            title="Select bug text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")
            return

        self.bug_text.delete("1.0", "end")
        self.bug_text.insert("1.0", content)

    def append_output(self, text: str):
        self.output_text.insert("end", text + "\n")
        self.output_text.see("end")
        self.update_idletasks()

    def save_output_to_file(self):
        content = self.output_text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Info", "There is no log content to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"query_log_{ts}.txt"
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save log:\n{e}")
            return

        messagebox.showinfo("Saved", f"Log saved to:\n{path}")

    def load_prompts_from_file(self):
        path = filedialog.askopenfilename(
            title="Load prompts",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load prompts:\n{e}")
            return

        summarizer = data.get("summarizer")
        chat = data.get("chat")

        if not isinstance(summarizer, str) or not isinstance(chat, str):
            messagebox.showerror("Error", "Prompt file must be JSON with 'summarizer' and 'chat' string fields.")
            return

        self.summarizer_text.delete("1.0", "end")
        self.summarizer_text.insert("1.0", summarizer)

        self.chat_prompt_text.delete("1.0", "end")
        self.chat_prompt_text.insert("1.0", chat)

        messagebox.showinfo("Loaded", f"Prompts loaded from:\n{path}")

    def save_prompts_to_file(self):
        summarizer = self.summarizer_text.get("1.0", "end-1c")
        chat = self.chat_prompt_text.get("1.0", "end-1c")

        data = {
            "summarizer": summarizer,
            "chat": chat,
        }

        path = filedialog.asksaveasfilename(
            title="Save prompts",
            defaultextension=".json",
            initialfile="prompts.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save prompts:\n{e}")
            return

        messagebox.showinfo("Saved", f"Prompts saved to:\n{path}")

    def _compute_relative_scores(self, distances):
        """
        Compute per query relative scores 0 to 100.

        Smallest distance -> 100
        Largest distance  -> 0
        """
        if not distances:
            return []

        min_d = min(distances)
        max_d = max(distances)

        if max_d == min_d:
            return [100.0 for _ in distances]

        scores = []
        for d in distances:
            score = 100.0 * (max_d - d) / (max_d - min_d)
            if score < 0:
                score = 0.0
            if score > 100:
                score = 100.0
            scores.append(score)
        return scores

    def run_query(self):
        # Clear both summary and log
        self.output_text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")
        self.last_results = None

        index_dir = self.index_dir_var.get().strip()
        repo_root = self.repo_root_var.get().strip()
        bug = self.bug_text.get("1.0", "end-1c").strip()

        if not index_dir:
            messagebox.showerror("Error", "Index directory is required.")
            return
        if not bug:
            messagebox.showerror("Error", "Bug or question text is required.")
            return

        try:
            top_k = int(self.top_k_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Top K must be an integer.")
            return

        try:
            max_chars = int(self.max_chars_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Max chars must be an integer.")
            return

        if not os.path.isdir(index_dir):
            messagebox.showerror("Error", f"Index directory does not exist:\n{index_dir}")
            return

        if repo_root and not os.path.isdir(repo_root):
            messagebox.showerror("Error", f"Repo root directory does not exist:\n{repo_root}")
            return

        try:
            client = chromadb.PersistentClient(
                path=index_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            collections = client.list_collections()
            if not collections:
                messagebox.showerror(
                    "Error",
                    "No collections found in this index directory. "
                    "You may need to build an index first."
                )
                return

            if len(collections) > 1:
                self.append_output("[gui_query] Multiple collections found in this index directory.")
                self.append_output("[gui_query] Available collections:")
                for c in collections:
                    self.append_output(f"  {c.name}")
                self.append_output(f"[gui_query] Using first collection: {collections[0].name}")
            else:
                self.append_output(f"[gui_query] Using collection: {collections[0].name}")

            collection = client.get_collection(collections[0].name)

        except Exception as e:
            messagebox.showerror("Error", f"Could not open collection from index directory:\n{e}")
            return

        self.append_output(f"[gui_query] Using index directory: {index_dir}")

        query_for_embedding = bug
        if len(bug) > max_chars:
            self.append_output(f"[gui_query] Bug text is {len(bug)} chars, summarizing before embedding...")
            sum_template = self.summarizer_text.get("1.0", "end-1c")
            query_for_embedding = summarize_query(bug, sum_template)

        self.append_output("[gui_query] Embedding query text...")
        q_embedding = embed_text(query_for_embedding)
        if q_embedding is None:
            messagebox.showerror("Error", "Failed to obtain embedding from Ollama.")
            return

        self.append_output(f"[gui_query] Querying index for top {top_k} snippets...")

        res = collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_list = res.get("documents", [[]])
        metas_list = res.get("metadatas", [[]])
        dist_list = res.get("distances", [[]])

        if not docs_list or not docs_list[0]:
            self.append_output("[gui_query] No relevant snippets found in the index.")
            return

        docs = docs_list[0]
        metas = metas_list[0]
        dists = dist_list[0]

        scores = self._compute_relative_scores(dists)

        self.last_results = [
            {"meta": meta, "distance": dist}
            for meta, dist in zip(metas, dists)
        ]
        self.last_index_dir = index_dir
        self.last_repo_root = repo_root

        count = len(metas)
        self.append_output(f"Retrieved {count} snippet chunks.")

        self.append_output("Using snippets from:")
        for idx, (meta, dist, score) in enumerate(zip(metas, dists, scores), start=1):
            path = meta.get("path", "<unknown>")
            chunk_idx = meta.get("chunk_index", "?")
            self.append_output(
                f"  [{idx:02d}] {score:5.1f}%  {path} (chunk {chunk_idx}, distance {dist:.4f})"
            )

        best_score = max(scores) if scores else 0.0
        if best_score < 15.0:
            self.append_output("")
            self.append_output("[gui_query] WARNING: All retrieved snippets have very low relative scores.")
            self.append_output("[gui_query] The answer may rely mostly on the bug text and not on code context.")

        self.append_output("")
        self.append_output("[gui_query] Asking LLM with retrieved context...")

        chat_template = self.chat_prompt_text.get("1.0", "end-1c")
        try:
            answer = chat_with_context(bug, docs, metas, chat_template)
        except Exception as e:
            messagebox.showerror("Error", f"Error calling chat model:\n{e}")
            return

        self.summary_text.insert("1.0", answer)
        self.summary_text.see("1.0")

        self.append_output("\n=== ANSWER ===\n")
        self.append_output(answer)

    def show_files_window(self):
        if not self.last_results:
            messagebox.showinfo("Info", "No query results available yet.")
            return

        repo_root = self.last_repo_root or ""
        win = tk.Toplevel(self)
        win.title("Query result files")

        info_label = ttk.Label(
            win,
            text=f"Repo root: {repo_root or '(not set, paths are relative)'}"
        )
        info_label.pack(fill="x", padx=8, pady=4)

        listbox = tk.Listbox(win, selectmode="browse")
        listbox.pack(fill="both", expand=True, padx=8, pady=4)

        files_for_rows = []

        distances = [item["distance"] for item in self.last_results]
        scores_all = self._compute_relative_scores(distances)

        best_by_path = {}
        counts_by_path = {}

        for item, score in zip(self.last_results, scores_all):
            meta = item["meta"]
            path = meta.get("path", "<unknown>")
            if path not in best_by_path or score > best_by_path[path]:
                best_by_path[path] = score
            counts_by_path[path] = counts_by_path.get(path, 0) + 1

        sorted_items = sorted(best_by_path.items(), key=lambda x: x[1], reverse=True)

        for idx, (path, score) in enumerate(sorted_items, start=1):
            count = counts_by_path.get(path, 1)
            listbox.insert("end", f"[{idx:02d}] {score:5.1f}% ({count} chunks)  {path}")
            files_for_rows.append(path)

        def on_open_selected(event=None):
            selection = listbox.curselection()
            if not selection:
                return
            idx_sel = selection[0]
            rel_path = files_for_rows[idx_sel]

            if repo_root:
                full_path = os.path.join(repo_root, rel_path)
            else:
                full_path = rel_path

            if not os.path.isfile(full_path):
                messagebox.showerror("Error", f"File does not exist:\n{full_path}")
                return

            try:
                subprocess.Popen(["open", full_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")

        listbox.bind("<Double-Button-1>", on_open_selected)

        open_btn = ttk.Button(win, text="Open selected file", command=on_open_selected)
        open_btn.pack(padx=8, pady=4)


def main():
    app = CodeSearchGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
