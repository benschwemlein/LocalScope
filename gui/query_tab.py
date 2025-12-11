import os
import sys
import textwrap
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_REPO_ROOT,
    DEFAULT_TOP_K,
    DEFAULT_MAX_DIRECT_EMBED_CHARS,
)
from querying.query_engine import run_query


def open_file_cross_platform(path: str):
    """
    Open a file with the default system handler on macOS, Windows or Linux.
    """
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            subprocess.Popen(["cmd", "/c", "start", "", path], shell=True)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open file:\n{e}")


DEFAULT_SUMMARIZER_PROMPT = textwrap.dedent("""
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

DEFAULT_CHAT_PROMPT = textwrap.dedent("""
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


class QueryTab(ttk.Frame):
    """
    Query tab that performs code search and LLM augmented answering.

    It expects two callables:

        get_summarizer_prompt() -> str
        get_chat_prompt() -> str
    """

    def __init__(self, parent, get_summarizer_prompt, get_chat_prompt):
        super().__init__(parent)

        self.get_summarizer_prompt = get_summarizer_prompt
        self.get_chat_prompt = get_chat_prompt

        # Stores a list of { "meta": ..., "distance": float, "score": float }
        self.last_results = None
        self.last_index_dir = None
        self.last_repo_root = None

        self._build_query_tab()

    def _build_query_tab(self):
        params_frame = ttk.LabelFrame(self, text="Parameters")
        params_frame.pack(fill="x", padx=8, pady=8)

        ttk.Label(params_frame, text="Index directory:").grid(row=0, column=0, sticky="w")
        self.index_dir_var = tk.StringVar(value=DEFAULT_INDEX_DIR)
        self.index_dir_entry = ttk.Entry(params_frame, textvariable=self.index_dir_var, width=60)
        self.index_dir_entry.grid(row=0, column=1, sticky="we", padx=4)
        browse_idx_btn = ttk.Button(params_frame, text="Browse", command=self.browse_index_dir)
        browse_idx_btn.grid(row=0, column=2, padx=4)

        ttk.Label(params_frame, text="Repo root:").grid(row=1, column=0, sticky="w")
        self.repo_root_var = tk.StringVar(value=DEFAULT_REPO_ROOT)
        self.repo_root_entry = ttk.Entry(params_frame, textvariable=self.repo_root_var, width=60)
        self.repo_root_entry.grid(row=1, column=1, sticky="we", padx=4)
        browse_repo_btn = ttk.Button(params_frame, text="Browse", command=self.browse_repo_root)
        browse_repo_btn.grid(row=1, column=2, padx=4)

        ttk.Label(params_frame, text="Top K:").grid(row=2, column=0, sticky="w")
        self.top_k_var = tk.StringVar(value=str(DEFAULT_TOP_K))
        self.top_k_entry = ttk.Entry(params_frame, textvariable=self.top_k_var, width=10)
        self.top_k_entry.grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(params_frame, text="Max chars before summarize:").grid(row=3, column=0, sticky="w")
        self.max_chars_var = tk.StringVar(value=str(DEFAULT_MAX_DIRECT_EMBED_CHARS))
        self.max_chars_entry = ttk.Entry(params_frame, textvariable=self.max_chars_var, width=10)
        self.max_chars_entry.grid(row=3, column=1, sticky="w", padx=4)

        params_frame.columnconfigure(1, weight=1)

        bug_frame = ttk.LabelFrame(self, text="Bug or Question")
        bug_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.bug_text = ScrolledText(bug_frame, wrap="word", height=10)
        self.bug_text.pack(fill="both", expand=True, padx=4, pady=4)

        btn_frame = ttk.Frame(bug_frame)
        btn_frame.pack(fill="x", padx=4, pady=4)

        load_btn = ttk.Button(btn_frame, text="Load from file", command=self.load_bug_from_file)
        load_btn.pack(side="left")

        run_btn = ttk.Button(btn_frame, text="Run query", command=self.run_query)
        run_btn.pack(side="right")

        # Single combined response area
        output_frame = ttk.LabelFrame(self, text="Response")
        output_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.response_text = ScrolledText(output_frame, wrap="word", height=16, state="normal")
        self.response_text.pack(fill="both", expand=True, padx=4, pady=4)

        output_btn_frame = ttk.Frame(self)
        output_btn_frame.pack(fill="x", padx=8, pady=(0, 4))

        save_output_btn = ttk.Button(
            output_btn_frame,
            text="Save response to file",
            command=self.save_output_to_file,
        )
        save_output_btn.pack(side="left")

        view_files_btn = ttk.Button(output_btn_frame, text="View result files", command=self.show_files_window)
        view_files_btn.pack(side="right")

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
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
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
        self.response_text.insert("end", text + "\n")
        self.response_text.see("end")
        self.update_idletasks()

    def save_output_to_file(self):
        content = self.response_text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Info", "There is no response content to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"query_response_{ts}.txt"
        path = filedialog.asksaveasfilename(
            title="Save response",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save response:\n{e}")
            return

        messagebox.showinfo("Saved", f"Response saved to:\n{path}")

    def run_query(self):
        # Clear response and last results
        self.response_text.delete("1.0", "end")
        self.last_results = None

        index_dir = self.index_dir_var.get().strip()
        repo_root = self.repo_root_var.get().strip()
        bug = self.bug_text.get("1.0", "end-1c")

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

        sum_template = self.get_summarizer_prompt() or DEFAULT_SUMMARIZER_PROMPT
        chat_template = self.get_chat_prompt() or DEFAULT_CHAT_PROMPT

        try:
            result = run_query(
                bug_text=bug,
                index_dir=index_dir,
                repo_root=repo_root or "",
                top_k=top_k,
                max_chars=max_chars,
                summarizer_template=sum_template,
                chat_template=chat_template,
                log=self.append_output,
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        answer = result["answer"]
        docs = result["docs"]
        metas = result["metas"]
        distances = result["distances"]
        scores = result["scores"]

        self.last_results = [
            {"meta": meta, "distance": dist, "score": score}
            for meta, dist, score in zip(metas, distances, scores)
        ]
        self.last_index_dir = index_dir
        self.last_repo_root = repo_root

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
            text=f"Repo root: {repo_root or '(not set, paths are relative)'}",
        )
        info_label.pack(fill="x", padx=8, pady=4)

        listbox = tk.Listbox(win, selectmode="browse")
        listbox.pack(fill="both", expand=True, padx=8, pady=4)

        files_for_rows: list[str] = []

        best_by_path: dict[str, float] = {}
        counts_by_path: dict[str, int] = {}

        for item in self.last_results:
            meta = item["meta"]
            score = item["score"]
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

            open_file_cross_platform(full_path)

        listbox.bind("<Double-Button-1>", on_open_selected)

        open_btn = ttk.Button(win, text="Open selected file", command=on_open_selected)
        open_btn.pack(padx=8, pady=4)
