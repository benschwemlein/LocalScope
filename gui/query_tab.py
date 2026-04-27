import os
import sys
import threading
import subprocess
import config
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


class QueryTab(ttk.Frame):
    """
    Query tab that performs code search and LLM augmented answering.

    It expects two callables:

        get_summarizer_prompt() -> str
        get_chat_prompt() -> str
    """

    def __init__(self, parent, settings_mgr, get_summarizer_prompt, get_chat_prompt):
        super().__init__(parent)

        self.settings_mgr = settings_mgr
        self.get_summarizer_prompt = get_summarizer_prompt
        self.get_chat_prompt = get_chat_prompt

        self.last_results = None
        self.last_index_dir = None
        self.last_repo_root = None

        self._current_thread = None
        self._cancel_event = None
        self._loading_ui = True

        self._build_query_tab()
        self._wire_autosave()

        self._loading_ui = False

    def _build_query_tab(self):
        s = self.settings_mgr.data.get("query_tab") or {}

        # Parameters section
        params_frame = ttk.LabelFrame(self, text="Parameters")
        params_frame.pack(fill="x", padx=8, pady=8)

        # Index directory
        ttk.Label(params_frame, text="Index directory:").grid(row=0, column=0, sticky="w")
        self.index_dir_var = tk.StringVar(value=s.get("index_dir", DEFAULT_INDEX_DIR))
        self.index_dir_entry = ttk.Entry(params_frame, textvariable=self.index_dir_var, width=60)
        self.index_dir_entry.grid(row=0, column=1, sticky="we", padx=4)
        browse_idx_btn = ttk.Button(params_frame, text="Browse", command=self.browse_index_dir)
        browse_idx_btn.grid(row=0, column=2, padx=4)
        idx_help_btn = ttk.Button(params_frame, text="?", width=2, command=self._help_index_dir)
        idx_help_btn.grid(row=0, column=3, sticky="e", padx=2)

        # Repo root
        ttk.Label(params_frame, text="Repository root (optional):").grid(row=1, column=0, sticky="w")
        self.repo_root_var = tk.StringVar(value=s.get("repo_root", DEFAULT_REPO_ROOT))
        self.repo_root_entry = ttk.Entry(params_frame, textvariable=self.repo_root_var, width=60)
        self.repo_root_entry.grid(row=1, column=1, sticky="we", padx=4)
        browse_repo_btn = ttk.Button(params_frame, text="Browse", command=self.browse_repo_root)
        browse_repo_btn.grid(row=1, column=2, padx=4)
        repo_help_btn = ttk.Button(params_frame, text="?", width=2, command=self._help_repo_root)
        repo_help_btn.grid(row=1, column=3, sticky="e", padx=2)

        # Number of results (Top K)
        ttk.Label(params_frame, text="Number of results to retrieve:").grid(row=2, column=0, sticky="w")
        self.top_k_var = tk.StringVar(value=str(s.get("top_k", DEFAULT_TOP_K)))
        self.top_k_entry = ttk.Entry(params_frame, textvariable=self.top_k_var, width=10)
        self.top_k_entry.grid(row=2, column=1, sticky="w", padx=4)
        topk_help_btn = ttk.Button(params_frame, text="?", width=2, command=self._help_top_k)
        topk_help_btn.grid(row=2, column=3, sticky="e", padx=2)

        # Max chars before summarizing
        ttk.Label(params_frame, text="Summarize text longer than (characters):").grid(row=3, column=0, sticky="w")
        self.max_chars_var = tk.StringVar(value=str(s.get("max_chars", DEFAULT_MAX_DIRECT_EMBED_CHARS)))
        self.max_chars_entry = ttk.Entry(params_frame, textvariable=self.max_chars_var, width=10)
        self.max_chars_entry.grid(row=3, column=1, sticky="w", padx=4)
        maxchars_help_btn = ttk.Button(params_frame, text="?", width=2, command=self._help_max_chars)
        maxchars_help_btn.grid(row=3, column=3, sticky="e", padx=2)

        params_frame.columnconfigure(1, weight=1)

        # Bug text frame
        bug_frame = ttk.LabelFrame(self, text="Bug or Question")
        bug_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.bug_text = ScrolledText(bug_frame, wrap="word", height=10)
        self.bug_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Load saved bug text if present
        saved_bug = (s.get("bug_text") or "")
        if saved_bug.strip():
            self.bug_text.delete("1.0", "end")
            self.bug_text.insert("1.0", saved_bug)

        btn_frame = ttk.Frame(bug_frame)
        btn_frame.pack(fill="x", padx=4, pady=4)

        load_btn = ttk.Button(btn_frame, text="Load from file", command=self.load_bug_from_file)
        load_btn.pack(side="left")

        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self._cancel_query, state="disabled")
        self.cancel_btn.pack(side="right", padx=(4, 0))

        self.run_btn = ttk.Button(btn_frame, text="Run query", command=self.run_query)
        self.run_btn.pack(side="right")

        # Status + progress indicator
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", padx=8, pady=(0, 4))

        self.status_label = ttk.Label(status_frame, text="Idle")
        self.status_label.pack(side="left")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True, padx=(8, 0))

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

        # Ensure settings file has all fields for this tab
        self._save_query_to_settings(save_immediately=False)

    def _wire_autosave(self):
        def on_any_change(*_):
            self._save_query_to_settings(save_immediately=False)

        for v in [self.index_dir_var, self.repo_root_var, self.top_k_var, self.max_chars_var]:
            v.trace_add("write", on_any_change)

        def on_bug_modified(event=None):
            if self.bug_text.edit_modified():
                self.bug_text.edit_modified(False)
                self._save_query_to_settings(save_immediately=False)

        self.bug_text.bind("<<Modified>>", on_bug_modified)

    def _save_query_to_settings(self, save_immediately: bool):
        if getattr(self, "_loading_ui", False):
            return

        # store raw strings to avoid breaking on partially typed numbers
        self.settings_mgr.data["query_tab"] = {
            "index_dir": self.index_dir_var.get(),
            "repo_root": self.repo_root_var.get(),
            "top_k": self._safe_int(self.top_k_var.get(), DEFAULT_TOP_K),
            "max_chars": self._safe_int(self.max_chars_var.get(), DEFAULT_MAX_DIRECT_EMBED_CHARS),
            "bug_text": self.bug_text.get("1.0", "end-1c"),
        }

        if save_immediately:
            self.settings_mgr.save_now()
        else:
            self.settings_mgr.save_soon()

    @staticmethod
    def _safe_int(raw: str, fallback: int) -> int:
        try:
            v = int(str(raw).strip())
            return v
        except Exception:
            return fallback

    # Help popups for parameters

    def _help_index_dir(self):
        messagebox.showinfo(
            "Index directory",
            "Folder where the Chroma vector database is stored.\n\n"
            "This directory is created when you index a repository from the Index tab.\n"
            "Queries read embeddings from here. If you delete or move it, you need to\n"
            "re index the repository."
        )

    def _help_repo_root(self):
        messagebox.showinfo(
            "Repository root",
            "Optional root folder of the source repository you indexed.\n\n"
            "If this is set, double clicking a result path will try to open the\n"
            "underlying file from that repository. If it is empty, you will still\n"
            "see file paths, but double click may not open anything."
        )

    def _help_top_k(self):
        messagebox.showinfo(
            "Number of results to retrieve",
            "How many relevant code snippets to fetch from the index for each query.\n\n"
            "Typical values:\n"
            "  8 to 16  fast and focused\n"
            "  20 to 30 more thorough but slower\n\n"
            "Higher values give the model more context but can make answers slower\n"
            "and sometimes less precise."
        )

    def _help_max_chars(self):
        messagebox.showinfo(
            "Summarize text longer than",
            "If your bug description is longer than this many characters, the app\n"
            "asks the chat model to summarize it into a shorter query before\n"
            "embedding.\n\n"
            "Recommended range: 3000 to 6000 characters.\n"
            "Set a very large value if you prefer no summarization."
        )

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
        self._save_query_to_settings(save_immediately=False)

    # Thread safe logging callback used by query_engine.run_query
    def _log(self, text: str):
        self.after(0, lambda: self._append_output(text))

    def _append_output(self, text: str):
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

    def _cancel_query(self):
        if self._cancel_event:
            self._cancel_event.set()
        self.cancel_btn.config(state="disabled")
        self.status_label.config(text="Cancelling...")

    def run_query(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Info", "A query is already running.")
            return

        # Clear response and last results
        self.response_text.delete("1.0", "end")
        self.last_results = None

        index_dir = self.index_dir_var.get().strip()
        repo_root = self.repo_root_var.get().strip()
        bug = self.bug_text.get("1.0", "end-1c")

        # Ensure runtime config matches autosaved Settings tab values
        st = self.settings_mgr.data.get("settings_tab") or {}
        url = (st.get("ollama_url") or "").strip()
        embed = (st.get("embed_model") or "").strip()
        chat = (st.get("chat_model") or "").strip()
        if url:
            config.OLLAMA_URL = url
        if embed:
            config.EMBED_MODEL = embed
        if chat:
            config.CHAT_MODEL = chat

        # Show runtime model selection in the query output
        try:
            import config as _cfg
            self._append_output("[runtime]")
            self._append_output(f"Ollama URL: {_cfg.OLLAMA_URL}")
            self._append_output(f"Embedding model: {_cfg.EMBED_MODEL}")
            self._append_output(f"Chat model: {_cfg.CHAT_MODEL}")
            self._append_output("")
        except Exception:
            pass

        try:
            top_k = int(self.top_k_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Number of results must be an integer.")
            return

        try:
            max_chars = int(self.max_chars_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Summarize threshold must be an integer.")
            return

        # Save settings right before running so they reflect what ran
        self._save_query_to_settings(save_immediately=False)

        # Get prompts from Prompts tab
        sum_template = (self.get_summarizer_prompt() or "").strip()
        chat_template = (self.get_chat_prompt() or "").strip()

        if not sum_template:
            messagebox.showerror(
                "Error",
                "Summarizer prompt is empty.\n\n"
                "Open the Prompts tab and provide a summarizer prompt before running a query.",
            )
            return

        if not chat_template:
            messagebox.showerror(
                "Error",
                "Answer prompt is empty.\n\n"
                "Open the Prompts tab and provide an answer prompt before running a query.",
            )
            return

        # Disable UI and start progress indicator
        self._cancel_event = threading.Event()
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.status_label.config(text="Running query...")
        self.progress.start(10)

        cancel_event = self._cancel_event

        def on_token(token: str):
            self.after(0, lambda t=token: self._append_token(t))

        def worker():
            try:
                result = run_query(
                    bug_text=bug,
                    index_dir=index_dir,
                    repo_root=repo_root or "",
                    top_k=top_k,
                    max_chars=max_chars,
                    summarizer_template=sum_template,
                    chat_template=chat_template,
                    log=self._log,
                    cancel_event=cancel_event,
                    token_callback=on_token,
                )
            except Exception as e:
                cancelled = cancel_event.is_set()
                self.after(0, lambda err=e, c=cancelled: self._on_query_error(err, c))
                return

            self.after(0, lambda: self._on_query_done(result, index_dir, repo_root))

        self._current_thread = threading.Thread(target=worker, daemon=True)
        self._current_thread.start()

    def _append_token(self, token: str):
        self.response_text.insert("end", token)
        self.response_text.see("end")
        self.update_idletasks()

    def _on_query_error(self, error: Exception, cancelled: bool = False):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if cancelled:
            self.status_label.config(text="Cancelled")
        else:
            self.status_label.config(text="Failed")
            messagebox.showerror("Error", str(error))

    def _on_query_done(self, result: dict, index_dir: str, repo_root: str):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        cancelled = self._cancel_event and self._cancel_event.is_set()
        self.status_label.config(text="Cancelled" if cancelled else "Done")

        metas = result["metas"]
        distances = result["distances"]
        scores = result["scores"]

        self.last_results = [
            {"meta": meta, "distance": dist, "score": score}
            for meta, dist, score in zip(metas, distances, scores)
        ]
        self.last_index_dir = index_dir
        self.last_repo_root = repo_root

        # Answer was already streamed token-by-token; append clickable file list
        self._append_file_links(repo_root)

    def _append_file_links(self, repo_root: str):
        if not self.last_results:
            return

        self.response_text.insert("end", "\n\n=== RETRIEVED FILES ===\n")
        self.response_text.tag_config("file_link", foreground="blue", underline=True)

        for i, item in enumerate(self.last_results, 1):
            meta  = item["meta"]
            score = item["score"]
            rel_path = meta.get("source", meta.get("path", "<unknown>"))
            full_path = os.path.join(repo_root, rel_path) if repo_root else rel_path

            line = f"[{i:02d}] {score:5.1f}%  {rel_path}\n"
            tag = f"link_{i}"
            self.response_text.tag_config(tag, foreground="blue", underline=True)
            self.response_text.tag_bind(
                tag, "<Button-1>",
                lambda e, p=full_path: open_file_cross_platform(p) if os.path.isfile(p) else None,
            )
            self.response_text.tag_bind(tag, "<Enter>", lambda e: self.response_text.config(cursor="hand2"))
            self.response_text.tag_bind(tag, "<Leave>", lambda e: self.response_text.config(cursor=""))
            self.response_text.insert("end", line, tag)

        self.response_text.see("end")

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
            path = meta.get("source", meta.get("path", "<unknown>"))
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
