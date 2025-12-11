import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from config import DEFAULT_INDEX_DIR, DEFAULT_REPO_ROOT, DEFAULT_COLLECTION_NAME
from indexing.indexer import (
    index_repo,
    INDEX_EXTS,
    CHARS_PER_CHUNK,
    CHUNK_OVERLAP,
    MAX_FILE_BYTES,
)


class IndexTab(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self._current_thread = None
        self._ext_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self):
        # Parameters frame
        params_frame = ttk.LabelFrame(self, text="Index parameters")
        params_frame.pack(fill="x", padx=8, pady=8)

        # Repo root
        ttk.Label(params_frame, text="Repo root:").grid(row=0, column=0, sticky="w")
        self.repo_root_var = tk.StringVar(value=DEFAULT_REPO_ROOT)
        repo_entry = ttk.Entry(params_frame, textvariable=self.repo_root_var, width=60)
        repo_entry.grid(row=0, column=1, sticky="we", padx=4)
        browse_repo_btn = ttk.Button(params_frame, text="Browse", command=self.browse_repo_root)
        browse_repo_btn.grid(row=0, column=2, padx=4)
        repo_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Repo root",
                (
                    "This is the top folder of the source tree you want to index.\n\n"
                    "The indexer will walk this directory and all of its child folders and read files "
                    "that match the selected file types.\n\n"
                    "Example:\n"
                    "  C:/dev/myproject\n"
                    "  /home/user/code/myproject\n\n"
                    "If you point this at a very large monorepo the first run may take a while."
                ),
            ),
        )
        repo_help_btn.grid(row=0, column=3, padx=2)

        # Index directory
        ttk.Label(params_frame, text="Index directory:").grid(row=1, column=0, sticky="w")
        self.index_dir_var = tk.StringVar(value=DEFAULT_INDEX_DIR)
        index_entry = ttk.Entry(params_frame, textvariable=self.index_dir_var, width=60)
        index_entry.grid(row=1, column=1, sticky="we", padx=4)
        browse_index_btn = ttk.Button(params_frame, text="Browse", command=self.browse_index_dir)
        browse_index_btn.grid(row=1, column=2, padx=4)
        index_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Index directory",
                (
                    "Folder where the Chroma database files are stored.\n\n"
                    "Everything that the indexer computes ends up here. You can keep separate indexes "
                    "for different projects by using different index folders or different collection names.\n\n"
                    "You should choose a location that is local and writable. If you delete this folder "
                    "you lose the index but your source code is unchanged."
                ),
            ),
        )
        index_help_btn.grid(row=1, column=3, padx=2)

        # Collection name
        ttk.Label(params_frame, text="Collection name:").grid(row=2, column=0, sticky="w")
        self.collection_var = tk.StringVar(value=DEFAULT_COLLECTION_NAME)
        collection_entry = ttk.Entry(params_frame, textvariable=self.collection_var, width=40)
        collection_entry.grid(row=2, column=1, sticky="w", padx=4)
        collection_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Collection name",
                (
                    "Name of the collection inside the Chroma index directory.\n\n"
                    "You can reuse the same index directory for many repositories by giving each "
                    "one a different collection name. Each run of the indexer will drop and recreate "
                    "the collection with the given name.\n\n"
                    "If you change this value you will create a separate logical index."
                ),
            ),
        )
        collection_help_btn.grid(row=2, column=3, padx=2)

        # Chars per chunk
        ttk.Label(params_frame, text="Chars per chunk:").grid(row=3, column=0, sticky="w")
        self.chars_per_chunk_var = tk.StringVar(value=str(CHARS_PER_CHUNK))
        chars_entry = ttk.Entry(params_frame, textvariable=self.chars_per_chunk_var, width=12)
        chars_entry.grid(row=3, column=1, sticky="w", padx=4)
        chars_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Chars per chunk",
                (
                    "Target size of each text chunk that is embedded.\n\n"
                    "Larger chunks capture more context from a file but can be less precise when answering "
                    "small focused questions. Smaller chunks give more precise snippets but may lose some "
                    "surrounding context.\n\n"
                    "Typical values are between 800 and 2000 characters. The default is chosen as a balance "
                    "between context and precision.\n\n"
                    "If you increase this, the number of chunks and embeddings goes down but each chunk becomes "
                    "larger. If you decrease this, the index will contain more smaller chunks."
                ),
            ),
        )
        chars_help_btn.grid(row=3, column=3, padx=2)

        # Chunk overlap
        ttk.Label(params_frame, text="Chunk overlap:").grid(row=4, column=0, sticky="w")
        self.chunk_overlap_var = tk.StringVar(value=str(CHUNK_OVERLAP))
        overlap_entry = ttk.Entry(params_frame, textvariable=self.chunk_overlap_var, width=12)
        overlap_entry.grid(row=4, column=1, sticky="w", padx=4)
        overlap_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Chunk overlap",
                (
                    "Number of characters shared between neighboring chunks.\n\n"
                    "Overlap helps keep related code or comments together when they sit near a chunk boundary. "
                    "Without overlap you can lose important context that crosses the artificial break between chunks.\n\n"
                    "A moderate overlap, for example 150 to 300 characters, is usually a good compromise. "
                    "Larger overlap slightly increases index size and build time but improves recall across boundaries.\n\n"
                    "Set this to zero if you want completely independent chunks and a smaller index."
                ),
            ),
        )
        overlap_help_btn.grid(row=4, column=3, padx=2)

        # Max file size
        ttk.Label(params_frame, text="Max file size (bytes):").grid(row=5, column=0, sticky="w")
        self.max_file_bytes_var = tk.StringVar(value=str(MAX_FILE_BYTES))
        max_file_entry = ttk.Entry(params_frame, textvariable=self.max_file_bytes_var, width=12)
        max_file_entry.grid(row=5, column=1, sticky="w", padx=4)
        max_file_help_btn = ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Max file size",
                (
                    "Files larger than this many bytes are skipped and not indexed.\n\n"
                    "This protects the indexer from accidentally pulling in very large generated files, "
                    "logs, or vendor bundles that would slow everything down and add noise to search.\n\n"
                    "If you know your project uses some large but important files you can raise this limit. "
                    "If you are indexing a very large repository and only care about typical source files you "
                    "can lower it to avoid giant files."
                ),
            ),
        )
        max_file_help_btn.grid(row=5, column=3, padx=2)

        params_frame.columnconfigure(1, weight=1)

        # File types frame
        filetypes_frame = ttk.LabelFrame(self, text="File types to index")
        filetypes_frame.pack(fill="x", padx=8, pady=4)

        btns_row = 0
        select_all_btn = ttk.Button(filetypes_frame, text="Select all", command=self._select_all_filetypes)
        select_all_btn.grid(row=btns_row, column=0, sticky="w", padx=4, pady=(4, 2))

        clear_all_btn = ttk.Button(filetypes_frame, text="Clear all", command=self._clear_all_filetypes)
        clear_all_btn.grid(row=btns_row, column=1, sticky="w", padx=4, pady=(4, 2))

        filetypes_help_btn = ttk.Button(
            filetypes_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "File types",
                (
                    "Choose which file extensions should be included in the index.\n\n"
                    "Only files whose extension is checked here will be read and chunked. This lets you focus "
                    "the index on real source and documentation and avoid noisy files such as build artifacts.\n\n"
                    "Typical use:\n"
                    "  For a web application, keep languages like java, ts, js, tsx, jsx, html, css, scss.\n"
                    "  For a Python project, you might only keep py, md, txt, json, yaml.\n\n"
                    "You can use Select all to index every supported type or Clear all and then pick a small subset "
                    "if you want to experiment with a faster and smaller index."
                ),
            ),
        )
        filetypes_help_btn.grid(row=btns_row, column=2, sticky="w", padx=4, pady=(4, 2))

        # Checkboxes for each extension
        exts = sorted(INDEX_EXTS)
        cols_per_row = 6
        start_row = 1
        for i, ext in enumerate(exts):
            row = start_row + i // cols_per_row
            col = (i % cols_per_row)
            var = tk.BooleanVar(value=True)
            self._ext_vars[ext] = var
            chk = ttk.Checkbutton(filetypes_frame, text=ext, variable=var)
            chk.grid(row=row, column=col, sticky="w", padx=4, pady=2)

        # Button frame
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=4)

        self.run_btn = ttk.Button(btn_frame, text="Run index", command=self.run_index)
        self.run_btn.pack(side="left")

        self.status_label = ttk.Label(btn_frame, text="")
        self.status_label.pack(side="right")

        # Log frame
        log_frame = ttk.LabelFrame(self, text="Index log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.log_text = ScrolledText(log_frame, wrap="word", height=12)
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _show_help(self, title: str, msg: str):
        messagebox.showinfo(title, msg)

    def browse_repo_root(self):
        path = filedialog.askdirectory(initialdir=self.repo_root_var.get() or os.path.expanduser("~"))
        if path:
            self.repo_root_var.set(path)

    def browse_index_dir(self):
        path = filedialog.askdirectory(initialdir=self.index_dir_var.get() or os.path.expanduser("~"))
        if path:
            self.index_dir_var.set(path)

    def _log(self, msg: str):
        self.after(0, lambda: self._append_log(msg))

    def _append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _select_all_filetypes(self):
        for var in self._ext_vars.values():
            var.set(True)

    def _clear_all_filetypes(self):
        for var in self._ext_vars.values():
            var.set(False)

    def run_index(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Info", "Indexing already running.")
            return

        repo_root = self.repo_root_var.get().strip()
        index_base_dir = self.index_dir_var.get().strip()
        collection_name = self.collection_var.get().strip()

        if not repo_root:
            messagebox.showerror("Error", "Repo root is required.")
            return
        if not os.path.isdir(repo_root):
            messagebox.showerror("Error", f"Repo root does not exist:\n{repo_root}")
            return
        if not index_base_dir:
            messagebox.showerror("Error", "Index directory is required.")
            return
        if not collection_name:
            messagebox.showerror("Error", "Collection name is required.")
            return

        # Actual folder where this collection will live, named from the GUI
        index_dir = os.path.join(index_base_dir, collection_name)

        # Parse numeric parameters
        try:
            chars_per_chunk = int(self.chars_per_chunk_var.get())
            chunk_overlap = int(self.chunk_overlap_var.get())
            max_file_bytes = int(self.max_file_bytes_var.get())
        except ValueError:
            messagebox.showerror("Error", "Chars per chunk, chunk overlap, and max file size must be integers.")
            return

        if chars_per_chunk <= 0:
            messagebox.showerror("Error", "Chars per chunk must be greater than zero.")
            return
        if chunk_overlap < 0:
            messagebox.showerror("Error", "Chunk overlap cannot be negative.")
            return
        if max_file_bytes <= 0:
            messagebox.showerror("Error", "Max file size must be greater than zero.")
            return

        # Collect selected file extensions
        selected_exts = {ext for ext, var in self._ext_vars.items() if var.get()}
        if not selected_exts:
            messagebox.showerror("Error", "Select at least one file type to index.")
            return

        self.log_text.delete("1.0", "end")
        self.run_btn.config(state="disabled")
        self.status_label.config(text="Indexing...")

        def worker():
            try:
                index_repo(
                    repo_root=repo_root,
                    index_dir=index_dir,          # now includes the name from the GUI
                    collection_name=collection_name,
                    index_exts=selected_exts,
                    max_file_bytes=max_file_bytes,
                    chars_per_chunk=chars_per_chunk,
                    chunk_overlap=chunk_overlap,
                    log=self._log,
                )
                self._done(True)
            except Exception as e:
                self._log(f"Error: {e}")
                self._done(False)

        self._current_thread = threading.Thread(target=worker, daemon=True)
        self._current_thread.start()

    def _done(self, success):
        def finish():
            self.run_btn.config(state="normal")
            self.status_label.config(text="Done" if success else "Failed")
        self.after(0, finish)
