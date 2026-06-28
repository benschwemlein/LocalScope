import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from config import DEFAULT_INDEX_DIR, DEFAULT_REPO_ROOT, DEFAULT_COLLECTION_NAME
from indexing.incremental_indexer import index_repo_incremental
from indexing.indexer import (
    index_repo,
    CHARS_PER_CHUNK,
    CHUNK_OVERLAP,
    MAX_FILE_BYTES,
)

COMMON_FILETYPE_GROUPS = {
    "Microsoft / .NET": [
        ".cs", ".csx", ".vb",
        ".fs", ".fsi", ".fsx",
        ".xaml",
        ".cshtml",
        ".config",
        ".resx",
        ".sln",
        ".csproj", ".vbproj", ".fsproj",
    ],
    "PHP ecosystem": [
        ".php", ".phtml", ".twig", ".tpl",
    ],
    "Java / JVM": [
        ".java",
        ".kt", ".kts",
        ".groovy", ".gvy",
        ".scala", ".sc",
        ".gradle",
        ".xml",
        ".properties",
    ],
    "JavaScript / Web": [
        ".js", ".jsx",
        ".ts", ".tsx",
        ".mjs", ".cjs",
        ".vue", ".svelte",
        ".astro",
        ".html",
        ".css", ".scss", ".less",
        ".json", ".json5",
    ],
    "Python / Data": [
        ".py", ".pyi",
        ".ipynb",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
    ],
    "Mobile / UI": [
        ".xml",
        ".plist",
        ".storyboard",
        ".xib",
    ],
    "C / C++ / Embedded": [
        ".c", ".h",
        ".cpp", ".hpp",
        ".cc", ".cxx", ".hh",
        ".ino",
        ".mk",
    ],
    "Rust / Go / Ruby / Swift / ObjC": [
        ".rs",
        ".go",
        ".rb",
        ".swift",
        ".m", ".mm",
    ],
    "Cloud / IaC": [
        ".tf",
        ".tfvars",
        ".yaml",
        ".yml",
        ".json",
    ],
    "SQL / Query": [
        ".sql",
        ".psql",
        ".hql",
        ".cql",
    ],
    "Shell / Scripts": [
        ".sh", ".bash", ".zsh",
        ".ksh",
        ".fish",
        ".bat", ".cmd",
        ".ps1", ".psm1",
    ],
    "Documentation": [
        ".md", ".rst", ".adoc",
        ".txt",
        ".csv",
    ],
}


class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, text="", shown=True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._shown = tk.BooleanVar(value=bool(shown))

        header = ttk.Frame(self)
        header.pack(fill="x")

        self._btn = ttk.Checkbutton(
            header,
            text=text,
            variable=self._shown,
            command=self._toggle,
            style="Toolbutton",
        )
        self._btn.pack(side="left", padx=2, pady=2)

        self.body = ttk.Frame(self)

        if self._shown.get():
            self.body.pack(fill="both", expand=True)

    def _toggle(self):
        if self._shown.get():
            self.body.pack(fill="both", expand=True)
        else:
            self.body.forget()


class IndexTab(ttk.Frame):

    def __init__(self, parent, settings_mgr):
        super().__init__(parent)
        self.settings_mgr = settings_mgr

        self._current_thread = None
        self._ext_vars: dict[str, tk.BooleanVar] = {}

        self._loading_ui = True
        self._build_ui()
        self._wire_autosave()
        self._loading_ui = False

    def _build_ui(self):
        params_frame = ttk.LabelFrame(self, text="Index parameters")
        params_frame.pack(fill="x", padx=8, pady=8)

        s = self.settings_mgr.data.get("index_tab") or {}

        # Repo root
        ttk.Label(params_frame, text="Repo root:").grid(row=0, column=0, sticky="w")
        self.repo_root_var = tk.StringVar(value=s.get("repo_root", DEFAULT_REPO_ROOT))
        ttk.Entry(params_frame, textvariable=self.repo_root_var, width=60).grid(
            row=0, column=1, sticky="we", padx=4
        )
        ttk.Button(params_frame, text="Browse", command=self.browse_repo_root).grid(row=0, column=2, padx=4)
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Repo root",
                (
                    "Top folder of the source tree you want to index.\n\n"
                    "The indexer walks this directory and all child folders and reads files "
                    "that match the selected file types."
                ),
            ),
        ).grid(row=0, column=3)

        # Index directory
        ttk.Label(params_frame, text="Index directory:").grid(row=1, column=0, sticky="w")
        self.index_dir_var = tk.StringVar(value=s.get("index_dir_base", DEFAULT_INDEX_DIR))
        ttk.Entry(params_frame, textvariable=self.index_dir_var, width=60).grid(
            row=1, column=1, sticky="we", padx=4
        )
        ttk.Button(params_frame, text="Browse", command=self.browse_index_dir).grid(row=1, column=2, padx=4)
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Index directory",
                (
                    "Base folder where index data is stored.\n\n"
                    "A subfolder named with the collection name will be created here for this index."
                ),
            ),
        ).grid(row=1, column=3)

        # Collection name
        ttk.Label(params_frame, text="Collection name:").grid(row=2, column=0, sticky="w")
        self.collection_var = tk.StringVar(value=s.get("collection_name", DEFAULT_COLLECTION_NAME))
        ttk.Entry(params_frame, textvariable=self.collection_var, width=40).grid(
            row=2, column=1, sticky="w", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Collection name",
                (
                    "Logical name for this index.\n\n"
                    "Used as the Chroma collection name and as the subfolder name under the index directory."
                ),
            ),
        ).grid(row=2, column=3)

        # Chars per chunk
        ttk.Label(params_frame, text="Chars per chunk:").grid(row=3, column=0, sticky="w")
        self.chars_per_chunk_var = tk.StringVar(value=str(s.get("chars_per_chunk", CHARS_PER_CHUNK)))
        ttk.Entry(params_frame, textvariable=self.chars_per_chunk_var, width=10).grid(
            row=3, column=1, sticky="w", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Chars per chunk",
                (
                    "Target maximum characters in each text chunk sent to the embedding model.\n\n"
                    "Larger values capture more context in one chunk. Smaller values give tighter snippets "
                    "but may lose surrounding context. Typical range is about 800 to 2000."
                ),
            ),
        ).grid(row=3, column=3)

        # Chunk overlap
        ttk.Label(params_frame, text="Chunk overlap:").grid(row=4, column=0, sticky="w")
        self.chunk_overlap_var = tk.StringVar(value=str(s.get("chunk_overlap", CHUNK_OVERLAP)))
        ttk.Entry(params_frame, textvariable=self.chunk_overlap_var, width=10).grid(
            row=4, column=1, sticky="w", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Chunk overlap",
                (
                    "Number of characters shared between neighboring chunks.\n\n"
                    "Overlap helps keep related lines together when they cross an artificial chunk boundary. "
                    "Moderate overlap improves recall with a small cost in index size."
                ),
            ),
        ).grid(row=4, column=3)

        # Max file size
        ttk.Label(params_frame, text="Max file size (bytes):").grid(row=5, column=0, sticky="w")
        self.max_file_bytes_var = tk.StringVar(value=str(s.get("max_file_bytes", MAX_FILE_BYTES)))
        ttk.Entry(params_frame, textvariable=self.max_file_bytes_var, width=12).grid(
            row=5, column=1, sticky="w", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Max file size",
                (
                    "Files larger than this many bytes are skipped.\n\n"
                    "Useful to avoid huge bundles, logs or generated output bloating the index."
                ),
            ),
        ).grid(row=5, column=3)
        # Parallel workers
        ttk.Label(params_frame, text="Parallel workers:").grid(row=7, column=0, sticky="w")
        self.num_workers_var = tk.StringVar(value=str(s.get("num_workers", 4)))
        ttk.Entry(params_frame, textvariable=self.num_workers_var, width=5).grid(
            row=7, column=1, sticky="w", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Parallel workers",
                (
                    "Number of parallel threads for indexing.\n\n"
                    "Higher = faster for large codebases, but uses more CPU.\n\n"
                    "Recommended:\n"
                    "  Small repos (<1000 files): 1-2 workers\n"
                    "  Medium repos (1000-5000 files): 4-8 workers\n"
                    "  Large repos (5000+ files): 8-12 workers\n\n"
                    "Note: More workers = more memory usage"
                ),
            ),
        ).grid(row=7, column=3)


        # Exclude directories
        ttk.Label(params_frame, text="Exclude directories:").grid(row=8, column=0, sticky="w")
        default_excluded = ".git, .idea, .vscode, node_modules, build, dist, out, target, .gradle, .venv, venv, __pycache__"
        self.exclude_dirs_var = tk.StringVar(value=s.get("exclude_dirs_csv", default_excluded))
        ttk.Entry(params_frame, textvariable=self.exclude_dirs_var, width=60).grid(
            row=8, column=1, sticky="we", padx=4
        )
        ttk.Button(
            params_frame,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "Exclude directories",
                (
                    "Comma-separated list of directory names to skip while walking the repo.\n\n"
                    "The indexer ignores these folders entirely. Useful for build output, dependency folders, "
                    "virtual environments, and other noise."
                ),
            ),
        ).grid(row=6, column=3)

        params_frame.columnconfigure(1, weight=1)

        # Collapsible file types section (collapsed by default)
        filetypes_collapsible = CollapsibleFrame(self, text="File types to index", shown=False)
        filetypes_collapsible.pack(fill="x", padx=8, pady=4)

        filetypes_frame = filetypes_collapsible.body

        toolbar = ttk.Frame(filetypes_frame)
        toolbar.pack(fill="x", padx=4, pady=4)

        ttk.Button(toolbar, text="Select all", command=self._select_all_filetypes).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Clear all", command=self._clear_all_filetypes).pack(side="left", padx=4)
        ttk.Button(
            toolbar,
            text="?",
            width=2,
            command=lambda: self._show_help(
                "File types",
                (
                    "Choose which file extensions should be indexed.\n\n"
                    "Only files with these extensions are read and chunked. This keeps the index focused on "
                    "actual source, configuration and documentation files."
                ),
            ),
        ).pack(side="left", padx=4)

        groups_holder = ttk.Frame(filetypes_frame)
        groups_holder.pack(fill="x", padx=4, pady=(2, 4))

        saved_filetypes = (s.get("filetypes") or {})

        seen: set[str] = set()
        group_names = list(COMMON_FILETYPE_GROUPS.keys())
        cols_of_groups = 3

        for i, group in enumerate(group_names):
            exts = COMMON_FILETYPE_GROUPS[group]
            row = i // cols_of_groups
            col = i % cols_of_groups

            group_frame = ttk.LabelFrame(groups_holder, text=group)
            group_frame.grid(row=row, column=col, sticky="nwe", padx=4, pady=2)

            groups_holder.columnconfigure(col, weight=1)

            inner_cols = 4
            idx = 0
            for ext in sorted(exts):
                if ext in seen:
                    continue
                seen.add(ext)

                r = idx // inner_cols
                c = idx % inner_cols
                idx += 1

                var = tk.BooleanVar(value=bool(saved_filetypes.get(ext, True)))
                self._ext_vars[ext] = var
                chk = ttk.Checkbutton(group_frame, text=ext, variable=var)
                chk.grid(row=r, column=c, sticky="w", padx=2, pady=1)

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="x", padx=8, pady=8)

        # Incremental mode checkbox
        self.incremental_mode_var = tk.BooleanVar(value=True)
        self.incremental_chk = ttk.Checkbutton(
            bottom_frame,
            text="Incremental Update (only process changed files)",
            variable=self.incremental_mode_var
        )
        self.incremental_chk.pack(side="left", padx=(0, 10))

        # Verbose logging checkbox
        self.verbose_mode_var = tk.BooleanVar(value=False)
        self.verbose_chk = ttk.Checkbutton(
            bottom_frame,
            text="Show individual files",
            variable=self.verbose_mode_var
        )
        self.verbose_chk.pack(side="left", padx=(0, 10))

        # Index button
        self.run_btn = ttk.Button(bottom_frame, text="Index", command=self.run_index)
        self.run_btn.pack(side="left")

        # Full reindex button
        self.full_reindex_btn = ttk.Button(
            bottom_frame,
            text="Full Reindex",
            command=self.run_full_reindex
        )
        self.full_reindex_btn.pack(side="left", padx=(5, 0))

        self.status_label = ttk.Label(bottom_frame, text="")
        self.status_label.pack(side="right")


        log_frame = ttk.LabelFrame(self, text="Index log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.log_text = ScrolledText(log_frame, wrap="word", height=12)
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Ensure settings file contains all fields for this tab
        self._save_index_to_settings(save_immediately=False)

    def _show_help(self, title: str, text: str):
        messagebox.showinfo(title, text)

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
        for v in self._ext_vars.values():
            v.set(True)

    def _clear_all_filetypes(self):
        for v in self._ext_vars.values():
            v.set(False)

    def _save_index_to_settings(self, save_immediately: bool):
        if getattr(self, "_loading_ui", False):
            return

        self.settings_mgr.data["index_tab"] = {
            "repo_root": self.repo_root_var.get().strip(),
            "index_dir_base": self.index_dir_var.get().strip(),
            "collection_name": self.collection_var.get().strip(),
            "chars_per_chunk": int(self.chars_per_chunk_var.get() or CHARS_PER_CHUNK),
            "chunk_overlap": int(self.chunk_overlap_var.get() or CHUNK_OVERLAP),
            "max_file_bytes": int(self.max_file_bytes_var.get() or MAX_FILE_BYTES),
            "num_workers": int(self.num_workers_var.get() or 4),
            "exclude_dirs_csv": self.exclude_dirs_var.get(),
            "filetypes": {ext: var.get() for ext, var in self._ext_vars.items()},
        }

        if save_immediately:
            self.settings_mgr.save_now()
        else:
            self.settings_mgr.save_soon()

    def _wire_autosave(self):
        def on_any_change(*_):
            self._save_index_to_settings(save_immediately=False)

        for v in [
            self.repo_root_var,
            self.index_dir_var,
            self.collection_var,
            self.chars_per_chunk_var,
            self.chunk_overlap_var,
            self.max_file_bytes_var,
            self.num_workers_var,
            self.exclude_dirs_var,
        ]:
            v.trace_add("write", on_any_change)

        for _, b in self._ext_vars.items():
            b.trace_add("write", on_any_change)

    def run_index(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Info", "Indexing already running.")
            return

        repo_root = self.repo_root_var.get().strip()
        index_base = self.index_dir_var.get().strip()
        collection_name = self.collection_var.get().strip()

        if not repo_root:
            messagebox.showerror("Error", "Repo root is required.")
            return
        if not os.path.isdir(repo_root):
            messagebox.showerror("Error", f"Repo root does not exist:\n{repo_root}")
            return
        if not index_base:
            messagebox.showerror("Error", "Index directory is required.")
            return

        index_dir = os.path.join(index_base, collection_name)

        try:
            chars = int(self.chars_per_chunk_var.get())
            overlap = int(self.chunk_overlap_var.get())
            max_bytes = int(self.max_file_bytes_var.get())
            num_workers = int(self.num_workers_var.get())
        except ValueError:
            messagebox.showerror("Error", "Numeric fields must contain integers.")
            return

        if chars <= 0:
            messagebox.showerror("Error", "Chars per chunk must be greater than zero.")
            return
        if overlap < 0:
            messagebox.showerror("Error", "Chunk overlap cannot be negative.")
            return
        if max_bytes <= 0:
            messagebox.showerror("Error", "Max file size must be greater than zero.")
            return

        # Parse excluded directories
        raw_excluded = self.exclude_dirs_var.get().strip()
        excluded_dirs: set[str] = set()
        if raw_excluded:
            for part in raw_excluded.split(","):
                name = part.strip()
                if name:
                    excluded_dirs.add(name)

        selected_exts = {ext for ext, var in self._ext_vars.items() if var.get()}
        if not selected_exts:
            messagebox.showerror("Error", "Choose at least one file type.")
            return

        # Save settings right before running index so they reflect what ran
        self._save_index_to_settings(save_immediately=False)

        self.log_text.delete("1.0", "end")
        self.run_btn.config(state="disabled")
        self.status_label.config(text="Indexing...")

        def worker():
            try:
                # Get mode settings
                incremental = self.incremental_mode_var.get()
                force_full = not incremental
                verbose = self.verbose_mode_var.get()
                
                index_repo_incremental(
                    repo_root=repo_root,
                    index_dir=index_dir,
                    collection_name=collection_name,
                    index_exts=selected_exts,
                    excluded_dirs=excluded_dirs,
                    max_file_bytes=max_bytes,
                    chars_per_chunk=chars,
                    chunk_overlap=overlap,
                    use_ast_chunking=True,
                    skip_problematic_files=True,
                    force_full_reindex=force_full,
                    num_workers=num_workers,
                    verbose=verbose,
                    log=self._log,
                )
                self._done(True)
            except Exception as e:
                self._log(f"Error: {e}")
                self._done(False)


        self._current_thread = threading.Thread(target=worker, daemon=True)
        self._current_thread.start()

    def run_full_reindex(self):
        """Force a full reindex, ignoring incremental mode."""
        response = messagebox.askyesno(
            "Full Reindex",
            "This will DELETE the entire existing index and rebuild from scratch.\n\n"
            "This is useful if:\n"
            "- You changed embedding models\n"
            "- The index is corrupted\n"
            "- You want to ensure everything is fresh\n\n"
            "Continue with full reindex?"
        )
        
        if not response:
            return
        
        # Temporarily disable incremental mode
        original_value = self.incremental_mode_var.get()
        self.incremental_mode_var.set(False)
        
        # Run index (will use force_full_reindex=True)
        self.run_index()
        
        # Restore original value
        self.incremental_mode_var.set(original_value)

    def _done(self, ok: bool):
        def finish():
            self.run_btn.config(state="normal")
            self.status_label.config(text="Done" if ok else "Failed")
        self.after(0, finish)