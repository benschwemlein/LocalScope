# gui/settings_tab.py

import threading
import requests
import tkinter as tk
from tkinter import ttk, messagebox

import config

# Requires: ollama/__init__.py exists
from ollama_manager.download_manager import DownloadManager, PullProgress

# Curated metadata for common models.
# We match by base name before ":" tag (example: "llama3.1:8b" -> "llama3.1").
MODEL_INFO = {
    # Embeddings
    "nomic-embed-text": (
        "Nomic AI",
        "Text embedding model designed for high quality semantic search and retrieval.",
    ),
    "all-minilm": (
        "Microsoft / sentence-transformers",
        "Small and fast general purpose embedding model based on MiniLM.",
    ),
    "mxbai-embed-large": (
        "mixedbread.ai",
        "Strong general purpose embedding model.",
    ),
    "bge-large": (
        "BAAI",
        "Popular embedding model family used for semantic search.",
    ),
    "embeddinggemma": (
        "Google",
        "EmbeddingGemma embedding model.",
    ),
    "snowflake-arctic-embed2": (
        "Snowflake",
        "Frontier embedding model with multilingual support.",
    ),
    "qwen3-embedding": (
        "Alibaba Cloud",
        "Qwen3 embedding models in multiple sizes.",
    ),
    "granite-embedding": (
        "IBM",
        "Granite embedding models.",
    ),

    # General chat models
    "llama3.1": ("Meta", "Llama 3.1 family model, strong general purpose chat and coding."),
    "llama3.2": ("Meta", "Llama 3.2 small models."),
    "llama3": ("Meta", "Llama 3 family model for general chat and coding tasks."),
    "llama2": ("Meta", "Earlier Llama 2 series open model for chat and reasoning."),
    "qwen2.5": ("Alibaba Cloud", "Qwen 2.5 family model, strong at multilingual tasks and reasoning."),
    "qwen3": ("Alibaba Cloud", "Qwen 3 family models."),
    "mistral": ("Mistral AI", "Mistral family model, efficient general purpose LLM."),
    "mixtral": ("Mistral AI", "Mixture of experts model that trades size for higher quality."),
    "phi3": ("Microsoft", "Phi 3 series small model tuned for strong reasoning at small sizes."),
    "phi-3": ("Microsoft", "Phi 3 series small model tuned for strong reasoning at small sizes."),
    "phi4": ("Microsoft", "Phi 4 family models."),

    # Code focused models
    "qwen2.5-coder": ("Alibaba Cloud", "Code specific Qwen models for code gen, reasoning, fixing."),
    "qwen3-coder": ("Alibaba Cloud", "Qwen3 coder models for agentic and coding tasks."),
    "deepseek-coder": ("DeepSeek", "Code focused DeepSeek Coder models."),
    "deepseek-coder-v2": ("DeepSeek", "Newer DeepSeek Coder V2 models."),
    "codellama": ("Meta", "Code Llama models."),
    "starcoder2": ("BigCode", "StarCoder2 coding models."),
    "wizardcoder": ("WizardLM", "Code generation focused WizardCoder models."),
    "sqlcoder": ("Defog.ai", "SQL generation focused model based on StarCoder."),
    "codegemma": ("Google", "CodeGemma coding models."),
    "codeqwen": ("Alibaba Cloud", "CodeQwen models."),
    "codegeex4": ("THUDM", "CodeGeeX coding models."),
    "stable-code": ("Stability AI", "Stable Code coding models."),
    "yi-coder": ("01.AI", "Yi Coder models."),
    "dolphincoder": ("Dolphin", "Coding focused Dolphin variants."),
}

SUGGESTED_PULL_MODELS = [
    # Embeddings
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "bge-large",
    "embeddinggemma",

    # Code focused
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "qwen3-coder:30b",
    "deepseek-coder:6.7b",
    "deepseek-coder:33b",
    "deepseek-coder-v2:16b",
    "codellama:7b",
    "codellama:13b",
    "starcoder2:7b",
    "wizardcoder:33b",
    "sqlcoder:7b",
    "codegemma:7b",
    "codeqwen:7b",
    "stable-code:3b",
    "yi-coder:9b",

    # Solid general chat options
    "llama3.1:8b",
    "qwen2.5:7b",
    "mistral:7b",
    "mixtral:8x7b",
    "phi3:3.8b",
]


def describe_model(name: str):
    base = name.split(":", 1)[0].lower()

    if base in MODEL_INFO:
        return MODEL_INFO[base]

    for key, info in MODEL_INFO.items():
        if base.startswith(key):
            return info

    return (
        "Unknown or community model",
        "No specific metadata is available. It may be a community or custom model.",
    )


class SettingsTab(ttk.Frame):
    """
    Tab for configuring runtime settings such as Ollama URL and model names.

    Changes update the config module globals, which are read by the indexer
    and query engine on each call.
    """

    def __init__(self, parent, settings_mgr):
        super().__init__(parent)
        self.settings_mgr = settings_mgr

        self.available_models: list[str] = []
        self.embed_combo = None
        self.chat_combo = None

        self.ollama_online = False
        self.status_var: tk.StringVar | None = None
        self.status_label: ttk.Label | None = None

        self._loading_ui = True

        # Load saved URL into config before we try to fetch models
        saved = self.settings_mgr.data.get("settings_tab") or {}
        saved_url = (saved.get("ollama_url") or "").strip()
        if saved_url:
            config.OLLAMA_URL = saved_url

        # Download manager needs an initial URL
        self._dl_mgr = DownloadManager(config.OLLAMA_URL)

        self._load_models_from_ollama()
        self._build_ui()
        self._update_status_indicator()
        self._wire_autosave()

        self._loading_ui = False

    def _load_models_from_ollama(self):
        url = f"{config.OLLAMA_URL.rstrip('/')}/api/tags"
        self.ollama_online = False
        try:
            resp = requests.get(url, timeout=5)
        except Exception:
            self.available_models = []
            self._update_status_indicator()
            return

        if not resp.ok:
            self.available_models = []
            self._update_status_indicator()
            return

        try:
            data = resp.json()
        except Exception:
            self.available_models = []
            self._update_status_indicator()
            return

        models = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                models.append(name)

        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        self.available_models = sorted(unique_models)
        self.ollama_online = True
        self._update_status_indicator()

    def _build_ui(self):
        main_frame = ttk.LabelFrame(self, text="Ollama and model settings")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        s = self.settings_mgr.data.get("settings_tab") or {}

        ttk.Label(main_frame, text="Ollama URL:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.ollama_url_var = tk.StringVar(value=s.get("ollama_url", config.OLLAMA_URL))
        ttk.Entry(main_frame, textvariable=self.ollama_url_var, width=50).grid(
            row=0, column=1, sticky="we", padx=4, pady=4
        )

        ttk.Button(main_frame, text="?", width=2, command=self._show_ollama_help).grid(
            row=0, column=2, sticky="e", padx=4, pady=4
        )

        self.status_var = tk.StringVar(value="status unknown")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(main_frame, text="Embedding model:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.embed_model_var = tk.StringVar(value=s.get("embed_model", config.EMBED_MODEL))
        self.embed_combo = ttk.Combobox(
            main_frame, textvariable=self.embed_model_var, values=self.available_models, width=40, state="normal"
        )
        self.embed_combo.grid(row=1, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(main_frame, text="?", width=2, command=self._show_embed_model_help).grid(
            row=1, column=2, sticky="e", padx=4, pady=4
        )

        ttk.Label(main_frame, text="Chat model:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.chat_model_var = tk.StringVar(value=s.get("chat_model", config.CHAT_MODEL))
        self.chat_combo = ttk.Combobox(
            main_frame, textvariable=self.chat_model_var, values=self.available_models, width=40, state="normal"
        )
        self.chat_combo.grid(row=2, column=1, sticky="we", padx=4, pady=4)
        ttk.Button(main_frame, text="?", width=2, command=self._show_chat_model_help).grid(
            row=2, column=2, sticky="e", padx=4, pady=4
        )

        ttk.Button(main_frame, text="Refresh models from Ollama", command=self._refresh_models).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 8)
        )
        ttk.Button(main_frame, text="Download model…", command=self._download_model).grid(
            row=3, column=2, sticky="w", padx=4, pady=(4, 8)
        )

        main_frame.columnconfigure(1, weight=1)

    def _wire_autosave(self):
        last_url = {"value": (self.ollama_url_var.get() or "").strip()}

        def save_core(*_):
            if getattr(self, "_loading_ui", False):
                return

            url = (self.ollama_url_var.get() or "").strip()
            embed = (self.embed_model_var.get() or "").strip()
            chat = (self.chat_model_var.get() or "").strip()

            # Persist to settings file
            self.settings_mgr.data["settings_tab"] = {
                "ollama_url": url,
                "embed_model": embed,
                "chat_model": chat,
            }
            self.settings_mgr.save_soon()

            # Update runtime globals immediately
            if url:
                config.OLLAMA_URL = url
                self._dl_mgr.set_ollama_url(url)
            if embed:
                config.EMBED_MODEL = embed
            if chat:
                config.CHAT_MODEL = chat

            # If URL changed, refresh the model list from the new instance
            if url and url != last_url["value"]:
                last_url["value"] = url
                self._refresh_models(silent=True)

        self.ollama_url_var.trace_add("write", save_core)
        self.embed_model_var.trace_add("write", save_core)
        self.chat_model_var.trace_add("write", save_core)

        save_core()


    def _update_status_indicator(self):
        if self.status_var is None or self.status_label is None:
            return
        if self.ollama_online:
            self.status_var.set("Ollama running")
            self.status_label.configure(foreground="green")
        else:
            self.status_var.set("Ollama not reachable")
            self.status_label.configure(foreground="red")

    def _show_ollama_help(self):
        messagebox.showinfo(
            "Ollama URL",
            (
                "The base URL where Ollama is running on your machine.\n\n"
                "Examples:\n"
                "  http://localhost:11434\n\n"
                "The app calls:\n"
                "  {URL}/api/embeddings  for embeddings\n"
                "  {URL}/api/chat        for chat completions\n"
                "  {URL}/api/tags        to list installed models\n"
                "  {URL}/api/pull        to download models\n"
            ),
        )

    def _show_embed_model_help(self):
        if self.available_models:
            lines = ["Embedding models discovered on this Ollama instance:\n"]
            for name in self.available_models:
                company, desc = describe_model(name)
                lines.append(f"{name}\n  Company: {company}\n  Description: {desc}\n")
            lines.append("Note: The same embedding model must be used for indexing and querying.\nRe index if you change it.")
            text = "\n".join(lines)
        else:
            text = (
                "Name of the Ollama model used for embeddings.\n\n"
                "Examples:\n"
                "  nomic-embed-text\n"
                "  mxbai-embed-large\n\n"
                "See installed models:\n"
                "  ollama list"
            )
        messagebox.showinfo("Embedding model", text)

    def _show_chat_model_help(self):
        if self.available_models:
            lines = ["Chat models discovered on this Ollama instance:\n"]
            for name in self.available_models:
                company, desc = describe_model(name)
                lines.append(f"{name}\n  Company: {company}\n  Description: {desc}\n")
            lines.append("Note: You can change the chat model without re indexing.")
            text = "\n".join(lines)
        else:
            text = (
                "Name of the Ollama chat model used to answer questions.\n\n"
                "Examples:\n"
                "  llama3.1:8b\n"
                "  qwen2.5-coder:7b\n\n"
                "See installed models:\n"
                "  ollama list"
            )
        messagebox.showinfo("Chat model", text)

    def _download_model(self):
        base_url = self.ollama_url_var.get().strip() or config.OLLAMA_URL
        if not base_url:
            messagebox.showerror("Download model", "Ollama URL is empty.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Download model")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        ttk.Label(dialog, text="Choose a model to download (or type a custom name):").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 4)
        )

        model_var = tk.StringVar(value=SUGGESTED_PULL_MODELS[0] if SUGGESTED_PULL_MODELS else "")
        combo = ttk.Combobox(dialog, textvariable=model_var, values=SUGGESTED_PULL_MODELS, width=44, state="normal")
        combo.grid(row=1, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        combo.focus_set()

        prog_label_var = tk.StringVar(value="")
        prog_label = ttk.Label(dialog, textvariable=prog_label_var)
        prog = ttk.Progressbar(dialog, orient="horizontal", mode="determinate", maximum=100)
        status_text = tk.StringVar(value="")
        status_lbl = ttk.Label(dialog, textvariable=status_text)

        prog_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 0))
        prog.grid(row=3, column=0, columnspan=2, sticky="we", padx=8, pady=4)
        status_lbl.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        prog_label.grid_remove()
        prog.grid_remove()
        status_lbl.grid_remove()

        def on_start():
            name = model_var.get().strip()
            if not name:
                messagebox.showerror("Download model", "Please choose or type a model name.")
                return

            prog_label_var.set(f"Downloading: {name}")
            prog["value"] = 0
            status_text.set("Starting...")
            prog_label.grid()
            prog.grid()
            status_lbl.grid()

            try:
                dialog.grab_release()
            except Exception:
                pass

            start_btn.config(state="disabled")
            cancel_btn.config(text="Close", command=dialog.destroy)

            # Make sure manager is pointing at the current URL
            self._dl_mgr.set_ollama_url(base_url)

            def ui_progress(p: PullProgress):
                def apply():
                    prog["value"] = float(getattr(p, "percent", 0.0) or 0.0)
                    if getattr(p, "status", ""):
                        status_text.set(str(p.status))
                self.after(0, apply)

            def ui_done(p: PullProgress):
                def apply():
                    if getattr(p, "ok", False):
                        status_text.set("Done")
                        prog["value"] = 100
                    else:
                        status_text.set(p.error or "Failed")
                    self._refresh_models(silent=True)
                self.after(0, apply)

            self._dl_mgr.start_pull(
                model=name,
                on_progress=ui_progress,
                on_done=ui_done,
            )

        def on_cancel():
            dialog.destroy()

        start_btn = ttk.Button(dialog, text="Download", command=on_start)
        start_btn.grid(row=5, column=0, sticky="e", padx=8, pady=8)

        cancel_btn = ttk.Button(dialog, text="Cancel", command=on_cancel)
        cancel_btn.grid(row=5, column=1, sticky="w", padx=8, pady=8)

        dialog.columnconfigure(0, weight=1)
        dialog.columnconfigure(1, weight=1)

    def _refresh_models(self, silent: bool = False):
        new_url = self.ollama_url_var.get().strip()
        if new_url:
            config.OLLAMA_URL = new_url
            self._dl_mgr.set_ollama_url(new_url)

        self._load_models_from_ollama()

        if self.embed_combo is not None:
            self.embed_combo["values"] = self.available_models
        if self.chat_combo is not None:
            self.chat_combo["values"] = self.available_models

        if silent:
            return

        if self.available_models:
            messagebox.showinfo("Models refreshed", f"Loaded {len(self.available_models)} models from Ollama.")
        else:
            messagebox.showwarning("Models not found", "Could not load any models from Ollama. Check the URL and that Ollama is running.")

    def apply_settings(self):
        new_url = self.ollama_url_var.get().strip()
        if new_url:
            config.OLLAMA_URL = new_url
            self._dl_mgr.set_ollama_url(new_url)

        new_embed = self.embed_model_var.get().strip()
        if new_embed:
            config.EMBED_MODEL = new_embed

        new_chat = self.chat_model_var.get().strip()
        if new_chat:
            config.CHAT_MODEL = new_chat

        self.settings_mgr.data["settings_tab"] = {
            "ollama_url": (self.ollama_url_var.get() or "").strip(),
            "embed_model": (self.embed_model_var.get() or "").strip(),
            "chat_model": (self.chat_model_var.get() or "").strip(),
        }
        self.settings_mgr.save_soon()

        self._refresh_models()

        messagebox.showinfo(
            "Settings applied",
            f"Ollama URL set to: {config.OLLAMA_URL}\nEmbedding model: {config.EMBED_MODEL}\nChat model: {config.CHAT_MODEL}",
        )
