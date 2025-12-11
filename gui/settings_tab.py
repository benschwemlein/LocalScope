import tkinter as tk
from tkinter import ttk, messagebox

import config
import requests


# Simple curated metadata for common models.
# We match by base name before the ":" tag (e.g. "llama3.1:8b" -> "llama3.1").
MODEL_INFO = {
    "nomic-embed-text": (
        "Nomic AI",
        "Text embedding model designed for high quality semantic search and retrieval.",
    ),
    "all-minilm": (
        "Microsoft / Hugging Face",
        "Small and fast general purpose embedding model based on MiniLM.",
    ),
    "llama3.1": (
        "Meta",
        "Llama 3.1 generation model, good general purpose chat and coding assistant.",
    ),
    "llama3": (
        "Meta",
        "Llama 3 series model for general chat and coding tasks.",
    ),
    "llama2": (
        "Meta",
        "Earlier Llama 2 series open model for chat and reasoning.",
    ),
    "qwen2.5": (
        "Alibaba Cloud",
        "Qwen 2.5 family model, strong at multilingual tasks and reasoning.",
    ),
    "qwen2": (
        "Alibaba Cloud",
        "Qwen 2 series general chat and coding model.",
    ),
    "mistral": (
        "Mistral AI",
        "Mistral family model, efficient general purpose LLM.",
    ),
    "mixtral": (
        "Mistral AI",
        "Mixture of experts model that trades size for higher quality.",
    ),
    "phi3": (
        "Microsoft",
        "Phi 3 series small model tuned for strong reasoning at small sizes.",
    ),
    "phi-3": (
        "Microsoft",
        "Phi 3 series small model tuned for strong reasoning at small sizes.",
    ),
}


def describe_model(name: str):
    """
    Return (company, description) for a given Ollama model name.

    We strip any tag after ":" and match by prefix in MODEL_INFO.
    """
    base = name.split(":", 1)[0].lower()

    # Exact key match first
    if base in MODEL_INFO:
        return MODEL_INFO[base]

    # Prefix match as fallback
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

    def __init__(self, parent):
        super().__init__(parent)
        self.available_models: list[str] = []
        self.embed_combo = None
        self.chat_combo = None

        self._load_models_from_ollama()
        self._build_ui()

    def _load_models_from_ollama(self):
        """
        Query Ollama for the list of installed models.

        Populates self.available_models. Fails gracefully if Ollama is not reachable.
        """
        url = f"{config.OLLAMA_URL.rstrip('/')}/api/tags"
        try:
            resp = requests.get(url, timeout=5)
        except Exception:
            self.available_models = []
            return

        if not resp.ok:
            self.available_models = []
            return

        try:
            data = resp.json()
        except Exception:
            self.available_models = []
            return

        models = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if name:
                models.append(name)

        # De duplicate and sort for nicer UX
        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        self.available_models = sorted(unique_models)

    def _build_ui(self):
        main_frame = ttk.LabelFrame(self, text="Ollama and model settings")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Ollama URL
        ttk.Label(main_frame, text="Ollama URL:").grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        self.ollama_url_var = tk.StringVar(value=config.OLLAMA_URL)
        ollama_entry = ttk.Entry(main_frame, textvariable=self.ollama_url_var, width=50)
        ollama_entry.grid(row=0, column=1, sticky="we", padx=4, pady=4)

        ollama_help_btn = ttk.Button(
            main_frame,
            text="?",
            width=2,
            command=self._show_ollama_help,
        )
        ollama_help_btn.grid(row=0, column=2, sticky="e", padx=4, pady=4)

        # Embedding model dropdown
        ttk.Label(main_frame, text="Embedding model:").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        self.embed_model_var = tk.StringVar(value=config.EMBED_MODEL)

        self.embed_combo = ttk.Combobox(
            main_frame,
            textvariable=self.embed_model_var,
            values=self.available_models,
            width=40,
            state="normal",  # allow typing custom model names
        )
        self.embed_combo.grid(row=1, column=1, sticky="we", padx=4, pady=4)

        embed_help_btn = ttk.Button(
            main_frame,
            text="?",
            width=2,
            command=self._show_embed_model_help,
        )
        embed_help_btn.grid(row=1, column=2, sticky="e", padx=4, pady=4)

        # Chat model dropdown
        ttk.Label(main_frame, text="Chat model:").grid(
            row=2, column=0, sticky="w", padx=4, pady=4
        )
        self.chat_model_var = tk.StringVar(value=config.CHAT_MODEL)

        self.chat_combo = ttk.Combobox(
            main_frame,
            textvariable=self.chat_model_var,
            values=self.available_models,
            width=40,
            state="normal",  # allow typing custom model names
        )
        self.chat_combo.grid(row=2, column=1, sticky="we", padx=4, pady=4)

        chat_help_btn = ttk.Button(
            main_frame,
            text="?",
            width=2,
            command=self._show_chat_model_help,
        )
        chat_help_btn.grid(row=2, column=2, sticky="e", padx=4, pady=4)

        # Refresh models button
        refresh_btn = ttk.Button(
            main_frame,
            text="Refresh models from Ollama",
            command=self._refresh_models,
        )
        refresh_btn.grid(row=3, column=1, sticky="w", padx=4, pady=(4, 8))

        main_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(0, 8))

        apply_btn = ttk.Button(btn_frame, text="Apply settings", command=self.apply_settings)
        apply_btn.pack(side="left")

        info_label = ttk.Label(
            btn_frame,
            text="These settings affect new queries and indexing runs.",
        )
        info_label.pack(side="right")

    # Help popups

    def _show_ollama_help(self):
        messagebox.showinfo(
            "Ollama URL",
            (
                "The base URL where Ollama is running on your machine.\n\n"
                "Examples:\n"
                "  http://localhost:11434   (default on your own laptop)\n\n"
                "The app calls:\n"
                "  {URL}/api/embeddings  for embeddings\n"
                "  {URL}/api/chat        for chat completions\n\n"
                "If you do not have Ollama installed, download it from:\n"
                "  https://ollama.com/download\n\n"
                "After installing and starting Ollama, leave the URL as\n"
                "  http://localhost:11434\n"
                "unless you have configured Ollama to listen elsewhere.\n\n"
                "If you change the URL, click Refresh models to reload the\n"
                "model list from that instance."
            ),
        )

    def _show_embed_model_help(self):
        if self.available_models:
            lines = [
                "Embedding models discovered on this Ollama instance:\n"
            ]
            for name in self.available_models:
                company, desc = describe_model(name)
                lines.append(f"{name}")
                lines.append(f"  Company: {company}")
                lines.append(f"  Description: {desc}\n")

            lines.append(
                "Note: The same embedding model must be used for indexing and querying.\n"
                "If you change this, you should re index your repository."
            )
            text = "\n".join(lines)
        else:
            text = (
                "Name of the Ollama model used for embeddings.\n\n"
                "This model turns code and text into vectors for ChromaDB.\n"
                "All indexing and query similarity search depends on this.\n\n"
                "Examples:\n"
                "  nomic-embed-text\n"
                "  all-minilm\n\n"
                "The same embedding model must be used for both indexing and querying.\n"
                "You can see installed models in the Ollama app or by running:\n"
                "  ollama list"
            )

        messagebox.showinfo("Embedding model", text)

    def _show_chat_model_help(self):
        if self.available_models:
            lines = [
                "Chat models discovered on this Ollama instance:\n"
            ]
            for name in self.available_models:
                company, desc = describe_model(name)
                lines.append(f"{name}")
                lines.append(f"  Company: {company}")
                lines.append(f"  Description: {desc}\n")

            lines.append(
                "Note: You can change the chat model without re indexing, since it only\n"
                "affects how the answer is generated from the retrieved snippets."
            )
            text = "\n".join(lines)
        else:
            text = (
                "Name of the Ollama chat model used to answer questions.\n\n"
                "This model reads your bug description plus the retrieved code\n"
                "snippets and generates the final explanation or fix suggestion.\n\n"
                "Examples:\n"
                "  llama3.1\n"
                "  llama3\n"
                "  qwen2.5\n\n"
                "You can switch this without re indexing, because embeddings do not\n"
                "depend on the chat model.\n"
                "You can see installed models by running:\n"
                "  ollama list"
            )

        messagebox.showinfo("Chat model", text)

    def _refresh_models(self):
        """
        Reload models from Ollama and update the dropdown lists.
        """
        self._load_models_from_ollama()

        if self.embed_combo is not None:
            self.embed_combo["values"] = self.available_models
        if self.chat_combo is not None:
            self.chat_combo["values"] = self.available_models

        if self.available_models:
            messagebox.showinfo(
                "Models refreshed",
                f"Loaded {len(self.available_models)} models from Ollama.",
            )
        else:
            messagebox.showwarning(
                "Models not found",
                "Could not load any models from Ollama. Check the URL and that Ollama is running.",
            )

    # Apply

    def apply_settings(self):
        # Update config globals
        new_url = self.ollama_url_var.get().strip()
        if new_url:
            config.OLLAMA_URL = new_url

        new_embed = self.embed_model_var.get().strip()
        if new_embed:
            config.EMBED_MODEL = new_embed

        new_chat = self.chat_model_var.get().strip()
        if new_chat:
            config.CHAT_MODEL = new_chat

        # Refresh models in case the URL changed
        self._refresh_models()

        messagebox.showinfo(
            "Settings applied",
            f"Ollama URL set to: {config.OLLAMA_URL}\n"
            f"Embedding model: {config.EMBED_MODEL}\n"
            f"Chat model: {config.CHAT_MODEL}",
        )
