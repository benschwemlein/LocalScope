#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

from gui.prompts_tab import PromptsTab
from gui.query_tab import QueryTab
from gui.index_tab import IndexTab
from gui.settings_tab import SettingsTab

from settings_store import SettingsManager
from settings_defaults import build_default_settings


class CodeSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Local RAG LLM")
        self.minsize(1200, 900)

        defaults = build_default_settings()
        self.settings = SettingsManager(self, defaults)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # Create Prompts tab first so Query can read prompts
        prompts_tab = PromptsTab(notebook, settings_mgr=self.settings)

        def get_summarizer_prompt() -> str:
            return prompts_tab.summarizer_text.get("1.0", "end-1c")

        def get_chat_prompt() -> str:
            return prompts_tab.chat_prompt_text.get("1.0", "end-1c")

        query_tab = QueryTab(
            notebook,
            settings_mgr=self.settings,
            get_summarizer_prompt=get_summarizer_prompt,
            get_chat_prompt=get_chat_prompt,
        )

        index_tab = IndexTab(notebook, settings_mgr=self.settings)
        settings_tab = SettingsTab(notebook, settings_mgr=self.settings)

        # Add tabs in correct visible order
        notebook.add(query_tab, text="Query")
        notebook.add(index_tab, text="Index")
        notebook.add(prompts_tab, text="Prompts")
        notebook.add(settings_tab, text="Settings")

        # ALWAYS start on Query tab
        notebook.select(query_tab)

        # Still persist last active tab (harmless, just not used on startup)
        def on_tab_changed(event=None):
            try:
                current_id = notebook.select()
                label = notebook.tab(current_id, "text")
                ui = self.settings.data.setdefault("ui", {})
                ui["last_active_tab"] = label
                self.settings.save_soon()
            except Exception:
                pass

        notebook.bind("<<NotebookTabChanged>>", on_tab_changed)


def main():
    app = CodeSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
