#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

from gui.prompts_tab import PromptsTab
from gui.query_tab import QueryTab
from gui.index_tab import IndexTab
from gui.settings_tab import SettingsTab
from config import APP_TITLE


class CodeSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local RAG LLM")
        self.minsize(1200, 900)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)       

        def get_summarizer_prompt():
            return prompts_tab.summarizer_text.get("1.0", "end-1c")

        def get_chat_prompt():
            return prompts_tab.chat_prompt_text.get("1.0", "end-1c")

        query_tab = QueryTab(
            notebook,
            get_summarizer_prompt=get_summarizer_prompt,
            get_chat_prompt=get_chat_prompt,
        )
        notebook.add(query_tab, text="Query")

        index_tab = IndexTab(notebook)
        notebook.add(index_tab, text="Index")

        prompts_tab = PromptsTab(notebook)
        notebook.add(prompts_tab, text="Prompts")

        settings_tab = SettingsTab(notebook)
        notebook.add(settings_tab, text="Settings")

        notebook.select(query_tab)


def main():
    app = CodeSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
