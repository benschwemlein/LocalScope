#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

from gui.prompts_tab import PromptsTab
from gui.query_tab import QueryTab
from gui.index_tab import IndexTab


class CodeSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Local RAG LLM")

        # Optional: minimum size
        self.minsize(900, 600)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # Prompts tab first so QueryTab can read from it
        prompts_tab = PromptsTab(notebook)
        notebook.add(prompts_tab, text="Prompts")

        def get_summarizer_prompt():
            # Text without trailing newline
            return prompts_tab.summarizer_text.get("1.0", "end-1c")

        def get_chat_prompt():
            return prompts_tab.chat_prompt_text.get("1.0", "end-1c")

        # Query tab: uses prompts from PromptsTab
        query_tab = QueryTab(
            notebook,
            get_summarizer_prompt=get_summarizer_prompt,
            get_chat_prompt=get_chat_prompt,
        )
        notebook.add(query_tab, text="Query")

        # Index tab: runs repo indexing
        index_tab = IndexTab(notebook)
        notebook.add(index_tab, text="Index")


        # Select Query tab by default
        notebook.select(query_tab)


def main():
    app = CodeSearchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
