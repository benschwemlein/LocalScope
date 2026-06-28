import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


class PromptsTab(ttk.Frame):
    """
    Tab for viewing and editing the summarizer and chat prompt templates.

    These are used by the Query tab when building requests to the LLM.
    """

    def __init__(self, parent, settings_mgr):
        super().__init__(parent)
        self.settings_mgr = settings_mgr
        self._build_ui()
        self._load_from_settings_or_defaults()
        self._wire_autosave()

    def _build_ui(self):
        ctl_frame = ttk.Frame(self)
        ctl_frame.pack(fill="x", padx=8, pady=4)

        load_prompts_btn = ttk.Button(
            ctl_frame,
            text="Load prompts from file",
            command=self.load_prompts_from_file,
        )
        load_prompts_btn.pack(side="left", padx=4)

        save_prompts_btn = ttk.Button(
            ctl_frame,
            text="Save prompts to file",
            command=self.save_prompts_to_file,
        )
        save_prompts_btn.pack(side="left", padx=4)

        sum_frame = ttk.LabelFrame(self, text="Summarizer prompt template")
        sum_frame.pack(fill="both", expand=True, padx=8, pady=4)

        sum_label = ttk.Label(
            sum_frame,
            text="Use <<BUG_TEXT>> where the full input description should go.",
        )
        sum_label.pack(anchor="w", padx=4, pady=2)

        self.summarizer_text = ScrolledText(sum_frame, wrap="word", height=12)
        self.summarizer_text.pack(fill="both", expand=True, padx=4, pady=4)

        chat_frame = ttk.LabelFrame(self, text="Answer prompt template")
        chat_frame.pack(fill="both", expand=True, padx=8, pady=4)

        chat_label = ttk.Label(
            chat_frame,
            text="Use <<BUG_TEXT>> for the investigation description and <<SNIPPETS>> for retrieved code snippets.",
        )
        chat_label.pack(anchor="w", padx=4, pady=2)

        self.chat_prompt_text = ScrolledText(chat_frame, wrap="word", height=14)
        self.chat_prompt_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _set_default_prompts(self):
        summarizer_default = (
            "You are a senior engineer helping with code search.\n\n"
            "You will be given a long input that may be a question, Jira ticket text, investigation notes, logs, or SQL.\n"
            "Sometimes it describes a bug, other times it is simply trying to understand how a feature or flow works.\n"
            "Your job is to rewrite it as a shorter query that preserves all important technical details needed\n"
            "to search the codebase.\n\n"
            "Full input:\n\n"
            "<<BUG_TEXT>>\n\n"
            "Task:\n"
            "1) Rewrite this as a concise search query for a codebase, suitable for either tracking down a bug\n"
            "   or exploring how the code implements a behavior.\n"
            "2) Keep all important technical details such as class names, endpoints, error messages,\n"
            "   stack trace fragments, config keys, error codes, i18n string keys, table and column names,\n"
            "   and SQL fragments.\n"
            "3) Never remove or alter i18n keys, table or column names, or any text that looks like SQL.\n"
            "4) Remove obvious noise such as greetings, scheduling chatter, and unrelated context.\n"
            "5) Aim for a few sentences or a short paragraph.\n"
            "6) Do not invent new information.\n"
        )

        chat_default = (
            "You are a senior engineer working on a large Java and Angular based retail point of sale system.\n\n"
            "You will be given:\n"
            "1) A description of what the user is investigating, which may include SQL, database results,\n"
            "   UI behavior, observed bugs, performance concerns, or general questions about how a feature works.\n"
            "2) A set of code snippets retrieved as relevant to that description.\n\n"
            "Your job:\n"
            "1) Explain what the relevant code is doing and how it fits into the overall flow of the system.\n"
            "2) Point to specific files, classes, methods, configuration keys, and database objects that matter.\n"
            "3) Describe how data and control move through the code, including important preconditions,\n"
            "   side effects, and error handling.\n"
            "4) If the description mentions a bug or suspicious behavior, explain the likely cause using\n"
            "   the snippets as evidence and outline what would need to change and where.\n"
            "5) When it makes sense, explain how to make the behavior more configurable so that future changes\n"
            "   can be done through database or configuration updates instead of code changes.\n\n"
            "Investigation description:\n"
            "<<BUG_TEXT>>\n\n"
            "Relevant code and documentation snippets:\n"
            "<<SNIPPETS>>\n\n"
            "Answer:\n"
            "1) Start by summarizing what the user seems to be trying to understand or debug.\n"
            "2) Then explain, in concrete terms, what the shown code does.\n"
            "3) Clearly separate:\n"
            "   a) Facts that are directly supported by the snippets.\n"
            "   b) Hypotheses or guesses that go beyond the snippets, and label them as such.\n"
        )

        self.summarizer_text.delete("1.0", "end")
        self.summarizer_text.insert("1.0", summarizer_default)

        self.chat_prompt_text.delete("1.0", "end")
        self.chat_prompt_text.insert("1.0", chat_default)

    def _load_from_settings_or_defaults(self):
        self._set_default_prompts()

        s = self.settings_mgr.data.get("prompts_tab") or {}
        summarizer = (s.get("summarizer_prompt") or "").strip()
        chat = (s.get("chat_prompt") or "").strip()

        if summarizer:
            self.summarizer_text.delete("1.0", "end")
            self.summarizer_text.insert("1.0", s.get("summarizer_prompt", ""))

        if chat:
            self.chat_prompt_text.delete("1.0", "end")
            self.chat_prompt_text.insert("1.0", s.get("chat_prompt", ""))

        self._save_prompts_to_settings(save_immediately=False)

    def _save_prompts_to_settings(self, save_immediately: bool):
        self.settings_mgr.data["prompts_tab"] = {
            "summarizer_prompt": self.summarizer_text.get("1.0", "end-1c"),
            "chat_prompt": self.chat_prompt_text.get("1.0", "end-1c"),
        }
        if save_immediately:
            self.settings_mgr.save_now()
        else:
            self.settings_mgr.save_soon()

    def _wire_autosave(self):
        def on_modified(widget):
            if widget.edit_modified():
                widget.edit_modified(False)
                self._save_prompts_to_settings(save_immediately=False)

        self.summarizer_text.bind("<<Modified>>", lambda e: on_modified(self.summarizer_text))
        self.chat_prompt_text.bind("<<Modified>>", lambda e: on_modified(self.chat_prompt_text))

    def load_prompts_from_file(self):
        path = filedialog.askopenfilename(
            title="Load prompts",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
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
            messagebox.showerror(
                "Error",
                "Prompt file must be JSON with 'summarizer' and 'chat' string fields.",
            )
            return

        self.summarizer_text.delete("1.0", "end")
        self.summarizer_text.insert("1.0", summarizer)

        self.chat_prompt_text.delete("1.0", "end")
        self.chat_prompt_text.insert("1.0", chat)

        self._save_prompts_to_settings(save_immediately=False)
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
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Could not save prompts:\n{e}")
            return

        self._save_prompts_to_settings(save_immediately=False)
        messagebox.showinfo("Saved", f"Prompts saved to:\n{path}")
