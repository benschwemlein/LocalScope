# settings_store.py
import json
from pathlib import Path
from typing import Any, Callable

CONFIG_PATH = Path.home() / ".local-rag-llm" / "config.json"


def _deep_merge(defaults: dict, loaded: dict) -> dict:
    out = dict(defaults)
    for k, v in (loaded or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class SettingsManager:
    def __init__(self, tk_root, defaults: dict):
        self._tk = tk_root
        self._defaults = defaults
        self.data = _deep_merge(defaults, self._load_file())
        self._pending_after_id = None
        self._subscribers: list[Callable[[dict], Any]] = []

    def _load_file(self) -> dict:
        if not CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_now(self):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def save_soon(self, delay_ms: int = 350):
        if self._pending_after_id is not None:
            try:
                self._tk.after_cancel(self._pending_after_id)
            except Exception:
                pass
        self._pending_after_id = self._tk.after(delay_ms, self._flush_save)

    def _flush_save(self):
        self._pending_after_id = None
        self.save_now()

    def subscribe(self, fn: Callable[[dict], Any]):
        self._subscribers.append(fn)

    def notify_all(self):
        for fn in self._subscribers:
            fn(self.data)
