#!/usr/bin/env python3
"""
download_manager.py

A small download manager for Ollama model pulls.

Goals
1. Non blocking downloads (thread per download)
2. Streaming progress parsing from Ollama /api/pull
3. Simple callbacks so the UI can show progress bars
4. Cancel support
5. Safe to call from Tkinter using after

Ollama endpoint used
POST {OLLAMA_URL}/api/pull
Body: {"name": "<model>"}
Response: newline delimited JSON objects

Typical objects include keys like:
status, digest, total, completed, error

This module does not require Tkinter. It is UI agnostic.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import requests


@dataclass
class PullProgress:
    model: str
    status: str = ""
    digest: str = ""
    total: int = 0
    completed: int = 0
    percent: float = 0.0
    done: bool = False
    ok: bool = False
    error: str = ""
    updated_at_unix: float = field(default_factory=lambda: time.time())


ProgressCallback = Callable[[PullProgress], None]
DoneCallback = Callable[[PullProgress], None]


class DownloadHandle:
    """
    Returned from DownloadManager.start_pull.
    Can be used to cancel an in flight download.
    """

    def __init__(self, model: str):
        self.model = model
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()


class DownloadManager:
    """
    Manages concurrent pulls from an Ollama instance.

    Usage
        mgr = DownloadManager("http://localhost:11434")
        handle = mgr.start_pull(
            "deepseek-coder:6.7b",
            on_progress=my_progress_fn,
            on_done=my_done_fn,
        )
    """

    def __init__(self, ollama_url: str, timeout_seconds: int = 10):
        self.ollama_url = (ollama_url or "").rstrip("/")
        self.timeout_seconds = timeout_seconds

        self._lock = threading.Lock()
        self._active: Dict[str, DownloadHandle] = {}
        self._threads: Dict[str, threading.Thread] = {}

    def set_ollama_url(self, ollama_url: str) -> None:
        self.ollama_url = (ollama_url or "").rstrip("/")

    def active_models(self) -> list[str]:
        with self._lock:
            return sorted(self._active.keys())

    def is_active(self, model: str) -> bool:
        with self._lock:
            return model in self._active

    def cancel(self, model: str) -> bool:
        with self._lock:
            handle = self._active.get(model)
        if not handle:
            return False
        handle.cancel()
        return True

    def start_pull(
        self,
        model: str,
        on_progress: Optional[ProgressCallback] = None,
        on_done: Optional[DoneCallback] = None,
        headers: Optional[dict] = None,
    ) -> DownloadHandle:
        model = (model or "").strip()
        if not model:
            raise ValueError("model is required")

        with self._lock:
            if model in self._active:
                return self._active[model]

            handle = DownloadHandle(model)
            self._active[model] = handle

            t = threading.Thread(
                target=self._pull_worker,
                args=(handle, on_progress, on_done, headers),
                daemon=True,
            )
            self._threads[model] = t
            t.start()

        return handle

    def _pull_worker(
        self,
        handle: DownloadHandle,
        on_progress: Optional[ProgressCallback],
        on_done: Optional[DoneCallback],
        headers: Optional[dict],
    ) -> None:
        prog = PullProgress(model=handle.model)

        def emit_progress():
            if on_progress:
                try:
                    on_progress(prog)
                except Exception:
                    pass

        def emit_done():
            if on_done:
                try:
                    on_done(prog)
                except Exception:
                    pass

        base_url = self.ollama_url
        if not base_url:
            prog.done = True
            prog.ok = False
            prog.error = "Ollama URL is empty"
            emit_done()
            self._mark_inactive(handle.model)
            return

        url = f"{base_url}/api/pull"
        req_headers = headers or {}

        resp = None
        try:
            resp = requests.post(
                url,
                json={"name": handle.model},
                stream=True,
                timeout=self.timeout_seconds,
                headers=req_headers,
            )
        except requests.RequestException as e:
            prog.done = True
            prog.ok = False
            prog.error = f"Could not contact Ollama: {e}"
            emit_done()
            self._mark_inactive(handle.model)
            return

        if not resp.ok:
            body = ""
            try:
                body = (resp.text or "")[:800]
            except Exception:
                body = ""
            prog.done = True
            prog.ok = False
            prog.error = f"Ollama returned {resp.status_code}. {body}".strip()
            emit_done()
            self._mark_inactive(handle.model)
            return

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if handle.is_cancelled():
                    prog.status = "cancelled"
                    prog.done = True
                    prog.ok = False
                    prog.error = "Cancelled by user"
                    emit_progress()
                    emit_done()
                    self._mark_inactive(handle.model)
                    try:
                        resp.close()
                    except Exception:
                        pass
                    return

                if not raw_line:
                    continue

                obj = None
                try:
                    obj = json.loads(raw_line)
                except Exception:
                    continue

                if isinstance(obj, dict) and "error" in obj:
                    prog.status = "error"
                    prog.done = True
                    prog.ok = False
                    prog.error = str(obj.get("error") or "Unknown error").strip()
                    prog.updated_at_unix = time.time()
                    emit_progress()
                    emit_done()
                    self._mark_inactive(handle.model)
                    return

                status = obj.get("status") or obj.get("message") or ""
                digest = obj.get("digest") or ""
                total = obj.get("total") or 0
                completed = obj.get("completed") or 0

                prog.status = str(status)
                prog.digest = str(digest)
                prog.total = int(total) if isinstance(total, (int, float)) else prog.total
                prog.completed = int(completed) if isinstance(completed, (int, float)) else prog.completed

                if prog.total > 0:
                    prog.percent = max(0.0, min(100.0, (prog.completed * 100.0) / prog.total))

                prog.updated_at_unix = time.time()
                emit_progress()

                if isinstance(status, str) and "success" in status.lower():
                    prog.done = True
                    prog.ok = True
                    emit_done()
                    self._mark_inactive(handle.model)
                    return

            prog.done = True
            prog.ok = True if prog.percent >= 99.9 else False
            if not prog.ok and not prog.error:
                prog.error = "Pull ended without explicit success"
            emit_done()
            self._mark_inactive(handle.model)
            return

        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _mark_inactive(self, model: str) -> None:
        with self._lock:
            self._active.pop(model, None)
            self._threads.pop(model, None)


def format_bytes(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return "0 B"
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        f /= 1024.0
        if f < 1024.0:
            return f"{f:.1f} {u}"
    return f"{f:.1f} PB"
