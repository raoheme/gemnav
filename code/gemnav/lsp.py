"""Minimal synchronous LSP client for pylsp (Jedi), over stdio JSON-RPC. Fully offline.

Supports: initialize, didOpen, textDocument/definition, textDocument/references, documentSymbol.
Positions are 1-based lines / 0-based columns at the API boundary of this module (LSP itself is 0-based).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlparse
from pathlib import Path


def _uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _path(uri: str) -> str:
    p = unquote(urlparse(uri).path)
    if os.name == "nt" and p.startswith("/") and len(p) > 2 and p[2] == ":":
        p = p[1:]
    return p


class PylspClient:
    def __init__(self, root: str, timeout: float = 30.0):
        self.root = os.path.abspath(root)
        self.timeout = timeout
        cmd = [shutil.which("pylsp")] if shutil.which("pylsp") else [sys.executable, "-m", "pylsp"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.root)
        self._id = 0
        self._lock = threading.Lock()
        self._opened: set[str] = set()
        self.stats = {"requests": 0, "seconds": 0.0}
        self._request("initialize", {"processId": os.getpid(), "rootUri": _uri(self.root), "capabilities": {},
                                     "initializationOptions": {},
                                     "workspaceFolders": [{"uri": _uri(self.root), "name": "repo"}]})
        self._notify("initialized", {})
        # keep the server lean: no linters/formatters, only Jedi navigation
        self._notify("workspace/didChangeConfiguration", {"settings": {"pylsp": {"plugins": {
            k: {"enabled": False} for k in ("pycodestyle", "pyflakes", "mccabe", "pydocstyle", "autopep8", "yapf", "pylint", "flake8", "rope_completion")}}}})

    # ------------------------------------------------------------------ transport
    def _send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        self.p.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        self.p.stdin.flush()

    def _read(self) -> dict:
        headers = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("pylsp closed the connection")
            line = line.strip()
            if not line:
                break
            k, _, v = line.decode().partition(":")
            headers[k.lower()] = v.strip()
        return json.loads(self.p.stdout.read(int(headers["content-length"])))

    def _notify(self, method: str, params: dict) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict):
        t0 = time.perf_counter()
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                msg = self._read()
                if msg.get("id") == rid and ("result" in msg or "error" in msg):
                    self.stats["requests"] += 1
                    self.stats["seconds"] += time.perf_counter() - t0
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", "lsp error"))
                    return msg["result"]
                if "id" in msg and "method" in msg:  # server->client request: reply null
                    self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        raise TimeoutError(method)

    # ------------------------------------------------------------------ API
    def open(self, rel: str) -> None:
        full = os.path.join(self.root, rel)
        if full in self._opened:
            return
        text = open(full, encoding="utf-8", errors="replace").read()
        self._notify("textDocument/didOpen", {"textDocument": {"uri": _uri(full), "languageId": "python", "version": 1, "text": text}})
        self._opened.add(full)

    def changed(self, rel: str) -> None:
        """Re-open a file after an edit so Jedi sees the new contents."""
        full = os.path.join(self.root, rel)
        if full in self._opened:
            self._notify("textDocument/didClose", {"textDocument": {"uri": _uri(full)}})
            self._opened.discard(full)
        self.open(rel)

    def _locs(self, res) -> list[tuple[str, int]]:
        out = []
        for loc in res or []:
            uri = loc.get("uri") or loc.get("targetUri")
            rng = loc.get("range") or loc.get("targetSelectionRange")
            p = _path(uri)
            rel = os.path.relpath(p, self.root).replace(os.sep, "/") if p.startswith(self.root) else p
            out.append((rel, rng["start"]["line"] + 1))
        return out

    def definition(self, rel: str, line: int, col: int) -> list[tuple[str, int]]:
        self.open(rel)
        return self._locs(self._request("textDocument/definition", {
            "textDocument": {"uri": _uri(os.path.join(self.root, rel))}, "position": {"line": line - 1, "character": col}}))

    def references(self, rel: str, line: int, col: int) -> list[tuple[str, int]]:
        self.open(rel)
        return self._locs(self._request("textDocument/references", {
            "textDocument": {"uri": _uri(os.path.join(self.root, rel))}, "position": {"line": line - 1, "character": col},
            "context": {"includeDeclaration": False}}))

    def close(self) -> None:
        try:
            self._request("shutdown", {})
            self._notify("exit", {})
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
