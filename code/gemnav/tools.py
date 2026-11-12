"""Deterministic tools for the agent loop. Every tool returns a short, size-capped string.

Backends: ripgrep if installed (else a pure-python scan), Python `ast` symbol index (no ctags
dependency), BM25 over code entities for `semantic_search` (a dense-embedding backend can be
plugged in through `Toolbox.embedder`; it is NOT exercised in the paper's experiments).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swe_local.codegraph import build_graph, is_indexable  # noqa: E402
from swe_local.retrieval import BM25  # noqa: E402

MAX_OUT_CHARS = 3000
TOOL_NAMES = ["search_code", "get_definition", "find_references", "read_file", "list_symbols",
              "semantic_search", "get_file_structure", "get_callers"]


def _cap(s: str, n: int = MAX_OUT_CHARS) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


class Toolbox:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        files = {}
        for dp, dn, fn in os.walk(self.root):
            dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv", "venv")]
            for f in fn:
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                if is_indexable(rel):
                    try:
                        files[rel] = open(full, encoding="utf-8", errors="replace").read()
                    except OSError:
                        pass
        self.files = files
        self.graph = build_graph(files)
        self._bm25 = BM25(self.graph) if self.graph.entities else None
        self.has_rg = shutil.which("rg") is not None

    # -- path safety: tools may only read inside the repository root
    def _safe(self, rel: str) -> str:
        p = os.path.abspath(os.path.join(self.root, rel))
        if os.path.commonpath([p, self.root]) != self.root:
            raise ValueError("path escapes repository root")
        return p

    # ------------------------------------------------------------------ tools
    def search_code(self, pattern: str, path: str = ".") -> str:
        if self.has_rg:
            r = subprocess.run(["rg", "-n", "--no-heading", "-m", "5", "-e", pattern, self._safe(path)],
                               capture_output=True, text=True, cwd=self.root, timeout=20)
            out = r.stdout.replace(self.root + "/", "")
        else:
            rx = re.compile(pattern)
            hits = []
            for rel, src in self.files.items():
                if not rel.startswith(path.strip("./")):
                    continue
                for i, line in enumerate(src.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{i}:{line.strip()}")
                if len(hits) > 40:
                    break
            out = "\n".join(hits[:40])
        return _cap(out or "no matches")

    def get_definition(self, symbol: str) -> str:
        ids = self.graph.by_name.get(symbol.split(".")[-1], [])
        rows = [f"{e}  (lines {self.graph.entities[e].start}-{self.graph.entities[e].end})" for e in ids[:10]]
        return _cap("\n".join(rows) or "symbol not found")

    def find_references(self, symbol: str) -> str:
        return self.search_code(r"\b" + re.escape(symbol.split(".")[-1]) + r"\b")

    def get_callers(self, symbol: str) -> str:
        targets = set(self.graph.by_name.get(symbol.split(".")[-1], []))
        callers = [a if b in targets else b for k, a, b in self.graph.typed_edges
                   if k == "calls" and (b in targets or a in targets)]
        callers = [c for c in dict.fromkeys(callers) if c not in targets]
        return _cap("\n".join(callers[:30]) or "no callers found")

    def read_file(self, path: str, start: int | str = 1, end: int | str = 80) -> str:
        p = self._safe(path)
        start, end = int(start), min(int(end), int(start) + 200)
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        body = "\n".join(f"{i}: {l}" for i, l in enumerate(lines[start - 1 : end], start))
        return _cap(f"[{path} lines {start}-{min(end, len(lines))} of {len(lines)}]\n{body}")

    def list_symbols(self, path: str) -> str:
        rows = [f"{self.graph.entities[e].kind[:4]} {e.split('::', 1)[1]} L{self.graph.entities[e].start}"
                for e in self.graph._by_path.get(path, []) if e != path]  # type: ignore[attr-defined]
        return _cap("\n".join(rows) or "no symbols (file not indexed)")

    get_file_structure = list_symbols

    def semantic_search(self, query: str, k: int | str = 5) -> str:
        if not self._bm25:
            return "empty index"
        sc = self._bm25.scores(query)
        top = sorted(sc, key=lambda e: -sc[e])[: int(k)]
        return _cap("\n\n".join(f"{e}\n{self.graph.entities[e].text[:500]}" for e in top))

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, tool: str, args: dict) -> str:
        if tool not in TOOL_NAMES:
            return f"error: unknown tool {tool!r}"
        try:
            return str(getattr(self, tool)(**args))
        except TypeError as ex:
            return f"error: bad arguments for {tool}: {ex}"
        except Exception as ex:  # tool errors are observations, never crashes
            return f"error: {type(ex).__name__}: {ex}"

    def file_summary_map(self, max_files: int = 60) -> str:
        """One line per file: first docstring line (or top symbols). Deterministic, no LLM needed."""
        import ast
        rows = []
        for rel, src in sorted(self.files.items())[: max_files * 4]:
            try:
                doc = ast.get_docstring(ast.parse(src)) or ""
            except SyntaxError:
                doc = ""
            if doc:
                s = doc.strip().splitlines()[0][:80]
            else:
                names = [e.split("::", 1)[1] for e in self.graph._by_path.get(rel, []) if e != rel and "." not in e.split("::", 1)[1]]  # type: ignore[attr-defined]
                s = ", ".join(names[:6])
            rows.append(f"{rel}: {s}")
        return "\n".join(rows[:max_files])
