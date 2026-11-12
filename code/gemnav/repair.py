"""Guarded multi-turn repair agent (gemnav). One instance = one SWE-bench task.

Pipeline
  index   : tree-sitter chunks -> SQLite store (+ dense embeddings, content-cached) ; BM25 over chunks ;
            deterministic file-summary map ; pylsp (Jedi) for definition/references.
  phase 0 : model writes a standalone reproduction script -> frozen OUTSIDE the repo (read-only) ->
            classified on the clean checkout: ALIGNED (fails without import/syntax errors) | PASSES | BROKEN.
            Regression suite C_reg (<= 2 visible test files for the top retrieved modules) is run once -> B0.
  turns   : one GBNF-constrained JSON action per model call (<= max_steps), sliding-window context.
            Every edit: protected-file guard -> apply on host -> sync -> py_compile -> checks ->
            score (regression_clean, -n_broken, R) -> commit if not worse, else git rollback (digest-verified).
  export  : `git diff <base>` restricted to non-test paths -> SWE-bench prediction.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time

from .chunker import chunk_file, symbols as chunk_symbols
from .context import toks
from .grammar import REPAIR_ARGS, ProtocolError, build_strict_gbnf, parse_action
from .vector_store import VectorStore, chunk_text_for_embedding

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swe_local.codegraph import is_indexable, is_test_path  # noqa: E402
from swe_local.guards import CONFIG_NAMES, STARTUP_NAMES  # noqa: E402
from swe_local.retrieval import rrf, tokenize  # noqa: E402
from swe_local.transaction import GitNativeTransaction, regression_clean, turn_score  # noqa: E402

SYSTEM = """You are an autonomous software engineer fixing a GitHub issue in a Python repository you cannot see directly.
Reply with exactly ONE JSON object per turn: {"tool": NAME, "args": {...}}.
Tools:
  semantic_search {query, k}        find relevant code chunks
  search_code {pattern, path}       ripgrep regex search
  get_definition {symbol}           where a function/class is defined (with code)
  find_references {symbol}          where a symbol is used (LSP)
  read_file {path, start, end}      read numbered lines (max 200)
  list_symbols {path}               outline of a file
  replace_in_file {path, old, new}  replace ONE exact occurrence of `old` (copy it exactly, with indentation; a few lines, < 600 chars) by `new`
  create_file {path, content}       create a new non-test source file
  run_tests {}                      run the reproduction script and regression tests
  submit {}                         finish; the current diff is your answer
Rules: never edit or create test files or test configuration; keep the change minimal and general (no special-casing of
values from the issue); read code before editing it; after an edit, check the TEST RESULT; submit when tests pass."""

REPRO_SYSTEM = ("Write ONE standalone Python script that reproduces the bug in the issue. It must exit with a non-zero "
                "status (an uncaught AssertionError or the exception from the issue) while the bug is present and exit 0 once "
                "it is fixed. Import the package normally; for Django call settings.configure() and django.setup() first. "
                "No network, no test frameworks. Keep it under 40 lines. Reply with a single ```python code block and nothing else.")

REPRO_BAD = re.compile(r"SyntaxError|ModuleNotFoundError|ImportError|IndentationError|NameError: name '\w+' is not defined|"
                       r"ImproperlyConfigured|AppRegistryNotReady")


class Index:
    def __init__(self, workdir: str, db_path: str, embedder=None, cache_path: str | None = None, lsp: bool = True):
        t0 = time.perf_counter()
        self.workdir = workdir
        self.files: dict[str, str] = {}
        for dp, dn, fn in os.walk(workdir):
            dn[:] = [d for d in dn if d not in (".git", "__pycache__", "node_modules", ".tox", "build", "dist")]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), workdir).replace(os.sep, "/")
                if is_indexable(rel):
                    try:
                        self.files[rel] = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
                    except OSError:
                        pass
        self.chunks = []
        for rel, src in self.files.items():
            try:
                self.chunks += chunk_file(src, rel)
            except Exception:
                pass
        self.stats = {"files": len(self.files), "chunks": len(self.chunks), "chunk_s": time.perf_counter() - t0}
        t0 = time.perf_counter()
        self.store = VectorStore(db_path, cache_path)
        self.summaries = self._summaries()
        self.store.add_chunks(self.chunks, chunk_symbols(self.chunks), self.summaries)
        self.stats["sqlite_s"] = time.perf_counter() - t0
        from .bm25 import ChunkBM25  # same implementation as the E5 benchmark
        t0 = time.perf_counter()
        self.bm25 = ChunkBM25([chunk_text_for_embedding(c) for c in self.chunks])
        self.stats["bm25_s"] = time.perf_counter() - t0
        self.embedder = embedder
        if embedder is not None:
            self.stats["embed"] = self.store.embed_all(embedder)
        self.lsp = None
        if lsp:
            t0 = time.perf_counter()
            try:
                from .lsp import PylspClient
                self.lsp = PylspClient(workdir)
            except Exception as ex:
                self.stats["lsp_error"] = str(ex)[:200]
            self.stats["lsp_init_s"] = time.perf_counter() - t0

    def _summaries(self):
        import ast
        out = {}
        for rel, src in self.files.items():
            try:
                tree = ast.parse(src)
                doc = (ast.get_docstring(tree) or "").strip().splitlines()
                names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
                imports = sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
            except (SyntaxError, ValueError, RecursionError):
                doc, names, imports = [], [], 0
            s = doc[0][:90] if doc else ", ".join(names[:6])
            out[rel] = (s, len(names), imports, src.count("\n") + 1)
        return out

    def hybrid(self, query: str, k: int = 20) -> list[int]:
        r_bm = self.bm25.rank(query)[:200]
        if self.embedder is None:
            return r_bm[:k]
        q = self.embedder.embed([query])[0]
        r_d = [i for i, _ in self.store.search(self.embedder.name, q, 200)]
        return rrf(r_bm, r_d)[:k]

    def file_map(self, ranked_chunks: list[int], budget_tokens: int = 700) -> str:
        files = []
        for i in ranked_chunks:
            p = self.chunks[i].file_path
            if p not in files:
                files.append(p)
        lines, used = [], 0
        for p in files:
            line = f"{p}: {self.summaries.get(p, ('',))[0]}"
            if used + toks(line) > budget_tokens:
                break
            lines.append(line)
            used += toks(line)
        return "\n".join(lines)

    def close(self):
        if self.lsp:
            self.lsp.close()


def _block(text: str) -> str | None:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", text, flags=re.S)   # tolerate an unterminated block
    code = m.group(1) if m else (text if "import" in text and "```" not in text else None)
    return code if code and code.strip() else None


class RepairAgent:
    def __init__(self, env, issue: str, client, index: Index, max_steps: int = 15, wall_s: int = 1800,
                 ctx_budget: int = 7000, test_timeout: int = 240, log=None):
        self.env, self.issue, self.client, self.ix = env, issue, client, index
        self.max_steps, self.wall_s, self.ctx_budget, self.test_timeout = max_steps, wall_s, ctx_budget, test_timeout
        self.t0 = time.time()
        self.log = log or (lambda *a: None)
        self.events: list[dict] = []
        self.grammar = build_strict_gbnf([t for t in REPAIR_ARGS if t not in ("get_file_structure", "get_callers")])
        self.allowed = {k: v for k, v in REPAIR_ARGS.items() if k not in ("get_file_structure", "get_callers")}
        self.tx = GitNativeTransaction(env.workdir)
        self.base = self._git("rev-parse", "HEAD").strip()
        self.repro_path, self.repro_status = None, "NONE"
        self.creg: list[str] = []
        self.b0: dict[str, str] = {}
        self.best = None
        self.edited: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _git(self, *a) -> str:
        return subprocess.run(["git", "-C", self.env.workdir, *a], capture_output=True, text=True).stdout

    def _ev(self, kind, **kw):
        kw.update(kind=kind, t=round(time.time() - self.t0, 2))
        self.events.append(kw)
        self.log(kind, {k: v for k, v in kw.items() if k not in ("kind",) and not isinstance(v, str) or len(str(v)) < 160})

    def _protected(self, rel: str, is_new: bool) -> str | None:
        base = os.path.basename(rel)
        if rel.startswith("/") or ".." in rel.split("/"):
            return "path escapes repository"
        if is_test_path(rel) or base.startswith("test_") or base.endswith("_test.py"):
            return "test files are protected"
        if base in CONFIG_NAMES or (is_new and base in STARTUP_NAMES):
            return "test/build configuration is protected"
        return None

    # ------------------------------------------------------------------ phase 0
    def synth_repro(self, ctx_text: str):
        t = time.time()
        out = self.client.chat(REPRO_SYSTEM, f"ISSUE:\n{self.issue[:4000]}\n\nRELEVANT CODE:\n{ctx_text[:6000]}", grammar=None, n_predict=1200)
        code = _block(out)
        if not code:
            self.repro_status = "NO_SCRIPT"
        else:
            self.repro_path = self.env.put_external("repro.py", code)
            rc, o = self.env.run(f"timeout 120 python {self.repro_path}", timeout=180)
            if rc == 0:
                self.repro_status = "PASSES_ON_BUGGY"   # unusable as a stopping signal
            elif REPRO_BAD.search(o) or rc == 124:
                self.repro_status = "BROKEN"
            else:
                self.repro_status = "ALIGNED"
            self.repro_baseline_out = o[-1500:]
        self._ev("repro", status=self.repro_status, s=round(time.time() - t, 1))

    def pick_creg(self, ranked_chunks: list[int]):
        mods = []
        for i in ranked_chunks:
            p = self.ix.chunks[i].file_path
            m = os.path.basename(p)[:-3]
            if m not in mods and m != "__init__":
                mods.append(m)
            if len(mods) >= 3:
                break
        cands = []
        for dp, dn, fn in os.walk(self.env.workdir):
            dn[:] = [d for d in dn if d != ".git"]
            for f in fn:
                if f.endswith(".py") and (f[5:-3] in mods if f.startswith("test_") else False):
                    cands.append(os.path.relpath(os.path.join(dp, f), self.env.workdir).replace(os.sep, "/"))
        self.creg = sorted(cands, key=lambda p: (len(p), p))[:2]
        t = time.time()
        self.b0 = self._run_creg() if self.creg else {}
        self._ev("creg", files=self.creg, n_tests=len(self.b0), n_pass=sum(v == "PASS" for v in self.b0.values()), s=round(time.time() - t, 1))

    def _run_creg(self) -> dict[str, str]:
        out = {}
        is_django = os.path.exists(os.path.join(self.env.workdir, "tests", "runtests.py"))
        for f in self.creg:
            if is_django and f.startswith("tests/"):
                label = f[len("tests/"):-3].replace("/", ".")
                rc, o = self.env.run(f"timeout {self.test_timeout} python tests/runtests.py {label} --verbosity 2 --parallel 1", self.test_timeout + 60)
                for m in re.finditer(r"^(\w+) \(([\w.]+)\)\s*(?:\n.*?)?\.\.\. (ok|FAIL|ERROR|skipped)", o, flags=re.M):
                    out[f"{m.group(2)}.{m.group(1)}"] = {"ok": "PASS", "FAIL": "FAIL", "ERROR": "ERROR", "skipped": "SKIP"}[m.group(3)]
            else:
                rc, o = self.env.run(f"timeout {self.test_timeout} python -m pytest -rA -q -p no:cacheprovider {f}", self.test_timeout + 60)
                for m in re.finditer(r"^(PASSED|FAILED|ERROR|SKIPPED) (\S+)", o, flags=re.M):
                    out[m.group(2)] = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERROR", "SKIPPED": "SKIP"}[m.group(1)]
        return out

    def checks(self) -> tuple[tuple, str]:
        t = time.time()
        r_status, r_out = "NO_REPRO", ""
        if self.repro_path and self.repro_status == "ALIGNED":
            rc, o = self.env.run(f"timeout 120 python {self.repro_path}", timeout=180)
            r_status = "PASS" if rc == 0 else ("TARGET_FAILURE" if not REPRO_BAD.search(o) else "OTHER")
            r_out = o[-800:]
        outcome = self._run_creg() if self.creg else {}
        ok, broken = regression_clean(self.b0, outcome)
        score = turn_score(self.b0, outcome, r_status)
        msg = (f"reproduction: {r_status}\n{r_out[-500:]}\nregression: {'clean' if ok else 'BROKE ' + ', '.join(broken[:5])} "
               f"({sum(v == 'PASS' for v in outcome.values())}/{len(outcome)} passing)")
        self._ev("checks", repro=r_status, regression_clean=ok, broken=len(broken), s=round(time.time() - t, 1))
        return score, msg

    # ------------------------------------------------------------------ tools
    def _code_at(self, rel, line, n=40):
        lines = self.ix.files.get(rel, "").splitlines() or open(os.path.join(self.env.workdir, rel), errors="replace").read().splitlines()
        return "\n".join(f"{i}: {l}" for i, l in enumerate(lines[line - 1: line - 1 + n], line))

    def tool(self, name: str, a: dict) -> str:
        wd = self.env.workdir
        if name == "semantic_search":
            ids = self.ix.hybrid(str(a.get("query", "")), int(a.get("k", 5) or 5))
            return "\n\n".join(f"{self.ix.chunks[i].file_path}:{self.ix.chunks[i].start_line}-{self.ix.chunks[i].end_line}\n"
                               f"{self.ix.chunks[i].raw_text[:900]}" for i in ids[:6])
        if name == "search_code":
            return _search(wd, str(a.get("pattern", "")), str(a.get("path") or "."))
        if name in ("get_definition", "find_references"):
            sym = str(a.get("symbol", "")).split(".")[-1]
            rows = self.ix.store.lookup_symbol(sym)
            if not rows:
                return "symbol not found in index"
            if name == "get_definition":
                return "\n\n".join(f"{r[1]}:{r[2]} ({r[3]}{' in ' + r[4] if r[4] else ''})\n{self._code_at(r[1], r[2], 30)}" for r in rows[:3])
            if self.ix.lsp is None:
                return "LSP unavailable"
            refs = []
            for r in rows[:2]:
                src = self.ix.files.get(r[1], "").splitlines()
                ln = r[2]
                for k in range(ln, min(ln + 5, len(src) + 1)):  # decorators: find the def line
                    col = src[k - 1].find(sym)
                    if col >= 0 and ("def " in src[k - 1] or "class " in src[k - 1]):
                        refs += self.ix.lsp.references(r[1], k, col)
                        break
            refs = sorted(set(refs))[:25]
            return "\n".join(f"{p}:{l}: {(self.ix.files.get(p, '').splitlines() or [''] * l)[l - 1].strip()[:120] if l - 1 < len(self.ix.files.get(p, '').splitlines()) else ''}" for p, l in refs) or "no references"
        if name == "read_file":
            rel = str(a.get("path", ""))
            p = os.path.normpath(os.path.join(wd, rel))
            if not p.startswith(os.path.normpath(wd)) or not os.path.isfile(p):
                return "error: no such file"
            s, e = int(a.get("start", 1) or 1), int(a.get("end", 0) or 0)
            e = min(e if e >= s else s + 79, s + 199)
            lines = open(p, errors="replace").read().splitlines()
            return f"[{rel} {s}-{min(e, len(lines))} of {len(lines)}]\n" + "\n".join(f"{i}: {l}" for i, l in enumerate(lines[s - 1:e], s))
        if name == "list_symbols":
            rel = str(a.get("path", ""))
            rows = [c for c in self.ix.chunks if c.file_path == rel and c.node_type != "module"]
            return "\n".join(f"{c.node_type} {c.parent_scope + '.' if c.parent_scope else ''}{c.identifier} L{c.start_line}-{c.end_line}" for c in rows[:80]) or "no symbols"
        if name in ("replace_in_file", "create_file"):
            return self.edit(name, a)
        if name == "run_tests":
            return self.checks()[1]
        return f"error: unknown tool {name}"

    def edit(self, name: str, a: dict) -> str:
        rel = str(a.get("path", "")).lstrip("./")
        full = os.path.join(self.env.workdir, rel)
        is_new = name == "create_file"
        why = self._protected(rel, is_new and not os.path.exists(full))
        if why:
            self._ev("guard_reject", path=rel, reason=why)
            return f"REJECTED: {why}"
        if is_new:
            if os.path.exists(full):
                return "error: file exists; use replace_in_file"
            new_text = str(a.get("content", ""))
        else:
            if not os.path.isfile(full):
                return "error: no such file"
            cur = open(full, encoding="utf-8", errors="replace").read()
            old, new = str(a.get("old", "")), str(a.get("new", ""))
            n = cur.count(old) if old else 0
            if n == 1:
                new_text = cur.replace(old, new, 1)
            else:
                new_text, how = _fuzzy_replace(cur, old, new)
                if new_text is None:
                    self._ev("edit_nomatch", path=rel, found=n, how=how)
                    return (f"error: `old` matched {n} times exactly and {how}. Copy `old` exactly from these lines "
                            f"of {rel} (line numbers are not part of the file):\n" + _closest(cur, old))
                self._ev("edit_fuzzy", path=rel, how=how)
        self.tx.begin({"edited": sorted(self.edited)})
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        self.env.sync([rel])
        rc, o = self.env.run(f"python -m py_compile {rel}", 120)
        if rc != 0:
            self.tx.rollback(); self.env.sync([rel]) if not is_new else self.env.sync([], [rel])
            self._ev("edit_rollback", path=rel, reason="syntax")
            return "ROLLED BACK (syntax error):\n" + o[-600:]
        score, msg = self.checks()
        if self.best is not None and score < self.best:
            self.tx.rollback()
            self.env.sync([rel]) if not is_new else self.env.sync([], [rel])
            self._ev("edit_rollback", path=rel, reason="score", score=list(score), best=list(self.best))
            return "ROLLED BACK (made checks worse):\n" + msg
        self.tx.commit(f"edit {rel}")
        self.best = score
        self.edited.add(rel)
        self._ev("edit_commit", path=rel, score=list(score))
        return "APPLIED.\nTEST RESULT:\n" + msg

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        q = self.issue[:3000]
        top = self.ix.hybrid(q, 30)
        ctx_chunks, used = [], 0
        for i in top:
            c = self.ix.chunks[i]
            t = min(toks(c.raw_text) + 12, 900)
            if used + t > 2400:
                continue
            ctx_chunks.append(f"### {c.file_path}:{c.start_line}-{c.end_line}\n{c.raw_text[:3600]}")
            used += t
        retrieved = "\n\n".join(ctx_chunks)
        fmap = self.ix.file_map(top, 600)
        self.synth_repro(retrieved)
        self.pick_creg(top)
        self.best = None
        if self.repro_status == "ALIGNED" or self.creg:
            self.best, _ = self.checks()
        history: list[tuple[str, str]] = []
        steps = errors = dups = 0
        seen: set[str] = set()
        submitted = False
        while steps < self.max_steps and time.time() - self.t0 < self.wall_s:
            hist_txt = ""
            for k, (act, obs) in enumerate(history):
                full = k >= len(history) - 2
                hist_txt += f"\nACTION {act}\nRESULT:\n{obs if full else obs[:160] + ' ...[collapsed]'}\n"
            user = (f"ISSUE:\n{self.issue[:2000]}\n\nREPRODUCTION SCRIPT STATUS: {self.repro_status}\n"
                    f"FILE MAP (most relevant first):\n{fmap}\n\nRETRIEVED CODE:\n{retrieved}\n\nHISTORY:{hist_txt or ' (none)'}\n\n"
                    f"Steps left: {self.max_steps - steps}. Next action (JSON):")
            while toks(SYSTEM) + toks(user) > self.ctx_budget and len(retrieved) > 800:
                retrieved = retrieved[: int(len(retrieved) * 0.8)]
                user = user.split("RETRIEVED CODE:")[0] + "RETRIEVED CODE:\n" + retrieved + "\n\nHISTORY:" + user.split("\n\nHISTORY:", 1)[1]
            t = time.time()
            raw = self.client.chat(SYSTEM, user, grammar=self.grammar, dry=True)
            steps += 1
            try:
                act = parse_action(raw, self.allowed)
            except ProtocolError as ex:
                errors += 1
                trunc = bool(getattr(self.client, "calls", None)) and self.client.calls[-1].get("finish") == "length"
                hint = " Your output was cut off: keep `old`/`new` short (a few lines, under 600 characters)." if trunc else ""
                history.append((raw[:200], f"invalid action: {ex}.{hint}"))
                self._ev("protocol_error", err=str(ex)[:120], raw=raw[:300], truncated=trunc)
                if errors >= 3:
                    break
                continue
            errors = 0
            if act.get("tool") == "submit":
                submitted = True
                self._ev("submit", step=steps)
                break
            t1 = time.time()
            key = json.dumps(act, sort_keys=True)
            if key in seen and act["tool"] not in ("run_tests",):
                dups += 1
                obs = "(duplicate of an earlier call - its result is in HISTORY; choose a different action)"
                self._ev("duplicate", name=act["tool"])
                history.append((json.dumps(act)[:600], obs))
                if dups >= 6:
                    break
                continue
            seen.add(key)
            try:
                obs = self.tool(act["tool"], act["args"])
            except Exception as ex:  # tool failures are observations
                obs = f"error: {type(ex).__name__}: {ex}"
            self._ev("tool", name=act["tool"], model_s=round(t1 - t, 2), tool_s=round(time.time() - t1, 2), obs_chars=len(obs))
            history.append((json.dumps(act)[:600], obs[:3000]))
        patch = self.export()
        return {"model_patch": patch, "steps": steps, "submitted": submitted, "repro_status": self.repro_status,
                "creg_files": self.creg, "edits_committed": sum(e["kind"] == "edit_commit" for e in self.events),
                "edits_rolled_back": sum(e["kind"] == "edit_rollback" for e in self.events),
                "guard_rejects": sum(e["kind"] == "guard_reject" for e in self.events),
                "duplicates": sum(e["kind"] == "duplicate" for e in self.events),
                "protocol_errors": sum(e["kind"] == "protocol_error" for e in self.events),
                "final_score": list(self.best) if self.best else None, "wall_s": time.time() - self.t0,
                "events": self.events}

    def export(self) -> str:
        names = [n for n in self._git("diff", "--name-only", self.base).split() if n]
        untracked = [n for n in self._git("ls-files", "--others", "--exclude-standard").split() if n]
        keep = [n for n in names + untracked if not self._protected(n, False) and not n.endswith((".pyc", ".orig", ".rej"))]
        if not keep:
            return ""
        if untracked:
            self._git("add", "-N", *[n for n in untracked if n in keep])
        return self._git("diff", self.base, "--", *keep)


def _norm(l: str) -> str:
    return " ".join(l.split())


def _fuzzy_replace(cur: str, old: str, new: str):
    """Unique line-sequence match ignoring whitespace differences; re-indents `new` to the file's indentation."""
    if not old.strip():
        return None, "`old` is empty"
    flines = cur.splitlines(keepends=True)
    olines = [l for l in old.splitlines()]
    while olines and not olines[0].strip():
        olines.pop(0)
    while olines and not olines[-1].strip():
        olines.pop()
    if not olines:
        return None, "`old` is blank"
    key = [_norm(l) for l in olines]
    hits = [i for i in range(len(flines) - len(key) + 1) if all(_norm(flines[i + k]) == key[k] for k in range(len(key)))]
    if len(hits) != 1:
        return None, f"matched {len(hits)} times ignoring whitespace"
    i = hits[0]
    file_ind = flines[i][: len(flines[i]) - len(flines[i].lstrip())]
    old_ind = olines[0][: len(olines[0]) - len(olines[0].lstrip())]
    nl = "\r\n" if flines[i].endswith("\r\n") else "\n"
    out = []
    for l in new.splitlines():
        if l.startswith(old_ind):
            l = file_ind + l[len(old_ind):]
        elif l.strip():
            l = file_ind + l.lstrip() if not old_ind else l
        out.append(l)
    repl = nl.join(out) + (nl if new.endswith("\n") or flines[i + len(key) - 1].endswith(("\n", "\r\n")) else "")
    return "".join(flines[:i]) + repl + "".join(flines[i + len(key):]), "whitespace-normalised match"


def _closest(cur: str, old: str, ctx: int = 6) -> str:
    import difflib
    lines = cur.splitlines()
    first = next((l for l in old.splitlines() if l.strip()), "")
    best = max(range(len(lines)), key=lambda i: difflib.SequenceMatcher(None, _norm(lines[i]), _norm(first)).ratio()) if lines else 0
    lo, hi = max(0, best - ctx), min(len(lines), best + ctx + 1)
    return "\n".join(f"{k + 1}: {lines[k]}" for k in range(lo, hi))


def _search(wd: str, pattern: str, path: str) -> str:
    if not pattern:
        return "error: empty pattern"
    path = path.lstrip("/") or "."
    if not os.path.exists(os.path.join(wd, path)):
        path = "."
    try:
        re.compile(pattern)
        fixed = False
    except re.error:
        fixed = True
    if _has_rg():
        cmd = ["rg", "-n", "--no-heading", "-m", "4", "-g", "*.py", "--max-columns", "200"] + (["-F"] if fixed else []) + ["-e", pattern, path]
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=30)
        out = r.stdout
        if not out and path != ".":
            out = subprocess.run(cmd[:-1] + ["."], cwd=wd, capture_output=True, text=True, timeout=30).stdout
    else:
        out = _pygrep(wd, re.escape(pattern) if fixed else pattern)
    note = " (searched as a literal string)" if fixed else ""
    return (out[:2500] + note) if out else f"no matches for {pattern!r}{note} in {path}"


_RG = None


def _has_rg():
    global _RG
    if _RG is None:
        import shutil
        _RG = shutil.which("rg") is not None
    return _RG


def _pygrep(wd, pat):
    rx = re.compile(pat)
    out = []
    for dp, dn, fn in os.walk(wd):
        dn[:] = [d for d in dn if d != ".git"]
        for f in fn:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                for i, l in enumerate(open(p, errors="replace"), 1):
                    if rx.search(l):
                        out.append(f"{os.path.relpath(p, wd)}:{i}:{l.strip()[:160]}")
                        if len(out) >= 40:
                            return "\n".join(out)
    return "\n".join(out)
