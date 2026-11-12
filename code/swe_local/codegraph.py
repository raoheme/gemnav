"""Typed repository code graph built with Python's stdlib `ast`.

Entities (graph nodes)
  module   : one per .py file (module-level code outside defs)        id = "path"
  class    : class header + class-level statements (methods excluded) id = "path::Class"
  function : function / method full source                             id = "path::Class.meth"

Typed edges (stored undirected for traversal, typed for export)
  contains : module->class/function, class->method
  calls    : function->function/class by callee simple name (ambiguity-capped)
  imports  : module->module, module->symbol (from-imports)
  inherits : class->base class (ambiguity-capped)

Name resolution is deliberately conservative: a simple name that maps to more than
`max_ambiguity` entities produces no edge (avoids hubs like `get`, `save`).
"""
from __future__ import annotations

import ast
import builtins
import keyword
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

TEST_PAT = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
SKIP_PAT = re.compile(r"(^|/)(docs?|doc_src|examples?|benchmarks?|asv_bench|\.github)/")
BUILTIN_NAMES = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls", "args", "kwargs"}


def is_test_path(path: str) -> bool:
    return bool(TEST_PAT.search(path))


def is_indexable(path: str) -> bool:
    return path.endswith(".py") and not is_test_path(path) and not SKIP_PAT.search(path)


def path_to_module(path: str) -> str:
    p = path[:-3] if path.endswith(".py") else path
    for pre in ("src/", "lib/"):
        if p.startswith(pre):
            p = p[len(pre):]
    p = p.replace("/", ".")
    if p.endswith(".__init__"):
        p = p[: -len(".__init__")]
    return p


@dataclass
class Entity:
    id: str
    kind: str  # module | class | function
    name: str
    path: str
    start: int
    end: int
    text: str = ""
    stack_depth: int = 0


@dataclass
class CodeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    adj: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    typed_edges: list[tuple[str, str, str]] = field(default_factory=list)
    by_name: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    module_of: dict[str, str] = field(default_factory=dict)  # dotted module -> path
    parse_failures: int = 0

    def add_edge(self, kind: str, a: str, b: str) -> None:
        if a == b or a not in self.entities or b not in self.entities:
            return
        if b in self.adj[a]:
            return
        self.adj[a].add(b)
        self.adj[b].add(a)
        self.typed_edges.append((kind, a, b))

    def degree(self, n: str) -> int:
        return len(self.adj.get(n, ()))

    def bfs_dist(self, seeds: dict[str, float] | list[str], max_depth: int = 6) -> dict[str, int]:
        dist: dict[str, int] = {}
        q: deque[str] = deque()
        for s in seeds:
            if s in self.entities and s not in dist:
                dist[s] = 0
                q.append(s)
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in self.adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def innermost_entity(self, path: str, line: int) -> str | None:
        best = None
        best_span = 10**9
        for eid in self._by_path.get(path, ()):  # type: ignore[attr-defined]
            e = self.entities[eid]
            if e.kind == "module":
                continue
            if e.start <= line <= e.end and (e.end - e.start) < best_span:
                best, best_span = eid, e.end - e.start
        if best is None and path in self.entities:
            return path
        return best

    def to_metta(self, limit: int | None = None) -> str:
        """Export typed edges as MeTTa/AtomSpace expressions (OpenCog Hyperon syntax)."""
        name = {"contains": "Contains", "calls": "Calls", "imports": "Imports", "inherits": "Inherits"}
        out = []
        for i, (k, a, b) in enumerate(self.typed_edges):
            if limit is not None and i >= limit:
                break
            out.append(f'({name[k]} "{a}" "{b}")')
        return "\n".join(out)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, lines: list[str]):
        self.path = path
        self.lines = lines
        self.stack: list[str] = []  # qualified name parts
        self.defs: list[Entity] = []
        self.contains: list[tuple[str, str]] = []
        self.calls: dict[str, set[str]] = defaultdict(set)
        self.bases: dict[str, set[str]] = defaultdict(set)
        self.cur_func: list[str] = []
        self.imports: list[tuple[str, list[str] | None, int]] = []

    def _eid(self, name: str) -> str:
        return f"{self.path}::{'.'.join(self.stack + [name])}"

    def _parent(self) -> str:
        return f"{self.path}::{'.'.join(self.stack)}" if self.stack else self.path

    def _src(self, node: ast.AST) -> str:
        s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
        return "\n".join(self.lines[s - 1 : e])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        eid = self._eid(node.name)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        ent = Entity(eid, "class", node.name, self.path, start, node.end_lineno or node.lineno, stack_depth=len(self.stack))
        # class text: header + non-def statements
        parts = [self.lines[node.lineno - 1]]
        for st in node.body:
            if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.append(self._src(st))
        ent.text = "\n".join(parts)
        self.defs.append(ent)
        self.contains.append((self._parent(), eid))
        for b in node.bases:
            n = b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", None)
            if n:
                self.bases[eid].add(n)
        self.stack.append(node.name)
        self.cur_func.append(eid)
        self.generic_visit(node)
        self.cur_func.pop()
        self.stack.pop()

    def _func(self, node) -> None:
        eid = self._eid(node.name)
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        ent = Entity(eid, "function", node.name, self.path, start, node.end_lineno or node.lineno, stack_depth=len(self.stack))
        ent.text = "\n".join(self.lines[start - 1 : ent.end])
        self.defs.append(ent)
        self.contains.append((self._parent(), eid))
        self.stack.append(node.name)
        self.cur_func.append(eid)
        self.generic_visit(node)
        self.cur_func.pop()
        self.stack.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imports.append((a.name, None, 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append((node.module or "", [a.name for a in node.names], node.level or 0))

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        n = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if n and n not in BUILTIN_NAMES:
            owner = self.cur_func[-1] if self.cur_func else self.path
            self.calls[owner].add(n)
        self.generic_visit(node)


def build_graph(files: dict[str, str], max_ambiguity: int = 5) -> CodeGraph:
    """files: mapping repo-relative path -> source text (already filtered to indexable files)."""
    g = CodeGraph()
    by_path: dict[str, list[str]] = defaultdict(list)
    visitors: list[_Visitor] = []
    imports: list[tuple[str, str, list[str] | None, int]] = []  # (path, module, names, level)
    for path, src in files.items():
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError):
            g.parse_failures += 1
            continue
        mod = Entity(path, "module", path.rsplit("/", 1)[-1][:-3], path, 1, max(len(lines), 1))
        g.entities[path] = mod
        by_path[path].append(path)
        g.module_of[path_to_module(path)] = path
        v = _Visitor(path, lines)
        try:
            v.visit(tree)
        except RecursionError:
            g.parse_failures += 1
            continue
        covered = bytearray(len(lines) + 2)
        for e in v.defs:
            g.entities[e.id] = e
            by_path[path].append(e.id)
            g.by_name[e.name].append(e.id)
            if e.stack_depth == 0:  # top-level defs cover all nested ones
                hi = min(e.end, len(lines))
                covered[e.start : hi + 1] = b"\x01" * (hi + 1 - e.start)
        mod.text = "\n".join(l for i, l in enumerate(lines, 1) if not covered[i])[:4000]
        visitors.append(v)
        imports.extend((path, m, names, lvl) for (m, names, lvl) in v.imports)
    g._by_path = by_path  # type: ignore[attr-defined]

    def resolve(name: str, prefer_path: str | None = None) -> list[str]:
        cands = g.by_name.get(name, [])
        if prefer_path:
            local = [c for c in cands if g.entities[c].path == prefer_path]
            if local:
                return local
        return cands if 0 < len(cands) <= max_ambiguity else []

    for v in visitors:
        for a, b in v.contains:
            g.add_edge("contains", a, b)
        for owner, names in v.calls.items():
            for n in names:
                for tgt in resolve(n, v.path):
                    g.add_edge("calls", owner, tgt)
        for cls, bases in v.bases.items():
            for n in bases:
                for tgt in resolve(n, v.path):
                    if g.entities[tgt].kind == "class":
                        g.add_edge("inherits", cls, tgt)

    # top-level symbols per module path (for `from m import name` edges)
    top_level: dict[str, dict[str, str]] = defaultdict(dict)
    for eid, e in g.entities.items():
        if e.kind != "module" and "." not in eid.split("::", 1)[1]:
            top_level[e.path][e.name] = eid

    for path, module, names, level in imports:
        if level:
            base = path_to_module(path).split(".")
            # for a package __init__, path_to_module already dropped __init__
            drop = level if not path.endswith("__init__.py") else level - 1
            base = base[: len(base) - drop] if drop else base
            module = ".".join(base + ([module] if module else []))
        tgt_mod = g.module_of.get(module)
        if tgt_mod:
            g.add_edge("imports", path, tgt_mod)
        if names:
            for n in names:
                sub = g.module_of.get(f"{module}.{n}")
                if sub:
                    g.add_edge("imports", path, sub)
                    continue
                if tgt_mod and n in top_level.get(tgt_mod, {}):
                    g.add_edge("imports", path, top_level[tgt_mod][n])
    return g


# ----------------------------------------------------------------------------- issue anchoring
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_PATH = re.compile(r"[\w./-]+\.py")
_CODEY = re.compile(r"`([^`]+)`")


def _looks_like_code(tok: str) -> bool:
    return "_" in tok or (any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok))


def issue_seeds(g: CodeGraph, issue: str, max_ambiguity: int = 8) -> dict[str, float]:
    """Map identifiers / paths mentioned in an issue to graph entities with weights 1/ambiguity."""
    seeds: dict[str, float] = {}
    code_spans = " ".join(_CODEY.findall(issue))
    code_blocks = " ".join(re.findall(r"```(.*?)```", issue, flags=re.S))
    codeish_text = code_spans + " " + code_blocks

    def add(eids: list[str]) -> None:
        if not eids or len(eids) > max_ambiguity:
            return
        w = 1.0 / len(eids)
        for e in eids:
            seeds[e] = max(seeds.get(e, 0.0), w)

    # explicit file paths (also matches stack-trace frames)
    for m in _PATH.findall(issue):
        m = m.lstrip("./")
        hits = [p for p in g.entities if g.entities[p].kind == "module" and (p == m or p.endswith("/" + m) or m.endswith("/" + p) or m.endswith(p))]
        add(hits[:max_ambiguity])
    # dotted references: module, module.Symbol, Class.method
    for tok in set(_IDENT.findall(issue)):
        parts = tok.split(".")
        if len(parts) >= 2:
            for k in range(len(parts), 0, -1):
                mod = ".".join(parts[:k])
                if mod in g.module_of:
                    path = g.module_of[mod]
                    rest = parts[k:]
                    if not rest:
                        add([path])
                    else:
                        eid = f"{path}::{'.'.join(rest)}"
                        add([eid] if eid in g.entities else [path])
                    break
            else:
                # Class.method style
                q = ".".join(parts[-2:])
                cands = [e for e in g.by_name.get(parts[-1], []) if e.endswith("::" + q) or e.endswith("." + q)]
                add(cands)
        for p in parts:
            if p in BUILTIN_NAMES or len(p) < 3:
                continue
            cands = g.by_name.get(p)
            if not cands:
                continue
            if _looks_like_code(p) or re.search(r"\b" + re.escape(p) + r"\b", codeish_text) or re.search(re.escape(p) + r"\s*\(", issue) or (p[0].isupper() and any(g.entities[c].kind == "class" for c in cands)):
                add(cands)
    return seeds
