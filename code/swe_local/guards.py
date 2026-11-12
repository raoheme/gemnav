"""Deterministic anti-gaming guards and the non-gameable complexity proxy C(Delta).

Guards run on the candidate diff *before* any test is executed.
  protected_file_guard : reject edits/deletions of pre-existing tests and test/runner config,
                         and creation of config/startup files that can alter test collection.
  import_shadow_guard  : reject new top-level modules whose name shadows a stdlib module or a
                         module imported by the reproduction / regression tests.
  literal_matches      : non-trivial literals introduced by the patch that also occur in the
                         issue text or the reproduction test (hard-coding signal).
complexity(...)          : C = dN_add + 2 dN_branch + 0.5 dN_del + 3 N_literal  (full-file ASTs).
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field

from .codegraph import is_test_path
from .patches import parse_unified_diff

CONFIG_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", ".coveragerc",
                "noxfile.py", "setup.py", "sitecustomize.py", "usercustomize.py", "manage.py"}
STARTUP_NAMES = {"conftest.py", "sitecustomize.py", "usercustomize.py", "pytest.ini", "tox.ini"}
STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"test", "tests"}
TRIVIAL_NUMS = {0, 1, 2, -1, 10, 100, 0.0, 1.0, 0.5, 2.0}
TRIVIAL_STRS = {"", " ", "\n", "utf-8", "utf8", "ascii", "self", "name", "value", "default", "true", "false",
                "none", "error", "%s", "{}", ".", ",", ":", "/", "-", "_"}


@dataclass
class GuardVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    excluded_from_export: list[str] = field(default_factory=list)


def protected_file_guard(diff: str, existing_paths: set[str] | None = None) -> GuardVerdict:
    v = GuardVerdict(True)
    for fp in parse_unified_diff(diff):
        path = fp.path
        base = os.path.basename(path)
        existed = (not fp.is_new_file) if existing_paths is None else (fp.old_path in existing_paths)
        if existed and (is_test_path(path) or base in CONFIG_NAMES):
            v.ok = False
            v.reasons.append(f"PROTECTED_EDIT:{path}" if not fp.is_deleted_file else f"PROTECTED_DELETE:{path}")
        elif not existed and base in STARTUP_NAMES:
            v.ok = False
            v.reasons.append(f"CONFIG_INJECTION:{path}")
        elif not existed and is_test_path(path):
            v.excluded_from_export.append(path)
    return v


def _imported_modules(src: str) -> set[str]:
    out: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.add(n.module.split(".")[0])
    return out


def import_shadow_guard(diff: str, test_sources: list[str], package_roots: set[str] | None = None) -> GuardVerdict:
    """A new .py file (or package dir) at a sys.path root whose name collides with an import target."""
    v = GuardVerdict(True)
    imported = set().union(*[_imported_modules(s) for s in test_sources]) if test_sources else set()
    roots = package_roots or {""}
    for fp in parse_unified_diff(diff):
        if not fp.is_new_file or not fp.path.endswith(".py"):
            continue
        d, base = os.path.split(fp.path)
        if d in roots:
            mod = base[:-3]
        elif base == "__init__.py" and os.path.dirname(d) in roots:
            mod = os.path.basename(d)
        else:
            continue
        if mod in imported or mod in STDLIB:
            v.ok = False
            v.reasons.append(f"IMPORT_SHADOW:{fp.path}->{mod}")
    return v


_STR = re.compile(r"""(?:[rbuRBU]{0,2})("(?:[^"\\\n]|\\.){1,200}"|'(?:[^'\\\n]|\\.){1,200}')""")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")


def _literals_in(code: str) -> set[str]:
    lits: set[str] = set()
    for m in _STR.finditer(code):
        s = m.group(1)[1:-1]
        if len(s) >= 4 and s.lower() not in TRIVIAL_STRS:
            lits.add("s:" + s)
    for m in _NUM.finditer(_STR.sub('""', code)):
        try:
            x = float(m.group(0))
        except ValueError:
            continue
        if x not in TRIVIAL_NUMS:
            lits.add("n:" + m.group(0))
    return lits


def literal_matches(diff: str, issue: str, repro_test: str = "") -> list[str]:
    """Literals in added (non-comment) lines that the patch did not merely keep from deleted lines."""
    added, deleted = [], []
    for fp in parse_unified_diff(diff):
        if is_test_path(fp.path):
            continue
        added += [l for l in fp.added_lines if not l.strip().startswith("#")]
        deleted += fp.deleted_lines
    new = _literals_in("\n".join(added)) - _literals_in("\n".join(deleted))
    ref = issue + "\n" + repro_test
    hits = []
    for lit in sorted(new):
        kind, val = lit.split(":", 1)
        if kind == "s":
            if val in ref:
                hits.append(lit)
        else:
            if re.search(r"(?<![\w.])" + re.escape(val) + r"(?![\w.])", ref):
                hits.append(lit)
    return hits


_BRANCH = (ast.If, ast.For, ast.While, ast.Try, ast.IfExp, ast.BoolOp, ast.With, ast.comprehension,
           ast.AsyncFor, ast.AsyncWith, ast.ExceptHandler) + ((ast.Match,) if hasattr(ast, "Match") else ())


def _counts(src: str) -> tuple[int, int]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (0, 0)
    n = b = 0
    for node in ast.walk(tree):
        n += 1
        if isinstance(node, _BRANCH):
            b += 1
    return n, b


def complexity(before: dict[str, str], after: dict[str, str], diff: str, issue: str, repro_test: str = "") -> dict:
    """C(Delta) over full-file ASTs of every touched non-test .py file."""
    add = dele = br = 0
    for path in set(before) | set(after):
        if not path.endswith(".py") or is_test_path(path):
            continue
        n0, b0 = _counts(before.get(path, ""))
        n1, b1 = _counts(after.get(path, ""))
        add += max(0, n1 - n0)
        dele += max(0, n0 - n1)
        br += max(0, b1 - b0)
    lits = literal_matches(diff, issue, repro_test)
    c = add + 2 * br + 0.5 * dele + 3.0 * len(lits)
    return {"C": c, "ast_added": add, "ast_deleted": dele, "branches_added": br, "literal_matches": lits}
