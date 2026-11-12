import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemnav.chunker import chunk_file  # noqa: E402
from gemnav.client import ScriptedClient  # noqa: E402
from gemnav.envs import LocalEnv  # noqa: E402
from gemnav.repair import Index, RepairAgent  # noqa: E402
from gemnav.vector_store import VectorStore  # noqa: E402

LIB = "def mean(xs):\n    \"\"\"Arithmetic mean.\"\"\"\n    return sum(xs) / (len(xs) - 1)\n\n\ndef total(xs):\n    return sum(xs)\n"
TEST = "from calc.stats import mean, total\n\ndef test_total():\n    assert total([1, 2]) == 3\n\ndef test_mean_single():\n    assert mean([4, 4]) == 4\n"


def make_repo(tmp):
    r = os.path.join(tmp, "repo")
    os.makedirs(os.path.join(r, "calc"))
    os.makedirs(os.path.join(r, "tests"))
    open(os.path.join(r, "calc", "__init__.py"), "w").write("")
    open(os.path.join(r, "calc", "stats.py"), "w").write(LIB)
    open(os.path.join(r, "tests", "test_stats.py"), "w").write(TEST)
    for c in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
        subprocess.run(["git", "-C", r, *c], check=True, capture_output=True)
    return r


def test_chunker_and_store(tmp_path):
    big_cls = "class Big:\n    x = 1\n" + "".join(f"    def m{i}(self):\n        return {i}\n" for i in range(120))
    ch = chunk_file("import os\n\n@dec\ndef f():\n    pass\n\n" + big_cls, "a.py")
    kinds = [c.node_type for c in ch]
    assert "function" in kinds and kinds.count("method") == 120 and "class_header" in kinds and "module" in kinds
    vs = VectorStore(str(tmp_path / "i.db"))
    vs.add_chunks(ch, [("f", "a.py", 3, "function", "")], {"a.py": ("x", 1, 1, 10)})
    assert vs.lookup_symbol("f")[0][1] == "a.py"


def test_repair_end_to_end_local(tmp_path):
    repo = make_repo(str(tmp_path))
    env = LocalEnv(repo)
    ix = Index(repo, str(tmp_path / "ix.db"), embedder=None, lsp=True)
    repro = "```python\nimport sys\nsys.path.insert(0, '.')\nfrom calc.stats import mean\nassert mean([2, 4]) == 3, mean([2, 4])\n```"
    acts = [
        repro,                                                             # phase-0 reproduction script (no grammar)
        json.dumps({"tool": "get_definition", "args": {"symbol": "mean"}}),
        json.dumps({"tool": "replace_in_file", "args": {"path": "tests/test_stats.py", "old": "== 4", "new": "== 2"}}),  # gaming
        json.dumps({"tool": "replace_in_file", "args": {"path": "calc/stats.py", "old": "def total(xs):\n    return sum(xs)", "new": "def total(xs):\n    return 0"}}),  # breaks regression
        json.dumps({"tool": "replace_in_file", "args": {"path": "calc/stats.py", "old": "return sum(xs) / (len(xs) - 1)", "new": "return sum(xs) / (len(xs) - 1"}}),  # syntax
        json.dumps({"tool": "replace_in_file", "args": {"path": "calc/stats.py", "old": "(len(xs) - 1)", "new": "len(xs)"}}),  # the fix
        json.dumps({"tool": "submit", "args": {}}),
    ]
    agent = RepairAgent(env, "mean([2, 4]) returns 6 instead of 3", ScriptedClient(acts), ix, max_steps=10)
    res = agent.run()
    ix.close()
    kinds = [e["kind"] for e in res["events"]]
    assert res["repro_status"] == "ALIGNED"
    assert res["creg_files"] == ["tests/test_stats.py"]
    assert res["guard_rejects"] == 1
    reasons = [e.get("reason") for e in res["events"] if e["kind"] == "edit_rollback"]
    assert reasons == ["score", "syntax"], reasons
    assert res["edits_committed"] == 1 and res["submitted"]
    assert res["final_score"] == [1, 0, 2]                                # regression clean, repro passes
    assert "-    return sum(xs) / (len(xs) - 1)" in res["model_patch"] and "+    return sum(xs) / len(xs)" in res["model_patch"]
    assert "tests/" not in res["model_patch"] and "return 0" not in res["model_patch"]
    assert "creg" in kinds and "checks" in kinds


def test_fuzzy_replace_reindents_and_refuses_ambiguity():
    from gemnav.repair import _closest, _fuzzy_replace
    cur = "class A:\n    def f(self):\n        x = 1\n        return x\n"
    out, how = _fuzzy_replace(cur, "x = 1\nreturn x", "x = 2\nreturn x * 2")      # model dropped indentation
    assert out == "class A:\n    def f(self):\n        x = 2\n        return x * 2\n", out
    out, how = _fuzzy_replace("a = 1\nb = 2\na = 1\n", "a  =  1", "a = 3")
    assert out is None and "2 times" in how
    assert "3:         x = 1" in _closest(cur, "x =1") or "x = 1" in _closest(cur, "x =1")


def test_search_literal_fallback(tmp_path):
    from gemnav.repair import _search
    (tmp_path / "m.py").write_text("def f(a, b):\n    return g(a\n")
    assert "m.py:1" in _search(str(tmp_path), "def f(a", ".")             # invalid regex -> literal
    assert "literal" in _search(str(tmp_path), "def f(a", ".")
    assert "m.py" in _search(str(tmp_path), "return g", "no/such/dir")      # bad path -> whole repo
    assert _search(str(tmp_path), "zzz_none", ".").startswith("no matches")
