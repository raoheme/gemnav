import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemnav.client import ScriptedClient  # noqa: E402
from gemnav.context import Context, toks  # noqa: E402
from gemnav.grammar import GBNF, ProtocolError, parse_action  # noqa: E402
from gemnav.loop import run_agent  # noqa: E402
from gemnav.tools import Toolbox  # noqa: E402
from swe_local.codegraph import build_graph, issue_seeds  # noqa: E402
from swe_local.guards import complexity, import_shadow_guard, literal_matches, protected_file_guard  # noqa: E402
from swe_local.patches import parse_unified_diff  # noqa: E402
from swe_local.retrieval import BM25, graph_boost_rank  # noqa: E402
from swe_local.router import LogisticRouter, MLPRouter, PCRouter, auroc  # noqa: E402
from swe_local.transaction import GitNativeTransaction, RollbackIntegrityError, regression_clean, turn_score  # noqa: E402

SRC = {
    "pkg/__init__.py": '"""pkg"""\n',
    "pkg/core.py": "from pkg.util import helper\n\nclass Base:\n    def run(self):\n        return 1\n\nclass Child(Base):\n    def go(self):\n        return helper(3)\n",
    "pkg/util.py": "def helper(x):\n    return x * 2\n\ndef unused():\n    pass\n",
}


def test_graph_edges_and_gold_lookup():
    g = build_graph(SRC)
    kinds = {k for k, _, _ in g.typed_edges}
    assert {"contains", "calls", "imports", "inherits"} <= kinds
    assert ("calls", "pkg/core.py::Child.go", "pkg/util.py::helper") in g.typed_edges
    assert g.innermost_entity("pkg/util.py", 2) == "pkg/util.py::helper"
    assert "Inherits" in g.to_metta()


def test_seeds_and_boost_reaches_lexically_silent_neighbour():
    g = build_graph(SRC)
    seeds = issue_seeds(g, "Calling `Child.go` returns the wrong value")
    assert "pkg/core.py::Child.go" in seeds
    dist = g.bfs_dist(seeds)
    assert dist["pkg/util.py::helper"] == 1
    bm = BM25(g).scores("wrong value")
    r = graph_boost_rank(g, seeds, bm, dist)
    assert "pkg/util.py::helper" in r and "pkg/util.py::unused" not in r[: r.index("pkg/util.py::helper")]


DIFF_TEST = "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,1 +1,1 @@\n-assert f(1) == 2\n+assert True\n"
DIFF_IMPL = "--- a/pkg/util.py\n+++ b/pkg/util.py\n@@ -1,2 +1,2 @@\n def helper(x):\n-    return x * 2\n+    return x * 3\n"


def test_guards():
    assert protected_file_guard(DIFF_IMPL).ok
    v = protected_file_guard(DIFF_IMPL + DIFF_TEST)
    assert not v.ok and "PROTECTED_EDIT:tests/test_x.py" in v.reasons
    new_conf = "--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1,1 @@\n+import os\n"
    assert not protected_file_guard(DIFF_IMPL + new_conf).ok
    shadow = "--- /dev/null\n+++ b/requests.py\n@@ -0,0 +1,1 @@\n+x=1\n"
    assert not import_shadow_guard(shadow, ["import requests"]).ok
    assert import_shadow_guard(DIFF_IMPL, ["import requests"]).ok


def test_literal_and_complexity():
    issue = "f(4242) must return 'ValueError raised'"
    hc = "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,3 @@\n def f(x):\n+    if x == 4242:\n+        return 'ValueError raised'\n"
    assert len(literal_matches(hc, issue)) == 2
    assert literal_matches(DIFF_IMPL, issue) == []
    before = {"m.py": "def f(x):\n    return x\n"}
    after = {"m.py": "def f(x):\n    return x + 1\n"}
    c = complexity(before, after, DIFF_IMPL, issue)
    assert c["C"] > 0 and c["literal_matches"] == []


def test_diff_parser_old_lines():
    fp = parse_unified_diff(DIFF_IMPL)[0]
    assert fp.old_changed_lines == {2} and fp.path == "pkg/util.py"


def _git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = str(tmp_path / "r")
    os.makedirs(r)
    _git(r, "init", "-q")
    open(r + "/a.py", "w").write("a=1\n")
    open(r + "/b.py", "w").write("b=1\n")
    _git(r, "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return r


def test_rollback_restores_tracked_untracked_and_index(repo):
    tx = GitNativeTransaction(repo)
    d0 = tx.begin({"t": 1})
    open(repo + "/a.py", "a").write("x\n")
    os.remove(repo + "/b.py")
    open(repo + "/new.py", "w").write("n\n")
    _git(repo, "add", "a.py")
    open(repo + "/a.py", "a").write("y\n")
    out = tx.rollback()
    assert out["digest"] == d0 and not os.path.exists(repo + "/new.py") and open(repo + "/a.py").read() == "a=1\n"


def test_rollback_detects_state_tamper(repo):
    tx = GitNativeTransaction(repo)
    tx.begin({"t": 1})
    tx._state = {"t": 2}
    with pytest.raises(RollbackIntegrityError):
        tx.rollback()


def test_commit_then_rollback(repo):
    tx = GitNativeTransaction(repo)
    tx.begin()
    open(repo + "/k.py", "w").write("k\n")
    tx.commit()
    tx.begin()
    open(repo + "/d.py", "w").write("d\n")
    tx.rollback()
    assert os.path.exists(repo + "/k.py") and not os.path.exists(repo + "/d.py")


def test_baseline_identity_and_score():
    b0 = {"t1": "PASS", "t2": "FAIL", "t3": "PASS"}
    ok, broken = regression_clean(b0, {"t1": "PASS", "t2": "PASS", "t3": "FAIL"})
    assert not ok and broken == ["t3"]  # fixing t2 does not excuse breaking t3 (failure swapping)
    assert turn_score(b0, {"t1": "PASS", "t2": "PASS", "t3": "PASS"}, "PASS") > turn_score(b0, {"t1": "PASS", "t3": "FAIL"}, "PASS")


def test_routers_learn_separable_data_and_auroc():
    rng = __import__("numpy").random.default_rng(0)
    X = rng.normal(size=(300, 10))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    for M in (LogisticRouter, MLPRouter, PCRouter):
        m = M(seed=0).fit(X[:200], y[:200])
        assert auroc(y[200:], m.predict_proba(X[200:])) > 0.9, M.__name__
    assert auroc(__import__("numpy").array([0, 1]), __import__("numpy").array([0.1, 0.9])) == 1.0


# ---------------------------------------------------------------- gemnav
@pytest.fixture
def tb(tmp_path):
    for p, s in SRC.items():
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(s)
    return Toolbox(str(tmp_path))


def test_grammar_and_parse():
    assert "tool-call" in GBNF and "get_callers" in GBNF
    assert parse_action('{"answer": "x"}')["answer"] == "x"
    assert parse_action('{"tool": "read_file", "args": {"path": "a.py"}}')
    for bad in ['{"tool": "rm", "args": {}}', '{"tool": "read_file", "args": {"evil": "1"}}', "not json", "[1]", '{"answer": 3}']:
        with pytest.raises(ProtocolError):
            parse_action(bad)


def test_tools(tb):
    assert "pkg/util.py::helper" in tb.get_definition("helper")
    assert "pkg/core.py::Child.go" in tb.get_callers("helper")
    assert "1: def helper" in tb.read_file("pkg/util.py", 1, 2)
    assert "error" in tb.dispatch("read_file", {"path": "../../etc/passwd"})
    assert "error" in tb.dispatch("nope", {})
    assert "helper" in tb.search_code("def helper")
    assert "pkg/util.py" in tb.file_summary_map()


def test_loop_end_to_end_with_scripted_client(tb):
    c = ScriptedClient([
        json.dumps({"tool": "get_definition", "args": {"symbol": "helper"}}),
        json.dumps({"tool": "get_definition", "args": {"symbol": "helper"}}),  # duplicate
        "garbage",
        json.dumps({"tool": "get_callers", "args": {"symbol": "helper"}}),
        json.dumps({"answer": "helper is defined in pkg/util.py:1 and called by Child.go"}),
    ])
    res = run_agent("Where is helper used?", tb, c, max_iters=8)
    assert "Child.go" in res["answer"] and res["iterations"] == 5
    assert any("duplicate" in p for p in c.prompts)
    assert any("invalid" in p for p in c.prompts)


def test_context_stays_under_budget():
    ctx = Context("q", "f.py: x\n" * 50, budget=800)
    for i in range(12):
        ctx.add("read_file", {"path": f"f{i}.py"}, "line\n" * 2000)
        assert toks(ctx.render()) <= 800
    assert "HISTORY" in ctx.render()


def test_loop_stops_after_repeated_protocol_errors(tb):
    res = run_agent("q", tb, ScriptedClient(["x"] * 10), max_iters=10)
    assert res["answer"] is None and res["iterations"] == 3


def test_diff_parser_pure_insertion_and_replacement():
    ins = "--- a/m.py\n+++ b/m.py\n@@ -3,2 +3,3 @@\n a\n+new\n b\n"
    assert parse_unified_diff(ins)[0].old_changed_lines == {3}  # after line 3, not before line 4
    rep = "--- a/m.py\n+++ b/m.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n+B2\n c\n"
    assert parse_unified_diff(rep)[0].old_changed_lines == {2}
