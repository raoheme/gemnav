"""swe_local: graph-guided retrieval and guarded repair harness for small local coding agents.

Modules
-------
codegraph   : AST-based typed code graph (files/classes/functions; contains/calls/imports/inherits).
retrieval   : BM25, graph-propagation and hybrid retrievers over code entities, token-budgeted context.
router      : per-task retrieval router (predictive-coding network, logistic regression, MLP).
guards      : protected-file diff guard, import-shadowing guard, literal-overfit detector, complexity C(Delta).
transaction : git-native transactional checkpoint/rollback with canonical state digest.
patches     : unified-diff parsing helpers.
"""
__version__ = "1.0.0"
