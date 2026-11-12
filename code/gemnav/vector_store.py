"""SQLite index (blueprint schema) + numpy cosine search, with a content-addressed embedding cache.

Tables: chunks, embeddings (float32 BLOB), symbols, file_summaries — as in the blueprint — plus
`emb_cache(text_sha, model, vector)` so unchanged chunks are never re-embedded across commits/tasks.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, file_path TEXT, start_line INT, end_line INT,
    node_type TEXT, identifier TEXT, parent_scope TEXT, raw_text TEXT);
CREATE TABLE IF NOT EXISTS embeddings (chunk_id INTEGER, model TEXT, vector BLOB,
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id));
CREATE TABLE IF NOT EXISTS symbols (symbol_name TEXT, file_path TEXT, line INT, kind TEXT, scope TEXT);
CREATE TABLE IF NOT EXISTS file_summaries (file_path TEXT PRIMARY KEY, summary TEXT, export_count INT, import_count INT, loc INT);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(symbol_name);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
"""
CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS emb_cache (text_sha TEXT, model TEXT, vector BLOB, PRIMARY KEY (text_sha, model));"


def chunk_text_for_embedding(c) -> str:
    scope = f"{c.parent_scope}." if c.parent_scope else ""
    return f"{c.file_path} {scope}{c.identifier}\n{c.raw_text[:4000]}"


class VectorStore:
    def __init__(self, db_path: str, cache_path: str | None = None):
        self.db = sqlite3.connect(db_path)
        self.db.executescript(SCHEMA)
        self.cache = sqlite3.connect(cache_path) if cache_path else self.db
        self.cache.execute(CACHE_SCHEMA)
        self._mat: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add_chunks(self, chunks, symbols_rows, summaries: dict[str, tuple[str, int, int, int]]):
        cur = self.db.cursor()
        cur.execute("DELETE FROM chunks"); cur.execute("DELETE FROM embeddings")
        cur.execute("DELETE FROM symbols"); cur.execute("DELETE FROM file_summaries")
        cur.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                        [(i, c.file_path, c.start_line, c.end_line, c.node_type, c.identifier, c.parent_scope, c.raw_text)
                         for i, c in enumerate(chunks)])
        cur.executemany("INSERT INTO symbols VALUES (?,?,?,?,?)", symbols_rows)
        cur.executemany("INSERT INTO file_summaries VALUES (?,?,?,?,?)", [(k, *v) for k, v in summaries.items()])
        self.db.commit()
        self.chunks = list(chunks)
        self._mat.clear()

    def embed_all(self, embedder, batch: int = 64) -> dict:
        """Embed every chunk, reusing cached vectors. Returns timing/count stats."""
        t0 = time.perf_counter()
        texts = [chunk_text_for_embedding(c) for c in self.chunks]
        shas = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        cached = {}
        for i in range(0, len(shas), 900):
            q = shas[i:i + 900]
            rows = self.cache.execute(f"SELECT text_sha, vector FROM emb_cache WHERE model=? AND text_sha IN ({','.join('?' * len(q))})",
                                      [embedder.name, *q]).fetchall()
            cached.update({s: np.frombuffer(v, dtype=np.float32) for s, v in rows})
        todo = [i for i, s in enumerate(shas) if s not in cached]
        stats = {"chunks": len(texts), "cache_hits": len(texts) - len(todo)}
        if todo:
            vecs = embedder.embed([texts[i] for i in todo], batch=batch, stats=stats)
            self.cache.executemany("INSERT OR REPLACE INTO emb_cache VALUES (?,?,?)",
                                   [(shas[i], embedder.name, vecs[k].tobytes()) for k, i in enumerate(todo)])
            self.cache.commit()
            for k, i in enumerate(todo):
                cached[shas[i]] = vecs[k]
        mat = np.stack([cached[s] for s in shas]).astype(np.float32) if shas else np.zeros((0, embedder.dim), np.float32)
        self.db.executemany("INSERT INTO embeddings VALUES (?,?,?)", [(i, embedder.name, mat[i].tobytes()) for i in range(len(mat))])
        self.db.commit()
        self._mat[embedder.name] = (np.arange(len(mat)), mat)
        stats["total_seconds"] = time.perf_counter() - t0
        return stats

    def search(self, model: str, qvec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if model not in self._mat:
            rows = self.db.execute("SELECT chunk_id, vector FROM embeddings WHERE model=? ORDER BY chunk_id", (model,)).fetchall()
            ids = np.array([r[0] for r in rows])
            mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, len(qvec)), np.float32)
            self._mat[model] = (ids, mat)
        ids, mat = self._mat[model]
        if len(ids) == 0:
            return []
        s = mat @ qvec.astype(np.float32)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(ids[i]), float(s[i])) for i in top]

    def lookup_symbol(self, name: str) -> list[tuple]:
        return self.db.execute("SELECT symbol_name, file_path, line, kind, scope FROM symbols WHERE symbol_name=?", (name,)).fetchall()
