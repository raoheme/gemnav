"""Local embedding models (no network at query time once weights are cached).

Registry of the four offline embedders compared in the paper. Backend selection:
  * "onnx"  : onnxruntime CPUExecutionProvider + HF `tokenizers` (MiniLM; Jina if an ONNX file is present)
  * "torch" : transformers AutoModel (CPU or CUDA) — used for CodeBERT / GraphCodeBERT and as fallback
Pooling is mean-pooling over the attention mask followed by L2 normalisation for every model, so cosine
similarity == dot product. Texts are truncated to `max_len` tokens; batches of 64 by default.
"""
from __future__ import annotations

import os
import time

import numpy as np

REGISTRY = {
    # name: (hf_repo, onnx_file_or_None, max_len, dim, trust_remote_code)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "onnx/model.onnx", 256, 384, False),
    "codebert": ("microsoft/codebert-base", None, 512, 768, False),
    "graphcodebert": ("microsoft/graphcodebert-base", None, 512, 768, False),
    "jina-code": ("jinaai/jina-embeddings-v2-base-code", "onnx/model.onnx", 512, 768, True),
}


def _mean_pool(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


class Embedder:
    def __init__(self, name: str, backend: str | None = None, device: str = "cpu", max_len: int | None = None,
                 threads: int | None = None, cache_dir: str | None = None):
        repo, onnx_file, ml, dim, trust = REGISTRY[name]
        self.name, self.dim, self.max_len = name, dim, max_len or ml
        self.backend = backend or ("onnx" if onnx_file else "torch")
        from huggingface_hub import hf_hub_download, snapshot_download
        if self.backend == "onnx":
            import onnxruntime as ort
            from tokenizers import Tokenizer
            path = hf_hub_download(repo, onnx_file, cache_dir=cache_dir)
            tok = hf_hub_download(repo, "tokenizer.json", cache_dir=cache_dir)
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if str(device).startswith("cuda") else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(path, so, providers=prov)
            self.providers = self.sess.get_providers()
            self.inputs = {i.name for i in self.sess.get_inputs()}
            self.tok = Tokenizer.from_file(tok)
            self.tok.enable_truncation(self.max_len)
            self.tok.enable_padding()
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if threads:
                torch.set_num_threads(threads)
            local = snapshot_download(repo, cache_dir=cache_dir,
                                      allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.py", "*.model", "merges.txt", "vocab*"])
            self.tok = AutoTokenizer.from_pretrained(local, trust_remote_code=trust)
            self.model = AutoModel.from_pretrained(local, trust_remote_code=trust).to(device).eval()
            self.device = device
            self.torch = torch

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self.backend == "onnx":
            enc = self.tok.encode_batch(texts)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            out = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            return _mean_pool(out, mask)
        with self.torch.inference_mode():
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state.float().cpu().numpy()
            return _mean_pool(h, enc["attention_mask"].cpu().numpy())

    def embed(self, texts: list[str], batch: int = 64, stats: dict | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        # sort by length so padding is minimal, then restore order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            out[idx] = self.embed_batch([texts[i] for i in idx])
        if stats is not None:
            stats["embed_seconds"] = stats.get("embed_seconds", 0.0) + time.perf_counter() - t0
            stats["embedded"] = stats.get("embedded", 0) + len(texts)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


def available() -> list[str]:
    return list(REGISTRY)


if __name__ == "__main__":  # quick self-check: python -m gemnav.embedder minilm
    import sys
    e = Embedder(sys.argv[1] if len(sys.argv) > 1 else "minilm", threads=os.cpu_count())
    v = e.embed(["def add(a, b): return a + b", "def sum_two(x, y): return x + y", "class HTTPServer: pass"])
    print(e.name, e.backend, v.shape, "sim(add,sum)=%.3f sim(add,http)=%.3f" % (v[0] @ v[1], v[0] @ v[2]))
