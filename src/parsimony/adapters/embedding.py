"""Embedding backends.

The default ``HashingEmbedder`` is deterministic, dependency-free (numpy only),
and torch-free, so benchmark numbers are reproducible by anyone without a model
download. For true semantic embeddings, ``SentenceTransformerEmbedder`` wraps
sentence-transformers (the ``[embed]`` extra) but is never imported by the core.
"""

from __future__ import annotations

import re

import numpy as np

from .._hashing import hash64

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Signed feature-hashing bag-of-tokens embedder (deterministic, L2-normalized)."""

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = hash64(tok, self.seed)
            idx = h % self.dim
            sign = 1.0 if ((h >> 33) & 1) else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0.0 else v

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    """Wrapper over sentence-transformers (optional, lazily imported)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import (  # ty: ignore[unresolved-import]
            SentenceTransformer,  # lazy optional dep ([embed] extra), not a core import
        )

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        v = np.asarray(self._model.encode(text, normalize_embeddings=True), dtype=np.float32)
        return v.ravel()
