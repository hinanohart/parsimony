from __future__ import annotations

import numpy as np

from parsimony.adapters.embedding import HashingEmbedder


def test_deterministic_and_normalized():
    e = HashingEmbedder(dim=64, seed=0)
    a = e.encode("the quick brown fox")
    b = e.encode("the quick brown fox")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_similar_text_higher_similarity():
    e = HashingEmbedder(dim=256, seed=0)
    base = e.encode("the user enjoys hiking in the mountains")
    near = e.encode("the user enjoys hiking up the mountains")
    far = e.encode("quarterly revenue rose to four thousand dollars")
    assert float(base @ near) > float(base @ far)


def test_empty_text_zero_vector():
    e = HashingEmbedder(dim=32, seed=0)
    v = e.encode("")
    assert float(np.linalg.norm(v)) == 0.0


def test_batch_shape():
    e = HashingEmbedder(dim=16, seed=1)
    m = e.encode_batch(["a b", "c d", "e f"])
    assert m.shape == (3, 16)
