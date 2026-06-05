"""Deterministic hashing primitives.

The single integer ``seed`` is the only source of "randomness" in parsimony.
All hashes are platform-independent (blake2b + pure-int mixing), so traces are
reproducible across OSes and Python builds.
"""

from __future__ import annotations

import hashlib

_MASK64 = (1 << 64) - 1


def hash64(key: str, seed: int = 0) -> int:
    """Stable 64-bit hash of ``key`` under ``seed`` (blake2b, byte-exact)."""
    data = f"{seed}\x00{key}".encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def mix64(a: int, b: int) -> int:
    """splitmix64-style integer mixer. Deterministic, dependency-free."""
    x = (a ^ ((b * 0x9E3779B97F4A7C15) & _MASK64)) & _MASK64
    x = (x ^ (x >> 30)) & _MASK64
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x = (x ^ (x >> 27)) & _MASK64
    x = (x * 0x94D049BB133111EB) & _MASK64
    x = (x ^ (x >> 31)) & _MASK64
    return x


def next_pow2(n: int) -> int:
    """Smallest power of two >= max(n, 1)."""
    p = 1
    while p < n:
        p <<= 1
    return p
