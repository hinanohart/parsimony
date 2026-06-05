"""Frequency estimation: a 4-bit Count-Min sketch with a doorkeeper.

This is a numpy port of caffeine's ``FrequencySketch`` idea:

* a ``depth x width`` table of 4-bit saturating counters (stored as ``uint8``,
  capped at 15);
* a one-pass *doorkeeper* bloom filter so one-hit-wonders never pollute the
  sketch (their first sighting only sets the doorkeeper);
* periodic *aging*: once the running sample size reaches ``sample_multiplier *
  capacity`` every counter is halved and the doorkeeper cleared, giving recency
  decay without TTLs.

All hashing is seeded and deterministic.
"""

from __future__ import annotations

import numpy as np

from ._hashing import hash64, mix64, next_pow2

_MAX = 15


class Doorkeeper:
    """A tiny seeded bloom filter that admits an item to the sketch on its 2nd sighting."""

    def __init__(self, capacity: int, seed: int = 0, bits_per_item: int = 8, num_hashes: int = 2):
        self.size = next_pow2(max(capacity, 1) * max(bits_per_item, 1))
        self.bits = np.zeros(self.size, dtype=bool)
        self.seed = seed ^ 0x9E3779B9
        self.k = max(num_hashes, 1)

    def _idx(self, key: str) -> list[int]:
        h = hash64(key, self.seed)
        return [mix64(h, i) & (self.size - 1) for i in range(self.k)]

    def contains(self, key: str) -> bool:
        return all(bool(self.bits[i]) for i in self._idx(key))

    def add(self, key: str) -> bool:
        """Insert ``key``; return True if it was already present."""
        idx = self._idx(key)
        present = all(bool(self.bits[i]) for i in idx)
        for i in idx:
            self.bits[i] = True
        return present

    def clear(self) -> None:
        self.bits[:] = False


class CountMinSketch4Bit:
    """Seeded, aging, 4-bit Count-Min frequency sketch."""

    def __init__(
        self,
        capacity: int,
        depth: int = 4,
        sample_multiplier: int = 10,
        seed: int = 0,
        doorkeeper_bits_per_item: int = 8,
        doorkeeper_hashes: int = 2,
    ):
        self.depth = max(depth, 1)
        self.width = next_pow2(max(capacity, 1))
        self.sample_size = max(sample_multiplier, 1) * max(capacity, 1)
        self.table = np.zeros((self.depth, self.width), dtype=np.uint8)
        self.size = 0
        self.seed = seed
        self._row_seeds = [mix64(seed, r + 1) for r in range(self.depth)]
        self.doorkeeper = Doorkeeper(
            capacity, seed, bits_per_item=doorkeeper_bits_per_item, num_hashes=doorkeeper_hashes
        )

    def _indices(self, key: str) -> list[int]:
        h = hash64(key, self.seed)
        return [mix64(h, self._row_seeds[r]) & (self.width - 1) for r in range(self.depth)]

    def increment(self, key: str) -> None:
        """Record one access of ``key``."""
        if not self.doorkeeper.add(key):
            # first ever sighting: doorkeeper only, do not touch the sketch
            self.size += 1
            self._maybe_age()
            return
        idx = self._indices(key)
        bumped = False
        for r in range(self.depth):
            c = int(self.table[r, idx[r]])
            if c < _MAX:
                self.table[r, idx[r]] = c + 1
                bumped = True
        if bumped:
            self.size += 1
            self._maybe_age()

    def estimate(self, key: str) -> int:
        """Estimated frequency of ``key`` in ``[0, 16]``."""
        idx = self._indices(key)
        m = min(int(self.table[r, idx[r]]) for r in range(self.depth))
        return m + (1 if self.doorkeeper.contains(key) else 0)

    def _maybe_age(self) -> None:
        if self.size >= self.sample_size:
            odd = int(np.count_nonzero(self.table & 1))
            self.table >>= 1  # uint8 halving
            self.size = (self.size - (odd >> 2)) >> 1
            self.doorkeeper.clear()
