"""Policy configuration.

Every knob lives here so a single ``PolicyConfig`` (plus the seed) fully
determines behaviour. The ``a1`` defaults reduce the unified objective to a
pure W-TinyLFU utility (``w_freq=1``, others ``0``), the simplest calibratable
starting point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    capacity: int = 128
    seed: int = 0

    # --- frequency sketch (caffeine-style) ---
    cms_depth: int = 4
    sample_multiplier: int = 10
    doorkeeper_bits_per_item: int = 8
    doorkeeper_hashes: int = 2

    # --- semantic dedup (write-time) ---
    dedup_threshold: float = 0.92

    # --- unified objective weights (a1 default = pure W-TinyLFU utility) ---
    w_freq: float = 1.0
    w_cover: float = 0.0
    w_salience: float = 0.0

    # --- rate-distortion compression ---
    lambda0: float = 1.0
    compression_levels: tuple[int, ...] = (0, 1, 2)  # 0 verbatim, 1 skeleton, 2 drop-stub
    skeleton_keep_ratio: float = 0.5
    numeric_type_weight: float = 2.0
    default_type_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 <= self.dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be in [0, 1]")
        if self.cms_depth < 1:
            raise ValueError("cms_depth must be >= 1")

    def digest(self) -> str:
        """Stable short hash of the configuration (for trace reproducibility)."""
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()
