from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from policyOrProxy.core.world.arena import Arena


def quantize_window(
    window: np.ndarray,
    arena: Arena,
    grid_size: int,
    *,
    jitter: float = 0.0,
    length_scale: float = 1.0,   # kept for signature symmetry; hashing scales separately
    clamp: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if window.ndim != 4:
        raise ValueError(f"expected window with 4 dims, got {window.shape}")

    rng = rng or np.random.default_rng()

    pos = window[..., :2]
    norm = arena.normalize(pos)  # not clamped unless requested below
    grid = np.round(norm * (grid_size - 1))

    if jitter > 0.0:
        grid = grid + rng.normal(scale=jitter, size=grid.shape)

    if clamp:
        grid = np.clip(grid, 0.0, float(grid_size - 1))

    return grid.astype(np.int32)


def hash_window(quantized_window: np.ndarray, length_scale: float = 1.0) -> int:
    scaled = np.asarray(quantized_window, dtype=np.float32) * float(length_scale)
    data = scaled.tobytes(order="C")
    digest = hashlib.sha1(data).hexdigest()
    return int(digest[:16], 16)


def bucket_id(hash_value: int, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    return int(hash_value % num_buckets)


@dataclass
class WindowHashRegionizer:
    arena: Arena
    num_buckets: int
    grid_size: int
    length_scale: float = 1.0
    jitter: float = 0.0
    clamp: bool = True

    prototypes: Optional[np.ndarray] = None  # (buckets, K, agents, 2)

    def __post_init__(self) -> None:
        self._cache: Dict[int, int] = {}

    def to_bucket(self, window: np.ndarray, *, rng: Optional[np.random.Generator] = None) -> int:
        q = quantize_window(
            window,
            self.arena,
            self.grid_size,
            jitter=self.jitter,
            clamp=self.clamp,
            rng=rng,
        )
        h = hash_window(q, self.length_scale)
        if h not in self._cache:
            self._cache[h] = bucket_id(h, self.num_buckets)
        return self._cache[h]

    def register_prototypes(self, prototypes: np.ndarray) -> None:
        if prototypes.shape[0] != self.num_buckets:
            raise ValueError("Prototype table must match number of buckets")
        self.prototypes = prototypes.astype(np.float32)

    def get_action(self, bucket: int, component: int) -> np.ndarray:
        if self.prototypes is None:
            raise RuntimeError("Prototypes not initialized")
        return self.prototypes[bucket, component]
