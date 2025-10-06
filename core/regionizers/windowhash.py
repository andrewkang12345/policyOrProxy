"""Window hashing utilities that back the ego policy prototypes."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from policyOrProxy.core.world.arena import Arena


def quantize_window(window: np.ndarray, arena: Arena, grid_size: int, jitter: float = 0.0) -> np.ndarray:
    """Quantize a window of states onto a regular grid."""
    if window.ndim != 4:
        raise ValueError(f"expected window with 4 dims, got {window.shape}")
    pos = window[..., :2]
    norm = arena.normalize(pos)
    grid = np.round(norm * (grid_size - 1))
    if jitter > 0.0:
        noise = np.random.default_rng().normal(scale=jitter, size=grid.shape)
        grid += noise
    return grid.astype(np.int32)


def hash_window(quantized_window: np.ndarray, length_scale: float = 1.0) -> int:
    """Hash a quantized window using SHA1 and map it to an integer."""
    scaled = np.asarray(quantized_window, dtype=np.float32) * float(length_scale)
    data = scaled.tobytes(order="C")
    digest = hashlib.sha1(data).hexdigest()
    return int(digest[:16], 16)


def bucket_id(hash_value: int, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    return hash_value % num_buckets


@dataclass
class WindowHashRegionizer:
    arena: Arena
    num_buckets: int
    grid_size: int
    length_scale: float = 1.0
    jitter: float = 0.0
    prototypes: Optional[np.ndarray] = None
    prototype_weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self._cache: Dict[int, int] = {}

    def to_bucket(self, window: np.ndarray) -> int:
        quantized = quantize_window(window, self.arena, self.grid_size, jitter=self.jitter)
        hashed = hash_window(quantized, self.length_scale)
        if hashed not in self._cache:
            self._cache[hashed] = bucket_id(hashed, self.num_buckets)
        return self._cache[hashed]

    def register_prototypes(self, prototypes: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        if prototypes.shape[0] != self.num_buckets:
            raise ValueError("Prototype table must match number of buckets")
        self.prototypes = prototypes.astype(np.float32)
        if weights is not None:
            self.prototype_weights = weights.astype(np.float32)
        else:
            num_components = prototypes.shape[1]
            uniform = np.full((self.num_buckets, num_components), 1.0 / max(num_components, 1), dtype=np.float32)
            self.prototype_weights = uniform

    def get_actions(self, bucket: int, deterministic: bool = False, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if self.prototypes is None or self.prototype_weights is None:
            raise RuntimeError("Prototype table not registered")
        if rng is None:
            rng = np.random.default_rng()
        weights = self.prototype_weights[bucket]
        weights = weights / np.maximum(weights.sum(), 1e-6)
        if deterministic:
            choice = int(np.argmax(weights))
        else:
            choice = int(rng.choice(len(weights), p=weights))
        return self.prototypes[bucket, choice]

    def dump_cache(self) -> Dict[int, int]:
        return dict(self._cache)
