"""Window-hash prototype policy used for the ego team."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer

LOGGER = logging.getLogger(__name__)


@dataclass
class WindowHashPolicy:
    regionizer: WindowHashRegionizer
    num_agents: int
    num_prototypes: int
    max_speed: float
    noise_std: float = 0.0
    sampling: str = "stochastic"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        prototypes = self.rng.uniform(-1.0, 1.0, size=(self.regionizer.num_buckets, self.num_prototypes, self.num_agents, 2)).astype(np.float32)
        prototypes *= self.max_speed
        self.regionizer.register_prototypes(prototypes)
        self.identifier = "ego_window_hash"
        LOGGER.info("Initialized WindowHashPolicy with %d buckets", self.regionizer.num_buckets)

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        bucket = self.regionizer.to_bucket(window)
        deterministic = deterministic or self.sampling == "argmax"
        action = self.regionizer.get_actions(bucket, deterministic=deterministic, rng=self.rng)
        if self.noise_std > 0.0:
            noise = self.rng.normal(scale=self.noise_std, size=action.shape)
            action = action + noise
        action = np.clip(action, -self.max_speed, self.max_speed)
        return action.astype(np.float32)

    def set_prototype(self, bucket: int, component: int, value: np.ndarray, weight: Optional[float] = None) -> None:
        if self.regionizer.prototypes is None:
            raise RuntimeError("Prototypes not initialized")
        self.regionizer.prototypes[bucket, component] = value.astype(np.float32)
        if weight is not None and self.regionizer.prototype_weights is not None:
            self.regionizer.prototype_weights[bucket, component] = float(weight)

    def get_prototype_table(self) -> np.ndarray:
        if self.regionizer.prototypes is None:
            raise RuntimeError("Prototypes not initialized")
        return self.regionizer.prototypes

    def export_state(self) -> dict:
        return {
            "prototypes": self.regionizer.prototypes.tolist() if self.regionizer.prototypes is not None else None,
            "weights": self.regionizer.prototype_weights.tolist() if self.regionizer.prototype_weights is not None else None,
        }

    def load_state(self, state: dict) -> None:
        prototypes = np.asarray(state["prototypes"], dtype=np.float32)
        weights = None
        if state.get("weights") is not None:
            weights = np.asarray(state["weights"], dtype=np.float32)
        self.regionizer.register_prototypes(prototypes, weights)
