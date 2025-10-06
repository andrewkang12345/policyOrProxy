"""Arena geometry utilities for the policy invariance benchmark."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class CircularObstacle:
    """Simple circular obstacle used to keep agents out of forbidden regions."""

    center: Sequence[float]
    radius: float

    def project(self, positions: np.ndarray) -> np.ndarray:
        """Push any position inside the obstacle back to the boundary."""
        center = np.asarray(self.center, dtype=np.float32)
        delta = positions - center
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)
        mask = dist < self.radius
        if not np.any(mask):
            return positions
        safe_delta = np.where(mask, delta, 0.0)
        safe_dist = np.maximum(dist, 1e-6)
        scaled = safe_delta / safe_dist * self.radius * 1.01
        adjusted = center + scaled
        return np.where(mask, adjusted, positions)


@dataclass
class Arena:
    """Rectangular arena with optional circular obstacles."""

    width: float
    height: float
    obstacles: Iterable[CircularObstacle] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.obstacles = [
            obs if isinstance(obs, CircularObstacle) else CircularObstacle(**obs)
            for obs in self.obstacles
        ]
        self._lower = np.array([0.0, 0.0], dtype=np.float32)
        self._upper = np.array([self.width, self.height], dtype=np.float32)

    @property
    def lower(self) -> np.ndarray:
        return self._lower

    @property
    def upper(self) -> np.ndarray:
        return self._upper

    def clamp_positions(self, positions: np.ndarray) -> np.ndarray:
        """Keep positions inside the arena bounds and outside obstacles."""
        positions = np.clip(positions, self.lower, self.upper)
        for obstacle in self.obstacles:
            positions = obstacle.project(positions)
        return positions

    def sample_spawn(self, rng: np.random.Generator, teams: int, agents: int, margin: float = 1.0) -> np.ndarray:
        """Sample non-colliding spawn positions."""
        positions = np.zeros((teams, agents, 2), dtype=np.float32)
        attempts = 0
        while True:
            attempts += 1
            proposal = rng.uniform(self.lower + margin, self.upper - margin, size=(teams, agents, 2)).astype(np.float32)
            if self._is_valid_spawn(proposal, min_dist=margin * 0.8):
                positions = proposal
                break
            if attempts > 1000:
                raise RuntimeError("Could not sample valid spawn positions after many attempts")
        return positions

    def _is_valid_spawn(self, proposal: np.ndarray, min_dist: float) -> bool:
        flat = proposal.reshape(-1, 2)
        if flat.shape[0] <= 1:
            return True
        dists = np.linalg.norm(flat[None, :, :] - flat[:, None, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < min_dist:
            return False
        for obstacle in self.obstacles:
            center = np.asarray(obstacle.center, dtype=np.float32)
            if np.any(np.linalg.norm(flat - center, axis=-1) < obstacle.radius * 1.1):
                return False
        return True

    def normalize(self, positions: np.ndarray) -> np.ndarray:
        span = self.upper - self.lower
        normalized = (positions - self.lower) / np.maximum(span, 1e-6)
        return np.clip(normalized, 0.0, 1.0)

    def denormalize(self, positions: np.ndarray) -> np.ndarray:
        span = self.upper - self.lower
        return self.lower + positions * span

    def area(self) -> float:
        return self.width * self.height

    def distance_to_boundary(self, positions: np.ndarray) -> np.ndarray:
        lower_dist = positions - self.lower
        upper_dist = self.upper - positions
        return np.minimum(lower_dist, upper_dist).min(axis=-1)

    def contains(self, positions: np.ndarray) -> np.ndarray:
        inside = np.logical_and(positions >= self.lower, positions <= self.upper).all(axis=-1)
        if not self.obstacles:
            return inside
        result = inside.copy()
        for obstacle in self.obstacles:
            center = np.asarray(obstacle.center, dtype=np.float32)
            mask = np.linalg.norm(positions - center, axis=-1) >= obstacle.radius
            result = np.logical_and(result, mask)
        return result


def build_arena(config: dict) -> Arena:
    obstacles: List[CircularObstacle] = []
    for raw in config.get("obstacles", []):
        if isinstance(raw, CircularObstacle):
            obstacles.append(raw)
        else:
            obstacles.append(CircularObstacle(center=raw["center"], radius=float(raw["radius"])))
    return Arena(width=float(config["width"]), height=float(config["height"]), obstacles=obstacles)
