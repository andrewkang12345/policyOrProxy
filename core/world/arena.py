from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Arena:
    width: float
    height: float
    obstacles: Optional[List[Dict[str, Any]]] = None  # list of {center:[x,y], radius:r}

    def normalize(self, pos: np.ndarray) -> np.ndarray:
        """
        Map world coordinates to ~[0,1] range (not clamped unless caller clamps).
        pos shape (...,2)
        """
        s = np.array([self.width, self.height], dtype=np.float32)
        return pos.astype(np.float32) / s

    def clamp_positions(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.astype(np.float32)
        pos[..., 0] = np.clip(pos[..., 0], 0.0, self.width)
        pos[..., 1] = np.clip(pos[..., 1], 0.0, self.height)
        return pos

    def _reflect_1d(self, x: np.ndarray, L: float) -> np.ndarray:
        """
        Reflect x into [0, L] using period-2L folding:
          x' = x mod (2L)
          if x' > L: x'' = 2L - x'
        Works for negative x too.
        """
        twoL = 2.0 * float(L)
        xm = np.mod(x, twoL)
        xr = np.where(xm <= L, xm, twoL - xm)
        return xr.astype(np.float32)

    def reflect_positions(self, pos: np.ndarray) -> np.ndarray:
        """
        Reflect positions to keep them inside the rectangle. Then project out of obstacles.
        pos shape (...,2)
        """
        pos = pos.astype(np.float32)
        pos[..., 0] = self._reflect_1d(pos[..., 0], self.width)
        pos[..., 1] = self._reflect_1d(pos[..., 1], self.height)
        return self._project_out_of_obstacles(pos)

    def _project_out_of_obstacles(self, pos: np.ndarray) -> np.ndarray:
        if not self.obstacles:
            return pos

        out = pos.astype(np.float32, copy=True)
        eps = 1e-4

        for obs in self.obstacles:
            c = np.array(obs["center"], dtype=np.float32)  # (2,)
            r = float(obs["radius"])

            d = out - c                                   # (...,2)
            dist = np.linalg.norm(d, axis=-1, keepdims=True)  # (...,1)
            inside = dist < (r + eps)

            # Safe direction: normalize if possible, else arbitrary unit vector
            denom = np.maximum(dist, 1e-6)
            d_unit = d / denom
            fallback = np.array([1.0, 0.0], dtype=np.float32)
            d_safe = np.where(dist > 1e-6, d_unit, fallback)

            out = np.where(inside, c + d_safe * (r + eps), out)

        return out

    def sample_spawn(
        self,
        rng: np.random.Generator,
        teams: int,
        agents_per_team: int,
        margin: float = 1.0,
    ) -> np.ndarray:
        lo = np.array([margin, margin], dtype=np.float32)
        hi = np.array([self.width - margin, self.height - margin], dtype=np.float32)
        pos = rng.uniform(lo, hi, size=(teams, agents_per_team, 2)).astype(np.float32)

        # Ensure spawns are not inside obstacles
        pos = self._project_out_of_obstacles(pos)
        return pos


def build_arena(cfg: Dict) -> Arena:
    return Arena(
        width=float(cfg["width"]),
        height=float(cfg["height"]),
        obstacles=cfg.get("obstacles", None),
    )
