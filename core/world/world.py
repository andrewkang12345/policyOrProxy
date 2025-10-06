"""Simple kinematic world used to generate training data."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from policyOrProxy.core.world.arena import Arena

LOGGER = logging.getLogger(__name__)


@dataclass
class WorldConfig:
    dt: float
    max_speed: float
    history: int
    perturbation_std: float
    teams: int
    agents_per_team: int


@dataclass
class WorldState:
    positions: np.ndarray
    velocities: np.ndarray

    def as_tensor(self) -> np.ndarray:
        return np.concatenate([self.positions, self.velocities], axis=-1)


class World:
    """Two-team kinematic world with velocity commands."""

    def __init__(self, arena: Arena, config: WorldConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.arena = arena
        self.config = config
        self.rng = rng or np.random.default_rng()
        self._history: Deque[WorldState] = deque(maxlen=config.history)
        self.state = self._initial_state()
        self._history_extend(self.state)

    def _initial_state(self) -> WorldState:
        positions = self.arena.sample_spawn(self.rng, self.config.teams, self.config.agents_per_team, margin=1.0)
        velocities = np.zeros_like(positions)
        return WorldState(positions=positions, velocities=velocities)

    def reset(self) -> None:
        LOGGER.debug("Resetting world")
        self.state = self._initial_state()
        self._history.clear()
        self._history_extend(self.state)

    def _history_extend(self, state: WorldState) -> None:
        while len(self._history) < self.config.history:
            self._history.append(state)

    @property
    def history_len(self) -> int:
        return self._history.maxlen or self.config.history

    def observe_window(self) -> np.ndarray:
        frames = list(self._history)
        stacked = np.stack([frame.as_tensor() for frame in frames], axis=0)
        return stacked

    def step(self, actions: np.ndarray) -> None:
        assert actions.shape == (self.config.teams, self.config.agents_per_team, 2), "invalid action shape"
        clipped = np.clip(actions, -1.0, 1.0) * self.config.max_speed
        noise = self.rng.normal(scale=self.config.perturbation_std, size=clipped.shape)
        velocities = clipped + noise
        positions = self.state.positions + velocities * self.config.dt
        positions = self.arena.clamp_positions(positions)
        velocities = (positions - self.state.positions) / max(self.config.dt, 1e-6)
        self.state = WorldState(positions=positions, velocities=velocities)
        self._history.append(self.state)

    def rollout(self, ego_policy, opponent_policy, steps: int, deterministic: bool = False, policy_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        windows = []
        ego_actions = []
        opponent_actions = []
        positions = []
        for step in range(steps):
            window = self.observe_window()
            ego_action = ego_policy.act(window, deterministic=deterministic)
            opp_action = opponent_policy.act(window, deterministic=deterministic)
            action_stack = np.stack([ego_action, opp_action], axis=0)
            self.step(action_stack)
            windows.append(window)
            ego_actions.append(ego_action)
            opponent_actions.append(opp_action)
            positions.append(self.state.positions.copy())
            LOGGER.debug("Rollout step %d complete", step)
        return {
            "windows": np.asarray(windows, dtype=np.float32),
            "ego_actions": np.asarray(ego_actions, dtype=np.float32),
            "opponent_actions": np.asarray(opponent_actions, dtype=np.float32),
            "positions": np.asarray(positions, dtype=np.float32),
            "policy_id": policy_id or opponent_policy.identifier,
        }

    def clone(self) -> "World":
        clone = World(self.arena, self.config, self.rng)
        clone.state = WorldState(positions=self.state.positions.copy(), velocities=self.state.velocities.copy())
        clone._history = deque([WorldState(positions=frame.positions.copy(), velocities=frame.velocities.copy()) for frame in self._history], maxlen=self.history_len)
        return clone


def build_world(arena: Arena, config: dict, rng: Optional[np.random.Generator] = None) -> World:
    world_cfg = WorldConfig(
        dt=float(config["dt"]),
        max_speed=float(config["max_speed"]),
        history=int(config["history"]),
        perturbation_std=float(config.get("perturbation_std", 0.0)),
        teams=int(config["teams"]),
        agents_per_team=int(config["agents_per_team"]),
    )
    return World(arena=arena, config=world_cfg, rng=rng)
