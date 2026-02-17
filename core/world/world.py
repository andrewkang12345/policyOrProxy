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
    window_len: int
    perturbation_std: float
    teams: int
    agents_per_team: int
    # New behavior: terminate episode on first wall hit (clamp event).
    terminate_on_wall: bool = True
    # Numerical tolerance for detecting clamp events.
    wall_eps: float = 1e-6


@dataclass
class WorldState:
    positions: np.ndarray
    velocities: np.ndarray

    def as_tensor(self) -> np.ndarray:
        return np.concatenate([self.positions, self.velocities], axis=-1)


class World:
    """
    Two-team kinematic world with velocity commands.

    - actions are physical velocity commands (units/sec), shape (teams, agents, 2)
    - optional execution noise in physical units
    - speed cap in physical units
    - integration uses dt
    - bounds handled by CLAMP (no reflection/bounce)
    - episode TERMINATES when a clamp occurs (i.e., something hits a wall), if terminate_on_wall=True
    - stored velocity is realized displacement / dt
    """

    def __init__(self, arena: Arena, config: WorldConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.arena = arena
        self.config = config
        self.rng = rng or np.random.default_rng()
        self._history: Deque[WorldState] = deque(maxlen=config.window_len)
        self.state = self._initial_state()
        self._terminated = False
        self._history_extend(self.state)

    def _initial_state(self) -> WorldState:
        positions = self.arena.sample_spawn(self.rng, self.config.teams, self.config.agents_per_team, margin=1.0)
        velocities = np.zeros_like(positions)
        return WorldState(positions=positions.astype(np.float32), velocities=velocities.astype(np.float32))

    def reset(self) -> None:
        self.state = self._initial_state()
        self._history.clear()
        self._terminated = False
        self._history_extend(self.state)

    def reset_to(self, start_state: np.ndarray) -> None:
        start_state = np.asarray(start_state, dtype=np.float32)
        assert start_state.shape[0] == self.config.teams and start_state.shape[1] == self.config.agents_per_team, \
            f"Start state shape mismatch: {start_state.shape} vs (teams={self.config.teams}, agents={self.config.agents_per_team})"

        pos = start_state[..., :2]
        vel = start_state[..., 2:4] if start_state.shape[-1] >= 4 else np.zeros_like(pos, dtype=np.float32)

        # Always clamp the provided start position into bounds.
        pos = self.arena.clamp_positions(pos)

        self.state = WorldState(positions=pos.astype(np.float32), velocities=vel.astype(np.float32))
        self._history.clear()
        self._terminated = False
        self._history_extend(self.state)

    def set_state(self, start_state: np.ndarray) -> None:
        self.reset_to(start_state)

    def _history_extend(self, state: WorldState) -> None:
        while len(self._history) < self.config.window_len:
            self._history.append(state)

    @property
    def history_len(self) -> int:
        return self.config.window_len

    @property
    def terminated(self) -> bool:
        return bool(self._terminated)

    def observe_window(self) -> np.ndarray:
        frames = list(self._history)
        return np.stack([frame.as_tensor() for frame in frames], axis=0)

    def _cap_speed(self, velocities: np.ndarray) -> np.ndarray:
        max_s = float(self.config.max_speed)
        if max_s <= 0:
            return np.zeros_like(velocities, dtype=np.float32)
        speed = np.linalg.norm(velocities, axis=-1, keepdims=True)
        scale = np.minimum(1.0, max_s / np.maximum(speed, 1e-6))
        return (velocities * scale).astype(np.float32)

    def step(self, actions: np.ndarray) -> None:
        """
        Advance one step. If terminate_on_wall=True, sets self._terminated when any
        position is clamped by the arena (interpreted as a wall hit).
        """
        assert actions.shape == (self.config.teams, self.config.agents_per_team, 2), "invalid action shape"
        velocities_cmd = np.asarray(actions, dtype=np.float32)

        if self.config.perturbation_std > 0.0:
            velocities_cmd = velocities_cmd + self.rng.normal(
                scale=self.config.perturbation_std, size=velocities_cmd.shape
            ).astype(np.float32)

        velocities_cmd = self._cap_speed(velocities_cmd)

        dt = float(self.config.dt)
        prev_pos = self.state.positions
        raw_positions = prev_pos + velocities_cmd * dt

        # No reflection: clamp into bounds.
        clamped_positions = self.arena.clamp_positions(raw_positions).astype(np.float32)

        # Detect wall hit: any coordinate changed by clamping.
        if self.config.terminate_on_wall:
            eps = float(self.config.wall_eps)
            if np.any(np.abs(clamped_positions - raw_positions) > eps):
                self._terminated = True

        velocities = (clamped_positions - prev_pos) / max(dt, 1e-6)

        self.state = WorldState(positions=clamped_positions, velocities=velocities.astype(np.float32))
        self._history.append(self.state)

    def rollout(
        self,
        ego_policy,
        opponent_policy,
        steps: int,
        deterministic: bool = False,
        policy_id: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Roll out up to `steps` transitions, but may terminate early if a wall hit occurs.
        Returned arrays have variable first dimension L (L <= steps).
        """
        windows = []
        ego_actions = []
        opponent_actions = []
        positions = []

        for _t in range(int(steps)):
            if self.terminated:
                break

            window = self.observe_window()
            ego_action = ego_policy.act(window, deterministic=deterministic)
            opp_action = opponent_policy.act(window, deterministic=deterministic)

            action_stack = np.stack([ego_action, opp_action], axis=0)
            self.step(action_stack)

            windows.append(window)
            ego_actions.append(ego_action)
            opponent_actions.append(opp_action)
            positions.append(self.state.positions.copy())

            if self.terminated:
                # Include the collision-causing action/transition, then stop.
                break

        return {
            "windows": np.asarray(windows, dtype=np.float32),
            "ego_actions": np.asarray(ego_actions, dtype=np.float32),
            "opponent_actions": np.asarray(opponent_actions, dtype=np.float32),
            "positions": np.asarray(positions, dtype=np.float32),
            "policy_id": policy_id or getattr(ego_policy, "identifier", "unknown"),
        }


def build_world(arena: Arena, config: dict, rng: Optional[np.random.Generator] = None) -> World:
    if "window_len" in config:
        window_len = int(config["window_len"])
    elif "history" in config:
        window_len = int(config["history"])
    else:
        raise KeyError("world config must define 'window_len' (preferred) or legacy 'history'")

    world_cfg = WorldConfig(
        dt=float(config["dt"]),
        max_speed=float(config["max_speed"]),
        window_len=window_len,
        perturbation_std=float(config.get("perturbation_std", 0.0)),
        teams=int(config["teams"]),
        agents_per_team=int(config["agents_per_team"]),
        terminate_on_wall=bool(config.get("terminate_on_wall", True)),
        wall_eps=float(config.get("wall_eps", 1e-6)),
    )
    return World(arena=arena, config=world_cfg, rng=rng)
