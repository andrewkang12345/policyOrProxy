from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from policyOrProxy.core.policies.mixture import choose_component

_FeatureMode = Literal["last", "mean", "flatten"]


def _flatten_window(window: np.ndarray, mode: _FeatureMode) -> np.ndarray:
    if window.ndim != 4:
        raise ValueError(f"Expected window shape (T,teams,agents,state_dim), got {window.shape}")
    if mode == "last":
        x = window[-1].reshape(-1)
    elif mode == "mean":
        x = window.mean(axis=0).reshape(-1)
    elif mode == "flatten":
        x = window.reshape(-1)
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")
    return x.astype(np.float32)


@dataclass
class LinearPolicy:
    identifier: str
    teams: int
    agents: int
    state_dim: int
    window_len: int
    max_speed: float
    noise_std: float

    policy_purity: str
    mixture_temperature: float
    num_components: int
    feature_mode: _FeatureMode

    W: np.ndarray       # (K, out_dim, F)
    b: np.ndarray       # (K, out_dim)
    gate_W: np.ndarray  # (K, F)
    gate_b: np.ndarray  # (K,)
    rng: np.random.Generator

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = _flatten_window(window, self.feature_mode)  # (F,)

        logits = (self.gate_W @ x) + self.gate_b        # (K,)
        k = choose_component(
            logits=logits,
            rng=self.rng,
            policy_purity=self.policy_purity,
            deterministic=deterministic,
            temperature=self.mixture_temperature,
        )

        y = (self.W[k] @ x) + self.b[k]                 # (out_dim,)

        # ---- DEBUG: y magnitude / tanh saturation ----
        y_abs = np.abs(y)
        frac_sat2 = float(np.mean(y_abs > 2.0))
        frac_sat3 = float(np.mean(y_abs > 3.0))
        print(
            f"[{self.identifier}] k={k} "
            f"|y|_2={np.linalg.norm(y):.3f} "
            f"y_max={y.max():.3f} y_min={y.min():.3f} "
            f"mean|y|={y_abs.mean():.3f} "
            f"frac(|y|>2)={frac_sat2:.2%} frac(|y|>3)={frac_sat3:.2%}"
        )
        # ---------------------------------------------
        action = np.tanh(y).reshape(self.agents, 2) * float(self.max_speed)

        # action = y.reshape(self.agents, 2)           # linear in x
        # action = np.clip(action, -self.max_speed, self.max_speed).astype(np.float32)

        # action = (y / (1.0 + np.abs(y))).reshape(self.agents, 2) * float(self.max_speed)

        det = bool(deterministic) or (str(self.policy_purity).lower() == "pure")
        if self.noise_std > 0.0 and not det:
            action = action + self.rng.normal(scale=self.noise_std, size=action.shape).astype(np.float32)

        return np.clip(action, -self.max_speed, self.max_speed).astype(np.float32)

    @staticmethod
    def from_cfg(
        *,
        policy_cfg: Dict,
        world_cfg: Dict,
        identifier: str,
        init_seed: int,
        max_speed: float,
        noise_std: float,
    ) -> "LinearPolicy":
        teams = int(world_cfg["teams"])
        agents = int(world_cfg["agents_per_team"])
        window_len = int(world_cfg.get("window_len", world_cfg.get("history")))
        state_dim = int(policy_cfg.get("state_dim", 4))

        feature_mode = str(policy_cfg.get("feature_mode", "last")).lower()
        if feature_mode not in ("last", "mean", "flatten"):
            raise ValueError("linear policy feature_mode must be one of: last, mean, flatten")

        if feature_mode == "flatten":
            F = window_len * teams * agents * state_dim
        else:
            F = teams * agents * state_dim

        K = int(policy_cfg.get("num_components", 1))
        rng = np.random.default_rng(int(init_seed))

        w_scale = float(policy_cfg.get("weight_scale", 0.5))
        b_scale = float(policy_cfg.get("bias_scale", 0.0))
        g_scale = float(policy_cfg.get("gate_weight_scale", 0.5))
        gb_scale = float(policy_cfg.get("gate_bias_scale", 0.0))

        out_dim = agents * 2
        W = rng.normal(scale=w_scale, size=(K, out_dim, F)).astype(np.float32)
        b = rng.normal(scale=b_scale, size=(K, out_dim)).astype(np.float32)

        gate_W = rng.normal(scale=g_scale, size=(K, F)).astype(np.float32)
        gate_b = rng.normal(scale=gb_scale, size=(K,)).astype(np.float32)

        return LinearPolicy(
            identifier=identifier,
            teams=teams,
            agents=agents,
            state_dim=state_dim,
            window_len=window_len,
            max_speed=float(max_speed),
            noise_std=float(noise_std),
            policy_purity=str(policy_cfg.get("policy_purity", "pure")),
            mixture_temperature=float(policy_cfg.get("mixture_temperature", 1.0)),
            num_components=K,
            feature_mode=feature_mode,  # type: ignore
            W=W, b=b, gate_W=gate_W, gate_b=gate_b,
            rng=rng,
        )
