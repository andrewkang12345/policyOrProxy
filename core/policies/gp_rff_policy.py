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
class GPRFFPolicy:
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

    rff_dim: int
    length_scale: float
    W: np.ndarray            # (D, F)
    u: np.ndarray            # (D,)

    A: np.ndarray            # (K, out_dim, D)
    b: np.ndarray            # (K, out_dim)

    G: np.ndarray            # (K, D)
    g0: np.ndarray           # (K,)

    rng: np.random.Generator

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _phi(self, x: np.ndarray) -> np.ndarray:
        D = self.rff_dim
        z = self.W @ x + self.u
        return (np.sqrt(2.0 / max(D, 1)) * np.cos(z)).astype(np.float32)

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = _flatten_window(window, self.feature_mode)
        phi = self._phi(x)  # (D,)

        logits = (self.G @ phi) + self.g0
        k = choose_component(
            logits=logits,
            rng=self.rng,
            policy_purity=self.policy_purity,
            deterministic=deterministic,
            temperature=self.mixture_temperature,
        )

        y = (self.A[k] @ phi) + self.b[k]  # (out_dim,)
        action = np.tanh(y).reshape(self.agents, 2) * float(self.max_speed)

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
    ) -> "GPRFFPolicy":
        teams = int(world_cfg["teams"])
        agents = int(world_cfg["agents_per_team"])
        window_len = int(world_cfg.get("window_len", world_cfg.get("history")))
        state_dim = int(policy_cfg.get("state_dim", 4))

        feature_mode = str(policy_cfg.get("feature_mode", "last")).lower()
        if feature_mode not in ("last", "mean", "flatten"):
            raise ValueError("gp_rff policy feature_mode must be one of: last, mean, flatten")

        if feature_mode == "flatten":
            F = window_len * teams * agents * state_dim
        else:
            F = teams * agents * state_dim

        rng = np.random.default_rng(int(init_seed))

        D = int(policy_cfg.get("rff_dim", 512))
        length_scale = float(policy_cfg.get("length_scale", 1.0))

        # mixture
        K = int(policy_cfg.get("num_components", 1))
        out_dim = agents * 2

        output_scale = float(policy_cfg.get("output_scale", 1.0))
        bias_scale = float(policy_cfg.get("bias_scale", 0.0))
        gate_scale = float(policy_cfg.get("gate_weight_scale", 1.0))
        gate_bias_scale = float(policy_cfg.get("gate_bias_scale", 0.0))

        W = rng.normal(scale=1.0 / max(length_scale, 1e-6), size=(D, F)).astype(np.float32)
        u = rng.uniform(0.0, 2.0 * np.pi, size=(D,)).astype(np.float32)

        A = rng.normal(scale=output_scale / np.sqrt(max(D, 1)), size=(K, out_dim, D)).astype(np.float32)
        b = rng.normal(scale=bias_scale, size=(K, out_dim)).astype(np.float32)

        G = rng.normal(scale=gate_scale / np.sqrt(max(D, 1)), size=(K, D)).astype(np.float32)
        g0 = rng.normal(scale=gate_bias_scale, size=(K,)).astype(np.float32)

        return GPRFFPolicy(
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
            rff_dim=D,
            length_scale=length_scale,
            W=W, u=u,
            A=A, b=b,
            G=G, g0=g0,
            rng=rng,
        )
