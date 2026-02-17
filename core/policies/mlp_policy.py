from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from policyOrProxy.core.policies.mixture import choose_component

_FeatureMode = Literal["last", "mean", "flatten"]
_Activation = Literal["tanh", "gelu", "relu"]


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


def _act_fn(name: _Activation, x: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class RandomMLPPolicy:
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
    activation: _Activation

    trunk_W: list[np.ndarray]    # list of (d_out, d_in)
    trunk_b: list[np.ndarray]    # list of (d_out,)
    head_W: np.ndarray           # (K, out_dim, H)
    head_b: np.ndarray           # (K, out_dim)
    gate_W: np.ndarray           # (K, H)
    gate_b: np.ndarray           # (K,)
    rng: np.random.Generator

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _trunk(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i in range(len(self.trunk_W)):
            h = (self.trunk_W[i] @ h) + self.trunk_b[i]
            h = _act_fn(self.activation, h)
        return h.astype(np.float32)  # (H,)

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = _flatten_window(window, self.feature_mode)
        h = self._trunk(x)

        logits = (self.gate_W @ h) + self.gate_b
        k = choose_component(
            logits=logits,
            rng=self.rng,
            policy_purity=self.policy_purity,
            deterministic=deterministic,
            temperature=self.mixture_temperature,
        )

        y = (self.head_W[k] @ h) + self.head_b[k]  # (out_dim,)
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
    ) -> "RandomMLPPolicy":
        teams = int(world_cfg["teams"])
        agents = int(world_cfg["agents_per_team"])
        window_len = int(world_cfg.get("window_len", world_cfg.get("history")))
        state_dim = int(policy_cfg.get("state_dim", 4))

        feature_mode = str(policy_cfg.get("feature_mode", "last")).lower()
        if feature_mode not in ("last", "mean", "flatten"):
            raise ValueError("mlp policy feature_mode must be one of: last, mean, flatten")

        activation = str(policy_cfg.get("activation", "tanh")).lower()
        if activation not in ("tanh", "relu", "gelu"):
            raise ValueError("mlp policy activation must be one of: tanh, relu, gelu")

        if feature_mode == "flatten":
            in_dim = window_len * teams * agents * state_dim
        else:
            in_dim = teams * agents * state_dim

        hidden_dims = policy_cfg.get("hidden_dims", [128, 128])
        if not isinstance(hidden_dims, list) or not hidden_dims:
            raise ValueError("mlp policy hidden_dims must be a non-empty list")

        K = int(policy_cfg.get("num_components", 1))
        out_dim = agents * 2

        rng = np.random.default_rng(int(init_seed))
        w_scale = float(policy_cfg.get("weight_scale", 0.5))
        b_scale = float(policy_cfg.get("bias_scale", 0.0))

        # trunk
        layer_dims = [in_dim] + [int(h) for h in hidden_dims]
        trunk_W: list[np.ndarray] = []
        trunk_b: list[np.ndarray] = []
        for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:]):
            trunk_W.append(rng.normal(scale=w_scale / np.sqrt(max(d_in, 1)), size=(d_out, d_in)).astype(np.float32))
            trunk_b.append(rng.normal(scale=b_scale, size=(d_out,)).astype(np.float32))

        H = int(layer_dims[-1])

        # K action heads
        head_W = rng.normal(scale=w_scale / np.sqrt(max(H, 1)), size=(K, out_dim, H)).astype(np.float32)
        head_b = rng.normal(scale=b_scale, size=(K, out_dim)).astype(np.float32)

        # gating head (state-dependent mixture weights)
        gate_W = rng.normal(scale=float(policy_cfg.get("gate_weight_scale", w_scale)) / np.sqrt(max(H, 1)),
                            size=(K, H)).astype(np.float32)
        gate_b = rng.normal(scale=float(policy_cfg.get("gate_bias_scale", b_scale)),
                            size=(K,)).astype(np.float32)

        return RandomMLPPolicy(
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
            activation=activation,      # type: ignore
            trunk_W=trunk_W,
            trunk_b=trunk_b,
            head_W=head_W,
            head_b=head_b,
            gate_W=gate_W,
            gate_b=gate_b,
            rng=rng,
        )
