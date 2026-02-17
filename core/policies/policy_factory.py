from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Callable, Optional

import numpy as np

from policyOrProxy.core.policies.linear_policy import LinearPolicy
from policyOrProxy.core.policies.mlp_policy import RandomMLPPolicy
from policyOrProxy.core.policies.gp_rff_policy import GPRFFPolicy
from policyOrProxy.core.policies.window_hash_policy import WindowHashPolicy


class Policy(Protocol):
    identifier: str

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def set_rng(self, rng: np.random.Generator) -> None: ...


PolicyBuilder = Callable[..., Policy]


def _require_policy_class(cfg: Dict) -> str:
    if "policy_class" not in cfg:
        raise KeyError(
            "Policy config must include `policy_class` (e.g., window_hash, linear, mlp_random, gp_rff). "
            "Window hashing is now an explicit class and is not inferred."
        )
    return str(cfg["policy_class"]).strip().lower()


def build_policy(
    *,
    arena: Any,
    world_cfg: Dict,
    policy_cfg: Dict,
    role: str,
    identifier: str,
    init_seed: int,
) -> Policy:
    """
    Build a policy from a config dict with explicit `policy_class`.

    role: "ego" | "opponent" (informational; can be used by future builders)
    init_seed: deterministic seed for parameter initialization (random weights/prototypes)
    """
    pclass = _require_policy_class(policy_cfg)

    # Common defaults
    max_speed = float(policy_cfg.get("max_speed", world_cfg.get("max_speed", 1.0)))
    noise_std = float(policy_cfg.get("noise_std", 0.0))

    if pclass == "window_hash":
        return WindowHashPolicy.from_cfg(
            arena=arena,
            policy_cfg=policy_cfg,
            world_cfg=world_cfg,
            identifier=identifier,
            init_seed=int(policy_cfg.get("init_seed", init_seed)),
            max_speed=max_speed,
            noise_std=noise_std,
        )

    if pclass == "linear":
        return LinearPolicy.from_cfg(
            policy_cfg=policy_cfg,
            world_cfg=world_cfg,
            identifier=identifier,
            init_seed=int(policy_cfg.get("init_seed", init_seed)),
            max_speed=max_speed,
            noise_std=noise_std,
        )

    if pclass in ("mlp_random", "mlp"):
        return RandomMLPPolicy.from_cfg(
            policy_cfg=policy_cfg,
            world_cfg=world_cfg,
            identifier=identifier,
            init_seed=int(policy_cfg.get("init_seed", init_seed)),
            max_speed=max_speed,
            noise_std=noise_std,
        )

    if pclass in ("gp_rff", "rff_gp"):
        return GPRFFPolicy.from_cfg(
            policy_cfg=policy_cfg,
            world_cfg=world_cfg,
            identifier=identifier,
            init_seed=int(policy_cfg.get("init_seed", init_seed)),
            max_speed=max_speed,
            noise_std=noise_std,
        )

    raise ValueError(f"Unknown policy_class='{pclass}'. Add a builder in policy_factory.py.")
