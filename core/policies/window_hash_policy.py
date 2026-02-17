from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.policies.mixture import choose_component

LOGGER = logging.getLogger(__name__)


@dataclass
class WindowHashPolicy:
    identifier: str
    regionizer: WindowHashRegionizer
    num_agents: int
    num_prototypes: int
    max_speed: float
    noise_std: float
    policy_purity: str          # "pure" | "mixed"
    mixture_temperature: float
    rng: np.random.Generator

    prototypes: np.ndarray | None = None     # (buckets, K, agents, 2)
    mix_logits: np.ndarray | None = None     # (buckets, K)

    def __post_init__(self) -> None:
        # prototypes: random continuous action modes per bucket
        prot = self.rng.uniform(
            -1.0, 1.0,
            size=(self.regionizer.num_buckets, self.num_prototypes, self.num_agents, 2),
        ).astype(np.float32)
        prot *= float(self.max_speed)
        self.prototypes = prot
        self.regionizer.register_prototypes(self.prototypes)

        # state-dependent mixture weights via bucket-dependent logits
        self.mix_logits = self.rng.normal(
            scale=1.0,
            size=(self.regionizer.num_buckets, self.num_prototypes),
        ).astype(np.float32)

        LOGGER.info("Initialized WindowHashPolicy id=%s buckets=%d K=%d",
                    self.identifier, self.regionizer.num_buckets, self.num_prototypes)

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.prototypes is None or self.mix_logits is None:
            raise RuntimeError("Policy not initialized")

        bucket = self.regionizer.to_bucket(window, rng=self.rng)
        logits = self.mix_logits[bucket]  # (K,)

        k = choose_component(
            logits=logits,
            rng=self.rng,
            policy_purity=self.policy_purity,
            deterministic=deterministic,
            temperature=self.mixture_temperature,
        )

        action = self.prototypes[bucket, k]

        det = bool(deterministic) or (str(self.policy_purity).lower() == "pure")
        if self.noise_std > 0.0 and not det:
            action = action + self.rng.normal(scale=self.noise_std, size=action.shape).astype(np.float32)

        return np.clip(action, -self.max_speed, self.max_speed).astype(np.float32)

    @staticmethod
    def from_cfg(
        *,
        arena: Any,
        policy_cfg: Dict,
        world_cfg: Dict,
        identifier: str,
        init_seed: int,
        max_speed: float,
        noise_std: float,
    ) -> "WindowHashPolicy":
        p = dict(policy_cfg)
        quant = p.get("quantization", {}) or {}
        proto = p.get("prototype_init", {}) or {}

        proto_seed = int(proto.get("seed", init_seed))
        rng = np.random.default_rng(proto_seed)

        regionizer = WindowHashRegionizer(
            arena=arena,
            num_buckets=int(p["num_buckets"]),
            grid_size=int(quant["grid_size"]),
            length_scale=float(quant.get("length_scale", 1.0)),
            jitter=float(quant.get("jitter", 0.0)),
            clamp=bool(quant.get("clamp", True)),
        )

        return WindowHashPolicy(
            identifier=identifier,
            regionizer=regionizer,
            num_agents=int(world_cfg["agents_per_team"]),
            num_prototypes=int(p.get("num_prototypes", p.get("num_components", 1))),
            max_speed=float(proto.get("max_speed", max_speed)),
            noise_std=float(p.get("noise_std", noise_std)),
            policy_purity=str(p.get("policy_purity", "pure")),
            mixture_temperature=float(p.get("mixture_temperature", 1.0)),
            rng=rng,
        )
