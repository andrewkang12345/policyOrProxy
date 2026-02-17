"""Window-hash opponent policies used to inject controlled distribution shifts."""
from __future__ import annotations

from typing import Dict, Optional

from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy, build_window_hash_policy

DEFAULT_IDENTIFIER = "opponent_window_hash"

__all__ = ["WindowHashPolicy", "build_hash_policy", "DEFAULT_IDENTIFIER"]


def build_hash_policy(
    arena,
    world_cfg: Dict,
    opp_cfg: Dict,
    *,
    identifier: Optional[str] = None,
    seed_offset: int = 0,
) -> WindowHashPolicy:
    """Create an opponent window-hash policy mirroring the ego policy type."""
    identifier = identifier or opp_cfg.get("identifier", DEFAULT_IDENTIFIER)
    return build_window_hash_policy(
        arena=arena,
        world_cfg=world_cfg,
        policy_cfg=opp_cfg,
        identifier=identifier,
        seed_offset=seed_offset,
    )
