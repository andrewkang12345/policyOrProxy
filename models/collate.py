"""Custom dataloader collate helpers."""
from __future__ import annotations

from typing import Dict, List

import torch


def next_frame_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    windows = torch.stack([item["window"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    episode_ids = torch.stack([item["episode_id"] for item in batch], dim=0)
    timesteps = torch.stack([item["timestep"] for item in batch], dim=0)
    output = {"window": windows, "action": actions, "episode_id": episode_ids, "timestep": timesteps}
    if "opponent_action" in batch[0]:
        output["opponent_action"] = torch.stack([item["opponent_action"] for item in batch], dim=0)
    if "policy_id" in batch[0]:
        output["policy_id"] = torch.stack([item["policy_id"] for item in batch], dim=0)
    return output


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
