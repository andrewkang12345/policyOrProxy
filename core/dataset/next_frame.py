"""Dataset providing (window, next action) pairs for CVAE training."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from policyOrProxy.core.dataset.indexer import EpisodeIndexer, EpisodeRecord


@dataclass
class _EpisodeCache:
    record: EpisodeRecord
    windows: np.ndarray
    actions: np.ndarray
    opponent_actions: Optional[np.ndarray]


class NextFrameDataset(Dataset):
    """Dataset that exposes windows and the corresponding ego action."""

    def __init__(
        self,
        root: Path,
        indexer: EpisodeIndexer,
        split: str,
        device: Optional[torch.device] = None,
        preload: bool = False,
        include_policy_id: bool = False,
    ) -> None:
        self.root = root
        self.indexer = indexer
        self.records: List[EpisodeRecord] = list(indexer.iter_split(split))
        if not self.records:
            raise ValueError(f"No episodes found for split {split}")
        self.device = device
        self.include_policy_id = include_policy_id
        self._policy_mapping: Dict[str, int] = {}
        if include_policy_id:
            unique_ids = sorted({record.policy_id or "unknown" for record in self.records})
            self._policy_mapping = {pid: idx for idx, pid in enumerate(unique_ids)}
            self._policy_inverse = {idx: pid for pid, idx in self._policy_mapping.items()}
        else:
            self._policy_inverse = {}
        self._episodes: List[_EpisodeCache] = []
        self._lengths: List[int] = []
        self._offsets: List[int] = []
        total = 0
        for record in self.records:
            episode = self._load_episode(record, preload=preload)
            self._episodes.append(episode)
            steps = episode.actions.shape[0]
            total += steps
            self._lengths.append(steps)
            self._offsets.append(total)
        self._total = total

    def _load_episode(self, record: EpisodeRecord, preload: bool) -> _EpisodeCache:
        path = self.root / record.path
        with np.load(path, allow_pickle=False) as data:
            windows = data["windows"]
            actions = data["ego_actions"]
            opponent_actions = data.get("opponent_actions")
            if preload:
                windows = np.asarray(windows)
                actions = np.asarray(actions)
                opponent_actions = np.asarray(opponent_actions) if opponent_actions is not None else None
        return _EpisodeCache(record=record, windows=windows, actions=actions, opponent_actions=opponent_actions)

    def __len__(self) -> int:
        return self._total

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index = self._total + index
        if index < 0 or index >= self._total:
            raise IndexError(index)
        episode_idx = bisect.bisect_right(self._offsets, index)
        prev = self._offsets[episode_idx - 1] if episode_idx > 0 else 0
        local_idx = index - prev
        return episode_idx, local_idx

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_idx, local_idx = self._locate(index)
        episode = self._episodes[episode_idx]
        window = episode.windows[local_idx]
        action = episode.actions[local_idx]
        sample: Dict[str, torch.Tensor] = {
            "window": torch.from_numpy(window).float(),
            "action": torch.from_numpy(action).float(),
            "episode_id": torch.tensor(episode_idx, dtype=torch.long),
            "timestep": torch.tensor(local_idx, dtype=torch.long),
        }
        if episode.opponent_actions is not None:
            sample["opponent_action"] = torch.from_numpy(episode.opponent_actions[local_idx]).float()
        if self.include_policy_id and episode.record.policy_id is not None:
            mapped = self._policy_mapping.get(episode.record.policy_id, 0)
            sample["policy_id"] = torch.tensor(mapped, dtype=torch.long)
        if self.device is not None:
            sample = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        return sample

    def describe(self) -> Dict[str, int]:
        return {record.path: length for record, length in zip(self.records, self._lengths)}

    def policy_name(self, index: int) -> str:
        return self._policy_inverse.get(int(index), "unknown")
