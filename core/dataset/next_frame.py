"""
Dataset providing (window, action) pairs aligned per timestep.

Option B:
- window = stored npz["windows"][t]         (T = stored window length)
- target = stored npz["ego_actions"][t]
"""

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
    windows: np.ndarray                    # (N, T, teams, agents, state_dim)
    actions: np.ndarray                    # (N, agents, action_dim)
    opponent_actions: Optional[np.ndarray] # (N, agents, action_dim) if present
    usable_len: int                        # N


def _ensure_windows_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "windows" not in data:
        raise KeyError(
            "NPZ is missing 'windows'. Option B requires storing per-step observation windows."
        )
    windows = np.asarray(data["windows"])
    if windows.ndim != 5:
        raise ValueError(
            f"Expected windows with 5 dims (N,T,teams,agents,state_dim), got {windows.shape}"
        )
    return windows


class NextFrameDataset(Dataset):
    """
    Option B dataset:
      sample[t] = (windows[t], ego_actions[t])

    The authoritative temporal window length is the stored NPZ window dimension:
      windows.shape == (N, T, teams, agents, state_dim)

    Behavior:
      - T is inferred from the first loaded episode
      - enforced consistent across all episodes in this dataset
    """

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

        # Inferred from NPZ on first episode load.
        self.window_len: Optional[int] = None

        self._policy_mapping: Dict[str, int] = {}
        if include_policy_id:
            unique_ids = sorted({rec.policy_id or "unknown" for rec in self.records})
            self._policy_mapping = {pid: idx for idx, pid in enumerate(unique_ids)}
            self._policy_inverse = {idx: pid for pid, idx in self._policy_mapping.items()}
        else:
            self._policy_inverse = {}

        self._episodes: List[_EpisodeCache] = []
        self._offsets: List[int] = []
        self._lengths: List[int] = []
        total = 0

        for rec in self.records:
            ep = self._load_episode(rec, preload=preload)
            self._episodes.append(ep)

            total += ep.usable_len
            self._lengths.append(ep.usable_len)
            self._offsets.append(total)

        self._total = total
        if self._total <= 0:
            raise ValueError(f"Dataset split '{split}' has no usable samples.")

        if self.window_len is None:
            raise RuntimeError("Failed to infer window_len from NPZ windows.")

    def _load_episode(self, record: EpisodeRecord, preload: bool) -> _EpisodeCache:
        path = self.root / record.path
        with np.load(path, allow_pickle=False) as data:
            windows = _ensure_windows_from_npz(data)
            actions = np.asarray(data["ego_actions"])
            opponent_actions = np.asarray(data["opponent_actions"]) if "opponent_actions" in data else None

        N = int(windows.shape[0])
        T = int(windows.shape[1])

        # Infer window_len from first episode and enforce for all subsequent episodes.
        if self.window_len is None:
            self.window_len = T
        elif T != int(self.window_len):
            raise ValueError(
                f"Episode {record.path}: stored window_len={T} != dataset window_len={self.window_len}."
            )

        if actions.shape[0] != N:
            raise ValueError(
                f"Episode {record.path}: actions length ({actions.shape[0]}) != windows length ({N})."
            )
        if opponent_actions is not None and opponent_actions.shape[0] != N:
            raise ValueError(
                f"Episode {record.path}: opponent_actions length ({opponent_actions.shape[0]}) != windows length ({N})."
            )

        if preload:
            windows = np.asarray(windows)
            actions = np.asarray(actions)
            opponent_actions = np.asarray(opponent_actions) if opponent_actions is not None else None

        return _EpisodeCache(
            record=record,
            windows=windows,
            actions=actions,
            opponent_actions=opponent_actions,
            usable_len=N,
        )

    def __len__(self) -> int:
        return self._total

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index = self._total + index
        if index < 0 or index >= self._total:
            raise IndexError(index)

        ep_idx = bisect.bisect_right(self._offsets, index)
        prev = self._offsets[ep_idx - 1] if ep_idx > 0 else 0
        local = index - prev
        return ep_idx, local

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self._locate(index)
        ep = self._episodes[ep_idx]

        window = ep.windows[t]     # (T, teams, agents, state_dim)
        action = ep.actions[t]     # (agents, action_dim)

        sample: Dict[str, torch.Tensor] = {
            "window": torch.from_numpy(window).float(),
            "action": torch.from_numpy(action).float(),
            "episode_id": torch.tensor(ep_idx, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
        }

        if ep.opponent_actions is not None:
            sample["opponent_action"] = torch.from_numpy(ep.opponent_actions[t]).float()

        if self.include_policy_id:
            pid = ep.record.policy_id or "unknown"
            mapped = self._policy_mapping.get(pid, 0)
            sample["policy_id"] = torch.tensor(mapped, dtype=torch.long)

        if self.device is not None:
            sample = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in sample.items()
            }

        return sample

    def describe(self) -> Dict[str, int]:
        return {str(r.path): l for r, l in zip(self.records, self._lengths)}

    def policy_name(self, index: int) -> str:
        return self._policy_inverse.get(int(index), "unknown")
