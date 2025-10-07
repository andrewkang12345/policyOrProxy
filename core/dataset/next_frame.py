"""Dataset providing (window, next action) pairs for CVAE training."""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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


def _build_sliding_windows(frames: np.ndarray, window_len: int) -> np.ndarray:
    """
    Build sliding windows over the time dimension (first axis).
    Input:  frames shape = (N, *feat_shape)
    Output: windows shape = (N - T + 1, T, *feat_shape), where each window[i] = frames[i : i+T]
    """
    if frames.ndim < 1:
        raise ValueError(f"frames must have time as first dimension; got shape {frames.shape}")
    N = frames.shape[0]
    T = int(window_len)
    if T <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    if N < T:
        raise ValueError(f"sequence too short for windowing: N={N} < T={T}")

    # Simple, explicit construction (avoids stride-tricks surprises)
    windows = np.stack([frames[i - T + 1 : i + 1] for i in range(T - 1, N)], axis=0)
    return windows


class NextFrameDataset(Dataset):
    """Dataset that exposes time windows and the corresponding ego action (aligned to the window's last frame)."""

    def __init__(
        self,
        root: Path,
        indexer: EpisodeIndexer,
        split: str,
        device: Optional[torch.device] = None,
        preload: bool = False,
        include_policy_id: bool = False,
        window_len: int = 1,
    ) -> None:
        self.root = root
        self.indexer = indexer
        self.records: List[EpisodeRecord] = list(indexer.iter_split(split))
        if not self.records:
            raise ValueError(f"No episodes found for split {split}")
        self.device = device
        self.include_policy_id = include_policy_id
        self.window_len = int(window_len)

        # Map policy_id -> contiguous int (optional)
        self._policy_mapping: Dict[str, int] = {}
        if include_policy_id:
            unique_ids = sorted({record.policy_id or "unknown" for record in self.records})
            self._policy_mapping = {pid: idx for idx, pid in enumerate(unique_ids)}
            self._policy_inverse = {idx: pid for pid, idx in self._policy_mapping.items()}
        else:
            self._policy_inverse = {}

        # Load all episodes and compute global indexing
        self._episodes: List[_EpisodeCache] = []
        self._lengths: List[int] = []
        self._offsets: List[int] = []
        total = 0
        for record in self.records:
            episode = self._load_episode(record, preload=preload)
            self._episodes.append(episode)
            steps = int(episode.actions.shape[0])
            total += steps
            self._lengths.append(steps)
            self._offsets.append(total)
        self._total = total

    def _load_episode(self, record: EpisodeRecord, preload: bool) -> _EpisodeCache:
        """
        Loads one episode from NPZ. Expected keys:
          - "windows": either (N, F), (N, teams, agents, state_dim), or (N, T, ...)
          - "ego_actions": (N, agents, action_dim)
          - optional "opponent_actions": (N, agents, action_dim) or similar
        Produces time-windowed outputs where actions are aligned to the last frame of each window.
        """
        path = self.root / record.path
        with np.load(path, allow_pickle=False) as data:
            raw_windows: np.ndarray = data["windows"]
            actions: np.ndarray = data["ego_actions"]
            opponent_actions: Optional[np.ndarray] = data.get("opponent_actions")

        # Normalize windows to shape (N', T, *feat_shape)
        if raw_windows.ndim >= 3 and raw_windows.shape[1] == self.window_len:
            # Already windowed: (N', T, ...) — validate T and use as-is
            if raw_windows.shape[1] != self.window_len:
                raise ValueError(
                    f"Episode {record.path}: expected window_len={self.window_len}, got T={raw_windows.shape[1]}"
                )
            windows = raw_windows
            # Actions should already be aligned per window; if actions still have length N (unwindowed),
            # try to trim them to (N') by dropping the first T-1 to match the last frame of each window.
            if actions.shape[0] == windows.shape[0] + self.window_len - 1:
                actions = actions[self.window_len - 1 :]
                if opponent_actions is not None and opponent_actions.shape[0] == windows.shape[0] + self.window_len - 1:
                    opponent_actions = opponent_actions[self.window_len - 1 :]
            elif actions.shape[0] != windows.shape[0]:
                # If lengths don't match, try the standard alignment (drop first T-1)
                if actions.shape[0] >= self.window_len and (actions.shape[0] - self.window_len + 1) == windows.shape[0]:
                    actions = actions[self.window_len - 1 :]
                    if opponent_actions is not None and opponent_actions.shape[0] >= self.window_len:
                        opponent_actions = opponent_actions[self.window_len - 1 :]
                else:
                    raise ValueError(
                        f"Episode {record.path}: actions length {actions.shape[0]} doesn't match windows {windows.shape[0]}"
                    )

        elif raw_windows.ndim >= 2:
            # Per-frame features: (N, F) or (N, teams, agents, state_dim) -> build sliding windows
            windows = _build_sliding_windows(raw_windows, self.window_len)  # (N', T, ...)
            # Align actions/opponent_actions to the last frame of each window
            N_prime = windows.shape[0]
            expected_actions = actions.shape[0] - self.window_len + 1
            if expected_actions <= 0:
                raise ValueError(
                    f"Episode {record.path}: actions too short for window_len={self.window_len} "
                    f"(actions N={actions.shape[0]})"
                )
            actions = actions[self.window_len - 1 :]
            if actions.shape[0] != N_prime:
                raise ValueError(
                    f"Episode {record.path}: actions/window length mismatch after alignment: "
                    f"{actions.shape[0]} vs {N_prime}"
                )
            if opponent_actions is not None:
                opponent_actions = opponent_actions[self.window_len - 1 :]
                if opponent_actions.shape[0] != N_prime:
                    raise ValueError(
                        f"Episode {record.path}: opponent_actions/window length mismatch after alignment: "
                        f"{opponent_actions.shape[0]} vs {N_prime}"
                    )
        else:
            raise ValueError(f"Episode {record.path}: unexpected windows shape {raw_windows.shape}")

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
        window = episode.windows[local_idx]  # (T, *feat_shape)
        action = episode.actions[local_idx]  # (agents, action_dim) or similar

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
            sample = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }

        return sample

    def describe(self) -> Dict[str, int]:
        return {str(record.path): length for record, length in zip(self.records, self._lengths)}

    def policy_name(self, index: int) -> str:
        return self._policy_inverse.get(int(index), "unknown")