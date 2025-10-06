"""Manually shift state distributions to match target Wasserstein divergences."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.metrics.metrics import wasserstein_distance_numpy
from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy
from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.world.arena import build_arena

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict) -> WindowHashPolicy:
    regionizer = WindowHashRegionizer(
        arena=arena,
        num_buckets=int(ego_cfg["num_buckets"]),
        grid_size=int(ego_cfg["quantization"]["grid_size"]),
        length_scale=float(ego_cfg["quantization"].get("length_scale", 1.0)),
        jitter=float(ego_cfg["quantization"].get("jitter", 0.0)),
    )
    return WindowHashPolicy(
        regionizer=regionizer,
        num_agents=int(world_cfg["agents_per_team"]),
        num_prototypes=int(ego_cfg["num_prototypes"]),
        max_speed=float(ego_cfg["prototype_init"].get("max_speed", world_cfg["max_speed"])),
        noise_std=float(ego_cfg.get("noise_std", 0.0)),
        sampling=ego_cfg.get("sampling", "stochastic"),
        seed=int(ego_cfg["prototype_init"].get("seed", 0)),
    )


def apply_translation_to_states(states: np.ndarray, arena, teams: int, agents: int, delta: np.ndarray) -> np.ndarray:
    reshaped = states.reshape(states.shape[0], teams, agents, 4).copy()
    positions = reshaped[..., :2]
    positions += delta
    for idx in range(reshaped.shape[0]):
        positions[idx] = arena.clamp_positions(positions[idx])
    reshaped[..., :2] = positions
    return reshaped.reshape(states.shape)


def translate_window(window: np.ndarray, arena, delta: np.ndarray) -> np.ndarray:
    shifted = window.copy()
    for t in range(window.shape[0]):
        positions = shifted[t, :, :, :2]
        positions += delta
        shifted[t, :, :, :2] = arena.clamp_positions(positions)
    return shifted


def find_translation_magnitude(
    target: float,
    baseline_states: np.ndarray,
    arena,
    teams: int,
    agents: int,
    delta: np.ndarray,
) -> float:
    if target <= 0:
        return 0.0
    max_shift = 0.5 * min(arena.width, arena.height)
    low, high = 0.0, max_shift
    for _ in range(30):
        mid = 0.5 * (low + high)
        shifted = apply_translation_to_states(baseline_states, arena, teams, agents, delta * mid)
        dist = wasserstein_distance_numpy(shifted, baseline_states)
        if dist < target:
            low = mid
        else:
            high = mid
    return high


def generate_manual_shift(
    source_root: Path,
    target_root: Path,
    data_config: Dict,
    ego_config: Dict,
    shift_config: Dict,
) -> Dict[str, Dict[str, float]]:
    arena = build_arena(data_config["arena"])
    world_cfg = data_config["world"]
    teams = int(world_cfg["teams"])
    agents = int(world_cfg["agents_per_team"])

    baseline_stats_path = source_root / "baseline_stats.npz"
    if not baseline_stats_path.exists():
        raise FileNotFoundError(f"Missing baseline stats at {baseline_stats_path}")
    baseline_stats = np.load(baseline_stats_path, allow_pickle=False)
    baseline_states = baseline_stats["states"]
    baseline_actions = baseline_stats["actions"]

    delta_matrix = np.ones((teams, agents, 2), dtype=np.float32)

    indexer = EpisodeIndexer.load(source_root)
    target_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}

    for shift_name, targets in shift_config["shifts"].items():
        LOGGER.info("Generating manual state shift %s", shift_name)
        state_target = float(targets["state_wasserstein"])
        delta_magnitude = find_translation_magnitude(
            state_target, baseline_states, arena, teams, agents, delta_matrix
        )
        delta_offsets = delta_matrix * delta_magnitude
        ego_policy = build_ego_policy(arena, world_cfg, ego_config)
        new_indexer = EpisodeIndexer(root=target_root / shift_name)
        all_states = []
        all_actions = []
        for split in indexer.splits():
            records = list(indexer.iter_split(split))
            split_dir = (target_root / shift_name / split)
            split_dir.mkdir(parents=True, exist_ok=True)
            for episode_idx, record in enumerate(records):
                source_path = source_root / record.path
                with np.load(source_path, allow_pickle=False) as data:
                    windows = data["windows"].astype(np.float32)
                    opponent_actions = data.get("opponent_actions")
                    positions = data.get("positions")
                shifted_windows = np.asarray([translate_window(win, arena, delta_offsets) for win in windows])
                ego_actions = np.asarray([ego_policy.act(win, deterministic=False) for win in shifted_windows])
                episode_path = split_dir / f"episode_{episode_idx:05d}.npz"
                payload = {
                    "windows": shifted_windows,
                    "ego_actions": ego_actions,
                    "policy_id": shift_name,
                }
                if opponent_actions is not None:
                    payload["opponent_actions"] = opponent_actions
                if positions is not None:
                    payload["positions"] = positions
                np.savez_compressed(episode_path, **payload)
                new_indexer.add_episode(split, episode_path, length=ego_actions.shape[0], policy_id=shift_name)
                all_states.append(shifted_windows[:, -1].reshape(shifted_windows.shape[0], -1))
                all_actions.append(ego_actions.reshape(ego_actions.shape[0], -1))
        new_indexer.save()
        states_concat = np.concatenate(all_states, axis=0)
        actions_concat = np.concatenate(all_actions, axis=0)
        achieved_state = wasserstein_distance_numpy(states_concat, baseline_states)
        achieved_action = wasserstein_distance_numpy(actions_concat, baseline_actions)
        np.savez_compressed(target_root / shift_name / "baseline_stats.npz", states=states_concat, actions=actions_concat)
        summary[shift_name] = {
            "target_state": state_target,
            "target_action": float(targets["action_wasserstein"]),
            "achieved_state": achieved_state,
            "achieved_action": achieved_action,
            "delta_magnitude": delta_magnitude,
        }
    with (target_root / "manual_shift_report.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def main(data_cfg: Path, ego_cfg: Path, shift_cfg: Path, source: Path, target: Path) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    ego_config = load_yaml(ego_cfg)
    shift_config = load_yaml(shift_cfg)
    source_root = resolve_path(source)
    target_root = resolve_path(target)
    summary = generate_manual_shift(source_root, target_root, data_config, ego_config, shift_config)
    LOGGER.info("Manual shift generation complete: %s", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually shift dataset state distributions")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--ego", type=str, default=str(PACKAGE_ROOT / "cfg" / "ego_policy.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--source", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()))
    parser.add_argument("--target", type=str, default=str((PACKAGE_ROOT / "../output/data/ood_manual").resolve()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.ego), Path(args.shift), Path(args.source), Path(args.target))
