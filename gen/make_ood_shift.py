"""Generate OOD datasets by swapping in tuned opponent policies."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
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
from policyOrProxy.core.policies.oppPolicy import WindowNN
from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict, rng: np.random.Generator) -> WindowHashPolicy:
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


def write_episode(root: Path, shift: str, split: str, episode_id: int, rollout: Dict[str, np.ndarray]) -> Path:
    folder = root / shift / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"episode_{episode_id:05d}.npz"
    np.savez_compressed(
        path,
        windows=rollout["windows"],
        ego_actions=rollout["ego_actions"],
        opponent_actions=rollout["opponent_actions"],
        positions=rollout["positions"],
        policy_id=rollout["policy_id"],
    )
    return path


def main(data_cfg: Path, ego_cfg: Path, shift_cfg: Path, opponents_dir: Path, opponent_cfg: Path) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    ego_config = load_yaml(ego_cfg)
    shift_config = load_yaml(shift_cfg)
    opponent_config = load_yaml(opponent_cfg)
    rng = np.random.default_rng(13)
    arena = build_arena(data_config["arena"])
    world = build_world(arena, data_config["world"], rng)
    ego_policy = build_ego_policy(arena, data_config["world"], ego_config, rng)
    baseline_root = resolve_path(Path(data_config.get("output_root", "output/data/iid")))
    baseline_stats = np.load(baseline_root / "baseline_stats.npz", allow_pickle=False)
    target_root = baseline_root.parent / "ood"
    target_root.mkdir(parents=True, exist_ok=True)
    divergence_report = {}
    opponents_dir = resolve_path(opponents_dir)
    for shift_name, spec in shift_config["shifts"].items():
        LOGGER.info("Generating OOD split %s", shift_name)
        ckpt_path = opponents_dir / f"{shift_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing opponent checkpoint {ckpt_path}")
        opponent = WindowNN(
            window_len=int(opponent_config.get("window_len", data_config["world"]["history"])),
            teams=int(data_config["world"]["teams"]),
            agents=int(data_config["world"]["agents_per_team"]),
            state_dim=int(opponent_config.get("state_dim", 4)),
            hidden_dim=int(opponent_config.get("hidden_dim", 256)),
            layers=int(opponent_config.get("layers", 3)),
            heads=int(opponent_config.get("heads", 4)),
            arch=opponent_config.get("arch", "mlp"),
            dropout=float(opponent_config.get("dropout", 0.1)),
            max_speed=float(opponent_config.get("max_speed", data_config["world"]["max_speed"])),
            activation=opponent_config.get("activation", "gelu"),
            identifier=shift_name,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        opponent.load_state_dict(state_dict)
        indexer = EpisodeIndexer(root=target_root / shift_name)
        states = []
        actions = []
        for split, count in data_config["rollout"]["episodes"].items():
            for episode_idx in range(int(count)):
                world.reset()
                rollout = world.rollout(ego_policy, opponent, steps=int(data_config["rollout"]["steps"]), deterministic=False, policy_id=shift_name)
                path = write_episode(target_root, shift_name, split, episode_idx, rollout)
                indexer.add_episode(split, path, length=rollout["ego_actions"].shape[0], policy_id=shift_name)
                states.append(rollout["windows"][:, -1].reshape(rollout["windows"].shape[0], -1))
                actions.append(rollout["ego_actions"].reshape(rollout["ego_actions"].shape[0], -1))
        indexer.save()
        state_concat = np.concatenate(states, axis=0)
        action_concat = np.concatenate(actions, axis=0)
        div_state = wasserstein_distance_numpy(state_concat, baseline_stats["states"])
        div_action = wasserstein_distance_numpy(action_concat, baseline_stats["actions"])
        divergence_report[shift_name] = {
            "target_state": spec["state_wasserstein"],
            "target_action": spec["action_wasserstein"],
            "achieved_state": div_state,
            "achieved_action": div_action,
        }
    with (target_root / "divergence_report.json").open("w", encoding="utf-8") as fp:
        json.dump(divergence_report, fp, indent=2)
    LOGGER.info("OOD dataset generation complete. Reports saved to %s", target_root / "divergence_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOD datasets")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--ego", type=str, default=str(PACKAGE_ROOT / "cfg" / "ego_policy.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--opponents", type=str, default=str((PACKAGE_ROOT / "../output/opponents").resolve()))
    parser.add_argument("--opponent_cfg", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.ego), Path(args.shift), Path(args.opponents), Path(args.opponent_cfg))
