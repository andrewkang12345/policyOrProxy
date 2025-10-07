"""Generate IID baselines for every available ego policy configuration."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"
EGO_PATTERN = "ego_policy*.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy
from policyOrProxy.core.policies.oppPolicy import WindowNN
from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def find_ego_configs(explicit: Iterable[str] | None = None) -> List[Path]:
    if explicit:
        return [resolve_path(Path(p)) for p in explicit]
    configs = sorted(CFG_DIR.glob(EGO_PATTERN))
    if not configs:
        raise FileNotFoundError(f"No ego policy configs matching {EGO_PATTERN}")
    return configs


def policy_name_from_path(path: Path) -> str:
    return path.stem  # e.g. ego_policy1


def build_policies(
    arena_cfg: Dict,
    ego_cfg: Dict,
    opp_cfg: Dict,
    world_cfg: Dict,
    rng: np.random.Generator,
) -> tuple:
    arena = build_arena(arena_cfg)
    world = build_world(arena, world_cfg, rng=rng)
    regionizer = WindowHashRegionizer(
        arena=arena,
        num_buckets=int(ego_cfg["num_buckets"]),
        grid_size=int(ego_cfg["quantization"]["grid_size"]),
        length_scale=float(ego_cfg["quantization"].get("length_scale", 1.0)),
        jitter=float(ego_cfg["quantization"].get("jitter", 0.0)),
    )
    ego_policy = WindowHashPolicy(
        regionizer=regionizer,
        num_agents=world.config.agents_per_team,
        num_prototypes=int(ego_cfg["num_prototypes"]),
        max_speed=float(ego_cfg["prototype_init"].get("max_speed", world.config.max_speed)),
        noise_std=float(ego_cfg.get("noise_std", 0.0)),
        sampling=ego_cfg.get("sampling", "stochastic"),
        seed=int(ego_cfg["prototype_init"].get("seed", 0)),
    )
    opp_policy = WindowNN(
        window_len=int(opp_cfg["window_len"]),
        teams=world.config.teams,
        agents=world.config.agents_per_team,
        state_dim=int(opp_cfg.get("state_dim", 4)),
        hidden_dim=int(opp_cfg.get("hidden_dim", 256)),
        layers=int(opp_cfg.get("layers", 3)),
        heads=int(opp_cfg.get("heads", 4)),
        dropout=float(opp_cfg.get("dropout", 0.1)),
        arch=opp_cfg.get("arch", "mlp"),
        activation=opp_cfg.get("activation", "gelu"),
        max_speed=float(opp_cfg.get("max_speed", world.config.max_speed)),
        identifier="baseline",
    )
    return arena, world, ego_policy, opp_policy


def write_episode(root: Path, split: str, episode_id: int, rollout: Dict[str, np.ndarray]) -> Path:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"episode_{episode_id:05d}.npz"
    np.savez_compressed(
        path,
        windows=rollout["windows"],
        ego_actions=rollout["ego_actions"],
        opponent_actions=rollout["opponent_actions"],
        positions=rollout["positions"],
        policy_id=rollout["policy_id"],
    )
    return path


def generate_for_policy(
    policy_config: Path,
    data_cfg: Dict,
    opp_cfg: Dict,
) -> None:
    policy_cfg = load_yaml(policy_config)
    name = policy_name_from_path(policy_config)
    output_root = resolve_path(Path(data_cfg.get("output_root", "output/data"))) / name / "iid"
    output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(data_cfg["rollout"].get("seed", 0)))
    arena, world, ego_policy, opp_policy = build_policies(
        data_cfg["arena"],
        policy_cfg,
        opp_cfg,
        data_cfg["world"],
        rng,
    )

    rollout_cfg = data_cfg["rollout"]
    episodes_cfg = rollout_cfg["episodes"]
    indexer = EpisodeIndexer(root=output_root)
    baseline_states = []
    baseline_actions = []
    policy_id = name

    LOGGER.info("Generating IID data for %s into %s", name, output_root)
    for split, count in episodes_cfg.items():
        for episode_idx in range(int(count)):
            world.reset()
            rollout = world.rollout(
                ego_policy,
                opp_policy,
                steps=int(rollout_cfg["steps"]),
                deterministic=False,
                policy_id=policy_id,
            )
            path = write_episode(output_root, split, episode_idx, rollout)
            indexer.add_episode(split, path, length=rollout["ego_actions"].shape[0], policy_id=policy_id)
            final_states = rollout["windows"][:, -1].reshape(rollout["windows"].shape[0], -1)
            baseline_states.append(final_states)
            baseline_actions.append(rollout["ego_actions"].reshape(rollout["ego_actions"].shape[0], -1))
    indexer.save()
    states = np.concatenate(baseline_states, axis=0)
    actions = np.concatenate(baseline_actions, axis=0)
    sample_size = min(5000, states.shape[0])
    sample_indices = np.random.default_rng().choice(states.shape[0], size=sample_size, replace=False)
    np.savez_compressed(output_root / "baseline_stats.npz", states=states[sample_indices], actions=actions[sample_indices])
    LOGGER.info("Finished %s", name)


def main(data_cfg: Path, opponent_cfg: Path, ego_cfgs: List[str]) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opp_config = load_yaml(opponent_cfg)
    configs = find_ego_configs(ego_cfgs)
    for cfg in configs:
        generate_for_policy(cfg, data_config, opp_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IID baselines for all ego policies")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy.yaml"))
    parser.add_argument("--ego", action="append", help="Specific ego policy yaml(s) to use (defaults to all ego_policy*.yaml)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.opponent), args.ego or [])
