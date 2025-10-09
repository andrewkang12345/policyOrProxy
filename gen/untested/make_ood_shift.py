"""Generate OOD datasets for every ego policy configuration."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.metrics.metrics import wasserstein_distance_numpy
from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy, build_window_hash_policy
from policyOrProxy.core.policies.oppPolicy import build_hash_policy
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


def policy_name_from_path(path: Path) -> str:
    return path.stem


def find_ego_configs(explicit: List[str] | None) -> List[Path]:
    if explicit:
        return [resolve_path(Path(p)) for p in explicit]
    configs = sorted(CFG_DIR.glob("ego_policy*.yaml"))
    if not configs:
        raise FileNotFoundError("No ego_policy*.yaml configs found")
    return configs


def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict) -> WindowHashPolicy:
    return build_window_hash_policy(
        arena=arena,
        world_cfg=world_cfg,
        policy_cfg=ego_cfg,
        identifier=ego_cfg.get("identifier", "ego_window_hash"),
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


def generate_policy_ood(
    policy_cfg_path: Path,
    data_config: Dict,
    shift_config: Dict,
    opponent_config: Dict,
    opponents_root: Path,
) -> None:
    name = policy_name_from_path(policy_cfg_path)
    ego_config = load_yaml(policy_cfg_path)
    rng = np.random.default_rng(13)

    arena = build_arena(data_config["arena"])
    world = build_world(arena, data_config["world"], rng)
    ego_policy = build_ego_policy(arena, data_config["world"], ego_config)

    base_output = resolve_path(Path(data_config.get("output_root", "output/data")))
    policy_root = base_output / name
    baseline_root = policy_root / "iid"
    if not baseline_root.exists():
        raise FileNotFoundError(f"Missing baseline dataset for {name} at {baseline_root}")
    baseline_stats = np.load(baseline_root / "baseline_stats.npz", allow_pickle=False)
    target_root = policy_root / "ood"
    target_root.mkdir(parents=True, exist_ok=True)

    policy_opponents = opponents_root / name
    if not policy_opponents.exists():
        raise FileNotFoundError(f"Missing opponents for {name} at {policy_opponents}")

    divergence_report = {}
    for shift_idx, (shift_name, spec) in enumerate(shift_config["shifts"].items(), start=1):
        LOGGER.info("[%s] Generating OOD split %s", name, shift_name)
        ckpt_path = policy_opponents / f"{shift_name}.npz"
        opponent = build_hash_policy(
            arena=arena,
            world_cfg=data_config["world"],
            opp_cfg=opponent_config,
            seed_offset=shift_idx,
            identifier=f"{name}_{shift_name}",
        )
        if ckpt_path.exists():
            with np.load(ckpt_path, allow_pickle=False) as opponent_state:
                state_payload = {"prototypes": opponent_state["prototypes"]}
                if "weights" in opponent_state.files:
                    state_payload["weights"] = opponent_state["weights"]
            opponent.load_state(state_payload)
        else:
            LOGGER.warning("Missing opponent state at %s; using freshly initialized policy.", ckpt_path)
        indexer = EpisodeIndexer(root=target_root / shift_name)
        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        for split, count in data_config["rollout"]["episodes"].items():
            for episode_idx in range(int(count)):
                world.reset()
                rollout = world.rollout(
                    ego_policy,
                    opponent,
                    steps=int(data_config["rollout"]["steps"]),
                    deterministic=False,
                    policy_id=f"{name}_{shift_name}",
                )
                path = write_episode(target_root, shift_name, split, episode_idx, rollout)
                indexer.add_episode(split, path, length=rollout["ego_actions"].shape[0], policy_id=f"{name}_{shift_name}")
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
    LOGGER.info("[%s] OOD dataset generation complete", name)


def main(data_cfg: Path, opponent_cfg: Path, shift_cfg: Path, opponents_dir: Path, ego_cfgs: List[str] | None) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opponent_config = load_yaml(opponent_cfg)
    shift_config = load_yaml(shift_cfg)
    opponents_root = resolve_path(opponents_dir)
    configs = find_ego_configs(ego_cfgs)
    for cfg_path in configs:
        generate_policy_ood(cfg_path, data_config, shift_config, opponent_config, opponents_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOD datasets")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent_cfg", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--opponents", type=str, default=str((PACKAGE_ROOT / "../output/opponents").resolve()))
    parser.add_argument("--ego", action="append", help="Specific ego policy configs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.opponent_cfg), Path(args.shift), Path(args.opponents), args.ego)
