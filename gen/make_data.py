#!/usr/bin/env python3
"""
Generate IID baselines for every available ego policy configuration.

Key features:
- Policy CLASS is explicit in YAML via `policy_class`.
- Continuable: appends missing episodes per split; does not overwrite existing episodes/index.
- Deterministic appends:
    episode seed = f(base_seed, policy_name, split, episode_id, role)
  so generating more later yields the same earlier episodes.
- Episodes are now VARIABLE LENGTH:
    the World terminates early when a wall hit occurs (no reflection),
    so `rollout.steps` in data.yaml is a MAXIMUM.
- Rebuilds baseline_stats.npz deterministically (reservoir sampling).

Expected YAML:
- data.yaml: arena/world/rollout/output_root
- opponent YAML: policy_class + params
- ego YAML(s): policy_class + params
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"
EGO_PATTERN = "ego_policy*.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.policies.policy_factory import build_policy
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world

LOGGER = logging.getLogger(__name__)
EP_RE = re.compile(r"episode_(\d+)\.npz$")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


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


def stable_int_seed(*parts: object, bits: int = 32) -> int:
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(msg).digest()
    if bits <= 32:
        return int.from_bytes(digest[:4], "little", signed=False)
    if bits <= 64:
        return int.from_bytes(digest[:8], "little", signed=False)
    return int.from_bytes(digest[:16], "little", signed=False)


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


def _parse_episode_id_from_path_str(p: str) -> Optional[int]:
    m = EP_RE.search(p.replace("\\", "/"))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _existing_episode_ids_from_index(indexer: EpisodeIndexer, split: str) -> List[int]:
    ids: List[int] = []
    for rec in indexer.iter_split(split):
        eid = _parse_episode_id_from_path_str(rec.path)
        if eid is not None:
            ids.append(eid)
    return sorted(set(ids))


def _existing_episode_ids_from_disk(root: Path, split: str) -> List[int]:
    ids: List[int] = []
    split_dir = root / split
    if not split_dir.exists():
        return []
    for p in split_dir.glob("episode_*.npz"):
        m = EP_RE.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def _load_or_create_indexer(output_root: Path) -> EpisodeIndexer:
    idx_path = output_root / "index.json"
    if idx_path.exists():
        try:
            return EpisodeIndexer.load(output_root)
        except Exception as e:
            LOGGER.warning("Found index.json but failed to load (%s). Rebuilding index from disk.", e)
    return EpisodeIndexer(root=output_root)


def _ensure_index_matches_disk(indexer: EpisodeIndexer, output_root: Path) -> EpisodeIndexer:
    indexed_paths = {Path(rec.path).as_posix() for rec in indexer.entries}
    for split_dir in ("train", "val", "test"):
        for p in (output_root / split_dir).glob("episode_*.npz"):
            rel = p.relative_to(output_root).as_posix()
            if rel in indexed_paths:
                continue
            try:
                with np.load(p, allow_pickle=False) as data:
                    ego_actions = np.asarray(data["ego_actions"])
                    length = int(ego_actions.shape[0])
                    policy_id = None
                    if "policy_id" in data:
                        policy_id = str(np.asarray(data["policy_id"]).item())
            except Exception:
                continue
            indexer.add_episode(split_dir, p, length=length, policy_id=policy_id)
            indexed_paths.add(rel)
    return indexer


def _reservoir_update(
    rng: np.random.Generator,
    states_row: np.ndarray,
    actions_row: np.ndarray,
    K: int,
    seen: int,
    filled: int,
    states_res: Optional[np.ndarray],
    actions_res: Optional[np.ndarray],
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    seen += 1
    if states_res is None or actions_res is None:
        states_res = np.zeros((K, states_row.shape[0]), dtype=np.float32)
        actions_res = np.zeros((K, actions_row.shape[0]), dtype=np.float32)

    if filled < K:
        states_res[filled] = states_row
        actions_res[filled] = actions_row
        filled += 1
        return seen, filled, states_res, actions_res

    j = int(rng.integers(0, seen))
    if j < K:
        states_res[j] = states_row
        actions_res[j] = actions_row
    return seen, filled, states_res, actions_res


def rebuild_baseline_stats(
    output_root: Path,
    indexer: EpisodeIndexer,
    policy_name: str,
    base_seed: int,
    max_samples: int = 5000,
) -> None:
    stats_rng = np.random.default_rng(stable_int_seed(base_seed, "baseline_stats", policy_name))

    K = int(max_samples)
    seen = 0
    filled = 0
    states_res: Optional[np.ndarray] = None
    actions_res: Optional[np.ndarray] = None

    entries = sorted(indexer.entries, key=lambda r: (r.split, r.path))
    for rec in entries:
        p = output_root / rec.path
        if not p.exists():
            continue
        try:
            with np.load(p, allow_pickle=False) as data:
                windows = np.asarray(data["windows"], dtype=np.float32)          # (L,T,teams,agents,state_dim)
                ego_actions = np.asarray(data["ego_actions"], dtype=np.float32)  # (L,agents,2)
        except Exception:
            continue

        if windows.ndim != 5 or ego_actions.ndim != 3:
            continue

        # Use the last frame of each window as the "next-state" proxy (unchanged),
        # and pair with action at that timestep.
        final_states = windows[:, -1].reshape(windows.shape[0], -1).astype(np.float32)
        flat_actions = ego_actions.reshape(ego_actions.shape[0], -1).astype(np.float32)

        n = int(min(final_states.shape[0], flat_actions.shape[0]))
        for i in range(n):
            seen, filled, states_res, actions_res = _reservoir_update(
                stats_rng, final_states[i], flat_actions[i], K, seen, filled, states_res, actions_res
            )

    if states_res is None or actions_res is None or filled == 0:
        LOGGER.warning("No samples available to write baseline_stats.npz for %s", policy_name)
        return

    states_out = states_res[:filled].copy()
    actions_out = actions_res[:filled].copy()
    np.savez_compressed(output_root / "baseline_stats.npz", states=states_out, actions=actions_out)
    LOGGER.info("Wrote baseline_stats.npz with %d samples (seen=%d)", filled, seen)


def build_arena_world_and_policies(
    data_cfg: Dict,
    ego_cfg: Dict,
    opp_cfg: Dict,
    *,
    base_seed: int,
    ego_name: str,
) -> tuple:
    arena = build_arena(data_cfg["arena"])

    init_rng = np.random.default_rng(stable_int_seed(base_seed, "init_world", ego_name))
    world = build_world(arena, data_cfg["world"], rng=init_rng)

    ego_init_seed = stable_int_seed(base_seed, ego_name, "init", "ego")
    opp_init_seed = stable_int_seed(base_seed, ego_name, "init", "opp")

    ego_policy = build_policy(
        arena=arena,
        world_cfg=data_cfg["world"],
        policy_cfg=ego_cfg,
        role="ego",
        identifier=ego_cfg.get("identifier", ego_name),
        init_seed=ego_init_seed,
    )
    opp_policy = build_policy(
        arena=arena,
        world_cfg=data_cfg["world"],
        policy_cfg=opp_cfg,
        role="opponent",
        identifier=opp_cfg.get("identifier", "opponent"),
        init_seed=opp_init_seed,
    )
    return arena, world, ego_policy, opp_policy


def generate_for_policy(
    policy_config: Path,
    data_cfg: Dict,
    opp_cfg: Dict,
    *,
    overwrite: bool = False,
    refresh_stats: bool = True,
) -> None:
    ego_cfg = load_yaml(policy_config)
    name = policy_name_from_path(policy_config)

    output_root = resolve_path(Path(data_cfg.get("output_root", "output/data"))) / name / "iid"
    if overwrite and output_root.exists():
        LOGGER.warning("Overwrite enabled: removing %s", output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rollout_cfg = data_cfg["rollout"]
    episodes_cfg: Dict[str, int] = rollout_cfg["episodes"]
    base_seed = int(rollout_cfg.get("seed", 0))
    max_steps = int(rollout_cfg["steps"])

    indexer = _load_or_create_indexer(output_root)
    indexer = _ensure_index_matches_disk(indexer, output_root)

    arena, world, ego_policy, opp_policy = build_arena_world_and_policies(
        data_cfg=data_cfg,
        ego_cfg=ego_cfg,
        opp_cfg=opp_cfg,
        base_seed=base_seed,
        ego_name=name,
    )

    policy_id = name
    LOGGER.info("Generating IID data for %s into %s", name, output_root)
    LOGGER.info("Episodes terminate early on wall-hit; rollout.steps is max_steps=%d", max_steps)

    for split, target_count_raw in episodes_cfg.items():
        target_count = int(target_count_raw)

        existing_ids = _existing_episode_ids_from_index(indexer, split)
        if not existing_ids:
            existing_ids = _existing_episode_ids_from_disk(output_root, split)

        existing_n = len(existing_ids)
        if existing_n >= target_count:
            LOGGER.info("Split '%s': already have %d episodes (target=%d). Skipping.", split, existing_n, target_count)
            continue

        next_id = (max(existing_ids) + 1) if existing_ids else 0
        to_make = target_count - existing_n
        LOGGER.info(
            "Split '%s': have %d, target %d -> generating %d more (starting at id %d).",
            split, existing_n, target_count, to_make, next_id
        )

        for k in range(to_make):
            episode_id = next_id + k

            ep_seed_world = stable_int_seed(base_seed, name, split, episode_id, "world")
            ep_seed_ego = stable_int_seed(base_seed, name, split, episode_id, "ego")
            ep_seed_opp = stable_int_seed(base_seed, name, split, episode_id, "opp")

            world.rng = np.random.default_rng(ep_seed_world)
            world.reset()

            if hasattr(ego_policy, "set_rng"):
                ego_policy.set_rng(np.random.default_rng(ep_seed_ego))
            elif hasattr(ego_policy, "rng"):
                ego_policy.rng = np.random.default_rng(ep_seed_ego)

            if hasattr(opp_policy, "set_rng"):
                opp_policy.set_rng(np.random.default_rng(ep_seed_opp))
            elif hasattr(opp_policy, "rng"):
                opp_policy.rng = np.random.default_rng(ep_seed_opp)

            rollout = world.rollout(
                ego_policy,
                opp_policy,
                steps=max_steps,          # maximum; may end earlier
                deterministic=False,
                policy_id=policy_id,
            )

            L = int(rollout["ego_actions"].shape[0])
            if L == 0:
                LOGGER.warning("Episode %s/%s/%05d produced length 0; check terminate logic / dt / spawn.", name, split, episode_id)

            path = write_episode(output_root, split, episode_id, rollout)
            indexer.add_episode(split, path, length=L, policy_id=policy_id)

    indexer.save()
    LOGGER.info("Saved index.json for %s", name)

    if refresh_stats:
        rebuild_baseline_stats(output_root, indexer, policy_name=name, base_seed=base_seed, max_samples=5000)

    LOGGER.info("Finished %s", name)


def main(data_cfg: Path, opponent_cfg: Path, ego_cfgs: List[str], overwrite: bool, refresh_stats: bool) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opp_config = load_yaml(opponent_cfg)

    configs = find_ego_configs(ego_cfgs)
    for cfg in configs:
        generate_for_policy(cfg, data_config, opp_config, overwrite=overwrite, refresh_stats=refresh_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate (continuable) IID baselines for ego policies (multi-class)")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    parser.add_argument("--ego", action="append", help="Specific ego policy yaml(s) (defaults to all ego_policy*.yaml)")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output and regenerate from scratch")
    parser.add_argument("--no_refresh_stats", action="store_true", help="Do not rebuild baseline_stats.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        Path(args.data),
        Path(args.opponent),
        args.ego or [],
        overwrite=bool(args.overwrite),
        refresh_stats=not bool(args.no_refresh_stats),
    )
