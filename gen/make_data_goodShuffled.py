"""Shuffle IID dataset within each episode while preserving state-action pairs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


from policyOrProxy.core.dataset.indexer import EpisodeIndexer

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def shuffle_episode(
    rng: np.random.Generator,
    windows: np.ndarray,
    actions: np.ndarray,
    opponent_actions: np.ndarray | None,
    positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    perm = rng.permutation(actions.shape[0])
    win = windows[perm]
    act = actions[perm]
    opp = opponent_actions[perm] if opponent_actions is not None else None
    pos = positions[perm] if positions is not None else None
    return win, act, opp, pos


def write_episode(
    path: Path,
    windows: np.ndarray,
    actions: np.ndarray,
    opponent_actions: np.ndarray | None,
    positions: np.ndarray | None,
    policy_id,
) -> None:
    payload = {
        "windows": windows,
        "ego_actions": actions,
        "policy_id": policy_id,
    }
    if opponent_actions is not None:
        payload["opponent_actions"] = opponent_actions
    if positions is not None:
        payload["positions"] = positions
    np.savez_compressed(path, **payload)


def iter_policies(base: Path, policies: list[str] | None) -> List[Path]:
    if policies:
        return [base / name for name in policies]
    return [p for p in sorted(base.iterdir()) if p.is_dir()]


def process_policy(policy_root: Path, dataset: str, target_name: str, rng: np.random.Generator) -> None:
    source = policy_root / dataset
    if not source.exists():
        LOGGER.warning("Skipping %s (missing %s)", policy_root.name, dataset)
        return
    target = policy_root / target_name
    indexer = EpisodeIndexer.load(source)
    target.mkdir(parents=True, exist_ok=True)
    new_indexer = EpisodeIndexer(root=target)
    for split in indexer.splits():
        records = list(indexer.iter_split(split))
        split_dir = target / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, record in enumerate(records):
            path = source / record.path
            with np.load(path, allow_pickle=False) as data:
                windows = data["windows"]
                actions = data["ego_actions"]
                opponent_actions = data.get("opponent_actions")
                positions = data.get("positions")
                policy = data.get("policy_id", policy_root.name)
                if hasattr(policy, "item"):
                    policy = policy.item()
            shuffled = shuffle_episode(rng, windows, actions, opponent_actions, positions)
            episode_path = split_dir / f"episode_{idx:05d}.npz"
            write_episode(episode_path, *shuffled, policy_id=policy)
            new_indexer.add_episode(split, episode_path, length=actions.shape[0], policy_id=policy)
        LOGGER.info("[%s] per-episode shuffled split %s", policy_root.name, split)
    new_indexer.save()
    baseline_stats = source / "baseline_stats.npz"
    if baseline_stats.exists():
        (target / "baseline_stats.npz").write_bytes(baseline_stats.read_bytes())
    LOGGER.info("Per-episode shuffled dataset stored in %s", target)


def main(base: Path, dataset: str, target: str, policies: list[str] | None) -> None:
    configure_logging()
    rng = np.random.default_rng(29)
    base = resolve_path(base)
    for policy_root in iter_policies(base, policies):
        process_policy(policy_root, dataset, target, rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle dataset within each episode")
    parser.add_argument("--base", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()))
    parser.add_argument("--dataset", type=str, default="iid")
    parser.add_argument("--target", type=str, default="iid_good_shuffle")
    parser.add_argument("--policy", action="append", help="Specific policy directories to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.base), args.dataset, args.target, args.policy)
