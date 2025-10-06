"""Sanity-check: shuffle IID dataset actions across the entire split (breaks causality)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from shutil import copyfile
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


def load_split(root: Path, records):
    windows = []
    actions = []
    meta = []
    for record in records:
        path = root / record.path
        with np.load(path, allow_pickle=False) as data:
            windows.append(data["windows"])
            actions.append(data["ego_actions"])
            meta.append({"len": data["ego_actions"].shape[0], "policy": data["policy_id"].item() if "policy_id" in data else None})
    return windows, actions, meta


def write_split(output_root: Path, split: str, windows, shuffled_actions, opponent_actions, positions, policies) -> None:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx, (win, act) in enumerate(zip(windows, shuffled_actions)):
        path = split_dir / f"episode_{idx:05d}.npz"
        np.savez_compressed(
            path,
            windows=win,
            ego_actions=act,
            opponent_actions=opponent_actions[idx],
            positions=positions[idx],
            policy_id=policies[idx],
        )


def main(source: Path, target: Path) -> None:
    configure_logging()
    rng = np.random.default_rng(17)
    source = resolve_path(source)
    target = resolve_path(target)
    indexer = EpisodeIndexer.load(source)
    target.mkdir(parents=True, exist_ok=True)
    new_indexer = EpisodeIndexer(root=target)
    for split in indexer.splits():
        records = list(indexer.iter_split(split))
        windows = []
        actions = []
        opponents = []
        positions = []
        policies = []
        for record in records:
            path = source / record.path
            with np.load(path, allow_pickle=False) as data:
                windows.append(data["windows"])
                actions.append(data["ego_actions"])
                opponents.append(data["opponent_actions"])
                positions.append(data["positions"])
                policy = data["policy_id"] if "policy_id" in data else "baseline"
                policies.append(policy.item() if hasattr(policy, "item") else policy)
        concatenated = np.concatenate([a.reshape(a.shape[0], -1) for a in actions], axis=0)
        perm = rng.permutation(concatenated.shape[0])
        shuffled_flat = concatenated[perm]
        pointer = 0
        shuffled_actions = []
        for act in actions:
            count = act.shape[0]
            chunk = shuffled_flat[pointer:pointer + count]
            shuffled_actions.append(chunk.reshape(act.shape))
            pointer += count
        write_split(target, split, windows, shuffled_actions, opponents, positions, policies)
        for idx, record in enumerate(records):
            path = target / split / f"episode_{idx:05d}.npz"
            new_indexer.add_episode(split, path, length=actions[idx].shape[0], policy_id=policies[idx])
        LOGGER.info("Shuffled split %s", split)
    new_indexer.save()
    baseline = source / "baseline_stats.npz"
    if baseline.exists():
        copyfile(baseline, target / "baseline_stats.npz")
    LOGGER.info("Shuffled dataset written to %s", target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle dataset actions across samples")
    parser.add_argument("--source", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()))
    parser.add_argument("--target", type=str, default=str((PACKAGE_ROOT / "../output/data/iid_bad_shuffle").resolve()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.source), Path(args.target))
