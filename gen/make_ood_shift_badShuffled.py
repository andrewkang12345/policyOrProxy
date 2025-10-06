"""Sanity-check: shuffle OOD datasets across entire splits (break causality)."""
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


def shuffle_split(source_root: Path, target_root: Path) -> None:
    indexer = EpisodeIndexer.load(source_root)
    target_root.mkdir(parents=True, exist_ok=True)
    new_indexer = EpisodeIndexer(root=target_root)
    rng = np.random.default_rng(23)
    for split in indexer.splits():
        records = list(indexer.iter_split(split))
        windows = []
        actions = []
        opponents = []
        positions = []
        policies = []
        for record in records:
            path = source_root / record.path
            with np.load(path, allow_pickle=False) as data:
                windows.append(data["windows"])
                actions.append(data["ego_actions"])
                opponents.append(data["opponent_actions"])
                positions.append(data["positions"])
                policy = data["policy_id"] if "policy_id" in data else "baseline"
                policies.append(policy.item() if hasattr(policy, "item") else policy)
        concat = np.concatenate([a.reshape(a.shape[0], -1) for a in actions], axis=0)
        perm = rng.permutation(concat.shape[0])
        shuffled = concat[perm]
        pointer = 0
        shuffled_actions = []
        for act in actions:
            count = act.shape[0]
            chunk = shuffled[pointer:pointer + count]
            shuffled_actions.append(chunk.reshape(act.shape))
            pointer += count
        split_dir = target_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx, (win, act) in enumerate(zip(windows, shuffled_actions)):
            path = split_dir / f"episode_{idx:05d}.npz"
            np.savez_compressed(
                path,
                windows=win,
                ego_actions=act,
                opponent_actions=opponents[idx],
                positions=positions[idx],
                policy_id=policies[idx],
            )
            new_indexer.add_episode(split, path, length=act.shape[0], policy_id=policies[idx])
    new_indexer.save()
    stats_path = source_root / "baseline_stats.npz"
    if stats_path.exists():
        copyfile(stats_path, target_root / "baseline_stats.npz")


def main(source: Path, target: Path) -> None:
    configure_logging()
    source = resolve_path(source)
    target = resolve_path(target)
    target.mkdir(parents=True, exist_ok=True)
    for shift_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        LOGGER.info("Shuffling shift %s", shift_dir.name)
        shuffle_split(shift_dir, target / shift_dir.name)
    report = source / "divergence_report.json"
    if report.exists():
        copyfile(report, target / "divergence_report.json")
    LOGGER.info("Shuffled OOD datasets stored in %s", target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle OOD datasets")
    parser.add_argument("--source", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()))
    parser.add_argument("--target", type=str, default=str((PACKAGE_ROOT / "../output/data/ood_bad_shuffle").resolve()))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.source), Path(args.target))
