"""Basic dataset sanity checks for distributions and metrics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.metrics.metrics import wasserstein_distance_numpy

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def summarize_split(root: Path, records):
    positions = []
    actions = []
    for record in records:
        with np.load(root / record.path, allow_pickle=False) as data:
            positions.append(data["positions"])
            actions.append(data["ego_actions"])
    pos = np.concatenate(positions, axis=0)
    act = np.concatenate(actions, axis=0)
    pos_mean = pos.mean(axis=(0, 1, 2))
    pos_std = pos.std(axis=(0, 1, 2))
    act_mean = act.mean(axis=(0, 1))
    act_std = act.std(axis=(0, 1))
    return {
        "positions": {"mean": pos_mean.tolist(), "std": pos_std.tolist()},
        "actions": {"mean": act_mean.tolist(), "std": act_std.tolist()},
    }


def main(root: Path, baseline: Path | None) -> None:
    configure_logging()
    indexer = EpisodeIndexer.load(root)
    report = {}
    for split in indexer.splits():
        records = list(indexer.iter_split(split))
        report[split] = summarize_split(root, records)
        LOGGER.info("Split %s: pos std=%s action std=%s", split, report[split]["positions"]["std"], report[split]["actions"]["std"])
    if baseline is not None and baseline.exists():
        stats = np.load(baseline, allow_pickle=False)
        states_ref = stats["states"]
        actions_ref = stats["actions"]
        states = []
        actions = []
        for record in indexer.entries:
            with np.load(root / record.path, allow_pickle=False) as data:
                states.append(data["windows"][:, -1].reshape(data["windows"].shape[0], -1))
                actions.append(data["ego_actions"].reshape(data["ego_actions"].shape[0], -1))
        states_concat = np.concatenate(states, axis=0)
        actions_concat = np.concatenate(actions, axis=0)
        w_state = wasserstein_distance_numpy(states_concat, states_ref)
        w_action = wasserstein_distance_numpy(actions_concat, actions_ref)
        LOGGER.info("Wasserstein(state)=%.3f action=%.3f", w_state, w_action)
    else:
        LOGGER.info("Baseline stats not provided; skipping Wasserstein comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic dataset sanity checks")
    parser.add_argument("--root", type=str, default="output/data/iid")
    parser.add_argument("--baseline", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    baseline = Path(args.baseline) if args.baseline else None
    main(Path(args.root), baseline)
