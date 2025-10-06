"""Compute Wasserstein distances for each dataset split relative to baseline."""
from __future__ import annotations

import argparse
import json
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


def collect_samples(root: Path) -> tuple[np.ndarray, np.ndarray]:
    indexer = EpisodeIndexer.load(root)
    states = []
    actions = []
    for record in indexer.entries:
        with np.load(root / record.path, allow_pickle=False) as data:
            states.append(data["windows"][:, -1].reshape(data["windows"].shape[0], -1))
            actions.append(data["ego_actions"].reshape(data["ego_actions"].shape[0], -1))
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def compute_split_report(dataset_root: Path, baseline_states: np.ndarray, baseline_actions: np.ndarray) -> dict:
    states, actions = collect_samples(dataset_root)
    return {
        "state_wasserstein": float(wasserstein_distance_numpy(states, baseline_states)),
        "action_wasserstein": float(wasserstein_distance_numpy(actions, baseline_actions)),
    }


def main(dataset_root: Path, baseline_root: Path, output: Path) -> None:
    configure_logging()
    baseline_states, baseline_actions = collect_samples(baseline_root)
    report = {}
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            LOGGER.info("Evaluating dataset %s", path.name)
            report[path.name] = compute_split_report(path, baseline_states, baseline_actions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    LOGGER.info("Wasserstein report stored at %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Wasserstein divergences")
    parser.add_argument("--datasets", type=str, default="output/data/ood")
    parser.add_argument("--baseline", type=str, default="output/data/iid")
    parser.add_argument("--output", type=str, default="output/eval/wasserstein.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.datasets), Path(args.baseline), Path(args.output))
