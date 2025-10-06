"""Duplicate dataset into multiple policy-labelled variants."""
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

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def main(source: Path, target: Path, variants: int) -> None:
    configure_logging()
    source = source.expanduser()
    target = target.expanduser()
    indexer = EpisodeIndexer.load(source)
    for variant in range(variants):
        variant_name = f"policy_{variant}"
        variant_root = target / variant_name
        variant_root.mkdir(parents=True, exist_ok=True)
        new_indexer = EpisodeIndexer(root=variant_root)
        for record in indexer.entries:
            source_path = source / record.path
            data = np.load(source_path, allow_pickle=False)
            payload = {key: data[key] for key in data.files}
            payload["policy_id"] = np.array(f"{record.policy_id or 'policy'}_{variant}", dtype=object)
            variant_split_dir = variant_root / Path(record.path).parent
            variant_split_dir.mkdir(parents=True, exist_ok=True)
            target_path = variant_root / record.path
            np.savez_compressed(target_path, **payload)
            new_indexer.add_episode(
                record.split,
                target_path,
                length=record.length,
                policy_id=f"{record.policy_id or 'policy'}_{variant}",
                meta=record.meta,
            )
        new_indexer.save(variant_root / "index.json")
        baseline_stats = source / "baseline_stats.npz"
        if baseline_stats.exists():
            stats = np.load(baseline_stats, allow_pickle=False)
            np.savez_compressed(
                variant_root / "baseline_stats.npz",
                **{key: stats[key] for key in stats.files},
            )
        LOGGER.info("Created variant %s at %s", variant_name, variant_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate dataset into multiple policy-labelled variants")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--variants", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.source), Path(args.target), args.variants)
