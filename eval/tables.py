"""Render comparison tables for global vs hierarchical models."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)


REQUIRED_KEYS = ["ade", "fde", "smoothness", "collisions", "invariance"]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    return metrics


def render_table(global_metrics: Dict[str, Dict[str, float]], hier_metrics: Dict[str, Dict[str, float]], order: List[str]) -> str:
    header = ["Shift"]
    for key in REQUIRED_KEYS:
        header.extend([f"Global {key}", f"Hier {key}"])
    widths = [max(len(h), 8) for h in header]
    rows = []
    header_row = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    rows.append(header_row)
    rows.append("-+-".join("-" * w for w in widths))
    for shift in order:
        row = [shift.ljust(widths[0])]
        for idx, key in enumerate(REQUIRED_KEYS):
            g_val = global_metrics.get(shift, {}).get(key, float("nan"))
            h_val = hier_metrics.get(shift, {}).get(key, float("nan"))
            row.append(f"{g_val:.3f}".ljust(widths[2 * idx + 1]))
            row.append(f"{h_val:.3f}".ljust(widths[2 * idx + 2]))
        rows.append(" | ".join(row))
    return "\n".join(rows)


def main(global_path: Path, hier_path: Path) -> None:
    configure_logging()
    global_metrics = load_metrics(global_path)
    hier_metrics = load_metrics(hier_path)
    order = sorted(set(global_metrics.keys()) | set(hier_metrics.keys()))
    table = render_table(global_metrics, hier_metrics, order)
    print(table)
    LOGGER.info("Comparison table generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render evaluation tables")
    parser.add_argument("--global", dest="global_path", type=str, required=True)
    parser.add_argument("--hier", dest="hier_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.global_path), Path(args.hier_path))
