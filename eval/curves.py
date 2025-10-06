"""Plot metric curves across shift severities."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

METRICS = ["ade", "fde", "smoothness", "collisions", "invariance"]


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_metric(metric: str, order, global_metrics, hier_metrics, output: Path) -> None:
    x = list(range(len(order)))
    g_vals = [global_metrics.get(key, {}).get(metric, float("nan")) for key in order]
    h_vals = [hier_metrics.get(key, {}).get(metric, float("nan")) for key in order]
    plt.figure(figsize=(6, 4))
    plt.plot(x, g_vals, marker="o", label="Global")
    plt.plot(x, h_vals, marker="s", label="Hierarchical")
    plt.xticks(x, order)
    plt.ylabel(metric)
    plt.xlabel("Shift severity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    LOGGER.info("Saved curve for %s to %s", metric, output)


def main(global_path: Path, hier_path: Path, output_dir: Path) -> None:
    configure_logging()
    global_metrics = load_metrics(global_path)
    hier_metrics = load_metrics(hier_path)
    order = sorted(set(global_metrics.keys()) | set(hier_metrics.keys()))
    for metric in METRICS:
        plot_metric(metric, order, global_metrics, hier_metrics, output_dir / f"{metric}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metric curves")
    parser.add_argument("--global", dest="global_path", type=str, required=True)
    parser.add_argument("--hier", dest="hier_path", type=str, required=True)
    parser.add_argument("--output", dest="output", type=str, default="output/eval/curves")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.global_path), Path(args.hier_path), Path(args.output))
