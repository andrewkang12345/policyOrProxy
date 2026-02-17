#!/usr/bin/env python3
"""
Plot training loss curves (Reconstruction+beta*KL) for Global and Hier CVAEs.

- Reads metrics CSVs like:
    output/runs/global_cvae_experiment1/metrics.csv

- Filters to split == "train"
- Plots a single figure with 6 curves total:
    Global: experiments 1, 4, 5
    Hier:   experiments 1, 4, 5
- Y-axis label: "Reconstruction+beta*KL"

Usage:
  python policyOrProxy/eval/plot_training_loss.py \
    --runs-base output/runs \
    --out output/eval/training_loss_overview.png

Options:
  --models global hier
  --experiments 1 4 5
  --title "Training Loss (Reconstruction+beta*KL)"
  --dpi 180
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training loss curves for Global/Hier CVAEs.")
    p.add_argument("--runs-base", type=str, default="output/runs",
                   help="Base directory containing *cvae_experiment*/metrics.csv subfolders.")
    p.add_argument("--out", type=str, default="output/eval/training_loss_overview.png",
                   help="Output path for the PNG chart.")
    p.add_argument("--models", nargs="+", default=["global", "hier"],
                   help="Which model families to include (supports 'global', 'hier').")
    p.add_argument("--experiments", nargs="+", type=int, default=[1, 4, 5],
                   help="Experiment numbers to include per model.")
    p.add_argument("--title", type=str, default="Training Loss (Reconstruction+beta*KL)")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def path_for(runs_base: Path, model: str, exp_num: int) -> Path:
    """
    Build the metrics.csv path for a given model/experiment.
    """
    if model not in {"global", "hier"}:
        raise ValueError(f"Unsupported model type: {model}")
    folder = f"{model}_cvae_experiment{exp_num}"
    return runs_base / folder / "metrics.csv"


def read_train_curve(csv_path: Path) -> Tuple[List[int], List[float]]:
    """
    Read (epoch, loss) for rows where split == 'train'.
    CSV header expected: epoch,split,loss,recon,kl,beta_example,lr
    """
    epochs: List[int] = []
    losses: List[float] = []
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if (row.get("split") or "").strip().lower() != "train":
                continue
            try:
                ep = int(float(row["epoch"]))
                lo = float(row["loss"])
            except Exception:
                continue
            epochs.append(ep)
            losses.append(lo)
    # De-dupe by epoch keeping last occurrence (in case of multiple rows per epoch)
    collapsed: Dict[int, float] = {}
    for ep, lo in zip(epochs, losses):
        collapsed[ep] = lo
    seps = sorted(collapsed.keys())
    return seps, [collapsed[e] for e in seps]


def collect_curves(runs_base: Path, models: List[str], experiments: List[int]) -> Dict[str, Dict[int, Tuple[List[int], List[float]]]]:
    """
    per_model -> {exp_num: (epochs, losses)}
    """
    per_model: Dict[str, Dict[int, Tuple[List[int], List[float]]]] = {m: {} for m in models}
    for m in models:
        for e in experiments:
            csv_path = path_for(runs_base, m, e)
            if not csv_path.exists():
                print(f"[WARN] Missing metrics: {csv_path}")
                continue
            ep, lo = read_train_curve(csv_path)
            if not ep:
                print(f"[WARN] No train rows in: {csv_path}")
                continue
            per_model[m][e] = (ep, lo)
    return per_model


def plot_curves(per_model: Dict[str, Dict[int, Tuple[List[int], List[float]]]],
                out_path: Path, title: str, dpi: int) -> None:
    # Determine common x-range for nicer comparisons; still plot actual epochs.
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=dpi)

    # Style: different linestyles per model, distinct labels per exp
    model_style = {
        "global": {"linestyle": "-",  "label_prefix": "Global"},
        "hier":   {"linestyle": "--", "label_prefix": "Hier"},
    }

    plotted_any = False
    for model, exps in per_model.items():
        style = model_style.get(model, {"linestyle": "-", "label_prefix": model.capitalize()})
        for exp_num in sorted(exps.keys()):
            epochs, losses = exps[exp_num]
            lbl = f"{style['label_prefix']} exp{exp_num}"
            ax.plot(epochs, losses, linestyle=style["linestyle"], linewidth=2.0, label=lbl)
            plotted_any = True

    if not plotted_any:
        raise SystemExit("No curves found to plot (check paths).")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction+beta*KL")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[OK] Saved plot -> {out_path}")


def main():
    args = parse_args()
    runs_base = Path(args.runs_base)
    models = [m.lower() for m in args.models]
    experiments = args.experiments

    per_model = collect_curves(runs_base, models, experiments)
    plot_curves(per_model, Path(args.out), args.title, args.dpi)


if __name__ == "__main__":
    main()
