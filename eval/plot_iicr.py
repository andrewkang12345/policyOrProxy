#!/usr/bin/env python3
"""
Make a holistic IICR bar chart for Global/Hier/Grover across experiments.

- Scans JSONs like:
    output/eval/global/experiment1.json
    output/eval/hier/experiment1.json
    output/eval/grover/experiment1.json
  (You can point the script at any base directory; it will glob subdirs.)

- X axis: experiment number (e.g., 1, 2, 3, ...)
- For each experiment: adjacent bars for [global, hier, grover] IICR.

Usage:
  python policyOrProxy/eval/plot_iicr.py \
    --base output/eval \
    --out output/eval/iicr_overview.png

Optional:
  --models global hier grover
  --title "IICR by Model and Experiment"

Notes:
- Missing files/models for an experiment are skipped (with a warning).
- Also writes a CSV summary next to the image for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


EXPR_RE = re.compile(r"experiment(\d+)\.json$", re.IGNORECASE)

# Default subdirectories for each model under --base
DEFAULT_MODEL_DIRS = {
    "global": "global",
    "hier": "hier",
    "grover": "grover",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IICR across experiments for multiple models.")
    p.add_argument("--base", type=str, default="output/eval",
                   help="Base directory containing model subfolders (global/hier/grover).")
    p.add_argument("--out", type=str, default="output/eval/iicr_overview.png",
                   help="Output path for the PNG chart.")
    p.add_argument("--models", nargs="+", default=["global", "hier", "grover"],
                   help="Which models to include (subdirs under --base).")
    p.add_argument("--title", type=str, default="IICR by Model and Experiment")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def find_experiment_files(base: Path, model: str) -> Dict[int, Path]:
    """
    Return mapping: experiment_number -> json_path for a specific model.
    Looks under base/<model>/experiment*.json
    """
    model_dir = base / DEFAULT_MODEL_DIRS.get(model, model)
    out: Dict[int, Path] = {}
    if not model_dir.exists():
        print(f"[WARN] Missing model dir: {model_dir}")
        return out
    for p in model_dir.glob("experiment*.json"):
        m = EXPR_RE.search(p.name)
        if not m:
            continue
        num = int(m.group(1))
        out[num] = p
    return out


def load_iicr(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        metrics = data.get("metrics", {})
        iicr = metrics.get("IICR", None)
        if iicr is None:
            print(f"[WARN] No IICR in {path}")
            return None
        return float(iicr)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def collect_all(base: Path, models: List[str]) -> Tuple[List[int], Dict[str, Dict[int, float]]]:
    """
    Returns:
      - sorted list of experiment numbers
      - per_model -> {exp_num: iicr}
    Only includes experiments where at least one model has data.
    """
    per_model_files = {m: find_experiment_files(base, m) for m in models}
    all_exps = set()
    for m, mp in per_model_files.items():
        all_exps.update(mp.keys())
    exp_nums = sorted(all_exps)
    per_model_iicr: Dict[str, Dict[int, float]] = {m: {} for m in models}

    for m in models:
        for exp in exp_nums:
            p = per_model_files[m].get(exp)
            if not p:
                continue
            iicr = load_iicr(p)
            if iicr is not None:
                per_model_iicr[m][exp] = iicr

    return exp_nums, per_model_iicr


def plot_bars(exp_nums: List[int], per_model_iicr: Dict[str, Dict[int, float]],
              models: List[str], out_path: Path, title: str, dpi: int) -> None:
    if not exp_nums:
        raise SystemExit("No experiments found to plot.")

    n_models = len(models)
    x = range(len(exp_nums))

    width = 0.8 / max(n_models, 1)  # total cluster width ~0.8
    offsets = [(-0.4 + width/2) + i * width for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(exp_nums)), 4.5), dpi=dpi)

    # Bars per model
    for i, m in enumerate(models):
        vals = [per_model_iicr[m].get(exp, float("nan")) for exp in exp_nums]
        # Matplotlib will skip NaNs automatically (leaves holes)
        ax.bar([xi + offsets[i] for xi in x], vals, width=width, label=m.capitalize(), edgecolor="black")

    ax.set_title(title)
    ax.set_xlabel("Config ID")
    ax.set_ylabel("IICR (intra / inter-centroid)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(e) for e in exp_nums])

    ax.legend(frameon=False, ncol=min(3, len(models)))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[OK] Saved plot -> {out_path}")


def write_csv(exp_nums: List[int], per_model_iicr: Dict[str, Dict[int, float]],
              models: List[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    header = ["experiment"] + models
    lines.append(",".join(header))
    for exp in exp_nums:
        row = [str(exp)]
        for m in models:
            v = per_model_iicr[m].get(exp)
            row.append("" if v is None else f"{v:.6f}")
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote CSV -> {csv_path}")


def main():
    args = parse_args()
    base = Path(args.base)
    out_img = Path(args.out)
    out_csv = out_img.with_suffix(".csv")

    models = [m.lower() for m in args.models]
    exp_nums, per_model_iicr = collect_all(base, models)
    plot_bars(exp_nums, per_model_iicr, models, out_img, args.title, args.dpi)
    write_csv(exp_nums, per_model_iicr, models, out_csv)


if __name__ == "__main__":
    main()
