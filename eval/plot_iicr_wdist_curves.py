#!/usr/bin/env python3
"""
Plot IICR vs. Wasserstein distance for multiple configs and architectures.

Expected files:
  output/eval/<model>/experiment<cfg>-<sev>.json
Where:
  <model> in {global, hier, grover}
  <cfg>   is a configuration id (e.g., 3, 6)
  <sev>   in {1..5} mapping to WD: tiny..extreme

X-axis (WD):
  1 -> 1.85 (tiny)
  2 -> 2.07 (mild)
  3 -> 2.45 (moderate)
  4 -> 2.63 (strong)
  5 -> 2.90 (extreme)

Styling:
  - Color groups by configuration id (same color across models).
  - Line style groups by architecture (global/hier/grover).

Usage:
  python policyOrProxy/eval/plot_iicr_wdist_curves.py \
    --base output/eval \
    --out output/eval/iicr_wdist_curves.png \
    --configs 3 6 \
    --models global hier grover \
    --title "IICR vs. Wasserstein Distance"

Notes:
- Missing JSONs are skipped gracefully.
- Also writes a CSV next to the image with the plotted data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

# Severity index -> (label, wasserstein distance)
SEV_TO_WD = {
    1: ("tiny",     1.85),
    2: ("mild",     2.07),
    3: ("moderate", 2.45),
    4: ("strong",   2.63),
    5: ("extreme",  2.90),
}

DEFAULT_MODELS = ["global", "hier", "grover"]

# Fixed linestyles per architecture
MODEL_STYLE = {
    "global": ("solid", 2.2),
    "hier":   ("dashed", 2.2),
    "grover": ("dotted", 2.4),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot IICR vs. Wasserstein distance across configs and models.")
    p.add_argument("--base", type=str, default="output/eval",
                   help="Base directory containing model subfolders (global/hier/grover).")
    p.add_argument("--out", type=str, default="output/eval/iicr_wdist_curves.png",
                   help="Output path for the PNG chart.")
    p.add_argument("--configs", nargs="+", type=int, default=[3, 6],
                   help="Configuration IDs to include (e.g., 3 6).")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="Models to include (any subset of global hier grover).")
    p.add_argument("--title", type=str, default="IICR vs. Wasserstein Distance")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def load_iicr(json_path: Path) -> Optional[float]:
    try:
        with json_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        m = data.get("metrics", {})
        iicr = m.get("IICR", None)
        if iicr is None:
            print(f"[WARN] No IICR in {json_path}")
            return None
        return float(iicr)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}")
        return None


def collect_data(base: Path, models: List[str], configs: List[int]) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Returns:
      per_cfg[cfg][model] -> list of (wd, iicr) points sorted by wd
    """
    per_cfg: Dict[int, Dict[str, List[Tuple[float, float]]]] = {cfg: {m: [] for m in models} for cfg in configs}
    for cfg in configs:
        for model in models:
            model_dir = base / model
            for sev in sorted(SEV_TO_WD.keys()):
                json_path = model_dir / f"experiment{cfg}-{sev}.json"
                if not json_path.exists():
                    # silently skip missing; may not have been run
                    continue
                iicr = load_iicr(json_path)
                if iicr is None:
                    continue
                wd = SEV_TO_WD[sev][1]
                per_cfg[cfg][model].append((wd, iicr))
            # sort by wd (x-axis)
            per_cfg[cfg][model].sort(key=lambda t: t[0])
    return per_cfg


def distinct_colors(n: int) -> List[str]:
    # Get a distinct palette from Matplotlib (tab20 cycling as needed)
    import itertools
    palette = plt.get_cmap("tab20").colors
    return list(itertools.islice((palette[i % len(palette)] for i in range(n)), n))


def plot_curves(per_cfg: Dict[int, Dict[str, List[Tuple[float, float]]]],
                models: List[str], out_path: Path, title: str, dpi: int) -> None:
    cfg_ids = sorted(per_cfg.keys())
    colors = {cfg: c for cfg, c in zip(cfg_ids, distinct_colors(len(cfg_ids)))}

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=dpi)

    plotted_any = False
    for cfg in cfg_ids:
        color = colors[cfg]
        for model in models:
            points = per_cfg[cfg][model]
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            linestyle, lw = MODEL_STYLE.get(model, ("solid", 2.0))
            lbl = f"{model.capitalize()} (cfg {cfg})"
            ax.plot(xs, ys, linestyle=linestyle, linewidth=lw, color=color, marker="o", label=lbl)
            plotted_any = True

    if not plotted_any:
        raise SystemExit("No data to plot. Are the JSON files present?")

    # X ticks at the WD anchors with severity labels
    xticks = [SEV_TO_WD[s][1] for s in sorted(SEV_TO_WD)]
    xticklabels = [f"{SEV_TO_WD[s][0]}\n{SEV_TO_WD[s][1]:.2f}" for s in sorted(SEV_TO_WD)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_title(title)
    ax.set_xlabel("Wasserstein distance of state distribution shift")
    ax.set_ylabel("IICR (intra / inter-centroid)")
    ax.grid(True, linestyle="--", alpha=0.4)

    # One legend with combined entries (color=cfg, linestyle=model)
    ax.legend(frameon=False, ncol=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[OK] Saved plot -> {out_path}")

    # Also dump a CSV with the plotted points
    csv_path = out_path.with_suffix(".csv")
    lines = ["cfg,model,wd,severity,iicr"]
    inv_map = {v[1]: k for k, v in SEV_TO_WD.items()}
    for cfg in cfg_ids:
        for model in models:
            for wd, iicr in per_cfg[cfg][model]:
                sev = inv_map.get(wd, None)
                sev_name = SEV_TO_WD[sev][0] if sev in SEV_TO_WD else ""
                lines.append(f"{cfg},{model},{wd:.2f},{sev_name},{iicr:.6f}")
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote CSV -> {csv_path}")


def main():
    args = parse_args()
    base = Path(args.base)
    models = [m.lower() for m in args.models]
    per_cfg = collect_data(base, models, args.configs)
    plot_curves(per_cfg, models, Path(args.out), args.title, args.dpi)


if __name__ == "__main__":
    main()
