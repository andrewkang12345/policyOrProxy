#!/usr/bin/env python3
"""
plot_position_heatmaps_multi.py

Plot heatmaps of agent (x,y) positions for:
  - IID baseline (Original)
  - ALL manual shift folders detected for that policy (from shifts.yaml / alpha_sweep.csv / dir scan)

Improvements vs older versions:
- Detects current manual shift naming (e.g., right_bias_alpha_0p0, ...)
- Better visibility via robust vmax (percentile) and optional log-like intensity scaling.

Intensity/normalization options:
- --intensity linear|log1p:
    linear  : display raw counts
    log1p   : display log(1 + count) (recommended for visibility)
- --normalize global|per_panel:
    global   : same color scale across all panels (best for comparison)
    per_panel: each panel gets its own vmax (best for visibility, worse for cross-panel comparison)
- --vmax_percentile:
    vmax is set to the chosen percentile of displayed values (ignores extreme outliers)

Usage:
  python policyOrProxy/viz/plot_position_heatmaps_multi.py \
    --base_iid output/data/ego_policy1/iid \
    --base_ood output/data/ood_manual/ego_policy1 \
    --split test --num 10 --start 0 --use_index \
    --out output/viz/ego_policy1_heatmaps.png \
    --intensity log1p --normalize global --vmax_percentile 99.5 --colorbar
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None  # type: ignore


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot heatmaps for IID + detected manual shifts (robust color scaling).")
    p.add_argument("--base_iid", type=str, default="output/data/ego_policy1/iid")
    p.add_argument("--base_ood", type=str, default="output/data/ood_manual/ego_policy1")
    p.add_argument("--split", type=str, default="test", help="train|val|test")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", type=str, default="output/viz/ego_policy1_position_heatmaps_multi.png")
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--cmap", type=str, default="inferno")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--use_index", action="store_true", help="Use index.json if present (recommended).")
    p.add_argument("--pattern", type=str, default="episode_{:05d}.npz")

    p.add_argument("--shifts_cfg", type=str, default="", help="Optional shifts.yaml to enforce shift ordering/alphas.")

    p.add_argument(
        "--extract",
        type=str,
        default="last",
        choices=["last", "all", "mean"],
        help=(
            "How to extract positions from 'windows': "
            "'last' uses last frame per window; "
            "'all' uses all frames in history; "
            "'mean' uses mean over history per window."
        ),
    )

    # Visibility controls
    p.add_argument(
        "--intensity",
        type=str,
        default="log1p",
        choices=["linear", "log1p"],
        help="Display transform applied to histogram counts. log1p recommended.",
    )
    p.add_argument(
        "--normalize",
        type=str,
        default="global",
        choices=["global", "per_panel"],
        help="Color scaling: global shared vmax or per-panel vmax.",
    )
    p.add_argument(
        "--vmax_percentile",
        type=float,
        default=99.5,
        help="Percentile used to set vmax (robust against outliers).",
    )
    p.add_argument(
        "--colorbar",
        action="store_true",
        help="Add a shared colorbar (only meaningful with --normalize global).",
    )

    p.add_argument(
        "--max_panels",
        type=int,
        default=0,
        help="Optional cap on number of plotted panels (0 = no cap).",
    )
    return p.parse_args()


# -------------------------
# Index-based episode discovery
# -------------------------

def try_load_index_paths(condition_dir: Path, split: str) -> Optional[List[Path]]:
    idx_path = condition_dir / "index.json"
    if not idx_path.exists():
        return None
    try:
        from policyOrProxy.core.dataset.indexer import EpisodeIndexer  # type: ignore
        indexer = EpisodeIndexer.load(condition_dir)
        paths: List[Path] = []
        for rec in indexer.entries:
            if getattr(rec, "split", None) != split:
                continue
            paths.append(condition_dir / rec.path)
        return sorted(paths, key=lambda p: p.as_posix())
    except Exception:
        return None


def iter_episode_paths(
    condition_dir: Path,
    split: str,
    pattern: str,
    start: int,
    num: int,
    use_index: bool,
) -> List[Path]:
    if use_index:
        indexed = try_load_index_paths(condition_dir, split)
        if indexed:
            return indexed[start:start + num]
    subdir = condition_dir / split
    return [subdir / pattern.format(i) for i in range(start, start + num)]


# -------------------------
# Shift detection
# -------------------------

def _load_shifts_from_yaml(shifts_cfg: Path) -> Optional[List[Tuple[str, Optional[float]]]]:
    if not shifts_cfg.exists() or yaml is None:
        return None
    try:
        with shifts_cfg.open("r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp)
        shifts = cfg.get("shifts", {})
        if not isinstance(shifts, dict) or not shifts:
            return None
        out: List[Tuple[str, Optional[float]]] = []
        for name, spec in shifts.items():
            alpha = None
            if isinstance(spec, dict) and "alpha" in spec:
                try:
                    alpha = float(spec["alpha"])
                except Exception:
                    alpha = None
            out.append((str(name), alpha))
        return out
    except Exception:
        return None


def _load_shifts_from_alpha_csv(alpha_csv: Path) -> Optional[List[Tuple[str, Optional[float]]]]:
    if not alpha_csv.exists():
        return None
    rows: List[Tuple[str, float]] = []
    try:
        with alpha_csv.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                shift = (r.get("shift") or "").strip()
                alpha_s = (r.get("alpha") or "").strip()
                if not shift:
                    continue
                try:
                    alpha = float(alpha_s)
                except Exception:
                    continue
                rows.append((shift, alpha))
        if not rows:
            return None
        rows.sort(key=lambda x: x[1])
        return [(s, a) for (s, a) in rows]
    except Exception:
        return None


def _scan_shift_dirs(base_ood: Path) -> List[Tuple[str, Optional[float]]]:
    if not base_ood.exists():
        return []
    shifts: List[str] = []
    for p in base_ood.iterdir():
        if not p.is_dir():
            continue
        if (p / "index.json").exists() or (p / "train").exists() or (p / "val").exists() or (p / "test").exists():
            shifts.append(p.name)
    shifts = sorted(set(shifts))
    return [(s, None) for s in shifts]


def detect_shifts(base_ood: Path, shifts_cfg: Optional[Path]) -> List[Tuple[str, Optional[float]]]:
    if shifts_cfg is not None and shifts_cfg.as_posix().strip():
        y = _load_shifts_from_yaml(shifts_cfg)
        if y:
            return y
    c = _load_shifts_from_alpha_csv(base_ood / "alpha_sweep.csv")
    if c:
        return c
    return _scan_shift_dirs(base_ood)


# -------------------------
# Data extraction
# -------------------------

def load_xy_from_npz(npz_path: Path, extract: str = "last") -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "windows" in data:
            W = np.asarray(data["windows"], dtype=np.float32)  # (L,T,teams,agents,F)
            if W.ndim != 5 or W.shape[-1] < 2:
                raise ValueError(f"'windows' has unexpected shape {W.shape} in {npz_path}")
            if extract == "last":
                xy = W[:, -1, :, :, :2].reshape(-1, 2)
            elif extract == "mean":
                xy = W[:, :, :, :, :2].mean(axis=1).reshape(-1, 2)
            else:  # all
                xy = W[..., :2].reshape(-1, 2)

        elif "positions" in data:
            P = np.asarray(data["positions"], dtype=np.float32)
            if P.ndim < 2 or P.shape[-1] < 2:
                raise ValueError(f"'positions' has unexpected shape {P.shape} in {npz_path}")
            xy = P.reshape(-1, P.shape[-1])[..., :2]

        else:
            best_k, best_size = None, -1
            for k in data.files:
                v = data[k]
                if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[-1] >= 2 and v.size > best_size:
                    best_k, best_size = k, v.size
            if best_k is None:
                raise ValueError(f"No suitable array found in {npz_path}; keys={list(data.files)}")
            arr = np.asarray(data[best_k], dtype=np.float32)
            xy = arr.reshape(-1, arr.shape[-1])[..., :2]

    m = np.isfinite(xy).all(axis=1)
    return xy[m]


# -------------------------
# Bounds + histogram
# -------------------------

def compute_global_bounds(cond_paths: Sequence[Tuple[str, List[Path]]], extract: str) -> Tuple[float, float, float, float]:
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    have_any = False

    for _label, paths in cond_paths:
        for p in paths:
            if not p.exists():
                continue
            try:
                xy = load_xy_from_npz(p, extract=extract)
            except Exception:
                continue
            if xy.size == 0:
                continue
            have_any = True
            x_min = min(x_min, float(np.min(xy[:, 0])))
            x_max = max(x_max, float(np.max(xy[:, 0])))
            y_min = min(y_min, float(np.min(xy[:, 1])))
            y_max = max(y_max, float(np.max(xy[:, 1])))

    if not have_any:
        raise SystemExit("No valid data found across any NPZ files. Check paths/split/num/start.")

    pad_x = 0.02 * (x_max - x_min + 1e-6)
    pad_y = 0.02 * (y_max - y_min + 1e-6)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def accumulate_hist(paths: Sequence[Path], bins: int, bounds: Tuple[float, float, float, float], extract: str) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    H = np.zeros((bins, bins), dtype=np.float64)

    for p in paths:
        if not p.exists():
            continue
        try:
            xy = load_xy_from_npz(p, extract=extract)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue
        if xy.size == 0:
            continue

        h, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1],
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        H += h.T
    return H


def apply_intensity(H: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return H.astype(np.float64)
    if mode == "log1p":
        return np.log1p(H.astype(np.float64))
    raise ValueError(f"Unknown intensity mode: {mode}")


def robust_vmax(values: np.ndarray, percentile: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 1.0
    # For robustness, ignore exact zeros if possible (common with sparse histograms)
    nz = v[v > 0]
    use = nz if nz.size > 0 else v
    q = float(np.clip(percentile, 0.0, 100.0)) / 100.0
    vmax = float(np.quantile(use, q))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(use)) if use.size else 1.0
    return max(vmax, 1e-12)


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    base_iid = Path(args.base_iid)
    base_ood = Path(args.base_ood)

    shifts_cfg = Path(args.shifts_cfg) if args.shifts_cfg.strip() else None
    shift_list = detect_shifts(base_ood, shifts_cfg)

    existing_shifts: List[Tuple[str, Optional[float]]] = []
    for s, a in shift_list:
        if (base_ood / s).exists():
            existing_shifts.append((s, a))
        else:
            print(f"[WARN] shift listed but not found on disk: {base_ood / s}")

    if not existing_shifts:
        print("[WARN] No shifts found; will plot IID only.")
    else:
        print(f"[OK] Detected {len(existing_shifts)} shifts under {base_ood}")

    max_panels = int(args.max_panels)
    if max_panels > 0:
        keep = max(0, max_panels - 1)  # reserve 1 for IID
        existing_shifts = existing_shifts[:keep]

    # Conditions
    conds: List[Tuple[str, Path]] = [("Original (iid)", base_iid)]
    for s, a in existing_shifts:
        label = f"{s}\nα={a:g}" if a is not None else s
        conds.append((label, base_ood / s))

    # Episode paths
    cond_paths: List[Tuple[str, List[Path]]] = []
    for label, cond_dir in conds:
        eps = iter_episode_paths(cond_dir, args.split, args.pattern, args.start, args.num, args.use_index)
        cond_paths.append((label, eps))

    # Bounds
    bounds = compute_global_bounds(cond_paths, extract=args.extract)
    x_min, x_max, y_min, y_max = bounds
    extent = (x_min, x_max, y_min, y_max)

    # Heatmaps (raw counts) -> displayed maps (intensity transform)
    raw_maps: List[np.ndarray] = []
    disp_maps: List[np.ndarray] = []
    labels: List[str] = []

    for label, paths in cond_paths:
        H = accumulate_hist(paths, args.bins, bounds, extract=args.extract)
        raw_maps.append(H)
        disp_maps.append(apply_intensity(H, args.intensity))
        labels.append(label)

    # Determine color scaling
    if args.normalize == "global":
        all_vals = np.concatenate([m.reshape(-1) for m in disp_maps], axis=0)
        vmax_global = robust_vmax(all_vals, args.vmax_percentile)
        vmax_list = [vmax_global for _ in disp_maps]
    else:
        vmax_list = [robust_vmax(m.reshape(-1), args.vmax_percentile) for m in disp_maps]

    # Plot grid
    n = len(disp_maps)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=args.dpi)
    axes = np.atleast_1d(axes).ravel()

    first_im = None
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue

        im = ax.imshow(
            disp_maps[i],
            origin="lower",
            extent=extent,
            cmap=args.cmap,
            aspect="equal",
            vmin=0.0,
            vmax=float(vmax_list[i]),
            interpolation="nearest",
        )
        if first_im is None:
            first_im = im

        ax.set_title(labels[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        # Small panel annotation so you remember what you're seeing
        if args.intensity == "log1p":
            ax.text(0.01, 0.01, "log1p(count)", transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
        else:
            ax.text(0.01, 0.01, "count", transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

    policy_name = base_iid.parent.name if base_iid.name == "iid" else base_iid.name
    # fig.suptitle(
    #     f"Agent Position Heatmaps — {policy_name}  (split={args.split}, episodes={args.num}, extract={args.extract}, "
    #     f"intensity={args.intensity}, normalize={args.normalize}, vmax_p={args.vmax_percentile})",
    #     y=0.98,
    # )
    fig.suptitle(
        f"Agent Position Heatmaps",
        y=0.98,
    )    

    # Optional global colorbar
    if args.colorbar and args.normalize == "global" and first_im is not None:
        cbar = fig.colorbar(first_im, ax=axes.tolist(), shrink=0.9, pad=0.02)
        cbar.set_label("log1p(count)" if args.intensity == "log1p" else "count")

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[OK] Saved -> {out}")


if __name__ == "__main__":
    main()
