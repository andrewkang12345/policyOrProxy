#!/usr/bin/env python3
"""
verify_sampling_probs.py

Diagnose why numpy.random.Generator.choice complains:
  ValueError: Probabilities do not sum to 1

This script:
- Loads IID dataset from output/data/<policy>/iid using EpisodeIndexer
- For each requested split, scans windows for non-finite values (NaN/Inf) at t=0 in (x,y)
- Computes score = mean_x(t=0) per candidate window
- Builds probabilities p = softmax(alpha * (scores - mean(scores)))
  using:
    (A) the original (bug-prone) softmax
    (B) a robust float64 softmax with non-finite guarding
- Prints diagnostics and tries rng.choice with each p to reproduce the crash

Usage examples:
  python policyOrProxy/debug/verify_sampling_probs.py \
    --root output/data/ego_policy1/iid --split train val test --alpha 0

  python policyOrProxy/debug/verify_sampling_probs.py \
    --root output/data/ego_policy1/iid --split train --shifts policyOrProxy/cfg/shifts.yaml

Notes:
- If you see any NaNs in start positions or scores, that is the primary cause.
- If scores are clean but p_sum drifts or choice fails only for the "original" softmax,
  you are likely hitting float32 exp/sum precision issues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Repo paths (match your other scripts)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify sampling probabilities p used in make_state_shift_manual.py")
    p.add_argument("--root", type=str, default="output/data/ego_policy1/iid", help="IID root like output/data/ego_policy1/iid (must contain index.json)")
    p.add_argument("--split", nargs="+", default=["train"], help="One or more splits to scan (default: train)")
    p.add_argument("--alpha", type=float, action="append", help="Alpha(s) to test (can repeat). If omitted and --shifts provided, uses shifts.yaml alphas.")
    p.add_argument("--shifts", type=str, help="Path to shifts.yaml to extract alpha sweep")
    p.add_argument("--max_bad", type=int, default=20, help="Max number of bad candidates to print (default 20)")
    p.add_argument("--max_episodes", type=int, default=0, help="If >0, only scan first N episodes per split (debug speed)")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for choice() tests")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def feature_mean_x(start_xy: np.ndarray) -> float:
    # start_xy: [teams, agents, 2]
    return float(np.mean(start_xy[..., 0]))


def original_stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    A replica of the bug-prone style softmax:
    - works in input dtype unless you promote
    - doesn't handle NaNs robustly
    """
    logits = np.asarray(logits)
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    s = float(np.sum(ex))
    if s <= 0:
        return np.ones_like(logits, dtype=np.float64) / float(len(logits))
    return (ex / s).astype(np.float64)


def robust_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Float64, non-finite safe softmax. Forces sum(p)=1.
    """
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    M = x.size
    if M == 0:
        raise ValueError("robust_softmax: empty logits")

    # All equal => exact uniform
    if np.all(x == x[0]):
        p = np.full((M,), 1.0 / float(M), dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))
        return p

    finite = np.isfinite(x)
    if not finite.any():
        p = np.full((M,), 1.0 / float(M), dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))
        return p
    if not finite.all():
        x = np.where(finite, x, -np.inf)

    m = np.max(x)
    y = np.clip(x - m, -745.0, 0.0)  # avoid overflow/underflow extremes
    ex = np.exp(y)
    s = ex.sum(dtype=np.float64)
    if not np.isfinite(s) or s <= 0.0:
        p = np.full((M,), 1.0 / float(M), dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))
        return p

    p = ex / s
    p = np.maximum(p, 0.0)
    ps = p.sum(dtype=np.float64)
    if not np.isfinite(ps) or ps <= 0.0:
        p = np.full((M,), 1.0 / float(M), dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))
        return p

    p /= ps
    p[-1] += 1.0 - float(p.sum(dtype=np.float64))
    if p[-1] < 0.0:
        p = np.clip(p, 0.0, None)
        p /= p.sum(dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))
    return p


def print_p_stats(tag: str, p: np.ndarray) -> None:
    p = np.asarray(p)
    s = float(np.sum(p)) if p.size else float("nan")
    finite = bool(np.isfinite(p).all()) if p.size else False
    n_nan = int(np.isnan(p).sum()) if p.size else 0
    n_inf = int(np.isinf(p).sum()) if p.size else 0
    pmin = float(np.min(p)) if p.size else float("nan")
    pmax = float(np.max(p)) if p.size else float("nan")
    n_neg = int((p < 0).sum()) if p.size else 0
    print(f"  [{tag}] len={p.size} sum={s:.15f} min={pmin:.3e} max={pmax:.3e} finite={finite} nan={n_nan} inf={n_inf} neg={n_neg}")


def try_choice(tag: str, rng: np.random.Generator, p: np.ndarray, k: int = 10) -> None:
    try:
        _ = rng.choice(len(p), size=min(k, len(p)), replace=True, p=p)
        print(f"  [{tag}] rng.choice OK")
    except Exception as e:
        print(f"  [{tag}] rng.choice FAILED: {type(e).__name__}: {e}")


def scan_split(root: Path, indexer: EpisodeIndexer, split: str, max_bad: int, max_episodes: int) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """
    Returns:
      scores: [M]
      bad: list of (episode_relpath, window_idx, reason)
    """
    records = list(indexer.iter_split(split))
    if max_episodes and max_episodes > 0:
        records = records[:max_episodes]

    scores: List[float] = []
    bad: List[Tuple[str, int, str]] = []

    for rec in records:
        ep_path = root / rec.path
        with np.load(ep_path, allow_pickle=False) as data:
            W = np.asarray(data["windows"], dtype=np.float32)  # [N,T,teams,agents,F]
        if W.ndim != 5:
            bad.append((rec.path, -1, f"windows.ndim={W.ndim} shape={W.shape}"))
            continue

        N = int(W.shape[0])
        starts_xy = W[:, 0, :, :, :2]  # [N,teams,agents,2]

        # detect any non-finite in start positions
        finite_mask = np.isfinite(starts_xy).all(axis=(1, 2, 3))
        for i in range(N):
            if not finite_mask[i]:
                if len(bad) < max_bad:
                    # pinpoint if possible
                    vals = starts_xy[i].reshape(-1, 2)
                    reason = f"non-finite start_xy: nan={np.isnan(vals).sum()} inf={np.isinf(vals).sum()}"
                    bad.append((rec.path, i, reason))
                # still compute score? no—skip candidate entirely
                continue
            scores.append(feature_mean_x(starts_xy[i]))

    scores_arr = np.asarray(scores, dtype=np.float64)
    # detect non-finite scores (can still happen if feature logic changes)
    if not np.isfinite(scores_arr).all():
        idxs = np.where(~np.isfinite(scores_arr))[0]
        for j in idxs[:max_bad]:
            bad.append(("<scores_array>", int(j), "non-finite score"))
        scores_arr = scores_arr[np.isfinite(scores_arr)]

    return scores_arr, bad


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    if not (root / "index.json").exists():
        raise SystemExit(f"Missing index.json under {root} (root must be the IID dataset dir).")

    indexer = EpisodeIndexer.load(root)

    # Determine alphas
    alphas: List[float] = []
    if args.alpha:
        alphas.extend([float(a) for a in args.alpha])
    if args.shifts:
        shifts = load_yaml(Path(args.shifts).expanduser())
        for shift_name, spec in (shifts.get("shifts", {}) or {}).items():
            if "alpha" in spec:
                alphas.append(float(spec["alpha"]))
    if not alphas:
        alphas = [0.0]

    # De-dupe and sort
    alphas = sorted(set(alphas))

    rng = np.random.default_rng(int(args.seed))

    for split in args.split:
        print("=" * 90)
        print(f"SPLIT: {split}")
        scores, bad = scan_split(root, indexer, split, max_bad=int(args.max_bad), max_episodes=int(args.max_episodes))

        print(f"Candidates scanned (after skipping bad start_xy): M={scores.size}")
        if bad:
            print(f"Found {len(bad)} bad candidates/episodes (showing up to {args.max_bad}):")
            for ep, i, reason in bad[: args.max_bad]:
                print(f"  - {ep}  window={i}  reason={reason}")
        else:
            print("No non-finite start positions detected in this split.")

        if scores.size == 0:
            print("No valid candidates left; cannot build probabilities.")
            continue

        # score stats
        print(f"Score stats: mean={scores.mean():.6f} std={scores.std():.6f} min={scores.min():.6f} max={scores.max():.6f} finite={np.isfinite(scores).all()}")

        scores_centered = scores - float(scores.mean())

        for alpha in alphas:
            print("-" * 90)
            print(f"alpha={alpha:g}")

            logits64 = np.asarray(alpha * scores_centered, dtype=np.float64)

            # Build p using original vs robust
            p_orig = original_stable_softmax(logits64.astype(np.float32))  # emulate float32 path
            p_rob = robust_softmax(logits64)

            print_p_stats("orig_softmax", p_orig)
            try_choice("orig_softmax", rng, p_orig)

            print_p_stats("robust_softmax", p_rob)
            try_choice("robust_softmax", rng, p_rob)

            # Extra: show how far from 1 we are
            orig_sum = float(np.sum(p_orig))
            rob_sum = float(np.sum(p_rob))
            print(f"  sum deviation: orig={orig_sum - 1.0:+.3e} robust={rob_sum - 1.0:+.3e}")

            # Extra: check if any NaNs in intermediate computations (for debugging)
            if not np.isfinite(logits64).all():
                print(f"  WARNING: non-finite logits detected: nan={np.isnan(logits64).sum()} inf={np.isinf(logits64).sum()}")

    print("=" * 90)
    print("Done. If orig_softmax fails but robust_softmax passes, your failure is numeric/guarding.")
    print("If both fail (or scores/start_xy report NaNs), your dataset contains non-finite values that must be fixed upstream.")


if __name__ == "__main__":
    main()
