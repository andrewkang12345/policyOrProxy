#!/usr/bin/env python3
"""
make_state_shift_manual.py

Create OOD datasets by:
  (resample_weighted) Reweighting & resampling *start frames* from real IID windows,
then constructing windows by actually ROLLING OUT the arena/world for `window_len` steps
using the existing ego/opponent policies (no added noise).

Changes vs previous:
- Target divergence proxy uses the **mean position over the window** (time-mean x,y).
- Manual shifting biases the **starting frame** (t=0) of each window with softmax(alpha * score).
- For every sample, we **restart** the world at the biased start and roll forward `window_len` steps.
- **No artificial noise** is added; trajectories are from policies and dynamics only.
- Saved NPZ `policy_id` == original policy name (unchanged).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import sys

import numpy as np
import yaml

# Repo paths
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local deps
from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.metrics.metrics import wasserstein_distance_numpy
from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy, build_window_hash_policy
from policyOrProxy.core.policies.oppPolicy import build_hash_policy
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world  # needed for real rollouts

LOGGER = logging.getLogger(__name__)

# -----------------------
# Helpers / config loaders
# -----------------------

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)

def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

def find_ego_configs(explicit: List[str] | None) -> List[Path]:
    if explicit:
        return [resolve_path(Path(p)) for p in explicit]
    configs = sorted(CFG_DIR.glob("ego_policy*.yaml"))
    if not configs:
        raise FileNotFoundError("No ego_policy*.yaml configs found")
    return configs

# -----------------------
# Build policies
# -----------------------

def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict) -> WindowHashPolicy:
    return build_window_hash_policy(
        arena=arena,
        world_cfg=world_cfg,
        policy_cfg=ego_cfg,
        identifier=ego_cfg.get("identifier", "ego_window_hash"),
    )

def build_opp_policy(arena, world_cfg: Dict, opp_cfg: Dict, seed_offset: int = 0) -> WindowHashPolicy:
    return build_hash_policy(
        arena=arena,
        world_cfg=world_cfg,
        opp_cfg=opp_cfg,
        seed_offset=seed_offset,
        identifier=opp_cfg.get("identifier", "manual_shift_opponent"),
    )

# -----------------------
# Feature scoring (for biased starts)
# -----------------------

def feature_mean_x(xy: np.ndarray) -> float:
    """Average x over teams*agents (for scoring starts)."""
    return float(np.mean(xy[..., 0]))

def feature_neg_dist_to_point(xy: np.ndarray, point: Tuple[float, float]) -> float:
    """Higher score closer to point (for scoring starts)."""
    dif = xy - np.asarray(point, dtype=np.float32)
    d = np.linalg.norm(dif, axis=-1).mean()
    return float(-d)

# -----------------------
# Candidate extraction
# -----------------------

def extract_start_and_mean_xy(ep_windows_path: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      start_xy: [n, teams, agents, 2]   (t=0 positions per window for scoring/biasing starts)
      mean_xy:  [n, teams, agents, 2]   (time-mean positions per window for W-proxy)
    """
    with np.load(ep_windows_path, allow_pickle=False) as data:
        W = data["windows"]  # [N, T, teams, agents, F]
        N, T, _, _, _ = W.shape
        idx = np.arange(0, N, max(1, stride), dtype=int)
        starts = W[idx, 0, :, :, :2]
        means  = W[idx, :, :, :, :2].mean(axis=1)
        return starts, means

def build_weighted_sampler(
    indexer_src: EpisodeIndexer,
    split: str,
    stride: int = 1,
) -> Tuple[List[Tuple[Path, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      records_simple:    [(episode_path, N), ...]
      candidates_start:  [M, teams, agents, 2]
      candidates_mean:   [M, teams, agents, 2]
      per_map:           [M, 2] -> (ep_idx, local_idx_in_subsample)
    """
    records = list(indexer_src.iter_split(split))
    if not records:
        raise ValueError(f"No episodes in split {split}")
    policy_source = indexer_src.root

    start_list, mean_list, map_list, episode_paths, Ns = [], [], [], [], []
    for ep_idx, r in enumerate(records):
        ep_path = policy_source / r.path
        episode_paths.append(ep_path)
        with np.load(ep_path, allow_pickle=False) as ep:
            W = ep["windows"]
            Ns.append(W.shape[0])
        starts, means = extract_start_and_mean_xy(ep_path, stride=stride)
        m = np.stack([
            np.full((starts.shape[0],), ep_idx, dtype=np.int64),
            np.arange(starts.shape[0], dtype=np.int64)
        ], axis=1)
        map_list.append(m)
        start_list.append(starts)
        mean_list.append(means)

    candidates_start = np.concatenate(start_list, axis=0)
    candidates_mean  = np.concatenate(mean_list, axis=0)
    per_map = np.concatenate(map_list, axis=0)
    return [(episode_paths[i], Ns[i]) for i in range(len(records))], candidates_start, candidates_mean, per_map

def compute_candidate_scores(
    candidates_start_xy: np.ndarray,
    feature: str,
    arena,
    feature_point: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    M = candidates_start_xy.shape[0]
    scores = np.zeros((M,), dtype=np.float32)
    if feature == "mean_x":
        for i in range(M):
            scores[i] = feature_mean_x(candidates_start_xy[i])
    elif feature == "neg_dist_to_point":
        point = feature_point if feature_point is not None else (arena.width/2.0, arena.height/2.0)
        for i in range(M):
            scores[i] = feature_neg_dist_to_point(candidates_start_xy[i], np.asarray(point, dtype=np.float32))
    else:
        raise ValueError(f"Unknown feature '{feature}'. Use 'mean_x' or 'neg_dist_to_point'.")
    return scores

# -----------------------
# Alpha calibration on MEAN position proxy
# -----------------------

def calibrated_alpha_for_target_W_meanpos(
    baseline_mean_xy_flat: np.ndarray,        # [B, D]
    candidates_mean_xy_flat: np.ndarray,      # [M, D]
    scores_on_starts: np.ndarray,             # [M]
    target_w: float,
    sample_count: int = 4096,
    alpha_hi: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Binary-search alpha so that mean-position resamples achieve target W vs baseline."""
    if target_w <= 0:
        return 0.0
    rng = np.random.default_rng() if rng is None else rng

    base = baseline_mean_xy_flat.astype(np.float32)
    cand = candidates_mean_xy_flat.astype(np.float32)

    def wdist_for_alpha(alpha: float) -> float:
        logits = alpha * (scores_on_starts - scores_on_starts.mean())
        p = np.exp(logits - logits.max()); p /= np.sum(p)
        idx = rng.choice(len(p), size=min(sample_count, len(p)), replace=True, p=p)
        frames = cand[idx]
        return float(wasserstein_distance_numpy(frames, base))

    lo, hi = 0.0, alpha_hi
    if wdist_for_alpha(hi) < target_w:
        return hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if wdist_for_alpha(mid) < target_w:
            lo = mid
        else:
            hi = mid
    return hi

# -----------------------
# World rollout helpers (no added noise)
# -----------------------

def _reset_world_to_start(world, start_state: np.ndarray) -> None:
    if hasattr(world, "reset_to"):
        world.reset_to(start_state)
        return
    if hasattr(world, "set_state"):
        world.reset()
        world.set_state(start_state)
        return
    raise RuntimeError("World does not support resetting to an arbitrary start state.")

def rollout_window_from_start(
    arena,
    world_cfg: Dict,
    ego_policy: WindowHashPolicy,
    opp_policy: WindowHashPolicy,
    start_state: np.ndarray,      # [teams, agents, F]
    window_len: int,
    rng: np.random.Generator,
):
    """
    Start world at start_state, roll for `window_len` steps using real dynamics/policies.
    The world is constructed with history == window_len so observe_window() matches T.
    """
    # Ensure the world's history equals the dataset window length
    world_cfg_local = dict(world_cfg)
    world_cfg_local["history"] = int(window_len)
    world = build_world(arena, world_cfg_local, rng=rng)

    _reset_world_to_start(world, start_state.astype(np.float32))

    # Roll forward window_len steps; at each step, policies see a T-long window (left-padded initially)
    ego_action_last = None
    opp_action_last = None
    for _ in range(window_len):
        window_for_policy = world.observe_window()
        ego_action = ego_policy.act(window_for_policy, deterministic=False).astype(np.float32)
        opp_action = opp_policy.act(window_for_policy, deterministic=False).astype(np.float32)
        action_stack = np.stack([ego_action, opp_action], axis=0)  # (teams, agents, 2)
        world.step(action_stack)
        ego_action_last = ego_action
        opp_action_last = opp_action

    final_window = world.observe_window().astype(np.float32)
    return final_window, ego_action_last, opp_action_last

def _build_policy_window(history: List[np.ndarray], window_len: int) -> np.ndarray:
    """
    Build a (T, teams, agents, F) window from a list of states (each [teams,agents,F]).
    If history shorter than T, left-pad by repeating the earliest state.
    """
    T = window_len
    if len(history) >= T:
        seq = history[-T:]
    else:
        pad = [history[0]] * (T - len(history))
        seq = pad + history
    return np.stack(seq, axis=0).astype(np.float32)


# -----------------------
# Baseline mean-position collector (for proxy)
# -----------------------

def collect_baseline_mean_positions(indexer: EpisodeIndexer, root: Path, max_samples: int = 5000) -> np.ndarray:
    all_means = []
    for rec in indexer.entries:
        with np.load(root / rec.path, allow_pickle=False) as data:
            W = data["windows"]  # [N, T, teams, agents, F]
            means = W[:, :, :, :, :2].mean(axis=1)  # [N, teams, agents, 2]
            all_means.append(means.reshape(means.shape[0], -1))
    full = np.concatenate(all_means, axis=0)
    if full.shape[0] > max_samples:
        sel = np.random.default_rng(1234).choice(full.shape[0], size=max_samples, replace=False)
        full = full[sel]
    return full.astype(np.float32)

# -----------------------
# Core processing
# -----------------------

def process_policy(
    ego_cfg_path: Path,
    data_config: Dict,
    generators_cfg: Dict,
    base_root: Path,
    target_root: Path,
    opp_cfg: Dict,
    seed_offset: int = 0,
) -> Dict[str, Dict[str, float]]:
    name = ego_cfg_path.stem
    policy_source = base_root / name / "iid"
    if not policy_source.exists():
        raise FileNotFoundError(f"Missing baseline dataset for {name} at {policy_source}")
    LOGGER.info("Processing policy %s", name)

    arena = build_arena(data_config["arena"])
    world_cfg = data_config["world"]
    teams = int(world_cfg["teams"])
    agents = int(world_cfg["agents_per_team"])

    # baseline actions (for achieved action-W)
    baseline_stats_path = policy_source / "baseline_stats.npz"
    if not baseline_stats_path.exists():
        raise FileNotFoundError(f"Missing baseline stats at {baseline_stats_path}")
    baseline_stats = np.load(baseline_stats_path, allow_pickle=False)
    baseline_actions = baseline_stats["actions"]

    ego_cfg = load_yaml(ego_cfg_path)
    ego_policy = build_ego_policy(arena, world_cfg, ego_cfg)
    opp_policy = build_opp_policy(arena, world_cfg, opp_cfg, seed_offset=seed_offset)

    indexer_src = EpisodeIndexer.load(policy_source)
    policy_target = target_root / name
    policy_target.mkdir(parents=True, exist_ok=True)

    splits = list(indexer_src.splits())
    if not splits:
        raise ValueError(f"No splits found for policy {name}")

    # infer window len & state_dim from a sample episode
    first_split = splits[0]
    first_records = list(indexer_src.iter_split(first_split))
    sample_episode_path = policy_source / first_records[0].path
    with np.load(sample_episode_path, allow_pickle=False) as sample_episode:
        sample_windows = sample_episode["windows"]
    if sample_windows.ndim != 5:
        raise ValueError(f"Expected windows with 5 dims, got {sample_windows.shape}")
    window_len = int(sample_windows.shape[1])
    state_dim = int(sample_windows.shape[-1])

    # Baseline mean positions for proxy (time-mean x,y)
    baseline_mean_xy_flat = collect_baseline_mean_positions(indexer_src, policy_source, max_samples=5000)

    summary: Dict[str, Dict[str, float]] = {}
    rng_global = np.random.default_rng(int(generators_cfg.get("seed", 0)))

    for shift_name, spec in generators_cfg["shifts"].items():
        LOGGER.info("[%s] Generating shift '%s'", name, shift_name)
        target_state_w = float(spec["state_wasserstein"])
        target_action_w = float(spec.get("action_wasserstein", 0.0))
        gen_type = spec["generator"]["type"]
        gen_params = dict(spec["generator"].get("params", {}))
        local_seed = int(spec.get("seed", rng_global.integers(0, 2**31-1)))
        rng = np.random.default_rng(local_seed)

        if gen_type != "resample_weighted":
            raise ValueError("Use generator.type = 'resample_weighted' for biased-start on-manifold generation.")

        # --- Build candidate bank from baseline windows (per split) ---
        feature = gen_params.get("feature", "mean_x")
        feature_point = tuple(gen_params.get("feature_point", [])) if "feature_point" in gen_params else None
        stride = int(gen_params.get("candidate_stride", 1))
        alpha_hi = float(gen_params.get("alpha_max", 50.0))

        # Build candidates across ALL splits to calibrate alpha
        all_starts, all_means = [], []
        per_split_payload = {}  # split -> (records_simple, candidates_start, candidates_mean, per_map)
        for split in splits:
            records_simple, candidates_start, candidates_mean, per_map = build_weighted_sampler(
                indexer_src=indexer_src,
                split=split,
                stride=stride,
            )
            per_split_payload[split] = (records_simple, candidates_start, candidates_mean, per_map)
            all_starts.append(candidates_start)
            all_means.append(candidates_mean)
        all_candidates_start = np.concatenate(all_starts, axis=0)
        all_candidates_mean  = np.concatenate(all_means, axis=0)

        # Scores on starts; W-proxy on mean over time (positions only)
        scores = compute_candidate_scores(all_candidates_start, feature, arena, feature_point)
        all_means_flat = all_candidates_mean.reshape(all_candidates_mean.shape[0], -1).astype(np.float32)

        # Calibrate alpha
        alpha = calibrated_alpha_for_target_W_meanpos(
            baseline_mean_xy_flat=baseline_mean_xy_flat,
            candidates_mean_xy_flat=all_means_flat,
            scores_on_starts=scores,
            target_w=target_state_w,
            sample_count=4096,
            alpha_hi=alpha_hi,
            rng=rng,
        )
        LOGGER.info("[%s/%s] calibrated alpha=%.3f to target mean-pos W≈%.3f", name, shift_name, alpha, target_state_w)

        new_indexer = EpisodeIndexer(root=policy_target / shift_name)
        all_mean_pos = []
        all_actions_ego = []

        # For each split, compute probabilities with the same alpha and generate windows
        for split in splits:
            records_simple, candidates_start, candidates_mean, per_map = per_split_payload[split]
            split_scores = compute_candidate_scores(candidates_start, feature, arena, feature_point)
            logits = alpha * (split_scores - split_scores.mean())
            p = np.exp(logits - logits.max()); p /= np.sum(p)

            split_dir = (policy_target / shift_name / split)
            split_dir.mkdir(parents=True, exist_ok=True)

            # We'll need access to the raw source episodes to fetch start states and full feature dim
            episode_paths = [ep for ep, _N in records_simple]

            mean_pos_accum = []
            actions_ego_accum = []

            for ep_output_idx, (_ep_path, N_prime) in enumerate(records_simple):
                # Draw N' *independent* starts (restart each time)
                idx_global = rng.choice(len(p), size=N_prime, replace=True, p=p)
                picks = [(int(per_map[j, 0]), int(per_map[j, 1])) for j in idx_global]  # (ep_idx, local_idx)

                # Build arrays
                # infer dims from first pick
                first_ep_idx, first_local = picks[0]
                with np.load(episode_paths[first_ep_idx], allow_pickle=False) as first_npz:
                    W0 = first_npz["windows"]  # [N, T, teams, agents, F]
                _, _, teams_, agents_, F_ = W0.shape
                sd = min(state_dim, F_)
                windows = np.zeros((N_prime, window_len, teams_, agents_, sd), dtype=np.float32)
                ego_actions = np.zeros((N_prime, agents_, 2), dtype=np.float32)
                opp_actions = np.zeros((N_prime, agents_, 2), dtype=np.float32)
                positions_mean = np.zeros((N_prime, teams_, agents_, 2), dtype=np.float32)

                # Cache NPZ files
                cache: Dict[Path, np.lib.npyio.NpzFile] = {}

                for i, (ep_idx, local_idx) in enumerate(picks):
                    path = episode_paths[ep_idx]
                    if path not in cache:
                        cache[path] = np.load(path, allow_pickle=False)
                    W = cache[path]["windows"]  # [N, T, teams, agents, F]
                    N_src = W.shape[0]
                    src_idx = min(local_idx * max(1, stride), N_src - 1)

                    # Start state: take the full feature slice to sd dims
                    start_state = W[src_idx, 0, :, :, :sd]  # [teams, agents, sd]

                    # Rollout for window_len with real policies/dynamics (no noise)
                    win, ea, oa = rollout_window_from_start(
                        arena=arena,
                        world_cfg=world_cfg,
                        ego_policy=ego_policy,
                        opp_policy=opp_policy,
                        start_state=start_state,
                        window_len=window_len,
                        rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
                    )

                    windows[i] = win
                    ego_actions[i] = ea
                    opp_actions[i] = oa
                    positions_mean[i] = win[:, :, :, :2].mean(axis=0)

                for npz in cache.values():
                    npz.close()

                out_path = split_dir / f"episode_{ep_output_idx:05d}.npz"
                np.savez_compressed(
                    out_path,
                    windows=windows,
                    ego_actions=ego_actions,
                    opponent_actions=opp_actions,
                    positions=positions_mean,  # time-mean positions (convenience)
                    policy_id=name,            # preserve original policy label
                )
                new_indexer.add_episode(split, out_path, length=N_prime, policy_id=name)

                mean_pos_accum.append(positions_mean.reshape(N_prime, -1))
                actions_ego_accum.append(ego_actions.reshape(N_prime, -1))

            all_mean_pos.append(np.concatenate(mean_pos_accum, axis=0))
            all_actions_ego.append(np.concatenate(actions_ego_accum, axis=0))

        new_indexer.save()

        # Save a small sample of states/actions for convenience
        states_concat = np.concatenate(all_mean_pos, axis=0)  # mean-pos flattened
        actions_concat = np.concatenate(all_actions_ego, axis=0)
        sample_size = min(5000, states_concat.shape[0])
        sel = np.random.default_rng(1234).choice(states_concat.shape[0], size=sample_size, replace=False)
        np.savez_compressed(
            policy_target / shift_name / "baseline_stats.npz",
            states=states_concat[sel],
            actions=actions_concat[sel],
        )

        # Achieved divergences
        achieved_state = float(wasserstein_distance_numpy(states_concat, baseline_mean_xy_flat))
        achieved_action = float(wasserstein_distance_numpy(actions_concat, baseline_actions))

        summary_entry = {
            "target_state": target_state_w,
            "target_action": target_action_w,
            "achieved_state": achieved_state,
            "achieved_action": achieved_action,
            "note": "resample_weighted (biased starts; rollout windows; mean-pos proxy)",
            "alpha_info": {"alpha_max": alpha_hi, "calibrated": float(alpha)},
        }
        LOGGER.info("[%s/%s] achieved mean-pos W=%.3f, action W=%.3f", name, shift_name, achieved_state, achieved_action)
        # Store
        if name not in summary:
            summary[name] = {}
        summary[shift_name] = summary_entry

    return summary

# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OOD datasets via biased starts + policy rollouts (on-manifold).")
    p.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    p.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    p.add_argument("--generators", type=str, default=str(PACKAGE_ROOT / "cfg" / "generators.yaml"),
                   help="YAML specifying shifts, targets, generator types/params")
    p.add_argument("--base", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()),
                   help="Base root containing per-policy IID datasets (for counts + baseline_stats)")
    p.add_argument("--target", type=str, default=str((PACKAGE_ROOT / "../output/data/ood_manual").resolve()))
    p.add_argument("--ego", action="append", help="Specific ego policy config paths (defaults to all ego_policy*.yaml)")
    return p.parse_args()

def main() -> None:
    configure_logging()
    args = parse_args()
    data_cfg = load_yaml(Path(args.data))
    opp_cfg = load_yaml(Path(args.opponent))
    gens_cfg = load_yaml(Path(args.generators))
    base = resolve_path(Path(args.base))
    target = resolve_path(Path(args.target))
    target.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for idx, ego_cfg_path in enumerate(find_ego_configs(args.ego)):
        s = process_policy(
            ego_cfg_path=ego_cfg_path,
            data_config=data_cfg,
            generators_cfg=gens_cfg,
            base_root=base,
            target_root=target,
            opp_cfg=opp_cfg,
            seed_offset=idx + 1,
        )
        summaries[ego_cfg_path.stem] = s
    LOGGER.info("Manual state generation complete: %s", json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()