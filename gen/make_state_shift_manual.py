#!/usr/bin/env python3
"""
make_state_shift_manual.py

Create OOD datasets by *state-biased* start selection from IID windows, with TWO generation modes:

DEFAULT (window-iid, current behavior):
  - Reweight & resample *start frames* from IID windows
  - For EACH sampled start, rebuild world, reset to start, roll forward `window_len` steps
  - Each stored "window" is independent (not trajectory-like)

OPTIONAL (trajectory-like episodes; enable with --trajectory_episodes):
  - For EACH output episode:
      * sample ONE biased start frame
      * rebuild world, reset to start
      * roll forward for L steps (L matched to baseline episode length)
      * store sliding windows at every step (trajectory-like, IID-style)

Key behaviors:
- Start-state bias uses: softmax(alpha * centered_score), where score = mean_x at t=0.
- Deterministic generation via stable hashing seeds.
- Outputs remain compatible with training code:
    windows, ego_actions, opponent_actions, positions, policy_id
  and writes index.json + baseline_stats.npz under each shift root.

Output structure:
  <target_root>/<ego_policy_name>/<shift_name>/{train,val,test}/episode_XXXXX.npz
  <target_root>/<ego_policy_name>/<shift_name>/index.json
  <target_root>/<ego_policy_name>/<shift_name>/baseline_stats.npz
  <target_root>/<ego_policy_name>/alpha_sweep.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Protocol

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
from policyOrProxy.core.policies.policy_factory import build_policy
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world

LOGGER = logging.getLogger(__name__)


# -----------------------
# Protocols
# -----------------------

class Policy(Protocol):
    identifier: str
    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def set_rng(self, rng: np.random.Generator) -> None: ...


# -----------------------
# Helpers / config loaders
# -----------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


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


def stable_int_seed(*parts: object, bits: int = 32) -> int:
    """
    Stable integer seed derived from arbitrary parts (independent of Python's randomized hash()).
    """
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(msg).digest()
    if bits <= 32:
        return int.from_bytes(digest[:4], "little", signed=False)
    if bits <= 64:
        return int.from_bytes(digest[:8], "little", signed=False)
    return int.from_bytes(digest[:16], "little", signed=False)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax that guarantees:
      - finite probabilities
      - non-negative probabilities
      - sum(p) == 1.0 exactly (within float64 arithmetic), via last-element correction
    """
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    M = x.size
    if M == 0:
        raise ValueError("stable_softmax: empty logits")

    # Special-case: all equal -> exact uniform (helps alpha=0)
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
    y = x - m
    y = np.clip(y, -745.0, 0.0)  # exp(-745) ~ min float64

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
    p[-1] += 1.0 - float(p.sum(dtype=np.float64))  # exactness correction

    if p[-1] < 0.0:
        p = np.clip(p, 0.0, None)
        p /= p.sum(dtype=np.float64)
        p[-1] += 1.0 - float(p.sum(dtype=np.float64))

    return p


# -----------------------
# Policies (multi-class via factory)
# -----------------------

def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict, *, init_seed: int, identifier: str) -> Policy:
    return build_policy(
        arena=arena,
        world_cfg=world_cfg,
        policy_cfg=ego_cfg,
        role="ego",
        identifier=identifier,
        init_seed=int(ego_cfg.get("init_seed", init_seed)),
    )


def build_opp_policy(arena, world_cfg: Dict, opp_cfg: Dict, *, init_seed: int, identifier: str) -> Policy:
    return build_policy(
        arena=arena,
        world_cfg=world_cfg,
        policy_cfg=opp_cfg,
        role="opponent",
        identifier=identifier,
        init_seed=int(opp_cfg.get("init_seed", init_seed)),
    )


# -----------------------
# Candidate extraction + scoring (state-only bias)
# -----------------------

def extract_start_xy(ep_windows_path: Path) -> np.ndarray:
    """
    Returns:
      start_xy: [N, teams, agents, 2] (t=0 positions per window)
    """
    with np.load(ep_windows_path, allow_pickle=False) as data:
        W = np.asarray(data["windows"], dtype=np.float32)  # [N,T,teams,agents,F]
        starts = W[:, 0, :, :, :2]
        return starts


def feature_mean_x(xy: np.ndarray) -> float:
    """Average x over teams*agents."""
    return float(np.mean(xy[..., 0]))


def build_candidate_bank_for_split(
    indexer_src: EpisodeIndexer,
    split: str,
) -> Tuple[List[Tuple[Path, int]], np.ndarray, np.ndarray]:
    """
    Returns:
      records_simple: [(episode_path, N_windows), ...]
      scores:         [M] score per candidate window start (mean_x on t=0)
      per_map:        [M, 2] -> (ep_idx, local_window_idx)
    """
    records = list(indexer_src.iter_split(split))
    if not records:
        raise ValueError(f"No episodes in split {split}")

    policy_source = indexer_src.root

    score_list: List[np.ndarray] = []
    map_list: List[np.ndarray] = []
    episode_paths: List[Path] = []
    Ns: List[int] = []

    for ep_idx, r in enumerate(records):
        ep_path = policy_source / r.path
        episode_paths.append(ep_path)

        with np.load(ep_path, allow_pickle=False) as ep:
            W = np.asarray(ep["windows"], dtype=np.float32)
            N = int(W.shape[0])
            Ns.append(N)

        starts = extract_start_xy(ep_path)  # [N,teams,agents,2]
        scores = np.zeros((starts.shape[0],), dtype=np.float32)
        for i in range(starts.shape[0]):
            scores[i] = feature_mean_x(starts[i])

        m = np.stack(
            [
                np.full((starts.shape[0],), ep_idx, dtype=np.int64),
                np.arange(starts.shape[0], dtype=np.int64),
            ],
            axis=1,
        )

        score_list.append(scores)
        map_list.append(m)

    all_scores = np.concatenate(score_list, axis=0).astype(np.float32)
    per_map = np.concatenate(map_list, axis=0).astype(np.int64)

    records_simple = [(episode_paths[i], Ns[i]) for i in range(len(records))]
    return records_simple, all_scores, per_map


# -----------------------
# World rollout helpers
# -----------------------

def _reset_world_to_start(world, start_state: np.ndarray) -> None:
    if hasattr(world, "reset_to"):
        world.reset_to(start_state)
        return
    if hasattr(world, "set_state"):
        world.reset()
        world.set_state(start_state)
        return
    raise RuntimeError("World does not support resetting to an arbitrary start state (reset_to or set_state missing).")


def _maybe_done(world) -> bool:
    """
    Best-effort early termination support.
    If your World implements a done/terminated flag or method, we respect it.
    Otherwise this always returns False.
    """
    for attr in ("done", "terminated", "is_done", "is_terminated"):
        if hasattr(world, attr):
            v = getattr(world, attr)
            try:
                return bool(v() if callable(v) else v)
            except Exception:
                pass
    return False


def rollout_window_from_start_iid(
    arena,
    world_cfg: Dict,
    ego_policy: Policy,
    opp_policy: Policy,
    start_state: np.ndarray,      # [teams, agents, F]
    window_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DEFAULT MODE helper:
    - Build a world
    - Reset to start_state
    - Step for `window_len` steps
    - Return final observed window (history), and last actions
    """
    world_cfg_local = dict(world_cfg)
    world_cfg_local["history"] = int(window_len)
    world_cfg_local["window_len"] = int(window_len)

    world_rng = np.random.default_rng(int(seed))
    world = build_world(arena, world_cfg_local, rng=world_rng)

    ego_policy.set_rng(np.random.default_rng(int(stable_int_seed(seed, "ego"))))
    opp_policy.set_rng(np.random.default_rng(int(stable_int_seed(seed, "opp"))))

    _reset_world_to_start(world, start_state.astype(np.float32))

    ego_action_last = None
    opp_action_last = None
    for _ in range(int(window_len)):
        window_for_policy = world.observe_window()
        ego_action = ego_policy.act(window_for_policy, deterministic=False).astype(np.float32)
        opp_action = opp_policy.act(window_for_policy, deterministic=False).astype(np.float32)
        action_stack = np.stack([ego_action, opp_action], axis=0)  # (teams, agents, 2)
        world.step(action_stack)
        ego_action_last = ego_action
        opp_action_last = opp_action
        if _maybe_done(world):
            break

    final_window = world.observe_window().astype(np.float32)
    if ego_action_last is None or opp_action_last is None:
        raise RuntimeError("Rollout produced no actions; window_len may be invalid.")
    return final_window, ego_action_last, opp_action_last


def rollout_episode_from_start_trajectory(
    arena,
    world_cfg: Dict,
    ego_policy: Policy,
    opp_policy: Policy,
    start_state: np.ndarray,       # [teams, agents, F]
    window_len: int,
    steps: int,                    # target episode length (matched to baseline episode windows)
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    TRAJECTORY MODE:
    - Build a world with history/window_len = `window_len`
    - Reset to start_state
    - For t in [0..steps):
        * observe_window (sliding history)
        * act
        * step
        * record window, actions, positions
    - Returns arrays shaped like IID rollout:
        windows: (L, window_len, teams, agents, state_dim)
        ego_actions: (L, agents, 2)
        opp_actions: (L, agents, 2)
        positions: (L, teams, agents, 2)   (true per-step positions, trajectory-like)
    """
    world_cfg_local = dict(world_cfg)
    world_cfg_local["history"] = int(window_len)
    world_cfg_local["window_len"] = int(window_len)

    world_rng = np.random.default_rng(int(seed))
    world = build_world(arena, world_cfg_local, rng=world_rng)

    ego_policy.set_rng(np.random.default_rng(int(stable_int_seed(seed, "ego_ep"))))
    opp_policy.set_rng(np.random.default_rng(int(stable_int_seed(seed, "opp_ep"))))

    _reset_world_to_start(world, start_state.astype(np.float32))

    windows: List[np.ndarray] = []
    ego_actions: List[np.ndarray] = []
    opp_actions: List[np.ndarray] = []
    positions: List[np.ndarray] = []

    for _t in range(int(steps)):
        window_for_policy = world.observe_window().astype(np.float32)
        ego_action = ego_policy.act(window_for_policy, deterministic=False).astype(np.float32)
        opp_action = opp_policy.act(window_for_policy, deterministic=False).astype(np.float32)

        action_stack = np.stack([ego_action, opp_action], axis=0)  # (teams, agents, 2)
        world.step(action_stack)

        windows.append(window_for_policy)
        ego_actions.append(ego_action)
        opp_actions.append(opp_action)
        # record true positions after step (matches IID rollout semantics)
        pos = getattr(world, "state", None)
        if pos is not None and hasattr(pos, "positions"):
            positions.append(np.asarray(pos.positions, dtype=np.float32)[..., :2].copy())
        else:
            # fallback: take last frame xy from window (less ideal, but keeps shape)
            positions.append(window_for_policy[-1, :, :, :2].copy())

        if _maybe_done(world):
            break

    if not windows:
        raise RuntimeError("Trajectory rollout produced length 0. Check world termination/spawn/config.")

    W = np.asarray(windows, dtype=np.float32)
    EA = np.asarray(ego_actions, dtype=np.float32)
    OA = np.asarray(opp_actions, dtype=np.float32)
    P = np.asarray(positions, dtype=np.float32)
    return W, EA, OA, P


# -----------------------
# Baseline collector (mean position proxy)
# -----------------------

def collect_baseline_mean_positions(indexer: EpisodeIndexer, root: Path, max_samples: int = 5000) -> np.ndarray:
    all_means: List[np.ndarray] = []
    for rec in indexer.entries:
        with np.load(root / rec.path, allow_pickle=False) as data:
            W = np.asarray(data["windows"], dtype=np.float32)  # [L,T,teams,agents,F]
            means = W[:, :, :, :, :2].mean(axis=1)            # [L,teams,agents,2]
            all_means.append(means.reshape(means.shape[0], -1))
    full = np.concatenate(all_means, axis=0).astype(np.float32)
    if full.shape[0] > int(max_samples):
        sel = np.random.default_rng(1234).choice(full.shape[0], size=int(max_samples), replace=False)
        full = full[sel]
    return full


# -----------------------
# Reservoir sampling (for baseline_stats + metric computation)
# -----------------------

def reservoir_update(
    rng: np.random.Generator,
    row: np.ndarray,
    K: int,
    seen: int,
    filled: int,
    buf: Optional[np.ndarray],
) -> tuple[int, int, np.ndarray]:
    seen += 1
    if buf is None:
        buf = np.zeros((K, row.shape[0]), dtype=np.float32)
    if filled < K:
        buf[filled] = row
        filled += 1
        return seen, filled, buf
    j = int(rng.integers(0, seen))
    if j < K:
        buf[j] = row
    return seen, filled, buf


# -----------------------
# Core processing
# -----------------------

def process_policy(
    ego_cfg_path: Path,
    data_config: Dict,
    shifts_cfg: Dict,
    base_root: Path,
    target_root: Path,
    opp_cfg: Dict,
    seed_offset: int = 0,
    trajectory_episodes: bool = False,
) -> Dict[str, Dict]:
    policy_name = ego_cfg_path.stem
    policy_source = base_root / policy_name / "iid"
    if not policy_source.exists():
        raise FileNotFoundError(f"Missing baseline dataset for {policy_name} at {policy_source}")

    LOGGER.info("Processing policy %s", policy_name)
    LOGGER.info("Generation mode: %s", "TRAJECTORY_EPISODES" if trajectory_episodes else "WINDOW_IID")

    arena = build_arena(data_config["arena"])
    world_cfg = data_config["world"]

    baseline_stats_path = policy_source / "baseline_stats.npz"
    baseline_actions: Optional[np.ndarray] = None
    if baseline_stats_path.exists():
        with np.load(baseline_stats_path, allow_pickle=False) as baseline_stats:
            if "actions" in baseline_stats:
                baseline_actions = np.asarray(baseline_stats["actions"], dtype=np.float32)

    ego_cfg = load_yaml(ego_cfg_path)
    base_seed = int(data_config.get("rollout", {}).get("seed", 0))

    ego_init_seed = stable_int_seed(base_seed, policy_name, "init", "ego")
    opp_init_seed = stable_int_seed(base_seed, policy_name, "init", "opp", seed_offset)

    ego_policy = build_ego_policy(
        arena, world_cfg, ego_cfg,
        init_seed=ego_init_seed,
        identifier=str(ego_cfg.get("identifier", policy_name)),
    )
    opp_policy = build_opp_policy(
        arena, world_cfg, opp_cfg,
        init_seed=opp_init_seed,
        identifier=str(opp_cfg.get("identifier", "manual_shift_opponent")),
    )

    indexer_src = EpisodeIndexer.load(policy_source)
    splits = list(indexer_src.splits())
    if not splits:
        raise ValueError(f"No splits found for policy {policy_name}")

    first_split = splits[0]
    first_records = list(indexer_src.iter_split(first_split))
    if not first_records:
        raise ValueError(f"No records in split '{first_split}' for policy {policy_name}")

    sample_episode_path = policy_source / first_records[0].path
    with np.load(sample_episode_path, allow_pickle=False) as sample_episode:
        sample_windows = np.asarray(sample_episode["windows"], dtype=np.float32)
    if sample_windows.ndim != 5:
        raise ValueError(f"Expected windows with 5 dims, got {sample_windows.shape}")

    window_len = int(sample_windows.shape[1])
    state_dim = int(sample_windows.shape[-1])
    teams = int(sample_windows.shape[2])
    agents = int(sample_windows.shape[3])

    LOGGER.info(
        "[%s] inferred window_len=%d state_dim=%d teams=%d agents=%d",
        policy_name, window_len, state_dim, teams, agents
    )

    baseline_mean_xy_flat = collect_baseline_mean_positions(indexer_src, policy_source, max_samples=5000)

    policy_target = target_root / policy_name
    policy_target.mkdir(parents=True, exist_ok=True)
    sweep_csv_path = policy_target / "alpha_sweep.csv"
    if not sweep_csv_path.exists():
        sweep_csv_path.write_text("shift,alpha,achieved_state_w,achieved_action_w,mode\n", encoding="utf-8")

    global_shift_seed = int(shifts_cfg.get("seed", 0))
    summary: Dict[str, Dict] = {}

    for shift_name, spec in shifts_cfg.get("shifts", {}).items():
        alpha = float(spec.get("alpha", 0.0))
        shift_seed = stable_int_seed(global_shift_seed, policy_name, shift_name)

        LOGGER.info("[%s/%s] generating (alpha=%.6g)", policy_name, shift_name, alpha)

        out_root = policy_target / shift_name
        out_root.mkdir(parents=True, exist_ok=True)
        new_indexer = EpisodeIndexer(root=out_root)

        # Reservoir buffers for metrics + baseline_stats
        K = 5000
        stats_rng = np.random.default_rng(int(stable_int_seed(shift_seed, "reservoir")))
        seen_s = filled_s = 0
        seen_a = filled_a = 0
        states_res: Optional[np.ndarray] = None
        actions_res: Optional[np.ndarray] = None

        for split in splits:
            records_simple, scores, per_map = build_candidate_bank_for_split(indexer_src, split=split)

            scores_centered = scores - float(scores.mean())
            p = stable_softmax(alpha * scores_centered)

            split_dir = out_root / split
            split_dir.mkdir(parents=True, exist_ok=True)

            episode_paths = [ep for (ep, _N) in records_simple]

            for ep_output_idx, (_ep_path, N_prime) in enumerate(records_simple):
                N_prime = int(N_prime)

                ep_rng = np.random.default_rng(int(stable_int_seed(shift_seed, split, ep_output_idx, "ep_rng")))

                # ---------- NEW: trajectory mode samples ONE start per episode ----------
                if trajectory_episodes:
                    j = int(ep_rng.choice(len(p), size=1, replace=True, p=p)[0])
                    src_ep_idx = int(per_map[j, 0])
                    src_win_idx = int(per_map[j, 1])

                    src_path = episode_paths[src_ep_idx]
                    with np.load(src_path, allow_pickle=False) as z:
                        Wsrc = np.asarray(z["windows"], dtype=np.float32)  # [L,T,teams,agents,F]
                    src_win_idx = min(src_win_idx, int(Wsrc.shape[0]) - 1)
                    start_state = Wsrc[src_win_idx, 0, :, :, :].astype(np.float32)  # [teams,agents,F]

                    ep_seed = stable_int_seed(shift_seed, split, ep_output_idx, "traj_episode")
                    W, EA, OA, POS = rollout_episode_from_start_trajectory(
                        arena=arena,
                        world_cfg=world_cfg,
                        ego_policy=ego_policy,
                        opp_policy=opp_policy,
                        start_state=start_state,
                        window_len=window_len,
                        steps=N_prime,          # match baseline episode length
                        seed=int(ep_seed),
                    )

                    # Ensure dims consistent
                    W = W[:, :, :, :, :state_dim].astype(np.float32)
                    POS = POS[:, :, :, :2].astype(np.float32)

                    out_path = split_dir / f"episode_{ep_output_idx:05d}.npz"
                    np.savez_compressed(
                        out_path,
                        windows=W,
                        ego_actions=EA.astype(np.float32),
                        opponent_actions=OA.astype(np.float32),
                        positions=POS.astype(np.float32),  # TRAJECTORY positions
                        policy_id=policy_name,
                    )
                    new_indexer.add_episode(split, out_path, length=int(W.shape[0]), policy_id=policy_name)

                    # Reservoir updates: mean-pos over time-within-window, flattened; actions flattened
                    # (keeps metric comparable to your existing pipeline)
                    positions_mean = W[:, :, :, :, :2].mean(axis=1)  # [L,teams,agents,2]
                    for i in range(int(W.shape[0])):
                        srow = positions_mean[i].reshape(-1).astype(np.float32)
                        arow = EA[i].reshape(-1).astype(np.float32)
                        seen_s, filled_s, states_res = reservoir_update(stats_rng, srow, K, seen_s, filled_s, states_res)
                        seen_a, filled_a, actions_res = reservoir_update(stats_rng, arow, K, seen_a, filled_a, actions_res)

                    continue  # next baseline episode

                # ---------- DEFAULT: iid windows (old behavior) ----------
                idx_global = ep_rng.choice(len(p), size=N_prime, replace=True, p=p)
                picks = [(int(per_map[j, 0]), int(per_map[j, 1])) for j in idx_global]

                windows = np.zeros((N_prime, window_len, teams, agents, state_dim), dtype=np.float32)
                ego_actions = np.zeros((N_prime, agents, 2), dtype=np.float32)
                opp_actions = np.zeros((N_prime, agents, 2), dtype=np.float32)
                positions_mean = np.zeros((N_prime, teams, agents, 2), dtype=np.float32)

                cache: Dict[Path, np.lib.npyio.NpzFile] = {}

                for i, (src_ep_idx, src_win_idx) in enumerate(picks):
                    src_path = episode_paths[src_ep_idx]
                    if src_path not in cache:
                        cache[src_path] = np.load(src_path, allow_pickle=False)

                    Wsrc = np.asarray(cache[src_path]["windows"], dtype=np.float32)  # [L,T,teams,agents,F]
                    N_src = int(Wsrc.shape[0])
                    src_idx = min(int(src_win_idx), N_src - 1)

                    start_state = Wsrc[src_idx, 0, :, :, :].astype(np.float32)  # [teams,agents,F]

                    sample_seed = stable_int_seed(shift_seed, split, ep_output_idx, i, "sample")
                    win, ea, oa = rollout_window_from_start_iid(
                        arena=arena,
                        world_cfg=world_cfg,
                        ego_policy=ego_policy,
                        opp_policy=opp_policy,
                        start_state=start_state,
                        window_len=window_len,
                        seed=int(sample_seed),
                    )

                    win = win[:, :, :, :state_dim].astype(np.float32)

                    windows[i] = win
                    ego_actions[i] = ea.astype(np.float32)
                    opp_actions[i] = oa.astype(np.float32)
                    positions_mean[i] = win[:, :, :, :2].mean(axis=0)

                    srow = positions_mean[i].reshape(-1).astype(np.float32)
                    arow = ego_actions[i].reshape(-1).astype(np.float32)

                    seen_s, filled_s, states_res = reservoir_update(stats_rng, srow, K, seen_s, filled_s, states_res)
                    seen_a, filled_a, actions_res = reservoir_update(stats_rng, arow, K, seen_a, filled_a, actions_res)

                for npz in cache.values():
                    npz.close()

                out_path = split_dir / f"episode_{ep_output_idx:05d}.npz"
                np.savez_compressed(
                    out_path,
                    windows=windows,
                    ego_actions=ego_actions,
                    opponent_actions=opp_actions,
                    positions=positions_mean,  # mean-per-window (non-trajectory)
                    policy_id=policy_name,
                )
                new_indexer.add_episode(split, out_path, length=N_prime, policy_id=policy_name)

        new_indexer.save()

        if states_res is None or actions_res is None or filled_s == 0 or filled_a == 0:
            raise RuntimeError(f"[{policy_name}/{shift_name}] No samples collected for baseline_stats.")

        states_out = states_res[:filled_s].copy()
        actions_out = actions_res[:filled_a].copy()
        np.savez_compressed(out_root / "baseline_stats.npz", states=states_out, actions=actions_out)

        achieved_state_w = float(wasserstein_distance_numpy(states_out, baseline_mean_xy_flat))
        achieved_action_w = float("nan")
        if baseline_actions is not None:
            achieved_action_w = float(wasserstein_distance_numpy(actions_out, baseline_actions))

        metrics = {
            "policy": policy_name,
            "shift": shift_name,
            "alpha": float(alpha),
            "mode": "trajectory_episodes" if trajectory_episodes else "window_iid",
            "achieved_state_w": achieved_state_w,
            "achieved_action_w": achieved_action_w,
            "notes": (
                "start bias = softmax(alpha * centered mean_x(t0)); "
                "default window_iid = independent windows; "
                "trajectory_episodes = one biased start per episode then sliding rollout"
            ),
        }

        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with sweep_csv_path.open("a", encoding="utf-8") as fp:
            fp.write(
                f"{shift_name},{alpha:.8g},{achieved_state_w:.6f},{achieved_action_w:.6f},"
                f"{'trajectory_episodes' if trajectory_episodes else 'window_iid'}\n"
            )

        LOGGER.info(
            "[%s/%s] achieved_state_w=%.4f achieved_action_w=%s",
            policy_name, shift_name, achieved_state_w,
            f"{achieved_action_w:.4f}" if np.isfinite(achieved_action_w) else "nan"
        )

        summary[shift_name] = metrics

    return summary


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OOD datasets via biased starts + policy rollouts (alpha sweep).")
    p.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    p.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    p.add_argument("--shifts", type=str, default=str(PACKAGE_ROOT / "cfg" / "shifts.yaml"))
    p.add_argument(
        "--base",
        type=str,
        default=str((PACKAGE_ROOT / "../output/data").resolve()),
        help="Base root containing per-policy IID datasets (for counts + baseline_stats)",
    )
    p.add_argument(
        "--target",
        type=str,
        default=str((PACKAGE_ROOT / "../output/data/ood_manual").resolve()),
        help="Target root for generated OOD datasets",
    )
    p.add_argument("--ego", action="append", help="Specific ego policy config paths (defaults to all ego_policy*.yaml)")
    p.add_argument(
        "--trajectory_episodes",
        action="store_true",
        help=(
            "If set, generate OOD data as trajectory-like episodes: one biased start per episode, "
            "then a sliding rollout for the episode length. Default is independent windows."
        ),
    )
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    data_cfg = load_yaml(Path(args.data))
    opp_cfg = load_yaml(Path(args.opponent))
    shifts_cfg = load_yaml(Path(args.shifts))

    base = resolve_path(Path(args.base))
    target = resolve_path(Path(args.target))
    target.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict] = {}
    for idx, ego_cfg_path in enumerate(find_ego_configs(args.ego)):
        s = process_policy(
            ego_cfg_path=ego_cfg_path,
            data_config=data_cfg,
            shifts_cfg=shifts_cfg,
            base_root=base,
            target_root=target,
            opp_cfg=opp_cfg,
            seed_offset=idx + 1,
            trajectory_episodes=bool(args.trajectory_episodes),
        )
        summaries[ego_cfg_path.stem] = s

    LOGGER.info("Manual state generation complete:\n%s", json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
