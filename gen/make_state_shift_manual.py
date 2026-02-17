#!/usr/bin/env python3
"""
make_state_shift_manual.py

Create OOD datasets by *manually generating* all-agent states (positions/velocities),
then evaluating ego/opponent policies on those windows to get actions.

NEW: Each sampled state is repeated across the window length *with noise* so frames
aren't identical. Noise can be i.i.d. or AR(1)-correlated across time.

See "Generators YAML example" at the end for per-shift noise settings.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
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
# Manual state generators (positions -> state)
# -----------------------

def clamp_positions(arena, xy: np.ndarray) -> np.ndarray:
    return arena.clamp_positions(xy)

def make_state_from_xy(xy: np.ndarray, vel_mode: str = "zero", vel_std: float = 0.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    xy: [teams, agents, 2] -> state [teams, agents, 4] as [x,y,vx,vy]
    """
    teams, agents, _ = xy.shape
    state = np.zeros((teams, agents, 4), dtype=np.float32)
    state[..., :2] = xy.astype(np.float32)
    if vel_mode == "gaussian":
        assert rng is not None
        state[..., 2:] = rng.normal(0.0, vel_std, size=(teams, agents, 2)).astype(np.float32)
    return state

def gen_gaussian_shift(arena, teams, agents, scale, params, rng):
    arena_scale = min(arena.width, arena.height)
    # center can be "arena" or explicit [x, y]
    if params.get("center", "arena") == "arena":
        center = np.array([arena.width/2.0, arena.height/2.0], dtype=np.float32)
    else:
        center = np.array(params["center"], dtype=np.float32)

    mean_dir = np.array(params.get("mean_dir", [1.0, 0.0]), dtype=np.float32)
    mean_dir /= (np.linalg.norm(mean_dir) or 1.0)

    base_sigma = float(params.get("base_sigma", 0.05))
    if params.get("base_sigma_relative", True):     # NEW: make relative by default
        base_sigma = base_sigma * arena_scale

    # Shift around center
    mus = center + np.stack([+scale*mean_dir, -scale*mean_dir], axis=0)[:teams]

    xy = rng.normal(loc=mus[:, None, :], scale=base_sigma, size=(teams, agents, 2)).astype(np.float32)
    xy = clamp_positions(arena, xy)
    return make_state_from_xy(xy, vel_mode=params.get("vel_mode", "zero"),
                              vel_std=float(params.get("vel_std", 0.0)), rng=rng)

def gen_ring(arena, teams: int, agents: int, scale: float, params: Dict, rng: np.random.Generator) -> np.ndarray:
    cx, cy = arena.width/2.0, arena.height/2.0
    theta = rng.uniform(0, 2*np.pi, size=(teams, agents))
    r = scale + rng.normal(0.0, float(params.get("radial_noise", 0.02*min(arena.width, arena.height))), size=(teams, agents))
    xy = np.zeros((teams, agents, 2), dtype=np.float32)
    xy[..., 0] = cx + r * np.cos(theta)
    xy[..., 1] = cy + r * np.sin(theta)
    xy = clamp_positions(arena, xy)
    return make_state_from_xy(xy, vel_mode=params.get("vel_mode", "zero"),
                              vel_std=float(params.get("vel_std", 0.0)), rng=rng)

def gen_grid_jitter(arena, teams: int, agents: int, scale: float, params: Dict, rng: np.random.Generator) -> np.ndarray:
    rows = int(params.get("rows", max(1, int(np.sqrt(agents)))))
    cols = int(params.get("cols", max(1, int(np.ceil(agents / rows)))))
    xs = np.linspace(0.1*arena.width, 0.9*arena.width, cols, dtype=np.float32)
    ys = np.linspace(0.1*arena.height, 0.9*arena.height, rows, dtype=np.float32)
    base = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)[:agents]
    xy = np.zeros((teams, agents, 2), dtype=np.float32)
    for k in range(teams):
        jitter = rng.uniform(low=-scale, high=scale, size=(agents, 2)).astype(np.float32)
        xy[k] = base + jitter
    xy = clamp_positions(arena, xy)
    return make_state_from_xy(xy, vel_mode=params.get("vel_mode", "zero"),
                              vel_std=float(params.get("vel_std", 0.0)), rng=rng)

def gen_corners(arena, teams: int, agents: int, scale: float, params: Dict, rng: np.random.Generator) -> np.ndarray:
    corners = np.array([
        [0.05*arena.width, 0.95*arena.height],
        [0.95*arena.width, 0.05*arena.height],
        [0.05*arena.width, 0.05*arena.height],
        [0.95*arena.width, 0.95*arena.height]
    ], dtype=np.float32)
    xy = np.zeros((teams, agents, 2), dtype=np.float32)
    for k in range(teams):
        target = corners[k % len(corners)]
        jitter = rng.normal(0.0, 0.02*min(arena.width, arena.height) + 0.25*scale, size=(agents, 2)).astype(np.float32)
        xy[k] = target + jitter
    xy = clamp_positions(arena, xy)
    return make_state_from_xy(xy, vel_mode=params.get("vel_mode", "zero"),
                              vel_std=float(params.get("vel_std", 0.0)), rng=rng)

GENERATOR_REGISTRY: Dict[str, Callable] = {
    "gaussian_shift": gen_gaussian_shift,
    "ring": gen_ring,
    "grid_jitter": gen_grid_jitter,
    "corners": gen_corners,
}

# -----------------------
# Window construction with noise
# -----------------------

def build_noisy_window(
    arena,
    base_state: np.ndarray,        # [teams, agents, F], F>=2, first 2 are x,y; if F>=4, next 2 are vx,vy
    window_len: int,
    rng: np.random.Generator,
    pos_std: float,
    vel_std: float,
    rho: float = 0.0,              # AR(1) temporal correlation (0=iid noise)
    clamp: bool = True,
) -> np.ndarray:
    """
    Repeat base_state across T with additive Gaussian noise.
    - Position noise: N(0, pos_std^2) per step; if rho>0, use AR(1): eps_t = rho*eps_{t-1}+sqrt(1-rho^2)*z_t
    - Velocity noise (if dims exist): N(0, vel_std^2) similarly.
    """
    teams, agents, F = base_state.shape
    T = int(window_len)
    window = np.repeat(base_state[None, ...], T, axis=0).astype(np.float32)

    if pos_std <= 0.0 and (vel_std <= 0.0 or F < 4):
        # nothing to do
        return window

    # Initialize AR(1) noise state
    pos_eps = np.zeros((teams, agents, 2), dtype=np.float32)
    vel_eps = np.zeros((teams, agents, 2), dtype=np.float32) if (vel_std > 0.0 and F >= 4) else None
    ar_scale = np.sqrt(max(0.0, 1.0 - rho * rho))

    for t in range(T):
        # position noise
        z_pos = rng.normal(0.0, pos_std, size=(teams, agents, 2)).astype(np.float32)
        pos_eps = rho * pos_eps + ar_scale * z_pos
        window[t, :, :, :2] = window[t, :, :, :2] + pos_eps

        # velocity noise (if present)
        if vel_eps is not None:
            z_vel = rng.normal(0.0, vel_std, size=(teams, agents, 2)).astype(np.float32)
            vel_eps = rho * vel_eps + ar_scale * z_vel
            window[t, :, :, 2:4] = window[t, :, :, 2:4] + vel_eps

        if clamp:
            window[t, :, :, :2] = clamp_positions(arena, window[t, :, :, :2])

    return window

def last_frame(state_window: np.ndarray) -> np.ndarray:
    return state_window[-1]

def flatten_state_frame(frame: np.ndarray) -> np.ndarray:
    return frame.reshape(-1).astype(np.float32)

# -----------------------
# State-W targeting
# -----------------------

def binary_search_scale_for_state_w(
    target_w: float,
    sample_count: int,
    make_state_fn: Callable[[float], np.ndarray],  # returns [teams, agents, 4]
    baseline_states_flat: np.ndarray,
    max_scale: float,
) -> float:
    if target_w <= 0:
        return 0.0
    lo, hi = 0.0, float(max_scale)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        frames = []
        for _i in range(min(sample_count, 2048)):
            st = make_state_fn(mid)                # [teams, agents, 4]
            frm = flatten_state_frame(st)          # compare last frame stats only
            frames.append(frm)
        frames = np.stack(frames, axis=0)
        dist = wasserstein_distance_numpy(frames, baseline_states_flat)
        if dist < target_w:
            lo = mid
        else:
            hi = mid
    return hi

# -----------------------
# Policy eval helpers
# -----------------------

def compute_actions_ego(ego: WindowHashPolicy, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
    return ego.act(window, deterministic=deterministic).astype(np.float32)

def compute_actions_opp(opp: WindowHashPolicy, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
    return opp.act(window, deterministic=deterministic).astype(np.float32)

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

    baseline_stats_path = policy_source / "baseline_stats.npz"
    if not baseline_stats_path.exists():
        raise FileNotFoundError(f"Missing baseline stats at {baseline_stats_path}")
    baseline_stats = np.load(baseline_stats_path, allow_pickle=False)
    baseline_states = baseline_stats["states"]
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
    first_split = splits[0]
    first_records = list(indexer_src.iter_split(first_split))
    if not first_records:
        raise ValueError(f"No episodes available in split {first_split} for policy {name}")
    sample_episode_path = policy_source / first_records[0].path
    with np.load(sample_episode_path, allow_pickle=False) as sample_episode:
        sample_windows = sample_episode["windows"]
    if sample_windows.ndim != 5:
        raise ValueError(f"Expected windows with 5 dims, got {sample_windows.shape}")
    window_len = int(sample_windows.shape[1])
    state_dim = int(sample_windows.shape[-1])

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

        # window noise config (defaults are small but non-zero)
        wn = spec.get("window_noise", {})
        # positional noise can be given absolute (arena units) or relative to arena size
        arena_scale = min(arena.width, arena.height)
        pos_std = float(wn.get("pos_std", 0.01 * arena_scale))
        if wn.get("pos_std_relative", False):
            pos_std = float(wn.get("pos_std", 0.01)) * arena_scale
        vel_std = float(wn.get("vel_std", 0.0))
        rho = float(wn.get("rho", 0.0))  # AR(1)

        if gen_type not in GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator type '{gen_type}'. Available: {list(GENERATOR_REGISTRY)}")
        gen_fn = GENERATOR_REGISTRY[gen_type]

        def make_state(scale: float) -> np.ndarray:
            st4 = gen_fn(arena, teams, agents, scale, gen_params, rng)  # [teams,agents,4]
            # pad/trim to state_dim
            if state_dim > 4:
                pad = np.zeros((teams, agents, state_dim), dtype=np.float32)
                pad[..., :4] = st4
                return pad
            elif state_dim < 4:
                return st4[..., :state_dim]
            return st4

        # find scale to match target state Wasserstein (using last-frame state vectors)
        max_scale = 0.75 * arena_scale
        scale = binary_search_scale_for_state_w(
            target_w=target_state_w,
            sample_count=min(len(baseline_states), 4096),
            make_state_fn=make_state,
            baseline_states_flat=baseline_states,
            max_scale=max_scale,
        )
        LOGGER.info("[%s/%s] scale=%.4f for target state W≈%.3f; pos_std=%.4g vel_std=%.4g rho=%.2f",
                    name, shift_name, scale, target_state_w, pos_std, vel_std, rho)

        new_indexer = EpisodeIndexer(root=policy_target / shift_name)
        all_states_last = []
        all_actions_ego = []

        for split in splits:
            records = list(indexer_src.iter_split(split))
            split_dir = (policy_target / shift_name / split)
            split_dir.mkdir(parents=True, exist_ok=True)

            for episode_idx, record in enumerate(records):
                src_path = policy_source / record.path
                with np.load(src_path, allow_pickle=False) as src:
                    raw_windows = src["windows"]
                    if raw_windows.ndim < 3:
                        raise ValueError(f"Expected windowed input in baseline to infer counts; got {raw_windows.shape}")
                    N_prime = raw_windows.shape[0]

                windows = np.zeros((N_prime, window_len, teams, agents, state_dim), dtype=np.float32)
                ego_actions = np.zeros((N_prime, agents, 2), dtype=np.float32)
                opp_actions = np.zeros((N_prime, agents, 2), dtype=np.float32)
                positions_last = np.zeros((N_prime, teams, agents, 2), dtype=np.float32)

                for i in range(N_prime):
                    base_state = make_state(scale)  # [teams,agents,F]
                    win = build_noisy_window(
                                    arena=arena,
                                    base_state=base_state,
                                    window_len=window_len,
                                    rng=rng,
                        pos_std=pos_std,
                        vel_std=vel_std,
                        rho=rho,
                        clamp=True,
                    )  # [T,teams,agents,F]

                    ego_actions[i] = compute_actions_ego(ego_policy, win, deterministic=False)
                    opp_actions[i] = compute_actions_opp(opp_policy, win, deterministic=False)
                    windows[i] = win
                    positions_last[i] = last_frame(win)[..., :2]

                out_path = split_dir / f"episode_{episode_idx:05d}.npz"
                np.savez_compressed(
                    out_path,
                    windows=windows,
                    ego_actions=ego_actions,
                    opponent_actions=opp_actions,
                    positions=positions_last,
                    policy_id=f"{name}_{shift_name}"
                )
                new_indexer.add_episode(split, out_path, length=N_prime, policy_id=f"{name}_{shift_name}")

                all_states_last.append(windows[:, -1].reshape(N_prime, -1))
                all_actions_ego.append(ego_actions.reshape(N_prime, -1))

        new_indexer.save()

        states_concat = np.concatenate(all_states_last, axis=0)
        actions_concat = np.concatenate(all_actions_ego, axis=0)
        sample_size = min(5000, states_concat.shape[0])
        sel = np.random.default_rng(1234).choice(states_concat.shape[0], size=sample_size, replace=False)
        np.savez_compressed(policy_target / shift_name / "baseline_stats.npz",
                            states=states_concat[sel], actions=actions_concat[sel])

        achieved_state = float(wasserstein_distance_numpy(states_concat, baseline_states))
        achieved_action = float(wasserstein_distance_numpy(actions_concat, baseline_actions))

        summary[shift_name] = {
            "target_state": target_state_w,
            "target_action": target_action_w,
            "achieved_state": achieved_state,
            "achieved_action": achieved_action,
            "scale": float(scale),
            "pos_std": float(pos_std),
            "vel_std": float(vel_std),
            "rho": float(rho),
        }
        LOGGER.info("[%s/%s] achieved W: state=%.3f action=%.3f", name, shift_name, achieved_state, achieved_action)

    with (policy_target / "manual_shift_report.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return summary

# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manually generate OOD state distributions (noisy repeated windows) and evaluate policies.")
    p.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    p.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    p.add_argument("--generators", type=str, default=str(PACKAGE_ROOT / "cfg" / "generators.yaml"),
                   help="YAML specifying shifts, targets, generator types/params, and window_noise")
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
