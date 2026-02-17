#!/usr/bin/env python3
"""
plot_gif.py

Visualize an episode NPZ as a GIF of agents moving.

Compatible NPZ format:
  - windows: (N, T, teams, agents, F), where (x,y) are windows[..., :2]
  - ego_actions: (N, agents, 2)   (optional but expected in your datasets)
  - opponent_actions: (N, agents, 2) (optional but expected in your datasets)
  - policy_id (optional)

Animation source modes (pick via --source):

1) --source episode_last   (default)
   Uses the LAST frame of each window: windows[i, -1, ...] for i=0..N-1
   This produces a coherent episode animation for IID rollouts (world.rollout output),
   because consecutive windows correspond to consecutive timesteps.

2) --source window
   Picks a single window index (random or --window_idx) and animates its T frames:
   windows[window_idx, t, ...] for t=0..T-1

   NEW BEHAVIOR:
   - If actions are available, appends ONE additional "action frame" showing the
     next position predicted by applying the action following the window.
   - Also overlays an arrow (quiver) showing the action vector, anchored at the
     last observed position in the window.

Bounds / camera behavior:
- Default: tight bounds computed from the frames being animated.
- If --fixed_bounds is provided (or defaulted): use those bounds (useful to show the whole arena).
- If --data is provided: attempt to build the arena from data.yaml and extract arena bounds.

Usage examples:
  # IID, animate continuous episode
  python policyOrProxy/eval/plot_gif.py \
    --root output/data/ego_policy1/iid --split test --out output/viz/ego_policy1_iid.gif \
    --use_index --source episode_last

  # OOD-manual, animate a single sampled window + action step
  python policyOrProxy/eval/plot_gif.py \
    --root output/data/ood_manual/ego_policy1/right_bias_alpha_0p0 --split test \
    --out output/viz/ood_window_plus_action.gif --use_index --source window

  # Force full-arena view (example bounds)
  python policyOrProxy/eval/plot_gif.py \
    --root output/data/ood_manual/ego_policy1/right_bias_alpha_0p0 --split test \
    --out output/viz/full_arena.gif --use_index \
    --fixed_bounds 0 40 0 25 --source window
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from matplotlib.animation import PillowWriter
    _HAS_PILLOW = True
except Exception:
    _HAS_PILLOW = False

# Optional dependencies for --data / arena bounds
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    from policyOrProxy.core.world.arena import build_arena  # type: ignore
    _HAS_ARENA = True
except Exception:
    _HAS_ARENA = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate a random episode NPZ as a GIF.")
    p.add_argument("--root", type=str, default="output/data/ego_policy1/iid",
                   help="Dataset root (contains index.json and/or <split>/)")
    p.add_argument("--split", type=str, default="test", help="train|val|test")
    p.add_argument("--out", type=str, default="output/viz/ego_policy1_iid.gif", help="Output GIF path")
    p.add_argument("--seed", type=int, default=0, help="Random seed for picking episode/window")
    p.add_argument("--episode_id", type=int, default=-1,
                   help="Episode integer id (e.g., 0 for episode_00000). -1=random")
    p.add_argument("--window_idx", type=int, default=-1,
                   help="Window index within NPZ. -1=random (used for --source window)")
    p.add_argument("--pattern", type=str, default="episode_{:05d}.npz",
                   help="Filename pattern if not using index.json")
    p.add_argument("--num_candidates", type=int, default=2000,
                   help="Max episode ids to scan when not using index.json")
    p.add_argument("--use_index", action="store_true",
                   help="Use <root>/index.json to find episodes (recommended)")
    p.add_argument("--source", type=str, default="episode_last",
                   choices=["episode_last", "window"],
                   help="episode_last: animate windows[:, -1]; window: animate windows[window_idx, :]")

    p.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    p.add_argument("--marker_size", type=float, default=30.0, help="Scatter marker size")
    p.add_argument("--pad", type=float, default=0.05, help="Padding fraction around bounds")
    p.add_argument("--title", type=str, default="", help="Optional title override")

    # Fixed/full-arena options
    p.add_argument(
        "--fixed_bounds",
        type=float,
        nargs=4,
        default=(0, 40, 0, 25),  # set your arena bounds here (or pass via CLI)
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Override plot bounds (forces full arena view): XMIN XMAX YMIN YMAX",
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Optional path to data.yaml; if set, attempt to use arena bounds from config",
    )

    # NEW: action visualization controls (used for --source window)
    p.add_argument(
        "--dt",
        type=float,
        default=-1.0,
        help="Time step for converting action (vel) to displacement. If <0, tries to read from --data; else defaults to 1.0.",
    )
    p.add_argument(
        "--action_scale",
        type=float,
        default=0.5,
        help="Extra multiplier on action displacement for visualization (after dt).",
    )
    p.add_argument(
        "--show_action_arrow",
        action="store_true",
        help="If set, draw action arrow(s) for --source window. (Recommended.)",
    )
    p.add_argument(
        "--append_action_frame",
        action="store_true",
        help="If set, append an extra frame showing positions after applying the action following the window.",
    )

    return p.parse_args()


def load_index_episode_paths(root: Path, split: str) -> Optional[List[Path]]:
    idx_path = root / "index.json"
    if not idx_path.exists():
        return None
    try:
        obj = json.loads(idx_path.read_text(encoding="utf-8"))
        entries = obj.get("entries", obj.get("episodes", obj.get("data", None)))
        if entries is None:
            return None

        paths: List[Path] = []
        if isinstance(entries, list):
            for rec in entries:
                try:
                    if rec.get("split") != split:
                        continue
                    rel = rec.get("path")
                    if not rel:
                        continue
                    paths.append(root / rel)
                except Exception:
                    continue

        paths = sorted(paths, key=lambda p: p.as_posix())
        return paths if paths else None
    except Exception:
        return None


def discover_episode_paths(root: Path, split: str, pattern: str, num_candidates: int) -> List[Path]:
    split_dir = root / split
    if split_dir.exists():
        globs = sorted(split_dir.glob("episode_*.npz"), key=lambda p: p.name)
        if globs:
            return globs
    return [(split_dir / pattern.format(i)) for i in range(num_candidates) if (split_dir / pattern.format(i)).exists()]


def pick_episode_path(
    root: Path,
    split: str,
    rng: np.random.Generator,
    use_index: bool,
    pattern: str,
    num_candidates: int,
    episode_id: int,
) -> Path:
    if use_index:
        indexed = load_index_episode_paths(root, split)
        if indexed:
            if episode_id >= 0:
                target_name = pattern.format(episode_id)
                for p in indexed:
                    if p.name == target_name:
                        return p
                if episode_id < len(indexed):
                    return indexed[episode_id]
                raise FileNotFoundError(f"Episode id {episode_id} not found via index.json under {root} split={split}")
            return indexed[int(rng.integers(0, len(indexed)))]

    candidates = discover_episode_paths(root, split, pattern, num_candidates)
    if not candidates:
        raise FileNotFoundError(f"No episode NPZ files found under {root}/{split}")
    if episode_id >= 0:
        target = root / split / pattern.format(episode_id)
        if target.exists():
            return target
        if episode_id < len(candidates):
            return candidates[episode_id]
        raise FileNotFoundError(f"Episode id {episode_id} not found under {root}/{split}")
    return candidates[int(rng.integers(0, len(candidates)))]


def _load_yaml(path: Path) -> dict:
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not available, but --data was provided. Install pyyaml or use --fixed_bounds.")
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def try_get_arena_bounds(arena) -> Optional[Tuple[float, float, float, float]]:
    # Method: arena.bounds() -> (x0,x1,y0,y1)
    if hasattr(arena, "bounds") and callable(getattr(arena, "bounds")):
        try:
            b = arena.bounds()
            if b is not None and len(b) == 4:
                return tuple(float(v) for v in b)
        except Exception:
            pass

    # Attributes: x_min/x_max/y_min/y_max
    for names in [
        ("x_min", "x_max", "y_min", "y_max"),
        ("xmin", "xmax", "ymin", "ymax"),
    ]:
        if all(hasattr(arena, n) for n in names):
            try:
                return (
                    float(getattr(arena, names[0])),
                    float(getattr(arena, names[1])),
                    float(getattr(arena, names[2])),
                    float(getattr(arena, names[3])),
                )
            except Exception:
                pass

    # Width/height centered at origin
    if hasattr(arena, "width") and hasattr(arena, "height"):
        try:
            w = float(getattr(arena, "width"))
            h = float(getattr(arena, "height"))
            return (-w / 2.0, w / 2.0, -h / 2.0, h / 2.0)
        except Exception:
            pass

    # Half extents centered at origin
    if hasattr(arena, "half_width") and hasattr(arena, "half_height"):
        try:
            hw = float(getattr(arena, "half_width"))
            hh = float(getattr(arena, "half_height"))
            return (-hw, hw, -hh, hh)
        except Exception:
            pass

    return None


def compute_bounds(frames: np.ndarray, pad_frac: float) -> Tuple[float, float, float, float]:
    xy = frames.reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size == 0:
        return -1.0, 1.0, -1.0, 1.0
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    dx = float(x_max - x_min + 1e-6)
    dy = float(y_max - y_min + 1e-6)
    return (
        float(x_min - pad_frac * dx),
        float(x_max + pad_frac * dx),
        float(y_min - pad_frac * dy),
        float(y_max + pad_frac * dy),
    )


def apply_pad_to_bounds(x0: float, x1: float, y0: float, y1: float, pad_frac: float) -> Tuple[float, float, float, float]:
    dx = float((x1 - x0) + 1e-6)
    dy = float((y1 - y0) + 1e-6)
    return (
        float(x0 - pad_frac * dx),
        float(x1 + pad_frac * dx),
        float(y0 - pad_frac * dy),
        float(y1 + pad_frac * dy),
    )


def choose_plot_bounds(args: argparse.Namespace, frames: np.ndarray) -> Tuple[float, float, float, float]:
    pad_frac = float(args.pad)

    if args.fixed_bounds is not None:
        x0, x1, y0, y1 = (float(args.fixed_bounds[0]), float(args.fixed_bounds[1]),
                          float(args.fixed_bounds[2]), float(args.fixed_bounds[3]))
        return apply_pad_to_bounds(x0, x1, y0, y1, pad_frac=pad_frac)

    if str(args.data).strip():
        if not _HAS_ARENA:
            raise RuntimeError("Arena code not importable, but --data was provided. Use --fixed_bounds instead.")
        data_cfg = _load_yaml(Path(args.data))
        arena = build_arena(data_cfg["arena"])
        b = try_get_arena_bounds(arena)
        if b is not None:
            return apply_pad_to_bounds(*b, pad_frac=pad_frac)
        return compute_bounds(frames, pad_frac=pad_frac)

    return compute_bounds(frames, pad_frac=pad_frac)


def resolve_dt(args: argparse.Namespace) -> float:
    # Priority: explicit --dt >= 0, else read from --data, else default 1.0
    if float(args.dt) >= 0.0:
        return float(args.dt)

    if str(args.data).strip():
        try:
            cfg = _load_yaml(Path(args.data))
            dt = cfg.get("world", {}).get("dt", None)
            if dt is not None:
                return float(dt)
        except Exception:
            pass

    return 1.0


def extract_frames_and_actions_from_npz(
    npz_path: Path, rng: np.random.Generator, source: str, window_idx: int
) -> Tuple[np.ndarray, dict, Optional[np.ndarray]]:
    """
    Returns:
      frames: (F, teams, agents, 2) float32
      meta:   dict
      actions_next: Optional[(teams, agents, 2)] float32
        - Only populated for source == "window" if action arrays exist.
        - Convention: team 0 uses ego_actions, team 1 uses opponent_actions.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        if "windows" not in data:
            raise ValueError(f"{npz_path} missing 'windows' key; cannot animate.")
        W = np.asarray(data["windows"], dtype=np.float32)  # (N,T,teams,agents,F)
        if W.ndim != 5 or W.shape[-1] < 2:
            raise ValueError(f"{npz_path} 'windows' has unexpected shape {W.shape}")
        N, T, teams, agents, _F = W.shape

        policy_id = None
        if "policy_id" in data:
            try:
                policy_id = str(np.asarray(data["policy_id"]).item())
            except Exception:
                policy_id = None

        meta = {
            "npz": str(npz_path),
            "N": int(N),
            "T": int(T),
            "teams": int(teams),
            "agents": int(agents),
            "policy_id": policy_id,
        }

        if source == "episode_last":
            frames = W[:, -1, :, :, :2]  # (N,teams,agents,2)
            return frames.astype(np.float32), meta, None

        # source == "window"
        if window_idx < 0:
            window_idx = int(rng.integers(0, N))
        window_idx = int(np.clip(window_idx, 0, N - 1))
        meta["window_idx"] = int(window_idx)

        frames = W[window_idx, :, :, :, :2]  # (T,teams,agents,2)
        frames = frames.astype(np.float32)

        actions_next: Optional[np.ndarray] = None
        if "ego_actions" in data or "opponent_actions" in data:
            actions_next = np.zeros((teams, agents, 2), dtype=np.float32)

            if "ego_actions" in data:
                EA = np.asarray(data["ego_actions"], dtype=np.float32)  # (N,agents,2) or (L,agents,2)
                if EA.ndim == 3 and EA.shape[0] > window_idx:
                    actions_next[0] = EA[window_idx]
                    meta["has_ego_action"] = True
            if "opponent_actions" in data:
                OA = np.asarray(data["opponent_actions"], dtype=np.float32)
                if OA.ndim == 3 and OA.shape[0] > window_idx and teams >= 2:
                    actions_next[1] = OA[window_idx]
                    meta["has_opp_action"] = True

            # If we failed to populate anything meaningful, drop it.
            if not np.isfinite(actions_next).any():
                actions_next = None

        return frames, meta, actions_next


def clip_positions_to_bounds(pos: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    out = pos.copy()
    out[..., 0] = np.clip(out[..., 0], x0, x1)
    out[..., 1] = np.clip(out[..., 1], y0, y1)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    ep_path = pick_episode_path(
        root=root,
        split=args.split,
        rng=rng,
        use_index=bool(args.use_index),
        pattern=args.pattern,
        num_candidates=int(args.num_candidates),
        episode_id=int(args.episode_id),
    )

    frames, meta, actions_next = extract_frames_and_actions_from_npz(
        npz_path=ep_path,
        rng=rng,
        source=str(args.source),
        window_idx=int(args.window_idx),
    )

    # Determine plot bounds
    x0, x1, y0, y1 = choose_plot_bounds(args, frames)

    # If window mode: optionally append an "action frame" (predicted next position)
    base_T = None
    action_disp = None  # (teams,agents,2), displacement used for arrow + action frame
    if args.source == "window":
        base_T = int(frames.shape[0])
        dt = resolve_dt(args)
        meta["dt_used"] = float(dt)

        if actions_next is not None:
            # Convert action (vel) to displacement for visualization
            action_disp = (actions_next.astype(np.float32) * float(dt) * float(args.action_scale)).astype(np.float32)
            meta["action_scale"] = float(args.action_scale)

            if bool(args.append_action_frame):
                last_pos = frames[-1]  # (teams,agents,2)
                next_pos = last_pos + action_disp
                next_pos = clip_positions_to_bounds(next_pos, x0, x1, y0, y1)
                frames = np.concatenate([frames, next_pos[None, ...]], axis=0)
                meta["action_frame_appended"] = True
        else:
            meta["action_frame_appended"] = False

    # ---- build figure ----
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=int(args.dpi))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    teams = int(meta["teams"])
    agents = int(meta["agents"])

    # One scatter per team
    scatters = []
    for team in range(teams):
        xy0 = frames[0, team].reshape(agents, 2)
        sc = ax.scatter(xy0[:, 0], xy0[:, 1], s=float(args.marker_size), label=f"team{team}")
        scatters.append(sc)

    ax.legend(loc="upper right", frameon=True)

    # Optional quiver (action arrows) for window mode
    quivers = None
    if args.source == "window" and bool(args.show_action_arrow) and action_disp is not None:
        quivers = []
        # Anchor at last observed window position (not the action frame position)
        anchor_pos = frames[base_T - 1] if base_T is not None and base_T > 0 else frames[0]
        for team in range(teams):
            xy = anchor_pos[team].reshape(agents, 2)
            uv = action_disp[team].reshape(agents, 2)
            q = ax.quiver(
                xy[:, 0], xy[:, 1],
                uv[:, 0], uv[:, 1],
                angles="xy", scale_units="xy", scale=1.0,
                width=0.004
            )
            # Start hidden; show near the end of the window
            q.set_alpha(0.0)
            quivers.append(q)

    # Title
    if args.title.strip():
        title = args.title.strip()
    else:
        pid = meta.get("policy_id") or ""
        mode = args.source
        extra = ""
        if mode == "window" and "window_idx" in meta:
            extra = f", window_idx={meta['window_idx']}"
        title = f"{pid} {args.split} — {mode}{extra}\n{ep_path.name}"
    ttl = ax.set_title(title)

    # Frame counter + action text
    txt = ax.text(0.01, 0.01, "", transform=ax.transAxes, ha="left", va="bottom")
    a_txt = ax.text(0.01, 0.05, "", transform=ax.transAxes, ha="left", va="bottom")

    def format_action_line(actions: np.ndarray) -> str:
        # actions: (teams,agents,2)
        parts = []
        for t in range(actions.shape[0]):
            # For readability: if one agent, show one vector; else show mean vector
            if actions.shape[1] == 1:
                v = actions[t, 0]
                parts.append(f"team{t} a=({v[0]:+.3f},{v[1]:+.3f})")
            else:
                m = actions[t].mean(axis=0)
                parts.append(f"team{t} a_mean=({m[0]:+.3f},{m[1]:+.3f})")
        return "  ".join(parts)

    def update(frame_i: int):
        fr = frames[frame_i]  # (teams,agents,2)
        for team in range(teams):
            xy = fr[team].reshape(agents, 2)
            scatters[team].set_offsets(xy)

        # Frame label logic
        if args.source == "window" and base_T is not None:
            if frame_i < base_T:
                txt.set_text(f"window frame {frame_i+1}/{base_T}")
            else:
                txt.set_text("action frame (predicted next position)")
        else:
            txt.set_text(f"frame {frame_i+1}/{frames.shape[0]}")

        # Action overlay (window mode only)
        if args.source == "window" and base_T is not None and action_disp is not None:
            # Show arrow(s) on the last observed window frame, and on the appended action frame if present
            show = (frame_i >= base_T - 1)
            if quivers is not None:
                anchor_pos = frames[base_T - 1]  # anchor at last observed position
                for team in range(teams):
                    xy = anchor_pos[team].reshape(agents, 2)
                    uv = action_disp[team].reshape(agents, 2)
                    quivers[team].set_offsets(xy)
                    quivers[team].set_UVC(uv[:, 0], uv[:, 1])
                    quivers[team].set_alpha(1.0 if show else 0.0)

            # if show:
            #     a_txt.set_text(
            #         f"dt={meta.get('dt_used', 1.0):g}, scale={meta.get('action_scale', 1.0):g}  "
            #         + format_action_line(action_disp)
            #     )
            # else:
            a_txt.set_text("")
        else:
            a_txt.set_text("")

        artists = [*scatters, txt, a_txt, ttl]
        if quivers is not None:
            artists.extend(quivers)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=int(frames.shape[0]),
        interval=1000.0 / max(1, int(args.fps)),
        blit=True,
    )

    if not _HAS_PILLOW:
        raise RuntimeError(
            "PillowWriter unavailable. Install pillow or use a matplotlib build with Pillow support. "
            "Common fix: pip install pillow"
        )

    writer = PillowWriter(fps=int(args.fps))
    anim.save(out.as_posix(), writer=writer)
    plt.close(fig)

    print("[OK] Saved GIF ->", out)
    print("[INFO] Episode:", ep_path)
    print("[INFO] Meta:", meta)
    print("[INFO] Bounds:", (x0, x1, y0, y1))

    if args.source == "window":
        if action_disp is None:
            print("[INFO] No action vectors found in NPZ (ego_actions/opponent_actions missing or incompatible).")
        else:
            print("[INFO] Action overlay enabled:", bool(args.show_action_arrow))
            print("[INFO] Action frame appended:", bool(args.append_action_frame))


if __name__ == "__main__":
    main()
