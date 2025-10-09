"""Prepare window-hash opponent policies for each ego policy and shift."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"
EGO_PATTERN = "ego_policy*.yaml"

if str(REPO_ROOT) not in sys.path:  # local import guard
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.policies.oppPolicy import build_hash_policy
from policyOrProxy.core.world.arena import build_arena

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def find_ego_configs(explicit: Iterable[str] | None = None) -> List[Path]:
    if explicit:
        return [resolve_path(Path(p)) for p in explicit]
    configs = sorted(CFG_DIR.glob(EGO_PATTERN))
    if not configs:
        raise FileNotFoundError(f"No ego policy configs matching {EGO_PATTERN}")
    return configs


def apply_opponent_modifications(policy, spec: Dict) -> Dict[str, float]:
    """Apply optional prototype or noise adjustments described in the shift spec."""
    opp_spec = dict(spec.get("opponent", {}))
    applied: Dict[str, float] = {}

    prototypes = policy.get_prototype_table().copy()
    weights = None
    if policy.regionizer.prototype_weights is not None:
        weights = policy.regionizer.prototype_weights.copy()

    scale = float(opp_spec.get("action_scale", 1.0))
    if scale != 1.0:
        prototypes *= scale
        np.clip(prototypes, -policy.max_speed, policy.max_speed, out=prototypes)
        applied["action_scale"] = scale

    bias = opp_spec.get("action_bias")
    if bias is not None:
        bias_arr = np.asarray(bias, dtype=np.float32)
        if bias_arr.shape == (2,):
            prototypes += bias_arr
        elif bias_arr.shape == (policy.num_agents, 2):
            prototypes += bias_arr[np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported action_bias shape {bias_arr.shape}")
        np.clip(prototypes, -policy.max_speed, policy.max_speed, out=prototypes)
        applied["action_bias"] = float(np.linalg.norm(bias_arr))

    temp = opp_spec.get("weights_temperature")
    if temp is not None:
        if weights is None:
            weights = np.ones_like(prototypes[..., 0])
        temperature = float(temp)
        logits = weights / max(temperature, 1e-6)
        logits = logits - logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)
        applied["weights_temperature"] = temperature

    if "noise_std" in opp_spec:
        policy.noise_std = float(opp_spec["noise_std"])
        applied["noise_std"] = policy.noise_std

    if applied:
        policy.regionizer.register_prototypes(prototypes, weights)

    return applied


def save_policy_state(path: Path, policy, metadata: Dict) -> None:
    state = policy.export_state()
    payload = {"prototypes": np.asarray(state["prototypes"], dtype=np.float32)}
    if state.get("weights") is not None:
        payload["weights"] = np.asarray(state["weights"], dtype=np.float32)
    np.savez_compressed(path, **payload)
    with path.with_suffix(".json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def process_policy(
    cfg_path: Path,
    data_config: Dict,
    shift_config: Dict,
    opponent_config: Dict,
    opponents_base: Path,
) -> None:
    name = cfg_path.stem
    arena = build_arena(data_config["arena"])
    world_cfg = data_config["world"]
    out_dir = opponents_base / name
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("[%s] Preparing opponents in %s", name, out_dir)
    for idx, (shift_name, spec) in enumerate(shift_config["shifts"].items(), start=1):
        policy = build_hash_policy(
            arena=arena,
            world_cfg=world_cfg,
            opp_cfg=opponent_config,
            seed_offset=idx,
            identifier=f"{name}_{shift_name}",
        )
        applied = apply_opponent_modifications(policy, spec)
        metadata = {
            "seed_offset": idx,
            "identifier": policy.identifier,
            "noise_std": policy.noise_std,
            "applied": applied,
        }
        target_path = out_dir / f"{shift_name}.npz"
        save_policy_state(target_path, policy, metadata)
        LOGGER.info("[%s] Saved opponent state for %s at %s", name, shift_name, target_path)


def main(data_cfg: Path, opponent_cfg: Path, shift_cfg: Path, base: Path, ego_cfgs: List[str] | None) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opponent_config = load_yaml(opponent_cfg)
    shift_config = load_yaml(shift_cfg)
    opponents_base = resolve_path(base)
    opponents_base.mkdir(parents=True, exist_ok=True)

    for cfg_path in find_ego_configs(ego_cfgs):
        process_policy(cfg_path, data_config, shift_config, opponent_config, opponents_base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare window-hash opponents for each ego policy shift.")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy_hash.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--base", type=str, default=str((PACKAGE_ROOT / "../output/opponents").resolve()))
    parser.add_argument("--ego", action="append", help="Specific ego policy yaml(s) to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.opponent), Path(args.shift), Path(args.base), args.ego or [])
