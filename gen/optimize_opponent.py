"""Tune opponents for each ego policy to hit shift targets."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
CFG_DIR = PACKAGE_ROOT / "cfg"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.core.metrics.metrics import wasserstein_distance_numpy
from policyOrProxy.core.policies.egoPolicy import WindowHashPolicy
from policyOrProxy.core.policies.oppPolicy import DivergenceLoss, WindowNN, optimize_to_target
from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.core.world.world import build_world
from policyOrProxy.models.collate import move_batch, next_frame_collate

LOGGER = logging.getLogger(__name__)


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


def build_ego_policy(arena, world_cfg: Dict, ego_cfg: Dict, rng: np.random.Generator) -> WindowHashPolicy:
    regionizer = WindowHashRegionizer(
        arena=arena,
        num_buckets=int(ego_cfg["num_buckets"]),
        grid_size=int(ego_cfg["quantization"]["grid_size"]),
        length_scale=float(ego_cfg["quantization"].get("length_scale", 1.0)),
        jitter=float(ego_cfg["quantization"].get("jitter", 0.0)),
    )
    return WindowHashPolicy(
        regionizer=regionizer,
        num_agents=int(world_cfg["agents_per_team"]),
        num_prototypes=int(ego_cfg["num_prototypes"]),
        max_speed=float(ego_cfg["prototype_init"].get("max_speed", world_cfg["max_speed"])),
        noise_std=float(ego_cfg.get("noise_std", 0.0)),
        sampling=ego_cfg.get("sampling", "stochastic"),
        seed=int(ego_cfg["prototype_init"].get("seed", 0)),
    )


def iter_data_roots(base: Path, policy_name: str) -> Path:
    policy_root = base / policy_name
    iid_root = policy_root / "iid"
    if not iid_root.exists():
        raise FileNotFoundError(f"Missing IID dataset for {policy_name} at {iid_root}")
    return iid_root


def process_policy(
    policy_cfg: Path,
    data_cfg: Dict,
    shift_cfg: Dict,
    opponent_cfg: Dict,
    base_output: Path,
    opponents_base: Path,
) -> None:
    name = policy_cfg.stem
    rng = np.random.default_rng(11)
    iid_root = iter_data_roots(base_output, name)
    indexer = EpisodeIndexer.load(iid_root)
    train_dataset = NextFrameDataset(iid_root, indexer, split="train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        train_dataset,
        batch_size=int(opponent_cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=next_frame_collate,
    )

    def loader_iter():
        while True:
            for batch in dataloader:
                yield move_batch(batch, device)

    iterator = loader_iter()
    baseline_stats = np.load(iid_root / "baseline_stats.npz", allow_pickle=False)
    baseline_states_np = baseline_stats["states"]
    baseline_actions_np = baseline_stats["actions"]
    baseline_states = torch.from_numpy(baseline_states_np).float().to(device)
    baseline_actions = torch.from_numpy(baseline_actions_np).float().to(device)
    divergence = DivergenceLoss(baseline_states, baseline_actions)

    save_root = opponents_base / name
    save_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    arena = build_arena(data_cfg["arena"])
    world_cfg = data_cfg["world"]
    ego_cfg = load_yaml(policy_cfg)

    training_cfg = opponent_cfg.get("training", {})
    eval_every = int(training_cfg.get("eval_every", 200))
    eval_episodes = int(training_cfg.get("eval_episodes", 6))
    eval_steps = int(training_cfg.get("eval_steps", data_cfg["rollout"]["steps"]))
    eval_seed = int(training_cfg.get("eval_seed", 314159))

    def build_ego_policy_instance() -> WindowHashPolicy:
        return build_ego_policy(arena, world_cfg, ego_cfg, rng)

    class TorchOpponentWrapper:
        def __init__(self, net: WindowNN, device: torch.device) -> None:
            self.net = net
            self.device = device

        def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:  # noqa: D401
            with torch.no_grad():
                tensor = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                actions = self.net(tensor)
            return actions.squeeze(0).detach().cpu().numpy()

    def closed_loop_evaluator(model: torch.nn.Module, shift_name: str) -> Dict[str, float]:
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        was_training = base_model.training
        base_model.eval()
        opponent_wrapper = TorchOpponentWrapper(base_model, device)
        ego_policy = build_ego_policy_instance()
        world_rng = np.random.default_rng(eval_seed)
        world = build_world(arena, world_cfg, rng=world_rng)
        states = []
        actions = []
        for episode_idx in range(eval_episodes):
            world.reset()
            rollout = world.rollout(
                ego_policy,
                opponent_wrapper,
                steps=eval_steps,
                deterministic=False,
                policy_id=f"{name}_{shift_name}",
            )
            windows = rollout["windows"]
            states.append(windows[:, -1].reshape(windows.shape[0], -1))
            actions.append(rollout["ego_actions"].reshape(rollout["ego_actions"].shape[0], -1))
        if was_training:
            base_model.train()
        state_concat = np.concatenate(states, axis=0)
        action_concat = np.concatenate(actions, axis=0)
        return {
            "state": wasserstein_distance_numpy(state_concat, baseline_states_np),
            "action": wasserstein_distance_numpy(action_concat, baseline_actions_np),
        }

    for shift_name, target in shift_cfg["shifts"].items():
        LOGGER.info("[%s] Optimizing opponent for shift %s", name, shift_name)
        model = WindowNN(
            window_len=int(opponent_cfg["window_len"]),
            teams=int(world_cfg["teams"]),
            agents=int(world_cfg["agents_per_team"]),
            state_dim=int(opponent_cfg.get("state_dim", 4)),
            hidden_dim=int(opponent_cfg.get("hidden_dim", 256)),
            layers=int(opponent_cfg.get("layers", 3)),
            heads=int(opponent_cfg.get("heads", 4)),
            dropout=float(opponent_cfg.get("dropout", 0.1)),
            arch=opponent_cfg.get("arch", "mlp"),
            activation=opponent_cfg.get("activation", "gelu"),
            max_speed=float(opponent_cfg.get("max_speed", world_cfg["max_speed"])),
            identifier=f"{name}_{shift_name}",
        ).to(device)
        if torch.cuda.device_count() > 1:
            LOGGER.info("Using DataParallel for opponent tuning")
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(opponent_cfg["optimizer"]["lr"]),
            betas=tuple(opponent_cfg["optimizer"].get("betas", [0.9, 0.999])),
            weight_decay=float(opponent_cfg["optimizer"].get("weight_decay", 0.0)),
        )
        target_dict = {
            "state": float(target["state_wasserstein"]),
            "action": float(target["action_wasserstein"]),
        }

        def evaluate_model(current_model: torch.nn.Module) -> Dict[str, float]:
            return closed_loop_evaluator(current_model, shift_name)

        achieved = optimize_to_target(
            model,
            iterator,
            divergence,
            target_dict,
            optimizer,
            max_steps=int(opponent_cfg["training"]["steps"]),
            log_every=int(opponent_cfg["training"].get("log_every", 50)),
            tol=0.05,
            closed_loop_eval=evaluate_model,
            eval_every=eval_every,
        )
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, save_root / f"{shift_name}.pt")
        summary[shift_name] = {"targets": target_dict, **achieved}

    with (save_root / "sa_divergences.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Opponent tuning complete for %s", name)


def main(data_cfg: Path, opponent_cfg: Path, shift_cfg: Path, base: Path, ego_cfgs: List[str] | None) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opponent_config = load_yaml(opponent_cfg)
    shift_config = load_yaml(shift_cfg)
    base_output = resolve_path(base)
    opponents_base = resolve_path(Path(data_config.get("opponents_root", "output/opponents")))
    opponents_base.mkdir(parents=True, exist_ok=True)

    for cfg_path in find_ego_configs(ego_cfgs):
        process_policy(cfg_path, data_config, shift_config, opponent_config, base_output, opponents_base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize opponent policies for shift targets")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent_cfg", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--base", type=str, default=str((PACKAGE_ROOT / "../output/data").resolve()))
    parser.add_argument("--ego", action="append", help="Specific ego policy configs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.opponent_cfg), Path(args.shift), Path(args.base), args.ego)
