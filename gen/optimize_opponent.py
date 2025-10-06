"""Tune opponent networks to reach desired Wasserstein shifts."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)

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


def load_yaml(path: Path) -> Dict:
    with resolve_path(path).open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main(data_cfg: Path, opponent_cfg: Path, shift_cfg: Path, ego_cfg: Path) -> None:
    configure_logging()
    data_config = load_yaml(data_cfg)
    opp_config = load_yaml(opponent_cfg)
    shift_config = load_yaml(shift_cfg)
    ego_config = load_yaml(ego_cfg)
    output_root = resolve_path(Path(data_config.get("output_root", "output/data/iid")))
    data_root = output_root
    indexer = EpisodeIndexer.load(data_root)
    train_dataset = NextFrameDataset(data_root, indexer, split="train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(
        train_dataset,
        batch_size=int(opp_config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=next_frame_collate,
    )

    def loader_iter():
        while True:
            for batch in dataloader:
                yield move_batch(batch, device)

    iterator = loader_iter()
    baseline_stats = np.load(data_root / "baseline_stats.npz", allow_pickle=False)
    baseline_states_np = baseline_stats["states"]
    baseline_actions_np = baseline_stats["actions"]
    baseline_states = torch.from_numpy(baseline_states_np).float().to(device)
    baseline_actions = torch.from_numpy(baseline_actions_np).float().to(device)
    divergence = DivergenceLoss(baseline_states, baseline_actions)
    save_root = output_root.parent / "opponents"
    save_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    training_cfg = opp_config.get("training", {})
    eval_every = int(training_cfg.get("eval_every", 200))
    eval_episodes = int(training_cfg.get("eval_episodes", 6))
    eval_steps = int(training_cfg.get("eval_steps", data_config["rollout"]["steps"]))
    eval_seed = int(training_cfg.get("eval_seed", 314159))

    arena = build_arena(data_config["arena"])
    world_cfg = data_config["world"]

    def build_ego_policy_instance() -> WindowHashPolicy:
        regionizer = WindowHashRegionizer(
            arena=arena,
            num_buckets=int(ego_config["num_buckets"]),
            grid_size=int(ego_config["quantization"]["grid_size"]),
            length_scale=float(ego_config["quantization"].get("length_scale", 1.0)),
            jitter=float(ego_config["quantization"].get("jitter", 0.0)),
        )
        return WindowHashPolicy(
            regionizer=regionizer,
            num_agents=int(world_cfg["agents_per_team"]),
            num_prototypes=int(ego_config["num_prototypes"]),
            max_speed=float(ego_config["prototype_init"].get("max_speed", world_cfg["max_speed"])),
            noise_std=float(ego_config.get("noise_std", 0.0)),
            sampling=ego_config.get("sampling", "stochastic"),
            seed=int(ego_config["prototype_init"].get("seed", 0)),
        )

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
                policy_id=shift_name,
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
    for name, target in shift_config["shifts"].items():
        LOGGER.info("Optimizing opponent for shift %s", name)
        model = WindowNN(
            window_len=int(opp_config["window_len"]),
            teams=int(data_config["world"]["teams"]),
            agents=int(data_config["world"]["agents_per_team"]),
            state_dim=int(opp_config.get("state_dim", 4)),
            hidden_dim=int(opp_config.get("hidden_dim", 256)),
            layers=int(opp_config.get("layers", 3)),
            heads=int(opp_config.get("heads", 4)),
            dropout=float(opp_config.get("dropout", 0.1)),
            arch=opp_config.get("arch", "mlp"),
            activation=opp_config.get("activation", "gelu"),
            max_speed=float(opp_config.get("max_speed", data_config["world"]["max_speed"])),
            identifier=name,
        ).to(device)
        if torch.cuda.device_count() > 1:
            LOGGER.info("Using DataParallel for opponent tuning")
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(opp_config["optimizer"]["lr"]),
            betas=tuple(opp_config["optimizer"].get("betas", [0.9, 0.999])),
            weight_decay=float(opp_config["optimizer"].get("weight_decay", 0.0)),
        )
        target_dict = {
            "state": float(target["state_wasserstein"]),
            "action": float(target["action_wasserstein"]),
        }
        def evaluate_model(current_model: torch.nn.Module) -> Dict[str, float]:
            return closed_loop_evaluator(current_model, name)

        achieved = optimize_to_target(
            model,
            iterator,
            divergence,
            target_dict,
            optimizer,
            max_steps=int(opp_config["training"]["steps"]),
            log_every=int(opp_config["training"].get("log_every", 50)),
            tol=0.05,
            closed_loop_eval=evaluate_model,
            eval_every=eval_every,
        )
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, save_root / f"{name}.pt")
        summary[name] = {"targets": target_dict, **achieved}
    with (save_root / "sa_divergences.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Opponent tuning complete. Results saved to %s", save_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize opponent policies for shift targets")
    parser.add_argument("--data", type=str, default=str(PACKAGE_ROOT / "cfg" / "data.yaml"))
    parser.add_argument("--opponent", type=str, default=str(PACKAGE_ROOT / "cfg" / "opponent_policy.yaml"))
    parser.add_argument("--shift", type=str, default=str(PACKAGE_ROOT / "cfg" / "shift.yaml"))
    parser.add_argument("--ego", type=str, default=str(PACKAGE_ROOT / "cfg" / "ego_policy.yaml"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.opponent), Path(args.shift), Path(args.ego))
