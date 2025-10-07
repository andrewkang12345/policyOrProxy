"""Evaluate trained models on IID and OOD splits."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.core.metrics.metrics import (
    action_smoothness,
    ade_fde,
    collision_rate,
    representation_invariance_score,
)
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.models.collate import move_batch, next_frame_collate
from policyOrProxy.models.train_global_cvae import GlobalCVAE
from policyOrProxy.models.train_hier_cvae import HierarchicalCVAE
from policyOrProxy.models.train_cvae_mapd import (
    MAPDCVAE,
    build_regionizer,
    build_action_bank,
    sample_distribution_features,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_model(model_type: str, model_cfg: Dict, device: torch.device):
    if model_type == "global":
        model = GlobalCVAE(
            window_len=int(model_cfg["window_len"]),
            teams=int(model_cfg["model"]["teams"]),
            agents=int(model_cfg["model"]["agents"]),
            state_dim=int(model_cfg["model"]["state_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            d_model=int(model_cfg["model"]["encoder_dim"]),
            layers=int(model_cfg["model"]["encoder_layers"]),
            heads=int(model_cfg["model"]["encoder_heads"]),
            dropout=float(model_cfg["model"]["dropout"]),
            action_dim=int(model_cfg["model"]["action_dim"]),
        ).to(device)
        latent_key = "z"
        extras: Dict[str, object] = {}
    elif model_type == "hier":
        model = HierarchicalCVAE(
            window_len=int(model_cfg["window_len"]),
            teams=int(model_cfg["model"]["teams"]),
            agents=int(model_cfg["model"]["agents"]),
            state_dim=int(model_cfg["model"]["state_dim"]),
            latent_dim_global=int(model_cfg["latent_dim"]["global"]),
            latent_dim_local=int(model_cfg["latent_dim"]["local"]),
            d_model=int(model_cfg["model"]["encoder_dim"]),
            layers=int(model_cfg["model"]["encoder_layers"]),
            heads=int(model_cfg["model"]["encoder_heads"]),
            dropout=float(model_cfg["model"]["dropout"]),
            action_dim=int(model_cfg["model"]["action_dim"]),
        ).to(device)
        latent_key = "z_global"
        extras = {}
    elif model_type == "mapd":
        agents = int(model_cfg["model"]["agents"])
        model = MAPDCVAE(
            window_len=int(model_cfg["window_len"]),
            teams=int(model_cfg["model"]["teams"]),
            agents=agents,
            state_dim=int(model_cfg["model"]["state_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            d_model=int(model_cfg["model"]["encoder_dim"]),
            layers=int(model_cfg["model"]["encoder_layers"]),
            heads=int(model_cfg["model"]["encoder_heads"]),
            dropout=float(model_cfg["model"]["dropout"]),
            action_dim=int(model_cfg["model"]["action_dim"]),
            dist_feature_dim=agents * 4,
        ).to(device)
        latent_key = "z"
        extras = {
            "samples_per_state": int(model_cfg.get("samples_per_state", 8)),
            "data_config": model_cfg.get("data_config", "policyOrProxy/cfg/data.yaml"),
            "ego_policy": model_cfg.get("ego_policy", "policyOrProxy/cfg/ego_policy.yaml"),
        }
    else:
        raise ValueError(f"Unsupported model type {model_type}")
    if torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel with %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
    return model, latent_key, extras


@torch.no_grad()
def evaluate_split(model, dataloader, device, arena, dt: float, forward_fn, latent_key: str) -> Dict[str, float]:
    episode_preds: Dict[int, List[torch.Tensor]] = defaultdict(list)
    episode_targets: Dict[int, List[torch.Tensor]] = defaultdict(list)
    episode_steps: Dict[int, List[int]] = defaultdict(list)
    latent_embeddings: List[np.ndarray] = []
    collision_samples = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = forward_fn(batch)
        mean = outputs["mean"]
        for idx in range(mean.size(0)):
            episode = int(batch["episode_id"][idx].item())
            timestep = int(batch["timestep"][idx].item())
            episode_preds[episode].append(mean[idx].detach().cpu())
            episode_targets[episode].append(batch["action"][idx].detach().cpu())
            episode_steps[episode].append(timestep)
        if latent_key in outputs:
            latent_embeddings.append(outputs[latent_key].detach().cpu().numpy())
        positions = batch["window"][:, -1, 0, :, :2].detach().cpu()
        predicted_positions = positions + mean.detach().cpu() * dt
        collision_samples.append(predicted_positions)
    metrics = {"ade": 0.0, "fde": 0.0, "smoothness": 0.0}
    total_episodes = len(episode_preds)
    smoothness_values = []
    ade_values = []
    fde_values = []
    for episode, preds in episode_preds.items():
        order = np.argsort(episode_steps[episode])
        preds_seq = torch.stack([preds[i] for i in order], dim=0)
        targets_seq = torch.stack([episode_targets[episode][i] for i in order], dim=0)
        adefde = ade_fde(preds_seq.unsqueeze(0), targets_seq.unsqueeze(0))
        ade_values.append(adefde["ade"])
        fde_values.append(adefde["fde"])
        if preds_seq.size(0) > 1:
            smoothness_values.append(action_smoothness(preds_seq.unsqueeze(0)))
    metrics["ade"] = float(np.mean(ade_values)) if ade_values else 0.0
    metrics["fde"] = float(np.mean(fde_values)) if fde_values else 0.0
    metrics["smoothness"] = float(np.mean(smoothness_values)) if smoothness_values else 0.0
    if collision_samples:
        predicted = torch.cat(collision_samples, dim=0)
        metrics["collisions"] = collision_rate(predicted.unsqueeze(0), arena)
    else:
        metrics["collisions"] = 0.0
    embeddings = np.concatenate(latent_embeddings, axis=0) if latent_embeddings else np.zeros((0, 1))
    return metrics, embeddings


def run_evaluation(model_type: str, model_cfg: Dict, ckpt: Path, eval_cfg: Dict, data_cfg: Dict) -> Dict[str, Dict[str, float]]:
    device_setting = eval_cfg["rollout"].get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
    if model_type == "grover":
        raise ValueError("Grover model does not support action rollout evaluation")
    model, latent_key, extras = build_model(model_type, model_cfg, device)
    state = torch.load(ckpt, map_location=device)
    (model.module if isinstance(model, torch.nn.DataParallel) else model).load_state_dict(state)
    arena = build_arena(data_cfg["arena"])
    dt = float(data_cfg["world"]["dt"])
    batch_size = int(eval_cfg["rollout"]["batch_size"])
    num_workers = int(eval_cfg["rollout"].get("num_workers", 0))
    results = {}
    invariance_features = {}
    eval_split = eval_cfg["rollout"].get("split", "test")

    if model_type == "mapd":
        data_cfg_path = Path(extras["data_config"])
        ego_cfg_path = Path(extras["ego_policy"])
        data_cfg_full = load_yaml(data_cfg_path)
        ego_cfg_full = load_yaml(ego_cfg_path)
        regionizer = build_regionizer(data_cfg_full, ego_cfg_full)
        base_root = Path(model_cfg["paths"]["data_root"]).expanduser()
        base_indexer = EpisodeIndexer.load(base_root)
        train_dataset = NextFrameDataset(base_root, base_indexer, split="train", include_policy_id=False)
        bank = build_action_bank(train_dataset, regionizer)
        samples_per_state = extras["samples_per_state"]

        def forward(batch):
            dist_features = sample_distribution_features(
                batch["window"],
                batch["action"],
                regionizer,
                bank,
                samples_per_state,
            )
            return model(batch["window"], dist_features)
    elif model_type == "hier":
        def forward(batch):
            return model(batch["window"], episode_ids=batch["episode_id"])
    else:
        def forward(batch):
            return model(batch["window"])

    def evaluate_root(root: Path, tag: str):
        indexer = EpisodeIndexer.load(root)
        dataset = NextFrameDataset(root, indexer, split=eval_split)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=next_frame_collate,
        )
        metrics, embeddings = evaluate_split(model, loader, device, arena, dt, forward, latent_key)
        results[tag] = metrics
        invariance_features[tag] = embeddings
    baseline_root = Path(eval_cfg["datasets"]["baseline"])
    evaluate_root(baseline_root, "baseline")
    for tag, path in eval_cfg["datasets"].get("ood_splits", {}).items():
        evaluate_root(Path(path), tag)
    invariance = representation_invariance_score(invariance_features)
    for tag in results:
        results[tag]["invariance"] = float(invariance)
    return results


def main(model_type: str, model_config_path: Path, checkpoint: Path, eval_config_path: Path, data_config_path: Path, output: Path) -> None:
    configure_logging()
    model_cfg = load_yaml(model_config_path)
    eval_cfg = load_yaml(eval_config_path)
    data_cfg = load_yaml(data_config_path)
    results = run_evaluation(model_type, model_cfg, checkpoint, eval_cfg, data_cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    LOGGER.info("Evaluation results stored in %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CVAE models")
    parser.add_argument("--model_type", choices=["global", "hier", "mapd"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval", type=str, default="policyOrProxy/cfg/eval.yaml")
    parser.add_argument("--data", type=str, default="policyOrProxy/cfg/data.yaml")
    parser.add_argument("--output", type=str, default="output/eval/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model_type, Path(args.config), Path(args.checkpoint), Path(args.eval), Path(args.data), Path(args.output))
