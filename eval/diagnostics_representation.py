"""Diagnostics for learned policy representations across multiple policies."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List
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
    cluster_purity,
    linear_probe_accuracy,
    representation_invariance_score,
)
from policyOrProxy.models.collate import move_batch, next_frame_collate
from policyOrProxy.models.train_global_cvae import GlobalCVAE
from policyOrProxy.models.train_hier_cvae import HierarchicalCVAE
from policyOrProxy.models.train_cvae_mapd import (
    MAPDCVAE,
    build_regionizer,
    build_action_bank,
    sample_distribution_features,
)
from policyOrProxy.models.train_cvae_grover import GroverEncoder, load_episodes

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_feature_model(model_type: str, model_cfg: Dict, device: torch.device):
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
        latent_key = "z"  # leave as-is for global model unless you prefer "mu"
        return model, latent_key, {}

    if model_type == "hier":
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
        # Use the deterministic mean of the global latent for representations
        latent_key = "mu_global"
        return model, latent_key, {}

    if model_type == "mapd":
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
        return model, latent_key, extras

    if model_type == "grover":
        # encoder built later in grover-specific flow
        return None, "embedding", {}

    raise ValueError(f"Unsupported model type {model_type}")


def _normalize_spec(spec) -> List[Dict[str, str]]:
    if isinstance(spec, str):
        return [{"path": spec}]
    if isinstance(spec, dict) and "path" in spec:
        return [spec]
    if isinstance(spec, Iterable):
        normalized: List[Dict[str, str]] = []
        for item in spec:
            normalized.extend(_normalize_spec(item))
        return normalized
    raise ValueError(f"Invalid dataset specification: {spec}")


@torch.no_grad()
def collect_episode_embeddings(
    loader: DataLoader,
    dataset: NextFrameDataset,
    device: torch.device,
    forward_fn,
    latent_key: str,
    tag: str,
) -> Dict[str, List[np.ndarray]]:
    per_episode: Dict[tuple[int, str], Dict[str, torch.Tensor]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_fn(batch)
        if latent_key not in outputs:
            raise RuntimeError(f"Latent key '{latent_key}' missing from model outputs")
        latents = outputs[latent_key]
        if latents.dim() == 3:
            latents = latents.mean(dim=1)
        latents_cpu = latents.detach().cpu()
        episode_ids = batch["episode_id"].detach().cpu().numpy()
        policy_indices = (
            batch["policy_id"].detach().cpu().numpy() if "policy_id" in batch else np.zeros_like(episode_ids)
        )
        for idx in range(latents_cpu.size(0)):
            episode_id = int(episode_ids[idx])
            policy_idx = int(policy_indices[idx])
            policy_name = dataset.policy_name(policy_idx)
            label = f"{tag}:{policy_name}"
            key = (episode_id, label)
            entry = per_episode.setdefault(
                key,
                {"sum": torch.zeros_like(latents_cpu[idx]), "count": 0},
            )
            entry["sum"] += latents_cpu[idx]
            entry["count"] += 1
    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    for (_episode_id, label), stats in per_episode.items():
        mean_emb = (stats["sum"] / max(stats["count"], 1)).numpy()
        grouped[label].append(mean_emb)
    return grouped


def run_grover_diagnostics(model_cfg: Dict, checkpoint: Path, eval_cfg: Dict, output: Path, device: torch.device) -> None:
    data_root = Path(model_cfg["paths"]["data_root"]).expanduser()
    indexer = EpisodeIndexer.load(data_root)
    train_policies = load_episodes(data_root, indexer, split="train", device=device)
    sample_policy = next(iter(train_policies.values()))
    if not sample_policy:
        raise RuntimeError("No episodes found in training data for Grover model")
    sample_episode = sample_policy[0]
    state_dim = sample_episode.states.size(-1)
    action_dim = sample_episode.actions.size(-1)

    embed_dim = int(model_cfg["model"]["embed_dim"])
    hidden_dim = int(model_cfg["model"]["hidden_dim"])

    encoder = GroverEncoder(
        input_dim=state_dim + action_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    dataset_specs = eval_cfg.get("datasets", {})
    if not dataset_specs:
        raise ValueError("Evaluation config must specify datasets")
    split_name = eval_cfg["rollout"].get("split", "test")
    max_samples = eval_cfg.get("max_samples")

    grouped_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for distribution, spec in dataset_specs.items():
        for entry in _normalize_spec(spec):
            dataset_path = Path(entry["path"])
            label = entry.get("label")
            tag = f"{distribution}:{label}" if label else distribution
            indexer_eval = EpisodeIndexer.load(dataset_path)
            policies = load_episodes(dataset_path, indexer_eval, split=split_name, device=device)
            for policy_name, episodes in policies.items():
                for episode in episodes:
                    embedding = encoder(episode.states, episode.actions).detach().cpu().numpy()
                    key = f"{tag}:{policy_name}"
                    grouped_embeddings[key].append(embedding)

    if len(grouped_embeddings) < 2:
        raise RuntimeError("Grover diagnostics require embeddings from at least two policies")

    all_features = []
    all_labels = []
    for label, feats in grouped_embeddings.items():
        all_features.extend(feats)
        all_labels.extend([label] * len(feats))
    features_array = np.asarray(all_features, dtype=np.float32)
    labels_array = np.asarray(all_labels)
    if max_samples is not None and features_array.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(features_array.shape[0], size=max_samples, replace=False)
        features_array = features_array[idx]
        labels_array = labels_array[idx]
    unique_labels = sorted(set(labels_array))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = np.asarray([label_to_int[label] for label in labels_array], dtype=np.int64)
    num_clusters = len(unique_labels)
    probe_acc = linear_probe_accuracy(features_array, label_ids)
    purity = cluster_purity(features_array, label_ids, num_clusters=num_clusters)
    invariance = representation_invariance_score(
        {label: np.asarray(feats, dtype=np.float32) for label, feats in grouped_embeddings.items()}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "linear_probe_accuracy": probe_acc,
                "cluster_purity": purity,
                "invariance": invariance,
                "policy_counts": {label: len(feats) for label, feats in grouped_embeddings.items()},
            },
            fp,
            indent=2,
        )
    LOGGER.info("Representation diagnostics saved to %s", output)


def main(model_type: str, model_cfg_path: Path, checkpoint: Path, eval_cfg_path: Path, output: Path) -> None:
    configure_logging()
    model_cfg = load_yaml(model_cfg_path)
    eval_cfg = load_yaml(eval_cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, latent_key, extras = build_feature_model(model_type, model_cfg, device)

    if model_type == "grover":
        run_grover_diagnostics(model_cfg, checkpoint, eval_cfg, output, device)
        return

    ckpt = torch.load(checkpoint, map_location=device)

    # Unwrap common checkpoint formats
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model"]:
            if k in ckpt:
                state_dict = ckpt[k]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # If the keys are prefixed with "module.", strip them for non-DataParallel models
    def strip_module_prefix(sd):
        if not len(sd):
            return sd
        needs_strip = next(iter(sd)).startswith("module.")
        if not needs_strip:
            return sd
        return {k[len("module."):]: v for k, v in sd.items()}

    state_dict = strip_module_prefix(state_dict)

    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning("Loaded with missing keys: %s", missing)
        LOGGER.warning("Loaded with unexpected keys: %s", unexpected)

    # IMPORTANT: disable dropout / use running stats
    target.eval()

    batch_size = int(eval_cfg["rollout"]["batch_size"])
    num_workers = int(eval_cfg["rollout"].get("num_workers", 0))
    split_name = eval_cfg["rollout"].get("split", "test")
    max_samples = eval_cfg.get("max_samples")

    dataset_specs = eval_cfg.get("datasets", {})
    if not dataset_specs:
        raise ValueError("Evaluation config must specify datasets")

    if model_type == "mapd":
        data_cfg = load_yaml(Path(extras["data_config"]))
        ego_cfg = load_yaml(Path(extras["ego_policy"]))
        regionizer = build_regionizer(data_cfg, ego_cfg)
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
            # keep any stochasticity in dist features but disable model sampling with deterministic forward if supported
            return model(batch["window"], dist_features)
    else:
        # Deterministic forward: bypass latent sampling noise
        def forward(batch):
            return model(batch["window"], deterministic=True)

    grouped_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for distribution, spec in dataset_specs.items():
        for entry in _normalize_spec(spec):
            dataset_path = Path(entry["path"])
            label = entry.get("label")
            tag = f"{distribution}:{label}" if label else distribution
            indexer = EpisodeIndexer.load(dataset_path)
            dataset = NextFrameDataset(
                dataset_path,
                indexer,
                split=split_name,
                include_policy_id=True,
                window_len=int(model_cfg["window_len"]),  # align eval window with model
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=next_frame_collate,
            )
            groups = collect_episode_embeddings(loader, dataset, device, forward, latent_key, tag)
            for policy_label, feats in groups.items():
                grouped_embeddings[policy_label].extend(feats)

    if len(grouped_embeddings) < 2:
        raise RuntimeError(
            "Representation diagnostics require embeddings from at least two distinct policies; "
            "check that each distribution provides multiple policy datasets."
        )

    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    for policy_label, feats in grouped_embeddings.items():
        all_features.extend(feats)
        all_labels.extend([policy_label] * len(feats))
    features_array = np.asarray(all_features, dtype=np.float32)
    labels_array = np.asarray(all_labels)

    if max_samples is not None and features_array.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(features_array.shape[0], size=max_samples, replace=False)
        features_array = features_array[idx]
        labels_array = labels_array[idx]

    unique_labels = sorted(set(labels_array))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = np.asarray([label_to_int[label] for label in labels_array], dtype=np.int64)

    num_clusters = len(unique_labels)
    probe_acc = linear_probe_accuracy(features_array, label_ids)
    purity = cluster_purity(features_array, label_ids, num_clusters=num_clusters)
    invariance = representation_invariance_score(
        {label: np.asarray(feats, dtype=np.float32) for label, feats in grouped_embeddings.items()}
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "linear_probe_accuracy": probe_acc,
                "cluster_purity": purity,
                "invariance": invariance,
                "policy_counts": {label: len(feats) for label, feats in grouped_embeddings.items()},
            },
            fp,
            indent=2,
        )
    LOGGER.info("Representation diagnostics saved to %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate representation diagnostics")
    parser.add_argument("--model_type", choices=["global", "hier", "mapd", "grover"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval", type=str, default="policyOrProxy/cfg/eval.yaml")
    parser.add_argument("--output", type=str, default="output/eval/representation.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.model_type, Path(args.config), Path(args.checkpoint), Path(args.eval), Path(args.output))