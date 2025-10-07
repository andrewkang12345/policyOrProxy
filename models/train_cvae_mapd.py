"""Training loop for CVAE with policy action distribution (MAPD) inputs."""
from __future__ import annotations

import argparse
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.core.regionizers.windowhash import WindowHashRegionizer
from policyOrProxy.core.world.arena import build_arena
from policyOrProxy.models.collate import move_batch, next_frame_collate
from policyOrProxy.models.checkpoint import CheckpointManager

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MAPDCVAE(nn.Module):
    def __init__(
        self,
        window_len: int,
        teams: int,
        agents: int,
        state_dim: int,
        latent_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        action_dim: int,
        dist_feature_dim: int,
    ) -> None:
        super().__init__()
        self.window_len = window_len
        self.teams = teams
        self.agents = agents
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.action_dim = action_dim
        self.dist_feature_dim = dist_feature_dim
        input_dim = teams * agents * state_dim + dist_feature_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout, max_len=window_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, agents * action_dim * 2),
        )

    def encode(self, window: torch.Tensor, dist_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = window.size(0)
        flattened = window.view(batch, self.window_len, self.teams * self.agents * self.state_dim)
        dist = dist_features.view(batch, 1, self.dist_feature_dim).expand(-1, self.window_len, -1)
        combined = torch.cat([flattened, dist], dim=-1)
        embedded = self.input_proj(combined)
        embedded = self.positional(embedded)
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return {"mu": mu, "logvar_z": logvar, "context": pooled}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, context: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = torch.cat([context, z], dim=-1)
        out = self.decoder(latent)
        out = out.view(out.size(0), self.agents, self.action_dim, 2)
        mean = out[..., 0]
        logvar = out[..., 1]
        return {"mean": mean, "logvar_action": logvar}

    def forward(self, window: torch.Tensor, dist_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(window, dist_features)
        z = self.reparameterize(encoded["mu"], encoded["logvar_z"])
        decoded = self.decode(encoded["context"], z)
        return {**encoded, **decoded, "z": z}

    def loss(self, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = outputs["mean"]
        logvar_action = outputs["logvar_action"]
        recon = 0.5 * ((actions - mean).pow(2) * torch.exp(-logvar_action) + logvar_action)
        recon = recon.sum(dim=[1, 2]).mean()
        kl = -0.5 * torch.sum(1 + outputs["logvar_z"] - outputs["mu"].pow(2) - outputs["logvar_z"].exp(), dim=1).mean()
        return {"loss": recon + kl, "recon": recon, "kl": kl}

    def sample(self, window: torch.Tensor, dist_features: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        encoded = self.encode(window, dist_features)
        z = encoded["mu"] if deterministic else self.reparameterize(encoded["mu"], encoded["logvar_z"])
        decoded = self.decode(encoded["context"], z)
        return decoded["mean"]


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def build_regionizer(data_cfg: Dict, ego_cfg: Dict) -> WindowHashRegionizer:
    arena = build_arena(data_cfg["arena"])
    return WindowHashRegionizer(
        arena=arena,
        num_buckets=int(ego_cfg["num_buckets"]),
        grid_size=int(ego_cfg["quantization"]["grid_size"]),
        length_scale=float(ego_cfg["quantization"].get("length_scale", 1.0)),
        jitter=float(ego_cfg["quantization"].get("jitter", 0.0)),
    )


def build_action_bank(dataset: NextFrameDataset, regionizer: WindowHashRegionizer) -> Dict[int, np.ndarray]:
    LOGGER.info("Precomputing action distributions per bucket")
    bank: Dict[int, List[np.ndarray]] = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        window = sample["window"].numpy()
        action = sample["action"].numpy()
        bucket = regionizer.to_bucket(window)
        bank[bucket].append(action)
    packed: Dict[int, np.ndarray] = {}
    for bucket, actions in bank.items():
        packed[bucket] = np.stack(actions, axis=0)
    return packed


def sample_distribution_features(
    windows: torch.Tensor,
    actions: torch.Tensor,
    regionizer: WindowHashRegionizer,
    bank: Dict[int, np.ndarray],
    samples_per_state: int,
) -> torch.Tensor:
    device = windows.device
    batch = windows.size(0)
    agents = actions.size(1)
    dist_features = torch.zeros(batch, agents, 4, device=device, dtype=windows.dtype)
    for idx in range(batch):
        window_np = windows[idx].detach().cpu().numpy()
        bucket = regionizer.to_bucket(window_np)
        bank_actions = bank.get(bucket)
        if bank_actions is None or bank_actions.size == 0:
            selected = actions[idx].detach().cpu().numpy()[None, ...]
        else:
            count = bank_actions.shape[0]
            replace = count < samples_per_state
            choice = np.random.choice(count, size=samples_per_state, replace=replace)
            selected = bank_actions[choice]
        selected_tensor = torch.from_numpy(selected).to(device=windows.device, dtype=windows.dtype)
        mean = selected_tensor.mean(dim=0)
        std = selected_tensor.std(dim=0, unbiased=False)
        dist_features[idx, :, :2] = mean
        dist_features[idx, :, 2:] = std
    return dist_features.view(batch, -1)


def linear_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)
        return max(min_lr / optimizer.defaults["lr"], 1e-3)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, device, regionizer, bank, samples_per_state):
    model.train()
    total = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        dist_features = sample_distribution_features(
            batch["window"],
            batch["action"],
            regionizer,
            bank,
            samples_per_state,
        )
        outputs = model(batch["window"], dist_features)
        losses = model.loss(outputs, batch["action"])
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for key in total:
            total[key] += float(losses[key].detach())
    steps = len(dataloader)
    return {key: value / steps for key, value in total.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, regionizer, bank, samples_per_state):
    model.eval()
    total = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        dist_features = sample_distribution_features(
            batch["window"],
            batch["action"],
            regionizer,
            bank,
            samples_per_state,
        )
        outputs = model(batch["window"], dist_features)
        losses = model.loss(outputs, batch["action"])
        for key in total:
            total[key] += float(losses[key].detach())
    steps = len(dataloader)
    return {key: value / steps for key, value in total.items()}


def train(
    config: Dict,
    data_root: Path | None = None,
    run_dir: Path | None = None,
    data_cfg_path: Path | None = None,
    ego_cfg_path: Path | None = None,
) -> None:
    cfg_data_path = Path(config.get("data_config", "policyOrProxy/cfg/data.yaml"))
    cfg_ego_path = Path(config.get("ego_policy", "policyOrProxy/cfg/ego_policy.yaml"))
    data_cfg = load_config(data_cfg_path or cfg_data_path)
    ego_cfg = load_config(ego_cfg_path or cfg_ego_path)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    paths_cfg = config.get("paths", {})
    data_root = (data_root or Path(paths_cfg.get("data_root", "output/data"))).expanduser()
    run_dir = (run_dir or Path(paths_cfg.get("run_dir", "output/runs/mapd_cvae"))).expanduser()
    configure_logging(run_dir)

    regionizer = build_regionizer(data_cfg, ego_cfg)

    indexer = EpisodeIndexer.load(data_root)
    train_dataset = NextFrameDataset(data_root, indexer, split="train", include_policy_id=False)
    val_dataset = NextFrameDataset(data_root, indexer, split="val", include_policy_id=False)

    bank = build_action_bank(train_dataset, regionizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=next_frame_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=next_frame_collate,
    )

    samples_per_state = int(config.get("samples_per_state", 8))
    agents = int(config["model"]["agents"])
    dist_feature_dim = agents * 4

    model = MAPDCVAE(
        window_len=int(config["window_len"]),
        teams=int(config["model"]["teams"]),
        agents=agents,
        state_dim=int(config["model"]["state_dim"]),
        latent_dim=int(config["latent_dim"]),
        d_model=int(config["model"]["encoder_dim"]),
        layers=int(config["model"]["encoder_layers"]),
        heads=int(config["model"]["encoder_heads"]),
        dropout=float(config["model"]["dropout"]),
        action_dim=int(config["model"]["action_dim"]),
        dist_feature_dim=dist_feature_dim,
    ).to(device)
    if torch.cuda.device_count() > 1:
        LOGGER.info("Wrapping MAPD CVAE with DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    optimizer_cfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    scheduler_cfg = config.get("scheduler", {})
    scheduler = linear_warmup_scheduler(
        optimizer,
        warmup_steps=int(scheduler_cfg.get("warmup_steps", 0)),
        min_lr=float(scheduler_cfg.get("min_lr", optimizer_cfg["lr"])),
    )

    ckpt = CheckpointManager(run_dir / "checkpoints", direction="min")
    best_val = None
    validate_interval = int(config.get("validate_interval", len(train_loader)))
    steps_since_val = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, regionizer, bank, samples_per_state)
        scheduler.step()
        LOGGER.info(
            "Epoch %d train: loss=%.4f recon=%.4f kl=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["recon"],
            train_metrics["kl"],
        )
        steps_since_val += len(train_loader)
        if steps_since_val >= validate_interval:
            val_metrics = evaluate(model, val_loader, device, regionizer, bank, samples_per_state)
            LOGGER.info(
                "Epoch %d val: loss=%.4f recon=%.4f kl=%.4f",
                epoch,
                val_metrics["loss"],
                val_metrics["recon"],
                val_metrics["kl"],
            )
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            ckpt.save(base_model, optimizer, epoch, metric=val_metrics["loss"])
            best_val = val_metrics["loss"] if best_val is None else min(best_val, val_metrics["loss"])
            steps_since_val = 0
    LOGGER.info("Training complete. Best validation loss %.4f", best_val if best_val is not None else float("nan"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVAE with policy action distributions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, help="Override data root")
    parser.add_argument("--run_dir", type=str, help="Override run directory")
    parser.add_argument("--data_cfg", type=str, help="Override data config path")
    parser.add_argument("--ego_cfg", type=str, help="Override ego policy config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    data_root = Path(args.data_root).expanduser() if args.data_root else None
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    data_cfg = Path(args.data_cfg).expanduser() if args.data_cfg else None
    ego_cfg = Path(args.ego_cfg).expanduser() if args.ego_cfg else None
    train(config, data_root=data_root, run_dir=run_dir, data_cfg_path=data_cfg, ego_cfg_path=ego_cfg)


if __name__ == "__main__":
    main()
