"""Training loop for the hierarchical latent CVAE."""
from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Dict
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


class HierarchicalCVAE(nn.Module):
    def __init__(
        self,
        window_len: int,
        teams: int,
        agents: int,
        state_dim: int,
        latent_dim_global: int,
        latent_dim_local: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.window_len = window_len
        self.teams = teams
        self.agents = agents
        self.state_dim = state_dim
        self.latent_dim_global = latent_dim_global
        self.latent_dim_local = latent_dim_local
        self.action_dim = action_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(teams * agents * state_dim, d_model)
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
        self.to_mu_global = nn.Linear(d_model, latent_dim_global)
        self.to_logvar_global = nn.Linear(d_model, latent_dim_global)
        local_input_dim = teams * state_dim + latent_dim_global
        self.local_hidden = nn.Linear(local_input_dim, d_model)
        self.to_mu_local = nn.Linear(d_model, latent_dim_local)
        self.to_logvar_local = nn.Linear(d_model, latent_dim_local)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_global + latent_dim_local + teams * state_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim * 2),
        )

    def encode(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = window.size(0)
        flattened = window.view(batch, self.window_len, self.teams * self.agents * self.state_dim)
        embedded = self.input_proj(flattened)
        embedded = self.positional(embedded)
        encoded = self.encoder(embedded)
        context_global = encoded.mean(dim=1)
        mu_global = self.to_mu_global(context_global)
        logvar_global = self.to_logvar_global(context_global)
        last_frame = window[:, -1].view(batch, self.teams, self.agents, self.state_dim)
        local_context = last_frame.permute(0, 2, 1, 3).reshape(batch, self.agents, self.teams * self.state_dim)
        return {
            "mu_global": mu_global,
            "logvar_global": logvar_global,
            "context_global": context_global,
            "local_context": local_context,
        }

    def sample_global(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_local(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, local_context: torch.Tensor, z_global: torch.Tensor, z_local: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_global_expanded = z_global.unsqueeze(1).expand(-1, self.agents, -1)
        decoder_input = torch.cat([local_context, z_global_expanded, z_local], dim=-1)
        hidden = self.decoder(decoder_input)
        hidden = hidden.view(hidden.size(0), self.agents, self.action_dim, 2)
        mean = hidden[..., 0]
        logvar = hidden[..., 1]
        return {"mean": mean, "logvar": logvar}

    def forward(self, window: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        encoded = self.encode(window)
        z_global = self.sample_global(encoded["mu_global"], encoded["logvar_global"], deterministic=deterministic)
        local_input = torch.cat([encoded["local_context"], z_global.unsqueeze(1).expand(-1, self.agents, -1)], dim=-1)
        local_hidden = torch.tanh(self.local_hidden(local_input))
        mu_local = self.to_mu_local(local_hidden)
        logvar_local = self.to_logvar_local(local_hidden)
        z_local = self.sample_local(mu_local, logvar_local, deterministic=deterministic)
        decoded = self.decode(encoded["local_context"], z_global, z_local)
        return {
            **encoded,
            "mu_local": mu_local,
            "logvar_local": logvar_local,
            "z_global": z_global,
            "z_local": z_local,
            **decoded,
        }

    def loss(self, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = outputs["mean"]
        logvar = outputs["logvar"]
        recon = 0.5 * ((actions - mean).pow(2) * torch.exp(-logvar) + logvar)
        recon = recon.sum(dim=[1, 2]).mean()
        kl_global = -0.5 * torch.sum(1 + outputs["logvar_global"] - outputs["mu_global"].pow(2) - outputs["logvar_global"].exp(), dim=1).mean()
        kl_local = -0.5 * torch.sum(1 + outputs["logvar_local"] - outputs["mu_local"].pow(2) - outputs["logvar_local"].exp(), dim=[1, 2]).mean()
        loss = recon + kl_global + kl_local
        return {"loss": loss, "recon": recon, "kl_global": kl_global, "kl_local": kl_local}

    def sample(self, window: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        outputs = self.forward(window, deterministic=deterministic)
        return outputs["mean"]


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


def build_model(config: Dict, device: torch.device) -> nn.Module:
    model_cfg = config["model"]
    latent_cfg = config["latent_dim"]
    model = HierarchicalCVAE(
        window_len=int(config["window_len"]),
        teams=int(model_cfg["teams"]),
        agents=int(model_cfg["agents"]),
        state_dim=int(model_cfg["state_dim"]),
        latent_dim_global=int(latent_cfg["global"]),
        latent_dim_local=int(latent_cfg["local"]),
        d_model=int(model_cfg["encoder_dim"]),
        layers=int(model_cfg["encoder_layers"]),
        heads=int(model_cfg["encoder_heads"]),
        dropout=float(model_cfg["dropout"]),
        action_dim=int(model_cfg["action_dim"]),
    )
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        LOGGER.info("Wrapping hierarchical CVAE with DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)
    return model


def linear_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)
        return max(min_lr / optimizer.defaults["lr"], 1e-3)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total = {"loss": 0.0, "recon": 0.0, "kl_global": 0.0, "kl_local": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"])
        losses = model.module.loss(outputs, batch["action"]) if isinstance(model, nn.DataParallel) else model.loss(outputs, batch["action"])
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for key in total:
            total[key] += float(losses[key].detach())
    steps = len(dataloader)
    return {key: value / steps for key, value in total.items()}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total = {"loss": 0.0, "recon": 0.0, "kl_global": 0.0, "kl_local": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"])
        losses = model.module.loss(outputs, batch["action"]) if isinstance(model, nn.DataParallel) else model.loss(outputs, batch["action"])
        for key in total:
            total[key] += float(losses[key].detach())
    steps = len(dataloader)
    return {key: value / steps for key, value in total.items()}


def train(config: Dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    set_seed(int(config["seed"]))
    data_root = Path(config["paths"]["data_root"]).expanduser()
    run_dir = Path(config["paths"]["run_dir"]).expanduser()
    configure_logging(run_dir)
    LOGGER.info("Starting hierarchical CVAE training on %s", device)
    indexer = EpisodeIndexer.load(data_root)
    train_dataset = NextFrameDataset(data_root, indexer, split="train")
    val_dataset = NextFrameDataset(data_root, indexer, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )
    model = build_model(config, device)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
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
    global_step = 0
    last_val_step = 0
    validate_interval = int(config.get("validate_interval", len(train_loader)))
    best_val = None
    for epoch in range(1, int(config["epochs"]) + 1):
        metrics_train = train_one_epoch(model, train_loader, optimizer, device)
        global_step += len(train_loader)
        scheduler.step()
        LOGGER.info(
            "Epoch %d train: loss=%.4f recon=%.4f kl_g=%.4f kl_l=%.4f",
            epoch,
            metrics_train["loss"],
            metrics_train["recon"],
            metrics_train["kl_global"],
            metrics_train["kl_local"],
        )
        if global_step - last_val_step >= validate_interval:
            metrics_val = evaluate(model, val_loader, device)
            LOGGER.info(
                "Epoch %d val: loss=%.4f recon=%.4f kl_g=%.4f kl_l=%.4f",
                epoch,
                metrics_val["loss"],
                metrics_val["recon"],
                metrics_val["kl_global"],
                metrics_val["kl_local"],
            )
            ckpt.save(base_model, optimizer, epoch, metric=metrics_val["loss"])
            best_val = metrics_val["loss"] if best_val is None else min(best_val, metrics_val["loss"])
            last_val_step = global_step
    LOGGER.info("Training complete. Best validation loss %.4f", best_val if best_val is not None else float("nan"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical CVAE")
    parser.add_argument("--config", type=str, default="policyOrProxy/cfg/train_hier_cvae.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()
