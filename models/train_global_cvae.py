#!/usr/bin/env python3
"""
Train a global latent CVAE across *selected policies* for configurable categories.

Features:
- Auto-discovers policy roots (iid or specific OOD shift)
- Or uses explicit per-split experiment lists (train/val only; test is ignored here)
- ConcatDataset over selected policies (keeps policy_id in batches)
- beta-VAE (fixed or scheduled)
- Resumable training (model/opt/sched/global_step/epoch/RNG)
- CSV metrics + train.log
- Picks emptiest CUDA device (no DataParallel)

YAML keys you can set (examples):
  data:
    category: "iid" | "ood_manual+right_bias_moderate"
    # OR an explicit experiment split config (train/val only)
    experiment:
      train:
        - {path: "output/data/ego_policy1/iid"}
        - {policy: "ego_policy2", category: "ood_manual+right_bias_mild"}
      val:
        - {path: "output/data/ego_policy1/iid"}
      # test: ...   <-- IGNORED by this script

  loss:
    beta_kl: 1.0
    beta_schedule: none | linear_warmup | cosine
    beta_min: 0.0
    beta_max: 1.0
    beta_warmup_steps: 5000
    beta_period_steps: 20000
  prefer_free_gpu: true
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Callable, Optional, List, Tuple, Iterable, Union
import sys

import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.models.collate import move_batch, next_frame_collate

LOGGER = logging.getLogger(__name__)


# ---------- utils / logging / device ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def pick_emptiest_cuda_device() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    best, best_free = 0, -1
    for i in range(n):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best, best_free = i, free
        except Exception:
            pass
    return best


# ---------- data helpers ----------

def parse_category(cat: str) -> Tuple[str, Optional[str]]:
    """
    'iid' -> ('iid', None)
    'ood_manual+right_bias_moderate' -> ('ood_manual', 'right_bias_moderate')
    """
    parts = cat.split("+")
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Unrecognized category format: {cat}")


def category_to_root(data_root: Path, policy: str, category: str) -> Path:
    base, sub = parse_category(category)
    if base == "iid":
        return data_root / policy / "iid"
    if sub is None:
        raise ValueError("OOD category must be 'ood_manual+<shift_name>'")
    return data_root / base / policy / sub


def _normalize_entries(entries: Union[Dict, List, str]) -> List[Dict]:
    """Accepts dict/list/str and returns a list of dict entries."""
    if isinstance(entries, str):
        return [{"policy": entries}]
    if isinstance(entries, dict):
        return [entries]
    if isinstance(entries, Iterable):
        out = []
        for e in entries:
            out.extend(_normalize_entries(e))
        return out
    raise ValueError(f"Cannot parse entries of type {type(entries)}")


def expand_experiment_entries(data_root: Path, split_entries: List[Dict]) -> List[Tuple[str, Path]]:
    """
    From a split spec, build a list of (policy_name, dataset_root) pairs.
    Each entry can be:
      - {policy: "ego_policy1", category: "iid"}
      - {policy: "ego_policy*", category: "ood_manual+right_bias_mild"}  # glob
      - {path: "output/data/ood_manual/ego_policy2/right_bias_moderate"} # explicit
      - {policy: "ego_policy3", exclude: true}                            # skip
    """
    resolved: List[Tuple[str, Path]] = []
    for ent in _normalize_entries(split_entries):
        if ent.get("exclude", False):
            continue
        if "path" in ent:
            rp = Path(ent["path"]).expanduser()
            if not (rp / "index.json").exists():
                LOGGER.warning("Missing index.json for explicit path: %s", rp)
                continue
            policy = rp.parent.name if rp.parent.name.startswith("ego_policy") else rp.name
            resolved.append((policy, rp))
            continue
        if "policy" in ent and "category" in ent:
            pattern = ent["policy"]
            category = ent["category"]
            for pol in sorted(data_root.glob(pattern)):
                if not pol.is_dir():
                    continue
                policy_name = pol.name
                rp = category_to_root(data_root, policy_name, category)
                if (rp / "index.json").exists():
                    resolved.append((policy_name, rp))
                else:
                    LOGGER.warning("No index.json for (%s, %s) at %s", policy_name, category, rp)
            continue
        if "policy" in ent and "category" not in ent:
            LOGGER.warning("Ignoring entry with 'policy' but no 'category': %s", ent)
            continue
        LOGGER.warning("Unrecognized split entry, ignored: %s", ent)
    # dedupe
    seen = set()
    uniq = []
    for pol, rp in resolved:
        key = (pol, rp.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((pol, rp))
    return uniq


def discover_policy_roots(data_root: Path, category: str) -> List[Path]:
    """
    Returns a list of per-policy directories that each contain an index.json
    with splits train/val/test for the requested category.

    iid:             <data_root>/ego_policy*/iid
    ood_manual+NAME: <data_root>/ood_manual/ego_policy*/NAME
    """
    base, sub = parse_category(category)
    roots: List[Path] = []

    if base == "iid":
        for p in sorted((data_root).glob("ego_policy*/iid")):
            if (p / "index.json").exists():
                roots.append(p)
    else:
        if sub is None:
            raise ValueError("OOD category must be like 'ood_manual+<shift_name>'")
        base_dir = data_root / base
        for p in sorted(base_dir.glob(f"ego_policy*/{sub}")):
            if (p / "index.json").exists():
                roots.append(p)

    if not roots:
        raise FileNotFoundError(f"No policy roots found for category='{category}' under {data_root}")
    return roots


def build_concat_dataset(
    roots: List[Path],
    split: str,
    window_len: int,
    device: Optional[torch.device] = None,
    include_policy_id: bool = True,
) -> Tuple[ConcatDataset, Dict[int, str]]:
    """
    For each policy root, load its EpisodeIndexer and wrap a NextFrameDataset.
    Concatenate them, and build a global (int_id -> policy_name) mapping.
    """
    datasets: List[NextFrameDataset] = []
    id_map: Dict[int, str] = {}
    next_id = 0

    for root in roots:
        # policy name is parent of category dir:
        #   .../ego_policyX/iid   or   .../ood_manual/ego_policyX/shift
        if root.name in ("iid", "train", "val", "test"):
            policy = root.parent.name
        else:
            policy = root.parent.name if (root.parent.name.startswith("ego_policy")) else root.name

        indexer = EpisodeIndexer.load(root)
        has_split = any(rec.split == split for rec in indexer.entries)
        if not has_split:
            LOGGER.warning("Skipping %s — no '%s' split found.", root, split)
            continue

        ds = NextFrameDataset(
            root=root,
            indexer=indexer,
            split=split,
            device=None,                 # moved later in collate
            preload=False,
            include_policy_id=include_policy_id,
            window_len=window_len,
        )
        id_map[next_id] = policy
        next_id += 1
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No datasets found for split '{split}' across provided roots.")

    return ConcatDataset(datasets), id_map


# ---------- model ----------

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


class GlobalCVAE(nn.Module):
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
        reconstruction_loss: str = "smooth_l1",
    ) -> None:
        super().__init__()
        self.window_len = int(window_len)
        self.teams = int(teams)
        self.agents = int(agents)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        self.reconstruction_loss = reconstruction_loss

        self.input_proj = nn.Linear(self.teams * self.agents * self.state_dim, d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout, max_len=max(self.window_len, 500))

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

    def _flatten_to_BTF(self, window: torch.Tensor) -> torch.Tensor:
        B = window.size(0)
        F = self.teams * self.agents * self.state_dim
        if window.dim() == 5:
            T = window.size(1)
            return window.view(B, T, F)
        elif window.dim() == 3 and window.size(-1) == F:
            return window
        elif window.dim() == 2 and window.size(-1) == F:
            return window.unsqueeze(1)
        else:
            raise ValueError(
                f"Unexpected window shape {tuple(window.shape)}; expected last dim {F} "
                f"or [B, T, {self.teams}, {self.agents}, {self.state_dim}]"
            )

    def encode(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = self._flatten_to_BTF(window)
        embedded = self.input_proj(flat)
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

    def forward(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(window)
        z = self.reparameterize(encoded["mu"], encoded["logvar_z"])
        decoded = self.decode(encoded["context"], z)
        return {**encoded, **decoded, "z": z}

    def loss(self, outputs: Dict[str, torch.Tensor], actions: torch.Tensor, beta_kl: float = 1.0) -> Dict[str, torch.Tensor]:
        mean = outputs["mean"]
        logvar = outputs["logvar_action"]
        recon = 0.5 * ((actions - mean).pow(2) * torch.exp(-logvar) + logvar)
        recon = recon.sum(dim=[1, 2]).mean()
        kl = -0.5 * torch.sum(1 + outputs["logvar_z"] - outputs["mu"].pow(2) - outputs["logvar_z"].exp(), dim=1).mean()
        loss = recon + beta_kl * kl
        return {"loss": loss, "recon": recon, "kl": kl, "beta": torch.tensor(beta_kl)}


# ---------- beta schedule ----------

def make_beta_fn(cfg: Dict, base_steps_per_epoch: int) -> Callable[[int], float]:
    loss_cfg = cfg.get("loss", {})
    beta_static = float(loss_cfg.get("beta_kl", 1.0))
    schedule = str(loss_cfg.get("beta_schedule", "none")).lower()

    beta_min = float(loss_cfg.get("beta_min", 0.0))
    beta_max = float(loss_cfg.get("beta_max", beta_static))
    warmup_steps = int(loss_cfg.get("beta_warmup_steps", 0))
    period_steps = int(loss_cfg.get("beta_period_steps", base_steps_per_epoch * 10))

    if schedule == "linear_warmup" and warmup_steps > 0:
        def beta_fn(step: int) -> float:
            if step <= 0: return beta_min
            if step >= warmup_steps: return beta_max
            return beta_min + (beta_max - beta_min) * (step / warmup_steps)
        return beta_fn

    if schedule == "cosine" and period_steps > 0:
        def beta_fn(step: int) -> float:
            cos = 0.5 * (1 + math.cos(2 * math.pi * (step % period_steps) / max(1, period_steps)))
            return beta_min + (beta_max - beta_min) * cos
        return beta_fn

    return lambda _step: beta_static


def linear_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)
        return max(min_lr / optimizer.defaults["lr"], 1e-3)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------- train / eval ----------

def train_one_epoch(model, dataloader, optimizer, device, beta_fn, start_step=0, max_grad_norm: float = 5.0):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    step = start_step
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"])
        beta = float(beta_fn(step))
        losses = model.loss(outputs, batch["action"], beta)

        optimizer.zero_grad()
        losses["loss"].backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss  += float(losses["loss"].detach())
        total_recon += float(losses["recon"].detach())
        total_kl    += float(losses["kl"].detach())
        step += 1
    steps = len(dataloader)
    return {"loss": total_loss / steps, "recon": total_recon / steps, "kl": total_kl / steps}, step


@torch.no_grad()
def evaluate(model, dataloader, device, beta_fn, start_step=0):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    step = start_step
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"])
        beta = float(beta_fn(step))
        losses = model.loss(outputs, batch["action"], beta)
        total_loss  += float(losses["loss"].detach())
        total_recon += float(losses["recon"].detach())
        total_kl    += float(losses["kl"].detach())
        step += 1
    steps = len(dataloader)
    return {"loss": total_loss / steps, "recon": total_recon / steps, "kl": total_kl / steps}, step


# ---------- checkpointing ----------

def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    best_val: Optional[float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> None:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    map_location: str | torch.device,
) -> tuple[int, int, Optional[float]]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    try:
        rng = ckpt.get("rng", {})
        if rng:
            random.setstate(rng.get("python"))
            np.random.set_state(rng.get("numpy"))
            torch.set_rng_state(rng.get("torch"))
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng.get("torch_cuda"))
    except Exception as e:
        LOGGER.warning("Could not fully restore RNG states: %s", e)

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_val = ckpt.get("best_val", None)
    return epoch, global_step, best_val


# ---------- orchestration ----------

def train_loop(config: Dict, data_root: Path | None = None, run_dir: Path | None = None, resume: bool = True) -> None:
    # device
    prefer_free_gpu = bool(config.get("prefer_free_gpu", True))
    if torch.cuda.is_available() and config.get("device", "auto") != "cpu":
        dev_idx = pick_emptiest_cuda_device() if prefer_free_gpu else 0
        torch.cuda.set_device(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
    else:
        device = torch.device("cpu")
    LOGGER.info("Using device: %s", device)

    # seeds, dirs
    set_seed(int(config["seed"]))
    paths_cfg = config.get("paths", {})
    data_root = (data_root or Path(paths_cfg.get("data_root", "output/data"))).expanduser()
    run_dir = (run_dir or Path(paths_cfg.get("run_dir", "output/runs/global_cvae"))).expanduser()
    configure_logging(run_dir)

    # persist config
    try:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp)
    except Exception as e:
        LOGGER.warning("Could not save run config: %s", e)

    # datasets
    T = int(config["window_len"])
    exp_cfg = config.get("data", {}).get("experiment")

    if exp_cfg:
        LOGGER.info("Using explicit experiment configuration.")
        train_pairs = expand_experiment_entries(data_root, exp_cfg.get("train", []))
        val_pairs   = expand_experiment_entries(data_root, exp_cfg.get("val",   exp_cfg.get("train", [])))
        if "test" in exp_cfg:
            LOGGER.info("Ignoring 'data.experiment.test' in training config (tests are evaluated separately).")
        # Convert (policy, root) -> roots
        train_roots = [rp for (_pol, rp) in train_pairs]
        val_roots   = [rp for (_pol, rp) in val_pairs]
    else:
        cat = config.get("data", {}).get("category", "iid")
        LOGGER.info("No experiment config; falling back to category='%s' for all policies.", cat)
        roots = discover_policy_roots(data_root, cat)
        train_roots = roots
        val_roots   = roots

    train_ds, train_idmap = build_concat_dataset(train_roots, split="train", window_len=T)
    val_ds,   val_idmap   = build_concat_dataset(val_roots,   split="val",   window_len=T)

    # Save a merged policy map (train/val only)
    try:
        with (run_dir / "policies.json").open("w", encoding="utf-8") as fp:
            json.dump({"train": train_idmap, "val": val_idmap}, fp, indent=2)
    except Exception as e:
        LOGGER.warning("Could not write policies.json: %s", e)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )

    # model/optim/sched
    model_cfg = config["model"]
    model = GlobalCVAE(
        window_len=T,
        teams=int(model_cfg["teams"]),
        agents=int(model_cfg["agents"]),
        state_dim=int(model_cfg["state_dim"]),
        latent_dim=int(config["latent_dim"]),
        d_model=int(model_cfg["encoder_dim"]),
        layers=int(model_cfg["encoder_layers"]),
        heads=int(model_cfg["encoder_heads"]),
        dropout=float(model_cfg["dropout"]),
        action_dim=int(model_cfg["action_dim"]),
        reconstruction_loss=model_cfg.get("reconstruction_loss", "smooth_l1"),
    ).to(device)

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

    beta_fn = make_beta_fn(config, base_steps_per_epoch=len(train_loader))

    # CSV logger
    metrics_csv_path = run_dir / "metrics.csv"
    new_file = not metrics_csv_path.exists()
    csv_f = metrics_csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    if new_file:
        csv_writer.writerow(["epoch", "split", "loss", "recon", "kl", "beta_example", "lr"])

    # resume
    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    start_epoch = 1
    global_step = 0
    best_val: Optional[float] = None

    if resume and last_ckpt.exists():
        LOGGER.info("Resuming from %s", last_ckpt)
        e, s, b = load_checkpoint(last_ckpt, model, optimizer, scheduler, map_location=device)
        start_epoch = e + 1
        global_step = s
        best_val = b

    validate_interval = int(config.get("validate_interval", len(train_loader)))
    max_grad_norm = float(config.get("max_grad_norm", 5.0))
    epochs = int(config["epochs"])

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, device, beta_fn,
            start_step=global_step, max_grad_norm=max_grad_norm
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        beta_example = beta_fn(global_step)
        LOGGER.info(
            "Epoch %d train: loss=%.4f recon=%.4f kl=%.4f beta≈%.4f lr=%.6f",
            epoch, train_metrics["loss"], train_metrics["recon"], train_metrics["kl"], beta_example, lr
        )
        csv_writer.writerow([
            epoch, "train",
            f'{train_metrics["loss"]:.6f}',
            f'{train_metrics["recon"]:.6f}',
            f'{train_metrics["kl"]:.6f}',
            f'{beta_example:.6f}',
            f'{lr:.8f}',
        ])
        csv_f.flush()

        # periodic validation + checkpointing
        if (epoch == start_epoch) or (global_step % validate_interval == 0) or (epoch == epochs):
            val_metrics, _ = evaluate(model, val_loader, device, beta_fn, start_step=global_step)
            LOGGER.info(
                "Epoch %d val:   loss=%.4f recon=%.4f kl=%.4f",
                epoch, val_metrics["loss"], val_metrics["recon"], val_metrics["kl"]
            )
            csv_writer.writerow([
                epoch, "val",
                f'{val_metrics["loss"]:.6f}',
                f'{val_metrics["recon"]:.6f}',
                f'{val_metrics["kl"]:.6f}',
                "", ""
            ])
            csv_f.flush()

            # save last/best
            save_checkpoint(last_ckpt, epoch, global_step, best_val, model, optimizer, scheduler)
            if best_val is None or val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(best_ckpt, epoch, global_step, best_val, model, optimizer, scheduler)

    csv_f.close()
    LOGGER.info("Training complete. Best validation loss: %.4f", best_val if best_val is not None else float("nan"))


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train global CVAE across selected policies")
    p.add_argument("--config", type=str, default="policyOrProxy/cfg/train_global_cvae.yaml")
    p.add_argument("--data_root", type=str, help="Override data root (default from YAML)")
    p.add_argument("--run_dir", type=str, help="Override run directory (default from YAML)")
    p.add_argument("--no_resume", action="store_true", help="Do not resume even if a checkpoint exists")
    return p.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.data_root:
        config.setdefault("paths", {})["data_root"] = args.data_root
    if args.run_dir:
        config.setdefault("paths", {})["run_dir"] = args.run_dir
    resume = not args.no_resume
    train_loop(config, resume=resume)


if __name__ == "__main__":
    main()
