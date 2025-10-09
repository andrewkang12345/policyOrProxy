#!/usr/bin/env python3
"""
Train a hierarchical latent CVAE across *multiple policies* with a clean config.

Key points
- NEVER reads output/data/index.json at the root.
- Discovers/uses child folders that each contain their own index.json:
    - iid:             <data_root>/ego_policy*/iid
    - ood_manual+NAME: <data_root>/ood_manual/ego_policy*/NAME
- Also supports explicit experiment lists (paths or (policy, category) globs).
- ConcatDataset over policies (keeps episode_ids to aggregate per-episode global latents).
- Resumable training (model/opt/sched/global_step/epoch/RNG).
- CSV metrics + train.log + policies.json.
- Picks emptiest CUDA device (no DataParallel by default).

Config (see your train_hier_cvae1.yaml):
data:
  # either a single category across all discovered policies:
  #   category: iid
  # or an explicit experiment with per-policy entries:
  experiment:
    train:
      - {path: "output/data/ego_policy1/iid"}
      - {policy: "ego_policy*", category: "iid"}
    val:
      - {path: "output/data/ego_policy2/iid"}
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

# repo bootstrap
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.models.collate import move_batch, next_frame_collate

LOGGER = logging.getLogger(__name__)


# ---------------- utils / logging / device ----------------

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


# ---------------- data helpers ----------------

def parse_category(cat: str) -> Tuple[str, Optional[str]]:
    """
    'iid' -> ('iid', None)
    'ood_manual+right_bias_mild' -> ('ood_manual', 'right_bias_mild')
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
    """Accepts dict/list/str and returns a flat list of dict entries."""
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
    Build a list of (policy_name, dataset_root) pairs.
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
            for pol_dir in sorted(data_root.glob(pattern)):
                if not pol_dir.is_dir():
                    continue
                policy = pol_dir.name
                rp = category_to_root(data_root, policy, category)
                if (rp / "index.json").exists():
                    resolved.append((policy, rp))
                else:
                    LOGGER.warning("No index.json for (%s, %s) at %s", policy, category, rp)
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
    pairs: List[Tuple[str, Path]],
    split: str,
    window_len: int,
    include_policy_id: bool = True,
) -> Tuple[ConcatDataset, Dict[int, str]]:
    """
    For each (policy, root), wrap NextFrameDataset and concat.
    Also collect a {int_id: policy_name} bookkeeping map.
    """
    datasets: List[NextFrameDataset] = []
    id_map: Dict[int, str] = {}
    next_id = 0

    for policy, root in pairs:
        indexer = EpisodeIndexer.load(root)  # <-- child folder's index.json
        if not any(rec.split == split for rec in indexer.entries):
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


# ---------------- model ----------------

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
        self.window_len = int(window_len)
        self.teams = int(teams)
        self.agents = int(agents)
        self.state_dim = int(state_dim)
        self.latent_dim_global = int(latent_dim_global)
        self.latent_dim_local = int(latent_dim_local)
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.teams * self.agents * self.state_dim, d_model)
        self.positional = PositionalEncoding(d_model, dropout=dropout, max_len=max(self.window_len, 500))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.to_mu_global = nn.Linear(d_model, self.latent_dim_global)
        self.to_logvar_global = nn.Linear(d_model, self.latent_dim_global)

        local_input_dim = self.teams * self.state_dim + self.latent_dim_global
        self.local_hidden = nn.Linear(local_input_dim, d_model)
        self.to_mu_local = nn.Linear(d_model, self.latent_dim_local)
        self.to_logvar_local = nn.Linear(d_model, self.latent_dim_local)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim_global + self.latent_dim_local + self.teams * self.state_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.agents * self.action_dim * 2),
        )

    def _normalize_window(self, window: torch.Tensor) -> torch.Tensor:
        # Accept (B,1,T,teams,agents,state) or (B,T,teams,agents,state)
        if window.dim() >= 3 and window.size(1) == 1:
            window = window.squeeze(1)
        return window

    def encode(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        window = self._normalize_window(window)
        B = window.size(0)
        F = self.teams * self.agents * self.state_dim
        flat = window.view(B, self.window_len, F)
        embedded = self.input_proj(flat)
        embedded = self.positional(embedded)
        encoded = self.encoder(embedded)
        sequence_context = encoded.mean(dim=1)                    # [B, d_model]

        last_frame = window[:, -1]                                # [B, teams, agents, state]
        # build per-agent context by reshaping
        per_agent = last_frame.reshape(B, self.teams * self.agents, self.state_dim)
        local_context = per_agent.view(B, self.agents, self.teams * self.state_dim)
        return {"sequence_context": sequence_context, "local_context": local_context}

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        window: torch.Tensor,
        episode_ids: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(window)
        seq_ctx = enc["sequence_context"]  # [B, d_model]
        device = window.device
        B = seq_ctx.size(0)

        if episode_ids is None:
            episode_ids = torch.arange(B, device=device, dtype=torch.long)
        else:
            episode_ids = episode_ids.to(device=device, dtype=torch.long)

        # aggregate context per episode id
        uniq, inv = torch.unique(episode_ids, sorted=True, return_inverse=True)
        sums = torch.zeros(uniq.size(0), seq_ctx.size(-1), device=device)
        sums.index_add_(0, inv, seq_ctx)
        counts = torch.bincount(inv, minlength=uniq.size(0)).float().to(device).unsqueeze(-1)
        epi_mean = sums / counts.clamp_min(1.0)

        mu_g = self.to_mu_global(epi_mean)
        logvar_g = self.to_logvar_global(epi_mean)
        z_g_epi = self._sample(mu_g, logvar_g, deterministic)
        # map back to batch
        mu_global = mu_g[inv]
        logvar_global = logvar_g[inv]
        z_global = z_g_epi[inv]

        local_in = torch.cat([enc["local_context"], z_global.unsqueeze(1).expand(-1, self.agents, -1)], dim=-1)
        local_h = torch.tanh(self.local_hidden(local_in))
        mu_l = self.to_mu_local(local_h)
        logvar_l = self.to_logvar_local(local_h)
        z_l = self._sample(mu_l, logvar_l, deterministic)

        dec_in = torch.cat([enc["local_context"], z_global.unsqueeze(1).expand(-1, self.agents, -1), z_l], dim=-1)
        out = self.decoder(dec_in).view(B, self.agents, self.action_dim, 2)
        mean = out[..., 0]
        logvar = out[..., 1]

        return {
            "mu_global": mu_global, "logvar_global": logvar_global,
            "mu_local": mu_l, "logvar_local": logvar_l,
            "z_global": z_global, "z_local": z_l,
            "mean": mean, "logvar": logvar,
        }

    def loss(self, outputs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = outputs["mean"]
        logvar = outputs["logvar"]
        recon = 0.5 * ((actions - mean).pow(2) * torch.exp(-logvar) + logvar)
        recon = recon.sum(dim=[1, 2]).mean()

        kl_g = -0.5 * torch.sum(
            1 + outputs["logvar_global"] - outputs["mu_global"].pow(2) - outputs["logvar_global"].exp(), dim=1
        ).mean()
        kl_l = -0.5 * torch.sum(
            1 + outputs["logvar_local"] - outputs["mu_local"].pow(2) - outputs["logvar_local"].exp(), dim=[1, 2]
        ).mean()
        total = recon + kl_g + kl_l
        return {"loss": total, "recon": recon, "kl_global": kl_g, "kl_local": kl_l}


# ---------------- sched / ckpt ----------------

def linear_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-3)
        return max(min_lr / optimizer.defaults["lr"], 1e-3)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0)), ckpt.get("best_val", None)


# ---------------- train / eval ----------------

def train_one_epoch(model, dataloader, optimizer, device, max_grad_norm: float = 5.0):
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl_global": 0.0, "kl_local": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"], episode_ids=batch["episode_id"])
        losses = model.loss(outputs, batch["action"])
        optimizer.zero_grad()
        losses["loss"].backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        for k in totals:
            totals[k] += float(losses[k].detach())
    steps = max(1, len(dataloader))
    return {k: v / steps for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl_global": 0.0, "kl_local": 0.0}
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["window"], episode_ids=batch["episode_id"])
        losses = model.loss(outputs, batch["action"])
        for k in totals:
            totals[k] += float(losses[k].detach())
    steps = max(1, len(dataloader))
    return {k: v / steps for k, v in totals.items()}


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
    run_dir = (run_dir or Path(paths_cfg.get("run_dir", "output/runs/hier_cvae"))).expanduser()
    configure_logging(run_dir)

    # persist config
    try:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp)
    except Exception as e:
        LOGGER.warning("Could not save run config: %s", e)

    # discover policies / roots from config (NO root-level index.json access!)
    T = int(config["window_len"])
    exp_cfg = config.get("data", {}).get("experiment")
    if exp_cfg:
        LOGGER.info("Using explicit experiment configuration.")
        train_pairs = expand_experiment_entries(data_root, exp_cfg.get("train", []))
        val_pairs   = expand_experiment_entries(data_root, exp_cfg.get("val",   exp_cfg.get("train", [])))
    else:
        cat = config.get("data", {}).get("category", "iid")
        LOGGER.info("No experiment config; falling back to category='%s' for all policies.", cat)
        roots = discover_policy_roots(data_root, cat)
        train_pairs = [(p.parent.name if p.parent.name.startswith("ego_policy") else p.name, p) for p in roots]
        val_pairs   = train_pairs

    # datasets & loaders (concat of child folders)
    train_ds, train_idmap = build_concat_dataset(train_pairs, split="train", window_len=T, include_policy_id=True)
    val_ds,   val_idmap   = build_concat_dataset(val_pairs,   split="val",   window_len=T, include_policy_id=True)

    train_loader = DataLoader(
        train_ds, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )

    # Save mapping for debugging later
    try:
        with (run_dir / "policies.json").open("w", encoding="utf-8") as fp:
            json.dump({"train": train_idmap, "val": val_idmap}, fp, indent=2)
    except Exception as e:
        LOGGER.warning("Could not write policies.json: %s", e)

    # model / opt / sched
    model_cfg = config["model"]
    latent_cfg = config["latent_dim"]
    model = HierarchicalCVAE(
        window_len=T,
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

    # CSV logger
    metrics_csv_path = run_dir / "metrics.csv"
    new_file = not metrics_csv_path.exists()
    csv_f = metrics_csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    if new_file:
        csv_writer.writerow(["epoch", "split", "loss", "recon", "kl_global", "kl_local", "lr"])

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
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=max_grad_norm)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        LOGGER.info(
            "Epoch %d train: loss=%.4f recon=%.4f kl_g=%.4f kl_l=%.4f lr=%.6f",
            epoch, train_metrics["loss"], train_metrics["recon"], train_metrics["kl_global"], train_metrics["kl_local"], lr
        )
        csv_writer.writerow([
            epoch, "train",
            f'{train_metrics["loss"]:.6f}',
            f'{train_metrics["recon"]:.6f}',
            f'{train_metrics["kl_global"]:.6f}',
            f'{train_metrics["kl_local"]:.6f}',
            f'{lr:.8f}',
        ])
        csv_f.flush()
        global_step += len(train_loader)

        if (epoch == start_epoch) or (global_step % validate_interval == 0) or (epoch == epochs):
            val_metrics = evaluate(model, val_loader, device)
            LOGGER.info(
                "Epoch %d val:   loss=%.4f recon=%.4f kl_g=%.4f kl_l=%.4f",
                epoch, val_metrics["loss"], val_metrics["recon"], val_metrics["kl_global"], val_metrics["kl_local"]
            )
            csv_writer.writerow([
                epoch, "val",
                f'{val_metrics["loss"]:.6f}',
                f'{val_metrics["recon"]:.6f}',
                f'{val_metrics["kl_global"]:.6f}',
                f'{val_metrics["kl_local"]:.6f}',
                ""
            ])
            csv_f.flush()

            # save last/best
            save_checkpoint(last_ckpt, epoch, global_step, best_val, model, optimizer, scheduler)
            if best_val is None or val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(best_ckpt, epoch, global_step, best_val, model, optimizer, scheduler)

    csv_f.close()
    LOGGER.info("Training complete. Best validation loss: %.4f", best_val if best_val is not None else float("nan"))


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hierarchical CVAE across multiple policies")
    p.add_argument("--config", type=str, default="policyOrProxy/cfg/train_hier_cvae.yaml")
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
