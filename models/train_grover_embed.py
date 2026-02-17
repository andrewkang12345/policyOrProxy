#!/usr/bin/env python3
"""
Train Grover et al. (2018) style policy representations.

Option B semantics:
- The authoritative temporal window length is NPZ windows.shape[1] (T).
- This script DOES NOT read config["window_len"].
- T is inferred from TRAIN dataset and enforced consistent across roots and splits.

Expected batch keys from NextFrameDataset/next_frame_collate:
  - window: [B, T, teams, agents, state_dim]
  - action: [B, agents, action_dim]
  - policy_id: [B]   (we inject stable IDs via WithPolicyId wrapper)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable, Union
import sys

import yaml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Sampler

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

def infer_concat_window_len(ds: ConcatDataset) -> int:
    if not hasattr(ds, "datasets") or not ds.datasets:
        raise ValueError("ConcatDataset is empty; cannot infer window length.")
    first = ds.datasets[0]
    T = getattr(first, "window_len", None)
    if T is None:
        raise ValueError("Underlying dataset did not set window_len; cannot infer.")
    for sub in ds.datasets[1:]:
        if getattr(sub, "window_len", None) != T:
            raise ValueError(
                f"Mismatched window_len across datasets: {getattr(sub,'window_len',None)} vs {T}"
            )
    return int(T)


class WithPolicyId(torch.utils.data.Dataset):
    """Wrap a dataset to force a stable per-policy integer ID."""
    def __init__(self, base_ds, policy_int_id: int):
        self.base = base_ds
        self.pid = int(policy_int_id)
        self.window_len = getattr(base_ds, "window_len", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]
        sample["policy_id"] = torch.tensor(self.pid, dtype=torch.long)
        return sample


def parse_category(cat: str) -> Tuple[str, Optional[str]]:
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
) -> Tuple[ConcatDataset, Dict[int, str]]:
    datasets: List[WithPolicyId] = []
    id_map: Dict[int, str] = {}
    next_id = 0
    for root in roots:
        # infer policy name
        if root.name in ("iid", "train", "val", "test"):
            policy = root.parent.name
        else:
            policy = root.parent.name if (root.parent.name.startswith("ego_policy")) else root.name

        indexer = EpisodeIndexer.load(root)
        has_split = any(rec.split == split for rec in indexer.entries)
        if not has_split:
            LOGGER.warning("Skipping %s — no '%s' split found.", root, split)
            continue

        base_ds = NextFrameDataset(
            root=root,
            indexer=indexer,
            split=split,
            device=None,
            preload=False,
            include_policy_id=False,   # we inject stable IDs
        )
        ds = WithPolicyId(base_ds, policy_int_id=next_id)
        id_map[next_id] = policy
        next_id += 1
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No datasets found for split '{split}' across provided roots.")
    return ConcatDataset(datasets), id_map


# ---------- grouped batch sampler ----------

class PolicyGroupedBatchSampler(Sampler[List[int]]):
    """
    Yields batches that contain at least two items from the same policy id.
    Assumes dataset is a ConcatDataset of WithPolicyId sub-datasets.
    Avoids indexing every sample by using sub-dataset spans.
    """
    def __init__(self, concat_ds: ConcatDataset, batch_size: int, seed: int = 0):
        assert isinstance(concat_ds, ConcatDataset), "PolicyGroupedBatchSampler expects a ConcatDataset"
        self.ds = concat_ds
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)

        self.pid_spans: Dict[int, List[int]] = {}  # pid -> list of indices
        self.idx2pid: Dict[int, int] = {}

        prev_cum = 0
        for sub, cum in zip(self.ds.datasets, self.ds.cumulative_sizes):
            assert hasattr(sub, "pid"), "Sub-datasets must be WithPolicyId wrappers exposing .pid"
            pid = int(sub.pid)
            indices = list(range(prev_cum, cum))
            self.pid_spans.setdefault(pid, []).extend(indices)
            for ix in indices:
                self.idx2pid[ix] = pid
            prev_cum = cum

        self.all_indices = list(range(prev_cum))
        self._epoch_state_reset()

    def _epoch_state_reset(self):
        self.remaining = set(self.all_indices)
        self.left_by_pid = {pid: set(ixs) for pid, ixs in self.pid_spans.items()}

    def __iter__(self):
        self._epoch_state_reset()
        rng = self.rng

        while self.remaining:
            candidate_pids = [pid for pid, s in self.left_by_pid.items() if len(s) >= 2]
            batch: List[int] = []

            if candidate_pids:
                pid = rng.choice(candidate_pids)
                take_k = min(len(self.left_by_pid[pid]), max(2, self.batch_size // 4))
                pick = rng.sample(list(self.left_by_pid[pid]), take_k)
                batch.extend(pick)

            rest_needed = self.batch_size - len(batch)
            if rest_needed > 0:
                others = list(self.remaining.difference(batch))
                if others:
                    fill = rng.sample(others, min(rest_needed, len(others)))
                    batch.extend(fill)

            for b in batch:
                self.remaining.discard(b)
                pid_b = self.idx2pid[b]
                self.left_by_pid[pid_b].discard(b)

            if batch:
                yield batch

    def __len__(self):
        import math
        return math.ceil(len(self.all_indices) / self.batch_size)


# ---------- model ----------

class EpisodeEncoder(nn.Module):
    """
    f_theta: map an episode window -> embedding.
    MLP per timestep + mean-pool.
    """
    def __init__(self, teams: int, agents: int, state_dim: int, action_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        in_dim = teams * agents * state_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        if window.dim() == 5:
            B, T, teams, agents, state = window.shape
            assert teams == self.teams and agents == self.agents and state == self.state_dim, (
                f"Unexpected window shape {tuple(window.shape)} vs config"
            )
            x = window.view(B, T, self.teams * self.agents * self.state_dim)
        elif window.dim() == 3:
            x = window
        else:
            raise ValueError(f"Unexpected window shape {tuple(window.shape)}")

        x = x.to(dtype=torch.float32)
        x = self.norm(x)
        h = self.mlp(x)            # [B, T, embed_dim]
        e = h.mean(dim=1)          # [B, embed_dim]
        return e


class ConditionalPolicy(nn.Module):
    """
    pi(a | o, z): Gaussian policy conditioned on embedding z and observation o (last-timestep obs).
    """
    def __init__(self, teams: int, agents: int, state_dim: int, embed_dim: int, hidden_dim: int, action_dim: int, dropout: float):
        super().__init__()
        self.teams = teams
        self.agents = agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        obs_dim = teams * agents * state_dim
        in_dim = obs_dim + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.to_mean   = nn.Linear(hidden_dim, agents * action_dim)
        self.to_logvar = nn.Linear(hidden_dim, agents * action_dim)

    def forward(self, obs_last: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs_last.dim() == 4:
            B, teams, agents, state = obs_last.shape
            assert teams == self.teams and agents == self.agents and state == self.state_dim
            o = obs_last.view(B, self.teams * self.agents * self.state_dim)
        elif obs_last.dim() == 2:
            o = obs_last
        else:
            raise ValueError(f"Unexpected obs_last shape {tuple(obs_last.shape)}")

        o = o.to(dtype=torch.float32)
        z = z.to(dtype=torch.float32)
        h = self.net(torch.cat([o, z], dim=-1))
        mean   = self.to_mean(h).view(o.size(0), self.agents, self.action_dim)
        logvar = self.to_logvar(h).view(o.size(0), self.agents, self.action_dim)
        return mean, logvar


class GroverModel(nn.Module):
    def __init__(
        self,
        teams: int,
        agents: int,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        encoder_hidden: int,
        policy_hidden: int,
        dropout: float,
        gaussian_min_logvar: float = -6.0,
    ):
        super().__init__()
        self.encoder = EpisodeEncoder(teams, agents, state_dim, action_dim, encoder_hidden, embed_dim, dropout)
        self.policy  = ConditionalPolicy(teams, agents, state_dim, embed_dim, policy_hidden, action_dim, dropout)
        self.gaussian_min_logvar = float(gaussian_min_logvar)

    def embed(self, window: torch.Tensor) -> torch.Tensor:
        return self.encoder(window)

    def act_params(self, obs_last: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.policy(obs_last, z)
        logvar = torch.clamp(logvar, min=self.gaussian_min_logvar, max=6.0)
        return mean, logvar

    @staticmethod
    def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        inv_var = torch.exp(-logvar)
        return 0.5 * ((target - mean).pow(2) * inv_var + logvar + eps)

    @staticmethod
    def triplet_soft_loss(p, n, r):
        d_rn = torch.sum((r - n) ** 2, dim=-1)
        d_rp = torch.sum((r - p) ** 2, dim=-1)
        return torch.nn.functional.softplus(d_rp - d_rn)

    def imitation_loss(self, obs_last: torch.Tensor, action: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.act_params(obs_last, z_ref)
        nll = self.gaussian_nll(mean, logvar, action).sum(dim=[1, 2]).mean()
        return nll

    def identification_loss(self, e_pos: torch.Tensor, e_neg: torch.Tensor, e_ref: torch.Tensor) -> torch.Tensor:
        d = self.triplet_soft_loss(e_pos, e_neg, e_ref)
        return d.mean()


# ---------- batch pairing helpers ----------

def flatten_last_obs(window: torch.Tensor) -> torch.Tensor:
    if window.dim() == 5:
        last = window[:, -1]
        return last.view(last.size(0), -1)
    elif window.dim() == 3:
        return window[:, -1]
    else:
        raise ValueError(f"Unexpected window shape {tuple(window.shape)}")


def build_triples(policy_ids: torch.Tensor) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    ids = policy_ids.cpu().tolist()
    by_policy: Dict[int, List[int]] = {}
    for i, pid in enumerate(ids):
        by_policy.setdefault(pid, []).append(i)
    all_indices = list(range(len(ids)))
    for pid, idxs in by_policy.items():
        if len(idxs) < 2:
            continue
        other = [i for i in all_indices if policy_ids[i] != pid]
        if not other:
            continue
        for t in range(len(idxs)):
            i = idxs[t]
            j = idxs[(t + 1) % len(idxs)]
            k = random.choice(other)
            triples.append((i, j, k))
    return triples


def _first_nonfinite(name_tensor_pairs: Dict[str, torch.Tensor]) -> Optional[str]:
    for name, t in name_tensor_pairs.items():
        if not torch.isfinite(t).all():
            return name
    return None


# ---------- train / eval ----------

def train_one_epoch(model, dataloader, optimizer, device, lambda_id: float, max_grad_norm: float = 5.0):
    model.train()
    m_tot = {"total": 0.0, "imitation": 0.0, "identification": 0.0}
    steps = 0
    for batch in dataloader:
        batch = move_batch(batch, device)
        window = batch["window"]
        action = batch["action"]
        policy_ids = batch["policy_id"].long()

        bad = _first_nonfinite({"window": window, "action": action})
        if bad:
            LOGGER.warning("Non-finite values in %s; skipping batch", bad)
            continue

        triples = build_triples(policy_ids)
        if not triples:
            continue

        embeds = model.embed(window)
        bad = _first_nonfinite({"embeds": embeds})
        if bad:
            LOGGER.warning("Non-finite values after encoder (%s); skipping batch", bad)
            continue

        pos_idx = torch.tensor([i for (i, j, k) in triples], device=device, dtype=torch.long)
        ref_idx = torch.tensor([j for (i, j, k) in triples], device=device, dtype=torch.long)
        neg_idx = torch.tensor([k for (i, j, k) in triples], device=device, dtype=torch.long)

        obs_last = flatten_last_obs(window[pos_idx])
        act_pos  = action[pos_idx]
        z_ref    = embeds[ref_idx]

        imitation = model.imitation_loss(obs_last, act_pos, z_ref)
        id_loss   = model.identification_loss(embeds[pos_idx], embeds[neg_idx], embeds[ref_idx])
        loss      = imitation + lambda_id * id_loss

        if not torch.isfinite(loss):
            LOGGER.warning("Non-finite total loss; skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        m_tot["total"] += float(loss.detach())
        m_tot["imitation"] += float(imitation.detach())
        m_tot["identification"] += float(id_loss.detach())
        steps += 1

    for k in list(m_tot.keys()):
        m_tot[k] = m_tot[k] / max(steps, 1)
    return m_tot


@torch.no_grad()
def evaluate(model, dataloader, device, lambda_id: float):
    model.eval()
    m_tot = {"total": 0.0, "imitation": 0.0, "identification": 0.0}
    steps = 0
    for batch in dataloader:
        batch = move_batch(batch, device)
        window = batch["window"]
        action = batch["action"]
        policy_ids = batch["policy_id"].long()

        bad = _first_nonfinite({"window": window, "action": action})
        if bad:
            continue

        triples = build_triples(policy_ids)
        if not triples:
            continue

        embeds = model.embed(window)
        bad = _first_nonfinite({"embeds": embeds})
        if bad:
            continue

        pos_idx = torch.tensor([i for (i, j, k) in triples], device=device, dtype=torch.long)
        ref_idx = torch.tensor([j for (i, j, k) in triples], device=device, dtype=torch.long)
        neg_idx = torch.tensor([k for (i, j, k) in triples], device=device, dtype=torch.long)

        obs_last = flatten_last_obs(window[pos_idx])
        act_pos  = action[pos_idx]
        z_ref    = embeds[ref_idx]

        imitation = model.imitation_loss(obs_last, act_pos, z_ref)
        id_loss   = model.identification_loss(embeds[pos_idx], embeds[neg_idx], embeds[ref_idx])
        loss      = imitation + lambda_id * id_loss

        if not torch.isfinite(loss):
            continue

        m_tot["total"] += float(loss.detach())
        m_tot["imitation"] += float(imitation.detach())
        m_tot["identification"] += float(id_loss.detach())
        steps += 1

    for k in list(m_tot.keys()):
        m_tot[k] = m_tot[k] / max(steps, 1)
    return m_tot


# ---------- checkpointing ----------

def save_checkpoint(
    path: Path,
    epoch: int,
    best_val: Optional[float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    state = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
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
    map_location: str | torch.device,
) -> tuple[int, Optional[float]]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
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
    best_val = ckpt.get("best_val", None)
    return epoch, best_val


# ---------- orchestration ----------

def train_loop(config: Dict, data_root: Path | None = None, run_dir: Path | None = None, resume: bool = True) -> None:
    prefer_free_gpu = bool(config.get("prefer_free_gpu", True))
    if torch.cuda.is_available() and config.get("device", "auto") != "cpu":
        dev_idx = pick_emptiest_cuda_device() if prefer_free_gpu else 0
        torch.cuda.set_device(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
    else:
        device = torch.device("cpu")

    set_seed(int(config["seed"]))
    paths_cfg = config.get("paths", {})
    data_root = (data_root or Path(paths_cfg.get("data_root", "output/data"))).expanduser()
    run_dir = (run_dir or Path(paths_cfg.get("run_dir", "output/runs/grover_embed"))).expanduser()
    configure_logging(run_dir)
    LOGGER.info("Using device: %s", device)

    try:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as fp:
            yaml.safe_dump(config, fp)
    except Exception as e:
        LOGGER.warning("Could not save run config: %s", e)

    exp_cfg = config.get("data", {}).get("experiment")
    if exp_cfg:
        LOGGER.info("Using explicit experiment configuration.")
        train_pairs = expand_experiment_entries(data_root, exp_cfg.get("train", []))
        val_pairs   = expand_experiment_entries(data_root, exp_cfg.get("val",   exp_cfg.get("train", [])))
        train_roots = [rp for (_pol, rp) in train_pairs]
        val_roots   = [rp for (_pol, rp) in val_pairs]
    else:
        cat = config.get("data", {}).get("category", "iid")
        LOGGER.info("No experiment config; falling back to category='%s' for all policies.", cat)
        roots = discover_policy_roots(data_root, cat)
        train_roots = roots
        val_roots   = roots

    train_ds, train_idmap = build_concat_dataset(train_roots, split="train")
    T = infer_concat_window_len(train_ds)
    val_ds,   val_idmap   = build_concat_dataset(val_roots,   split="val")
    _ = infer_concat_window_len(val_ds)

    LOGGER.info("Using inferred window_len T=%d (from stored NPZ windows).", T)

    try:
        with (run_dir / "policies.json").open("w", encoding="utf-8") as fp:
            json.dump({"train": train_idmap, "val": val_idmap}, fp, indent=2)
    except Exception as e:
        LOGGER.warning("Could not write policies.json: %s", e)

    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    seed = int(config.get("seed", 0))

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PolicyGroupedBatchSampler(train_ds, batch_size=batch_size, seed=seed),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PolicyGroupedBatchSampler(val_ds, batch_size=batch_size, seed=seed + 1),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=next_frame_collate,
    )

    mcfg = config["model"]
    model = GroverModel(
        teams=int(mcfg["teams"]),
        agents=int(mcfg["agents"]),
        state_dim=int(mcfg["state_dim"]),
        action_dim=int(mcfg["action_dim"]),
        embed_dim=int(mcfg["embed_dim"]),
        encoder_hidden=int(mcfg["encoder_hidden"]),
        policy_hidden=int(mcfg["policy_hidden"]),
        dropout=float(mcfg["dropout"]),
        gaussian_min_logvar=float(config.get("loss", {}).get("gaussian_min_logvar", -6.0)),
    ).to(device)

    ocfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ocfg["lr"]),
        betas=tuple(ocfg.get("betas", [0.9, 0.999])),
        weight_decay=float(ocfg.get("weight_decay", 0.0)),
    )

    lambda_id = float(config.get("loss", {}).get("lambda_id", 0.1))
    max_grad_norm = float(config.get("max_grad_norm", 5.0))
    epochs = int(config["epochs"])

    metrics_csv_path = run_dir / "metrics.csv"
    new_file = not metrics_csv_path.exists()
    csv_f = metrics_csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    if new_file:
        csv_writer.writerow(["epoch", "split", "total", "imitation", "identification", "lr"])

    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"
    start_epoch = 1
    best_val: Optional[float] = None

    if resume and last_ckpt.exists():
        LOGGER.info("Resuming from %s", last_ckpt)
        e, b = load_checkpoint(last_ckpt, model, optimizer, map_location=device)
        start_epoch = e + 1
        best_val = b

    for epoch in range(start_epoch, epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device, lambda_id, max_grad_norm=max_grad_norm)
        lr = optimizer.param_groups[0]["lr"]
        LOGGER.info(
            "Epoch %d train: total=%.4f imitation=%.4f identification=%.4f lr=%.6f",
            epoch, train_m["total"], train_m["imitation"], train_m["identification"], lr
        )
        csv_writer.writerow([epoch, "train",
                             f'{train_m["total"]:.6f}',
                             f'{train_m["imitation"]:.6f}',
                             f'{train_m["identification"]:.6f}',
                             f'{lr:.8f}'])
        csv_f.flush()

        val_m = evaluate(model, val_loader, device, lambda_id)
        LOGGER.info(
            "Epoch %d val:   total=%.4f imitation=%.4f identification=%.4f",
            epoch, val_m["total"], val_m["imitation"], val_m["identification"]
        )
        csv_writer.writerow([epoch, "val",
                             f'{val_m["total"]:.6f}',
                             f'{val_m["imitation"]:.6f}',
                             f'{val_m["identification"]:.6f}',
                             ""])
        csv_f.flush()

        save_checkpoint(last_ckpt, epoch, best_val, model, optimizer)
        if (best_val is None) or (val_m["total"] < best_val):
            best_val = val_m["total"]
            save_checkpoint(best_ckpt, epoch, best_val, model, optimizer)

    csv_f.close()
    LOGGER.info("Training complete. Best validation total loss: %.4f", best_val if best_val is not None else float("nan"))


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Grover et al. policy representations")
    p.add_argument("--config", type=str, default="policyOrProxy/cfg/train_grover_embed.yaml")
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
