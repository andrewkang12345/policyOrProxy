"""Grover-style policy representation: deterministic episode embeddings + conditioned imitation + triplet loss."""
from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Episode:
    states: torch.Tensor  # (T, state_dim)
    actions: torch.Tensor  # (T, action_dim)
    policy: str


def load_yaml(path: Path) -> Dict:
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


def flatten_state(window: np.ndarray) -> np.ndarray:
    # take the most recent frame and flatten agent features
    latest = window[-1]  # shape (teams, agents, state_dim)
    return latest.reshape(-1)


def load_episodes(root: Path, indexer: EpisodeIndexer, split: str, device: torch.device) -> Dict[str, List[Episode]]:
    episodes: Dict[str, List[Episode]] = {}
    records = list(indexer.iter_split(split))
    for record in records:
        data = np.load(root / record.path, allow_pickle=False)
        windows = data["windows"]  # (T, window_len, teams, agents, state_dim)
        actions = data["ego_actions"]  # (T, agents, action_dim)
        T = windows.shape[0]
        state_features = np.stack([flatten_state(windows[t]) for t in range(T)], axis=0)
        action_features = actions.reshape(T, -1)
        episode = Episode(
            states=torch.from_numpy(state_features).float().to(device),
            actions=torch.from_numpy(action_features).float().to(device),
            policy=record.policy_id or "unknown",
        )
        episodes.setdefault(episode.policy, []).append(episode)
    return episodes


class GroverEncoder(nn.Module):
    """Deterministic episode encoder: per-step MLP over [state, action], mean-pooled over time."""
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, l2_normalize: bool = True) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.l2_normalize = l2_normalize

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: (T, S), actions: (T, A)
        seq = torch.cat([states, actions], dim=-1)  # (T, S+A)
        h = self.network(seq)                       # (T, D)
        z = h.mean(dim=0)                           # (D,)
        if self.l2_normalize:
            z = F.normalize(z, dim=0)
        return z


class GaussianPolicyDecoder(nn.Module):
    """
    Policy network p(a|o, z): outputs mean and (diagonal) log-variance for a Gaussian over actions.
    This turns imitation into (negative) log-likelihood rather than MSE.
    """
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head_mean = nn.Linear(hidden_dim, action_dim)
        self.head_logvar = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor, embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # states: (T, S), embedding: (D,)
        T = states.size(0)
        emb = embedding.expand(T, -1)              # (T, D)
        inp = torch.cat([states, emb], dim=-1)     # (T, S+D)
        h = self.backbone(inp)                     # (T, H)
        mean = self.head_mean(h)                   # (T, A)
        logvar = self.head_logvar(h).clamp(min=-10.0, max=2.0)  # stability
        return mean, logvar


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    NLL for diagonal Gaussian across action dims, averaged over time.
    mean/logvar/target: (T, A)
    """
    inv_var = torch.exp(-logvar)
    per_dim = 0.5 * ((target - mean) ** 2 * inv_var + logvar)   # (T, A)
    return per_dim.sum(dim=-1).mean()                           # scalar


@dataclass
class GroverModel:
    encoder: GroverEncoder
    decoder: GaussianPolicyDecoder

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()


def sample_ref_pos_neg(policies: Dict[str, List[Episode]]) -> Tuple[Episode, Episode, Episode]:
    """
    Returns: reference (e*), positive (e+) from SAME policy (distinct episodes),
             negative (e-) from a DIFFERENT policy.
    """
    valid_policies = [p for p, eps in policies.items() if len(eps) >= 2]
    if len(valid_policies) < 1 or len(policies) < 2:
        raise RuntimeError("Training requires at least two policies and two episodes per policy")
    ref_policy = random.choice(valid_policies)
    ref, pos = random.sample(policies[ref_policy], 2)
    neg_policy = random.choice([p for p in policies.keys() if p != ref_policy])
    neg = random.choice(policies[neg_policy])
    return ref, pos, neg


def soft_triplet_loss(z_ref: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
    """
    Paper-style smooth objective (squared-softmax flavor):
    d = (1 + exp(||r - n||^2 - ||r - p||^2))^-2
    We minimize 1 - d to push positive closer than negative.
    """
    d_pos = torch.sum((z_ref - z_pos) ** 2)
    d_neg = torch.sum((z_ref - z_neg) ** 2)
    score = torch.pow(1.0 + torch.exp(d_neg - d_pos), -2)
    return 1.0 - score  # lower is better when pos closer than neg


def evaluate(
    policies: Dict[str, List[Episode]],
    model: GroverModel,
    triplet_weight: float,
    triplet_type: str = "soft",
    margin: float = 1.0,
) -> Dict[str, float]:
    model.encoder.eval()
    model.decoder.eval()
    total_imitation = 0.0
    total_triplet = 0.0
    count = 0
    with torch.no_grad():
        for ref_policy, episodes in policies.items():
            if len(episodes) < 2:
                continue
            for idx in range(len(episodes) - 1):
                ref = episodes[idx]
                pos = episodes[idx + 1]
                neg_policy = random.choice([p for p in policies.keys() if p != ref_policy])
                neg = random.choice(policies[neg_policy])

                z_ref = model.encoder(ref.states, ref.actions)
                z_pos = model.encoder(pos.states, pos.actions)
                z_neg = model.encoder(neg.states, neg.actions)

                mean, logvar = model.decoder(pos.states, z_ref)  # imitate e+ conditioned on e*
                imitation_loss = gaussian_nll(mean, logvar, pos.actions)

                if triplet_type == "margin":
                    t_loss = F.triplet_margin_loss(
                        z_ref.unsqueeze(0), z_pos.unsqueeze(0), z_neg.unsqueeze(0), margin=margin
                    )
                else:
                    t_loss = soft_triplet_loss(z_ref, z_pos, z_neg)

                total_imitation += float(imitation_loss)
                total_triplet += float(t_loss)
                count += 1
    if count == 0:
        return {"loss": float("nan"), "imitation": float("nan"), "triplet": float("nan")}
    imitation = total_imitation / count
    triplet = total_triplet / count
    return {"loss": imitation + triplet_weight * triplet, "imitation": imitation, "triplet": triplet}


def train(config: Dict) -> None:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    data_root = Path(config["paths"]["data_root"]).expanduser()
    run_dir = Path(config["paths"]["run_dir"]).expanduser()
    configure_logging(run_dir)

    root_entries = config["paths"].get("data_roots")
    if root_entries:
        roots = [Path(p).expanduser() for p in root_entries]
    else:
        roots = [data_root]

    def aggregate(split: str) -> Dict[str, List[Episode]]:
        combined: Dict[str, List[Episode]] = {}
        for root in roots:
            indexer = EpisodeIndexer.load(root)
            subset = load_episodes(root, indexer, split=split, device=device)
            for policy, episodes in subset.items():
                combined.setdefault(policy, []).extend(episodes)
        return combined

    train_policies = aggregate("train")
    val_policies = aggregate("val")

    # Infer dims
    first_episode = next(iter(train_policies.values()))[0]
    state_dim = first_episode.states.size(-1)
    action_dim = first_episode.actions.size(-1)

    encoder = GroverEncoder(
        input_dim=state_dim + action_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        embed_dim=int(config["model"]["embed_dim"]),
        l2_normalize=bool(config["model"].get("l2_normalize", True)),
    ).to(device)
    decoder = GaussianPolicyDecoder(
        state_dim=state_dim,
        embed_dim=int(config["model"]["embed_dim"]),
        hidden_dim=int(config["model"]["decoder_hidden"]),
        action_dim=action_dim,
    ).to(device)
    model = GroverModel(encoder=encoder, decoder=decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr"]))
    triplet_weight = float(config.get("triplet_weight", 1.0))
    triplet_type = str(config.get("triplet_type", "soft"))  # "soft" or "margin"
    triplet_margin = float(config.get("triplet_margin", 1.0))
    batch_size = int(config.get("batch_size", 16))

    best_val = None
    steps_per_epoch = int(config.get("steps_per_epoch", 200))
    epochs = int(config["epochs"])

    for epoch in range(1, epochs + 1):
        model.encoder.train()
        model.decoder.train()
        total_loss = 0.0
        total_imitation = 0.0
        total_triplet = 0.0
        steps = 0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            loss_accum = 0.0
            imitation_accum = 0.0
            triplet_accum = 0.0

            for _ in range(batch_size):
                ref, pos, neg = sample_ref_pos_neg(train_policies)

                z_ref = model.encoder(ref.states, ref.actions)
                z_pos = model.encoder(pos.states, pos.actions)
                z_neg = model.encoder(neg.states, neg.actions)

                mean, logvar = model.decoder(pos.states, z_ref)  # imitate e+ conditioned on e*
                imitation_loss = gaussian_nll(mean, logvar, pos.actions)

                if triplet_type == "margin":
                    t_loss = F.triplet_margin_loss(
                        z_ref.unsqueeze(0), z_pos.unsqueeze(0), z_neg.unsqueeze(0), margin=triplet_margin
                    )
                else:
                    t_loss = soft_triplet_loss(z_ref, z_pos, z_neg)

                loss = imitation_loss + triplet_weight * t_loss

                loss_accum += loss
                imitation_accum += imitation_loss
                triplet_accum += t_loss

            loss_accum /= batch_size
            loss_accum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss_accum.detach())
            total_imitation += float((imitation_accum / batch_size).detach())
            total_triplet += float((triplet_accum / batch_size).detach())
            steps += 1

        LOGGER.info(
            "Epoch %d train: loss=%.4f imitation=%.4f triplet=%.4f",
            epoch,
            total_loss / max(steps, 1),
            total_imitation / max(steps, 1),
            total_triplet / max(steps, 1),
        )

        val_metrics = evaluate(
            val_policies, model, triplet_weight=triplet_weight, triplet_type=triplet_type, margin=triplet_margin
        )
        LOGGER.info(
            "Epoch %d val: loss=%.4f imitation=%.4f triplet=%.4f",
            epoch,
            val_metrics["loss"],
            val_metrics["imitation"],
            val_metrics["triplet"],
        )

        current = val_metrics["loss"]
        if best_val is None or current < best_val:
            best_val = current
            ckpt_path = run_dir / "checkpoints"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "triplet_weight": triplet_weight,
                    "triplet_type": triplet_type,
                    "triplet_margin": triplet_margin,
                },
                ckpt_path / "best.pt",
            )

    LOGGER.info("Training complete. Best validation loss %.4f", best_val if best_val is not None else float("nan"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Grover-style policy representation model")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()