"""Grover-style CVAE with episode embeddings and triplet loss."""
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
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([states, actions], dim=-1)
        h = self.network(seq)
        return h.mean(dim=0)


class GroverDecoder(nn.Module):
    def __init__(self, state_dim: int, embed_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        T = states.size(0)
        emb = embedding.expand(T, -1)
        inp = torch.cat([states, emb], dim=-1)
        return self.mlp(inp)


@dataclass
class GroverModel:
    encoder: GroverEncoder
    decoder: GroverDecoder

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()


def sample_triplet(policies: Dict[str, List[Episode]]) -> Tuple[Episode, Episode, Episode]:
    valid_policies = [p for p, eps in policies.items() if len(eps) >= 2]
    if len(valid_policies) < 1 or len(policies) < 2:
        raise RuntimeError("Grover training requires at least two policies and two episodes per policy")
    anchor_policy = random.choice(valid_policies)
    positive_policy = anchor_policy
    negative_policy = random.choice([p for p in policies.keys() if p != anchor_policy])
    anchor, positive = random.sample(policies[anchor_policy], 2)
    negative = random.choice(policies[negative_policy])
    return anchor, positive, negative


def evaluate(policies: Dict[str, List[Episode]], model: GroverModel, margin: float, weight: float) -> Dict[str, float]:
    model.encoder.eval()
    model.decoder.eval()
    total_imitation = 0.0
    total_triplet = 0.0
    count = 0
    with torch.no_grad():
        for anchor_policy, episodes in policies.items():
            if len(episodes) < 2:
                continue
            for idx in range(len(episodes) - 1):
                anchor = episodes[idx]
                positive = episodes[idx + 1]
                negative_policy = random.choice([p for p in policies.keys() if p != anchor_policy])
                negative = random.choice(policies[negative_policy])
                anchor_embed = model.encoder(anchor.states, anchor.actions)
                positive_embed = model.encoder(positive.states, positive.actions)
                negative_embed = model.encoder(negative.states, negative.actions)
                pred_actions = model.decoder(anchor.states, positive_embed)
                imitation_loss = F.mse_loss(pred_actions, anchor.actions)
                triplet_loss = F.triplet_margin_loss(
                    anchor_embed.unsqueeze(0),
                    positive_embed.unsqueeze(0),
                    negative_embed.unsqueeze(0),
                    margin=margin,
                )
                total_imitation += float(imitation_loss)
                total_triplet += float(triplet_loss)
                count += 1
    if count == 0:
        return {"loss": float("nan"), "imitation": float("nan"), "triplet": float("nan")}
    imitation = total_imitation / count
    triplet = total_triplet / count
    return {"loss": imitation + weight * triplet, "imitation": imitation, "triplet": triplet}


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

    state_dim = next(iter(train_policies.values()))[0].states.size(-1)
    action_dim = next(iter(train_policies.values()))[0].actions.size(-1)

    encoder = GroverEncoder(
        input_dim=state_dim + action_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        embed_dim=int(config["model"]["embed_dim"]),
    ).to(device)
    decoder = GroverDecoder(
        state_dim=state_dim,
        embed_dim=int(config["model"]["embed_dim"]),
        hidden_dim=int(config["model"]["decoder_hidden"]),
        action_dim=action_dim,
    ).to(device)
    model = GroverModel(encoder=encoder, decoder=decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["optimizer"]["lr"]))
    triplet_margin = float(config.get("triplet_margin", 1.0))
    triplet_weight = float(config.get("triplet_weight", 1.0))
    batch_size = int(config.get("batch_size", 16))

    best_val = None
    for epoch in range(1, int(config["epochs"]) + 1):
        total_loss = 0.0
        total_imitation = 0.0
        total_triplet = 0.0
        steps = 0
        model.encoder.train()
        model.decoder.train()
        num_batches = int(config.get("steps_per_epoch", 200))
        for _ in range(num_batches):
            optimizer.zero_grad()
            loss_accum = 0.0
            imitation_accum = 0.0
            triplet_accum = 0.0
            for _ in range(batch_size):
                anchor, positive, negative = sample_triplet(train_policies)
                anchor_embed = model.encoder(anchor.states, anchor.actions)
                positive_embed = model.encoder(positive.states, positive.actions)
                negative_embed = model.encoder(negative.states, negative.actions)
                pred_actions = model.decoder(anchor.states, positive_embed)
                imitation_loss = F.mse_loss(pred_actions, anchor.actions)
                triplet_loss = F.triplet_margin_loss(
                    anchor_embed.unsqueeze(0),
                    positive_embed.unsqueeze(0),
                    negative_embed.unsqueeze(0),
                    margin=triplet_margin,
                )
                loss = imitation_loss + triplet_weight * triplet_loss
                loss_accum += loss
                imitation_accum += imitation_loss
                triplet_accum += triplet_loss
            loss_accum /= batch_size
            loss_accum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss_accum.detach())
            total_imitation += float(imitation_accum.detach() / batch_size)
            total_triplet += float(triplet_accum.detach() / batch_size)
            steps += 1
        LOGGER.info(
            "Epoch %d train: loss=%.4f imitation=%.4f triplet=%.4f",
            epoch,
            total_loss / max(steps, 1),
            total_imitation / max(steps, 1),
            total_triplet / max(steps, 1),
        )
        val_metrics = evaluate(val_policies, model, triplet_margin, triplet_weight)
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
                    "triplet_margin": triplet_margin,
                    "triplet_weight": triplet_weight,
                },
                ckpt_path / "best.pt",
            )
    LOGGER.info("Training complete. Best validation loss %.4f", best_val if best_val is not None else float("nan"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Grover-style CVAE")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()
