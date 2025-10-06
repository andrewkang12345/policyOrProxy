"""Opponent window networks used to inject controlled distribution shifts."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn

from policyOrProxy.core.metrics.metrics import wasserstein_distance_torch

LOGGER = logging.getLogger(__name__)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation {name}")


class WindowNN(nn.Module):
    """Neural opponent policy conditioned on recent multi-agent windows."""

    def __init__(
        self,
        window_len: int,
        teams: int,
        agents: int,
        state_dim: int,
        hidden_dim: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        arch: str = "mlp",
        activation: str = "gelu",
        max_speed: float = 3.5,
        identifier: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.window_len = window_len
        self.teams = teams
        self.agents = agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.arch = arch
        self.max_speed = max_speed
        self.identifier = identifier or f"opp_{arch}"
        self.activation_name = activation
        self.net = self._build_network()

    def _build_network(self) -> nn.Module:
        if self.arch == "mlp":
            return self._build_mlp()
        if self.arch == "transformer":
            return self._build_transformer()
        raise ValueError(f"Unsupported architecture {self.arch}")

    def _build_mlp(self) -> nn.Module:
        inp = self.window_len * self.teams * self.agents * self.state_dim
        layers = []
        current = inp
        for _ in range(self.layers - 1):
            layers.append(nn.Linear(current, self.hidden_dim))
            layers.append(_activation(self.activation_name))
            layers.append(nn.Dropout(self.dropout))
            current = self.hidden_dim
        layers.append(nn.Linear(current, self.agents * 2))
        return nn.Sequential(*layers)

    def _build_transformer(self) -> nn.Module:
        token_dim = self.state_dim * self.teams
        self.input_proj = nn.Linear(self.state_dim * self.teams, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation=self.activation_name,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.output_head = nn.Linear(self.hidden_dim, self.agents * 2)
        return encoder

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        batch = window.shape[0]
        flat = window.reshape(batch, self.window_len, self.teams * self.agents * self.state_dim)
        if self.arch == "mlp":
            flattened = flat.reshape(batch, -1)
            logits = self.net(flattened)
        else:
            tokens = flat.reshape(batch, self.window_len, self.teams, self.agents, self.state_dim)
            tokens = tokens.permute(0, 3, 1, 2, 4).contiguous()
            tokens = tokens.reshape(batch * self.agents, self.window_len, self.teams * self.state_dim)
            embedded = self.input_proj(tokens)
            encoded = self.net(embedded)
            pooled = encoded.mean(dim=1)
            logits = self.output_head(pooled)
            logits = logits.reshape(batch, self.agents * 2)
        actions = torch.tanh(logits) * self.max_speed
        actions = actions.view(batch, self.agents, 2)
        return actions

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del deterministic
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(window).float().unsqueeze(0)
            actions = self.forward(tensor)
        return actions.squeeze(0).cpu().numpy()


@dataclass
class DivergenceLoss:
    baseline_state_embeddings: torch.Tensor
    baseline_action_embeddings: torch.Tensor
    power: float = 2.0

    def __post_init__(self) -> None:
        self.baseline_state_embeddings = self.baseline_state_embeddings.float()
        self.baseline_action_embeddings = self.baseline_action_embeddings.float()

    def compute(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        state_emb = states[:, -1].reshape(states.size(0), -1)
        action_emb = actions.reshape(actions.size(0), -1)
        state_div = wasserstein_distance_torch(state_emb, self.baseline_state_embeddings, p=self.power)
        action_div = wasserstein_distance_torch(action_emb, self.baseline_action_embeddings, p=self.power)
        return {"state": state_div, "action": action_div}


def optimize_to_target(
    model: WindowNN,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    divergence_loss: DivergenceLoss,
    targets: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    log_every: int = 50,
    tol: float = 0.05,
    closed_loop_eval: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    eval_every: int = 200,
) -> Dict[str, float]:
    """Tune the opponent network to hit target Wasserstein levels."""
    device = next(model.parameters()).device
    model.train()
    running = {key: 0.0 for key in targets}
    closed_loop = {key: float("inf") for key in targets}
    for step, batch in enumerate(dataloader, start=1):
        if step > max_steps:
            break
        windows = batch["window"].to(device)
        optimizer.zero_grad()
        actions = model(windows)
        divs = divergence_loss.compute(windows, actions)
        loss = torch.zeros(1, device=device)
        for key, target in targets.items():
            diff = divs[key] - target
            running[key] = 0.9 * running[key] + 0.1 * float(divs[key].detach())
            loss = loss + diff.pow(2)
        loss = loss.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0:
            LOGGER.info("step %d: state=%.3f target=%.3f action=%.3f target=%.3f", step, running["state"], targets["state"], running["action"], targets["action"])
        if closed_loop_eval is not None and eval_every > 0 and step % eval_every == 0:
            try:
                metrics = closed_loop_eval(model)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Closed-loop evaluation failed: %s", exc)
            else:
                for key in targets:
                    closed_loop[key] = float(metrics.get(key, closed_loop[key]))
                LOGGER.info(
                    "closed-loop step %d: state=%.3f target=%.3f action=%.3f target=%.3f",
                    step,
                    closed_loop["state"],
                    targets["state"],
                    closed_loop["action"],
                    targets["action"],
                )
                if all(abs(closed_loop[k] - targets[k]) < tol for k in targets):
                    LOGGER.info("Closed-loop targets reached within tolerance after %d steps", step)
                    break
        if all(abs(running[k] - targets[k]) < tol for k in targets) and closed_loop_eval is None:
            LOGGER.info("Reached targets within tolerance after %d steps", step)
            break
    if closed_loop_eval is not None:
        try:
            metrics = closed_loop_eval(model)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Final closed-loop evaluation failed: %s", exc)
        else:
            for key in targets:
                closed_loop[key] = float(metrics.get(key, closed_loop[key]))
    return {"open_loop": running, "closed_loop": closed_loop}
