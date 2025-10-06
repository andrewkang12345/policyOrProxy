"""Checkpoint utilities for model training."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class CheckpointManager:
    root: Path
    max_to_keep: int = 5
    best_metric: Optional[float] = None
    direction: str = "min"
    history: Dict[int, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _should_replace(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.direction == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, metric: Optional[float] = None) -> Path:
        path = self.root / f"ckpt_{step:07d}.pt"
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": metric,
        }, path)
        self.history[step] = path
        self._prune()
        LOGGER.info("Saved checkpoint to %s", path)
        if metric is not None and self._should_replace(metric):
            self.best_metric = metric
            best_path = self.root / "best.pt"
            path.replace(best_path)
            self.history[step] = best_path
            LOGGER.info("Updated best checkpoint to %s (metric=%.4f)", best_path, metric)
        return path

    def _prune(self) -> None:
        if self.max_to_keep <= 0:
            return
        if len(self.history) <= self.max_to_keep:
            return
        steps = sorted(self.history.keys())
        for step in steps[:-self.max_to_keep]:
            path = self.history.pop(step)
            if path.exists():
                path.unlink()

    def load_latest(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[int]:
        checkpoints = sorted(self.root.glob("ckpt_*.pt"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        payload = torch.load(latest, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        LOGGER.info("Loaded checkpoint %s", latest)
        return int(payload.get("step", 0))
