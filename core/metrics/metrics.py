"""Metrics used across training and evaluation."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def ade_fde(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions = predictions.float()
    targets = targets.float()
    errors = torch.linalg.norm(predictions - targets, dim=-1)
    ade = errors.mean().item()
    if errors.ndim >= 2:
        fde = errors[..., -1].mean().item()
    else:
        fde = errors.mean().item()
    return {"ade": ade, "fde": fde}


def action_smoothness(actions: torch.Tensor) -> float:
    actions = actions.float()
    diffs = actions[:, 1:] - actions[:, :-1]
    smooth = torch.linalg.norm(diffs, dim=-1).mean().item()
    return smooth


def collision_rate(positions: torch.Tensor, arena) -> float:
    flat = positions.detach().cpu().numpy().reshape(-1, 2)
    mask = arena.contains(flat)
    collisions = (~mask).sum()
    return collisions / max(flat.shape[0], 1)


def wasserstein_distance_numpy(samples_a: np.ndarray, samples_b: np.ndarray, p: float = 2.0) -> float:
    tensor_a = torch.from_numpy(samples_a).float()
    tensor_b = torch.from_numpy(samples_b).float()
    return float(wasserstein_distance_torch(tensor_a, tensor_b, p=p))


def wasserstein_distance_torch(samples_a: torch.Tensor, samples_b: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    if samples_a.ndim == 1:
        samples_a = samples_a.unsqueeze(0)
    if samples_b.ndim == 1:
        samples_b = samples_b.unsqueeze(0)
    cost = torch.cdist(samples_a, samples_b, p=2)
    min_a = cost.min(dim=1)[0]
    min_b = cost.min(dim=0)[0]
    return 0.5 * (min_a.mean() + min_b.mean())


def linear_probe_accuracy(features: np.ndarray, labels: np.ndarray, epochs: int = 200, lr: float = 0.1) -> float:
    features_tensor = torch.from_numpy(features).float()
    labels_tensor = torch.from_numpy(labels).long()
    num_classes = int(labels_tensor.max()) + 1
    model = torch.nn.Linear(features_tensor.size(1), num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = model(features_tensor)
        loss = loss_fn(logits, labels_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = model(features_tensor).argmax(dim=1)
    return (preds == labels_tensor).float().mean().item()


def cluster_purity(features: np.ndarray, labels: np.ndarray, num_clusters: int) -> float:
    features_tensor = torch.from_numpy(features).float()
    kmeans = _kmeans(features_tensor, num_clusters, iterations=50)
    assignments = kmeans.assignments.numpy()
    purity = 0.0
    for c in range(num_clusters):
        mask = assignments == c
        if not mask.any():
            continue
        label_counts = np.bincount(labels[mask])
        purity += label_counts.max()
    return purity / max(len(labels), 1)


def representation_invariance_score(feature_groups: Dict[str, np.ndarray]) -> float:
    centroids = {}
    for key, feats in feature_groups.items():
        centroids[key] = np.mean(feats, axis=0)
    keys = list(centroids.keys())
    if len(keys) < 2:
        return 0.0
    distances = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff = centroids[keys[i]] - centroids[keys[j]]
            distances.append(np.linalg.norm(diff))
    return float(np.mean(distances))


class _KMeansResult:
    def __init__(self, centers: torch.Tensor, assignments: torch.Tensor) -> None:
        self.centers = centers
        self.assignments = assignments


def _kmeans(features: torch.Tensor, k: int, iterations: int = 50) -> _KMeansResult:
    rng = torch.Generator().manual_seed(7)
    indices = torch.randperm(features.size(0), generator=rng)[:k]
    centers = features[indices].clone()
    for _ in range(iterations):
        dists = torch.cdist(features, centers)
        assignments = dists.argmin(dim=1)
        for cluster in range(k):
            mask = assignments == cluster
            if mask.any():
                centers[cluster] = features[mask].mean(dim=0)
    return _KMeansResult(centers=centers, assignments=assignments)
