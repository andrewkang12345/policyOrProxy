# File: policyOrProxy/eval/eval.py
#!/usr/bin/env python3
"""
Universal evaluation for trained (Global/Hier/MAPD) CVAE-like models + Grover-style representation model.

What it does
------------
- Test data is configured independently from training (eval.yaml).
- Supports per-policy, per-distribution mixes via YAML "data.experiment.test".
- Embedding clustering metric: IICR = (mean intra-cluster distance) / (mean inter-centroid distance).
- Optional linear-probe policy classification accuracy on embeddings when all test policies
  were *seen during training* (based on the training YAML).
- Logs train/test distribution configuration per policy into the results JSON.

Supports model_type:
- "global" -> GlobalCVAE (latent_key="mu")
- "hier"   -> HierarchicalCVAE (latent_key="mu_global")
- "grover" -> GroverModel (latent_key="mu")  <-- added

Example eval.yaml (test-only) is at the end of this file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Union
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Repo path bootstrap
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policyOrProxy.core.dataset.indexer import EpisodeIndexer
from policyOrProxy.core.dataset.next_frame import NextFrameDataset
from policyOrProxy.models.collate import move_batch, next_frame_collate
from policyOrProxy.models.train_global_cvae import GlobalCVAE
from policyOrProxy.models.train_hier_cvae import HierarchicalCVAE
from policyOrProxy.core.metrics.metrics import linear_probe_accuracy

# NEW: import the Grover model (from your training script)
from policyOrProxy.models.train_grover_embed import GroverModel  # noqa: E402

LOGGER = logging.getLogger("eval")


# -----------------------
# Logging / YAML helpers
# -----------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


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
    """Accept dict/list/str and return a flat list of dict entries."""
    if isinstance(entries, str):
        return [{"policy": entries}]
    if isinstance(entries, dict):
        return [entries]
    if isinstance(entries, Iterable):
        out: List[Dict] = []
        for e in entries:
            out.extend(_normalize_entries(e))
        return out
    raise ValueError(f"Cannot parse entries of type {type(entries)}")


def _normalize_window_shape(w: torch.Tensor) -> torch.Tensor:
    """
    Ensure window shape is [B, T, teams, agents, state] (or [B, T, F] / [B, F]).
    Some dataloaders yield [B, 1, T, teams, agents, state]; squeeze that middle dim.
    """
    if w.dim() >= 3 and w.size(1) == 1:
        w = w.squeeze(1)
    return w


def expand_experiment_entries(
    data_root: Path, split_entries: List[Dict]
) -> List[Tuple[str, Path, str]]:
    """
    From a split spec, build a list of (policy_name, dataset_root, distribution_label).
    Each entry can be:
      - {policy: "ego_policy1", category: "iid"}
      - {policy: "ego_policy*", category: "ood_manual+right_bias_mild"}  # glob
      - {path: "output/data/ood_manual/ego_policy2/right_bias_moderate"} # explicit
      - {policy: "ego_policy3", exclude: true}                            # skip
    """
    resolved: List[Tuple[str, Path, str]] = []
    for ent in _normalize_entries(split_entries):
        if ent.get("exclude", False):
            continue

        if "path" in ent:
            rp = Path(ent["path"]).expanduser()
            if not (rp / "index.json").exists():
                LOGGER.warning("Missing index.json for explicit path: %s", rp)
                continue
            pol = rp.parent.name if rp.parent.name.startswith("ego_policy") else rp.name
            # distribution label from path tail
            dist = "iid" if rp.name == "iid" else (f"{rp.parents[2].name}+{rp.name}" if len(rp.parents) > 2 else rp.name)
            resolved.append((pol, rp, dist))
            continue

        if "policy" in ent and "category" in ent:
            pattern = ent["policy"]
            category = ent["category"]
            for pol_dir in sorted(data_root.glob(pattern)):
                if not pol_dir.is_dir():
                    continue
                pol = pol_dir.name
                rp = category_to_root(data_root, pol, category)
                if (rp / "index.json").exists():
                    resolved.append((pol, rp, category))
                else:
                    LOGGER.warning("No index.json for (%s, %s) at %s", pol, category, rp)
            continue

        if "policy" in ent and "category" not in ent:
            LOGGER.warning("Ignoring entry with 'policy' but no 'category': %s", ent)
            continue

        LOGGER.warning("Unrecognized split entry, ignored: %s", ent)

    uniq: List[Tuple[str, Path, str]] = []
    seen = set()
    for pol, rp, dist in resolved:
        key = (pol, rp.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((pol, rp, dist))
    return uniq


# -----------------------
# Model builders
# -----------------------

def build_feature_model(model_type: str, model_cfg: Dict, device: torch.device):
    """
    Returns (model, latent_key, extras) where forward_fn should return a dict containing latent_key.
    Deterministic latent keys by default:
      - global: 'mu'
      - hier:   'mu_global'
      - grover: 'mu' (we map GroverModel.embed(...) -> {'mu': ...})
    """
    if model_type == "global":
        model = GlobalCVAE(
            window_len=int(model_cfg["window_len"]),
            teams=int(model_cfg["model"]["teams"]),
            agents=int(model_cfg["model"]["agents"]),
            state_dim=int(model_cfg["model"]["state_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            d_model=int(model_cfg["model"]["encoder_dim"]),
            layers=int(model_cfg["model"]["encoder_layers"]),
            heads=int(model_cfg["model"]["encoder_heads"]),
            dropout=float(model_cfg["model"]["dropout"]),
            action_dim=int(model_cfg["model"]["action_dim"]),
        ).to(device)
        latent_key = "mu"
        return model, latent_key, {}

    if model_type == "hier":
        model = HierarchicalCVAE(
            window_len=int(model_cfg["window_len"]),
            teams=int(model_cfg["model"]["teams"]),
            agents=int(model_cfg["model"]["agents"]),
            state_dim=int(model_cfg["model"]["state_dim"]),
            latent_dim_global=int(model_cfg["latent_dim"]["global"]),
            latent_dim_local=int(model_cfg["latent_dim"]["local"]),
            d_model=int(model_cfg["model"]["encoder_dim"]),
            layers=int(model_cfg["model"]["encoder_layers"]),
            heads=int(model_cfg["model"]["encoder_heads"]),
            dropout=float(model_cfg["model"]["dropout"]),
            action_dim=int(model_cfg["model"]["action_dim"]),
        ).to(device)
        latent_key = "mu_global"
        return model, latent_key, {}

    if model_type == "grover":
        # Expect same keys you used in train_grover_embed.yaml
        mcfg = model_cfg["model"]
        loss_cfg = model_cfg.get("loss", {})
        model = GroverModel(
            teams=int(mcfg["teams"]),
            agents=int(mcfg["agents"]),
            state_dim=int(mcfg["state_dim"]),
            action_dim=int(mcfg["action_dim"]),
            embed_dim=int(mcfg["embed_dim"]),
            encoder_hidden=int(mcfg["encoder_hidden"]),
            policy_hidden=int(mcfg["policy_hidden"]),
            dropout=float(mcfg["dropout"]),
            gaussian_min_logvar=float(loss_cfg.get("gaussian_min_logvar", -6.0)),
        ).to(device)
        # We’ll expose embeddings under 'mu' for consistency with CVAEs
        latent_key = "mu"
        return model, latent_key, {}

    raise ValueError(f"Unsupported model type {model_type}")


# -----------------------
# Embedding collection
# -----------------------

@torch.no_grad()
def collect_embeddings_for_pair(
    loader: DataLoader,
    device: torch.device,
    forward_fn,
    latent_key: str,
    policy_label: str,
) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = forward_fn(batch)
        if latent_key not in out:
            raise RuntimeError(f"Latent key '{latent_key}' missing from model outputs")
        lat = out[latent_key]
        if lat.dim() == 3:
            lat = lat.mean(dim=1)
        feats.append(lat.detach().cpu().numpy())
    return [row for arr in feats for row in arr]


def build_test_loaders(
    data_root: Path,
    test_spec: List[Dict],
    batch_size: int,
    num_workers: int,
) -> Tuple[List[Tuple[str, str, DataLoader]], Dict[str, List[str]]]:
    """
    Returns:
      - plan: list of (policy, dist_label, dataloader) tuples
      - mapping: {policy: [dist_label, ...]} for logging
    """
    pairs = expand_experiment_entries(data_root, test_spec)
    plan: List[Tuple[str, str, DataLoader]] = []
    mapping: Dict[str, List[str]] = {}
    for pol, root, dist in pairs:
        indexer = EpisodeIndexer.load(root)
        ds = NextFrameDataset(root, indexer, split="test", include_policy_id=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=next_frame_collate)
        plan.append((pol, dist, loader))
        mapping.setdefault(pol, []).append(dist)
    return plan, mapping


# -----------------------
# Clustering metrics (IICR)
# -----------------------

def _pairwise_mean_distance(X: np.ndarray) -> float:
    n = X.shape[0]
    if n < 2:
        return 0.0
    norms = (X * X).sum(axis=1, keepdims=True)
    sq_dists = norms + norms.T - 2 * X @ X.T
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(np.maximum(sq_dists[iu], 0.0))
    return float(d.mean())


def compute_iicr(grouped: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
    clusters = {k: np.asarray(v, dtype=np.float32) for k, v in grouped.items() if len(v) > 0}
    policies = sorted(clusters.keys())
    if len(policies) < 2:
        return {"iicr": float("nan"), "intra": float("nan"), "inter": float("nan")}
    intra_vals, centroids, sizes = [], [], {}
    for pol in policies:
        X = clusters[pol]
        sizes[pol] = int(X.shape[0])
        intra_vals.append(_pairwise_mean_distance(X))
        centroids.append(X.mean(axis=0, keepdims=True))
    intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    C = np.concatenate(centroids, axis=0)
    inter = _pairwise_mean_distance(C)
    iicr = float(intra / inter) if inter > 0 else float("inf")
    out = {"iicr": iicr, "intra": intra, "inter": inter}
    for pol in policies:
        out[f"size[{pol}]"] = sizes[pol]
    return out


# -----------------------
# Checkpoint utilities
# -----------------------

def resolve_checkpoint_path(p: Union[str, Path]) -> Path:
    """Accept a file OR a run directory; return a concrete .pt file path."""
    p = Path(p)
    if p.is_file():
        return p
    # treat as directory; try common candidates
    candidates = [
        p / "checkpoints" / "best.pt",
        p / "best.pt",
        p / "checkpoints" / "last.pt",
        p / "last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve a checkpoint file under: {p}")


def safe_torch_load(path: Path, map_location: torch.device):
    """
    Robust torch.load that handles PyTorch 2.6+ (weights_only=True default) and
    allowlists numpy's reconstruct helper when needed.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            import numpy as np  # noqa
            add_safe_globals([np.core.multiarray._reconstruct])  # allowlist
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        except Exception:
            raise e


# -----------------------
# Main evaluation routine
# -----------------------

def main(args) -> None:
    configure_logging()

    # --- Load configs
    model_cfg = load_yaml(Path(args.config))
    eval_cfg = load_yaml(Path(args.eval))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and eval_cfg.get("device", "auto") != "cpu" else "cpu")
    LOGGER.info("Device: %s", device)

    # --- Model + checkpoint
    model_type = eval_cfg.get("model_type", "global")
    model, latent_key, extras = build_feature_model(model_type, model_cfg, device)

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    # Unwrap common formats
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt.get("model_state") or ckpt
    else:
        state = ckpt

    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    missing, unexpected = target.load_state_dict(state, strict=False)
    if missing:
        LOGGER.warning("Missing keys when loading: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys when loading: %s", unexpected)
    target.eval()

    # --- Forward function
    if model_type == "grover":
        @torch.no_grad()
        def forward_fn(batch):
            w = _normalize_window_shape(batch["window"])
            # Return dict to match downstream API
            return {"mu": model.embed(w)}
    elif model_type == "mapd":
        # (Left intact if you had MAPD utilities wired elsewhere.)
        raise NotImplementedError("model_type 'mapd' is not supported in this build.")
    else:
        @torch.no_grad()
        def forward_fn(batch):
            w = _normalize_window_shape(batch["window"])
            return model(w)

    # --- Determine seen (train) policies for logging/probe gating
    data_root = Path(model_cfg.get("paths", {}).get("data_root", "output/data")).expanduser()
    train_map: Dict[str, List[str]] = {}
    seen_policies: set[str] = set()
    exp_cfg = model_cfg.get("data", {}).get("experiment")
    if exp_cfg:
        train_pairs = expand_experiment_entries(data_root, exp_cfg.get("train", []))
        val_pairs = expand_experiment_entries(data_root, exp_cfg.get("val", exp_cfg.get("train", [])))
        for pol, _root, dist in train_pairs + val_pairs:
            train_map.setdefault(pol, [])
            if dist not in train_map[pol]:
                train_map[pol].append(dist)
            seen_policies.add(pol)
    else:
        cat = model_cfg.get("data", {}).get("category", "iid")
        base, sub = parse_category(cat)
        if base == "iid":
            for p in sorted((data_root).glob("ego_policy*/iid")):
                pol = p.parent.name
                train_map.setdefault(pol, []).append("iid")
                seen_policies.add(pol)
        else:
            base_dir = data_root / base
            for p in sorted(base_dir.glob(f"ego_policy*/{sub or ''}")):
                pol = p.parent.name
                train_map.setdefault(pol, []).append(cat)
                seen_policies.add(pol)

    # --- Build test plan from eval YAML
    test_spec = eval_cfg.get("data", {}).get("experiment", {}).get("test", [])
    if not test_spec:
        raise ValueError("eval.yaml must provide: data.experiment.test: [...]")
    batch_size = int(eval_cfg.get("batch_size", 128))
    num_workers = int(eval_cfg.get("num_workers", 0))
    plan, test_map = build_test_loaders(data_root, test_spec, batch_size, num_workers)

    # --- Collect embeddings grouped by policy
    grouped: Dict[str, List[np.ndarray]] = {}
    for pol, dist, loader in plan:
        feats = collect_embeddings_for_pair(loader, device, forward_fn, latent_key, policy_label=pol)
        grouped.setdefault(pol, []).extend(feats)
        LOGGER.info("Collected %d embeddings for policy=%s, dist=%s", len(feats), pol, dist)

    # --- IICR
    iicr_stats = compute_iicr(grouped)

    # --- Optional policy linear probe (only if no unseen policies)
    test_policies = set(test_map.keys())
    do_probe = test_policies.issubset(seen_policies) and len(test_policies) >= 2
    probe_acc = None
    if do_probe:
        X: List[np.ndarray] = []
        y: List[int] = []
        label_to_int = {pol: i for i, pol in enumerate(sorted(test_policies))}
        for pol, vecs in grouped.items():
            if pol not in label_to_int:
                continue
            X.extend(vecs)
            y.extend([label_to_int[pol]] * len(vecs))
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        probe_acc = float(linear_probe_accuracy(X_arr, y_arr))
        LOGGER.info("Linear probe policy accuracy (test policies ⊆ train policies): %.4f", probe_acc)
    else:
        LOGGER.info("Skipping policy probe (test introduces unseen policies or too few classes).")

    # --- Results JSON
    results = {
        "checkpoint": str(ckpt_path),
        "model_type": model_type,
        "latent_key": latent_key,
        "metrics": {
            "IICR": iicr_stats.get("iicr"),
            "intra_mean": iicr_stats.get("intra"),
            "inter_centroid_mean": iicr_stats.get("inter"),
            "policy_probe_accuracy": probe_acc,
        },
        "cluster_sizes": {k: iicr_stats[k] for k in iicr_stats if k.startswith("size[")},
        "config_logged": {
            "train_distribution_per_policy": train_map,
            "test_distribution_per_policy": test_map,
        },
    }

    out = Path(eval_cfg.get("output", args.output or "output/eval/representation.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    LOGGER.info("Saved results to %s", out)


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained model embeddings & clustering")
    p.add_argument("--config", required=True, help="Training YAML (used to know which policies were seen)")
    p.add_argument("--checkpoint", required=True, help="Path to a .pt file OR a run dir containing checkpoints/")
    p.add_argument("--eval", required=True, help="Eval YAML defining test mixes")
    p.add_argument("--output", type=str, help="Optional override of output JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
