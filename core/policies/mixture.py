from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = float(max(temperature, 1e-6))
    z = logits.astype(np.float32) / t
    z = z - np.max(z)
    e = np.exp(z)
    return (e / np.maximum(e.sum(), 1e-8)).astype(np.float32)


def choose_component(
    *,
    logits: np.ndarray,                 # (K,)
    rng: np.random.Generator,
    policy_purity: str,                 # "pure" | "mixed"
    deterministic: bool,
    temperature: float = 1.0,
) -> int:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    K = int(logits.shape[0])
    if K <= 1:
        return 0

    pure = str(policy_purity).strip().lower() == "pure"
    det = bool(deterministic) or pure

    if det:
        return int(np.argmax(logits))

    p = softmax(logits, temperature=temperature)
    return int(rng.choice(K, p=p))
