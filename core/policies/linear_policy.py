from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np

from policyOrProxy.core.policies.mixture import choose_component

_FeatureMode = Literal["last", "mean", "flatten"]


def _flatten_window(window: np.ndarray, mode: _FeatureMode) -> np.ndarray:
    if window.ndim != 4:
        raise ValueError(f"Expected window shape (T,teams,agents,state_dim), got {window.shape}")
    if mode == "last":
        x = window[-1].reshape(-1)
    elif mode == "mean":
        x = window.mean(axis=0).reshape(-1)
    elif mode == "flatten":
        x = window.reshape(-1)
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")
    return x.astype(np.float32)


@dataclass
class LinearPolicy:
    identifier: str
    teams: int
    agents: int
    state_dim: int
    window_len: int
    max_speed: float
    noise_std: float

    policy_purity: str
    mixture_temperature: float
    num_components: int
    feature_mode: _FeatureMode

    W: np.ndarray       # (K, out_dim, F)
    b: np.ndarray       # (K, out_dim)
    gate_W: np.ndarray  # (K, F)
    gate_b: np.ndarray  # (K,)
    rng: np.random.Generator

    # ---- opponent feature scaling (Option A) ----
    # Hard-coded defaults: treat team 1 as opponent, scale it up.
    opp_team_idx: int = 1
    opp_feature_scale: float = 8.0
    # --------------------------------------------

    # Debug controls (debug by default)
    debug: bool = True
    debug_every: int = 1  # print every act() call by default

    # Internal debug state
    _dbg_step: int = 0
    _y_prev: Optional[np.ndarray] = None
    _a_prev: Optional[np.ndarray] = None

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _dbg_print(self, msg: str) -> None:
        print(msg)

    @staticmethod
    def _cos_sim_and_angle_deg(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return float("nan"), float("nan")
        cos = float((a @ b) / (na * nb + 1e-12))
        cos = float(np.clip(cos, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos)))
        return cos, ang

    def act(self, window: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # ---- scale opponent team slice before flattening ----
        w = window
        if self.opp_feature_scale != 1.0:
            if window.ndim != 4:
                raise ValueError(f"Expected window shape (T,teams,agents,state_dim), got {window.shape}")
            if not (0 <= self.opp_team_idx < window.shape[1]):
                raise ValueError(
                    f"opp_team_idx={self.opp_team_idx} out of range for teams={window.shape[1]}"
                )
            w = np.array(window, copy=True, dtype=np.float32)
            w[:, self.opp_team_idx, :, :] *= float(self.opp_feature_scale)
        # -------------------------------------------------------------

        x = _flatten_window(w, self.feature_mode)  # (F,)

        logits = (self.gate_W @ x) + self.gate_b   # (K,)
        k = choose_component(
            logits=logits,
            rng=self.rng,
            policy_purity=self.policy_purity,
            deterministic=deterministic,
            temperature=self.mixture_temperature,
        )

        y = (self.W[k] @ x) + self.b[k]            # (out_dim,)

        # ---------------- DEBUG (enabled by default) ----------------
        if self.debug and (self._dbg_step % int(self.debug_every) == 0):
            y_abs = np.abs(y)
            frac_sat2 = float(np.mean(y_abs > 2.0))
            frac_sat3 = float(np.mean(y_abs > 3.0))

            dy_norm = float("nan")
            if self._y_prev is not None and self._y_prev.shape == y.shape:
                dy_norm = float(np.linalg.norm(y - self._y_prev))

            a_unit = np.tanh(y).astype(np.float32)
            frac_act_sat = float(np.mean(np.abs(a_unit) > 0.95))

            a_vec = a_unit.reshape(-1).astype(np.float32)
            cos_sim = float("nan")
            dangle_deg = float("nan")
            if self._a_prev is not None and self._a_prev.shape == a_vec.shape:
                cos_sim, dangle_deg = self._cos_sim_and_angle_deg(a_vec, self._a_prev)

            # Decompose contributions by team (meaningful for flatten mode)
            team0_norm = team1_norm = team_ratio = None
            if self.feature_mode == "flatten" and w.ndim == 4:
                T, teams, agents, sd = w.shape
                if teams == self.teams and agents == self.agents and sd == self.state_dim:
                    Wk = self.W[k].reshape(-1, T, teams, agents, sd).astype(np.float32)
                    X = w.astype(np.float32)

                    y_team0 = np.sum(Wk[:, :, 0, :, :] * X[:, 0, :, :], axis=(1, 2, 3))
                    team0_norm = float(np.linalg.norm(y_team0))

                    if teams > 1:
                        y_team1 = np.sum(Wk[:, :, 1, :, :] * X[:, 1, :, :], axis=(1, 2, 3))
                        team1_norm = float(np.linalg.norm(y_team1))
                        team_ratio = team1_norm / (team0_norm + 1e-9)

            self._dbg_print(
                f"[{self.identifier}] step={self._dbg_step} k={k} "
                f"|y|_2={np.linalg.norm(y):.3f} "
                f"y_max={y.max():.3f} y_min={y.min():.3f} "
                f"mean|y|={y_abs.mean():.3f} "
                f"frac(|y|>2)={frac_sat2:.2%} frac(|y|>3)={frac_sat3:.2%} "
                f"frac(|tanh(y)|>0.95)={frac_act_sat:.2%} "
                f"dy_norm={dy_norm:.4f} "
                f"cos_sim={cos_sim:.4f} dangle_deg={dangle_deg:.3f} "
                f"opp_team_idx={self.opp_team_idx} opp_scale={self.opp_feature_scale:.2f}"
                + (f" team0_norm={team0_norm:.3f}" if team0_norm is not None else "")
                + (f" team1_norm={team1_norm:.3f}" if team1_norm is not None else "")
                + (f" opp/ego={team_ratio:.3f}" if team_ratio is not None else "")
            )

            self._y_prev = y.copy()
            self._a_prev = a_vec.copy()

        self._dbg_step += 1
        # -----------------------------------------------------------

        action = np.tanh(y).reshape(self.agents, 2) * float(self.max_speed)

        det = bool(deterministic) or (str(self.policy_purity).lower() == "pure")
        if self.noise_std > 0.0 and not det:
            action = action + self.rng.normal(scale=self.noise_std, size=action.shape).astype(np.float32)

        return np.clip(action, -self.max_speed, self.max_speed).astype(np.float32)

    @staticmethod
    def from_cfg(
        *,
        policy_cfg: Dict,
        world_cfg: Dict,
        identifier: str,
        init_seed: int,
        max_speed: float,
        noise_std: float,
    ) -> "LinearPolicy":
        teams = int(world_cfg["teams"])
        agents = int(world_cfg["agents_per_team"])
        window_len = int(world_cfg.get("window_len", world_cfg.get("history")))
        state_dim = int(policy_cfg.get("state_dim", 4))

        feature_mode = str(policy_cfg.get("feature_mode", "last")).lower()
        if feature_mode not in ("last", "mean", "flatten"):
            raise ValueError("linear policy feature_mode must be one of: last, mean, flatten")

        if feature_mode == "flatten":
            F = window_len * teams * agents * state_dim
        else:
            F = teams * agents * state_dim

        K = int(policy_cfg.get("num_components", 1))
        rng = np.random.default_rng(int(init_seed))

        w_scale = float(policy_cfg.get("weight_scale", 0.5))
        b_scale = float(policy_cfg.get("bias_scale", 0.0))
        g_scale = float(policy_cfg.get("gate_weight_scale", 0.5))
        gb_scale = float(policy_cfg.get("gate_bias_scale", 0.0))

        out_dim = agents * 2
        W = rng.normal(scale=w_scale, size=(K, out_dim, F)).astype(np.float32)
        b = rng.normal(scale=b_scale, size=(K, out_dim)).astype(np.float32)

        gate_W = rng.normal(scale=g_scale, size=(K, F)).astype(np.float32)
        gate_b = rng.normal(scale=gb_scale, size=(K,)).astype(np.float32)

        return LinearPolicy(
            identifier=identifier,
            teams=teams,
            agents=agents,
            state_dim=state_dim,
            window_len=window_len,
            max_speed=float(max_speed),
            noise_std=float(noise_std),
            policy_purity=str(policy_cfg.get("policy_purity", "pure")),
            mixture_temperature=float(policy_cfg.get("mixture_temperature", 1.0)),
            num_components=K,
            feature_mode=feature_mode,  # type: ignore
            W=W,
            b=b,
            gate_W=gate_W,
            gate_b=gate_b,
            rng=rng,
            # opp_team_idx / opp_feature_scale are hard-coded defaults above
            # debug/debug_every are also hard-coded defaults above
        )
