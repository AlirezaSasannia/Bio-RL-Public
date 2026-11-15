"""Agents compared in the prototype.

* ``fixed``      : standard Q-learning on a designer reward (+1 food, -1 damage,
                   small per-step cost).
* ``homeostatic``: one Q-table per need (reward bases); need weights = current
                   urgency only. No endocrine layer.
* ``pearcehall`` : homeostatic agent whose learning rate tracks recent surprise
                   (Pearce-Hall associability). A non-biological adaptive
                   learner: the baseline the endocrine layer must beat.
* ``endocrine``  : homeostatic agent + endocrine layer. NE and cortisol change
                   the need weights (what the agent wants) and the learning rate
                   and exploration (how it learns).

Exploration is epsilon-greedy so that it does not depend on the scale of the
values, which differs between the designer reward and need-based rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bio_rl.appraisal.appraisal import Appraisal
from bio_rl.endocrine.hormones import Endocrine, EndocrineConfig
from bio_rl.envs.gridworld import N_ACTIONS
from bio_rl.needs.homeostasis import NeedConfig, Needs


@dataclass
class AgentConfig:
    kind: str = "endocrine"          # fixed | homeostatic | pearcehall | endocrine
    ph_rate: float = 0.05            # pearcehall: smoothing of associability
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.05
    living_cost: float = 0.01        # fixed-reward baseline only: standard per-step cost
    endocrine: EndocrineConfig = field(default_factory=EndocrineConfig)
    needs: NeedConfig = field(default_factory=NeedConfig)


class Agent:
    def __init__(self, n_states: int, cfg: AgentConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.needs = Needs(cfg.needs)
        ec = cfg.endocrine
        self.appraisal = Appraisal(n_states, kind=ec.k_estimator, tau_kp=ec.tau_kp) if cfg.kind == "endocrine" \
            else Appraisal(n_states)
        n_values = 1 if cfg.kind == "fixed" else len(self.needs.n)
        self.Q = np.zeros((n_values, n_states, N_ACTIONS))
        self.endo = Endocrine(cfg.endocrine) if cfg.kind == "endocrine" else None
        self.assoc, self.mean_abs = 1.0, 0.05   # pearcehall state

    # ------------------------------------------------------------------
    def weights(self) -> np.ndarray:
        if self.cfg.kind == "fixed":
            return np.ones(1)
        if self.endo is not None:
            k = len(self.needs.n)
            target = self.needs.cfg.setpoint + self.endo.setpoint_shift(k)
            w = 2.0 * np.maximum(target - self.needs.n, 0.0) + 0.05
            return w * self.endo.need_weight_multipliers(k)
        return self.needs.urgency()

    def alpha(self) -> float:
        a = self.cfg.alpha
        if self.endo is not None:
            a *= self.endo.alpha_multiplier()
        elif self.cfg.kind == "pearcehall":
            a *= float(np.clip(self.assoc, 0.25, 4.0))
        return float(min(a, 0.9))

    def epsilon(self) -> float:
        e = self.cfg.epsilon
        if self.endo is not None:
            e *= self.endo.temperature_multiplier()
            e /= 1.0 + self.cfg.endocrine.kp_explore * (1.0 - self.appraisal.Kp)
        return float(min(max(e, 0.0), 0.5))

    def act(self, s: int) -> int:
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(N_ACTIONS))
        q = self.weights() @ self.Q[:, s, :]
        if self.endo is not None and self.cfg.endocrine.kp_effort > 0:
            # Controllability prior (Huys & Dayan 2009): with probability 1 - Kp the
            # outcome does not depend on the action, so a move brings what staying
            # brings, minus the energy it costs. Action 0 = stay.
            kp = self.appraisal.Kp
            effort = self.cfg.endocrine.kp_effort * self.weights()[0] * self.needs.cfg.move_cost
            q = q.copy()
            q[1:] = kp * q[1:] + (1.0 - kp) * (q[0] - effort)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(self, s: int, a: int, s2: int, events: dict) -> dict:
        dn = self.needs.update(events)
        threat, K = self.appraisal.update(s2, events["damage"])
        if self.cfg.kind == "fixed":
            rewards = np.array([float(events["ate"]) - float(events["damage"]) - self.cfg.living_cost])
        else:
            rewards = dn                       # one reward per need dimension
        w = self.weights()
        # per-need TD errors: the vector "dopamine" signal
        a2 = int(np.argmax(w @ self.Q[:, s2, :]))
        deltas = rewards + self.cfg.gamma * self.Q[:, s2, a2] - self.Q[:, s, a]
        alphas = np.full(len(deltas), self.alpha())
        if self.endo is not None:
            alphas = np.minimum(alphas * self.endo.per_need_alpha(len(deltas)), 0.9)
            shrink = self.endo.reset_fraction()
            if shrink > 0:
                self.Q *= 1.0 - shrink
        self.Q[:, s, a] += alphas * deltas
        abs_delta = float(abs(np.dot(w, deltas)))
        if self.cfg.kind == "pearcehall":
            self.mean_abs += (abs_delta - self.mean_abs) / 200.0
            self.assoc += self.cfg.ph_rate * (abs_delta / (self.mean_abs + 1e-6) - self.assoc)
        if self.endo is not None:
            U = float(min(max(self.endo.NE, 0.0), 1.0))
            self.endo.step(abs_delta, threat, K, U, social=bool(events.get("social", False)))
        return {
            "K": K,
            "Kp": self.appraisal.Kp,
            "NE": self.endo.NE if self.endo else 0.0,
            "C": self.endo.C if self.endo else 0.0,
            "OT": self.endo.OT if self.endo else 0.0,
        }
