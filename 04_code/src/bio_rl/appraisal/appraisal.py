"""Appraisal modules: threat (amygdala-inspired) and controllability.

* Threat map T(s): Pavlovian estimate of the damage probability in each cell,
  learned with a delta rule. Appraised threat A(t) = T(s_t) + damage_t.
* Controllability K(t): how much of the damage the agent's own map can
  predict. If damage is tied to places the agent can avoid, the map explains
  it and K is high; if damage strikes at random, much of it stays surprising
  and K falls.

v0.5 adds a contingency estimator (``kind="contingency"``), closer to the
operational definition used in the helplessness literature (Maier & Seligman
2016: control = the outcome depends on what the animal does). Damage counts
as *escapable* if it happens in places the agent's map marks as dangerous
(somewhere it could have stayed away from) and as *inescapable* if it hits in
places the map marks as safe. With leaky counts and a Beta(1, 1) prior:
    K = 1 - (n_safe + 1) / (n_all + 2)
which is 0.5 with no evidence. Two time scales are kept: a fast K (tau_k)
that gates cortisol, and a slow controllability prior Kp (tau_kp) that
generalises to new situations (Huys & Dayan 2009).
"""
from __future__ import annotations

import numpy as np


class Appraisal:
    def __init__(self, n_states: int, lr_threat: float = 0.1, tau_k: float = 300.0,
                 kind: str = "explained", tau_kp: float = 1500.0, safe_threshold: float = 0.05) -> None:
        self.kind = kind
        self.tau_kp = tau_kp
        self.safe_threshold = safe_threshold
        self.n_all = self.n_safe = 0.0          # fast leaky counts
        self.p_all = self.p_safe = 0.0          # slow leaky counts (prior)
        self.Kp = 0.5
        self.T = np.zeros(n_states)
        self.lr = lr_threat
        self.tau_k = tau_k
        self.mean_damage = 0.02
        self.mean_unexplained = 0.02
        self.K = 0.5

    def update(self, s: int, damage: bool) -> tuple[float, float]:
        d = float(damage)
        expected = self.T[s]
        if self.kind == "contingency":
            return self._update_contingency(s, d, expected)
        self.T[s] += self.lr * (d - expected)
        # running rates of all damage and of the part the map did not predict
        a = 1.0 / self.tau_k
        self.mean_damage += a * (d - self.mean_damage)
        unexplained = max(d - expected, 0.0)
        self.mean_unexplained += a * (unexplained - self.mean_unexplained)
        self.K = min(max(1.0 - self.mean_unexplained / (self.mean_damage + 1e-3), 0.0), 1.0)
        threat = expected + d
        return threat, self.K

    def _update_contingency(self, s: int, d: float, expected: float) -> tuple[float, float]:
        safe_hit = d if expected < self.safe_threshold else 0.0
        self.T[s] += self.lr * (d - expected)
        f, g = 1.0 - 1.0 / self.tau_k, 1.0 - 1.0 / self.tau_kp
        self.n_all, self.n_safe = f * self.n_all + d, f * self.n_safe + safe_hit
        self.p_all, self.p_safe = g * self.p_all + d, g * self.p_safe + safe_hit
        self.K = 1.0 - (self.n_safe + 1.0) / (self.n_all + 2.0)
        self.Kp = 1.0 - (self.p_safe + 1.0) / (self.p_all + 2.0)
        return expected + d, self.K
