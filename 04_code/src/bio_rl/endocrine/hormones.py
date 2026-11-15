"""Endocrine layer: noradrenaline (fast) and cortisol (slow, delayed).

All variables follow a leaky integrator  H <- H + (H0 - H)/tau + input.

* Noradrenaline (NE): driven by *unexpected* surprise, i.e. TD-error magnitude
  above its own running average. Raises learning rate and exploration.
* Cortisol (C): driven by appraised threat, gated by low controllability and
  raised by uncertainty, delayed by ``delay`` steps, with quadratic negative
  feedback standing in for the HPA axis:
      input_C = kappa * A(t-d) * (1 - K) * (1 + U)
      C <- C + (C0 - C)/tau_C + input_C - f * C^2
  Effects: raises the weight of the safety need; inverted-U effect on learning
  rate; narrows exploration.
Each effect is a separate *channel* with its own gain, so experiments can switch
channels on and off and keep only those that earn their place:
  NE -> learning rate (ne_to_alpha), NE -> exploration (ne_to_temp),
  C -> safety weight (c_to_safety), C -> learning rate as inverted U
  (c_alpha_up / c_alpha_down) or as monotonic decrease (c_alpha_lower),
  C -> exploration (c_to_temp), C -> safety set-point (c_setpoint),
  NE -> network reset of values (ne_reset), C -> safety-value learning rate (c_safety_lr).

* Dopamine is not a separate state here: it is the per-need TD error computed
  by the learner (a vector of prediction errors, as in reward bases).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    """Scalar clip; much faster than np.clip on Python floats."""
    return lo if x < lo else hi if x > hi else x


@dataclass
class EndocrineConfig:
    use_ne: bool = True
    use_cortisol: bool = True
    # noradrenaline
    tau_ne: float = 8.0
    k_ne: float = 1.5
    tau_surprise: float = 200.0
    ne_to_alpha: float = 1.5        # alpha multiplier = 1 + ne_to_alpha * NE
    ne_to_temp: float = 1.0         # temperature multiplier = 1 + ne_to_temp * NE
    # cortisol
    tau_c: float = 150.0
    kappa_c: float = 0.05
    delay: int = 10
    feedback: float = 0.02
    c_to_safety: float = 2.0        # safety weight multiplier = 1 + c_to_safety * C
    c_alpha_up: float = 1.0         # inverted U: 1 + up*C - down*C^2
    c_alpha_down: float = 1.2
    c_to_temp: float = 1.0          # temperature divided by 1 + c_to_temp * C
    c_alpha_lower: float = 0.0      # monotonic decrease: alpha / (1 + c_alpha_lower * C)  (Park et al., 2017)
    ne_reset: float = 0.0           # network reset (Bouret & Sara 2005): values shrink by ne_reset*(NE-ne_reset_threshold)
    ne_reset_threshold: float = 1.0
    c_safety_lr: float = 0.0        # emotional-memory enhancement: safety-value learning rate x (1 + c_safety_lr*C)
    use_ot: bool = False            # oxytocin (social layer)
    tau_ot: float = 100.0
    k_ot: float = 0.05              # OT rise per step of social contact
    c_to_belonging: float = 0.0     # tend-and-befriend (Taylor et al. 2000): belonging weight x (1 + c_to_belonging*C)
    ot_buffer: float = 0.0          # social buffering: cortisol input x max(0, 1 - ot_buffer * OT)
    # v0.5: controllability prior (appraisal) -> action vigour (Huys & Dayan 2009; Niv et al. 2007)
    k_estimator: str = "explained"  # "explained" (v0.4) or "contingency" (v0.5)
    tau_kp: float = 1500.0          # time scale of the controllability prior Kp
    kp_effort: float = 0.0          # gain on the energy cost of moving under the no-control hypothesis (1 = physiological)
    kp_explore: float = 0.0         # exploration divided by 1 + kp_explore * (1 - Kp)
    c_setpoint: float = 0.0         # allostatic shift: safety set-point raised by c_setpoint * C (Khan & Lowe, 2024)


class Endocrine:
    def __init__(self, cfg: EndocrineConfig | None = None) -> None:
        self.cfg = cfg or EndocrineConfig()
        self.NE = 0.0
        self.C = 0.0
        self.OT = 0.0
        self.mean_abs_delta = 0.05
        self.buffer: deque[float] = deque([0.0] * max(self.cfg.delay, 1), maxlen=max(self.cfg.delay, 1))

    def step(self, abs_delta: float, threat: float, K: float, U: float, social: bool = False) -> None:
        c = self.cfg
        if c.use_ot:
            self.OT += -self.OT / c.tau_ot + (c.k_ot if social else 0.0)
            self.OT = _clip(float(self.OT), 0.0, 3.0)
        # NE: surprise relative to what is usual
        excess = max(abs_delta - self.mean_abs_delta, 0.0)
        self.mean_abs_delta += (abs_delta - self.mean_abs_delta) / c.tau_surprise
        if c.use_ne:
            self.NE += -self.NE / c.tau_ne + c.k_ne * excess
            self.NE = _clip(float(self.NE), 0.0, 3.0)
        # cortisol: delayed appraisal, gated by controllability
        self.buffer.append(threat)
        if c.use_cortisol:
            delayed = self.buffer[0]
            inp = c.kappa_c * delayed * (1.0 - K) * (1.0 + U)
            inp *= max(0.0, 1.0 - c.ot_buffer * self.OT)
            self.C += -self.C / c.tau_c + inp - c.feedback * self.C ** 2
            self.C = _clip(float(self.C), 0.0, 3.0)

    # ---- read-out functions ------------------------------------------
    def alpha_multiplier(self) -> float:
        c = self.cfg
        ne = 1.0 + c.ne_to_alpha * self.NE
        cort = max(0.2, 1.0 + c.c_alpha_up * self.C - c.c_alpha_down * self.C ** 2)
        cort /= 1.0 + c.c_alpha_lower * self.C
        return ne * cort

    def temperature_multiplier(self) -> float:
        c = self.cfg
        return (1.0 + c.ne_to_temp * self.NE) / (1.0 + c.c_to_temp * self.C)

    def reset_fraction(self) -> float:
        """Network-reset channel: fraction by which learned values shrink this step."""
        return float(min(0.5, self.cfg.ne_reset * max(self.NE - self.cfg.ne_reset_threshold, 0.0)))

    # read-outs are vectors over needs; index 1 is always safety
    def _vec(self, n: int, base: float, safety: float) -> np.ndarray:
        v = np.full(n, base)
        v[1] = safety
        return v

    def per_need_alpha(self, n: int = 2) -> np.ndarray:
        """Per-need learning-rate multipliers."""
        return self._vec(n, 1.0, 1.0 + self.cfg.c_safety_lr * self.C)

    def setpoint_shift(self, n: int = 2) -> np.ndarray:
        """Allostatic channel: anticipatory raise of the safety set-point."""
        return self._vec(n, 0.0, self.cfg.c_setpoint * self.C)

    def need_weight_multipliers(self, n: int = 2) -> np.ndarray:
        """Multipliers for the need weights."""
        v = self._vec(n, 1.0, 1.0 + self.cfg.c_to_safety * self.C)
        if n > 2:
            v[2] = 1.0 + self.cfg.c_to_belonging * self.C   # index 2 = belonging
        return v
