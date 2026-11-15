"""Need layer: internal state, set-points and homeostatic utility.

Utility is the distance of the internal state from its set-points,
D(n) = sum_k w_k (n*_k - n_k)^2, and reward is its reduction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEEDS = ("energy", "safety")


@dataclass
class NeedConfig:
    setpoint: float = 1.0
    energy_decay: float = 0.008      # per step
    move_cost: float = 0.002         # extra energy cost of moving
    food_gain: float = 0.35
    safety_recovery: float = 0.004   # passive recovery per step
    shelter_recovery: float = 0.03   # extra recovery inside the shelter
    damage_loss: float = 0.25
    social: bool = False             # add a third need, belonging
    belonging_decay: float = 0.004
    contact_gain: float = 0.05       # per step spent with the companion


class Needs:
    """Need vector in [0, 1]: (energy, safety) or (energy, safety, belonging)."""

    def __init__(self, cfg: NeedConfig | None = None) -> None:
        self.cfg = cfg or NeedConfig()
        self.n = np.array([0.8, 0.8, 0.8] if self.cfg.social else [0.8, 0.8])

    def update(self, events: dict) -> np.ndarray:
        """Apply one step of physiology; return the felt change in each need.

        The returned change is computed before clipping at 0, so a depleted
        need keeps signalling its loss (hunger keeps hurting at zero energy);
        gains above the set-point are not felt.
        """
        c = self.cfg
        old = self.n.copy()
        e, s = self.n[0], self.n[1]
        e -= c.energy_decay + (c.move_cost if events["moved"] else 0.0)
        if events["ate"]:
            e += c.food_gain
        s += c.safety_recovery + (c.shelter_recovery if events["in_shelter"] else 0.0)
        if events["damage"]:
            s -= c.damage_loss
        raw = [e, s]
        if c.social:
            b = self.n[2] - c.belonging_decay + (c.contact_gain if events.get("social") else 0.0)
            raw.append(b)
        raw = np.array(raw)
        felt = np.minimum(raw, 1.0) - old
        self.n = np.clip(raw, 0.0, 1.0)
        return felt

    def deficits(self) -> np.ndarray:
        return self.cfg.setpoint - self.n

    def urgency(self) -> np.ndarray:
        """Gradient of the quadratic drive: how much a unit gain in each need is worth now."""
        return 2.0 * np.maximum(self.deficits(), 0.0) + 0.05

    def drive(self) -> float:
        """Unweighted homeostatic deviation, used as the evaluation metric."""
        return float(np.sum(self.deficits() ** 2))
