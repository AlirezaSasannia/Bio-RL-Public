"""Search spaces for the evolution study and the CMA-ES fitness function.

Every agent kind gets its own free parameters and the same optimisation budget,
so no kind wins because it was tuned more carefully.
"""
from __future__ import annotations

import numpy as np

from bio_rl.fast.ecology import GRACE, INJURY_GRACE, sample
from bio_rl.fast.popsim import param_arrays, simulate

# Upper bounds equal the hard caps inside the simulator (alpha <= 0.9, epsilon <= 0.5), so a hormone
# channel can never push a parameter beyond what the no-hormone agents may choose directly.
LEARN = {"alpha": (0.02, 0.9), "epsilon": (0.0, 0.5)}
SPACES = {
    "homeostatic": ("needs", dict(LEARN)),
    "fixed": ("fixed", {**LEARN, "w_damage": (0.0, 5.0), "living_cost": (0.0, 0.05)}),
    "pearcehall": ("pearcehall", {**LEARN, "ph_rate": (0.005, 0.3)}),
    "endocrine": ("needs", {**LEARN,
                            "c_to_safety": (0.0, 6.0),     # cortisol -> weight of safety
                            "c_to_belonging": (0.0, 6.0),  # cortisol -> weight of belonging (tend-and-befriend)
                            "ot_buffer": (0.0, 4.0),       # oxytocin buffers cortisol input
                            "kp_effort": (0.0, 4.0),       # controllability prior -> vigour
                            "kp_explore": (0.0, 4.0),      # controllability prior -> exploration
                            "ne_to_alpha": (0.0, 4.0),     # noradrenaline -> learning rate
                            "ne_to_temp": (0.0, 6.0),      # noradrenaline -> exploration
                            "c_to_temp": (0.0, 4.0),       # cortisol narrows exploration
                            "kappa_c": (0.005, 0.3)}),     # cortisol gain
}
CHANNELS = ["c_to_safety", "c_to_belonging", "ot_buffer", "kp_effort", "kp_explore",
            "ne_to_alpha", "ne_to_temp", "c_to_temp"]


def decode(kind: str, u: np.ndarray) -> dict:
    """Map a point of the unit cube to named parameters."""
    _, space = SPACES[kind]
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return {k: float(lo + ui * (hi - lo)) for ui, (k, (lo, hi)) in zip(u, space.items())}


def evaluate(kind: str, candidates: list[dict], n_agents: int, bank, gen_seed: int, T: int = 5000,
             version: int = 1) -> np.ndarray:
    """Mean survival of each candidate over the same sampled ecologies and layouts
    (common random numbers within a generation)."""
    sim_kind, _ = SPACES[kind]
    rng = np.random.default_rng(gen_seed)
    eco1, _ = sample(n_agents, rng, version)
    lay1 = rng.integers(len(bank), size=n_agents)
    m = len(candidates)
    eco = {k: np.tile(v, m) for k, v in eco1.items()}
    lay = np.tile(lay1, m)
    keys = set().union(*candidates)
    params = param_arrays(m * n_agents, **{k: np.repeat([c.get(k, 0.0) for c in candidates], n_agents) for k in keys})
    out = simulate(sim_kind, params, eco, bank, lay, T, seed=gen_seed + 1, grace=GRACE,
                   injury_grace=INJURY_GRACE[version])
    return out["survival"].reshape(m, n_agents).mean(1)
