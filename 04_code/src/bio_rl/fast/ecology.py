"""Ecology distributions for the evolution study.

Five factors that an animal cannot see directly but that decide which
behaviour keeps it alive:
* danger type: none / controllable (threat zone that can be avoided) /
  uncontrollable (random damage anywhere; only the shelter helps recovery)
* lethality: safety lost per hit
* volatility: how often the world (food, shelter, threat, companion) moves
* scarcity: food regrowth time
* social: a companion is present or not
"""
from __future__ import annotations

import itertools

import numpy as np

from bio_rl.fast.popsim import ecology_arrays

DANGER = {"none": (0.0, 0.0), "controllable": (0.35, 0.0), "uncontrollable": (0.0, 0.015)}
GRID = dict(danger=list(DANGER), lethality=[0.15, 0.4], switch_every=[0, 1500], regrow=[25, 50],
            companion=[False, True])
ENERGY_DECAY = 0.005   # metabolism; calibrated on a no-hormone agent (experiments/evo/dev_calibrate_ecologies.py)
GRACE = 100            # steps at zero energy or safety before death


# v2 (evolution_v2): injuries are lethal (death as soon as safety reaches 0), calibrated on
# agents without hormones (results/logs/evo_v2_dev_calibration.out)
DANGER_V2 = {"none": (0.0, 0.0), "controllable": (0.2, 0.0), "uncontrollable": (0.0, 0.008)}
GRID_V2 = dict(GRID, lethality=[0.25, 0.5])
INJURY_GRACE = {1: GRACE, 2: 1}
RANGES = {1: dict(p_zone=(0.2, 0.5), p_random=(0.005, 0.025), loss=(0.1, 0.45)),
          2: dict(p_zone=(0.1, 0.3), p_random=(0.004, 0.012), loss=(0.2, 0.55))}


def sample(B: int, rng: np.random.Generator, version: int = 1) -> tuple[dict, list]:
    """Training distribution: continuous ranges around the test grid."""
    rg = RANGES[version]
    kind = rng.choice(list(DANGER), size=B)
    p_zone = np.where(kind == "controllable", rng.uniform(*rg["p_zone"], B), 0.0)
    p_random = np.where(kind == "uncontrollable", rng.uniform(*rg["p_random"], B), 0.0)
    switch = np.where(rng.random(B) < 0.5, 0, rng.integers(1000, 2000, B))
    eco = ecology_arrays(B, p_zone=p_zone, p_random=p_random, damage_loss=rng.uniform(*rg["loss"], B),
                         switch_every=switch, regrow=rng.integers(20, 56, B), companion=rng.random(B) < 0.5,
                         energy_decay=ENERGY_DECAY)
    return eco, list(kind)


def grid(reps: int, version: int = 1):
    """Test grid: every combination of the factor levels, ``reps`` agents each.
    Returns (ecology arrays, list of per-agent factor dicts)."""
    g, danger = (GRID, DANGER) if version == 1 else (GRID_V2, DANGER_V2)
    combos = list(itertools.product(*g.values()))
    rows = [dict(zip(g, c)) for c in combos for _ in range(reps)]
    B = len(rows)
    eco = ecology_arrays(B,
                         p_zone=[danger[r["danger"]][0] for r in rows],
                         p_random=[danger[r["danger"]][1] for r in rows],
                         damage_loss=[r["lethality"] for r in rows],
                         switch_every=[r["switch_every"] for r in rows],
                         regrow=[r["regrow"] for r in rows],
                         companion=[r["companion"] for r in rows], energy_decay=ENERGY_DECAY)
    for i, r in enumerate(rows):
        r["eco_id"] = i // reps
    return eco, rows
