"""Hidden regimes that switch within a life (single-agent study, stage A).

Each regime is one ecology. The agent is never told which regime it is in.
Durations and order are random; every life starts with a safe juvenile period.
"""
from __future__ import annotations

import numpy as np

from bio_rl.fast.popsim import ecology_arrays

DECAY = 0.005
LOSS = 0.4
REGIMES = {
    "safe":           dict(p_zone=0.0, p_random=0.0, regrow=25, switch_every=0),
    "danger":         dict(p_zone=0.3, p_random=0.0, regrow=25, switch_every=0),
    "uncontrollable": dict(p_zone=0.0, p_random=0.006, regrow=25, switch_every=0),
    "scarce":         dict(p_zone=0.0, p_random=0.0, regrow=60, switch_every=0),
    "volatile":       dict(p_zone=0.0, p_random=0.0, regrow=25, switch_every=300),
}
NAMES = list(REGIMES)
T_LIFE = 6000
JUVENILE = 500
DUR = (500, 1200)


def eco(name: str, B: int) -> dict:
    return ecology_arrays(B, damage_loss=LOSS, energy_decay=DECAY, companion=False, **REGIMES[name])


def schedule(rng: np.random.Generator, T: int = T_LIFE, dur: tuple = DUR,
             juvenile: int = JUVENILE) -> list[tuple[int, str]]:
    """[(start_step, regime), ...]: safe juvenile period, then random regimes, no immediate repeats."""
    out, t, prev = [(0, "safe")], juvenile, "safe"
    while t < T:
        r = rng.choice([n for n in NAMES if n != prev])
        out.append((t, str(r)))
        prev = r
        t += int(rng.integers(*dur))
    return out


def phases(sched, B: int, params_by_regime: dict | None = None):
    """Simulator phases for a schedule (the first entry is applied as the start ecology)."""
    ph = []
    for start, name in sched[1:]:
        P = params_by_regime[name] if params_by_regime else None
        ph.append((start, None, eco(name, B), P))
    return ph


def regime_trace(sched, T: int = T_LIFE) -> np.ndarray:
    """Regime index per step."""
    idx = np.zeros(T, int)
    for (s, name), nxt in zip(sched, sched[1:] + [(T, None)]):
        idx[s:nxt[0]] = NAMES.index(name)
    return idx


# ---------------------------------------------------------------------------
# Second environment E2 (12 x 12 grid, 7 x 7 view, 3 food cells, 2 threat clusters; stage E2-0 / C4).
# Hit probabilities and injury as above; food regrowth and layout changes scaled with distances (x 1.5).
# REGROW_SCARCE_E2 and DECAY_E2 are the uncalibrated defaults; the registered calibration rule may change them.
T_LIFE_E2 = 9000
JUVENILE_E2 = 750
DUR_E2 = (800, 2000)
REGROW_E2, REGROW_SCARCE_E2, SWITCH_E2 = 38, 90, 450
DECAY_E2 = DECAY


def reg_tab_e2(regrow_scarce: int = REGROW_SCARCE_E2, regrow: int = REGROW_E2, switch_every: int = SWITCH_E2):
    """Regime table (p_zone, p_random, regrow, switch_every) for E2, rows in NAMES order."""
    tab = {"safe": (0.0, 0.0, regrow, 0), "danger": (REGIMES["danger"]["p_zone"], 0.0, regrow, 0),
           "uncontrollable": (0.0, REGIMES["uncontrollable"]["p_random"], regrow, 0),
           "scarce": (0.0, 0.0, regrow_scarce, 0), "volatile": (0.0, 0.0, regrow, switch_every)}
    return np.array([tab[n] for n in NAMES], float)


def schedule_e2(rng: np.random.Generator) -> list[tuple[int, str]]:
    return schedule(rng, T=T_LIFE_E2, dur=DUR_E2, juvenile=JUVENILE_E2)
