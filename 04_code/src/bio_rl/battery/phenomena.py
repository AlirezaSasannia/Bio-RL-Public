"""Phenomenon battery: each function simulates one known human/animal finding.

Every function takes (architecture name, condition, seed) and returns a dict of
per-seed outcome measures. Predictions and tests live in the pre-registered
spec (experiments/battery_v1.py), not here.
"""
from __future__ import annotations

import numpy as np

from bio_rl.battery.architectures import agent_config
from bio_rl.envs.gridworld import EnvConfig, make_layout, with_companion
from bio_rl.runner import run_phases


def _slope(y):
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# PH1 + PH2  Learned helplessness, triadic yoked design (Seligman & Maier 1967;
# controllability and cortisol: Dickerson & Kemeny 2004)
# Phase 1 (3000 steps): escapable = avoidable threat cells; yoked = the exact
# damage sequence of the escapable run with the same seed, independent of
# behaviour; none = no damage. Phase 2 (2000 steps): a new world with avoidable
# threat, identical for all three groups.
def helplessness(arch: str, seed: int, **overrides) -> list[dict]:
    w1, w2 = make_layout(11000 + seed), make_layout(12000 + seed)
    p1, p2 = 3000, 2000
    test = EnvConfig()
    rows = []
    esc = run_phases(agent_config(arch, **overrides), seed, [(p1, w1, EnvConfig()), (p2, w2, test)])
    schedule = esc["damage"][:p1].astype(bool)
    runs = {
        "escapable": esc,
        # the yoked animal is a different individual: its own behaviour seed, the escapable animal's damage
        "yoked": run_phases(agent_config(arch, **overrides), seed + 100000,
                            [(p1, w1, EnvConfig(damage_schedule=schedule)), (p2, w2, test)]),
        "none": run_phases(agent_config(arch, **overrides), seed, [(p1, w1, EnvConfig(no_damage=True)), (p2, w2, test)]),
    }
    for cond, r in runs.items():
        ph2 = slice(p1, p1 + 1000)
        rows.append(dict(phenomenon="PH1", arch=arch, condition=cond, seed=seed,
                         phase1_damage=r["damage"][:p1].sum(), phase1_C=r["C"][:p1].mean(),
                         phase1_K=r["K"][p1 - 500:p1].mean(),
                         phase2_food=r["ate"][ph2].sum(), phase2_damage=r["damage"][ph2].sum(),
                         phase2_C=r["C"][ph2].mean(), phase2_energy=r["energy"][ph2].mean(),
                         phase1_Kp=r["Kp"][p1 - 1], phase2_Kp=r["Kp"][ph2].mean(),
                         phase2_moved=r["moved"][ph2].mean()))
    return rows


# ---------------------------------------------------------------------------
# PH3  Habituation of the HPA response to a repeated controllable stressor
# Six 1000-step blocks; threat active for the first 300 steps of each block.
def habituation(arch: str, cond: str, seed: int) -> dict:
    w = make_layout(13000 + seed)
    blocks = [(i * 1000, i * 1000 + 300) for i in range(6)]
    cfg = EnvConfig(threat_blocks=blocks, uncontrollable=(cond == "uncontrollable"), p_damage_uncontrollable=0.08)
    r = run_phases(agent_config(arch), seed, [(6000, w, cfg)])
    peaks = [r["C"][a:a + 500].max() for a, _ in blocks]
    return dict(phenomenon="PH3", arch=arch, condition=cond, seed=seed, slope_peak_C=_slope(peaks),
                first_peak=peaks[0], last_peak=peaks[-1])


# ---------------------------------------------------------------------------
# PH4  Social buffering of the cortisol response (Heinrichs et al. 2003)
def social_buffering(arch: str, cond: str, seed: int) -> dict:
    base = make_layout(14000 + seed)
    w = with_companion(base, 14000 + seed) if cond == "companion" else base
    r = run_phases(agent_config(arch, social=True), seed,
                   [(6000, w, EnvConfig(uncontrollable=True))])
    late = slice(4000, 6000)
    return dict(phenomenon="PH4", arch=arch, condition=cond, seed=seed, C=r["C"][late].mean(),
                energy=r["energy"][late].mean(), food=r["ate"][late].sum())


# ---------------------------------------------------------------------------
# PH5  Inverted U of stress on learning (Yerkes-Dodson; Lupien et al. 2007)
# Background random damage at four rates; the world switches at 3000 steps;
# learning = food found in the first 1000 steps of the new world.
STRESS_LEVELS = {"0": 0.0, "1": 0.01, "2": 0.02, "3": 0.04}


def inverted_u(arch: str, cond: str, seed: int) -> dict:
    p = STRESS_LEVELS[cond]
    w1, w2 = make_layout(15000 + seed), make_layout(15500 + seed)
    cfg = EnvConfig(uncontrollable=p > 0, p_damage_uncontrollable=p, no_damage=(p == 0))
    r = run_phases(agent_config(arch), seed, [(3000, w1, cfg), (1000, w2, cfg)])
    return dict(phenomenon="PH5", arch=arch, condition=cond, seed=seed,
                new_world_food=r["ate"][3000:].sum(), mean_C=r["C"][3000:].mean())


# ---------------------------------------------------------------------------
# PH7  Learning rate rises with volatility (Behrens et al. 2007)
def volatility(arch: str, cond: str, seed: int) -> dict:
    every = 500 if cond == "volatile" else 4000
    phases = [(every, make_layout(17000 + 100 * seed + i), EnvConfig()) for i in range(12000 // every)]
    r = run_phases(agent_config(arch), seed, phases)
    return dict(phenomenon="PH7", arch=arch, condition=cond, seed=seed,
                mean_alpha=r["alpha"][1000:].mean(), drive=r["drive"][1000:].mean())


# ---------------------------------------------------------------------------
# PH8  Social withdrawal under uncontrollable stress (unplanned in round 3)
def withdrawal(arch: str, seed: int) -> dict:
    w = with_companion(make_layout(18000 + seed), 18000 + seed)
    r = run_phases(agent_config(arch, social=True), seed, [(6000, w, EnvConfig(uncontrollable=True))])
    late = slice(4000, 6000)
    return dict(phenomenon="PH8", arch=arch, condition="uncontrollable", seed=seed,
                with_companion=r["social"][late].mean(), belonging=r["belonging"][late].mean(),
                energy=r["energy"][late].mean())


# ---------------------------------------------------------------------------
# PH0  Core adaptation: one world for 3000 steps, then a new world (v0.5 regression check)
def transfer(arch: str, seed: int) -> dict:
    w1, w2 = make_layout(19000 + seed), make_layout(19500 + seed)
    r = run_phases(agent_config(arch), seed, [(3000, w1, EnvConfig()), (2000, w2, EnvConfig())])
    new = slice(3000, 4000)
    return dict(phenomenon="PH0", arch=arch, condition="switch", seed=seed,
                drive=r["drive"][new].mean(), food=r["ate"][new].sum(), damage=r["damage"][new].sum())
