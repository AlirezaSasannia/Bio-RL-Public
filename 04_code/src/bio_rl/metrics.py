"""Adaptation metrics computed from run traces."""
from __future__ import annotations

import numpy as np


def rolling(x: np.ndarray, w: int = 200) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, np.nan), out])


def summarize(r: dict, train_steps: int, window: int = 1000, abs_threshold: float = 0.3) -> dict:
    T = len(r["drive"])
    pre = slice(train_steps - window, train_steps)
    early = slice(train_steps, train_steps + window)
    post = slice(train_steps, T)
    pre_drive = r["drive"][pre].mean()
    thresh = max(1.5 * pre_drive, pre_drive + 0.05)
    roll = rolling(r["drive"])[train_steps + 199:]
    ok = np.flatnonzero(roll <= thresh)
    recovery = int(ok[0] + 200) if len(ok) else T - train_steps
    # same absolute threshold for every agent, so agents that tolerate a larger
    # deviation before the switch are not given an easier target
    ok_abs = np.flatnonzero(roll <= abs_threshold)
    recovery_abs = int(ok_abs[0] + 200) if len(ok_abs) else T - train_steps
    return {
        "pre_drive": pre_drive,
        "early_drive": r["drive"][early].mean(),
        "post_drive": r["drive"][post].mean(),
        "recovery_steps": recovery,
        "recovery_abs": recovery_abs,
        "early_food": r["ate"][early].sum(),
        "early_damage": r["damage"][early].sum(),
        "post_damage": r["damage"][post].sum(),
        "pre_damage": r["damage"][pre].sum(),
        "peak_C": r["C"][train_steps:].max(),
    }
