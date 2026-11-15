"""Vectorised population simulator: B agents, each in its own world, stepped in lockstep.

Same model as the reference implementation (bio_rl.envs / needs / appraisal /
endocrine / learning, architecture v0.5), rewritten with arrays so that
hundreds of agents (e.g. a whole CMA-ES generation) run at once. Every
parameter is per agent, so one batch can hold many candidate architectures.

Additions that the reference implementation does not have:
* death: an agent dies when its energy or its safety reaches 0, and is frozen;
* ecologies: per-agent danger, lethality, volatility, food scarcity, companion;
* agent kinds: "needs" (homeostatic utility, with or without hormone
  channels), "fixed" (designer reward) and "pearcehall" (needs + adaptive
  associability).
Validated against the reference in experiments/evo/validate_popsim.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bio_rl.envs.gridworld import MOVES, make_layout, with_companion

S, NA, NN = 64, 5, 3          # states (8x8), actions, needs (energy, safety, belonging)
SIZE = 8


# ---------------------------------------------------------------------------
# worlds
@dataclass
class LayoutBank:
    food: np.ndarray       # (L, 2) state index of the two food cells
    shelter: np.ndarray    # (L,)
    threat: np.ndarray     # (L, S) bool
    companion: np.ndarray  # (L,)  state index
    size: int = 8          # grid side (8 in the original world, 12 in the second environment E2)

    @property
    def SIZE(self) -> int:
        return self.size

    @property
    def S(self) -> int:
        return self.size * self.size

    @classmethod
    def build_e2(cls, seeds) -> "LayoutBank":
        """Second environment E2: 12 x 12 grid, 3 food cells, 1 shelter, 2 threat clusters of 7 cells."""
        from bio_rl.envs.gridworld import make_layout_e2
        size = 12
        f, sh, th = [], [], []
        for sd in seeds:
            lay = make_layout_e2(int(sd))
            idx = lambda rc: rc[0] * size + rc[1]
            f.append([idx(x) for x in lay.food])
            sh.append(idx(lay.shelter))
            m = np.zeros(size * size, bool)
            m[[idx(x) for x in lay.threat]] = True
            th.append(m)
        sh = np.array(sh)
        return cls(np.array(f), sh, np.array(th), sh.copy(), size=size)

    @classmethod
    def build(cls, seeds) -> "LayoutBank":
        f, sh, th, co = [], [], [], []
        for sd in seeds:
            lay = with_companion(make_layout(int(sd)), int(sd))
            idx = lambda rc: rc[0] * SIZE + rc[1]
            f.append([idx(x) for x in lay.food])
            sh.append(idx(lay.shelter))
            m = np.zeros(S, bool)
            m[[idx(x) for x in lay.threat]] = True
            th.append(m)
            co.append(idx(lay.companion))
        return cls(np.array(f), np.array(sh), np.array(th), np.array(co))

    def __len__(self) -> int:
        return len(self.shelter)


NEED = dict(energy_decay=0.008, move_cost=0.002, food_gain=0.35, safety_recovery=0.004,
            shelter_recovery=0.03, belonging_decay=0.004, contact_gain=0.05)


ECO_KEYS = ("p_zone", "p_random", "damage_loss", "switch_every", "regrow", "companion", "energy_decay")


def ecology_arrays(B: int, **kw) -> dict:
    """Per-agent ecology parameters; scalars are broadcast."""
    d = dict(p_zone=0.35, p_random=0.0, damage_loss=0.25, switch_every=0, regrow=30, companion=False,
             energy_decay=NEED["energy_decay"])
    d.update(kw)
    out = {k: np.broadcast_to(np.asarray(v), (B,)).copy() for k, v in d.items()}
    out["switch_every"] = out["switch_every"].astype(int)
    out["regrow"] = out["regrow"].astype(int)
    out["companion"] = out["companion"].astype(bool)
    return out


# ---------------------------------------------------------------------------
# agents
PARAM_DEFAULTS = dict(
    alpha=0.1, epsilon=0.05, gamma=0.9,
    # hormone channels (0 = channel off)
    c_to_safety=0.0, c_to_belonging=0.0, ot_buffer=0.0, kp_effort=0.0, kp_explore=0.0,
    ne_to_alpha=0.0, ne_to_temp=0.0, c_to_temp=0.0,
    # endocrine dynamics
    kappa_c=0.05, tau_c=150.0, feedback=0.02, tau_ne=8.0, k_ne=1.5, tau_ot=100.0, k_ot=0.05,
    tau_k=300.0, tau_kp=1500.0,
    # fixed-reward agent
    w_damage=1.0, living_cost=0.01,
    # constant risk attitude (no hormones): safety weight x safety_mult
    safety_mult=1.0,
    # pearce-hall agent
    ph_rate=0.05,
)
DELAY = 10


def param_arrays(B: int, **kw) -> dict:
    d = dict(PARAM_DEFAULTS)
    d.update(kw)
    return {k: np.broadcast_to(np.asarray(v, float), (B,)).copy() for k, v in d.items()}


# ---------------------------------------------------------------------------
def simulate(kind: str, params: dict, eco: dict, bank: LayoutBank, layout_idx: np.ndarray, T: int,
             seed: int, death: bool = True, damage_schedule: np.ndarray | None = None,
             phases: list | None = None, record: tuple = (), grace: int = 30, juvenile: int = 500,
             injury_grace: int | None = None, social_need: bool = True) -> dict:
    """Run B agents for T steps.

    kind: "needs" | "fixed" | "pearcehall".
    layout_idx: (B,) starting layout of each agent in ``bank``.
    damage_schedule: optional (B, T) int; 1/0 = damage / no damage regardless of behaviour
        (yoked design), -1 = use the ecology.
    phases: optional list of (start_step, layout_idx (B,) or None, eco dict[, params dict]) to change
        world, ecology and (for oracle agents) parameters at fixed steps. layout None = keep.
    social_need: False removes the belonging need (held at its set-point).
    Returns also ``score``: mean over all T steps of 1 - drive/2 while alive, 0 after death
    (wellbeing; 1 = all needs met for the whole life).
    record: names of per-step traces to return, e.g. ("moved", "ate", "C", "Kp").
    Death: an agent dies when its energy or safety has been at 0 for ``grace``
    consecutive steps, but not during the first ``juvenile`` steps (parental care).
    ``injury_grace`` (default = ``grace``) sets the run length for safety separately;
    injury_grace = 1 means a hit that takes safety to 0 kills at once.
    Returns per-agent summaries plus requested traces (T, B).
    """
    rng = np.random.default_rng(seed)
    B = len(layout_idx)
    P = params
    ar = np.arange(B)
    lay = layout_idx.copy()
    food = bank.food[lay].copy()
    shelter = bank.shelter[lay].copy()
    threat = bank.threat[lay].copy()
    comp = np.where(eco["companion"], bank.companion[lay], -1)
    pos = shelter.copy()
    timers = np.zeros((B, 2), int)
    phases = sorted(phases or [], key=lambda p: p[0])

    n = np.full((B, NN), 0.8)
    if not social_need:
        n[:, 2] = 1.0
    sum_score = np.zeros(B)
    nq = 1 if kind == "fixed" else NN
    Q = np.zeros((B, nq, S, NA))
    T_map = np.zeros((B, S))
    n_all = np.zeros(B); n_safe = np.zeros(B); p_all = np.zeros(B); p_safe = np.zeros(B)
    K = np.full(B, 0.5); Kp = np.full(B, 0.5)
    NE = np.zeros(B); C = np.zeros(B); OT = np.zeros(B)
    mean_abs = np.full(B, 0.05)
    buf = np.zeros((B, DELAY)); bi = 0
    assoc = np.ones(B); ph_mean = np.full(B, 0.05)

    alive = np.ones(B, bool)
    zero_e = np.zeros(B, int)
    zero_s = np.zeros(B, int)
    injury_grace = grace if injury_grace is None else injury_grace
    life = np.full(B, T)
    sum_drive = np.zeros(B); sum_alive = np.zeros(B)
    tot = {k: np.zeros(B) for k in ("ate", "damage", "moved", "shelter", "social", "C", "Kp")}
    traces = {k: np.zeros((T, B), np.float32) for k in record}

    def weights():
        w = 2.0 * np.maximum(1.0 - n, 0.0) + 0.05
        w[:, 1] *= (1.0 + P["c_to_safety"] * C) * P["safety_mult"]
        w[:, 2] *= 1.0 + P["c_to_belonging"] * C
        return w

    for t in range(T):
        # ---- world changes
        while phases and phases[0][0] == t:
            ph = phases.pop(0)
            new_lay, eco = ph[1], ph[2]
            if len(ph) > 3 and ph[3] is not None:
                P = ph[3]
            if new_lay is not None:
                lay = new_lay.copy()
                food, shelter, threat = bank.food[lay].copy(), bank.shelter[lay].copy(), bank.threat[lay].copy()
                timers[:] = 0
            comp = np.where(eco["companion"], bank.companion[lay], -1)
        sw = (eco["switch_every"] > 0) & (t > 0)
        if sw.any():
            sw &= (t % np.maximum(eco["switch_every"], 1)) == 0
            if sw.any():
                lay[sw] = rng.integers(len(bank), size=int(sw.sum()))
                food[sw], shelter[sw], threat[sw] = bank.food[lay[sw]], bank.shelter[lay[sw]], bank.threat[lay[sw]]
                comp[sw] = np.where(eco["companion"][sw], bank.companion[lay[sw]], -1)
                timers[sw] = 0

        # ---- act
        s = pos
        if kind == "fixed":
            q = Q[ar, 0, s, :]
        else:
            w = weights()
            q = np.einsum("bk,bka->ba", w, Q[ar, :, s, :])
            ke = P["kp_effort"]
            if np.any(ke > 0):
                effort = ke * w[:, 0] * NEED["move_cost"]
                q[:, 1:] = Kp[:, None] * q[:, 1:] + (1.0 - Kp[:, None]) * (q[:, :1] - effort[:, None])
        eps = P["epsilon"] * (1.0 + P["ne_to_temp"] * NE) / (1.0 + P["c_to_temp"] * C)
        eps = np.clip(eps / (1.0 + P["kp_explore"] * (1.0 - Kp)), 0.0, 0.5)
        greedy = np.argmax(q + 1e-12 * rng.random((B, NA)), axis=1)
        a = np.where(rng.random(B) < eps, rng.integers(NA, size=B), greedy)

        # ---- environment
        r0 = np.clip(s // SIZE + MOVES[a, 0], 0, SIZE - 1)
        c0 = np.clip(s % SIZE + MOVES[a, 1], 0, SIZE - 1)
        s2 = r0 * SIZE + c0
        on_food = food == s2[:, None]
        ate_slot = on_food & (timers == 0)
        ate = ate_slot.any(1)
        timers = np.where(ate_slot, eco["regrow"][:, None], np.maximum(timers - 1, 0))
        u = rng.random(B)
        dmg = (threat[ar, s2] & (u < eco["p_zone"])) | (rng.random(B) < eco["p_random"])
        if damage_schedule is not None and t < damage_schedule.shape[1]:
            sched = damage_schedule[:, t]
            dmg = np.where(sched >= 0, sched > 0, dmg)
        in_sh = s2 == shelter
        social = s2 == comp
        moved = a != 0

        # ---- needs (felt change before clipping at 0)
        old = n.copy()
        raw = n.copy()
        raw[:, 0] += -eco["energy_decay"] - NEED["move_cost"] * moved + NEED["food_gain"] * ate
        raw[:, 1] += NEED["safety_recovery"] + NEED["shelter_recovery"] * in_sh - eco["damage_loss"] * dmg
        raw[:, 2] += -NEED["belonging_decay"] + NEED["contact_gain"] * social
        if not social_need:
            raw[:, 2] = 1.0
        felt = np.minimum(raw, 1.0) - old
        n_new = np.clip(raw, 0.0, 1.0)

        # ---- appraisal (contingency estimate, v0.5)
        d = dmg.astype(float)
        expected = T_map[ar, s2]
        safe_hit = d * (expected < 0.05)
        T_map[ar, s2] += 0.1 * (d - expected)
        f1, g1 = 1.0 - 1.0 / P["tau_k"], 1.0 - 1.0 / P["tau_kp"]
        n_all = f1 * n_all + d; n_safe = f1 * n_safe + safe_hit
        p_all = g1 * p_all + d; p_safe = g1 * p_safe + safe_hit
        K = 1.0 - (n_safe + 1.0) / (n_all + 2.0)
        Kp_new = 1.0 - (p_safe + 1.0) / (p_all + 2.0)
        threat_now = expected + d

        # ---- learning
        n = np.where(alive[:, None], n_new, n)
        if kind == "fixed":
            rew = (ate.astype(float) - P["w_damage"] * d - P["living_cost"])[:, None]
            Qs2 = Q[ar, :, s2, :]
            a2 = np.argmax(Qs2[:, 0, :], axis=1)
        else:
            rew = felt
            w2 = weights()
            Qs2 = Q[ar, :, s2, :]
            a2 = np.argmax(np.einsum("bk,bka->ba", w2, Qs2), axis=1)
        delta = rew + P["gamma"][:, None] * Qs2[ar, :, a2] - Q[ar, :, s, a]
        alpha = P["alpha"] * (1.0 + P["ne_to_alpha"] * NE)
        if kind == "pearcehall":
            alpha = P["alpha"] * np.clip(assoc, 0.25, 4.0)
        alpha = np.minimum(alpha, 0.9) * alive
        Q[ar, :, s, a] += alpha[:, None] * delta
        absd = np.abs((w2 * delta).sum(1)) if kind != "fixed" else np.abs(delta[:, 0])
        if kind == "pearcehall":
            ph_mean += (absd - ph_mean) / 200.0
            assoc += P["ph_rate"] * (absd / (ph_mean + 1e-6) - assoc)

        # ---- endocrine (U from NE before this step's update, as in the reference)
        U = np.clip(NE, 0.0, 1.0)
        OT = np.clip(OT - OT / P["tau_ot"] + P["k_ot"] * social, 0.0, 3.0)
        excess = np.maximum(absd - mean_abs, 0.0)
        mean_abs += (absd - mean_abs) / 200.0
        NE = np.clip(NE - NE / P["tau_ne"] + P["k_ne"] * excess, 0.0, 3.0)
        buf[:, bi] = threat_now
        bi = (bi + 1) % DELAY
        delayed = buf[:, bi]          # oldest entry = threat DELAY-1 steps ago
        inp = P["kappa_c"] * delayed * (1.0 - K) * (1.0 + U) * np.maximum(0.0, 1.0 - P["ot_buffer"] * OT)
        C = np.clip(C - C / P["tau_c"] + inp - P["feedback"] * C ** 2, 0.0, 3.0)
        Kp = Kp_new

        # ---- bookkeeping
        pos = np.where(alive, s2, pos)
        drive = ((1.0 - n) ** 2).sum(1)
        sum_drive += drive * alive
        sum_score += (1.0 - drive / 2.0) * alive
        sum_alive += alive
        for k, v in (("ate", ate), ("damage", dmg), ("moved", moved), ("shelter", in_sh),
                     ("social", social), ("C", C), ("Kp", Kp)):
            tot[k] += v * alive
        for k in record:
            traces[k][t] = {"ate": ate, "damage": dmg, "moved": moved, "shelter": in_sh, "social": social,
                            "C": C, "Kp": Kp, "K": K, "NE": NE, "OT": OT, "energy": n[:, 0], "safety": n[:, 1],
                            "belonging": n[:, 2], "alive": alive}[k]
        if death:
            zero_e = np.where(n[:, 0] <= 0.0, zero_e + 1, 0)
            zero_s = np.where(n[:, 1] <= 0.0, zero_s + 1, 0)
            died = alive & ((zero_e >= grace) | (zero_s >= injury_grace)) & (t >= juvenile)
            life[died] = t + 1
            alive &= ~died

    out = dict(survival=life / T, life=life, drive=sum_drive / np.maximum(sum_alive, 1), score=sum_score / T)
    for k, v in tot.items():
        out[k] = v / np.maximum(sum_alive, 1)
    out.update(traces)
    return out
