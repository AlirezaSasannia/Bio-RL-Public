"""Single-agent adaptation study: one learner, many ways of setting its knobs online.

The learner is the need-based agent of popsim (energy and safety, one value table
per need). What differs between agents is the *controller* that sets three knobs at
every step: learning rate alpha, exploration epsilon and the weight of safety
(risk attitude). Every controller sees only what the agent experiences; none is
told the hidden regime, except the "oracle" reference.

Controllers
-----------
fixed      constant knobs                                            (alpha0, eps0, s0)
reactive   endocrine v0.5 channels (cortisol, NE, controllability prior)
fuzzy      biologically derived fuzzy rules -> hormones (NE, cortisol, ghrelin)
           -> receptors (cortisol via MR and GR) -> knobs
plastic    fuzzy + receptor plasticity: the sensitivity of each hormone's effect
           on each knob is learned within the life by a reward-modulated
           three-factor rule (perturbation x hormone level x wellbeing change)
vdbe       value-difference based exploration (Tokic 2010), per-state epsilon
changept   Page-Hinkley change detection on surprise; boost alpha / epsilon after a change
pearcehall adaptive learning rate from recent surprise (Pearce & Hall 1980)
oracle     knobs set from a table indexed by the true regime (upper reference)
vdbe_metagrad  online SPSA meta-gradient on the three knob offsets (stage B9 baseline)
vdbe_bandit    sliding-window UCB over 7 knob settings (stage B9 baseline, Agent57-style)

Every controller also has a base risk attitude s0, so that all agents can choose a
constant safety weight; hormones only add *changes* of it.
"""
from __future__ import annotations

import numpy as np

from bio_rl.envs.gridworld import MOVES
from bio_rl.fast.popsim import NEED, LayoutBank
from bio_rl.fast.regimes import DECAY, LOSS, NAMES, REGIMES

S, NA, SIZE = 64, 5, 8
# regime table: p_zone, p_random, regrow, switch_every
REG_TAB = np.array([[REGIMES[n]["p_zone"], REGIMES[n]["p_random"], REGIMES[n]["regrow"], REGIMES[n]["switch_every"]]
                    for n in NAMES])

DEFAULTS = dict(alpha0=0.3, eps0=0.1, s0=1.0, gamma=0.9,
                # reactive (v0.5 channels)
                c_to_safety=0.0, ne_to_alpha=0.0, ne_to_temp=0.0, c_to_temp=0.0, kp_explore=0.0, kp_effort=0.0,
                kappa_c=0.05,
                # fuzzy: membership centres and gains (rows NE, MR, GR, ghrelin; cols alpha, eps, safety)
                c_thr=0.05, c_k=0.5, c_sur=0.05, c_intake=0.01,
                **{f"g_{h}_{k}": 0.0 for h in ("ne", "mr", "gr", "gh") for k in ("a", "e", "s")},
                # plasticity
                eta=0.0, sigma=0.0,
                # generic linear meta-controller (inputs threat, K, surprise, intake, energy deficit)
                **{f"w_{i}_{k}": 0.0 for i in range(6) for k in ("a", "e", "s")},
                # split surprise (B11): 0 off; 1 two-timescale |delta|; 2 signed mean / spread of delta
                split=0.0, tau_split=50.0,
                rule_seed=0, yoked=0.0, b_a=0.0, b_e=0.0, b_s=0.0, clamp_i=-1.0, clamp_v=0.0,
                # state augmentation (same 5 appraisal inputs fed to the learner instead of the knobs)
                **{f"thr_{i}": 0.5 for i in range(5)}, eta_th=0.01, x_scale=1.0,
                # meta-gradient (SPSA): initial offsets, step size, perturbation size, window length
                m_a=0.0, m_e=0.0, m_s=0.0, mg_lr=0.0, mg_c=0.0, mg_W=50.0,
                # bandit (sliding-window UCB): centre offsets, per-knob step, block length, window (pulls), UCB coef
                n_a=0.0, n_e=0.0, n_s=0.0, st_a=0.0, st_e=0.0, st_s=0.0, bd_N=50.0, bd_tau=50.0, bd_c=0.1,
                # vdbe
                vdbe_sigma=0.1,
                # change point
                ph_lambda=0.005, ph_h=0.5, boost_a=0.0, boost_e=0.0, boost_tau=100.0,
                # pearce-hall
                ph_rate=0.05)
HORMONES, KNOBS = ("ne", "mr", "gr", "gh"), ("a", "e", "s")
SC_CHG = {1: 0.001, 2: 0.003}     # input scales of the split signals (B11, fixed on development seeds)
SC_NOI = {1: 0.01, 2: 0.02}


def params(B: int, **kw) -> dict:
    d = dict(DEFAULTS)
    d.update(kw)
    return {k: np.broadcast_to(np.asarray(v, float), (B,)).copy() for k, v in d.items()}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def simulate(controller: str, P: dict, regimes: np.ndarray, bank: LayoutBank, layout_idx: np.ndarray, seed: int,
             oracle_table: np.ndarray | None = None, grace: int = 100, juvenile: int = 500,
             record: tuple = (), loss: float = LOSS, reg_tab: np.ndarray | None = None,
             decay: float | None = None, hook=None) -> dict:
    """regimes: (T, B) int regime index per step and agent (hidden from all but the oracle).
    oracle_table: (n_regimes, 3) knobs (alpha, eps, safety_mult) for the oracle.
    World size and food count come from the bank (default world: 8 x 8, 2 food cells); reg_tab and decay override
    the regime table and the energy decay (second environment E2).
    hook: optional read-only callback hook(t, locals()) at the end of every step (web viewer); draws no random
    numbers and changes no state, so results are identical with or without it."""
    S, SIZE = bank.S, bank.SIZE
    REG_TAB_ = REG_TAB if reg_tab is None else reg_tab
    DECAY_ = DECAY if decay is None else decay
    T, B = regimes.shape
    rng = np.random.default_rng(seed)
    ar = np.arange(B)
    lay = layout_idx.copy()
    food, shelter, threat = bank.food[lay].copy(), bank.shelter[lay].copy(), bank.threat[lay].copy()
    pos = shelter.copy()
    timers = np.zeros((B, bank.food.shape[1]), int)
    n = np.full((B, 2), 0.8)
    Q = np.zeros((B, 2, S * (32 if controller == 'vdbe_augtab' else 1), NA))
    T_map = np.zeros((B, S))
    n_all = np.zeros(B); n_safe = np.zeros(B); p_all = np.zeros(B); p_safe = np.zeros(B)
    K = np.full(B, 0.5); Kp = np.full(B, 0.5)
    NE = np.zeros(B); C = np.zeros(B); G = np.zeros(B)
    mean_abs = np.full(B, 0.05)
    buf = np.zeros((B, 10)); bi = 0
    thr_avg = np.zeros(B); intake = np.full(B, 0.02); sur = np.zeros(B)
    lin = np.zeros((B, 3))
    AUG = controller in ("vdbe_augtab", "vdbe_auglin")
    NC = 32 if controller == "vdbe_augtab" else 1
    ctx = np.zeros(B, int)                                  # augtab: discretised appraisal context
    xn = np.zeros((B, 5))                                   # normalised appraisal inputs
    Th = np.zeros((B, 2, NA, 5))                            # auglin: per-need, per-action linear weights
    SPLIT = int(P["split"][0])
    NIN = 6 if SPLIT else 5
    xin = np.zeros((B, NIN)); eff_rec = np.zeros((B, 3))    # recorded inputs and knob offsets
    d_m = np.zeros(B); d_v = np.full(B, 0.05 ** 2); f_abs = np.full(B, 0.05)    # split-surprise state
    thr_cut = np.stack([P[f"thr_{i}"] for i in range(5)], 1)
    Wg = np.stack([np.stack([P[f"w_{i}_{k}"] for i in range(NIN)], 1) for k in KNOBS], 1)   # (B, 3, 5 or 6)
    _rr = np.random.default_rng(int(P["rule_seed"][0]))
    rand_rules = [tuple((int(i), bool(_rr.random() < 0.5)) for i in _rr.choice(5, 2, replace=False)) for _ in range(3)]
    # plasticity state
    rho = np.ones((B, 4, 3)); elig = np.zeros((B, 4, 3)); xi = np.zeros((B, 3)); r_win = np.zeros(B)
    r_base = np.full(B, 0.8); W = 50
    gains = np.stack([np.stack([P[f"g_{h}_{k}"] for k in KNOBS], 1) for h in HORMONES], 1)   # (B, 4, 3)
    eps_tab = np.repeat(P["eps0"][:, None], S, 1)           # vdbe
    ph_m = np.zeros(B); ph_M = np.zeros(B); ph_mean = np.full(B, 0.05); boost = np.zeros(B)
    assoc = np.ones(B); pmean = np.full(B, 0.05)

    # meta-gradient (SPSA, antithetic pair of windows): theta are the knob offsets being learned
    theta = np.stack([P["m_a"], P["m_e"], P["m_s"]], 1)
    mg_W = np.maximum(np.round(P["mg_W"]), 1).astype(int)
    rng_c = np.random.default_rng(seed + 7_777_777)         # controller noise, separate so the world's noise stays common
    Dlt = rng_c.choice([-1.0, 1.0], size=(B, 3)) if controller == "vdbe_metagrad" else np.zeros((B, 3))
    mg_phase = np.zeros(B, int); mg_acc = np.zeros(B); mg_rA = np.zeros(B); mg_k = np.zeros(B, int)
    # bandit (sliding-window UCB over 7 arms: centre, +-step on one knob at a time)
    ARMV = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    centre = np.stack([P["n_a"], P["n_e"], P["n_s"]], 1)
    step = np.stack([P["st_a"], P["st_e"], P["st_s"]], 1)
    bd_N = np.maximum(np.round(P["bd_N"]), 1).astype(int)
    HMAX = T // max(int(bd_N.min()), 1) + 2 if controller == "vdbe_bandit" else 1
    h_arm = np.zeros((B, HMAX), int); h_r = np.zeros((B, HMAX)); n_pull = np.zeros(B, int)
    arm_now = np.zeros(B, int); bd_k = np.zeros(B, int); bd_acc = np.zeros(B)

    alive = np.ones(B, bool); life = np.full(B, T); zero_e = np.zeros(B, int)
    score = np.zeros(B)
    traces = {k: np.zeros((T, B), np.float32) for k in record}

    for t in range(T):
        reg = regimes[t]
        p_zone, p_rand, regrow, sw_every = (REG_TAB_[reg, i] for i in range(4))
        regrow = regrow.astype(int)
        sw = (sw_every > 0) & (t > 0) & ((t % np.maximum(sw_every, 1).astype(int)) == 0)
        if sw.any():
            lay[sw] = rng.integers(len(bank), size=int(sw.sum()))
            food[sw], shelter[sw], threat[sw] = bank.food[lay[sw]], bank.shelter[lay[sw]], bank.threat[lay[sw]]
            timers[sw] = 0

        # ---------------- knobs from the controller
        phi = None
        if controller == "oracle":
            a_t, e_t, s_t = (oracle_table[reg, i] for i in range(3))
        elif AUG:
            eff = np.zeros((B, 3))
            a_t, e_t, s_t = P["alpha0"], P["eps0"], P["s0"]
        elif controller in ("vdbe_generic", "vdbe_metagrad", "vdbe_bandit"):
            if controller == "vdbe_generic":
                eff = lin
            elif controller == "vdbe_metagrad":
                eff = theta + (1 - 2 * mg_phase)[:, None] * P["mg_c"][:, None] * Dlt
            else:
                eff = centre + step * ARMV[arm_now]
            a_t = np.exp(np.log(P["alpha0"]) + eff[:, 0])
            e_t = _sig(_logit(P["eps0"]) + eff[:, 1])
            s_t = P["s0"] * np.exp(eff[:, 2])
        elif controller in ("fuzzy", "plastic", "fixed_fuzzy", "vdbe_fuzzy", "vdbe_randrules", "vdbe_nomem"):
            MR = C / (C + 0.1)
            GR = C ** 2 / (C ** 2 + 0.36)
            phi = np.stack([NE, MR, GR, G], 1)                                   # (B, 4)
            eff = np.einsum("bh,bhk->bk", phi, gains * rho)                      # (B, 3)
            if controller == "plastic":
                eff = eff + xi
            a_t = np.exp(np.log(P["alpha0"]) + eff[:, 0])
            e_t = _sig(_logit(P["eps0"]) + eff[:, 1])
            s_t = P["s0"] * np.exp(eff[:, 2])
        elif controller == "reactive":
            a_t = P["alpha0"] * (1.0 + P["ne_to_alpha"] * NE)
            e_t = P["eps0"] * (1.0 + P["ne_to_temp"] * NE) / (1.0 + P["c_to_temp"] * C) / (1.0 + P["kp_explore"] * (1 - Kp))
            s_t = P["s0"] * (1.0 + P["c_to_safety"] * C)
        elif controller == "changept":
            a_t = P["alpha0"] * (1.0 + P["boost_a"] * boost)
            e_t = P["eps0"] + P["boost_e"] * boost
            s_t = P["s0"]
        elif controller == "pearcehall":
            a_t = P["alpha0"] * np.clip(assoc, 0.25, 4.0)
            e_t, s_t = P["eps0"], P["s0"]
        else:   # fixed, vdbe
            a_t, e_t, s_t = P["alpha0"], P["eps0"], P["s0"]
        if phi is not None or controller in ("vdbe_generic", "vdbe_metagrad", "vdbe_bandit"):
            eff_rec = eff
        a_t = np.minimum(np.broadcast_to(a_t, (B,)), 0.9)
        e_t = np.clip(np.broadcast_to(e_t, (B,)), 0.0, 0.5)
        s_t = np.broadcast_to(s_t, (B,))

        # ---------------- act
        s = pos
        sq = s + S * ctx                                        # state index of the value table
        w = 2.0 * np.maximum(1.0 - n, 0.0) + 0.05
        w[:, 1] *= s_t
        Qs = Q[ar, :, sq, :]
        if controller == "vdbe_auglin":
            Qs = Qs + np.einsum("bkai,bi->bka", Th, xn)
        q = np.einsum("bk,bka->ba", w, Qs)
        if controller == "reactive" and np.any(P["kp_effort"] > 0):
            effort = P["kp_effort"] * w[:, 0] * NEED["move_cost"]
            q[:, 1:] = Kp[:, None] * q[:, 1:] + (1 - Kp[:, None]) * (q[:, :1] - effort[:, None])
        if controller == "vdbe":
            eps_now = np.clip(eps_tab[ar, s], 0, 0.5)
        elif controller.startswith("vdbe_"):  # modulators shift the per-state VDBE exploration (logit space)
            eps_now = np.clip(_sig(_logit(eps_tab[ar, s]) + eff[:, 1]), 0, 0.5)
        else:
            eps_now = e_t
        greedy = np.argmax(q + 1e-12 * rng.random((B, NA)), 1)
        a = np.where(rng.random(B) < eps_now, rng.integers(NA, size=B), greedy)

        # ---------------- environment
        r0 = np.clip(s // SIZE + MOVES[a, 0], 0, SIZE - 1)
        c0 = np.clip(s % SIZE + MOVES[a, 1], 0, SIZE - 1)
        s2 = r0 * SIZE + c0
        ate_slot = (food == s2[:, None]) & (timers == 0)
        ate = ate_slot.any(1)
        timers = np.where(ate_slot, regrow[:, None], np.maximum(timers - 1, 0))
        dmg = (threat[ar, s2] & (rng.random(B) < p_zone)) | (rng.random(B) < p_rand)
        in_sh = s2 == shelter
        moved = a != 0

        # ---------------- needs
        old = n.copy()
        raw = n.copy()
        raw[:, 0] += -DECAY_ - NEED["move_cost"] * moved + NEED["food_gain"] * ate
        raw[:, 1] += NEED["safety_recovery"] + NEED["shelter_recovery"] * in_sh - loss * dmg
        felt = np.minimum(raw, 1.0) - old
        n = np.where(alive[:, None], np.clip(raw, 0.0, 1.0), n)

        # ---------------- appraisal
        d = dmg.astype(float)
        expected = T_map[ar, s2]
        safe_hit = d * (expected < 0.05)
        T_map[ar, s2] += 0.1 * (d - expected)
        n_all = (1 - 1 / 300) * n_all + d; n_safe = (1 - 1 / 300) * n_safe + safe_hit
        p_all = (1 - 1 / 1500) * p_all + d; p_safe = (1 - 1 / 1500) * p_safe + safe_hit
        K = 1 - (n_safe + 1) / (n_all + 2)
        Kp = 1 - (p_safe + 1) / (p_all + 2)
        threat_now = expected + d

        # ---------------- learning
        w2 = 2.0 * np.maximum(1.0 - n, 0.0) + 0.05
        w2[:, 1] *= s_t
        Qs2 = Q[ar, :, s2 + S * ctx, :]
        if controller == "vdbe_auglin":
            Qs2 = Qs2 + np.einsum("bkai,bi->bka", Th, xn)
        a2 = np.argmax(np.einsum("bk,bka->ba", w2, Qs2), 1)
        delta = felt + P["gamma"][:, None] * Qs2[ar, :, a2] - Qs[ar, :, a]
        Q[ar, :, sq, a] += (a_t * alive)[:, None] * delta
        if controller == "vdbe_auglin":
            Th[ar, :, a, :] += (P["eta_th"] * alive)[:, None, None] * delta[:, :, None] * xn[:, None, :]
        dsig = (w2 * delta).sum(1)
        absd = np.abs(dsig)

        # ---------------- controller state updates
        U = np.clip(NE, 0, 1)
        excess = np.maximum(absd - mean_abs, 0.0)
        mean_abs += (absd - mean_abs) / 200.0
        thr_avg += (threat_now - thr_avg) / 50.0
        intake += (ate - intake) / 100.0
        if controller == "reactive":
            NE = np.clip(NE - NE / 8.0 + 1.5 * excess, 0, 3)
            buf[:, bi] = threat_now
            bi = (bi + 1) % 10
            inp = P["kappa_c"] * buf[:, bi] * (1 - K) * (1 + U)
            C = np.clip(C - C / 150.0 + inp - 0.02 * C ** 2, 0, 3)
        elif AUG:
            # same appraisal inputs as the generic controller, fed to the learner (context held for one step)
            sur = np.clip(sur - sur / 8.0 + 1.5 * excess, 0, 3)
            x = np.stack([thr_avg / 0.05, K, sur / 0.1, intake / 0.02, 1 - n[:, 0]], 1)
            xn = np.clip(x, 0, 3) / 3.0 * (P["x_scale"][:, None] if controller == "vdbe_auglin" else 1.0)
            if controller == "vdbe_augtab":
                ctx = ((xn > thr_cut) * (2 ** np.arange(5))).sum(1)
        elif controller == "vdbe_generic":
            # generic linear meta-controller: same appraisal inputs, same output knobs, leaky memory (tau 50)
            sur = np.clip(sur - sur / 8.0 + 1.5 * excess, 0, 3)
            x = np.stack([thr_avg / 0.05, K, sur / 0.1, intake / 0.02, 1 - n[:, 0]], 1)        # (B, 5)
            if SPLIT:   # surprise split into change (column 2) and noise (column 5); Piray & Daw 2021
                tau = P["tau_split"]
                if SPLIT == 1:      # change: fast minus slow mean |delta|; noise: slow mean |delta|
                    f_abs += (absd - f_abs) / tau
                    chg, noi = np.maximum(f_abs - mean_abs, 0.0), mean_abs
                else:               # change: |running mean of signed delta|; noise: spread around that mean
                    d_m += (dsig - d_m) / tau
                    d_v += (dsig ** 2 - d_v) / tau
                    chg, noi = np.abs(d_m), np.sqrt(np.maximum(d_v - d_m ** 2, 0.0))
                x = np.concatenate([x[:, :2], np.clip(chg / SC_CHG[SPLIT], 0, 30)[:, None], x[:, 3:],
                                    np.clip(noi / SC_NOI[SPLIT], 0, 30)[:, None]], 1)
            if P["yoked"][0] > 0:      # yoked control: inputs of another agent (different schedule), same statistics
                x = np.roll(x, 24, axis=0)
            ci = int(P["clamp_i"][0])
            if ci >= 0:                 # causal test: one input held at a constant value for the whole life
                x[:, ci] = P["clamp_v"]
            xin = x
            bias = np.stack([P["b_a"], P["b_e"], P["b_s"]], 1)                    # constant offsets (B8 control)
            target = np.clip(np.einsum("bki,bi->bk", Wg, x) + bias, -6, 6)
            lin += (target - lin) / 50.0
        elif controller in ("fuzzy", "plastic", "fixed_fuzzy", "vdbe_fuzzy", "vdbe_randrules", "vdbe_nomem"):
            sur = np.clip(sur - sur / 8.0 + 1.5 * excess, 0, 3)          # surprise integrator (input signal)
            xin = np.stack([thr_avg / 0.05, K, sur / 0.1, intake / 0.02, 1 - n[:, 0]], 1)
            hi = lambda x, c: _sig((x - c) / (0.5 * c))
            h_thr, h_k = hi(thr_avg, P["c_thr"]), _sig((K - P["c_k"]) / 0.1)
            h_sur, h_in = hi(sur, P["c_sur"]), hi(intake, P["c_intake"])
            # rules from endocrinology (paper, Table of rules)
            if controller == "vdbe_randrules":
                # same architecture, rules drawn at random (fixed per run): each hormone = product of two
                # memberships with random polarity
                mem = np.stack([h_thr, h_k, h_sur, h_in, 1 - n[:, 0]], 1)
                tg = []
                for (i1, p1), (i2, p2) in rand_rules:
                    m1 = mem[:, i1] if p1 else 1 - mem[:, i1]
                    m2 = mem[:, i2] if p2 else 1 - mem[:, i2]
                    tg.append(m1 * m2)
                ne_t, c_t, g_t = tg
            else:
                ne_t = h_sur * (1 - h_thr)                              # unexpected change without threat -> NE
                c_t = 0.5 * h_thr * h_k + 1.0 * h_thr * (1 - h_k)        # controllable -> moderate, uncontrollable -> high
                g_t = (1 - h_in) * (1 - n[:, 0]) * (1 - h_thr)          # low intake + low energy, no threat -> ghrelin
            if controller == "vdbe_nomem":                              # no hormonal dynamics: instant hormones
                NE, C, G = ne_t.copy(), c_t.copy(), g_t.copy()
            else:
                NE += (ne_t - NE) / 10.0
                C += (c_t - C) / 100.0
                G += (g_t - G) / 50.0
        if controller == "vdbe" or controller.startswith("vdbe_"):
            x = np.abs(a_t * (w2 * delta).sum(1)) / P["vdbe_sigma"]
            f = (1 - np.exp(-x)) / (1 + np.exp(-x))
            eps_tab[ar, s] = 0.2 * f + 0.8 * eps_tab[ar, s]
        elif controller == "changept":
            ph_mean += (absd - ph_mean) / 200.0
            ph_m += absd - ph_mean - P["ph_lambda"]
            ph_M = np.minimum(ph_M, ph_m)
            det = (ph_m - ph_M) > P["ph_h"]
            boost = np.where(det, 1.0, boost * (1 - 1 / P["boost_tau"]))
            ph_m = np.where(det, 0.0, ph_m); ph_M = np.where(det, 0.0, ph_M)
        elif controller == "pearcehall":
            pmean += (absd - pmean) / 200.0
            assoc += P["ph_rate"] * (absd / (pmean + 1e-6) - assoc)

        wellbeing = 1.0 - ((1 - n) ** 2).sum(1) / 2.0
        if controller == "vdbe_metagrad":
            # SPSA: window A with theta + c*D, window B with theta - c*D; g = (rA - rB) / (2c) * D
            mg_acc += wellbeing; mg_k += 1
            end = mg_k >= mg_W
            if end.any():
                r_w = mg_acc / np.maximum(mg_k, 1)
                endA = end & (mg_phase == 0)
                endB = end & (mg_phase == 1)
                mg_rA = np.where(endA, r_w, mg_rA)
                g = ((mg_rA - r_w) / (2 * np.maximum(P["mg_c"], 1e-6)))[:, None] * Dlt
                upd = endB & alive
                theta = np.where(upd[:, None], np.clip(theta + P["mg_lr"][:, None] * g, -6, 6), theta)
                newD = rng_c.choice([-1.0, 1.0], size=(B, 3))
                Dlt = np.where(endB[:, None], newD, Dlt)
                mg_phase = np.where(endA, 1, np.where(endB, 0, mg_phase))
                mg_acc = np.where(end, 0.0, mg_acc); mg_k = np.where(end, 0, mg_k)
        elif controller == "vdbe_bandit":
            # sliding-window UCB (Garivier & Moulines 2011); reward = mean wellbeing over the block
            bd_acc += wellbeing; bd_k += 1
            end = bd_k >= bd_N
            if end.any():
                ix = np.flatnonzero(end)
                k = n_pull[ix]
                h_arm[ix, k] = arm_now[ix]
                h_r[ix, k] = bd_acc[ix] / bd_k[ix]
                n_pull[ix] = k + 1
                j = np.arange(HMAX)[None, :]
                valid = (j < (k + 1)[:, None]) & (j >= (k + 1 - np.maximum(np.round(P["bd_tau"][ix]), 1))[:, None])
                oh = (h_arm[ix][:, :, None] == np.arange(7)[None, None, :]) & valid[:, :, None]
                cnt = oh.sum(1)
                mean = (oh * h_r[ix][:, :, None]).sum(1) / np.maximum(cnt, 1)
                nwin = np.minimum(k + 1, np.maximum(np.round(P["bd_tau"][ix]), 1))
                ucb = mean + P["bd_c"][ix, None] * np.sqrt(np.log(np.maximum(nwin, 2))[:, None] / np.maximum(cnt, 1))
                ucb = np.where(cnt == 0, np.inf, ucb)
                arm_now[ix] = np.argmax(ucb + 1e-12 * rng_c.random(ucb.shape), 1)
                bd_acc[ix] = 0.0; bd_k[ix] = 0
        if controller == "plastic":
            elig += phi[:, :, None] * xi[:, None, :] * gains
            r_win += wellbeing
            if (t + 1) % W == 0:
                r_w = r_win / W
                adv = (r_w - r_base)[:, None, None]
                sig2 = np.maximum(P["sigma"], 1e-3)[:, None, None] ** 2
                rho = np.clip(rho + P["eta"][:, None, None] * adv * elig / (W * sig2) * alive[:, None, None], 0.0, 3.0)
                rho += (1.0 - rho) / 40.0            # slow return to baseline sensitivity
                r_base += (r_w - r_base) / 5.0
                elig[:] = 0; r_win[:] = 0
                xi = rng.normal(0.0, 1.0, (B, 3)) * P["sigma"][:, None]

        # ---------------- bookkeeping
        pos = np.where(alive, s2, pos)
        score += wellbeing * alive
        for k in record:
            traces[k][t] = {"alive": alive, "wellbeing": wellbeing * alive, "alpha": a_t, "eps": eps_now,
                            "safety_mult": s_t, "NE": NE, "C": C, "G": G, "ate": ate, "damage": dmg,
                            **{f"x{i}": xin[:, i] for i in range(NIN)},
                            "ea": eff_rec[:, 0], "ee": eff_rec[:, 1], "es": eff_rec[:, 2]}[k]
        zero_e = np.where(n[:, 0] <= 0, zero_e + 1, 0)
        died = alive & ((zero_e >= grace) | (n[:, 1] <= 0)) & (t >= juvenile)
        life[died] = t + 1
        alive &= ~died
        if hook is not None:
            hook(t, locals())

    out = dict(score=score / T, survival=life / T)
    out.update(traces)
    return out


def regime_matrix(schedules: list, T: int) -> np.ndarray:
    """(T, B) regime indices from a list of B schedules [(start, name), ...]."""
    M = np.zeros((T, len(schedules)), int)
    for b, sch in enumerate(schedules):
        for (st, name), nxt in zip(sch, sch[1:] + [(T, None)]):
            M[st:nxt[0], b] = NAMES.index(name)
    return M
