"""Batched PyTorch port of the regime-switching world of bio_rl.fast.adaptive (single-agent study, phase 2).

Same rules as adaptive.simulate: needs (energy, safety), hidden regimes with layout switches in the volatile
regime, damage, death (100 steps at zero energy or zero safety, never during the juvenile period), wellbeing
1 - sum((1 - n)^2) / 2 while alive. Many agents, each with its own world and parameters, are stepped together on
the GPU.

Learners
  tab   the tabular need-based learner of adaptive.py (one Q table per need), for validation against numpy
  dqnv  the dqn learner with a 5x5 egocentric view instead of the one-hot cell (channels: ready food, regrowing
        food, shelter, outside the grid, own remembered threat T_map), plus optional FiLM modulation of the hidden
        layer by the appraisal signals: pre' = pre * (1 + tanh(xf G)) + xf Fb, xf = clip(x, 0, 3) / 3 * fs; G, Fb
        start at 0 and are learned within the life by the same TD gradient (per-agent `fs_i` scales, 0 = off;
        `yoke` = index of the agent whose signals drive the FiLM and the global layer, default itself). Replayed
        views use the stored layout and food readiness and the current threat memory.
        Keyed memory (cfg mem_mode=True, dqnv only): replay buffer of M steps; each agent samples within its own
        window `mem` (<= M) of recent steps; the m - 1 non-newest samples are drawn from K = 64 uniform candidates
        by Gumbel-top-k on logits -sum_d (scale_d * (key_d(candidate) - key_d(now)))^2, key = 5 appraisal signals
        (clip 0-3, /3; of the `yoke` partner) with scales ks_0..4 and the state context (row/7, col/7, energy,
        safety) with scales kv_0..3; all scales 0 = uniform sampling within the window.
  gru   (stage C3) the dqnv learner with the hidden layer replaced by a GRU (hidden H) fed the view, cell
        coordinates, need levels, the last felt reward (2), the last action (one-hot) and the 5 signal slots (scale
        xs, 0 = no signals). Replay of stored-state sequences (R2D2-style, no burn-in): every step m // L sequences
        of L steps (the newest always included) start from the hidden state stored at their first step; gradients
        through the L steps by autograd, same Adam rule. No FiLM; the knobs as for dqnv.
  dqnv with cfg meta=True (stage C3 'metagrad'): the knob offsets are theta (start m_a, m_e, m_s) instead of the
        global layer; theta_a follows the one-step meta-gradient of the learning rate through the differentiable
        update (Xu et al. 2018, as in STAC): after each Adam step, d L_val / d theta_a = -<grad L_val(net'),
        lr * direction> on an independent replayed minibatch, RMS-normalised, step mg_lr. Exploration and safety
        weight have no gradient through the value loss and keep the constant offsets m_e, m_s.
  dqnv with cfg ra=True (stage C3b): the feed-forward view learner also receives the last felt reward (2) and the
        last action (one-hot), encoded as in the gru input; replayed transitions store them.
  dqnv with cfg seq=True (revision X2): replay sampled as in the gru (m // L contiguous sequences of L steps, the
        newest included) instead of m single transitions; no hidden state, so the update is unchanged otherwise.
  dqn   a small per-agent network: input = one-hot cell, cell coordinates, own need levels (and optionally the five
        appraisal signals); one value head per need (5 actions each), combined with the need weights exactly as in
        the table; felt need change as reward; per-agent replay buffer, double-DQN target, target network copied
        every C steps; Adam direction scaled by the learning-rate knob; manual backprop (2-layer MLP).
Both use per-cell VDBE exploration. Knob modulation (all per agent, zero = no effect):
  Wg (15) generic global layer weights, bias (3) constant offsets, xs (5) input scales of the appraisal signals
  fed to the network (dqn only; 0 = signals not given as input).
Random streams: `world` noise (damage, layout switches) and `agent` noise (actions, replay sampling) come from two
generators, so arms that differ only in the agent keep common random numbers for the world."""
from __future__ import annotations

import numpy as np
import torch

from bio_rl.fast.popsim import NEED
from bio_rl.fast.regimes import DECAY, LOSS, NAMES, REGIMES

S, NA, SIZE = 64, 5, 8
MOVES = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
REG_TAB = np.array([[REGIMES[n]["p_zone"], REGIMES[n]["p_random"], REGIMES[n]["regrow"], REGIMES[n]["switch_every"]]
                    for n in NAMES])
DQN_DEFAULTS = dict(H=32, M=1000, m=8, C=1, lr_max=0.1, adam_b1=0.9, adam_b2=0.999, needs_in=1.0, alias_target=True, mem_mode=False, K=64,
                    L=4, meta=False, ra=False, seq=False)


def _logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def simulate(learner: str, P: dict, regimes: np.ndarray, bank, layout_idx: np.ndarray, seed: int,
             grace: int = 100, juvenile: int = 500, loss: float = LOSS, record: tuple = (), device="cuda",
             dqn: dict | None = None, dtype=torch.float32, reg_tab: np.ndarray | None = None,
             decay: float | None = None, hook=None) -> dict:
    """P: dict of per-agent arrays (B,) with alpha0, eps0, s0, gamma, vdbe_sigma, w_{i}_{k} (generic weights),
    b_a, b_e, b_s (constant offsets) and, for dqn, xs_{i} (input scales). regimes: (T, B) regime indices.
    World size and food count come from the bank (default world 8 x 8, 2 food cells); dqn cfg 'view' is the view
    radius (2 = 5 x 5, default; 3 = 7 x 7 in E2); reg_tab and decay override the regime table and energy decay.
    hook: optional read-only callback hook(t, locals()) at the end of every step (web viewer); None = no effect."""
    S, SIZE = getattr(bank, "S", 64), getattr(bank, "SIZE", 8)
    NF = bank.food.shape[1]
    REG_TAB_ = REG_TAB if reg_tab is None else reg_tab
    DECAY_ = DECAY if decay is None else decay
    dev = torch.device(device)
    T, B = regimes.shape
    f = lambda k: torch.as_tensor(np.asarray(P[k], float), dtype=dtype, device=dev)
    gw = torch.Generator(device=dev).manual_seed(int(seed))
    ga = torch.Generator(device=dev).manual_seed(int(seed) + 1_000_003)
    rnd_w = lambda *s: torch.rand(*s, generator=gw, device=dev, dtype=dtype)
    rnd_a = lambda *s: torch.rand(*s, generator=ga, device=dev, dtype=dtype)
    ar = torch.arange(B, device=dev)
    REG = torch.as_tensor(regimes, device=dev)
    RT = torch.as_tensor(REG_TAB_, dtype=dtype, device=dev)
    bf, bs, bt = (torch.as_tensor(x, device=dev) for x in (bank.food, bank.shelter, bank.threat))
    L = len(bank)
    lay = torch.as_tensor(layout_idx, device=dev).long()
    food, shelter, threat = bf[lay].clone(), bs[lay].clone(), bt[lay].clone()
    pos = shelter.clone()
    timers = torch.zeros(B, NF, dtype=torch.long, device=dev)
    n = torch.full((B, 2), 0.8, dtype=dtype, device=dev)
    moves = MOVES.to(dev)

    alpha0, eps0, s0, gamma, vsig = f("alpha0"), f("eps0"), f("s0"), f("gamma"), f("vdbe_sigma")
    Wg = torch.stack([torch.stack([f(f"w_{i}_{k}") for i in range(5)], 1) for k in "aes"], 1)   # (B, 3, 5)
    bias = torch.stack([f("b_a"), f("b_e"), f("b_s")], 1)
    lin = torch.zeros(B, 3, dtype=dtype, device=dev)
    eps_tab = eps0[:, None].repeat(1, S)
    T_map = torch.zeros(B, S, dtype=dtype, device=dev)
    n_all, n_safe = torch.zeros(B, dtype=dtype, device=dev), torch.zeros(B, dtype=dtype, device=dev)
    K = torch.full((B,), 0.5, dtype=dtype, device=dev)
    mean_abs = torch.full((B,), 0.05, dtype=dtype, device=dev)
    thr_avg = torch.zeros(B, dtype=dtype, device=dev)
    intake = torch.full((B,), 0.02, dtype=dtype, device=dev)
    sur = torch.zeros(B, dtype=dtype, device=dev)
    x = torch.zeros(B, 5, dtype=dtype, device=dev)          # appraisal signals (as in adaptive.vdbe_generic)

    cfg = {**DQN_DEFAULTS, **(dqn or {})}
    GRU = learner == "gru"                            # learner flags defined for every learner (the tab learner
    VIEW = learner in ("dqnv", "gru")                 # crashed on the GRU flag after the C3 additions)
    META = bool(cfg["meta"]) and learner == "dqnv"
    RA = bool(cfg["ra"]) and learner == "dqnv"        # stage C3b: last felt reward and last action as input
    SEQ = bool(cfg["seq"]) and learner == "dqnv" and not cfg["mem_mode"]   # revision X2: GRU-style sequence replay
    if SEQ:
        assert cfg["m"] % cfg["L"] == 0, "seq replay needs m divisible by L"
    if learner == "tab":
        Q = torch.zeros(B, 2, S, NA, dtype=dtype, device=dev)
    else:
        H, M, m, C = cfg["H"], cfg["M"], cfg["m"], cfg["C"]
        xs = torch.stack([f(f"xs_{i}") if f"xs_{i}" in P else torch.zeros(B, dtype=dtype, device=dev)
                          for i in range(5)], 1)
        VR = int(cfg.get("view", 2))                     # view radius
        NV = (2 * VR + 1) ** 2
        D = (5 * NV + 2 + 2 + 5 + (2 + NA if RA else 0)) if VIEW else S + 2 + 2 + 5
        if RA:
            fprev = torch.zeros(B, 2, dtype=dtype, device=dev)
            aprev = torch.zeros(B, dtype=torch.long, device=dev)
            rb_fp = torch.zeros(B, M, 2, dtype=dtype, device=dev)
            rb_ap = torch.zeros(B, M, dtype=torch.long, device=dev)
        gi = torch.Generator(device=dev).manual_seed(int(seed) + 2_000_029)
        if GRU:
            D = 5 * NV + 2 + 2 + 2 + NA + 5
            Wx = torch.randn(B, D, 3 * H, generator=gi, device=dev, dtype=dtype) * (1.0 / np.sqrt(D))
            Wh = torch.randn(B, H, 3 * H, generator=gi, device=dev, dtype=dtype) * (1.0 / np.sqrt(H))
            bx = torch.zeros(B, 1, 3 * H, dtype=dtype, device=dev)
            bh = torch.zeros(B, 1, 3 * H, dtype=dtype, device=dev)
            W2 = torch.randn(B, H, 2 * NA, generator=gi, device=dev, dtype=dtype) * (0.1 / np.sqrt(H))
            b2 = (f("q_init") if "q_init" in P else torch.zeros(B, dtype=dtype, device=dev))[:, None, None].repeat(1, 1, 2 * NA)
            net = [Wx, Wh, bx, bh, W2, b2]
            h_roll = torch.zeros(B, H, dtype=dtype, device=dev)
            fprev = torch.zeros(B, 2, dtype=dtype, device=dev)
            aprev = torch.zeros(B, dtype=torch.long, device=dev)
            rb_h = torch.zeros(B, M, H, dtype=dtype, device=dev)
            rb_fp = torch.zeros(B, M, 2, dtype=dtype, device=dev)
            rb_ap = torch.zeros(B, M, dtype=torch.long, device=dev)
        else:
            W1 = torch.randn(B, D, H, generator=gi, device=dev, dtype=dtype) * (1.0 / np.sqrt(D))
            b1 = torch.zeros(B, 1, H, dtype=dtype, device=dev)
            W2 = torch.randn(B, H, 2 * NA, generator=gi, device=dev, dtype=dtype) * (0.1 / np.sqrt(H))
            b2 = (f("q_init") if "q_init" in P else torch.zeros(B, dtype=dtype, device=dev))[:, None, None].repeat(1, 1, 2 * NA)
            net = [W1, b1, W2, b2]
        if META:
            theta = torch.stack([f("m_a"), f("m_e"), f("m_s")], 1)
            mg_lr = f("mg_lr")
            mg_v = torch.zeros(B, dtype=dtype, device=dev)
        if VIEW:
            fs = torch.stack([f(f"fs_{i}") if f"fs_{i}" in P else torch.zeros(B, dtype=dtype, device=dev)
                              for i in range(5)], 1)
            yoke = torch.as_tensor(np.asarray(P["yoke"]), device=dev).long() if "yoke" in P else ar
            if not GRU:
                net += [torch.zeros(B, 5, H, dtype=dtype, device=dev), torch.zeros(B, 5, H, dtype=dtype, device=dev)]
            off = torch.arange(-VR, VR + 1, device=dev)
            d_r, d_c = off.repeat_interleave(2 * VR + 1), off.repeat(2 * VR + 1)    # NV offsets
            rb_l = torch.zeros(B, M, dtype=torch.long, device=dev)
            rb_fr = torch.zeros(B, M, NF, dtype=torch.bool, device=dev); rb_fr2 = torch.zeros_like(rb_fr)
            rb_xm = torch.zeros(B, M, 5, dtype=dtype, device=dev); rb_xm2 = torch.zeros_like(rb_xm)
            MEM = cfg["mem_mode"]
            if MEM:
                memw = torch.as_tensor(np.asarray(P["mem"]), device=dev).long() if "mem" in P \
                    else torch.full((B,), M, dtype=torch.long, device=dev)
                kscale = torch.stack([f(f"ks_{i}") if f"ks_{i}" in P else torch.zeros(B, dtype=dtype, device=dev)
                                      for i in range(5)] +
                                     [f(f"kv_{i}") if f"kv_{i}" in P else torch.zeros(B, dtype=dtype, device=dev)
                                      for i in range(4)], 1)                                     # (B, 9)
                rb_k = torch.zeros(B, M, 9, dtype=dtype, device=dev)
        # with C = 1 the target equals the online net at every step: alias it (identical results, no copies)
        alias = C == 1 and cfg["alias_target"]
        tgt = net if alias else [p.clone() for p in net]
        mom = [torch.zeros_like(p) for p in net]
        vel = [torch.zeros_like(p) for p in net]
        step_ct = 0
        rowcol = torch.stack([torch.arange(S, device=dev) // SIZE, torch.arange(S, device=dev) % SIZE], 1).to(dtype) / (SIZE - 1)
        # replay: cell, needs, signals before; action; felt (2); cell, needs, signals after; need weights after; alive
        rb_s = torch.zeros(B, M, dtype=torch.long, device=dev); rb_s2 = torch.zeros_like(rb_s)
        rb_a = torch.zeros_like(rb_s)
        rb_n = torch.zeros(B, M, 2, dtype=dtype, device=dev); rb_n2 = torch.zeros_like(rb_n)
        rb_x = torch.zeros(B, M, 5, dtype=dtype, device=dev); rb_x2 = torch.zeros_like(rb_x)
        rb_f = torch.zeros(B, M, 2, dtype=dtype, device=dev); rb_w2 = torch.zeros_like(rb_f)
        rb_al = torch.zeros(B, M, dtype=dtype, device=dev)

        def feats(cell, nn_, xx):
            """cell (B, k), nn_ (B, k, 2), xx (B, k, 5) -> (B, k, D)"""
            oh = torch.nn.functional.one_hot(cell, S).to(dtype)
            return torch.cat([oh, rowcol[cell], nn_ * cfg["needs_in"], (xx.clamp(0, 3) / 3.0) * xs[:, None, :]], -1)

        def view(cell, layv, frv):
            """egocentric 5x5 view: cell (B, k), layv (B, k), frv (B, k, 2) food ready -> (B, k, 125)"""
            rr = cell[..., None] // SIZE + d_r
            cc = cell[..., None] % SIZE + d_c
            oob = (rr < 0) | (rr >= SIZE) | (cc < 0) | (cc >= SIZE)
            idx = rr.clamp(0, SIZE - 1) * SIZE + cc.clamp(0, SIZE - 1)           # (B, k, NV)
            fd, sh = bf[layv], bs[layv]                                          # (B, k, NF), (B, k)
            on = idx[..., None] == fd[..., None, :]                              # (B, k, V, NF)
            fr = frv[..., None, :]
            inside = (~oob).to(dtype)
            ready = (on & fr).any(-1).to(dtype) * inside
            grow = (on & ~fr).any(-1).to(dtype) * inside
            shel = (idx == sh[..., None]).to(dtype) * inside
            tm = torch.gather(T_map, 1, idx.view(B, -1)).view(idx.shape) * inside
            return torch.cat([ready, grow, shel, oob.to(dtype), tm], -1)

        def featsv(cell, nn_, xx, layv, frv, fp=None, ap=None):
            parts = [view(cell, layv, frv), rowcol[cell], nn_ * cfg["needs_in"], (xx.clamp(0, 3) / 3.0) * xs[:, None, :]]
            if RA:          # same encoding as the GRU input: felt reward (2), action one-hot (5)
                parts += [fp, torch.nn.functional.one_hot(ap, NA).to(dtype)]
            return torch.cat(parts, -1)

        def fwdv(params, X, XM):
            """FiLM-modulated forward: returns out, pre, pre2, gain, th, xf, h"""
            w1, c1, w2, c2, Gm, Fm = params
            pre = torch.baddbmm(c1, X, w1)
            xf = (XM.clamp(0, 3) / 3.0) * fs[:, None, :]
            th = torch.tanh(torch.bmm(xf, Gm))
            gain = 1 + th
            pre2 = pre * gain + torch.bmm(xf, Fm)
            h = pre2.clamp_min(0)
            return torch.baddbmm(c2, h, w2), pre, pre2, gain, th, xf, h

        def featg(cell, nn_, xx, layv, frv, fp, ap):
            """GRU input: view, coordinates, needs, last felt reward, last action (one-hot), scaled signals"""
            return torch.cat([view(cell, layv, frv), rowcol[cell], nn_ * cfg["needs_in"], fp,
                              torch.nn.functional.one_hot(ap, NA).to(dtype), (xx.clamp(0, 3) / 3.0) * xs[:, None, :]], -1)

        def gru_step(params, X, hh):
            """X (B, k, D), hh (B, k, H) -> new hidden (B, k, H); PyTorch GRU gate convention"""
            wx, wh, cx, ch = params[:4]
            gx = torch.baddbmm(cx, X, wx)
            gh = torch.baddbmm(ch, hh, wh)
            r_ = torch.sigmoid(gx[..., :H] + gh[..., :H])
            z_ = torch.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
            n_ = torch.tanh(gx[..., 2 * H:] + r_ * gh[..., 2 * H:])
            return (1 - z_) * n_ + z_ * hh

        def head(params, hh):
            return torch.baddbmm(params[5], hh, params[4])

        def fwd(params, X):
            w1, c1, w2, c2 = params
            pre = torch.baddbmm(c1, X, w1)
            h = pre.clamp_min(0)
            return torch.baddbmm(c2, h, w2), pre, h          # (B, k, 10)

    alive = torch.ones(B, dtype=torch.bool, device=dev)
    life = torch.full((B,), T, dtype=torch.long, device=dev)
    zero_e = torch.zeros(B, dtype=torch.long, device=dev)
    score = torch.zeros(B, dtype=torch.float64, device=dev)
    traces = {k: torch.zeros(T, B, dtype=torch.float32, device=dev) for k in record}
    one = torch.ones((), dtype=dtype, device=dev)

    for t in range(T):
        reg = REG[t]
        rp = RT[reg]
        p_zone, p_rand, regrow, sw_every = rp[:, 0], rp[:, 1], rp[:, 2].long(), rp[:, 3].long()
        if t > 0:
            sw = (sw_every > 0) & ((t % sw_every.clamp_min(1)) == 0)
            if bool(sw.any()):
                new = torch.randint(L, (B,), generator=gw, device=dev)
                lay = torch.where(sw, new, lay)
                food = torch.where(sw[:, None], bf[lay], food)
                shelter = torch.where(sw, bs[lay], shelter)
                threat = torch.where(sw[:, None], bt[lay], threat)
                timers = torch.where(sw[:, None], torch.zeros_like(timers), timers)

        # ---------------- knobs (global layer: generic weights + constant offsets, leaky memory tau 50)
        eff = theta if learner != "tab" and META else lin
        a_t = torch.exp(torch.log(alpha0) + eff[:, 0])
        s_t = s0 * torch.exp(eff[:, 2])
        a_t = torch.minimum(a_t, one * 0.9) if learner == "tab" else torch.minimum(a_t, one * cfg["lr_max"])

        # ---------------- act
        s = pos
        w = 2.0 * (1.0 - n).clamp_min(0) + 0.05
        w = torch.stack([w[:, 0], w[:, 1] * s_t], 1)
        if learner == "tab":
            Qs = Q[ar, :, s, :]                                                  # (B, 2, 5)
        elif learner == "dqn":
            out, _, _ = fwd(net, feats(s[:, None], n[:, None, :], x[:, None, :]))
            Qs = out[:, 0, :].view(B, 2, NA)
        elif GRU:
            fr_now = timers == 0
            xm_old = x
            Xa = featg(s[:, None], n[:, None, :], x[:, None, :], lay[:, None], fr_now[:, None, :], fprev[:, None, :],
                       aprev[:, None])
            h_new = gru_step(net, Xa, h_roll[:, None, :])
            Qs = head(net, h_new)[:, 0, :].view(B, 2, NA)
        else:
            fr_now = timers == 0
            xm_old = x[yoke]
            ra_now = (fprev[:, None, :], aprev[:, None]) if RA else ()
            out = fwdv(net, featsv(s[:, None], n[:, None, :], x[:, None, :], lay[:, None], fr_now[:, None, :], *ra_now),
                       xm_old[:, None, :])[0]
            Qs = out[:, 0, :].view(B, 2, NA)
        q = torch.einsum("bk,bka->ba", w, Qs)
        eps_now = torch.sigmoid(_logit(eps_tab[ar, s]) + eff[:, 1]).clamp(0, 0.5)
        greedy = torch.argmax(q + 1e-12 * rnd_a(B, NA), 1)
        a = torch.where(rnd_a(B) < eps_now, torch.randint(NA, (B,), generator=ga, device=dev), greedy)

        # ---------------- environment
        r0 = (s // SIZE + moves[a, 0]).clamp(0, SIZE - 1)
        c0 = (s % SIZE + moves[a, 1]).clamp(0, SIZE - 1)
        s2 = r0 * SIZE + c0
        ate_slot = (food == s2[:, None]) & (timers == 0)
        ate = ate_slot.any(1)
        timers = torch.where(ate_slot, regrow[:, None].expand(-1, NF), (timers - 1).clamp_min(0))
        u1, u2 = rnd_w(B), rnd_w(B)
        dmg = (threat[ar, s2] & (u1 < p_zone)) | (u2 < p_rand)
        in_sh = s2 == shelter
        moved = (a != 0).to(dtype)

        # ---------------- needs
        old = n
        raw = n.clone()
        raw[:, 0] += -DECAY_ - NEED["move_cost"] * moved + NEED["food_gain"] * ate.to(dtype)
        raw[:, 1] += NEED["safety_recovery"] + NEED["shelter_recovery"] * in_sh.to(dtype) - loss * dmg.to(dtype)
        felt = raw.clamp_max(1.0) - old
        n = torch.where(alive[:, None], raw.clamp(0.0, 1.0), n)

        # ---------------- appraisal
        d = dmg.to(dtype)
        expected = T_map[ar, s2]
        safe_hit = d * (expected < 0.05).to(dtype)
        T_map[ar, s2] += 0.1 * (d - expected)
        n_all = (1 - 1 / 300) * n_all + d
        n_safe = (1 - 1 / 300) * n_safe + safe_hit
        K = 1 - (n_safe + 1) / (n_all + 2)
        threat_now = expected + d

        # ---------------- learning
        w2 = 2.0 * (1.0 - n).clamp_min(0) + 0.05
        w2 = torch.stack([w2[:, 0], w2[:, 1] * s_t], 1)
        x_old = x
        if learner == "tab":
            Qs2 = Q[ar, :, s2, :]
            a2 = torch.argmax(torch.einsum("bk,bka->ba", w2, Qs2), 1)
            delta = felt + gamma[:, None] * Qs2[ar, :, a2] - Qs[ar, :, a]
            Q[ar, :, s, a] += (a_t * alive)[:, None] * delta
            absd = (w2 * delta).sum(1).abs()
            vx = (a_t * (w2 * delta).sum(1)).abs() / vsig
        else:
            # signals after the step are computed below; the transition stores the signals the agent saw when acting
            pass

        # ---------------- controller state (appraisal signals; the generic layer)
        if learner == "tab":
            excess = (absd - mean_abs).clamp_min(0)
            mean_abs = mean_abs + (absd - mean_abs) / 200.0
        thr_avg = thr_avg + (threat_now - thr_avg) / 50.0
        intake = intake + (ate.to(dtype) - intake) / 100.0
        if learner == "dqnv":
            fr_after = timers == 0
            ra_next = (felt[:, None, :], a[:, None]) if RA else ()
            out2 = fwdv(net, featsv(s2[:, None], n[:, None, :], x_old[:, None, :], lay[:, None], fr_after[:, None, :],
                                    *ra_next), xm_old[:, None, :])[0]
            Qs2 = out2[:, 0, :].view(B, 2, NA)
            a2 = torch.argmax(torch.einsum("bk,bka->ba", w2, Qs2), 1)
            delta = felt + gamma[:, None] * Qs2[ar, :, a2] - Qs[ar, :, a]
            absd = (w2 * delta).sum(1).abs()
            vx = absd / vsig
            excess = (absd - mean_abs).clamp_min(0)
            mean_abs = mean_abs + (absd - mean_abs) / 200.0
        if GRU:
            fr_after = timers == 0
            with torch.no_grad():
                X2a = featg(s2[:, None], n[:, None, :], x_old[:, None, :], lay[:, None], fr_after[:, None, :],
                            felt[:, None, :], a[:, None])
                Qs2 = head(net, gru_step(net, X2a, h_new))[:, 0, :].view(B, 2, NA)
            a2 = torch.argmax(torch.einsum("bk,bka->ba", w2, Qs2), 1)
            delta = felt + gamma[:, None] * Qs2[ar, :, a2] - Qs[ar, :, a]
            absd = (w2 * delta).sum(1).abs()
            vx = absd / vsig
            excess = (absd - mean_abs).clamp_min(0)
            mean_abs = mean_abs + (absd - mean_abs) / 200.0
        if learner == "dqn":
            # current-transition TD error with the online net (no gradient): drives VDBE and surprise as in the table
            out2, _, _ = fwd(net, feats(s2[:, None], n[:, None, :], x_old[:, None, :]))
            Qs2 = out2[:, 0, :].view(B, 2, NA)
            a2 = torch.argmax(torch.einsum("bk,bka->ba", w2, Qs2), 1)
            if alias:
                Qt2 = Qs2
            else:
                outt, _, _ = fwd(tgt, feats(s2[:, None], n[:, None, :], x_old[:, None, :]))
                Qt2 = outt[:, 0, :].view(B, 2, NA)
            delta = felt + gamma[:, None] * Qt2[ar, :, a2] - Qs[ar, :, a]
            absd = (w2 * delta).sum(1).abs()
            vx = absd / vsig
            excess = (absd - mean_abs).clamp_min(0)
            mean_abs = mean_abs + (absd - mean_abs) / 200.0
        sur = (sur - sur / 8.0 + 1.5 * excess).clamp(0, 3)
        x = torch.stack([thr_avg / 0.05, K, sur / 0.1, intake / 0.02, 1 - n[:, 0]], 1)
        xg = x[yoke] if learner == "dqnv" else x                                # yoked signals drive the layers
        target = (torch.einsum("bki,bi->bk", Wg, xg) + bias).clamp(-6, 6)
        lin = lin + (target - lin) / 50.0

        if GRU:
            j = t % M
            rb_s[:, j], rb_n[:, j], rb_x[:, j], rb_a[:, j] = s, old, x_old, a
            rb_f[:, j], rb_s2[:, j], rb_n2[:, j], rb_x2[:, j] = felt, s2, n, x
            rb_w2[:, j], rb_al[:, j] = w2, alive.to(dtype)
            rb_l[:, j], rb_fr[:, j], rb_fr2[:, j] = lay, fr_now, fr_after
            rb_h[:, j], rb_fp[:, j], rb_ap[:, j] = h_roll, fprev, aprev
            h_roll, fprev, aprev = h_new[:, 0, :].detach(), felt, a
            Ls = cfg["L"]
            ns = max(m // Ls, 1)
            if t >= Ls - 1:
                nend = min(t + 1, M) - Ls + 1
                age = torch.randint(nend, (B, ns), generator=ga, device=dev)
                age[:, 0] = 0                                                     # the newest sequence
                idx = ((j - age)[..., None] - torch.arange(Ls - 1, -1, -1, device=dev)) % M       # (B, ns, L)
                idx = idx.reshape(B, ns * Ls)
                g = lambda buf: torch.gather(buf, 1, idx) if buf.dim() == 2 else \
                    torch.gather(buf, 1, idx[..., None].expand(-1, -1, buf.shape[-1]))
                sq = lambda v: v.view(B, ns, Ls, *v.shape[2:])
                lb = g(rb_l)
                X = sq(featg(g(rb_s), g(rb_n), g(rb_x), lb, g(rb_fr), g(rb_fp), g(rb_ap)))
                X2 = sq(featg(g(rb_s2), g(rb_n2), g(rb_x2), lb, g(rb_fr2), g(rb_f), g(rb_a)))
                ww2, aa, ff, al_ = sq(g(rb_w2)), sq(g(rb_a)), sq(g(rb_f)), sq(g(rb_al))
                hh = sq(g(rb_h))[:, :, 0, :]
                Pg = [p_.detach().requires_grad_(True) for p_ in net]
                td_loss = 0.0
                for k_ in range(Ls):
                    hh = gru_step(Pg, X[:, :, k_, :], hh)
                    q_on = head(Pg, hh).view(B, ns, 2, NA)
                    with torch.no_grad():
                        q2 = head(net, gru_step(net, X2[:, :, k_, :], hh.detach())).view(B, ns, 2, NA)
                        ab = torch.argmax((ww2[:, :, k_, :, None] * q2).sum(2), -1)
                        q_next = torch.gather(q2, 3, ab[:, :, None, None].expand(-1, -1, 2, 1))[..., 0]
                    q_sa = torch.gather(q_on, 3, aa[:, :, k_, None, None].expand(-1, -1, 2, 1))[..., 0]
                    dl = (ff[:, :, k_] + gamma[:, None, None] * q_next - q_sa) * al_[:, :, k_, None]
                    td_loss = td_loss + 0.5 * (dl ** 2).sum() / (ns * Ls)
                grads = torch.autograd.grad(td_loss, Pg)
                step_ct += 1
                b1c, b2c = 1 - cfg["adam_b1"] ** step_ct, 1 - cfg["adam_b2"] ** step_ct
                lr_b = (a_t * alive)[:, None, None]
                with torch.no_grad():
                    for p_, g_, m_, v_ in zip(net, grads, mom, vel):
                        m_.mul_(cfg["adam_b1"]).add_(g_, alpha=1 - cfg["adam_b1"])
                        v_.mul_(cfg["adam_b2"]).addcmul_(g_, g_, value=1 - cfg["adam_b2"])
                        p_.sub_(lr_b * (m_ / b1c) / ((v_ / b2c).sqrt() + 1e-8))
        if learner in ("dqn", "dqnv"):
            j = t % M
            rb_s[:, j], rb_n[:, j], rb_x[:, j], rb_a[:, j] = s, old, x_old, a
            rb_f[:, j], rb_s2[:, j], rb_n2[:, j], rb_x2[:, j] = felt, s2, n, x
            rb_w2[:, j], rb_al[:, j] = w2, alive.to(dtype)
            if VIEW:
                rb_l[:, j], rb_fr[:, j], rb_fr2[:, j] = lay, fr_now, fr_after
                rb_xm[:, j], rb_xm2[:, j] = xm_old, x[yoke]
            if RA:
                rb_fp[:, j], rb_ap[:, j] = fprev, aprev
                fprev, aprev = felt, a
            if VIEW and MEM:
                key_now = torch.cat([xm_old.clamp(0, 3) / 3.0, rowcol[s], old], 1)          # key of this step
                rb_k[:, j] = key_now
                win = torch.minimum(memw, torch.full_like(memw, t + 1)).to(dtype)        # (B,)
                age = (rnd_a(B, cfg["K"]) * win[:, None]).long()                          # (B, K) in [0, win)
                cand = (j - age) % M
                kc = torch.gather(rb_k, 1, cand[..., None].expand(-1, -1, 9))             # (B, K, 9)
                logit = -(((kc - key_now[:, None, :]) * kscale[:, None, :]) ** 2).sum(-1)
                gum = -torch.log(-torch.log(rnd_a(B, cfg["K"]).clamp(1e-12, 1 - 1e-7)))
                pick = torch.topk(logit + gum, m - 1, dim=1).indices
                idx = torch.cat([torch.full((B, 1), j, dtype=torch.long, device=dev), torch.gather(cand, 1, pick)], 1)
            elif SEQ:
                # revision X2: the GRU's replay sampling for the feed-forward net: m // L contiguous L-step sequences,
                # start ages uniform as in the GRU branch, the newest sequence always included. Slots not yet
                # written have alive weight 0 in rb_al, so they add no gradient (the GRU skips steps t < L - 1).
                Ls = cfg["L"]
                ns = m // Ls
                nend = max(min(t + 1, M) - Ls + 1, 1)
                age = torch.randint(nend, (B, ns), generator=ga, device=dev)
                age[:, 0] = 0
                idx = (((j - age)[..., None] - torch.arange(Ls - 1, -1, -1, device=dev)) % M).reshape(B, ns * Ls)
            else:
                hi = min(t + 1, M)
                idx = torch.randint(hi, (B, m), generator=ga, device=dev)
                idx[:, 0] = j                                                     # always learn from the newest
            g = lambda buf: torch.gather(buf, 1, idx) if buf.dim() == 2 else \
                torch.gather(buf, 1, idx[..., None].expand(-1, -1, buf.shape[-1]))
            ww2 = g(rb_w2)
            if VIEW:
                lb = g(rb_l)
                X = featsv(g(rb_s), g(rb_n), g(rb_x), lb, g(rb_fr), *((g(rb_fp), g(rb_ap)) if RA else ()))
                X2 = featsv(g(rb_s2), g(rb_n2), g(rb_x2), lb, g(rb_fr2), *((g(rb_f), g(rb_a)) if RA else ()))
                o_on, pre, pre2, gain, th, xf, h = fwdv(net, X, g(rb_xm))
                with torch.no_grad():
                    o2 = fwdv(net, X2, g(rb_xm2))[0]
                    ot = o2
            else:
                X = feats(g(rb_s), g(rb_n), g(rb_x))
                X2 = feats(g(rb_s2), g(rb_n2), g(rb_x2))
                o_on, pre, h = fwd(net, X)
                with torch.no_grad():
                    o2, _, _ = fwd(net, X2)
                    ot = o2 if alias else fwd(tgt, X2)[0]
            o_on3, o2, ot = o_on.view(B, m, 2, NA), o2.view(B, m, 2, NA), ot.view(B, m, 2, NA)
            ab = torch.argmax((ww2[..., None] * o2).sum(2), -1)                 # (B, m)
            q_next = torch.gather(ot, 3, ab[:, :, None, None].expand(-1, -1, 2, 1))[..., 0]
            aa = g(rb_a)
            q_sa = torch.gather(o_on3, 3, aa[:, :, None, None].expand(-1, -1, 2, 1))[..., 0]
            dl = (g(rb_f) + gamma[:, None, None] * q_next - q_sa) * g(rb_al)[..., None]      # (B, m, 2)
            gout = torch.zeros(B, m, 2, NA, dtype=dtype, device=dev)
            gout.scatter_(3, aa[:, :, None, None].expand(-1, -1, 2, 1), -dl[..., None])
            gout = gout.view(B, m, 2 * NA) / m
            gW2 = torch.bmm(h.transpose(1, 2), gout)
            gb2 = gout.sum(1, keepdim=True)
            if VIEW:
                gpre2 = torch.bmm(gout, net[2].transpose(1, 2)) * (pre2 > 0).to(dtype)
                gpre = gpre2 * gain
                gG = torch.bmm(xf.transpose(1, 2), gpre2 * pre * (1 - th * th))
                gF = torch.bmm(xf.transpose(1, 2), gpre2)
            else:
                gpre = torch.bmm(gout, net[2].transpose(1, 2)) * (pre > 0).to(dtype)
            gW1 = torch.bmm(X.transpose(1, 2), gpre)
            gb1 = gpre.sum(1, keepdim=True)
            grads = (gW1, gb1, gW2, gb2) + ((gG, gF) if VIEW else ())
            step_ct += 1
            b1c, b2c = 1 - cfg["adam_b1"] ** step_ct, 1 - cfg["adam_b2"] ** step_ct
            lr_b = (a_t * alive)[:, None, None]
            dirs = []
            for p_, g_, m_, v_ in zip(net, grads, mom, vel):
                m_.mul_(cfg["adam_b1"]).add_(g_, alpha=1 - cfg["adam_b1"])
                v_.mul_(cfg["adam_b2"]).addcmul_(g_, g_, value=1 - cfg["adam_b2"])
                if META:
                    dr = (m_ / b1c) / ((v_ / b2c).sqrt() + 1e-8)
                    p_.sub_(lr_b * dr)
                    dirs.append(dr)
                else:
                    p_.sub_(lr_b * (m_ / b1c) / ((v_ / b2c).sqrt() + 1e-8))
            if META:
                # one-step meta-gradient of the validation TD loss w.r.t. theta_a (log learning-rate offset)
                iv = torch.randint(min(t + 1, M), (B, m), generator=ga, device=dev)
                gv = lambda buf: torch.gather(buf, 1, iv) if buf.dim() == 2 else \
                    torch.gather(buf, 1, iv[..., None].expand(-1, -1, buf.shape[-1]))
                lv = gv(rb_l)
                Xv = featsv(gv(rb_s), gv(rb_n), gv(rb_x), lv, gv(rb_fr))
                X2v = featsv(gv(rb_s2), gv(rb_n2), gv(rb_x2), lv, gv(rb_fr2))
                Pm = [p_.detach().requires_grad_(True) for p_ in net]
                qv = fwdv(Pm, Xv, gv(rb_xm))[0].view(B, m, 2, NA)
                with torch.no_grad():
                    q2v = fwdv(net, X2v, gv(rb_xm2))[0].view(B, m, 2, NA)
                    abv = torch.argmax((gv(rb_w2)[..., None] * q2v).sum(2), -1)
                    qn = torch.gather(q2v, 3, abv[:, :, None, None].expand(-1, -1, 2, 1))[..., 0]
                av = gv(rb_a)
                qsa = torch.gather(qv, 3, av[:, :, None, None].expand(-1, -1, 2, 1))[..., 0]
                dlv = (gv(rb_f) + gamma[:, None, None] * qn - qsa) * gv(rb_al)[..., None]
                gm = torch.autograd.grad(0.5 * (dlv ** 2).sum() / m, Pm)
                with torch.no_grad():
                    gA = -sum((gp * dr).flatten(1).sum(1) for gp, dr in zip(gm, dirs)) * a_t * alive
                    gA = gA * (a_t < cfg["lr_max"]).to(dtype)
                    mg_v = 0.99 * mg_v + 0.01 * gA * gA
                    upd = mg_lr * gA / (mg_v.sqrt() + 1e-12) * (mg_v > 0).to(dtype)
                    theta = torch.cat([(theta[:, :1] - upd[:, None]).clamp(-6, 6), theta[:, 1:]], 1)
            if not alias and (t + 1) % C == 0:
                for p_, q_ in zip(net, tgt):
                    q_.copy_(p_)

        # ---------------- VDBE (per cell)
        fv = (1 - torch.exp(-vx)) / (1 + torch.exp(-vx))
        eps_tab[ar, s] = 0.2 * fv + 0.8 * eps_tab[ar, s]

        wellbeing = 1.0 - ((1 - n) ** 2).sum(1) / 2.0
        pos = torch.where(alive, s2, pos)
        score += (wellbeing * alive).double()
        for k in record:
            traces[k][t] = {"alive": alive.float(), "wellbeing": (wellbeing * alive).float(), "alpha": a_t.float(),
                            "eps": eps_now.float(), "safety_mult": s_t.float(),
                            **{f"x{i}": x_old[:, i].float() for i in range(5)},
                            "ea": eff[:, 0].float(), "ee": eff[:, 1].float(), "es": eff[:, 2].float()}[k]
        zero_e = torch.where(n[:, 0] <= 0, zero_e + 1, torch.zeros_like(zero_e))
        died = alive & ((zero_e >= grace) | (n[:, 1] <= 0)) & (t >= juvenile)
        life = torch.where(died, torch.full_like(life, t + 1), life)
        alive = alive & ~died
        if hook is not None:
            hook(t, locals())

    out = dict(score=(score / T).cpu().numpy(), survival=(life.double() / T).cpu().numpy())
    out.update({k: v.cpu().numpy() for k, v in traces.items()})
    return out


def params(B: int, **kw) -> dict:
    d = dict(alpha0=0.3, eps0=0.1, s0=1.0, gamma=0.9, vdbe_sigma=0.1, b_a=0.0, b_e=0.0, b_s=0.0, q_init=0.0,
             **{f"xs_{i}": 0.0 for i in range(5)}, **{f"fs_{i}": 0.0 for i in range(5)},
             **{f"ks_{i}": 0.0 for i in range(5)}, **{f"kv_{i}": 0.0 for i in range(4)}, mem=1000.0,
             m_a=0.0, m_e=0.0, m_s=0.0, mg_lr=0.0,
             **{f"w_{i}_{k}": 0.0 for i in range(5) for k in "aes"})
    d.update(kw)
    return {k: np.broadcast_to(np.asarray(v, float), (B,)).copy() for k, v in d.items()}
