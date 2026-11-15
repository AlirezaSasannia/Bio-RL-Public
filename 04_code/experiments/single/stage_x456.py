"""Revision v1.5 (family single_revision_v1_5, see revision_v15.py). Evaluation only, frozen solutions.
X4  truly switch-free lives (one regime from step 0) for the B10 arms control, bias, generic.
X5  mechanism: cause of death, time at zero energy, death rules G = 50 / 200, knob clamps, signal correlations.
X6  one common test set (4096 schedules) for all frozen eight-base tabular solutions; GPU subset for the networks.
Also (descriptive, used sets, deterministic re-computation with their own seeds; scores must equal the logs):
X1 lives (gain after step 1000) and the B10 test set (per-agent lives and causes -> Kaplan-Meier, RMST, hazards).
Run: python stage_x456.py [run|recompute|x6tab|x6gpu|report] [--workers N]
Smoke (dev seeds, short lives, scratch output, nothing registered): set V15_SMOKE=<scratch dir>."""
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pool import NWORK
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank
from reanalysis_runlevel import ci3, holm
from revision_v15 import SPEC

LOGS = Path(__file__).resolve().parents[2] / "results" / "logs" / "single"
B10D, B11D = LOGS / "stageB10_opt", LOGS / "stageB11_opt"
OUT = LOGS / "revV15"
CTRL = {"base": "vdbe", "control": "vdbe", "bias": "vdbe_generic", "generic": "vdbe_generic", "yoked": "vdbe_generic",
        "augtab": "vdbe_augtab", "auglin": "vdbe_auglin", "bio": "vdbe_fuzzy", "generic_split": "vdbe_generic"}
TAB_ARMS = ["augtab", "generic", "generic_split", "yoked", "bio", "bias", "auglin", "control", "base"]  # slowest first
NB, NA_, NS = 8, 4, 128
T, JUV, BIN = R.T_LIFE, R.JUVENILE, 500
NR = len(R.NAMES)
KN = ("a", "e", "s")
GRACE = (100, 50, 200)
X4 = dict(bank=5_400_000, sim=340_000)
X5 = dict(sched=56_000, bank=5_500_000, sim=350_000)
X6 = dict(sched=65_000, bank=5_600_000, sim=360_000, tsim=370_000, ns=4096, ch=256, ngpu=1024)
X1 = dict(bank=5_200_000, sim=320_000, sched=53_000)          # used set (X1), re-computation only
B10T = dict(sched=20_000, bank=2_000_000, sim=86_000)          # used set (B10 test), re-computation only
SIGNALS = ["threat", "K", "surprise", "intake", "deficit"]
SMOKE = os.environ.get("V15_SMOKE", "").strip()
if SMOKE:      # code check only: dev seeds, short lives, scratch output
    OUT, NS, T = Path(SMOKE), 6, 1500
    X4 = dict(bank=1_970_000, sim=135_000)
    X5 = dict(sched=97_500, bank=1_970_000, sim=136_000)
    X6 = dict(sched=97_600, bank=1_970_000, sim=137_000, tsim=138_000, ns=16, ch=8, ngpu=8)
    X1 = dict(bank=1_970_000, sim=139_000, sched=97_700)
    B10T = dict(sched=97_800, bank=1_970_000, sim=139_500)
    register = lambda spec: "smoke-no-registration"
_banks = {}


def bank(lo):
    if lo not in _banks:
        _banks[lo] = LayoutBank.build(range(lo, lo + 1000))
    return _banks[lo]


def params_of(arm, k):
    d = B11D if arm == "generic_split" else B10D
    return json.loads((d / f"{arm}_r{k}.json").read_text())["params"]


def layouts(ns, i0=0, agents=NA_):
    return np.concatenate([(np.arange(agents) * 37 + i * 11) % 1000 for i in range(i0, i0 + ns)])


def sched_matrix(seed0, ns, i0=0, agents=NA_):
    sch = [R.schedule(np.random.default_rng(seed0 + i)) for i in range(i0, i0 + ns)]
    return A.regime_matrix([s for s in sch for _ in range(agents)], R.T_LIFE)[:T]


def lives(kind, k, c, v):
    """regime matrix, layouts, bank, simulator seed for one job"""
    lay = layouts(NS)
    if kind == "x4":          # whole life in regime c, no safe prelude (juvenile death protection as usual)
        return np.full((T, NS * NA_), c, int), lay, bank(X4["bank"]), X4["sim"] + 10 * k + c
    if kind == "x5":          # new switching lives
        return sched_matrix(X5["sched"], NS), lay, bank(X5["bank"]), X5["sim"] + 10 * k + v
    if kind == "x1":          # X1 lives exactly as in stage_x1.py (used set)
        if c < NR:
            M = np.full((T, NS * NA_), c, int)
            M[:JUV] = R.NAMES.index("safe")
        else:
            M = sched_matrix(X1["sched"], NS)
        return M, lay, bank(X1["bank"]), X1["sim"] + 10 * k + c
    if kind == "b10":         # B10 test set exactly as in stage_b10.py (used set)
        return sched_matrix(B10T["sched"], NS), lay, bank(B10T["bank"]), B10T["sim"] + k
    raise ValueError(kind)


class Rec:
    """Read-only hook (draws no random numbers, changes no state): per agent and regime, alive steps, steps at zero
    energy, deaths by cause; longest zero-energy run; wellbeing in 500-step bins; sum of the realised knob offsets;
    optionally sums for the correlation of the five appraisal inputs per regime (over alive agent-steps)."""

    def __init__(self, B, sig):
        self.alive, self.zero, self.ds, self.di = (np.zeros((B, NR), int) for _ in range(4))
        self.maxzero = np.zeros(B, int)
        self.cause = np.zeros(B, int)            # 0 alive at the end, 1 starvation, 2 injury, 3 both in the same step
        self.bins = np.zeros((B, T // BIN))
        self.eff = np.zeros((B, 3))
        self.sig = sig
        self.S1, self.S2, self.N = np.zeros((NR, 5)), np.zeros((NR, 5, 5)), np.zeros(NR)

    def __call__(self, t, L):
        died = L["died"]
        ap = L["alive"] | died                   # alive during this step
        reg, n = L["reg"], L["n"]
        oh = (reg[:, None] == np.arange(NR)[None, :]) & ap[:, None]
        self.alive += oh
        self.zero += oh & (n[:, 0] <= 0)[:, None]
        if died.any():
            st, inj = died & (L["zero_e"] >= L["grace"]), died & (n[:, 1] <= 0)
            self.cause[died] = st[died].astype(int) + 2 * inj[died].astype(int)
            self.ds += oh & st[:, None]
            self.di += oh & inj[:, None]
        self.maxzero = np.maximum(self.maxzero, np.where(ap, L["zero_e"], 0))
        self.bins[:, t // BIN] += ap * L["wellbeing"]
        self.eff += ap[:, None] * L["eff_rec"]
        if self.sig:
            x = L["xin"][:, :5]
            for r in np.unique(reg[ap]):
                m = ap & (reg == r)
                X = x[m]
                self.S1[r] += X.sum(0)
                self.S2[r] += X.T @ X
                self.N[r] += m.sum()


def run_job(job):
    """job = (kind, arm, k, c, v, clamp_j, clamp_val); arm 'generic' with clamp_j >= 0 = knob clamp"""
    kind, arm, k, c, v, cj, cv = job
    prm = dict(params_of(arm, k))
    if cj >= 0:
        kn = KN[cj]
        prm.update({f"w_{i}_{kn}": 0.0 for i in range(5)}, **{f"b_{kn}": float(cv)})
    M, lay, bk, seed = lives(kind, k, c, v)
    B = M.shape[1]
    rec = Rec(B, sig=CTRL[arm] == "vdbe_generic" and kind in ("x4", "x5", "b10") and cj < 0)
    o = A.simulate(CTRL[arm], A.params(B, **prm), M, bk, lay, seed=seed, grace=GRACE[v], hook=rec)
    df = pd.DataFrame(dict(agent=np.arange(B), group=np.arange(B) // NA_, score=o["score"],
                           life=np.round(o["survival"] * T).astype(int), cause=rec.cause, maxzero=rec.maxzero,
                           alive_total=rec.alive.sum(1)))
    for j, kn in enumerate(KN):
        df[f"effsum_{kn}"] = rec.eff[:, j]
    for b in range(rec.bins.shape[1]):
        df[f"wb_{b}"] = rec.bins[:, b].round(4)
    for j, rn in enumerate(R.NAMES):
        df[f"alive_{rn}"], df[f"zero_{rn}"] = rec.alive[:, j], rec.zero[:, j]
        df[f"dstarv_{rn}"], df[f"dinj_{rn}"] = rec.ds[:, j], rec.di[:, j]
    df.insert(0, "clamp", cj)
    df.insert(0, "grace", GRACE[v])
    df.insert(0, "cond", c)
    df.insert(0, "base", k)
    df.insert(0, "arm", arm)
    df.insert(0, "kind", kind)
    sig = dict(S1=rec.S1, S2=rec.S2, N=rec.N) if rec.sig else None
    return df, sig, job


def _map(jobs, name):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sigs, t0 = [], {}, time.time()
    with Pool(NWORK) as p:
        for i, (df, sig, job) in enumerate(p.imap_unordered(run_job, jobs)):
            rows.append(df)
            if sig is not None:
                sigs[job] = sig
            if i % 20 == 0 or i == len(jobs) - 1:
                print(f"{name} {i + 1}/{len(jobs)} {job[:5]} score {df.score.mean():.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
    D = pd.concat(rows).sort_values(["kind", "arm", "base", "cond", "grace", "clamp", "agent"])
    D.to_csv(OUT / f"{name}_agents.csv.gz", index=False)
    if sigs:
        (OUT / f"{name}_signals.pkl").write_bytes(pickle.dumps(sigs))
    return D


def run():
    """X4 and X5 (registered): pass 1 (all arms, three death rules), then the knob clamps."""
    print("registered", register(SPEC)[:12], flush=True)
    jobs = [("x4", a, k, c, v, -1, 0.0) for v in range(3) for c in range(NR) for k in range(NB)
            for a in ("control", "bias", "generic")]
    jobs += [("x5", a, k, 5, v, -1, 0.0) for v in range(3) for k in range(NB) for a in ("control", "bias", "generic")]
    jobs += [("x5", "bio", k, 5, 0, -1, 0.0) for k in range(NB)]
    D = _map(jobs, "x45")
    # clamp values: mean realised offset of each knob over alive agent-steps of unclamped generic, per base and lives
    ref = {"x4": (D.kind == "x4") & (D.cond == R.NAMES.index("scarce")), "x5": D.kind == "x5"}
    rows, cj = [], []
    for kind, m in ref.items():
        g = D[m & (D.arm == "generic") & (D.grace == 100)].groupby("base")
        for k, sub in g:
            for j, kn in enumerate(KN):
                val = float(sub[f"effsum_{kn}"].sum() / sub.alive_total.sum())
                rows.append(dict(kind=kind, base=k, knob=kn, value=val))
                cj.append((kind, "generic", int(k), R.NAMES.index("scarce") if kind == "x4" else 5, 0, j, val))
    pd.DataFrame(rows).to_csv(OUT / "clamp_values.csv", index=False)
    _map(cj, "clamp")


def recompute():
    """Descriptive, used sets: X1 lives and the B10 test set, re-computed with their own seeds (scores checked)."""
    print("registered", register(SPEC)[:12], flush=True)
    X = _map([("x1", a, k, c, 0, -1, 0.0) for c in range(NR + 1) for k in range(NB) for a in ("control", "bias",
                                                                                               "generic")], "x1re")
    Bt = _map([("b10", a, k, 0, 0, -1, 0.0) for k in range(NB)
               for a in ("augtab", "generic", "yoked", "bio", "bias", "auglin", "control", "base")], "b10re")
    if SMOKE:
        return
    worst = 0.0
    G = pd.read_csv(LOGS / "revX1" / "x1_groups.csv")
    cond_name = R.NAMES + ["switching"]
    xs = X.groupby(["arm", "base", "cond", "group"]).score.mean()
    for (arm, k, c), sub in xs.groupby(level=[0, 1, 2]):
        lg = G[(G.arm == arm) & (G.base == k) & (G.cond == cond_name[c])].sort_values("group").score.to_numpy()
        worst = max(worst, float(np.abs(sub.to_numpy() - lg).max()))
    raw = pd.read_csv(LOGS / "stageB10_test_raw.csv")      # one row per agent, in agent order
    for (arm, k), sub in Bt.groupby(["arm", "base"]):
        lg = raw[(raw.arm == arm) & (raw.run == k)]
        worst = max(worst, float(np.abs(sub.sort_values("agent").score.to_numpy() - lg.score.to_numpy()).max()),
                    float(np.abs(sub.sort_values("agent").life.to_numpy() / T - lg.survival.to_numpy()).max()))
    print("re-computation, max |diff| to the logged X1 and B10 scores:", worst, flush=True)
    (OUT / "recompute_check.txt").write_text(f"max_abs_diff {worst}\n")
    if worst > 1e-9:
        raise SystemExit("re-computation does not reproduce the logged scores: descriptive analyses not reported")


# ------------------------------------------------------------------ X6
def x6_job(job):
    arm, k, ch = job
    ns, i0 = X6["ch"], ch * X6["ch"]
    M, lay = sched_matrix(X6["sched"], ns, i0), layouts(ns, i0)
    B = M.shape[1]
    o = A.simulate(CTRL[arm], A.params(B, **params_of(arm, k)), M, bank(X6["bank"]), lay,
                   seed=X6["sim"] + 100 * k + ch)
    sc, sv = o["score"].reshape(ns, NA_), o["survival"].reshape(ns, NA_)
    return pd.DataFrame(dict(arm=arm, base=k, schedule=np.arange(i0, i0 + ns), score=sc.mean(1), survival=sv.mean(1),
                             score_a0=sc[:, 0], survival_a0=sv[:, 0]))


def x6tab():
    print("registered", register(SPEC)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, k, ch) for a in TAB_ARMS for k in range(NB) for ch in range(X6["ns"] // X6["ch"])]
    rows, t0 = [], time.time()
    with Pool(NWORK) as p:
        for i, df in enumerate(p.imap_unordered(x6_job, jobs)):
            rows.append(df)
            if i % 100 == 0 or i == len(jobs) - 1:
                print(f"x6tab {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(rows).sort_values(["arm", "base", "schedule"]).to_csv(OUT / "x6_tab.csv.gz", index=False)


def x6gpu():
    """Network solutions on the GPU subset (schedules 0..ngpu-1 of the common set, agent 0), two bases per call."""
    print("registered", register(SPEC)[:12], flush=True)
    if SMOKE:
        os.environ["X2_SMOKE"] = str(Path(SMOKE) / "x2smoke")
    import stage_c as SC
    import stage_c3 as C3
    import stage_x2x3 as X2
    from bio_rl.torchsim import world as W
    if SMOKE:
        C3.TCUT = T
    cache = OUT / "x6gpu_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ns = X6["ngpu"]
    M = sched_matrix(X6["sched"], ns, agents=1)[:C3.TCUT]
    lay = layouts(ns, agents=1)
    bk = SC._bank(X6["bank"])
    arms = X2.ALL_ARMS + ([] if SMOKE else ["ff_big_seq"])
    rows = []
    for arm in arms:
        f = cache / f"{arm}.pkl"
        if not f.exists():
            learner, cfg, d = X2.spec_of(arm)
            out = {}
            for c, ks in enumerate([(0, 1), (2, 3), (4, 5), (6, 7)]):
                prms = [json.loads((d / f"{arm}_r{k}.json").read_text())["params"] for k in ks]
                d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
                keys = set(d0) | set().union(*prms)
                P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], ns) for k_ in keys}
                t0 = time.time()
                o = W.simulate(learner, P, np.tile(M, (1, len(ks))), bk, np.tile(lay, len(ks)),
                               seed=X6["tsim"] + c, dqn={**C3.DQN, **cfg})
                for i, k in enumerate(ks):
                    out[k] = dict(score=np.asarray(o["score"][i * ns:(i + 1) * ns]),
                                  survival=np.asarray(o["survival"][i * ns:(i + 1) * ns]))
                print(f"x6gpu {arm} bases {ks}: {len(ks) * ns} agents, {time.time() - t0:.0f}s", flush=True)
            f.write_bytes(pickle.dumps(out))
        out = pickle.loads(f.read_bytes())
        for k, o in out.items():
            rows.append(pd.DataFrame(dict(arm=arm, base=k, schedule=np.arange(ns), score=o["score"],
                                          survival=o["survival"])))
    pd.concat(rows).to_csv(OUT / "x6_gpu.csv.gz", index=False)


# ------------------------------------------------------------------ report
def tint(bm, alpha=0.05):
    bm = np.asarray(bm, float)
    n = len(bm)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * bm.std(ddof=1) / np.sqrt(n)
    return bm.mean() - h, bm.mean() + h


def signflip_p(bm):
    """exact two-sided sign-flip permutation p over the base means"""
    bm = np.asarray(bm, float)
    s = np.array(list(itertools.product([1, -1], repeat=len(bm))))
    m = np.abs((s * bm).mean(1))
    return float((m >= abs(bm.mean()) - 1e-12).mean())


def compare(d, name, desc):
    """d: DataFrame (bases x groups) of paired differences"""
    d = d.dropna(axis=1, how="all")
    bm = d.mean(axis=1).to_numpy()
    lo, hi = ci3(d.to_numpy(), np.asarray(d.index))
    tl, th = tint(bm)
    return dict(test=name, comparison=desc, diff=float(np.nanmean(d.to_numpy())), bases_better=int((bm > 0).sum()),
                n_bases=len(bm), p_t=float(stats.ttest_1samp(bm, 0).pvalue), p_signflip=signflip_p(bm),
                d_z=float(bm.mean() / bm.std(ddof=1)) if bm.std(ddof=1) > 0 else np.nan,
                ci3_lo=lo, ci3_hi=hi, t_lo=tl, t_hi=th)


def group_scores(D, col="score"):
    return D.groupby(["kind", "arm", "base", "cond", "grace", "clamp", "group"])[col].mean()


def late_score(D, t0=1000):
    nb = T // BIN
    cols = [f"wb_{b}" for b in range(t0 // BIN, nb)]
    D = D.copy()
    D["late"] = D[cols].sum(1) / (T - t0)
    return D


def pv(S, kind, arm, cond, grace=100, clamp=-1):
    """(base x group) table of group scores"""
    return S.xs((kind, arm, cond, grace, clamp), level=["kind", "arm", "cond", "grace", "clamp"]).unstack("group")


def report():
    D = pd.read_csv(OUT / "x45_agents.csv.gz")
    C = pd.read_csv(OUT / "clamp_agents.csv.gz")
    S = group_scores(pd.concat([D, C]))
    sc = R.NAMES.index("scarce")
    pooled = lambda arm, grace=100: sum(pv(S, "x4", arm, c, grace) for c in range(NR)) / NR
    rows = [compare(pv(S, "x4", "generic", c) - pv(S, "x4", "bias", c), f"X4-P{c + 1}", f"generic - bias, {rn}")
            for c, rn in enumerate(R.NAMES)]
    rows.append(compare(pooled("generic") - pooled("bias"), "X4-P6", "generic - bias, pooled switch-free"))
    rows += [compare(pv(S, "x4", "generic", c) - pv(S, "x4", "control", c), f"X4-S{c + 1}", f"generic - control, {rn}")
             for c, rn in enumerate(R.NAMES)]
    rows.append(compare(pooled("generic") - pooled("control"), "X4-S6", "generic - control, pooled switch-free"))
    rows.append(compare(pooled("bias") - pooled("control"), "X4-S7", "bias - control, pooled switch-free"))
    gsw = pv(S, "x5", "generic", 5) - pv(S, "x5", "bias", 5)
    gst = pooled("generic") - pooled("bias")
    rows.append(compare(gsw - gst, "X4-S8", "(generic - bias) switching - (generic - bias) pooled switch-free"))
    for j, kn in enumerate(("alpha", "exploration", "safety weight")):
        rows.append(compare(pv(S, "x4", "generic", sc) - pv(S, "x4", "generic", sc, clamp=j), f"X5-C{j + 1}",
                            f"generic - generic with {kn} clamped, switch-free scarcity"))
    for j, kn in enumerate(("alpha", "exploration", "safety weight")):
        rows.append(compare(pv(S, "x5", "generic", 5) - pv(S, "x5", "generic", 5, clamp=j), f"X5-C{j + 4}",
                            f"generic - generic with {kn} clamped, switching lives"))
    for i, g in enumerate((50, 200)):
        rows.append(compare(pv(S, "x5", "generic", 5, g) - pv(S, "x5", "bias", 5, g), f"X5-D{i + 1}",
                            f"generic - bias, switching lives, death rule G = {g}"))
    for i, g in enumerate((50, 200)):
        rows.append(compare(pooled("generic", g) - pooled("bias", g), f"X5-D{i + 3}",
                            f"generic - bias, pooled switch-free, death rule G = {g}"))
    Tt = pd.DataFrame(rows)
    Tt["family"] = np.where(Tt.test.str.contains("-P"), "primary", "secondary")
    Tt["p_holm"] = np.nan
    for fam, idx in Tt.groupby("family").groups.items():
        Tt.loc[idx, "p_holm"] = holm(Tt.loc[idx, "p_t"].to_numpy())
    excl = ((Tt.ci3_lo > 0) | (Tt.ci3_hi < 0)) & ((Tt.t_lo > 0) | (Tt.t_hi < 0))
    Tt["verdict"] = np.where((Tt.p_holm < 0.05) & excl, "PASS", np.where(excl, "n.e.", "n.d."))
    # what each t interval excludes (for non-detections)
    Tt["excludes"] = [f"effects below {r.t_lo:+.3f} and above {r.t_hi:+.3f}" for r in Tt.itertuples()]
    # X4-E1: TOST on the same difference as X4-S8, margin 0.025
    bm = (gsw - gst).mean(1).to_numpy()
    m = 0.025
    p1 = stats.ttest_1samp(bm, -m, alternative="greater").pvalue
    p2 = stats.ttest_1samp(bm, m, alternative="less").pvalue
    l90, h90 = tint(bm, 0.10)
    E = pd.DataFrame([dict(test="X4-E1", comparison="TOST (generic - bias) switching - pooled switch-free, margin 0.025",
                           diff=float(bm.mean()), t90_lo=l90, t90_hi=h90, p_lower=p1, p_upper=p2,
                           p_tost=max(p1, p2), verdict="equivalent" if max(p1, p2) < 0.05 else "not shown")])
    Tt.to_csv(OUT / "v15_tests.csv", index=False)
    E.to_csv(OUT / "v15_tost.csv", index=False)
    # scores per arm and condition (base means, then mean and t interval)
    srow = []
    for (kind, arm, cond, grace, clamp), sub in S.groupby(level=["kind", "arm", "cond", "grace", "clamp"]):
        b = sub.groupby(level="base").mean().to_numpy()
        lo, hi = tint(b)
        srow.append(dict(kind=kind, arm=arm, cond=cond, grace=grace, clamp=clamp, score=b.mean(), lo=lo, hi=hi))
    pd.DataFrame(srow).to_csv(OUT / "v15_scores.csv", index=False)
    mech(D)
    late(D)
    signals()
    if (OUT / "b10re_agents.csv.gz").exists():
        survival_tables()
    if (OUT / "x6_tab.csv.gz").exists():
        x6_report()
    pd.set_option("display.width", 250)
    print(Tt[["family", "test", "comparison", "diff", "bases_better", "p_t", "p_signflip", "d_z", "p_holm", "ci3_lo",
              "ci3_hi", "t_lo", "t_hi", "verdict"]].round(4).to_string())
    print(E.round(4).to_string())


def mech(D):
    """cause-specific hazards, zero-energy time, longest zero-energy run, per kind, arm, death rule and regime"""
    out = []
    for (kind, arm, grace, cond), sub in D[D.clamp < 0].groupby(["kind", "arm", "grace", "cond"]):
        for rn in R.NAMES:
            al = sub.groupby("base")[f"alive_{rn}"].sum()
            if al.sum() == 0:
                continue
            rec = dict(kind=kind, arm=arm, grace=grace, cond=cond, regime=rn, alive_steps=int(al.sum()))
            for key, col in (("starv_per_1000", "dstarv"), ("inj_per_1000", "dinj")):
                v = (sub.groupby("base")[f"{col}_{rn}"].sum() / al.replace(0, np.nan) * 1000).dropna()
                rec[key], (rec[key + "_lo"], rec[key + "_hi"]) = v.mean(), tint(v)
                rec[key.split("_")[0] + "_deaths"] = int(sub[f"{col}_{rn}"].sum())
            v = (sub.groupby("base")[f"zero_{rn}"].sum() / al.replace(0, np.nan)).dropna()
            rec["zero_share"], (rec["zero_lo"], rec["zero_hi"]) = v.mean(), tint(v)
            out.append(rec)
    M = pd.DataFrame(out)
    M.to_csv(OUT / "v15_causes.csv", index=False)
    mz = D[D.clamp < 0].groupby(["kind", "arm", "grace", "cond", "base"]).maxzero.mean().groupby(
        ["kind", "arm", "grace", "cond"]).agg(["mean", "max"])
    mz.to_csv(OUT / "v15_maxzero.csv")
    # paired generic - bias differences in cause-specific hazards (X4 and X5, G = 100)
    rows = []
    for kind, conds in (("x4", range(NR)), ("x5", [5])):
        for cond in conds:
            for rn in R.NAMES:
                h = {}
                for arm in ("generic", "bias", "control"):
                    sub = D[(D.kind == kind) & (D.arm == arm) & (D.grace == 100) & (D.cond == cond) & (D.clamp < 0)]
                    al = sub.groupby("base")[f"alive_{rn}"].sum()
                    if al.sum() == 0:
                        break
                    h[arm] = {c: sub.groupby("base")[f"{c}_{rn}"].sum() / al * 1000 for c in ("dstarv", "dinj")}
                if len(h) < 3:
                    continue
                for b in ("bias", "control"):
                    for c in ("dstarv", "dinj"):
                        d = (h["generic"][c] - h[b][c]).to_numpy()
                        lo, hi = tint(d)
                        rows.append(dict(kind=kind, cond=cond, regime=rn, cause=c, comparison=f"generic - {b}",
                                         diff=d.mean(), lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(OUT / "v15_cause_diffs.csv", index=False)


def late(D):
    """score after step 1000: X4 (new set) and X1 (re-computed used set), base-level t intervals, no test"""
    rows = []
    for name, f, kind, conds in (("X4", OUT / "x45_agents.csv.gz", "x4", range(NR)),
                                 ("X1", OUT / "x1re_agents.csv.gz", "x1", range(NR + 1))):
        if not f.exists():
            continue
        X = late_score(D if name == "X4" else pd.read_csv(f))
        X = X[(X.kind == kind) & (X.grace == 100) & (X.clamp < 0)]
        S = X.groupby(["arm", "cond", "base", "group"]).late.mean()
        tab = {c: S.xs(c, level="cond").unstack("group") for c in conds}
        tab["pooled"] = sum(tab[c] for c in range(NR)) / NR
        for c, P in tab.items():
            for b in ("bias", "control"):
                d = P.loc["generic"] - P.loc[b]
                bm = d.mean(axis=1).to_numpy()
                lo, hi = tint(bm)
                cname = c if c == "pooled" else (R.NAMES + ["switching"])[c]
                rows.append(dict(set=name, cond=cname, comparison=f"generic - {b}", diff=bm.mean(), t_lo=lo, t_hi=hi,
                                 bases_better=int((bm > 0).sum())))
    pd.DataFrame(rows).to_csv(OUT / "v15_late_gain.csv", index=False)


def corr_from(S1, S2, N):
    m = S1 / N
    cov = S2 / N - np.outer(m, m)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return cov / np.outer(sd, sd)


def signals():
    rows = []
    for name in ("x45", "b10re"):
        f = OUT / f"{name}_signals.pkl"
        if not f.exists():
            continue
        sig = pickle.loads(f.read_bytes())
        acc = {}
        for (kind, arm, k, c, v, cj, cv), s in sig.items():
            if v != 0:
                continue
            for r in range(NR):
                if s["N"][r] < 100:
                    continue
                acc.setdefault((kind, arm, c, r), []).append(corr_from(s["S1"][r], s["S2"][r], s["N"][r]))
        for (kind, arm, c, r), cs in acc.items():
            cs = np.array(cs)
            for i, j in itertools.combinations(range(5), 2):
                v = cs[:, i, j]
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    continue
                lo, hi = tint(v)
                rows.append(dict(kind=kind, arm=arm, cond=c, regime=R.NAMES[r], pair=f"{SIGNALS[i]}~{SIGNALS[j]}",
                                 r=v.mean(), lo=lo, hi=hi, n_bases=len(v)))
    pd.DataFrame(rows).to_csv(OUT / "v15_signal_corr.csv", index=False)


def survival_tables():
    """B10 test set (re-computed): Kaplan-Meier (no censoring before 6000), RMST(6000), cause-specific hazards"""
    D = pd.read_csv(OUT / "b10re_agents.csv.gz")
    grid = np.arange(0, T + 1, 100)
    km, rm = [], []
    for arm, sub in D.groupby("arm"):
        per = np.array([[(g.life.to_numpy() > t).mean() for t in grid] for _, g in sub.groupby("base")])
        lo, hi = zip(*[tint(per[:, i]) if per[:, i].std() > 0 else (per[:, i].mean(),) * 2 for i in range(len(grid))])
        km.append(pd.DataFrame(dict(arm=arm, t=grid, surv=per.mean(0), lo=lo, hi=hi)))
        r = sub.groupby("base").life.mean().to_numpy()
        rm.append(dict(arm=arm, rmst=r.mean(), lo=tint(r)[0], hi=tint(r)[1]))
    pd.concat(km).to_csv(OUT / "b10_km.csv", index=False)
    life = D.groupby(["arm", "base"]).life.mean().unstack("arm")
    for a, b in (("generic", "bias"), ("generic", "control"), ("bias", "control"), ("generic", "bio")):
        d = (life[a] - life[b]).to_numpy()
        lo, hi = tint(d)
        rm.append(dict(arm=f"{a} - {b}", rmst=d.mean(), lo=lo, hi=hi, p_t=float(stats.ttest_1samp(d, 0).pvalue),
                       p_signflip=signflip_p(d)))
    pd.DataFrame(rm).to_csv(OUT / "b10_rmst.csv", index=False)
    D = D.assign(kind="b10", grace=100, clamp=-1)
    out = []
    for arm, sub in D.groupby("arm"):
        for rn in R.NAMES:
            al = sub.groupby("base")[f"alive_{rn}"].sum()
            rec = dict(arm=arm, regime=rn)
            for key, col in (("starv", "dstarv"), ("inj", "dinj")):
                v = sub.groupby("base")[f"{col}_{rn}"].sum() / al * 1000
                rec[key], (rec[key + "_lo"], rec[key + "_hi"]) = v.mean(), tint(v)
            v = sub.groupby("base")[f"zero_{rn}"].sum() / al
            rec["zero_share"] = v.mean()
            out.append(rec)
    pd.DataFrame(out).to_csv(OUT / "b10_causes.csv", index=False)


X6_TAB_COMP = [("generic", "control"), ("generic", "yoked"), ("generic", "augtab"), ("generic", "auglin"),
               ("generic", "bias"), ("bio", "control"), ("bio", "bias"), ("generic", "bio"), ("bias", "control"),
               ("control", "base"), ("generic_split", "generic")]
X6_NET_COMP = [("gru", "control_ra"), ("aug_ra", "control_ra"), ("control_ra", "control"), ("aug_ra", "gru"),
               ("gru", "aug"), ("gru", "ff_big"), ("ff_big", "control_ra"), ("ff_big", "aug_ra"), ("gru", "aug_ra"),
               ("film", "control"), ("aug", "control"), ("aug", "generic"), ("film", "gru"), ("aug", "gru"),
               ("gru", "control"), ("gru_aug", "gru"), ("metagrad", "control"), ("aug", "bias"), ("film", "bias"),
               ("generic", "bias"), ("metagrad", "bias"), ("gru", "ff_big_seq"), ("ff_big_seq", "ff_big"),
               ("ff_big_seq", "control_ra")]


def x6_report():
    rows, summ = [], []
    for fam, f, comps, col in (("tabular", "x6_tab.csv.gz", X6_TAB_COMP, "score"),
                               ("network", "x6_gpu.csv.gz", X6_NET_COMP, "score")):
        if not (OUT / f).exists():
            continue
        X = pd.read_csv(OUT / f)
        for arm, sub in X.groupby("arm"):
            for c in ("score", "survival") + (("score_a0",) if fam == "tabular" else ()):
                b = sub.groupby("base")[c].mean().to_numpy()
                lo, hi = tint(b)
                summ.append(dict(family=fam, arm=arm, measure=c, mean=b.mean(), lo=lo, hi=hi))
            if fam == "tabular":    # GPU subset (schedules 0..ngpu-1, agent 0) for comparison with the networks
                b = sub[sub.schedule < X6["ngpu"]].groupby("base").score_a0.mean().to_numpy()
                lo, hi = tint(b)
                summ.append(dict(family=fam, arm=arm, measure="score_gpu_subset", mean=b.mean(), lo=lo, hi=hi))
        P = X.pivot_table(index=["base", "schedule"], columns="arm", values=col)
        for a, b in comps:
            if a in P and b in P:
                rows.append({**compare((P[a] - P[b]).unstack("schedule"), f"X6 {fam}", f"{a} - {b}"), "family": fam})
    Tx = pd.DataFrame(rows)
    if len(Tx):
        Tx["p_holm"] = np.nan
        for fam, idx in Tx.groupby("family").groups.items():
            Tx.loc[idx, "p_holm"] = holm(Tx.loc[idx, "p_t"].to_numpy())
    Tx.to_csv(OUT / "x6_comparisons.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "x6_summary.csv", index=False)


def extra():
    """Descriptive additions for the paper (no tests, no new runs): per-regime generic - bias / control under each
    death rule (X4), switching-lives gains per death rule (X5), zero-energy share and longest zero-energy run in
    switch-free scarcity, and X6 estimates beside the registered estimates of the same comparisons."""
    D = pd.read_csv(OUT / "x45_agents.csv.gz")
    S = group_scores(D)
    rows = []
    for g in GRACE:
        for c, rn in list(enumerate(R.NAMES)) + [("pooled", "pooled switch-free"), (5, "switching")]:
            for b in ("bias", "control"):
                if c == "pooled":
                    d = sum(pv(S, "x4", "generic", k, g) - pv(S, "x4", b, k, g) for k in range(NR)) / NR
                else:
                    kind = "x5" if c == 5 else "x4"
                    d = pv(S, kind, "generic", c, g) - pv(S, kind, b, c, g)
                bm = d.mean(axis=1).to_numpy()
                lo, hi = tint(bm)
                rows.append(dict(grace=g, cond=rn, comparison=f"generic - {b}", diff=bm.mean(), t_lo=lo, t_hi=hi,
                                 bases_better=int((bm > 0).sum())))
    pd.DataFrame(rows).to_csv(OUT / "v15_grace_regime.csv", index=False)
    sc = R.NAMES.index("scarce")
    rows = []
    for g in GRACE:
        sub = D[(D.kind == "x4") & (D.cond == sc) & (D.grace == g) & (D.clamp < 0)]
        z, mz = {}, {}
        for arm in ("generic", "bias", "control"):
            s = sub[sub.arm == arm]
            z[arm] = s.groupby("base")["zero_scarce"].sum() / s.groupby("base")["alive_scarce"].sum()
            mz[arm] = s.groupby("base").maxzero.mean()
        for meas, v in (("zero_share", z), ("longest_zero_run", mz)):
            for arm in ("generic", "bias", "control"):
                lo, hi = tint(v[arm].to_numpy())
                rows.append(dict(grace=g, measure=meas, arm=arm, value=v[arm].mean(), lo=lo, hi=hi))
            for b in ("bias", "control"):
                d = (v["generic"] - v[b]).to_numpy()
                lo, hi = tint(d)
                rows.append(dict(grace=g, measure=meas, arm=f"generic - {b}", value=d.mean(), lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(OUT / "v15_zero_scarce.csv", index=False)
    reg = pd.read_csv(OUT.parent / "reanalysis_v15" / "tests_v15.csv")
    reg = reg[~reg.stage.isin(["X1"])]
    x6 = pd.read_csv(OUT / "x6_comparisons.csv")
    x6["key"] = x6.comparison.str.strip()
    rows = []
    for r in reg.itertuples():
        m = x6[x6.key == r.comparison.split(",")[0].strip()]
        fam = "network" if r.stage.startswith(("C", "X2")) else "tabular"
        m = m[m.family == fam]
        if not len(m):
            continue
        x = m.iloc[0]
        rows.append(dict(stage=r.stage, pred=r.pred, comparison=r.comparison, verdict_registered=r.verdict_registered,
                         reg_diff=r.diff, reg_t_lo=r.t_lo, reg_t_hi=r.t_hi, x6_diff=x["diff"], x6_t_lo=x.t_lo,
                         x6_t_hi=x.t_hi, x6_bases_better=int(x.bases_better), same_sign=bool(np.sign(r.diff) == np.sign(x["diff"])),
                         x6_excludes_0=bool(x.t_lo > 0 or x.t_hi < 0)))
    pd.DataFrame(rows).to_csv(OUT / "x6_vs_registered.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    {"run": run, "recompute": recompute, "x6tab": x6tab, "x6gpu": x6gpu, "report": report,
     "extra": extra}[sys.argv[1]]()
