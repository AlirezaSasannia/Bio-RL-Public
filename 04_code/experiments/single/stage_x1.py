"""Revision X1 (family single_revision_v1_3, see revision_v13.py): frozen B10 agents (control, bias, generic) in
stationary single-regime lives, plus a switching reference, behaviour logs through the read-only simulate hook, and
conditional hazards for the B10 test set (deterministic re-computation with its own seeds).
Run: python stage_x1.py [run|hazard|report|all] [--workers N]
Smoke (dev seeds, scratch output, nothing registered): set X1_SMOKE=<scratch dir>."""
import json
import os
import sys
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
from reanalysis_runlevel import ci3
from revision_v13 import SPEC

LOGS = Path(__file__).resolve().parents[2] / "results" / "logs" / "single"
OPT = LOGS / "stageB10_opt"
ARMS = ["control", "bias", "generic"]
CTRL = {"control": "vdbe", "bias": "vdbe_generic", "generic": "vdbe_generic", "bio": "vdbe_fuzzy"}
NB, NS, NA_ = 8, 128, 4
T, JUV = R.T_LIFE, R.JUVENILE
BANK, SIM, SWITCH = 5_200_000, 320_000, 53_000
B10T = dict(sched=20000, bank=2_000_000, sim=86_000)
OUT = LOGS / "revX1"
SMOKE = os.environ.get("X1_SMOKE", "").strip()
if SMOKE:      # code check only: dev seeds, short lives, scratch output
    OUT, NS, T = Path(SMOKE), 6, 1500
    BANK, SIM, SWITCH = 1_970_000, 135_000, 97_500
    register = lambda spec: "smoke-no-registration"
NR = len(R.NAMES)
COND = R.NAMES + ["switching"]
BIN = 100
KEYS = ["alive", "wb", "move", "shelter", "dfood", "dready", "nready", "energy", "ea", "ee", "es", "alpha", "eps",
        "smult"]
BINKEYS = ["alive", "move", "shelter", "dfood", "energy", "ea", "ee", "es", "alpha", "eps", "smult"]
_banks = {}


def bank(lo):
    if lo not in _banks:
        _banks[lo] = LayoutBank.build(range(lo, lo + 1000))
    return _banks[lo]


def layouts(ns):
    return np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(ns)])


def regimes_for(c, ns):
    if c < NR:
        M = np.full((T, ns * NA_), c, int)
        M[:JUV] = R.NAMES.index("safe")
        return M
    sch = [R.schedule(np.random.default_rng(SWITCH + i)) for i in range(ns)]
    return A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:T]


class Recorder:
    """Read-only hook (draws no random numbers, changes no state): per agent and regime sums over steps alive, the
    wellbeing sum over steps >= JUV, and 100-step time bins of knobs and behaviour (sums over alive agents)."""

    def __init__(self, B, size):
        self.acc = {k: np.zeros((B, NR)) for k in KEYS}
        self.died = np.zeros((B, NR))
        self.wbpost = np.zeros(B)
        self.bins = {k: np.zeros(T // BIN) for k in BINKEYS}
        self.size = size

    def __call__(self, t, L):
        alive_pre = L["alive"] | L["died"]
        oh = (L["reg"][:, None] == np.arange(NR)[None, :]) & alive_pre[:, None]
        w = oh.astype(float)
        sz, s2, food = self.size, L["s2"], L["food"]
        dist = np.abs(s2[:, None] // sz - food // sz) + np.abs(s2[:, None] % sz - food % sz)      # (B, NF)
        ready = L["timers"] == 0
        has = ready.any(1)
        eff = L["eff_rec"]
        vals = {"alive": np.ones(len(s2)), "wb": L["wellbeing"], "move": (L["a"] != 0).astype(float),
                "shelter": L["in_sh"].astype(float), "dfood": dist.min(1).astype(float),
                "dready": np.where(has, np.where(ready, dist, 10 ** 6).min(1), 0).astype(float),
                "nready": has.astype(float), "energy": L["n"][:, 0], "ea": eff[:, 0], "ee": eff[:, 1],
                "es": eff[:, 2], "alpha": np.broadcast_to(L["a_t"], s2.shape), "eps": L["eps_now"],
                "smult": np.broadcast_to(L["s_t"], s2.shape)}
        for k, v in vals.items():
            self.acc[k] += w * np.asarray(v, float)[:, None]
        self.died += L["died"][:, None] & oh
        a = alive_pre.astype(float)
        if t >= JUV:
            self.wbpost += a * L["wellbeing"]
        b = t // BIN
        for k in BINKEYS:
            self.bins[k][b] += (a * np.asarray(vals[k], float)).sum()


def run_job(job):
    """job = (tag, arm, k, c): tag 'x1' (new set) or 'b10' (B10 test re-computation, its own seeds)."""
    tag, arm, k, c = job
    prm = json.loads((OPT / f"{arm}_r{k}.json").read_text())["params"]
    if tag == "x1":
        M, lay, bk, seed, cond = regimes_for(c, NS), layouts(NS), bank(BANK), SIM + 10 * k + c, COND[c]
    else:
        sch = [R.schedule(np.random.default_rng(B10T["sched"] + i)) for i in range(NS)]
        M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:T]
        lay, bk, seed, cond = layouts(NS), bank(B10T["bank"]), B10T["sim"] + k, "b10test"
    B = M.shape[1]
    rec = Recorder(B, bk.SIZE)
    o = A.simulate(CTRL[arm], A.params(B, **prm), M, bk, lay, seed=seed, hook=rec)
    ns = B // NA_
    g = lambda v: np.asarray(v, float).reshape(ns, NA_).sum(1)          # sums over the 4 agents of a group
    df = pd.DataFrame(dict(group=np.arange(ns), score=g(o["score"]) / NA_, survival=g(o["survival"]) / NA_,
                           score_post=g(rec.wbpost) / NA_ / (T - JUV)))
    for j, rn in enumerate(R.NAMES):
        for kk in KEYS:
            df[f"{kk}_{rn}"] = g(rec.acc[kk][:, j])
        df[f"died_{rn}"] = g(rec.died[:, j])
        df[f"steps_{rn}"] = g((M == j).sum(0))
    df["tag"], df["arm"], df["base"], df["cond"] = tag, arm, k, cond
    bins = pd.DataFrame(rec.bins)
    bins["bin"] = np.arange(len(bins))
    bins["tag"], bins["arm"], bins["base"], bins["cond"] = tag, arm, k, cond
    return df, bins, o["score"], o["survival"], job


def _run(jobs, name):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, bins, chk = [], [], []
    with Pool(NWORK) as p:
        for df, bn, sc, sv, job in p.imap_unordered(run_job, jobs):
            rows.append(df)
            bins.append(bn)
            chk.append((job, sc, sv))
            print(*job, round(float(df.score.mean()), 4), flush=True)
    pd.concat(rows).sort_values(["arm", "base", "cond", "group"]).to_csv(OUT / f"{name}_groups.csv", index=False)
    pd.concat(bins).sort_values(["arm", "base", "cond", "bin"]).to_csv(OUT / f"{name}_bins.csv", index=False)
    return chk


def run():
    print("registered", register(SPEC)[:12], flush=True)
    _run([("x1", a, k, c) for c in range(len(COND)) for k in range(NB) for a in ARMS], "x1")


def hazard():
    print("registered", register(SPEC)[:12], flush=True)
    chk = _run([("b10", a, k, 0) for a in ["control", "bias", "generic", "bio"] for k in range(NB)], "b10hazard")
    if SMOKE:
        return
    raw = pd.read_csv(LOGS / "stageB10_test_raw.csv")
    worst = 0.0
    for (tag, arm, k, c), sc, sv in chk:
        logged = raw[(raw.arm == arm) & (raw.run == k)]
        worst = max(worst, float(np.abs(logged.score.to_numpy() - sc).max()),
                    float(np.abs(logged.survival.to_numpy() - sv).max()))
    print("B10 re-computation, max |diff| to the logged score/survival:", worst, flush=True)
    (OUT / "b10hazard_check.txt").write_text(f"max_abs_diff {worst}\n")
    if worst > 1e-9:
        raise SystemExit("B10 re-computation does not reproduce the logged test: hazards not reported")


# ------------------------------------------------------------------ report
def tint(bm, alpha=0.05):
    n = len(bm)
    h = stats.t.ppf(1 - alpha / 2, n - 1) * bm.std(ddof=1) / np.sqrt(n)
    return bm.mean() - h, bm.mean() + h


def compare(d, name, desc):
    """d: DataFrame (bases x groups) of paired differences; two-sided base-level t-test, ci3 and t intervals."""
    bm = d.mean(1).to_numpy()
    lo, hi = ci3(d.to_numpy(), np.asarray(d.index))
    tl, th = tint(bm)
    return dict(test=name, comparison=desc, diff=float(d.to_numpy().mean()), bases_better=int((bm > 0).sum()),
                n_bases=len(bm), p_t=float(stats.ttest_1samp(bm, 0).pvalue),
                p_wilcoxon=float(stats.wilcoxon(bm, method="exact").pvalue) if np.any(bm != 0) else 1.0,
                ci3_lo=lo, ci3_hi=hi, t_lo=tl, t_hi=th)


def report():
    G = pd.read_csv(OUT / "x1_groups.csv")
    S = G.pivot_table(index=["cond", "base", "group"], columns="arm", values="score_post")
    sw = G[G.cond == "switching"].pivot_table(index=["base", "group"], columns="arm", values="score")
    st = {rn: S.loc[rn] for rn in R.NAMES}
    pooled = sum(st.values()) / NR
    D = lambda P, a, b: (P[a] - P[b]).unstack("group")
    rows = [compare(D(st[rn], "generic", "bias"), f"X1-P{c + 1}", f"generic - bias, stationary {rn}")
            for c, rn in enumerate(R.NAMES)]
    rows.append(compare(D(pooled, "generic", "bias"), "X1-P6", "generic - bias, pooled stationary"))
    rows += [compare(D(st[rn], "generic", "control"), f"X1-S{c + 1}", f"generic - control, stationary {rn}")
             for c, rn in enumerate(R.NAMES)]
    rows.append(compare(D(pooled, "bias", "control"), "X1-S6", "bias - control, pooled stationary"))
    rows.append(compare(D(sw, "generic", "bias") - D(pooled, "generic", "bias"), "X1-S7",
                        "(generic - bias) switching - (generic - bias) pooled stationary"))
    rows.append(compare(D(sw, "generic", "bias"), "X1-d1", "generic - bias, switching lives (descriptive)"))
    rows.append(compare(D(sw, "generic", "control"), "X1-d2", "generic - control, switching lives (descriptive)"))
    Tt = pd.DataFrame(rows)
    Tt.to_csv(OUT / "x1_tests_raw.csv", index=False)
    means = pd.concat({rn: st[rn].groupby("base").mean() for rn in R.NAMES} | {"pooled": pooled.groupby("base").mean(),
                      "switching": sw.groupby("base").mean()}, names=["cond"]).groupby("cond").mean()
    means.to_csv(OUT / "x1_scores.csv")
    # behaviour per arm, condition and regime: per-base rates (sum / alive steps), then mean and t interval over bases
    out = []
    for (arm, cond), sub in G.groupby(["arm", "cond"]):
        for rn in R.NAMES:
            al = sub.groupby("base")[f"alive_{rn}"].sum()
            if al.sum() == 0:
                continue
            rec = dict(arm=arm, cond=cond, regime=rn, alive_steps=float(al.sum()))
            for kk in KEYS[1:]:
                if kk == "nready":
                    continue
                den = sub.groupby("base")[f"nready_{rn}"].sum() if kk == "dready" else al
                v = (sub.groupby("base")[f"{kk}_{rn}"].sum() / den.replace(0, np.nan)).dropna()
                rec[kk], (rec[kk + "_lo"], rec[kk + "_hi"]) = v.mean(), tint(v.to_numpy())
            dz = sub.groupby("base")[f"died_{rn}"].sum() / al.replace(0, np.nan) * 1000
            rec["hazard_per_1000"], (rec["hazard_lo"], rec["hazard_hi"]) = dz.mean(), tint(dz.dropna().to_numpy())
            rec["deaths"] = float(sub[f"died_{rn}"].sum())
            out.append(rec)
    Bh = pd.DataFrame(out)
    Bh.to_csv(OUT / "x1_behaviour.csv", index=False)
    Bn = pd.read_csv(OUT / "x1_bins.csv")
    tr = Bn.groupby(["arm", "cond", "bin"])[BINKEYS].sum()
    tr = tr[BINKEYS[1:]].div(tr["alive"].replace(0, np.nan), axis=0)
    tr.to_csv(OUT / "x1_traces.csv")
    pd.set_option("display.width", 250)
    print(Tt.round(4).to_string())
    print(means.round(4).to_string())
    print(Bh[Bh.regime == "scarce"][["arm", "cond", "move", "shelter", "dfood", "dready", "energy", "es", "ee", "ea",
                                     "hazard_per_1000"]].round(3).to_string())


def report_hazard():
    G = pd.read_csv(OUT / "b10hazard_groups.csv")
    out = []
    for arm, sub in G.groupby("arm"):
        for rn in R.NAMES:
            al = sub.groupby("base")[f"alive_{rn}"].sum()
            dz = sub.groupby("base")[f"died_{rn}"].sum() / al * 1000
            lo, hi = tint(dz.to_numpy())
            out.append(dict(arm=arm, regime=rn, hazard_per_1000=dz.mean(), lo=lo, hi=hi,
                            deaths=float(sub[f"died_{rn}"].sum()), alive_steps=float(al.sum())))
    H = pd.DataFrame(out)
    # paired hazard differences generic - bias and generic - control per regime (base level, t interval)
    for b in ("bias", "control"):
        for rn in R.NAMES:
            hz = {a: (G[G.arm == a].groupby("base")[f"died_{rn}"].sum() / G[G.arm == a].groupby("base")[f"alive_{rn}"]
                      .sum() * 1000) for a in ("generic", b)}
            d = (hz["generic"] - hz[b]).to_numpy()
            lo, hi = tint(d)
            out.append(dict(arm=f"generic - {b}", regime=rn, hazard_per_1000=d.mean(), lo=lo, hi=hi,
                            p_t=float(stats.ttest_1samp(d, 0).pvalue)))
    H = pd.DataFrame(out)
    H.to_csv(OUT / "b10_hazards.csv", index=False)
    print(H.round(4).to_string())


if __name__ == "__main__":
    {"run": run, "hazard": hazard, "report": lambda: (report(), report_hazard()),
     "all": lambda: (run(), hazard(), report(), report_hazard())}[sys.argv[1]]()
