"""Stage B3 (placebo controls) and phase-2 warm start. Run: python experiments/single/stage_b3.py [opt|test]"""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import GENERIC, HORMONE_ONLY, SPACES, decode_space, encode
from bio_rl.fast.popsim import LayoutBank
from spec_b3 import AMENDMENT_B3, SPEC_B3
NRUN = 4

LOGS = ROOT / "results" / "logs" / "single"
OUT = LOGS / "stageB3_opt"
B2 = LOGS / "stageB2_opt"
ARMS = {"bio": ("vdbe_fuzzy", HORMONE_ONLY), "generic": ("vdbe_generic", GENERIC),
        "randrules": ("vdbe_randrules", HORMONE_ONLY), "nomem": ("vdbe_nomem", HORMONE_ONLY),
        "control": ("vdbe", SPACES["vdbe"])}
GENS, POP, NAG = 20, 12, 48
_bank = None


def bank():
    global _bank
    if _bank is None:
        _bank = LayoutBank.build(range(400000, 402000))
    return _bank


def score(ctrl, cands, gen_seed, gens_bank=None):
    rng = np.random.default_rng(gen_seed)
    scheds = [R.schedule(rng) for _ in range(NAG)]
    lay1 = rng.integers(len(bank()), size=NAG)
    m = len(cands)
    keys = set().union(*cands)
    P = A.params(m * NAG, **{k: np.repeat([c[k] for c in cands], NAG) for k in keys})
    o = A.simulate(ctrl, P, A.regime_matrix(scheds * m, R.T_LIFE), bank(), np.tile(lay1, m), seed=gen_seed + 1)
    return o["score"].reshape(m, NAG).mean(1)


def run(job):
    arm, r = job
    ctrl, space = ARMS[arm]
    base = json.loads((LOGS / "stageB_opt" / f"vdbe_r{r % 4}.json").read_text())["params"]
    extra = {"rule_seed": 100 + r} if arm == "randrules" else {}
    if arm == "control":
        x0, make = encode(space, base), (lambda u: decode_space(space, u))
    else:
        x0 = np.full(len(space), 0.5)
        make = lambda u: {**base, **extra, **decode_space(space, u)}
    es = cma.CMAEvolutionStrategy(x0, 0.15, {"bounds": [0, 1], "popsize": POP, "seed": 7000 + r, "verbose": -9})
    hist = []
    for g in range(GENS):
        X = es.ask()
        f = score(ctrl, [make(x) for x in X], 16_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    res = dict(arm=arm, run=r, controller=ctrl, params=make(es.mean), history=hist)
    (OUT / f"{arm}_r{r}.json").write_text(json.dumps(res, indent=1))
    return arm, r, hist[-1]


def warm(r):
    """Phase 2: continue joint optimisation from the B2 fixed+hormones solution in the stage-B fuzzy space."""
    space = SPACES["fuzzy"]
    p0 = json.loads((B2 / f"fixed_hormones_r{r}.json").read_text())["params"]
    es = cma.CMAEvolutionStrategy(encode(space, p0), 0.15, {"bounds": [0, 1], "popsize": POP, "seed": 8000 + r,
                                                            "verbose": -9})
    hist = []
    for g in range(30):
        X = es.ask()
        f = score("fuzzy", [decode_space(space, x) for x in X], 17_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    res = dict(arm="warm", run=r, controller="fuzzy", start=p0, params=decode_space(space, es.mean), history=hist)
    (OUT / f"warm_r{r}.json").write_text(json.dumps(res, indent=1))
    return "warm", r, hist[-1]


def _job(j):
    return warm(j[1]) if j[0] == "warm" else run(j)


def optimise():
    print("registered", register(SPEC_B3)[:12], register(AMENDMENT_B3)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for r in range(4):     # reuse B2 runs 0-3 for bio and control
        for arm, src in (("bio", "vdbe_hormones"), ("control", "vdbe_control")):
            p = OUT / f"{arm}_r{r}.json"
            if not p.exists():
                d = json.loads((B2 / f"{src}_r{r}.json").read_text())
                p.write_text(json.dumps(dict(arm=arm, run=r, controller=d["controller"], params=d["params"],
                                             history=d["history"], source="stage B2"), indent=1))
    jobs = [(a, r) for r in range(NRUN) for a in ARMS if not (OUT / f"{a}_r{r}.json").exists()]
    jobs += [("warm", r) for r in range(4) if not (OUT / f"warm_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for a, r, m in p.imap_unordered(_job, jobs):
            print(a, r, round(m, 3), flush=True)


# ------------------------------------------------------------------ test
NS, NA_ = 128, 4
TB = None


def test_set(kind):
    rng_seed = {"main": 3000, "short": 4000, "long": 5000, "lethal": 3000}[kind]
    dur = {"short": (200, 500), "long": (1500, 3000)}.get(kind, R.DUR)
    sch = [R.schedule(np.random.default_rng(rng_seed + i), dur=dur) for i in range(NS)]
    return sch, A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)


def evaluate(args):
    kind, ctrl, prm, tag, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(800000, 801000))
    sch, M = test_set(kind)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=41_000 + r,
                   loss=0.6 if kind == "lethal" else R.LOSS, record=("wellbeing",))
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
    df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
    for j, rn in enumerate(R.NAMES):
        m = M == j
        df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
    df["arm"], df["run"], df["set"] = tag, r, kind
    return df


def hier_ci(diff_by_run_sched, n=4000, seed=0):
    """diff_by_run_sched: (runs, schedules) differences; resample runs then schedules."""
    rng = np.random.default_rng(seed)
    R_, S_ = diff_by_run_sched.shape
    out = np.empty(n)
    for b in range(n):
        rr = rng.integers(R_, size=R_)
        ss = rng.integers(S_, size=S_)
        out[b] = diff_by_run_sched[np.ix_(rr, ss)].mean()
    return np.percentile(out, [2.5, 97.5])


def test():
    jobs = []
    for kind in ("main", "short", "long", "lethal"):
        for arm in ARMS:
            for r in range(NRUN):
                d = json.loads((OUT / f"{arm}_r{r}.json").read_text())
                jobs.append((kind, d["controller"], d["params"], arm, r))
        if kind == "main":
            for r in range(4):
                for tag, f in (("warm", OUT / f"warm_r{r}.json"),):
                    if f.exists():
                        jobs.append((kind, "fuzzy", json.loads(f.read_text())["params"], tag, r))
                    jobs.append((kind, "fuzzy", json.loads((B2 / f"fixed_hormones_r{r}.json").read_text())["params"],
                                 "warm_start_point", r))
                    jobs.append((kind, "fuzzy", json.loads((LOGS / "stageB_opt" / f"fuzzy_r{r}.json").read_text())["params"],
                                 "fuzzy_from_scratch", r))
    with Pool(NWORK) as p:
        D = pd.concat(p.map(evaluate, jobs))
    D.to_csv(LOGS / "stageB3_test_raw.csv", index=False)
    rows = []
    for kind in ("main", "short", "long", "lethal"):
        X = D[D.set == kind]
        piv = X.groupby(["arm", "run", "schedule"]).score.mean()
        for other in ("generic", "randrules", "nomem", "control"):
            d = (piv["bio"] - piv[other]).unstack()            # runs x schedules
            run_diff = d.mean(1).values
            lo, hi = hier_ci(d.values)
            p = float(stats.wilcoxon(d.mean(0).values, alternative="greater").pvalue)   # over schedules
            rows.append(dict(set=kind, comparison=f"bio - {other}", diff=run_diff.mean(), ci_lo=lo, ci_hi=hi,
                             runs_better=int((run_diff > 0).sum()), p=p))
    T = pd.DataFrame(rows)
    main = T.index[T.set == "main"]
    adj = holm({i: T.loc[i, "p"] for i in main})
    T["p_holm"] = np.nan
    for i in main:
        T.loc[i, "p_holm"] = adj[i]
    T["verdict"] = np.where(T.set == "main", np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL"), "secondary")
    T.to_csv(LOGS / "stageB3_tests.csv", index=False)
    S = D.groupby(["set", "arm"])[["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES]].mean()
    S.to_csv(LOGS / "stageB3_summary.csv")
    # per-regime bio - control on the main set, with hierarchical CI
    X = D[D.set == "main"]
    pr = []
    for rn in R.NAMES:
        piv = X.groupby(["arm", "run", "schedule"])[f"score_{rn}"].mean()
        d = (piv["bio"] - piv["control"]).unstack()
        d = d.loc[:, d.notna().all(0)]
        lo, hi = hier_ci(d.values)
        pr.append(dict(regime=rn, diff=d.values.mean(), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(pr).to_csv(LOGS / "stageB3_per_regime.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(4).to_string())
    print(S.round(3).to_string())
    print(pd.DataFrame(pr).round(4).to_string())


if __name__ == "__main__":
    {"opt": optimise, "test": test}[sys.argv[1]]()
