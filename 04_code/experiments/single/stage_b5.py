"""Stage B5 (registered after an external review of draft v0.3, before any B5 run).
Q1  information or capacity?   yoked: generic controller (15 params) fed another agent's appraisal inputs
Q2  which inputs are needed?   loo_i: generic controller without input i (12 params), i = threat, K, surprise,
                               intake, energy deficit
Q3  global modulation or just more information?
                               augtab: the 5 inputs, binarised at optimised thresholds, extend the learner's state
                                       (64 cells x 32 contexts); thresholds at 1 switch a signal off (5 params)
                               auglin: the 5 inputs enter the action values linearly, weights learned online by the
                                       same TD error (learning rate and input scale optimised, 2 params)
Q4  replication of B4 on an independent test set (new schedules, new layouts, new noise seeds).
All arms: the 4 frozen stage-B VDBE bases, CMA-ES 20 x 12 x 48, sigma 0.15, start u = 0.5.
Run: python stage_b5.py [opt|test]"""
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
import stage_b3 as B3
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space
from bio_rl.fast.popsim import LayoutBank

INPUTS = ["threat", "K", "surprise", "intake", "deficit"]
SPEC_B5 = {
    "id": "single_stageB5_info_vs_modulation", "registered": "after review of draft v0.3, before any B5 run",
    "arms": {"yoked": "generic, inputs from another agent (np.roll 24 over the batch), 15 params",
             **{f"loo_{i}": f"generic without input {n}, 12 params" for i, n in enumerate(INPUTS)},
             "augtab": "inputs binarised (optimised thresholds) appended to the Q-table state, 5 params",
             "auglin": "inputs as linear features of the action values, TD-learned weights, 2 params"},
    "budget": "as B3/B4: 20 x 12 x 48, sigma 0.15, start u=0.5, 4 runs (one per VDBE base)",
    "test": ("independent replication set: 128 new schedules (seeds 6000-6127) x 4 agents, layout bank "
             "900000-900999, simulator seeds 61000+run; never used before. B3/B4 arms re-tested on it"),
    "predictions": {"E1": "generic > yoked", "E2": "generic > augtab", "E3": "generic > auglin",
                    "E4": "generic > surprise_only (B4 replication)", "E5": "generic > appraisal_only (B4 replication)"},
    "inference": ("one-sided Wilcoxon over schedules of the run-averaged difference, Holm across E1-E5, plus "
                  "hierarchical bootstrap CI (runs, schedules); a claim needs both"),
    "secondary": {"loo": "generic - loo_i for each input, CI and Holm over the 5 (secondary family)",
                  "replication": "bio - control, generic - bio on the new set",
                  "tost": ("B3 randrules and nomem vs bio: equivalence margin +/-0.033 (half of the B3 bio - control "
                           "gain), 90% hierarchical CI, on the B3 set and on the new set")},
}
AMENDMENT_B5 = {"id": "single_stageB5_amendment1", "amends": "single_stageB5_info_vs_modulation",
                "when": "after the yoked and augtab training runs (training curves only), before any test",
                "change": ("add arm augtab0: as augtab but CMA starts at u=0.9 (thresholds 0.90, signals almost always "
                           "off, i.e. near zero effect like the generic arm's start), because augtab started with all "
                           "32 contexts active and was still improving at the end. Reported as secondary; E2 is "
                           "evaluated against the better of augtab and augtab0 (chosen on training score)")}
OUT = B3.LOGS / "stageB5_opt"
GEN_W = lambda idx: {f"w_{i}_{k}": (-3.0, 3.0, "lin") for i in idx for k in ("a", "e", "s")}
SPACE = {"yoked": GEN_W(range(5)), **{f"loo_{j}": GEN_W([i for i in range(5) if i != j]) for j in range(5)},
         "augtab": {f"thr_{i}": (0.02, 1.0, "lin") for i in range(5)},
         "augtab0": {f"thr_{i}": (0.02, 1.0, "lin") for i in range(5)},
         "auglin": {"eta_th": (1e-4, 0.3, "log"), "x_scale": (0.2, 5.0, "log")}}
CTRL = {"yoked": "vdbe_generic", "augtab": "vdbe_augtab", "augtab0": "vdbe_augtab", "auglin": "vdbe_auglin",
        **{f"loo_{j}": "vdbe_generic" for j in range(5)}}
EXTRA = {"yoked": {"yoked": 1.0}}


def run(job):
    arm, r = job
    space = SPACE[arm]
    base = json.loads((B3.LOGS / "stageB_opt" / f"vdbe_r{r % 4}.json").read_text())["params"]
    make = lambda u: {**base, **EXTRA.get(arm, {}), **decode_space(space, u)}
    es = cma.CMAEvolutionStrategy(np.full(len(space), 0.9 if arm == "augtab0" else 0.5), 0.15,
                                  {"bounds": [0, 1], "popsize": B3.POP, "seed": 9500 + r, "verbose": -9})
    hist = []
    for g in range(B3.GENS):
        X = es.ask()
        f = B3.score(CTRL[arm], [make(x) for x in X], 19_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    (OUT / f"{arm}_r{r}.json").write_text(json.dumps(dict(arm=arm, run=r, controller=CTRL[arm],
                                                          params=make(es.mean), history=hist), indent=1))
    return arm, r, hist[-1]


def optimise():
    print("registered", register(SPEC_B5)[:12], register(AMENDMENT_B5)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["yoked", "augtab", "auglin", "augtab0"] + [f"loo_{j}" for j in range(5)]
    jobs = [(a, r) for a in order for r in range(4) if not (OUT / f"{a}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for a, r, m in p.imap_unordered(run, jobs):
            print(a, r, round(m, 3), flush=True)


# ------------------------------------------------------------------ test on the independent set
NS, NA_ = 128, 4
TB = None


def evaluate(args):
    ctrl, prm, tag, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(900000, 901000))
    sch = [R.schedule(np.random.default_rng(6000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=61_000 + r)
    df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
    df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
    df["arm"], df["run"] = tag, r
    return df


def all_arms():
    src = {"control": B3.OUT, "bio": B3.OUT, "generic": B3.OUT, "randrules": B3.OUT, "nomem": B3.OUT,
           "surprise_only": B3.OUT, "appraisal_only": B3.OUT, **{a: OUT for a in SPACE}}
    jobs = []
    for a, d in src.items():
        for r in range(4):
            j = json.loads((d / f"{a}_r{r}.json").read_text())
            jobs.append((j["controller"], j["params"], a, r))
    return jobs


def comp(piv, a, b, alt="greater"):
    d = (piv[a] - piv[b]).unstack()
    lo, hi = B3.hier_ci(d.values)
    lo90, hi90 = np.percentile(_boot(d.values), [5, 95])
    p = float(stats.wilcoxon(d.mean(0).values, alternative=alt).pvalue)
    return dict(comparison=f"{a} - {b}", diff=d.values.mean(), ci_lo=lo, ci_hi=hi, ci90_lo=lo90, ci90_hi=hi90,
                runs_better=int((d.mean(1) > 0).sum()), p=p)


def _boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    R_, S_ = x.shape
    return np.array([x[np.ix_(rng.integers(R_, size=R_), rng.integers(S_, size=S_))].mean() for _ in range(n)])


def test():
    with Pool(NWORK) as p:
        D = pd.concat(p.map(evaluate, all_arms()))
    D.to_csv(B3.LOGS / "stageB5_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    tr = {a: np.mean([json.loads((OUT / f"{a}_r{r}.json").read_text())["history"][-1] for r in range(4)])
          for a in ("augtab", "augtab0")}
    best_tab = max(tr, key=tr.get)
    print("augtab arm used for E2 (training score):", best_tab, {k: round(v, 4) for k, v in tr.items()})
    prim = [comp(piv, "generic", b) for b in ("yoked", best_tab, "auglin", "surprise_only", "appraisal_only")]
    T = pd.DataFrame(prim)
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["family"] = "primary"
    loo = pd.DataFrame([comp(piv, "generic", f"loo_{j}") for j in range(5)])
    adj = holm(dict(enumerate(loo.p)))
    loo["p_holm"] = [adj[i] for i in range(len(loo))]
    loo["family"] = "loo"
    sec = pd.DataFrame([comp(piv, "bio", "control"), comp(piv, "generic", "bio"), comp(piv, "generic", "control"),
                        comp(piv, "bio", "randrules", "two-sided"), comp(piv, "bio", "nomem", "two-sided"),
                        comp(piv, "augtab", "control", "two-sided"), comp(piv, "augtab0", "control", "two-sided"),
                        comp(piv, "generic", "augtab0" if best_tab == "augtab" else "augtab"), comp(piv, "auglin", "control", "two-sided"),
                        comp(piv, "yoked", "control", "two-sided")])
    sec["family"] = "secondary"
    T = pd.concat([T, loo, sec], ignore_index=True)
    T.to_csv(B3.LOGS / "stageB5_tests.csv", index=False)
    S = D.groupby("arm")[["score", "survival", "alive_wellbeing"]].mean()
    S.to_csv(B3.LOGS / "stageB5_summary.csv")
    pd.set_option("display.width", 220)
    print(T.round(4).to_string())
    print(S.sort_values("score").round(3).to_string())


if __name__ == "__main__":
    {"opt": optimise, "test": test}[sys.argv[1]]()
