"""Stage B10 (registered after the 2026-10-21 simulated review, before any B10 run): independent bases.
Every earlier add-on run sat on one of four shared stage-B VDBE bases, so the base learner was never replicated. B10
tunes 8 NEW VDBE bases with the stage-B protocol (new CMA seeds, new training schedules) and runs ONE add-on
optimisation per arm on each base. The base is the replication unit (n = 8).
Run: python stage_b10.py [bases|opt|test|all] [--workers N]"""
import json
import os
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
import stage_b5 as B5
import stage_b8 as B8
from _resume import cma_run
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import GENERIC, HORMONE_ONLY, SPACES, decode_space, encode, evaluate
from bio_rl.fast.popsim import LayoutBank
from reanalysis_runlevel import ci3, holm

NB = 8
BGENS = 30
TEST = dict(sched=20000, bank=2_000_000, sim=86_000)
SPEC_B10 = {
    "id": "single_stageB10_independent_bases",
    "registered": "after the 2026-10-21 review and base-level re-analysis, before any B10 run",
    "question": ("Do the tabular add-on results hold when the base learner itself is replicated (8 independently tuned "
                 "VDBE bases, one add-on run per arm and base)?"),
    "bases": ("8 new VDBE bases k = 0..7, stage-B protocol and search space (alpha0 0.02-0.9, eps0 0-0.5, s0 0.25-8 log, "
              "vdbe_sigma 0.001-1 log; same bounds as stage B so the setting is replicated): CMA-ES 30 gens x 12 x 48 "
              "agents, sigma 0.3, start u ~ U(0.1, 0.9) from default_rng(23100 + k), CMA seed 23000 + k, generation "
              "seeds 50000000 + 1000 k + g (new training schedules), training layout bank 400000-401999"),
    "arms": {"control": "further tuning of the base in the stage-B vdbe space, start at the base",
             "bio": "vdbe_fuzzy, 16 params (4 centres + 12 gains), start u = 0.5",
             "generic": "vdbe_generic, 15 weights in (-3, 3), start u = 0.5",
             "bias": "vdbe_generic without inputs: 3 constant offsets in (-6, 6), start u = 0.5",
             "yoked": "generic with another agent's inputs (np.roll 24), 15 weights, start u = 0.5",
             "augtab": ("5 thresholds, signals binarised into the Q-table state; start u = 0.9 (near-zero effect, as "
                        "B5 augtab0, which had the better training score in 3/4 B5 bases)"),
             "auglin": "signals as linear TD-learned features, 2 params, start u = 0.5"},
    "budget": ("each arm on each base: CMA-ES 20 gens x 12 x 48, sigma 0.15, CMA seed 24000 + k, generation seeds "
               "51000000 + 1000 k + g shared by all arms of base k (common random numbers)"),
    "test": ("new independent set, never used: 128 schedules (seeds 20000-20127) x 4 agents, layout bank "
             "2000000-2000999, simulator seeds 86000 + k (shared by all arms of base k)"),
    "predictions": {"P1": "generic > control", "P2": "generic > yoked", "P3": "generic > augtab",
                    "P4": "generic > auglin", "P5": "generic > bias"},
    "secondary": {"S1": "bio > control (one-sided)", "S2": "bio vs bias (two-sided)", "S3": "generic > bio (one-sided)",
                  "S4": "bias > control (one-sided)"},
    "inference": ("d = difference of agent-averaged score per (base, schedule), paired by base and schedule. Unit = base "
                  "(n = 8): one-sided t-test and one-sided exact Wilcoxon signed-rank on the 8 base means, Holm "
                  "separately for each test over P1-P5 (and over S1-S4); 95% three-level bootstrap CI "
                  "(reanalysis_runlevel.ci3: bases, runs within base (one run here), schedules; 4000 draws, seed 0). "
                  "A primary claim holds if the base-level t-test p_Holm < 0.05 AND the CI excludes 0; the Wilcoxon on "
                  "base means is reported as a non-parametric check (disagreement reported). Also reported: the "
                  "registered schedule-level one-sided Wilcoxon (over the 128 schedules of the base-averaged "
                  "difference) with the 2-level hierarchical CI (stage_b3.hier_ci) and its Holm over P1-P5."),
    "descriptive": ("survival, alive wellbeing, per-regime score and deaths by regime; constant-shift fraction "
                    "(bias - control) / (generic - control) with a bootstrap over bases; parameters within 1% of a "
                    "search bound for every base and arm"),
    "interpretation": ("P1-P5 passing replicates the four-shared-base results with independent bases. S1/S4 decide "
                       "whether 'bio > control' and 'about 1/3 of the gain is a constant shift' can be stated without "
                       "the qualifier 'on four shared bases'. No detectable difference is not equivalence."),
}
ARMS = ["augtab", "generic", "yoked", "bio", "bias", "auglin", "control"]     # slowest first
SPACE = {"control": SPACES["vdbe"], "bio": HORMONE_ONLY, "generic": GENERIC, "bias": B8.SPACE["bias"],
         "yoked": GENERIC, "augtab": B5.SPACE["augtab"], "auglin": B5.SPACE["auglin"]}
CTRL = {"control": "vdbe", "bio": "vdbe_fuzzy", "generic": "vdbe_generic", "bias": "vdbe_generic",
        "yoked": "vdbe_generic", "augtab": "vdbe_augtab", "auglin": "vdbe_auglin"}
EXTRA = {"yoked": {"yoked": 1.0}}
LOGS = B3.LOGS
OUT = LOGS / "stageB10_opt"
if os.environ.get("B10_SMOKE"):     # code check only: 1 generation, dev seeds, scratch output, nothing registered
    OUT = LOGS = Path(os.environ["B10_SMOKE"])
    BGENS, B3.GENS, NS_SMOKE = 1, 1, 16
    TEST = dict(sched=98000, bank=1_900_000, sim=98_500)
    register = lambda spec: "smoke-no-registration"
_bank = None


def tbank():
    global _bank
    if _bank is None:
        _bank = LayoutBank.build(range(400000, 402000))
    return _bank


def base_job(k):
    space = SPACES["vdbe"]
    x0 = np.random.default_rng(23100 + k).uniform(0.1, 0.9, len(space))
    res = cma_run(OUT / f"base_r{k}.json", x0, 0.3, 12, 23000 + k, BGENS,
                  lambda X, g: evaluate("vdbe", [decode_space(space, x) for x in X], 48, tbank(),
                                        gen_seed=50_000_000 + 1000 * k + g),
                  lambda u: decode_space(space, u), dict(arm="base", run=k, controller="vdbe"))
    return "base", k, res["history"][-1]


def load_base(k):
    return json.loads((OUT / f"base_r{k}.json").read_text())["params"]


def addon_job(job):
    arm, k = job
    space, base = SPACE[arm], load_base(k)
    if arm == "control":
        x0, make = encode(space, base), (lambda u: decode_space(space, u))
    else:
        x0 = np.full(len(space), 0.9 if arm == "augtab" else 0.5)
        make = lambda u: {**base, **EXTRA.get(arm, {}), **decode_space(space, u)}
    res = cma_run(OUT / f"{arm}_r{k}.json", x0, 0.15, B3.POP, 24000 + k, B3.GENS,
                  lambda X, g: B3.score(CTRL[arm], [make(x) for x in X], 51_000_000 + 1000 * k + g),
                  make, dict(arm=arm, run=k, controller=CTRL[arm]))
    return arm, k, res["history"][-1]


def bases():
    print("registered", register(SPEC_B10)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(min(NWORK, NB)) as p:
        for a, k, m in p.imap_unordered(base_job, range(NB)):
            print(a, k, round(m, 3), flush=True)


def opt():
    print("registered", register(SPEC_B10)[:12], flush=True)
    jobs = [(a, k) for a in ARMS for k in range(NB) if not (OUT / f"{a}_r{k}.json").exists()]
    with Pool(NWORK) as p:
        for a, k, m in p.imap_unordered(addon_job, jobs):
            print(a, k, round(m, 3), flush=True)


# ------------------------------------------------------------------ test on the new independent set
NS, NA_ = (NS_SMOKE if os.environ.get("B10_SMOKE") else 128), 4
TB = None


def test_eval(args):
    ctrl, prm, tag, k = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(TEST["bank"], TEST["bank"] + 1000))
    sch = [R.schedule(np.random.default_rng(TEST["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=TEST["sim"] + k, record=("wellbeing", "alive"))
    return frame(o, M, tag, k)


def frame(o, M, tag, k):
    wb, al = o["wellbeing"], o["alive"].astype(bool)
    died = np.vstack([np.zeros((1, al.shape[1]), bool), al[:-1] & ~al[1:]])
    df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
    df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
    for j, rn in enumerate(R.NAMES):
        m = M == j
        df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
        df[f"death_{rn}"] = (died & m).any(0).astype(float)
    df["arm"], df["run"] = tag, k
    return df


def compare(piv, a, b, alt="greater"):
    d = (piv[a] - piv[b]).unstack()                       # bases x schedules
    bm = d.mean(1).values
    lo, hi = ci3(d.values, np.asarray(d.index))
    hlo, hhi = B3.hier_ci(d.values)
    return dict(comparison=f"{a} - {b}", alternative=alt, diff=d.values.mean(), bases_better=int((bm > 0).sum()),
                n_bases=len(bm), sd_base=bm.std(ddof=1),
                p_t=float(stats.ttest_1samp(bm, 0, alternative=alt).pvalue),
                p_wilcoxon=float(stats.wilcoxon(bm, alternative=alt, method="exact").pvalue),
                ci3_lo=lo, ci3_hi=hi,
                p_sched=float(stats.wilcoxon(d.mean(0).values, alternative=alt).pvalue), hci_lo=hlo, hci_hi=hhi)


def family(piv, pairs, name):
    T = pd.DataFrame([dict(pred=p, **compare(piv, a, b, alt)) for p, a, b, alt in pairs])
    for col in ("p_t", "p_wilcoxon", "p_sched"):
        T[col + "_holm"] = holm(T[col].to_numpy())
    excl = np.where(T.alternative == "greater", T.ci3_lo > 0, (T.ci3_lo > 0) | (T.ci3_hi < 0))
    T["verdict"] = np.where((T.p_t_holm < 0.05) & excl, "PASS", "FAIL")
    excl_s = np.where(T.alternative == "greater", T.hci_lo > 0, (T.hci_lo > 0) | (T.hci_hi < 0))
    T["verdict_sched"] = np.where((T.p_sched_holm < 0.05) & excl_s, "PASS", "FAIL")
    T["family"] = name
    return T


PRIMARY = [("P1", "generic", "control", "greater"), ("P2", "generic", "yoked", "greater"),
           ("P3", "generic", "augtab", "greater"), ("P4", "generic", "auglin", "greater"),
           ("P5", "generic", "bias", "greater")]
SECONDARY = [("S1", "bio", "control", "greater"), ("S2", "bio", "bias", "two-sided"),
             ("S3", "generic", "bio", "greater"), ("S4", "bias", "control", "greater")]


def bounds_table():
    rows = []
    for k in range(NB):
        for arm in ["base"] + ARMS:
            j = json.loads((OUT / f"{arm}_r{k}.json").read_text())
            sp = SPACES["vdbe"] if arm == "base" else SPACE[arm]
            for (name, (lo, hi, _)), u in zip(sp.items(), j["u"]):
                if u < 0.01 or u > 0.99:
                    rows.append(dict(arm=arm, base=k, param=name, value=j["params"][name], bound=lo if u < 0.5 else hi))
    return pd.DataFrame(rows, columns=["arm", "base", "param", "value", "bound"])


def test():
    print("registered", register(SPEC_B10)[:12], flush=True)
    jobs = [(CTRL[a], json.loads((OUT / f"{a}_r{k}.json").read_text())["params"], a, k)
            for a in ARMS for k in range(NB)]
    jobs += [("vdbe", load_base(k), "base", k) for k in range(NB)]
    with Pool(NWORK) as p:
        D = pd.concat(p.map(test_eval, jobs))
    D.to_csv(LOGS / "stageB10_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([family(piv, PRIMARY, "primary"), family(piv, SECONDARY, "secondary")], ignore_index=True)
    T.to_csv(LOGS / "stageB10_tests.csv", index=False)
    S = D.groupby("arm").mean(numeric_only=True).drop(columns=["run", "schedule"])
    S.to_csv(LOGS / "stageB10_summary.csv")
    # constant-shift fraction, bootstrap over bases
    bm = D.groupby(["arm", "run"]).score.mean().unstack(0)
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(4000):
        x = bm.iloc[rng.integers(NB, size=NB)].mean()
        fr.append((x["bias"] - x["control"]) / (x["generic"] - x["control"]))
    m = bm.mean()
    shift = pd.DataFrame([dict(fraction=(m["bias"] - m["control"]) / (m["generic"] - m["control"]),
                               ci_lo=np.percentile(fr, 2.5), ci_hi=np.percentile(fr, 97.5))])
    shift.to_csv(LOGS / "stageB10_shift_fraction.csv", index=False)
    bm.to_csv(LOGS / "stageB10_base_means.csv")
    Bt = bounds_table()
    Bt.to_csv(LOGS / "stageB10_bounds.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S[["score", "survival", "alive_wellbeing"] + [f"death_{r}" for r in R.NAMES]].sort_values("score")
          .round(3).to_string())
    print(bm.round(3).to_string())
    print(shift.round(3).to_string())
    print(Bt.groupby(["arm", "param"]).size().to_string())


def run_all():
    bases()
    for k in range(NB):
        print(k, {a: round(v, 4) for a, v in load_base(k).items()}, flush=True)
    opt()
    test()


if __name__ == "__main__":
    {"bases": bases, "opt": opt, "test": test, "all": run_all}[sys.argv[1]]()
