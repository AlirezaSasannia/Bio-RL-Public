"""Stage B9 (roadmap step 3): is the global layer better than standard ways of adapting hyperparameters online?
Arms on the frozen VDBE bases r % 4 (8 runs each), the same three knob offsets (log learning rate, exploration logit,
log safety weight) and the same CMA-ES budget (20 x 12 x 48, sigma 0.15) for each method's own meta-parameters:
  metagrad  online SPSA meta-gradient: the knob offsets theta are learned within the life by gradient ascent on
            recent wellbeing, estimated from an antithetic pair of windows (theta + cD, then theta - cD, D Rademacher).
            Tuned: initial offsets (3), step size, perturbation size, window length (6 params).
  bandit    sliding-window UCB (Garivier & Moulines 2011) over 7 knob settings (centre, +-step on one knob at a time);
            arms switch every N steps, reward = mean wellbeing over the block. Tuned: centre (3), steps (3), N,
            window, UCB coefficient (9 params).
  generic, bias, control   existing 8 runs (stages B3/B7/B8), re-tested.
Every step of the pipeline is resumable: CMA checkpoints per generation, evaluations cached per job.
Run: python stage_b9.py [opt|test|speed|report|all] [--workers N]"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
import stage_b5 as B5
import stage_b6 as B6
import stage_b7 as B7
from _resume import cached_map, cma_run
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space
from bio_rl.fast.popsim import LayoutBank

SPEC_B9 = {
    "id": "single_stageB9_modern_baselines",
    "registered": "after draft v0.5 and the move to the local machine, before any B9 run",
    "question": ("Is the generic global layer (appraisal signals -> knob offsets) better than standard online "
                 "hyperparameter adaptation (meta-gradient, bandit meta-controller) acting on the same knobs?"),
    "arms": {
        "metagrad": ("online SPSA meta-gradient on the 3 knob offsets: antithetic pair of windows of W steps "
                     "(theta + c*D, theta - c*D, D Rademacher), theta += lr * (r_A - r_B) / (2c) * D, theta clipped to "
                     "+-6, r = mean wellbeing in the window; perturbation noise from a separate RNG (common random "
                     "numbers for the world kept). Tuned: theta0 (3, (-6, 6)), lr (0.01, 100) log, c (0.01, 1) log, "
                     "W (10, 300) log; 6 params. Chosen over a one-step TD meta-gradient because the knobs act "
                     "through a non-differentiable tabular learner with per-state VDBE exploration and death"),
        "bandit": ("sliding-window UCB over 7 knob settings (centre, centre +- step on one knob at a time); arm "
                   "switches every N steps, reward = mean wellbeing over the block, window tau pulls, UCB bonus "
                   "c * sqrt(log(min(k, tau)) / n). Tuned: centre (3, (-6, 6)), steps (3, (0.05, 3) log), N (10, 300) "
                   "log, tau (5, 500) log, c (0.001, 1) log; 9 params"),
        "generic": "existing generic global layer, 8 runs (B3 runs 0-3, B7 runs 4-7), re-tested",
        "bias": "existing constant-offset control, 8 runs (B8 runs 0-3, B7 addendum runs 4-7), re-tested",
        "control": "existing further-tuning control, 8 runs (B3/B7), re-tested",
    },
    "budget": ("each new arm: CMA-ES 20 generations x 12 candidates x 48 agents, sigma 0.15, on VDBE base r % 4, "
               "runs 0-7; start u = 0.5 (offsets 0; metagrad lr 1, c 0.1, W 55; bandit steps 0.39, N 55, tau 50, "
               "c 0.03); CMA seeds 11000 + r (metagrad), 12000 + r (bandit); training schedules shared by both new "
               "arms, generation seed 21000000 + 1000 r + g. Same budget as generic, bias and control"),
    "test": ("new independent set, never used before: 128 schedules (seeds 7000-7127) x 4 agents, layout bank "
             "1000000-1000999, simulator seeds 81000 + run; common random numbers across arms"),
    "predictions": {"P1": "generic > metagrad", "P2": "generic > bandit", "P3": "metagrad > control",
                    "P4": "bandit > control", "P5": "generic > control"},
    "inference": ("one-sided Wilcoxon over the 128 schedules of the run-averaged difference, Holm across P1-P5, plus "
                  "hierarchical bootstrap 95% CI (runs, then schedules); a claim needs both p_Holm < 0.05 and the CI "
                  "excluding 0. Runs paired by index (same base r % 4)"),
    "secondary": {
        "vs_bias": "metagrad - bias, bandit - bias, generic - bias (two-sided, CI)",
        "decomposition": "survival, wellbeing while alive, per-regime score, deaths by regime (as per_regime_b7)",
        "per_regime": "generic - metagrad and generic - bandit inside each regime, hierarchical CI",
        "speed": ("B6 switch protocol with death disabled on new layouts 1100000-1100999, simulator seeds 91000 + "
                  "run; arms - control per switch and window, CI only"),
        "bounds": "parameters of the new arms within 2% of a search bound are reported",
        "bio": "the biological module (8 runs) is evaluated as a descriptive reference only",
    },
    "interpretation": ("P1 and P2 pass: reading appraisal signals beats standard online hyperparameter adaptation on "
                       "the same knobs in this task. P1 or P2 fails: the global layer's gain is not specific to "
                       "appraisal signals; a standard method does as well (no equivalence claimed without TOST). "
                       "metagrad or bandit not better than bias: the method mostly finds a constant shift"),
}

LOGS = B3.LOGS
OUT = LOGS / "stageB9_opt"
CACHE = LOGS / "stageB9_cache"
GENS, POP, NAG, SIG = 20, 12, 48, 0.15
SPACE = {
    "metagrad": {"m_a": (-6.0, 6.0, "lin"), "m_e": (-6.0, 6.0, "lin"), "m_s": (-6.0, 6.0, "lin"),
                 "mg_lr": (0.01, 100.0, "log"), "mg_c": (0.01, 1.0, "log"), "mg_W": (10.0, 300.0, "log")},
    "bandit": {"n_a": (-6.0, 6.0, "lin"), "n_e": (-6.0, 6.0, "lin"), "n_s": (-6.0, 6.0, "lin"),
               "st_a": (0.05, 3.0, "log"), "st_e": (0.05, 3.0, "log"), "st_s": (0.05, 3.0, "log"),
               "bd_N": (10.0, 300.0, "log"), "bd_tau": (5.0, 500.0, "log"), "bd_c": (0.001, 1.0, "log")},
}
CTRL = {"metagrad": "vdbe_metagrad", "bandit": "vdbe_bandit"}
CMA_SEED = {"metagrad": 11_000, "bandit": 12_000}
NEW = ("metagrad", "bandit")
OLD = ("generic", "bias", "control", "bio")
R8 = range(8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ optimisation
def run(job):
    arm, r = job
    space = SPACE[arm]
    base = json.loads((LOGS / "stageB_opt" / f"vdbe_r{r % 4}.json").read_text())["params"]
    make = lambda u: {**base, **decode_space(space, u)}
    res = cma_run(OUT / f"{arm}_r{r}.json", np.full(len(space), 0.5), SIG, POP, CMA_SEED[arm] + r, GENS,
                  lambda X, g: B3.score(CTRL[arm], [make(x) for x in X], 21_000_000 + 1000 * r + g), make,
                  dict(arm=arm, run=r, controller=CTRL[arm]))
    return arm, r, res["history"][-1]


def opt():
    log("registered", register(SPEC_B9)[:12])
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, r) for r in R8 for a in NEW if not (OUT / f"{a}_r{r}.json").exists()]
    log(f"opt: {len(jobs)} CMA runs to do, workers {NWORK}")
    with Pool(min(NWORK, max(len(jobs), 1))) as p:
        for a, r, m in p.imap_unordered(run, jobs):
            log("done", a, r, round(m, 4))


def load(arm, r):
    if arm in NEW:
        return json.loads((OUT / f"{arm}_r{r}.json").read_text())
    return B7.load(arm, r)


# ------------------------------------------------------------------ test on the new independent set
NS, NA_ = 128, 4
TB = None


def evaluate(args):
    ctrl, prm, tag, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(1_000_000, 1_001_000))
    sch = [R.schedule(np.random.default_rng(7000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=81_000 + r, record=("wellbeing",))
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
    df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
    life = np.round(o["survival"] * R.T_LIFE).astype(int)
    died = life < R.T_LIFE
    reg_at_death = M[np.clip(life - 1, 0, R.T_LIFE - 1), np.arange(len(lay))]
    for j, rn in enumerate(R.NAMES):
        m = M == j
        df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
        df[f"died_{rn}"] = (died & (reg_at_death == j)).astype(float)
    df["arm"], df["run"] = tag, r
    return df


def test():
    arms = NEW + OLD
    jobs = [(load(a, r)["controller"], load(a, r)["params"], a, r) for a in arms for r in R8]
    keys = [f"test_{a}_r{r}" for a in arms for r in R8]
    with Pool(NWORK) as p:
        D = pd.concat(cached_map(p, evaluate, jobs, keys, CACHE, log))
    D.to_csv(LOGS / "stageB9_test_raw.csv", index=False)
    log("test: raw written", len(D))


# ------------------------------------------------------------------ speed protocol (secondary)
TB2 = None


def speed_eval(args):
    ctrl, prm, tag, r, (a, b) = args
    global TB2
    if TB2 is None:
        TB2 = LayoutBank.build(range(1_100_000, 1_101_000))
    PRE, POST, NL = B6.PRE, B6.POST, B6.NL
    B = NL * NA_
    M = np.zeros((PRE + POST, B), int)
    M[:PRE], M[PRE:] = R.NAMES.index(a), R.NAMES.index(b)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NL)])
    o = A.simulate(ctrl, A.params(B, **prm), M, TB2, lay, seed=91_000 + r, juvenile=PRE + POST + 1,
                   record=("wellbeing",))
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(layout=np.repeat(np.arange(NL), NA_), pre=wb[PRE - 200:PRE].mean(0)))
    for lo, hi in B6.WIN:
        df[f"w{lo}_{hi}"] = wb[PRE + lo:PRE + hi].mean(0)
    df["arm"], df["run"], df["switch"] = tag, r, f"{a}->{b}"
    return df


def speed():
    arms = NEW + ("generic", "bias", "control")
    jobs, keys = [], []
    for a in arms:
        for r in R8:
            j = load(a, r)
            for s in B6.SW:
                jobs.append((j["controller"], j["params"], a, r, s))
                keys.append(f"speed_{a}_r{r}_{s[0]}-{s[1]}")
    with Pool(NWORK) as p:
        D = pd.concat(cached_map(p, speed_eval, jobs, keys, CACHE, log))
    D.to_csv(LOGS / "stageB9_speed_raw.csv", index=False)
    rows = []
    for s in D.switch.unique():
        X = D[D.switch == s]
        for col in ("pre", "w0_100", "w100_300", "w0_300", "w300_1000"):
            pv = X.groupby(["arm", "run", "layout"])[col].mean()
            for a in arms:
                if a != "control":
                    d = (pv[a] - pv["control"]).unstack()
                    lo, hi = B3.hier_ci(d.values, n=2000)
                    rows.append(dict(switch=s, window=col, comparison=f"{a} - control", diff=d.values.mean(),
                                     ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(LOGS / "stageB9_speed_tests.csv", index=False)
    log("speed: tests written")


# ------------------------------------------------------------------ analysis
def report():
    D = pd.read_csv(LOGS / "stageB9_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    prim = [("P1", "generic", "metagrad"), ("P2", "generic", "bandit"), ("P3", "metagrad", "control"),
            ("P4", "bandit", "control"), ("P5", "generic", "control")]
    T = pd.DataFrame([dict(pred=k, **B5.comp(piv, a, b)) for k, a, b in prim])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL")
    T["family"] = "primary"
    sec = pd.DataFrame([B5.comp(piv, a, b, "two-sided") for a, b in
                        (("metagrad", "bias"), ("bandit", "bias"), ("generic", "bias"), ("bias", "control"),
                         ("metagrad", "bandit"), ("bio", "control"), ("generic", "bio"))])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(LOGS / "stageB9_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageB9_summary.csv")
    # deaths by regime: runs with fewer deaths than bias / metagrad / bandit
    pr = []
    for rn in R.NAMES:
        pv = D.groupby(["arm", "run", "schedule"])[f"score_{rn}"].mean()
        for a, b in (("generic", "metagrad"), ("generic", "bandit"), ("metagrad", "control"), ("bandit", "control")):
            d = (pv[a] - pv[b]).unstack()
            d = d.loc[:, d.notna().all(0)]
            lo, hi = B3.hier_ci(d.values)
            pr.append(dict(regime=rn, comparison=f"{a} - {b}", diff=d.values.mean(), ci_lo=lo, ci_hi=hi))
        dr = D.groupby(["arm", "run"])[f"died_{rn}"].mean()
        for a, b in (("generic", "metagrad"), ("generic", "bandit"), ("generic", "bias")):
            dd = (dr[a] - dr[b])
            pr.append(dict(regime=rn, comparison=f"deaths {a} - {b}", diff=dd.mean(), runs_fewer=int((dd < 0).sum())))
    pd.DataFrame(pr).to_csv(LOGS / "stageB9_per_regime.csv", index=False)
    # parameters near the search bounds
    rows = []
    for a in NEW:
        for r in R8:
            u = np.array(load(a, r)["u"])
            for (k, (lo, hi, sc)), ui in zip(SPACE[a].items(), u):
                rows.append(dict(arm=a, run=r, param=k, value=load(a, r)["params"][k], u=ui,
                                 on_bound=bool(ui < 0.02 or ui > 0.98)))
    Bd = pd.DataFrame(rows)
    Bd.to_csv(LOGS / "stageB9_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(S.round(4).to_string())
    print(pd.DataFrame(pr).round(4).to_string())
    print(Bd.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (opt, test, report, speed):
        log("=== step", step.__name__)
        step()
    log("=== B9 finished")


if __name__ == "__main__":
    {"opt": opt, "test": test, "speed": speed, "report": report, "all": all_}[sys.argv[1]]()
