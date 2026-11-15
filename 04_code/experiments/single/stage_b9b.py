"""Stage B9b (registered after B9 results, before any B9b run): warm-started baselines.
In B9 the meta-gradient and bandit arms ended below the constant-offset control (bias), although both contain it
(metagrad with lr -> 0 and the bandit's centre arm are constant offsets): their CMA search, from offsets 0, did not
reach the bias solution within the budget. B9b asks whether online adaptation adds anything on top of the constant
shift, and whether the generic layer still wins, by starting the two baselines at the bias solution of run r.
Resumable as B9. Run: python stage_b9b.py [opt|test|report|all] [--workers N]"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
import stage_b5 as B5
import stage_b7 as B7
import stage_b9 as B9
from _resume import cached_map, cma_run
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.fast.popsim import LayoutBank

SPEC_B9B = {
    "id": "single_stageB9b_warmstart",
    "registered": "after B9 results (B9 test set used), before any B9b run",
    "observation": ("B9: metagrad and bandit below bias on the B9 test set (-0.040, -0.047) and below control in "
                    "training; optimiser shrank the SPSA step (lr 0.02-0.13); neither found the bias offsets "
                    "(bias raises log learning rate by ~+0.4 in 8/8 runs)"),
    "arms": {"metagrad_ws": "B9 metagrad space; CMA start: theta0 = bias_r offsets (b_a, b_e, b_s), other params u=0.5",
             "bandit_ws": "B9 bandit space; CMA start: centre = bias_r offsets, other params u=0.5"},
    "budget": ("20 x 12 x 48, sigma 0.15, 8 runs on VDBE base r % 4 (in addition to the bias budget they inherit, "
               "as the add-on arms inherit the base's budget); CMA seeds 13000 + r, 14000 + r; generation seed "
               "22000000 + 1000 r + g, shared by both arms"),
    "test": ("new independent set, never used before: 128 schedules (seeds 8000-8127) x 4 agents, layout bank "
             "1200000-1200999, simulator seeds 82000 + run; generic, bias, control and the B9 arms re-tested on it"),
    "predictions": {"Q1": "generic > metagrad_ws", "Q2": "generic > bandit_ws"},
    "inference": "as B9: one-sided Wilcoxon over schedules + hierarchical CI; Holm over Q1-Q2; both needed",
    "secondary": {"add_to_shift": "metagrad_ws - bias, bandit_ws - bias (two-sided, CI)",
                  "replication": "B9 P1-P5 re-estimated on the new set (descriptive)",
                  "decomposition": "survival, wellbeing while alive, deaths by regime"},
    "interpretation": ("Q1, Q2 pass: the generic layer beats standard online adaptation even when that adaptation "
                       "starts from the best constant shift. ws arms > bias: online adaptation adds to the shift; "
                       "ws arms <= bias: in this task online hyperparameter adaptation from wellbeing adds nothing"),
}
OUT = B3.LOGS / "stageB9b_opt"
CACHE = B3.LOGS / "stageB9b_cache"
WS = {"metagrad_ws": ("metagrad", 13_000, ("m_a", "m_e", "m_s")), "bandit_ws": ("bandit", 14_000, ("n_a", "n_e", "n_s"))}
R8 = range(8)
log = B9.log


def run(job):
    arm, r = job
    src, seed, keys = WS[arm]
    space = B9.SPACE[src]
    base = json.loads((B3.LOGS / "stageB_opt" / f"vdbe_r{r % 4}.json").read_text())["params"]
    bias = B7.load("bias", r)["params"]
    start = {k: 0.0 for k in space}
    start.update({k: v for k, v in zip(keys, (bias["b_a"], bias["b_e"], bias["b_s"]))})
    x0 = np.full(len(space), 0.5)
    x0[:3] = encode({k: space[k] for k in keys}, {k: start[k] for k in keys})
    make = lambda u: {**base, **decode_space(space, u)}
    res = cma_run(OUT / f"{arm}_r{r}.json", x0, B9.SIG, B9.POP, seed + r, B9.GENS,
                  lambda X, g: B3.score(B9.CTRL[src], [make(x) for x in X], 22_000_000 + 1000 * r + g), make,
                  dict(arm=arm, run=r, controller=B9.CTRL[src]))
    return arm, r, res["history"][-1]


def opt():
    log("registered", register(SPEC_B9B)[:12])
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, r) for r in R8 for a in WS if not (OUT / f"{a}_r{r}.json").exists()]
    log(f"opt: {len(jobs)} CMA runs to do, workers {NWORK}")
    with Pool(min(NWORK, max(len(jobs), 1))) as p:
        for a, r, m in p.imap_unordered(run, jobs):
            log("done", a, r, round(m, 4))


def load(arm, r):
    if arm in WS:
        return json.loads((OUT / f"{arm}_r{r}.json").read_text())
    return B9.load(arm, r)


NS, NA_ = 128, 4
TB = None


def evaluate(args):
    ctrl, prm, tag, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(1_200_000, 1_201_000))
    sch = [R.schedule(np.random.default_rng(8000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=82_000 + r, record=("wellbeing",))
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


ARMS = tuple(WS) + ("metagrad", "bandit", "generic", "bias", "control")


def test():
    jobs = [(load(a, r)["controller"], load(a, r)["params"], a, r) for a in ARMS for r in R8]
    keys = [f"test_{a}_r{r}" for a in ARMS for r in R8]
    with Pool(NWORK) as p:
        D = pd.concat(cached_map(p, evaluate, jobs, keys, CACHE, log))
    D.to_csv(B3.LOGS / "stageB9b_test_raw.csv", index=False)
    log("test: raw written", len(D))


def report():
    D = pd.read_csv(B3.LOGS / "stageB9b_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.DataFrame([dict(pred=k, **B5.comp(piv, "generic", b)) for k, b in (("Q1", "metagrad_ws"), ("Q2", "bandit_ws"))])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL")
    T["family"] = "primary"
    sec = pd.DataFrame([B5.comp(piv, a, b, "two-sided") for a, b in
                        (("metagrad_ws", "bias"), ("bandit_ws", "bias"), ("metagrad_ws", "control"),
                         ("bandit_ws", "control"), ("generic", "metagrad"), ("generic", "bandit"),
                         ("metagrad", "control"), ("bandit", "control"), ("generic", "control"), ("generic", "bias"))])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(B3.LOGS / "stageB9b_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(B3.LOGS / "stageB9b_summary.csv")
    rows = []
    for a, (src, _, keys) in WS.items():
        for r in R8:
            j = load(a, r)
            rows.append(dict(arm=a, run=r, **{k: j["params"][k] for k in B9.SPACE[src]},
                             on_bound=int(sum(u < 0.02 or u > 0.98 for u in j["u"]))))
    pd.DataFrame(rows).to_csv(B3.LOGS / "stageB9b_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(S.round(4).to_string())
    print(pd.DataFrame(rows).round(3).to_string())


def all_():
    for step in (opt, test, report):
        log("=== step", step.__name__)
        step()
    log("=== B9b finished")


if __name__ == "__main__":
    {"opt": opt, "test": test, "report": report, "all": all_}[sys.argv[1]]()
