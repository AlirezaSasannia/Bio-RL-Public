"""Stage B11 (registered after the B11 development runs, before any B11 confirmatory run): surprise split.
The generic layer's surprise input is split into a change (volatility) and a noise (stochasticity) component (Piray &
Daw 2021, Nat. Commun.: volatility should raise the learning rate, stochasticity lower it). Design chosen on development
seeds (b11_dev.py). Arm generic_split (6 inputs, 18 weights) is optimised on the 8 B10 bases with the B10 add-on budget
and the B10 generation seeds (common random numbers with the B10 generic and bias arms, whose B10 solutions are reused).
Run: python stage_b11.py [opt|test|all] [--workers N]"""
import json
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
import b11_dev as DEV
import stage_b3 as B3
import stage_b10 as B10
from _resume import cma_run
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space
from bio_rl.fast.popsim import LayoutBank

SPLIT = "split2"       # chosen on development seeds (decision log 2026-10-21)
NB = B10.NB
SPEC_B11 = {
    "id": "single_stageB11_surprise_split",
    "registered": "after the B11 development runs (dev seeds only), before any B11 confirmatory run",
    "question": ("Does splitting the surprise input into a change and a noise component improve the generic layer, "
                 "and does the layer learn the normative signs (change raises, noise lowers the learning rate)?"),
    "signal": {"split1": DEV.SPEC["candidates"]["split1"], "split2": DEV.SPEC["candidates"]["split2"]}[SPLIT],
    "scales": DEV.SPEC["scales"],
    "arms": {"generic_split": ("vdbe_generic with inputs threat, K, change, intake, energy deficit, noise; 18 weights "
                               "in (-3, 3), start u = 0.5; CMA seed 24000 + k, generation seeds 51000000 + 1000 k + g "
                               "(identical to the B10 arms of base k), 20 x 12 x 48, sigma 0.15, on B10 base k"),
             "generic": "B10 solution, reused (same budget and generation seeds)",
             "bias": "B10 solution, reused"},
    "test": ("new independent set, never used: 128 schedules (seeds 21000-21127) x 4 agents, layout bank "
             "2100000-2100999, simulator seeds 87000 + k"),
    "predictions": {"Q1": "generic_split > generic (score, one-sided)",
                    "Q2": "learned weight of change on log alpha (w_2_a) > 0 in at least 6 of 8 bases",
                    "Q3": "learned weight of noise on log alpha (w_5_a) < 0 in at least 6 of 8 bases"},
    "secondary": {"S1": "generic_split > bias (one-sided)", "S2": "generic > bias on the new set (one-sided)"},
    "inference": ("Q1, S1, S2 as B10: unit = base (n = 8), one-sided t-test (claim: p < 0.05 and three-level CI "
                  "excludes 0; Holm over S1-S2) plus exact Wilcoxon on base means and the schedule-level Wilcoxon "
                  "with hierarchical CI as checks. Q2/Q3: count of bases with the predicted sign; exact one-sided sign "
                  "test p reported (6/8 gives p = 0.145, so the count criterion is descriptive of consistency, not "
                  "significance). Weights within 1% of +-3 are reported."),
    "caveat": ("inputs are non-negative, so a weight mixes modulation with a constant offset; the per-base effective "
               "modulation w * sd(input over alive test steps) is reported beside the weights"),
    "interpretation": ("Q1 pass: the split carries usable information beyond the single surprise signal. Q2+Q3 pass "
                       "without Q1: the layer learns the normative direction but it does not improve the score. "
                       "No detectable difference is not equivalence."),
}
CTRL, SPACE, EXTRA = DEV.ARMS[SPLIT]
OUT = B3.LOGS / "stageB11_opt"
NS, NA_ = 128, 4
TB = None


def job(k):
    base = B10.load_base(k)
    make = lambda u: {**base, **EXTRA, **decode_space(SPACE, u)}
    res = cma_run(OUT / f"generic_split_r{k}.json", np.full(len(SPACE), 0.5), 0.15, B3.POP, 24000 + k, B3.GENS,
                  lambda X, g: B3.score(CTRL, [make(x) for x in X], 51_000_000 + 1000 * k + g),
                  make, dict(arm="generic_split", run=k, controller=CTRL))
    return "generic_split", k, res["history"][-1]


def opt():
    print("registered", register(SPEC_B11)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(min(NWORK, NB)) as p:
        for a, k, m in p.imap_unordered(job, [k for k in range(NB) if not (OUT / f"generic_split_r{k}.json").exists()]):
            print(a, k, round(m, 3), flush=True)


def load(arm, k):
    d = OUT if arm == "generic_split" else B10.OUT
    return json.loads((d / f"{arm}_r{k}.json").read_text())


def test_eval(args):
    ctrl, prm, tag, k = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(2_100_000, 2_101_000))
    sch = [R.schedule(np.random.default_rng(21000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    rec = ("wellbeing", "alive", "x2", "x5") if tag == "generic_split" else ("wellbeing", "alive")
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=87_000 + k, record=rec)
    df = B10.frame(o, M, tag, k)
    sd = {}
    if tag == "generic_split":
        al = o["alive"].astype(bool)
        sd = dict(sd_change=float(o["x2"][al].std()), sd_noise=float(o["x5"][al].std()),
                  mean_change=float(o["x2"][al].mean()), mean_noise=float(o["x5"][al].mean()))
    return df, sd


def test():
    print("registered", register(SPEC_B11)[:12], flush=True)
    jobs = [(load(a, k)["controller"], load(a, k)["params"], a, k)
            for a in ("generic_split", "generic", "bias") for k in range(NB)]
    with Pool(NWORK) as p:
        res = p.map(test_eval, jobs)
    D = pd.concat([r[0] for r in res])
    D.to_csv(B3.LOGS / "stageB11_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, [("Q1", "generic_split", "generic", "greater")], "primary"),
                   B10.family(piv, [("S1", "generic_split", "bias", "greater"), ("S2", "generic", "bias", "greater")],
                              "secondary")], ignore_index=True)
    T.to_csv(B3.LOGS / "stageB11_tests.csv", index=False)
    W = []
    for k in range(NB):
        p_ = load("generic_split", k)["params"]
        u = load("generic_split", k)["u"]
        sd = res[k][1]
        W.append(dict(base=k, w_change_a=p_["w_2_a"], w_noise_a=p_["w_5_a"], w_change_e=p_["w_2_e"],
                      w_noise_e=p_["w_5_e"], w_change_s=p_["w_2_s"], w_noise_s=p_["w_5_s"],
                      eff_change_a=p_["w_2_a"] * sd["sd_change"], eff_noise_a=p_["w_5_a"] * sd["sd_noise"],
                      n_on_bound=int(sum(x < 0.01 or x > 0.99 for x in u)), **sd))
    W = pd.DataFrame(W)
    W.to_csv(B3.LOGS / "stageB11_weights.csv", index=False)
    n2, n3 = int((W.w_change_a > 0).sum()), int((W.w_noise_a < 0).sum())
    Wt = pd.DataFrame([dict(pred="Q2", bases_predicted_sign=n2, n=NB, p_sign=stats.binomtest(n2, NB, 0.5, "greater").pvalue,
                            verdict="PASS" if n2 >= 6 else "FAIL"),
                       dict(pred="Q3", bases_predicted_sign=n3, n=NB, p_sign=stats.binomtest(n3, NB, 0.5, "greater").pvalue,
                            verdict="PASS" if n3 >= 6 else "FAIL")])
    Wt.to_csv(B3.LOGS / "stageB11_weight_tests.csv", index=False)
    S = D.groupby("arm").mean(numeric_only=True).drop(columns=["run", "schedule"])
    S.to_csv(B3.LOGS / "stageB11_summary.csv")
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(Wt.round(4).to_string())
    print(W.round(3).to_string())
    print(S[["score", "survival", "alive_wellbeing"] + [f"death_{r}" for r in R.NAMES]].round(3).to_string())


if __name__ == "__main__":
    {"opt": opt, "test": test, "all": lambda: (opt(), test())}[sys.argv[1]]()
