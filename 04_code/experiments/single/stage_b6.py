"""Stage B6 (registered after review of v0.3): adaptation speed from matched states, survival removed.
Every frozen agent lives 1500 steps in the safe regime (same layouts, same noise for all arms), then the hidden
regime switches to X and stays for 1000 steps. Death is disabled throughout, so the score measures how well and
how fast the agent re-adapts, not whether it survives. X in {danger, uncontrollable, scarce, volatile}; plus the
reverse switch danger -> safe.
Metric: mean wellbeing in post-switch windows [0,100), [100,300), [300,1000); pre-switch level (last 200 steps)
reported to check matched starts. Prediction (one-sided, Holm over the 5 switches, window [0,300)):
F1 generic > control, F2 bio > control. Secondary: all other arms vs control, per window."""
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
import stage_b3 as B3
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

SPEC_B6 = {"id": "single_stageB6_adaptation_speed", "registered": "after review of v0.3, before any B6 run",
           "protocol": "1500 steps safe, then regime X for 1000 steps, death disabled; 128 layouts x 4 agents",
           "switches": ["safe->danger", "safe->uncontrollable", "safe->scarce", "safe->volatile", "danger->safe"],
           "windows": [[0, 100], [100, 300], [300, 1000]],
           "predictions": {"F1": "generic > control in [0,300)", "F2": "bio > control in [0,300)"},
           "inference": "one-sided Wilcoxon over layouts of the run-averaged difference, Holm over 5 switches per "
                        "prediction; hierarchical CI (runs, layouts)"}
PRE, POST, NL, NA_ = 1500, 1000, 128, 4
SW = [("safe", "danger"), ("safe", "uncontrollable"), ("safe", "scarce"), ("safe", "volatile"), ("danger", "safe")]
WIN = [(0, 100), (100, 300), (300, 1000), (0, 300)]
TB = None


def evaluate(args):
    ctrl, prm, tag, r, (a, b) = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(950000, 951000))
    B = NL * NA_
    M = np.zeros((PRE + POST, B), int)
    M[:PRE], M[PRE:] = R.NAMES.index(a), R.NAMES.index(b)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NL)])
    o = A.simulate(ctrl, A.params(B, **prm), M, TB, lay, seed=71_000 + r, juvenile=PRE + POST + 1,
                   record=("wellbeing",))
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(layout=np.repeat(np.arange(NL), NA_), pre=wb[PRE - 200:PRE].mean(0)))
    for lo, hi in WIN:
        df[f"w{lo}_{hi}"] = wb[PRE + lo:PRE + hi].mean(0)
    df["arm"], df["run"], df["switch"] = tag, r, f"{a}->{b}"
    curve = wb[PRE - 100:].reshape(-1, 10, B).mean(1).mean(1)          # 10-step bins, mean over agents
    return df, (tag, r, f"{a}->{b}", curve)


def jobs(arms):
    out = []
    for arm, d in arms.items():
        for r in range(4):
            j = json.loads((d / f"{arm}_r{r}.json").read_text())
            for s in SW:
                out.append((j["controller"], j["params"], arm, r, s))
    return out


def main():
    print("registered", register(SPEC_B6)[:12], flush=True)
    arms = {"control": B3.OUT, "bio": B3.OUT, "generic": B3.OUT}
    b5 = B3.LOGS / "stageB5_opt"
    arms.update({a: b5 for a in ("yoked", "augtab", "auglin") if (b5 / f"{a}_r3.json").exists()})
    with Pool(NWORK) as p:
        res = p.map(evaluate, jobs(arms))
    D = pd.concat([r[0] for r in res])
    D.to_csv(B3.LOGS / "stageB6_raw.csv", index=False)
    C = pd.DataFrame([dict(arm=t, run=r, switch=s, **{f"b{i}": v for i, v in enumerate(c)}) for t, r, s, c in
                      (x[1] for x in res)])
    C.to_csv(B3.LOGS / "stageB6_curves.csv", index=False)
    rows = []
    for s in D.switch.unique():
        X = D[D.switch == s]
        for col in ["pre"] + [f"w{lo}_{hi}" for lo, hi in WIN]:
            piv = X.groupby(["arm", "run", "layout"])[col].mean()
            for arm in arms:
                if arm == "control":
                    continue
                d = (piv[arm] - piv["control"]).unstack()
                lo_, hi_ = B3.hier_ci(d.values, n=2000)
                rows.append(dict(switch=s, window=col, comparison=f"{arm} - control", diff=d.values.mean(),
                                 ci_lo=lo_, ci_hi=hi_,
                                 p=float(stats.wilcoxon(d.mean(0).values, alternative="greater").pvalue)))
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    for arm in ("generic", "bio"):
        idx = T.index[(T.window == "w0_300") & (T.comparison == f"{arm} - control")]
        adj = holm({i: T.loc[i, "p"] for i in idx})
        for i in idx:
            T.loc[i, "p_holm"] = adj[i]
    T.to_csv(B3.LOGS / "stageB6_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[T.window.isin(["pre", "w0_300", "w300_1000"])].round(4).to_string())
    print(D.groupby(["switch", "arm"])[["pre", "w0_100", "w100_300", "w300_1000"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
