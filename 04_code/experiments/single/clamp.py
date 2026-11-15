"""Clamp test (registered as exploratory follow-up of the mechanism analysis, before running): hold one input of the
frozen generic layers (8 runs) at its life-average value and measure the score per regime on the B5 independent set.
Expectations: clamping surprise or energy deficit costs most in the volatile regime (where the mechanism analysis
locates their joint effect on learning rate); clamping threat or intake costs little."""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

SPEC = {"id": "single_clamp_exploratory", "registered": "after mechanism analysis, before any clamp run",
        "design": "generic layers r0-7; each input clamped to its mean over the B5 test lives (from mechanism/per_regime)",
        "expectation": "surprise and deficit clamps cost most in volatile; threat and intake clamps cost little",
        "status": "exploratory"}
INP = ["threat", "K", "surprise", "intake", "deficit"]
NS, NA_ = 128, 4
TB = None


def run(args):
    r, ci, cv = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(900000, 901000))
    j = json.loads((B3.OUT / f"generic_r{r}.json").read_text())
    sch = [R.schedule(np.random.default_rng(6000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate("vdbe_generic", A.params(len(lay), **j["params"], clamp_i=ci, clamp_v=cv), M, TB, lay,
                   seed=61_000 + r, record=("wellbeing",))
    wb = o["wellbeing"]
    row = dict(run=r, clamp=INP[ci] if ci >= 0 else "none", score=o["score"].mean(), survival=o["survival"].mean())
    for g, rn in enumerate(R.NAMES):
        m = M == g
        row[rn] = float((wb * m).sum() / m.sum())
    return row


if __name__ == "__main__":
    print("registered", register(SPEC)[:12], flush=True)
    D = pd.read_csv(B3.LOGS / "mechanism" / "per_regime.csv")
    G = D[D.arm == "generic"]
    means = [float(np.average(G[f"x{i}"], weights=G.n)) for i in range(5)]
    print("clamp values", np.round(means, 3))
    jobs = [(r, -1, 0.0) for r in range(8)] + [(r, i, means[i]) for r in range(8) for i in range(5)]
    with Pool(NWORK) as p:
        T = pd.DataFrame(p.map(run, jobs))
    T.to_csv(B3.LOGS / "clamp_results.csv", index=False)
    base = T[T.clamp == "none"].set_index("run")
    rows = []
    for c in INP:
        X = T[T.clamp == c].set_index("run")
        d = X[["score"] + R.NAMES] - base[["score"] + R.NAMES]
        rows.append(dict(clamp=c, **d.mean().round(4).to_dict(), runs_worse=int((d.score < 0).sum())))
    S = pd.DataFrame(rows)
    S.to_csv(B3.LOGS / "clamp_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.to_string())
