"""Exploratory mechanism analysis (not a registered test): what the global layers do.
Records inputs and knob offsets of the frozen generic and biological layers on the B5 independent test set and in
the B6 switch protocol, and decomposes the generic layer's knob offsets into input contributions per regime."""
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
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

INP = ["threat", "K", "surprise", "intake", "deficit"]
KN = ["a", "e", "s"]
REC = ("alive", "alpha", "eps", "safety_mult", "ea", "ee", "es") + tuple(f"x{i}" for i in range(5))
OUT = B3.LOGS / "mechanism"
NS, NA_ = 128, 4
TB = None


def run(args):
    arm, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(900000, 901000))
    j = json.loads((B3.OUT / f"{arm}_r{r}.json").read_text())
    sch = [R.schedule(np.random.default_rng(6000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(j["controller"], A.params(len(lay), **j["params"]), M, TB, lay, seed=61_000 + r, record=REC)
    al = o["alive"].astype(bool) & (np.arange(R.T_LIFE)[:, None] >= R.JUVENILE)
    rows = []
    for g, rn in enumerate(R.NAMES):
        m = al & (M == g)
        row = dict(arm=arm, run=r, regime=rn, n=int(m.sum()))
        for k in REC[1:]:
            row[k] = float(o[k][m].mean())
        rows.append(row)
    # event-triggered average around layout changes inside the volatile regime (every 300 steps)
    vol = R.NAMES.index("volatile")
    ev = []
    for t in range(600, R.T_LIFE - 200, 300):
        m = (M[t] == vol) & (M[t - 100] == vol) & (M[t + 199] == vol) & al[t - 100]
        if m.any():
            ev.append(np.stack([o[k][t - 100:t + 200, m].mean(1) for k in ("ea", "ee", "es", "x2")], 0))
    eta = np.mean(ev, 0) if ev else np.full((4, 300), np.nan)
    return pd.DataFrame(rows), (arm, r, eta)


def main(runs=range(4)):
    OUT.mkdir(exist_ok=True)
    jobs = [(a, r) for a in ("generic", "bio") for r in runs]
    with Pool(NWORK) as p:
        res = p.map(run, jobs)
    D = pd.concat([x[0] for x in res])
    D.to_csv(OUT / "per_regime.csv", index=False)
    E = pd.DataFrame([dict(arm=a, run=r, signal=s, **{f"t{t - 100}": v for t, v in enumerate(eta[i])})
                      for a, r, eta in (x[1] for x in res) for i, s in enumerate(("ea", "ee", "es", "x2"))])
    E.to_csv(OUT / "volatile_events.csv", index=False)
    # generic weights and per-regime contributions W[k,i] * mean x_i
    W, C = [], []
    for r in runs:
        p = json.loads((B3.OUT / f"generic_r{r}.json").read_text())["params"]
        for i, nm in enumerate(INP):
            W.append(dict(run=r, input=nm, **{k: p[f"w_{i}_{k}"] for k in KN}))
            for _, row in D[(D.arm == "generic") & (D.run == r)].iterrows():
                C.append(dict(run=r, regime=row.regime, input=nm,
                              **{k: p[f"w_{i}_{k}"] * row[f"x{i}"] for k in KN}))
    W, C = pd.DataFrame(W), pd.DataFrame(C)
    W.to_csv(OUT / "generic_weights.csv", index=False)
    C.to_csv(OUT / "generic_contrib.csv", index=False)
    pd.set_option("display.width", 220)
    print(D.groupby(["arm", "regime"])[["alpha", "eps", "safety_mult", "ea", "ee", "es"]].mean().round(3).to_string())
    print(D[D.arm == "generic"].groupby("regime")[[f"x{i}" for i in range(5)]].mean().round(3).to_string())
    print(W.groupby("input")[KN].agg(["mean", lambda x: int((x > 0).sum())]).round(2).to_string())
    print(C.groupby(["regime", "input"])[KN].mean().round(2).unstack("input").to_string())


if __name__ == "__main__":
    main()
