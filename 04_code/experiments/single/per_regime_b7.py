"""Per-regime scores and survival on the B5 independent set for control, bias, bio and generic (8 runs each);
descriptive follow-up of B7/B8: where does reading the signals add to constant offsets?"""
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
import stage_b7 as B7
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

NS, NA_ = 128, 4
TB = None


def run(args):
    arm, r = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(900000, 901000))
    j = B7.load(arm, r)
    sch = [R.schedule(np.random.default_rng(6000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(j["controller"], A.params(len(lay), **j["params"]), M, TB, lay, seed=61_000 + r,
                   record=("wellbeing", "alive"))
    wb, al = o["wellbeing"], o["alive"].astype(bool)
    row = dict(arm=arm, run=r, score=o["score"].mean(), survival=o["survival"].mean())
    # deaths by the regime in which they happened, per agent-life
    died = al[:-1] & ~al[1:]
    for g, rn in enumerate(R.NAMES):
        m = M == g
        row[f"score_{rn}"] = float((wb * m).sum() / m.sum())
        row[f"deaths_{rn}"] = float((died & m[1:]).sum() / al.shape[1])
        alive_m = m & al
        row[f"alivewb_{rn}"] = float(wb[alive_m].mean())
    return row


if __name__ == "__main__":
    jobs = [(a, r) for a in ("control", "bias", "bio", "generic") for r in range(8)]
    with Pool(NWORK) as p:
        T = pd.DataFrame(p.map(run, jobs))
    T.to_csv(B3.LOGS / "per_regime_b7.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.groupby("arm").mean(numeric_only=True).drop(columns="run").round(4).T.to_string())
