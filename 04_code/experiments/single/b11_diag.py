"""Stage B11 development diagnostic (part of single_stageB11_dev_exploratory; development seeds only: schedules
98000-98031, layouts 1900000-1900399, simulator 98500 + r): behaviour of the candidate change / noise signals with a
zero-weight generic layer (knobs unaffected) on the stage-B VDBE bases r0-3. For each signal, the mean over alive
steps by regime, early (< 50 steps after a regime or layout switch) and late (>= 200 steps).
The first run of this diagnostic (2026-10-21, before the dev CMA) used unit input scales; this script uses the
registered scales, which rescale each signal by a constant and leave the regime comparisons unchanged up to clipping.
Run: python b11_diag.py [--workers N]"""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

LOGS = ROOT / "results" / "logs" / "single"
NS, NA_ = 32, 4
CASES = [(0, 50), (1, 20), (2, 50)]          # (split, tau): 0 = the single surprise signal


def run(job):
    split, tau, r = job
    TB = LayoutBank.build(range(1_900_000, 1_900_400))
    sch = [R.schedule(np.random.default_rng(98000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 400 for i in range(NS)])
    T, B = M.shape
    since = np.zeros((T, B), int)
    for t in range(1, T):
        sw = (M[t] != M[t - 1]) | ((M[t] == R.NAMES.index("volatile")) & (t % 300 == 0))
        since[t] = np.where(sw, 0, since[t - 1] + 1)
    base = json.loads((LOGS / "stageB_opt" / f"vdbe_r{r}.json").read_text())["params"]
    o = A.simulate("vdbe_generic", A.params(B, **base, split=split, tau_split=tau), M, TB, lay, seed=98_500 + r,
                   record=("alive", "x2", "x5") if split else ("alive", "x2"))
    al = o["alive"].astype(bool)
    sigs = {"surprise": o["x2"]} if not split else {"change": o["x2"], "noise": o["x5"]}
    rows = []
    for nm, x in sigs.items():
        row = dict(split=split, tau=tau, base=r, signal=nm, mean=x[al].mean())
        for g, rn in enumerate(R.NAMES):
            m = al & (M == g)
            row[f"{rn}_early"] = x[m & (since < 50)].mean()
            row[f"{rn}_late"] = x[m & (since >= 200)].mean()
            row[f"{rn}_all"] = x[m].mean()
        rows.append(row)
    if split:
        rows.append(dict(split=split, tau=tau, base=r, signal="corr(change, noise)",
                         mean=np.corrcoef(o["x2"][al], o["x5"][al])[0, 1]))
    return rows


if __name__ == "__main__":
    with Pool(NWORK) as p:
        res = p.map(run, [(s, t, r) for s, t in CASES for r in range(4)])
    D = pd.DataFrame([row for rows in res for row in rows])
    out = LOGS / "stageB11_dev" / "diag_signals.csv"
    D.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(D.groupby(["split", "tau", "signal"]).mean(numeric_only=True).drop(columns="base").round(3).to_string())
