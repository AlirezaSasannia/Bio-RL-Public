"""Exploratory diagnostic after B9b (registered before running): does switching on online SPSA adaptation on top of
the constant offsets cost survival? metagrad with theta0 = bias_r offsets, c = 0.05, W = 50 and step size lr in
{0, 0.01, 0.03, 0.1, 0.3, 1} (lr = 0 is exactly the bias agent); 8 bias runs; development seeds only (schedules
99100-99227, training layout bank, simulator seed 99300 + r), no test set. Reports score, survival, wellbeing while alive.
Run: python b9_dose.py [--workers N]"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
import stage_b7 as B7
from _resume import cached_map
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R

SPEC = {"id": "single_b9_dose_exploratory", "status": "exploratory",
        "registered": "after B9b results, before any run of this diagnostic",
        "design": ("metagrad from bias_r offsets (r = 0-7), c 0.05, W 50, lr in {0, 0.01, 0.03, 0.1, 0.3, 1}; dev seeds "
                   "(schedules 99100-99227 x 4 agents, layouts from the training bank, simulator seed 99300 + r)"),
        "expectation": ("if online SPSA optimises short-term wellbeing at the cost of survival, survival and score fall "
                        "with lr while wellbeing while alive does not")}
LRS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


def run(job):
    r, lr = job
    b = B7.load("bias", r)["params"]
    prm = {**b, "m_a": b["b_a"], "m_e": b["b_e"], "m_s": b["b_s"], "b_a": 0.0, "b_e": 0.0, "b_s": 0.0,
           "mg_lr": lr, "mg_c": 0.05, "mg_W": 50.0}
    sch = [R.schedule(np.random.default_rng(99100 + i)) for i in range(128)]
    M = A.regime_matrix([s for s in sch for _ in range(4)], R.T_LIFE)
    lay = np.concatenate([(np.arange(4) * 37 + i * 11) % 2000 for i in range(128)])
    o = A.simulate("vdbe_metagrad", A.params(len(lay), **prm), M, B3.bank(), lay, seed=99_300 + r)
    s, sv = o["score"], o["survival"]
    return dict(run=r, lr=lr, score=s.mean(), survival=sv.mean(), alive_wellbeing=(s / np.maximum(sv, 1e-9)).mean())


if __name__ == "__main__":
    print("registered", register(SPEC)[:12], flush=True)
    jobs = [(r, lr) for r in range(8) for lr in LRS]
    with Pool(NWORK) as p:
        T = pd.DataFrame(cached_map(p, run, jobs, [f"dose_r{r}_lr{lr}" for r, lr in jobs], B3.LOGS / "b9_dose_cache"))
    T.to_csv(B3.LOGS / "b9_dose.csv", index=False)
    g = T.groupby("lr")[["score", "survival", "alive_wellbeing"]].mean()
    d = T.pivot(index="run", columns="lr", values="score")
    g["runs_below_lr0"] = [int((d[lr] < d[0.0]).sum()) for lr in g.index]
    print(g.round(4).to_string())
