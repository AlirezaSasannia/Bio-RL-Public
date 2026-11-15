"""Stage A: how much can regime-dependent modulation gain at most? (see spec_a.py)

Run:  python experiments/single/stage_a_headroom.py
"""
import itertools
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
from bio_rl.battery.prereg import register
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank, param_arrays, simulate
from spec_a import SPEC_A

LOGS = ROOT / "results" / "logs" / "single"
GRID = list(itertools.product([0.1, 0.3, 0.6, 0.9], [0.02, 0.1, 0.2, 0.35, 0.5], [0.5, 1.0, 2.0, 4.0]))
KW = dict(grace=100, injury_grace=1, social_need=False)
_banks = {}


def bank(start):
    if start not in _banks:
        _banks[start] = LayoutBank.build(range(start, start + 1000))
    return _banks[start]


def params_for(configs, reps):
    a = np.repeat([c[0] for c in configs], reps)
    e = np.repeat([c[1] for c in configs], reps)
    s = np.repeat([c[2] for c in configs], reps)
    return param_arrays(len(a), alpha=a, epsilon=e, safety_mult=s)


def stationary(regime):
    reps = 64
    B = len(GRID) * reps
    lay = np.tile(np.arange(reps) * 13 % 1000, len(GRID))
    out = simulate("needs", params_for(GRID, reps), R.eco(regime, B), bank(400000), lay, 3000,
                   seed=50 + R.NAMES.index(regime), **KW)
    return regime, out["score"].reshape(len(GRID), reps).mean(1)


def switching(sch_seed):
    reps = 4
    B = len(GRID) * reps
    sched = R.schedule(np.random.default_rng(sch_seed))
    lay = np.tile((np.arange(reps) * 31 + sch_seed * 7) % 1000, len(GRID))
    out = simulate("needs", params_for(GRID, reps), R.eco("safe", B), bank(400000), lay, R.T_LIFE,
                   seed=100 + sch_seed, phases=R.phases(sched, B), **KW)
    return out["score"].reshape(len(GRID), reps).mean(1)


def test_run(args):
    """One test schedule: best-fixed, oracle and floor agents on identical layouts and seeds."""
    sch_seed, fixed, oracle = args
    reps = 8
    sched = R.schedule(np.random.default_rng(sch_seed))
    lay = (np.arange(reps) * 37 + sch_seed * 11) % 1000
    rec = ("alive", "energy", "safety")
    res = {}
    agents = {"best_fixed": param_arrays(reps, alpha=fixed[0], epsilon=fixed[1], safety_mult=fixed[2]),
              "floor": param_arrays(reps, alpha=0.0, epsilon=0.0)}
    for name, P in agents.items():
        res[name] = simulate("needs", P, R.eco("safe", reps), bank(700000), lay, R.T_LIFE, seed=9000 + sch_seed,
                             phases=R.phases(sched, reps), record=rec, **KW)
    Pr = {k: param_arrays(reps, alpha=v[0], epsilon=v[1], safety_mult=v[2]) for k, v in oracle.items()}
    res["oracle"] = simulate("needs", Pr["safe"], R.eco("safe", reps), bank(700000), lay, R.T_LIFE,
                             seed=9000 + sch_seed, phases=R.phases(sched, reps, Pr), record=rec, **KW)
    reg = R.regime_trace(sched)
    rows = []
    for name, o in res.items():
        contrib = o["alive"] * (1.0 - ((1 - o["energy"]) ** 2 + (1 - o["safety"]) ** 2) / 2.0)
        for i in range(reps):
            row = dict(schedule=sch_seed, agent=i, kind=name, score=float(o["score"][i]),
                       survival=float(o["survival"][i]))
            for j, rn in enumerate(R.NAMES):
                m = reg == j
                row[f"score_{rn}"] = float(contrib[m, i].mean()) if m.any() else np.nan
            rows.append(row)
    return rows


if __name__ == "__main__":
    print("registered", register(SPEC_A)[:12], flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with Pool(NWORK) as p:
        st = dict(p.map(stationary, R.NAMES))
        sw = np.mean(p.map(switching, range(16)), axis=0)
    grid = pd.DataFrame(GRID, columns=["alpha", "epsilon", "safety_mult"])
    for r in R.NAMES:
        grid[f"stationary_{r}"] = st[r]
    grid["switching"] = sw
    grid.to_csv(LOGS / "stageA_grid.csv", index=False)
    fixed = GRID[int(np.argmax(sw))]
    oracle = {r: GRID[int(np.argmax(st[r]))] for r in R.NAMES}
    json.dump(dict(best_fixed=fixed, oracle=oracle), open(LOGS / "stageA_choices.json", "w"), indent=1)
    print("best fixed", fixed, "oracle", oracle, flush=True)

    with Pool(NWORK) as p:
        rows = [r for rr in p.map(test_run, [(s, fixed, oracle) for s in range(1000, 1064)]) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(LOGS / "stageA_test_raw.csv", index=False)
    per = df.groupby(["schedule", "kind"]).score.mean().unstack()
    d = (per["oracle"] - per["best_fixed"]).values
    boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rel = d.mean() / (per["oracle"].mean() - per["floor"].mean())
    summary = dict(score_oracle=per["oracle"].mean(), score_best_fixed=per["best_fixed"].mean(),
                   score_floor=per["floor"].mean(), headroom=d.mean(), ci_lo=lo, ci_hi=hi, relative_headroom=rel,
                   decision="GO to stage B" if (lo > 0 and rel >= 0.10) else "REDESIGN")
    for rn in R.NAMES:
        pr = df.groupby(["schedule", "kind"])[f"score_{rn}"].mean().unstack()
        summary[f"headroom_{rn}"] = float((pr["oracle"] - pr["best_fixed"]).mean())
    summary.update({f"survival_{k}": float(v) for k, v in df.groupby("kind").survival.mean().items()})
    pd.Series(summary).to_csv(LOGS / "stageA_summary.csv")
    print(pd.Series(summary).to_string())
