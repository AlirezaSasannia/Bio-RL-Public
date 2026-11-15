"""Stage B4 (registered after B3): which appraisal signals carry the generic controller's gain?
Arms on the same frozen VDBE bases, same budget as B2/B3:
  surprise_only  - generic linear controller fed only the surprise signal (the signal standard adaptive
                   methods use); 3 parameters
  appraisal_only - fed threat, controllability, intake and energy deficit but not surprise; 12 parameters
Compared with the B3 generic (all 5 inputs) and control arms on the B3 fresh test set.
Prediction (one-sided, Holm across 2): generic(all) > surprise_only and generic(all) > appraisal_only."""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast.adaptive_search import decode_space

SPEC_B4 = {"id": "single_stageB4_inputs", "registered": "after B3 results, before any B4 run",
           "arms": {"surprise_only": "inputs: surprise", "appraisal_only": "inputs: threat, K, intake, energy deficit"},
           "budget": "as B3 (20 x 12 x 48, sigma 0.15, zero-effect start), 4 runs on the 4 VDBE bases",
           "test": "B3 fresh test set (128 schedules x 4 agents)",
           "predictions": {"D1": "generic(all inputs) > surprise_only", "D2": "generic(all inputs) > appraisal_only"},
           "inference": "as B3 amendment: Wilcoxon over schedules + hierarchical CI, Holm across D1-D2"}
SUB = {"surprise_only": [2], "appraisal_only": [0, 1, 3, 4]}
OUT = B3.OUT


def run(job):
    arm, r = job
    space = {f"w_{i}_{k}": (-3.0, 3.0, "lin") for i in SUB[arm] for k in ("a", "e", "s")}
    base = json.loads((B3.LOGS / "stageB_opt" / f"vdbe_r{r}.json").read_text())["params"]
    make = lambda u: {**base, **decode_space(space, u)}
    es = cma.CMAEvolutionStrategy(np.full(len(space), 0.5), 0.15,
                                  {"bounds": [0, 1], "popsize": B3.POP, "seed": 9000 + r, "verbose": -9})
    hist = []
    for g in range(B3.GENS):
        X = es.ask()
        f = B3.score("vdbe_generic", [make(x) for x in X], 18_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    (OUT / f"{arm}_r{r}.json").write_text(json.dumps(dict(arm=arm, run=r, controller="vdbe_generic",
                                                          params=make(es.mean), history=hist), indent=1))
    return arm, r, hist[-1]


if __name__ == "__main__":
    print("registered", register(SPEC_B4)[:12], flush=True)
    jobs = [(a, r) for r in range(4) for a in SUB if not (OUT / f"{a}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for a, r, m in p.imap_unordered(run, jobs):
            print(a, r, round(m, 3), flush=True)
    ev = [("main", "vdbe_generic", json.loads((OUT / f"{a}_r{r}.json").read_text())["params"], a, r)
          for a in SUB for r in range(4)]
    with Pool(NWORK) as p:
        D = pd.concat(p.map(B3.evaluate, ev))
    old = pd.read_csv(B3.LOGS / "stageB3_test_raw.csv")
    D = pd.concat([old[(old.set == "main") & old.arm.isin(["generic", "control", "bio"])], D])
    D.to_csv(B3.LOGS / "stageB4_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    rows = []
    for other in SUB:
        d = (piv["generic"] - piv[other]).unstack()
        lo, hi = B3.hier_ci(d.values)
        rows.append(dict(comparison=f"generic - {other}", diff=d.values.mean(), ci_lo=lo, ci_hi=hi,
                         runs_better=int((d.mean(axis=1) > 0).sum()),
                         p=float(stats.wilcoxon(d.mean(axis=0).values, alternative="greater").pvalue)))
    for other in SUB:
        d = (piv[other] - piv["control"]).unstack()
        lo, hi = B3.hier_ci(d.values)
        rows.append(dict(comparison=f"{other} - control", diff=d.values.mean(), ci_lo=lo, ci_hi=hi,
                         runs_better=int((d.mean(axis=1) > 0).sum()), p=np.nan))
    T = pd.DataFrame(rows)
    adj = holm({i: T.loc[i, "p"] for i in T.index[:2]})
    T["p_holm"] = [adj.get(i, np.nan) for i in T.index]
    T.to_csv(B3.LOGS / "stageB4_tests.csv", index=False)
    print(D.groupby("arm")[["score", "survival", "alive_wellbeing"]].mean().round(3).to_string())
    print(T.round(4).to_string())
