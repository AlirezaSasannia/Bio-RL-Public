"""Stage B7 (registered after review #2 of v0.3, before any B7 run): more optimisation runs per arm.
Runs 4-7 for the six key arms (control, bio, generic, yoked, augtab, auglin), each on VDBE base r % 4 with new CMA
seeds and new training schedules; identical protocol to B3/B5. All 8 runs re-tested on the B5 independent test set
and in the B6 adaptation-speed protocol. Inference as before, now with 8 runs in the hierarchical bootstrap.
Run: python stage_b7.py [opt|test|speed]"""
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
import stage_b5 as B5
import stage_b6 as B6
import stage_b8 as B8
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register

SPEC_B7 = {"id": "single_stageB7_more_runs", "registered": "after review #2 of v0.3, before any B7 run",
           "change": "runs 4-7 for control, bio, generic (B3 protocol) and yoked, augtab, auglin (B5 protocol)",
           "base": "VDBE base r % 4; CMA seeds and training schedules new (seed formulas of B3/B5 with r = 4..7)",
           "test": "B5 independent set and B6 speed protocol, all 8 runs",
           "claims": "B5 primary E1-E3 and B3/B5 secondary comparisons re-estimated with 8 runs; same criteria"}
SPEC_B7B = {"id": "single_stageB7_addendum_bias", "amends": "single_stageB7_more_runs",
            "when": "after B8 results on 4 runs, before any B7 test",
            "change": "add the B8 bias arm (runs 4-7, B8 protocol) to the 8-run tests; H1 generic > bias re-estimated"}
ARMS = {"control": "b3", "bio": "b3", "generic": "b3", "yoked": "b5", "augtab": "b5", "auglin": "b5", "bias": "b8"}
DIR = {"b3": B3.OUT, "b5": B5.OUT, "b8": B8.OUT}
R8 = range(8)


def _job(j):
    arm, r = j
    return {"b3": B3.run, "b5": B5.run, "b8": B8.run}[ARMS[arm]](j)


def opt():
    print("registered", register(SPEC_B7)[:12], register(SPEC_B7B)[:12], flush=True)
    jobs = [(a, r) for r in range(4, 8) for a in ARMS if not (DIR[ARMS[a]] / f"{a}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for a, r, m in p.imap_unordered(_job, jobs):
            print(a, r, round(m, 3), flush=True)


def load(arm, r):
    return json.loads((DIR[ARMS[arm]] / f"{arm}_r{r}.json").read_text())


def test():
    jobs = [(load(a, r)["controller"], load(a, r)["params"], a, r) for a in ARMS for r in R8]
    with Pool(NWORK) as p:
        D = pd.concat(p.map(B5.evaluate, jobs))
    D.to_csv(B3.LOGS / "stageB7_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.DataFrame([B5.comp(piv, "generic", b) for b in ("yoked", "augtab", "auglin", "bias")])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["family"] = "primary (E1-E3, H1; 8 runs)"
    sec = pd.DataFrame([B5.comp(piv, "bio", "control"), B5.comp(piv, "generic", "bio"),
                        B5.comp(piv, "generic", "control"), B5.comp(piv, "yoked", "control", "two-sided"),
                        B5.comp(piv, "augtab", "control", "two-sided"), B5.comp(piv, "auglin", "control", "two-sided"),
                        B5.comp(piv, "bias", "control", "two-sided"), B5.comp(piv, "bio", "bias", "two-sided")])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(B3.LOGS / "stageB7_tests.csv", index=False)
    S = D.groupby("arm")[["score", "survival", "alive_wellbeing"]].mean()
    S.to_csv(B3.LOGS / "stageB7_summary.csv")
    pd.set_option("display.width", 220)
    print(T.round(4).to_string())
    print(S.sort_values("score").round(3).to_string())


def speed():
    jobs = [(load(a, r)["controller"], load(a, r)["params"], a, r, s) for a in ARMS for r in R8 for s in B6.SW]
    with Pool(NWORK) as p:
        res = p.map(B6.evaluate, jobs)
    D = pd.concat([x[0] for x in res])
    D.to_csv(B3.LOGS / "stageB7_speed_raw.csv", index=False)
    rows = []
    for s in D.switch.unique():
        X = D[D.switch == s]
        for col in ("pre", "w0_100", "w100_300", "w0_300", "w300_1000"):
            pv = X.groupby(["arm", "run", "layout"])[col].mean()
            for a in ARMS:
                if a != "control":
                    d = (pv[a] - pv["control"]).unstack()
                    lo, hi = B3.hier_ci(d.values, n=2000)
                    rows.append(dict(switch=s, window=col, comparison=f"{a} - control", diff=d.values.mean(),
                                     ci_lo=lo, ci_hi=hi))
    T = pd.DataFrame(rows)
    T.to_csv(B3.LOGS / "stageB7_speed_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[T.window.isin(["w0_300", "w300_1000"]) & T.comparison.isin(["generic - control", "bio - control"])]
          .round(4).to_string())


if __name__ == "__main__":
    {"opt": opt, "test": test, "speed": speed}[sys.argv[1]]()
