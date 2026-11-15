"""Stage B8 (registered before any B8 run, after the exploratory mechanism analysis showed that the stage-B VDBE bases
sit on search bounds (alpha0 = 0.02 lower bound in 4/4, s0 = 0.25 lower bound in 3/4) and that the generic layer
holds large, almost constant knob offsets (safety weight x0.02-0.12, exploration -3.4 logits)).
Question: is the global layer's gain just a constant shift of the operating point beyond the base's search bounds?
Arms (frozen VDBE bases r % 4, budget 20 x 12 x 48, sigma 0.15, 4 runs):
  bias  the generic layer with no inputs: three constant knob offsets (same mapping, range +/-6), start 0
  wide  no module: re-tune the base (alpha0, eps0, s0, vdbe_sigma) with widened bounds, start at the base
Predictions (one-sided, Holm over 2, on the B5 independent set): H1 generic > bias, H2 generic > wide.
Run: python stage_b8.py [opt|test]"""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import cma
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
import stage_b5 as B5
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast.adaptive_search import decode_space, encode

SPEC_B8 = {"id": "single_stageB8_constant_shift", "registered": "after exploratory mechanism analysis, before any B8 run",
           "observation": ("VDBE bases on search bounds (alpha0 0.02 in 4/4; s0 0.25 in 3/4); generic layer holds large "
                           "near-constant offsets (safety weight x0.02-0.12; exploration -3.4 logits)"),
           "arms": {"bias": "generic layer without inputs: 3 constant knob offsets in (-6, 6), start 0",
                    "wide": ("re-tune base with widened bounds: alpha0 (0.002, 0.9) log, eps0 (0, 0.5), s0 (0.005, 8) "
                             "log, vdbe_sigma (0.001, 100) log; start at the base")},
           "budget": "20 x 12 x 48, sigma 0.15, 4 runs on VDBE bases r % 4",
           "predictions": {"H1": "generic > bias", "H2": "generic > wide"},
           "inference": "B5 independent set; one-sided Wilcoxon over schedules + hierarchical CI; Holm over H1-H2",
           "interpretation": ("if bias or wide match generic, the add-on gain is mainly a constant shift of the operating "
                              "point that the base search could not reach, not input-dependent modulation")}
SPACE = {"bias": {k: (-6.0, 6.0, "lin") for k in ("b_a", "b_e", "b_s")},
         "wide": {"alpha0": (0.002, 0.9, "log"), "eps0": (0.0, 0.5, "lin"), "s0": (0.005, 8.0, "log"),
                  "vdbe_sigma": (0.001, 100.0, "log")}}
CTRL = {"bias": "vdbe_generic", "wide": "vdbe"}
OUT = B3.LOGS / "stageB8_opt"


def run(job):
    arm, r = job
    space = SPACE[arm]
    base = json.loads((B3.LOGS / "stageB_opt" / f"vdbe_r{r % 4}.json").read_text())["params"]
    make = lambda u: {**base, **decode_space(space, u)}
    x0 = encode(space, base) if arm == "wide" else np.full(len(space), 0.5)
    es = cma.CMAEvolutionStrategy(x0, 0.15, {"bounds": [0, 1], "popsize": B3.POP, "seed": 9800 + r, "verbose": -9})
    hist = []
    for g in range(B3.GENS):
        X = es.ask()
        f = B3.score(CTRL[arm], [make(x) for x in X], 20_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    (OUT / f"{arm}_r{r}.json").write_text(json.dumps(dict(arm=arm, run=r, controller=CTRL[arm], params=make(es.mean),
                                                          history=hist), indent=1))
    return arm, r, hist[-1]


def opt():
    print("registered", register(SPEC_B8)[:12], flush=True)
    OUT.mkdir(exist_ok=True)
    jobs = [(a, r) for r in range(4) for a in SPACE if not (OUT / f"{a}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for a, r, m in p.imap_unordered(run, jobs):
            print(a, r, round(m, 3), flush=True)


def test():
    jobs = []
    for a, d in (("generic", B3.OUT), ("control", B3.OUT), ("bio", B3.OUT), ("bias", OUT), ("wide", OUT)):
        for r in range(4):
            j = json.loads((d / f"{a}_r{r}.json").read_text())
            jobs.append((j["controller"], j["params"], a, r))
    with Pool(NWORK) as p:
        D = pd.concat(p.map(B5.evaluate, jobs))
    D.to_csv(B3.LOGS / "stageB8_test_raw.csv", index=False)
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.DataFrame([B5.comp(piv, "generic", "bias"), B5.comp(piv, "generic", "wide")])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["family"] = "primary"
    sec = pd.DataFrame([B5.comp(piv, "bias", "control", "two-sided"), B5.comp(piv, "wide", "control", "two-sided"),
                        B5.comp(piv, "bio", "bias", "two-sided"), B5.comp(piv, "bio", "wide", "two-sided")])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(B3.LOGS / "stageB8_tests.csv", index=False)
    S = D.groupby("arm")[["score", "survival", "alive_wellbeing"]].mean()
    S.to_csv(B3.LOGS / "stageB8_summary.csv")
    pd.set_option("display.width", 220)
    print(T.round(4).to_string())
    print(S.sort_values("score").round(3).to_string())
    for a in ("bias", "wide"):
        for r in range(4):
            p = json.loads((OUT / f"{a}_r{r}.json").read_text())["params"]
            print(a, r, {k: round(p[k], 4) for k in SPACE[a]})


if __name__ == "__main__":
    {"opt": opt, "test": test}[sys.argv[1]]()
