"""Stage B, step 1: optimise every controller with the same CMA-ES budget."""
import json
import sys
import time
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import cma
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bio_rl.battery.prereg import register
from bio_rl.fast.adaptive_search import SPACES, decode, evaluate
from bio_rl.fast.popsim import LayoutBank
from spec_b import SPEC_B

OUT = ROOT / "results" / "logs" / "single" / "stageB_opt"
CTRLS = ["fixed", "reactive", "fuzzy", "plastic", "vdbe", "changept", "pearcehall"]
GENS, POP, NAG, RESTARTS = 30, 12, 48, 4
_bank = None


def run(job):
    global _bank
    ctrl, r = job
    if _bank is None:
        _bank = LayoutBank.build(range(400000, 402000))
    dim = len(SPACES[ctrl])
    x0 = np.random.default_rng(200 + r).uniform(0.1, 0.9, dim)
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"bounds": [0, 1], "popsize": POP, "seed": 3000 + r, "verbose": -9})
    hist, t0 = [], time.time()
    for g in range(GENS):
        X = es.ask()
        f = evaluate(ctrl, [decode(ctrl, x) for x in X], NAG, _bank, gen_seed=12_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(dict(gen=g, best=float(f.max()), mean=float(f.mean())))
    res = dict(controller=ctrl, restart=r, params=decode(ctrl, es.mean), history=hist, seconds=time.time() - t0)
    (OUT / f"{ctrl}_r{r}.json").write_text(json.dumps(res, indent=1))
    return ctrl, r, hist[-1]["mean"]


if __name__ == "__main__":
    print("registered", register(SPEC_B)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, r) for r in range(RESTARTS) for c in CTRLS if not (OUT / f"{c}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for c, r, m in p.imap_unordered(run, jobs):
            print(f"{c} restart {r}: last-generation mean score {m:.3f}", flush=True)
