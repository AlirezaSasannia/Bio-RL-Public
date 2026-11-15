"""Stage B11 development (registered as exploratory before the first dev run; development seeds only, never the B11
test set): choose how to split the surprise signal into a change (volatility) and a noise (stochasticity) component
(Piray & Daw 2021, Nat. Commun.). Candidates, both computed from the need-weighted TD error delta of every step:
  split 1  change = fast mean |delta| (tau 20) minus slow mean |delta| (tau 200, the baseline the surprise signal
           already uses), floored at 0; noise = slow mean |delta|
  split 2  change = |running mean of signed delta| (tau 50); noise = spread of delta around that mean
           (sqrt of running var, tau 50): E[delta^2] = mean^2 + var
Each replaces the surprise input (column 2) by change and adds noise as column 5 (6 inputs, 18 weights).
Arms generic, split1, split2 on B10 bases 0-3, B10 add-on budget with dev generation seeds 59000000 + 1000 k + g,
CMA seed 29000 + k; scored on the dev set (schedules 98000-98127, layouts 1900000-1900999, simulator 98500 + k).
Run: python b11_dev.py [--workers N]"""
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
import stage_b10 as B10
from _resume import cma_run
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space
from bio_rl.fast.popsim import LayoutBank

SPEC = {"id": "single_stageB11_dev_exploratory",
        "registered": "before any B11 development run; B10 bases exist, no B10 test result seen",
        "status": "exploratory design choice on development seeds; no claim",
        "candidates": {"split1": "change = fast (tau 20) - slow (tau 200) mean |delta|, noise = slow mean |delta|",
                       "split2": "change = |EMA_50(delta)|, noise = sqrt(EMA_50(delta^2) - EMA_50(delta)^2)"},
        "scales": ("input scales fixed from a zero-weight diagnostic on dev seeds (mean alive value ~1): split1 change "
                   "/0.001, noise /0.01; split2 change /0.003, noise /0.02; both clipped to [0, 30]"),
        "arms": "generic, split1, split2 on B10 bases 0-3, B10 add-on budget, dev generation seeds 59000000 + 1000 k + g",
        "dev_set": "schedules 98000-98127 x 4, layouts 1900000-1900999, simulator seeds 98500 + k",
        "choice_rule": ("register for B11 the candidate with the higher mean dev score; report weights on change and "
                        "noise for alpha; if neither candidate beats generic on dev, still register the better one "
                        "(the confirmatory test decides)")}
SPACE6 = {f"w_{i}_{k}": (-3.0, 3.0, "lin") for i in range(6) for k in ("a", "e", "s")}
ARMS = {"generic": ("vdbe_generic", B10.SPACE["generic"], {}),
        "split1": ("vdbe_generic", SPACE6, {"split": 1.0, "tau_split": 20.0}),
        "split2": ("vdbe_generic", SPACE6, {"split": 2.0, "tau_split": 50.0})}
OUT = B3.LOGS / "stageB11_dev"
NS, NA_ = 128, 4
TB = None


def job(j):
    arm, k = j
    ctrl, space, extra = ARMS[arm]
    base = B10.load_base(k)
    make = lambda u: {**base, **extra, **decode_space(space, u)}
    res = cma_run(OUT / f"{arm}_r{k}.json", np.full(len(space), 0.5), 0.15, B3.POP, 29000 + k, B3.GENS,
                  lambda X, g: B3.score(ctrl, [make(x) for x in X], 59_000_000 + 1000 * k + g),
                  make, dict(arm=arm, run=k, controller=ctrl))
    return arm, k, res["history"][-1]


def dev_eval(args):
    ctrl, prm, tag, k = args
    global TB
    if TB is None:
        TB = LayoutBank.build(range(1_900_000, 1_901_000))
    sch = [R.schedule(np.random.default_rng(98000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    o = A.simulate(ctrl, A.params(len(lay), **prm), M, TB, lay, seed=98_500 + k)
    return dict(arm=tag, run=k, score=o["score"].mean(), survival=o["survival"].mean())


if __name__ == "__main__":
    print("registered", register(SPEC)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, k) for a in ARMS for k in range(4) if not (OUT / f"{a}_r{k}.json").exists()]
    with Pool(NWORK) as p:
        for a, k, m in p.imap_unordered(job, jobs):
            print(a, k, round(m, 3), flush=True)
        ev = [(ARMS[a][0], json.loads((OUT / f"{a}_r{k}.json").read_text())["params"], a, k)
              for a in ARMS for k in range(4)]
        D = pd.DataFrame(p.map(dev_eval, ev))
    D.to_csv(OUT / "dev_scores.csv", index=False)
    print(D.pivot(index="run", columns="arm", values="score").round(4).to_string())
    print(D.groupby("arm")[["score", "survival"]].mean().round(4).to_string())
    for a in ("split1", "split2"):
        for k in range(4):
            p_ = json.loads((OUT / f"{a}_r{k}.json").read_text())["params"]
            print(a, k, "change->alpha", round(p_["w_2_a"], 3), "noise->alpha", round(p_["w_5_a"], 3),
                  "change->eps", round(p_["w_2_e"], 3), "noise->eps", round(p_["w_5_e"], 3))
