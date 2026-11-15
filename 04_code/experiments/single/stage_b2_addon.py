"""Stage B2: hormones as an add-on to tuned learners (see spec_b2.py)."""
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
from bio_rl.analysis.stats import holm
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import HORMONE_ONLY, SPACES, decode_space, encode
from bio_rl.fast.popsim import LayoutBank
from spec_b2 import SPEC_B2

LOGS = ROOT / "results" / "logs" / "single"
OUT = LOGS / "stageB2_opt"
GENS, POP, NAG = 20, 12, 48
_bank = None


def score(ctrl, cands, gen_seed):
    rng = np.random.default_rng(gen_seed)
    scheds = [R.schedule(rng) for _ in range(NAG)]
    lay1 = rng.integers(len(_bank), size=NAG)
    m = len(cands)
    keys = set().union(*cands)
    P = A.params(m * NAG, **{k: np.repeat([c[k] for c in cands], NAG) for k in keys})
    o = A.simulate(ctrl, P, A.regime_matrix(scheds * m, R.T_LIFE), _bank, np.tile(lay1, m), seed=gen_seed + 1)
    return o["score"].reshape(m, NAG).mean(1)


def run(job):
    global _bank
    base_ctrl, arm, r = job
    if _bank is None:
        _bank = LayoutBank.build(range(400000, 402000))
    base = json.loads((LOGS / "stageB_opt" / f"{base_ctrl}_r{r}.json").read_text())["params"]
    if arm == "hormones":
        space, ctrl = HORMONE_ONLY, base_ctrl + "_fuzzy"
        x0 = np.full(len(space), 0.5)                       # gains 0, centres mid-range
        make = lambda u: {**base, **decode_space(space, u)}
    else:
        space, ctrl = SPACES[base_ctrl], base_ctrl
        x0 = encode(space, base)
        make = lambda u: decode_space(space, u)
    es = cma.CMAEvolutionStrategy(x0, 0.15, {"bounds": [0, 1], "popsize": POP, "seed": 5000 + r, "verbose": -9})
    hist = []
    for g in range(GENS):
        X = es.ask()
        f = score(ctrl, [make(x) for x in X], 15_000_000 + 1000 * r + g)
        es.tell(X, list(-f))
        hist.append(float(f.mean()))
    res = dict(base=base_ctrl, arm=arm, restart=r, controller=ctrl, params=make(es.mean), history=hist)
    (OUT / f"{base_ctrl}_{arm}_r{r}.json").write_text(json.dumps(res, indent=1))
    return base_ctrl, arm, r, hist[-1]


if __name__ == "__main__":
    print("registered", register(SPEC_B2)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, a, r) for r in range(4) for b in ("fixed", "vdbe") for a in ("hormones", "control")
            if not (OUT / f"{b}_{a}_r{r}.json").exists()]
    with Pool(NWORK) as p:
        for b, a, r, m in p.imap_unordered(run, jobs):
            print(b, a, r, round(m, 3), flush=True)
    # held-out test
    sys.argv = [sys.argv[0]]
    import stage_b_test as T
    rows = []
    for b in ("fixed", "vdbe"):
        for a in ("hormones", "control"):
            fr = []
            for r in range(4):
                d = json.loads((OUT / f"{b}_{a}_r{r}.json").read_text())
                fr.append(T.run(d["controller"], d["params"], r, tag=f"{b}_{a}"))
            rows.append(pd.concat(fr))
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageB2_test_raw.csv", index=False)
    res = []
    for pid, b in (("A1", "fixed"), ("A2", "vdbe")):
        h = D[D.variant == f"{b}_hormones"].groupby("schedule").score.mean()
        c = D[D.variant == f"{b}_control"].groupby("schedule").score.mean()
        d = (h - c).values
        boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
        res.append(dict(test=pid, base=b, hormones=h.mean(), control=c.mean(), diff=d.mean(),
                        ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                        p=float(stats.wilcoxon(d, alternative="greater").pvalue),
                        **{f"diff_{rn}": D[D.variant == f"{b}_hormones"][f"score_{rn}"].mean()
                           - D[D.variant == f"{b}_control"][f"score_{rn}"].mean() for rn in R.NAMES}))
    Rz = pd.DataFrame(res)
    adj = holm({i: Rz.loc[i, "p"] for i in Rz.index})
    Rz["p_holm"] = [adj[i] for i in Rz.index]
    Rz["verdict"] = np.where(Rz.p_holm < 0.05, "PASS", "FAIL")
    Rz.to_csv(LOGS / "stageB2_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(Rz.round(4).to_string())
