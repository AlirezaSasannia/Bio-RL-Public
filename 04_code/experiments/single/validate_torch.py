"""Phase 2, step 1 (registered before running): the batched PyTorch world (bio_rl.torchsim.world, learner 'tab')
against the numpy simulator (bio_rl.fast.adaptive, controller 'vdbe_generic') with the same tabular VDBE learner.
Configurations: the 4 stage-B VDBE bases (global layer off) and the 4 generic layers r0-3 (appraisal-signal path on).
Same schedules and layouts, independent noise. Resumable (per-job cache). Run: python validate_torch.py [--workers N]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _resume import cached_map
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

SPEC = {"id": "single_torch_world_validation",
        "registered": "after phase 1 (B9), before any validation run; dev smoke tests only (seeds 99500+, 99800+)",
        "question": "Is the batched PyTorch regime world equivalent to the numpy simulator used for all stage-B results?",
        "design": ("same tabular VDBE learner in both (numpy controller vdbe_generic, torch learner tab, float32 on "
                   "GPU); 8 configurations: stage-B VDBE bases r0-3 (no global layer) and generic layers r0-3; "
                   "4096 schedules (seeds 30000-34095) x 4 agents per configuration, layout bank 300000-300999, "
                   "same schedules and layouts in both, independent noise (numpy seeds 31000 + chunk, torch 32000 + config)"),
        "primary": ("paired TOST over schedule-configuration pairs (32768) of the schedule-mean difference numpy - "
                    "torch, margin +-0.01 absolute, alpha 0.05, separately for score and survival; both must pass"),
        "margin_rationale": "0.01 is below every effect claimed so far (smallest: bias - control 0.028)",
        "secondary": "per-configuration differences with 95% CI; TOST per configuration reported, not required",
        "decision": "if both primary TOSTs pass, the torch world is used for phase 2; otherwise find the cause first"}
LOGS = ROOT / "results" / "logs" / "single"
NS, K, CH = 4096, 4, 1024
MARGIN = 0.01


def configs():
    out = []
    for r in range(4):
        out.append((f"base_r{r}", json.loads((LOGS / "stageB_opt" / f"vdbe_r{r}.json").read_text())["params"]))
    for r in range(4):
        out.append((f"generic_r{r}", json.loads((LOGS / "stageB3_opt" / f"generic_r{r}.json").read_text())["params"]))
    return out


def setup():
    sch = [R.schedule(np.random.default_rng(30000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(K)], R.T_LIFE)
    lay = np.concatenate([(np.arange(K) * 37 + i * 11) % 1000 for i in range(NS)])
    return M, lay


_cache = {}


def np_job(job):
    ci, c = job
    if "M" not in _cache:
        _cache["M"], _cache["lay"] = setup()
        _cache["bank"] = LayoutBank.build(range(300000, 301000))
    name, prm = configs()[ci]
    sl = slice(c * CH, (c + 1) * CH)
    o = A.simulate("vdbe_generic", A.params(CH, **prm), _cache["M"][:, sl], _cache["bank"], _cache["lay"][sl],
                   seed=31000 + 100 * ci + c)
    return o["score"], o["survival"]


def tost_paired(d, margin):
    se = d.std(ddof=1) / np.sqrt(len(d))
    df = len(d) - 1
    p = max(1 - stats.t.cdf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))
    return float(p), float(d.mean() - 1.645 * se), float(d.mean() + 1.645 * se)


if __name__ == "__main__":
    print("registered", register(SPEC)[:12], flush=True)
    cache = LOGS / "torch_validation_cache"
    C = configs()
    jobs = [(ci, c) for ci in range(len(C)) for c in range(NS * K // CH)]
    with Pool(NWORK) as p:
        res = cached_map(p, np_job, jobs, [f"np_{ci}_{c}" for ci, c in jobs], cache)
    from bio_rl.torchsim import world as W
    import pickle
    M, lay = setup()
    bank = LayoutBank.build(range(300000, 301000))
    rows, dsc, dsv = [], [], []
    for ci, (name, prm) in enumerate(C):
        f = cache / f"torch_{ci}.pkl"
        if not f.exists():
            prm_t = {k: v for k, v in prm.items() if k in W.params(1)}
            o = W.simulate("tab", W.params(NS * K, **prm_t), M, bank, lay, seed=32000 + ci)
            f.write_bytes(pickle.dumps((o["score"], o["survival"])))
            print("torch done", name, flush=True)
        ts, tv = pickle.loads(f.read_bytes())
        ns = np.concatenate([res[jobs.index((ci, c))][0] for c in range(NS * K // CH)])
        nv = np.concatenate([res[jobs.index((ci, c))][1] for c in range(NS * K // CH)])
        d_s = ns.reshape(NS, K).mean(1) - ts.reshape(NS, K).mean(1)
        d_v = nv.reshape(NS, K).mean(1) - tv.reshape(NS, K).mean(1)
        dsc.append(d_s); dsv.append(d_v)
        for meas, d, a, b in (("score", d_s, ns, ts), ("survival", d_v, nv, tv)):
            p, lo90, hi90 = tost_paired(d, MARGIN)
            se = d.std(ddof=1) / np.sqrt(NS)
            rows.append(dict(config=name, measure=meas, numpy=a.mean(), torch=b.mean(), diff=d.mean(),
                             ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se, p_tost=p, family="secondary"))
    for meas, ds in (("score", dsc), ("survival", dsv)):
        d = np.concatenate(ds)
        p, lo90, hi90 = tost_paired(d, MARGIN)
        rows.append(dict(config="pooled", measure=meas, diff=d.mean(), ci_lo=lo90, ci_hi=hi90, p_tost=p,
                         family="primary (90% CI)", verdict="EQUIVALENT" if p < 0.05 else "NOT SHOWN"))
    T = pd.DataFrame(rows)
    T.to_csv(LOGS / "validate_torch.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(5).to_string())
