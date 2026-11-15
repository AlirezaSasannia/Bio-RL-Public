"""Diagnosis of the failed E2-0 gate (exploratory, development seeds only; no registered data reused, nothing tested).
Same tabular learner (controller fixed) and 80-setting grid as stage A / E2-0, in the original world and in E2 (calibrated
values from E2-0). Per world:
  stationary  per-regime grid scores and survival (64 agents per setting, 3000 / 4500-step lives)
  contrast    loss in regime r from using regime r''s best setting; knob sensitivity (range of marginal means)
  switching   best fixed chosen on 16 dev schedules; dev headroom (oracle vs best fixed vs non-learner) on 32 others
Seeds: schedules 970000-970047, layout banks 3970000+ (E2) and 3980000+ (original), simulator 971000+.
Selection-stability replicates (env DIAG_REP=k > 0): selection on schedules 970000 + 100 k + (0-15), simulator
971000 + 10000 k, layout offset 7 k; the same 32 dev test schedules (970016-970047) for every replicate. Env
DIAG_GATE=1 also evaluates the E2-0 gate's choices (headroom_choices.json) on these dev test schedules.
Run: python diag_e2.py [--workers N]  ->  results/logs/single/diagE2/ (rep{k}/ for replicates)"""
import os
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

LOGS = ROOT / "results" / "logs" / "single"
REP = int(os.environ.get("DIAG_REP", "0"))
OUT = LOGS / "diagE2" / (f"rep{REP}" if REP else "")
GRID = list(itertools.product([0.1, 0.3, 0.6, 0.9], [0.02, 0.1, 0.2, 0.35, 0.5], [0.5, 1.0, 2.0, 4.0]))
SCH, SIM = 970_000, 971_000 + 10_000 * REP
SEL = SCH + 100 * REP
CAL = json.loads((LOGS / "stageE2_0" / "calibration.json").read_text())
_b = {}


def world(w):
    """(bank, schedule fn, T_life, simulate kwargs) for 'orig' or 'e2'."""
    if w not in _b:
        _b[w] = (LayoutBank.build(range(3_980_000, 3_982_000)) if w == "orig"
                 else LayoutBank.build_e2(range(3_970_000, 3_972_000)))
    if w == "orig":
        return _b[w], R.schedule, R.T_LIFE, 3000, {}
    return _b[w], R.schedule_e2, R.T_LIFE_E2, 4500, dict(
        reg_tab=R.reg_tab_e2(regrow_scarce=CAL["regrow_scarce"]), decay=CAL["decay"], juvenile=R.JUVENILE_E2)


def knobs(cfgs, reps):
    c = np.repeat(np.array(cfgs, float), reps, 0)
    return A.params(len(c), alpha0=c[:, 0], eps0=c[:, 1], s0=c[:, 2])


def stationary(job):
    w, regime = job
    bank, _, _, T, kw = world(w)
    reps = 64
    B = len(GRID) * reps
    M = np.full((T, B), R.NAMES.index(regime), int)
    lay = np.tile((np.arange(reps) * 13 + 7 * REP) % len(bank), len(GRID))
    o = A.simulate("fixed", knobs(GRID, reps), M, bank, lay, seed=SIM + 10 * R.NAMES.index(regime) + (w == "e2"), **kw)
    return dict(world=w, regime=regime, score=o["score"].reshape(-1, reps).mean(1),
                survival=o["survival"].reshape(-1, reps).mean(1))


def switching(job):
    w, s = job
    bank, sch, T, _, kw = world(w)
    reps = 4
    B = len(GRID) * reps
    M = A.regime_matrix([sch(np.random.default_rng(s))] * B, T)
    lay = np.tile((np.arange(reps) * 31 + s * 7 + 7 * REP) % len(bank), len(GRID))
    o = A.simulate("fixed", knobs(GRID, reps), M, bank, lay, seed=s + 500 + (w == "e2"), **kw)
    return o["score"].reshape(-1, reps).mean(1)


def test(job):
    w, s, fixed, oracle = job
    bank, sch, T, _, kw = world("e2" if w == "e2_gate" else w)
    reps = 8
    M = A.regime_matrix([sch(np.random.default_rng(s))] * reps, T)
    lay = (np.arange(reps) * 37 + s * 11) % len(bank)
    kw = dict(kw, record=("wellbeing",))
    seed = 971_000 + 100 + s                           # same test noise for every replicate
    res = {"fixed": A.simulate("fixed", knobs([fixed], reps), M, bank, lay, seed=seed, **kw),
           "floor": A.simulate("fixed", A.params(reps, alpha0=0.0, eps0=0.0), M, bank, lay, seed=seed, **kw),
           "oracle": A.simulate("oracle", A.params(reps), M, bank, lay, seed=seed,
                                oracle_table=np.array([oracle[r] for r in R.NAMES], float), **kw)}
    rows = []
    for kind, o in res.items():
        for i in range(reps):
            row = dict(world=w, schedule=s, kind=kind, score=o["score"][i], survival=o["survival"][i])
            for j, rn in enumerate(R.NAMES):
                m = M[:, i] == j
                row[f"score_{rn}"] = o["wellbeing"][m, i].mean() if m.any() else np.nan
                row[f"time_{rn}"] = m.mean()
            rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = ["orig", "e2"]
    WT = W + (["e2_gate"] if os.environ.get("DIAG_GATE") else [])
    with Pool(NWORK) as p:
        st = p.map(stationary, [(w, r) for w in W for r in R.NAMES])
        sw = p.map(switching, [(w, s) for w in W for s in range(SEL, SEL + 16)])
    G = pd.DataFrame(GRID * 2, columns=["alpha", "eps", "safety"])
    G.insert(0, "world", np.repeat(W, len(GRID)))
    for d in st:
        m = G.world == d["world"]
        G.loc[m, f"score_{d['regime']}"] = d["score"]
        G.loc[m, f"surv_{d['regime']}"] = d["survival"]
    G["switching"] = np.concatenate([np.mean(sw[:16], 0), np.mean(sw[16:], 0)])
    G.to_csv(OUT / "grid.csv", index=False)

    rows, choices, jobs = [], {}, []
    for w in W:
        g = G[G.world == w].reset_index(drop=True)
        best = {r: int(g[f"score_{r}"].idxmax()) for r in R.NAMES}
        bf = int(g.switching.idxmax())
        choices[w] = dict(best_fixed=GRID[bf], oracle={r: GRID[best[r]] for r in R.NAMES})
        for r in R.NAMES:
            sc = g[f"score_{r}"]
            other = [sc[best[q]] for q in R.NAMES if q != r]
            sens = {k: float(g.groupby(k)[f"score_{r}"].mean().pipe(lambda v: v.max() - v.min()))
                    for k in ("alpha", "eps", "safety")}
            rows.append(dict(world=w, regime=r, best=GRID[best[r]], best_score=sc.max(), median=sc.median(),
                             loss_vs_fixed=sc.max() - sc[bf], mean_cross_loss=sc.max() - np.mean(other),
                             max_cross_loss=sc.max() - min(other), surv_best=g[f"surv_{r}"][best[r]],
                             surv_fixed=g[f"surv_{r}"][bf], **{f"sens_{k}": v for k, v in sens.items()}))
        jobs += [(w, s, GRID[bf], choices[w]["oracle"]) for s in range(SCH + 16, SCH + 48)]
    if os.environ.get("DIAG_GATE"):
        gc = json.loads((LOGS / "stageE2_0" / "headroom_choices.json").read_text())
        choices["e2_gate"] = dict(best_fixed=gc["best_fixed"], oracle=gc["reference"])
        jobs += [("e2_gate", s, tuple(gc["best_fixed"]), {r: tuple(v) for r, v in gc["reference"].items()})
                 for s in range(SCH + 16, SCH + 48)]
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "contrast.csv", index=False)
    (OUT / "choices.json").write_text(json.dumps(choices, indent=1))
    with Pool(NWORK) as p:
        D = pd.DataFrame([r for rr in p.map(test, jobs) for r in rr])
    D.to_csv(OUT / "test_raw.csv", index=False)
    S = []
    for w in WT:
        per = D[D.world == w].groupby(["schedule", "kind"]).mean(numeric_only=True)
        sc = per.score.unstack()
        d = (sc.oracle - sc.fixed).values
        lo, hi = np.percentile(np.random.default_rng(0).choice(d, (4000, len(d))).mean(1), [2.5, 97.5])
        row = dict(world=w, oracle=sc.oracle.mean(), fixed=sc.fixed.mean(), floor=sc.floor.mean(), headroom=d.mean(),
                   ci_lo=lo, ci_hi=hi, rel=d.mean() / (sc.oracle.mean() - sc.floor.mean()))
        for k in ("oracle", "fixed"):
            row[f"surv_{k}"] = D[(D.world == w) & (D.kind == k)].survival.mean()
        for rn in R.NAMES:
            pr = per[f"score_{rn}"].unstack()
            row[f"head_{rn}"] = (pr.oracle - pr.fixed).mean()
        S.append(row)
    S = pd.DataFrame(S)
    S.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(C.round(4).to_string())
    print(json.dumps(choices))
    print(S.round(4).to_string())


if __name__ == "__main__":
    main()
