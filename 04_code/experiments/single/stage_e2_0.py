"""Stage E2-0 (registered before any E2-0 run): gate for the second environment E2 (12 x 12 grid, 7 x 7 view, 3 food
cells, 2 threat clusters of 7, regimes of 800-2000 steps, 9000-step life). No modulator is tuned here.
  calibrate  fixed rule on development seeds: choose energy decay and scarce regrowth so that the four stage-B VDBE
             bases (no modulator, knobs as tuned in the original world) survive in E2 as in the original world
  validate   numpy (bio_rl.fast.adaptive) vs torch (bio_rl.torchsim.world, learner tab) in E2, as validate_torch.py
  headroom   regime-informed grid reference vs best fixed setting for the tabular learner, as stage A
Resumable (per-job cache, atomic writes). Run: python stage_e2_0.py [calibrate|validate|headroom|all] [--workers N]"""
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _resume import _atomic_write, cached_map
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

LOGS = ROOT / "results" / "logs" / "single"
OUT = LOGS / "stageE2_0"
DECAYS = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]
SCARCE = [60, 75, 90, 105, 120]
GRID = list(itertools.product([0.1, 0.3, 0.6, 0.9], [0.02, 0.1, 0.2, 0.35, 0.5], [0.5, 1.0, 2.0, 4.0]))
SEEDS = dict(dev_sched=99600, dev_bank=3_900_000, train_bank=3_000_000, val_sched=200_000, val_bank=3_200_000,
             val_np=180_000, val_torch=190_000, head_sched=46000, head_bank=3_300_000, head_sim=195_000)
NCAL, NS_VAL, K_VAL, CH = 512, 4096, 4, 1024
if os.environ.get("E2_SMOKE", "").strip():      # code check only: tiny sizes, scratch output, nothing registered
    OUT = Path(os.environ["E2_SMOKE"].strip())
    DECAYS, SCARCE, NCAL, NS_VAL, CH = [0.004, 0.005], [90], 32, 16, 32
    GRID = GRID[::20]
    register = lambda spec: "smoke-no-registration"
SPEC = {
    "id": "single_stageE2_0_gate",
    "registered": "after the C3c results and the E2 code check (bit-identity of the original world), before any E2-0 run",
    "environment": ("E2: 12 x 12 grid, 3 food cells (pairwise and from the shelter >= 6 steps apart), 1 shelter, 2 "
                    "non-overlapping threat clusters of 7 cells (9.7% of cells; original 9.4%); same five regimes, hit "
                    "probabilities (danger 0.3 in threat cells, uncontrollable 0.006 anywhere) and injury 0.4; food "
                    "regrowth 38 (normal), layout change every 450 steps (volatile); regimes 800-2000 steps; life 9000 "
                    "steps, juvenile 750; view radius 3 (7 x 7) for the network learners; move cost and food gain as in "
                    "the original world. Energy decay and scarce regrowth set by the calibration rule"),
    "calibrate": ("development seeds only (schedules 99600+, layout bank 3900000-3900999). Reference: the four stage-B "
                  "VDBE bases (stageB_opt/vdbe_r0-3, no modulator) in the original world, 512 agents each on original "
                  "schedules: overall survival s_ref and the share of agents that die during scarcity d_ref. E2: the same "
                  "bases on E2 schedules for every pair (energy decay in {0.0015, 0.0020, 0.0025, 0.0030, 0.0035, "
                  "0.0040}) x (scarce regrowth in {60, 75, 90, 105, 120}); choose the pair minimising |s - s_ref| + "
                  "|d - d_ref| (means over the 4 bases); ties -> larger decay, then 90. Grid set after a development "
                  "probe (seeds 99600+) showed that decay 0.003-0.005 leaves E2 much harsher than the original world; "
                  "a choice on the grid edge is reported"),
    "validate": ("as single_torch_world_validation, in E2 with the calibrated values: tabular VDBE learner in numpy "
                 "(controller vdbe_generic) and torch (learner tab); 8 configurations (stage-B VDBE bases r0-3, stage-B3 "
                 "generic layers r0-3); 4096 E2 schedules (seeds 200000-204095) x 4 agents, layout bank 3200000-3200999, "
                 "same schedules and layouts, independent noise (numpy 180000 + 100 config + chunk, torch 190000 + "
                 "config); paired TOST over schedule-configuration pairs, margin +-0.01, alpha 0.05, score and survival; "
                 "both must pass"),
    "headroom": ("as stage A with the tabular learner (controller fixed; reference: controller oracle) in E2: 80 constant "
                 "settings (alpha {0.1, 0.3, 0.6, 0.9} x eps {0.02, 0.1, 0.2, 0.35, 0.5} x safety weight {0.5, 1, 2, "
                 "4}); per-regime best from 4500-step stationary lives (64 agents per setting and regime, training bank "
                 "3000000-3001999); best fixed from 16 switching E2 schedules x 4 agents; test on 64 new E2 schedules "
                 "(seeds 46000-46063) x 8 agents, layout bank 3300000-3300999, simulator 195000 + schedule: best fixed, "
                 "regime-informed grid reference and a non-learner (alpha 0, eps 0); headroom = reference - best fixed "
                 "with bootstrap CI over schedules"),
    "gate": ("E2 passes if (1) both validation TOSTs pass and (2) the headroom CI excludes 0 and the headroom is at "
             "least 10% of the range between the non-learner and the reference (as stage A). Only then are the C4 "
             "bases tuned (on the lab server, separately registered)"),
}


def bank(kind):
    lo = {"dev": SEEDS["dev_bank"], "train": SEEDS["train_bank"], "val": SEEDS["val_bank"], "head": SEEDS["head_bank"]}[kind]
    n = 2000 if kind == "train" else 1000
    return LayoutBank.build_e2(range(lo, lo + n))


_b = {}


def cached_bank(kind, e2=True):
    key = (kind, e2)
    if key not in _b:
        _b[key] = bank(kind) if e2 else LayoutBank.build(range(SEEDS["dev_bank"], SEEDS["dev_bank"] + 1000))
    return _b[key]


def vdbe_base(r):
    return json.loads((LOGS / "stageB_opt" / f"vdbe_r{r}.json").read_text())["params"]


def died_share(o, M, T, regime="scarce"):
    life = np.round(o["survival"] * T).astype(int)
    died = life < T
    rd = M[np.clip(life - 1, 0, T - 1), np.arange(M.shape[1])]
    return float((died & (rd == R.NAMES.index(regime))).mean())


def cal_job(job):
    """job = ('ref', r) or ('e2', r, decay, scarce): survival and scarce-death share of base r."""
    rng_off = 0
    if job[0] == "ref":
        r = job[1]
        M = A.regime_matrix([R.schedule(np.random.default_rng(SEEDS["dev_sched"] + i)) for i in range(NCAL)], R.T_LIFE)
        o = A.simulate("vdbe", A.params(NCAL, **vdbe_base(r)), M, cached_bank("dev", e2=False), np.arange(NCAL) % 1000,
                       seed=SEEDS["dev_sched"] + 10 * r)
        return dict(world="original", base=r, survival=float(o["survival"].mean()), died_scarce=died_share(o, M, R.T_LIFE))
    _, r, d, sc = job
    M = A.regime_matrix([R.schedule_e2(np.random.default_rng(SEEDS["dev_sched"] + i)) for i in range(NCAL)], R.T_LIFE_E2)
    o = A.simulate("vdbe", A.params(NCAL, **vdbe_base(r)), M, cached_bank("dev"), np.arange(NCAL) % 1000,
                   seed=SEEDS["dev_sched"] + 10 * r, reg_tab=R.reg_tab_e2(regrow_scarce=sc), decay=d,
                   juvenile=R.JUVENILE_E2)
    return dict(world="E2", base=r, decay=d, regrow_scarce=sc, survival=float(o["survival"].mean()),
                died_scarce=died_share(o, M, R.T_LIFE_E2))


def calibrate():
    print("registered", register(SPEC)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("ref", r) for r in range(4)] + [("e2", r, d, sc) for d in DECAYS for sc in SCARCE for r in range(4)]
    keys = ["ref_%d" % j[1] if j[0] == "ref" else "e2_%d_%g_%d" % j[1:] for j in jobs]
    with Pool(NWORK) as p:
        rows = cached_map(p, cal_job, jobs, keys, OUT / "cal_cache")
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "calibration_raw.csv", index=False)
    ref = D[D.world == "original"][["survival", "died_scarce"]].mean()
    G = D[D.world == "E2"].groupby(["decay", "regrow_scarce"])[["survival", "died_scarce"]].mean().reset_index()
    G["loss"] = (G.survival - ref.survival).abs() + (G.died_scarce - ref.died_scarce).abs()
    G["tie_scarce"] = (G.regrow_scarce - 90).abs()
    best = G.sort_values(["loss", "decay", "tie_scarce"], ascending=[True, False, True]).iloc[0]
    G.to_csv(OUT / "calibration_grid.csv", index=False)
    choice = dict(decay=float(best.decay), regrow_scarce=int(best.regrow_scarce), ref_survival=float(ref.survival),
                  ref_died_scarce=float(ref.died_scarce), e2_survival=float(best.survival),
                  e2_died_scarce=float(best.died_scarce), loss=float(best.loss))
    _atomic_write(OUT / "calibration.json", json.dumps(choice, indent=1).encode())
    print(G.round(4).to_string())
    print("choice", choice, flush=True)


def calib():
    return json.loads((OUT / "calibration.json").read_text())


def val_configs():
    out = [(f"base_r{r}", vdbe_base(r)) for r in range(4)]
    out += [(f"generic_r{r}", json.loads((LOGS / "stageB3_opt" / f"generic_r{r}.json").read_text())["params"])
            for r in range(4)]
    return out[:1] if os.environ.get("E2_SMOKE", "").strip() else out


def val_setup():
    sch = [R.schedule_e2(np.random.default_rng(SEEDS["val_sched"] + i)) for i in range(NS_VAL)]
    M = A.regime_matrix([s for s in sch for _ in range(K_VAL)], R.T_LIFE_E2)
    lay = np.concatenate([(np.arange(K_VAL) * 37 + i * 11) % 1000 for i in range(NS_VAL)])
    return M, lay


_vc = {}


def val_np_job(job):
    ci, c = job
    if "M" not in _vc:
        _vc["M"], _vc["lay"] = val_setup()
    cb = calib()
    name, prm = val_configs()[ci]
    sl = slice(c * CH, (c + 1) * CH)
    o = A.simulate("vdbe_generic", A.params(CH, **prm), _vc["M"][:, sl], cached_bank("val"), _vc["lay"][sl],
                   seed=SEEDS["val_np"] + 100 * ci + c, reg_tab=R.reg_tab_e2(regrow_scarce=cb["regrow_scarce"]),
                   decay=cb["decay"], juvenile=R.JUVENILE_E2)
    return o["score"], o["survival"]


def tost_paired(d, margin=0.01):
    se = d.std(ddof=1) / np.sqrt(len(d))
    df = len(d) - 1
    p = max(1 - stats.t.cdf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))
    return float(p), float(d.mean() - 1.645 * se), float(d.mean() + 1.645 * se)


def validate():
    print("registered", register(SPEC)[:12], flush=True)
    cb = calib()
    C = val_configs()
    nchunk = NS_VAL * K_VAL // CH
    jobs = [(ci, c) for ci in range(len(C)) for c in range(nchunk)]
    with Pool(NWORK) as p:
        res = cached_map(p, val_np_job, jobs, [f"np_{ci}_{c}" for ci, c in jobs], OUT / "val_cache")
    from bio_rl.torchsim import world as W
    M, lay = val_setup()
    vb = cached_bank("val")
    rows, dsc, dsv = [], [], []
    for ci, (name, prm) in enumerate(C):
        f = OUT / "val_cache" / f"torch_{ci}.pkl"
        if not f.exists():
            t = time.time()
            prm_t = {k: v for k, v in prm.items() if k in W.params(1)}
            o = W.simulate("tab", W.params(NS_VAL * K_VAL, **prm_t), M, vb, lay, seed=SEEDS["val_torch"] + ci,
                           reg_tab=R.reg_tab_e2(regrow_scarce=cb["regrow_scarce"]), decay=cb["decay"],
                           juvenile=R.JUVENILE_E2)
            _atomic_write(f, pickle.dumps((o["score"], o["survival"])))
            print("torch done", name, f"{time.time() - t:.0f}s", flush=True)
        ts, tv = pickle.loads(f.read_bytes())
        ns = np.concatenate([res[jobs.index((ci, c))][0] for c in range(nchunk)])
        nv = np.concatenate([res[jobs.index((ci, c))][1] for c in range(nchunk)])
        d_s = ns.reshape(NS_VAL, K_VAL).mean(1) - ts.reshape(NS_VAL, K_VAL).mean(1)
        d_v = nv.reshape(NS_VAL, K_VAL).mean(1) - tv.reshape(NS_VAL, K_VAL).mean(1)
        dsc.append(d_s); dsv.append(d_v)
        for meas, d, a, b in (("score", d_s, ns, ts), ("survival", d_v, nv, tv)):
            p, lo90, hi90 = tost_paired(d)
            rows.append(dict(config=name, measure=meas, numpy=a.mean(), torch=b.mean(), diff=d.mean(), ci90_lo=lo90,
                             ci90_hi=hi90, p_tost=p, family="secondary"))
    for meas, ds in (("score", dsc), ("survival", dsv)):
        d = np.concatenate(ds)
        p, lo90, hi90 = tost_paired(d)
        rows.append(dict(config="pooled", measure=meas, diff=d.mean(), ci90_lo=lo90, ci90_hi=hi90, p_tost=p,
                         family="primary", verdict="PASS" if p < 0.05 else "FAIL"))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "validation.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(5).to_string(), flush=True)


def head_stationary(job):
    regime, = job
    cb = calib()
    reps = 64
    B = len(GRID) * reps
    T = 4500
    M = np.full((T, B), R.NAMES.index(regime), int)
    a = np.repeat([g[0] for g in GRID], reps); e = np.repeat([g[1] for g in GRID], reps)
    s = np.repeat([g[2] for g in GRID], reps)
    lay = np.tile(np.arange(reps) * 13 % 2000, len(GRID))
    o = A.simulate("fixed", A.params(B, alpha0=a, eps0=e, s0=s), M, cached_bank("train"), lay,
                   seed=SEEDS["head_sim"] - 1000 + R.NAMES.index(regime),
                   reg_tab=R.reg_tab_e2(regrow_scarce=cb["regrow_scarce"]), decay=cb["decay"], juvenile=R.JUVENILE_E2)
    return o["score"].reshape(len(GRID), reps).mean(1)


def head_switching(job):
    sch_seed, = job
    cb = calib()
    reps = 4
    B = len(GRID) * reps
    sched = R.schedule_e2(np.random.default_rng(sch_seed))
    M = A.regime_matrix([sched] * B, R.T_LIFE_E2)
    a = np.repeat([g[0] for g in GRID], reps); e = np.repeat([g[1] for g in GRID], reps)
    s = np.repeat([g[2] for g in GRID], reps)
    lay = np.tile((np.arange(reps) * 31 + sch_seed * 7) % 2000, len(GRID))
    o = A.simulate("fixed", A.params(B, alpha0=a, eps0=e, s0=s), M, cached_bank("train"), lay, seed=sch_seed + 500,
                   reg_tab=R.reg_tab_e2(regrow_scarce=cb["regrow_scarce"]), decay=cb["decay"], juvenile=R.JUVENILE_E2)
    return o["score"].reshape(len(GRID), reps).mean(1)


def head_test(job):
    sch_seed, fixed, oracle = job
    cb = calib()
    reps = 8
    sched = R.schedule_e2(np.random.default_rng(sch_seed))
    M = A.regime_matrix([sched] * reps, R.T_LIFE_E2)
    lay = (np.arange(reps) * 37 + sch_seed * 11) % 1000
    kw = dict(reg_tab=R.reg_tab_e2(regrow_scarce=cb["regrow_scarce"]), decay=cb["decay"], juvenile=R.JUVENILE_E2,
              record=("wellbeing",))
    seed = SEEDS["head_sim"] + sch_seed
    res = {"best_fixed": A.simulate("fixed", A.params(reps, alpha0=fixed[0], eps0=fixed[1], s0=fixed[2]), M,
                                    cached_bank("head"), lay, seed=seed, **kw),
           "floor": A.simulate("fixed", A.params(reps, alpha0=0.0, eps0=0.0), M, cached_bank("head"), lay, seed=seed, **kw),
           "reference": A.simulate("oracle", A.params(reps), M, cached_bank("head"), lay, seed=seed,
                                   oracle_table=np.array([oracle[r] for r in R.NAMES], float), **kw)}
    rows = []
    for kind, o in res.items():
        wb = o["wellbeing"]
        for i in range(reps):
            row = dict(schedule=sch_seed, agent=i, kind=kind, score=float(o["score"][i]), survival=float(o["survival"][i]))
            for j, rn in enumerate(R.NAMES):
                m = M[:, i] == j
                row[f"score_{rn}"] = float(wb[m, i].mean()) if m.any() else np.nan
            rows.append(row)
    return rows


def headroom():
    print("registered", register(SPEC)[:12], flush=True)
    with Pool(NWORK) as p:
        st = cached_map(p, head_stationary, [(r,) for r in R.NAMES], [f"stat_{r}" for r in R.NAMES], OUT / "head_cache")
        sw = cached_map(p, head_switching, [(s,) for s in range(SEEDS["dev_sched"], SEEDS["dev_sched"] + 16)],
                        [f"sw_{s}" for s in range(16)], OUT / "head_cache")
    sw = np.mean(sw, axis=0)
    grid = pd.DataFrame(GRID, columns=["alpha", "epsilon", "safety_mult"])
    for r, v in zip(R.NAMES, st):
        grid[f"stationary_{r}"] = v
    grid["switching"] = sw
    grid.to_csv(OUT / "headroom_grid.csv", index=False)
    fixed = GRID[int(np.argmax(sw))]
    oracle = {r: GRID[int(np.argmax(v))] for r, v in zip(R.NAMES, st)}
    _atomic_write(OUT / "headroom_choices.json", json.dumps(dict(best_fixed=fixed, reference=oracle), indent=1).encode())
    nt = 64 if not os.environ.get("E2_SMOKE", "").strip() else 4
    with Pool(NWORK) as p:
        rows = [r for rr in p.map(head_test, [(s, fixed, oracle) for s in range(SEEDS["head_sched"],
                                                                                  SEEDS["head_sched"] + nt)]) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "headroom_test_raw.csv", index=False)
    per = df.groupby(["schedule", "kind"]).score.mean().unstack()
    d = (per["reference"] - per["best_fixed"]).values
    boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rel = d.mean() / (per["reference"].mean() - per["floor"].mean())
    S = dict(score_reference=per["reference"].mean(), score_best_fixed=per["best_fixed"].mean(),
             score_floor=per["floor"].mean(), headroom=d.mean(), ci_lo=lo, ci_hi=hi, relative_headroom=rel,
             headroom_pass=bool(lo > 0 and rel >= 0.10))
    for rn in R.NAMES:
        pr = df.groupby(["schedule", "kind"])[f"score_{rn}"].mean().unstack()
        S[f"headroom_{rn}"] = float((pr["reference"] - pr["best_fixed"]).mean())
    S.update({f"survival_{k}": float(v) for k, v in df.groupby("kind").survival.mean().items()})
    V = pd.read_csv(OUT / "validation.csv")
    vp = V[V.family == "primary"]
    S["validation_pass"] = bool((vp.verdict == "PASS").all())
    S["gate"] = "PASS" if (S["validation_pass"] and S["headroom_pass"]) else "FAIL"
    pd.Series(S).to_csv(OUT / "gate_summary.csv")
    print("best fixed", fixed, "reference", oracle)
    print(pd.Series(S).to_string(), flush=True)


def all_():
    for step in (calibrate, validate, headroom):
        print("=== step", step.__name__, time.strftime("%H:%M:%S"), flush=True)
        step()
    print("=== E2-0 finished", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    {"calibrate": calibrate, "validate": validate, "headroom": headroom, "all": all_}[sys.argv[1]]()
