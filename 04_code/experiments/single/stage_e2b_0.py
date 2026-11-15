"""Stage E2b-0 (second and last gate attempt for the second environment; design in proposals/E2b_proposal.md).
E2 world (12 x 12, 3 food, 2 threat clusters, regimes 800-2000 steps, 9000-step life) with
  - energy decay fixed a priori by distance scaling (0.0026), and
  - per-regime ceiling calibration: danger p_zone, uncontrollable p_random, scarce regrowth and volatile layout-switch
    interval, each chosen so that the tabular learner's survival at the best grid setting of that regime (stationary
    lives) matches the original world's value measured in the same run (development seeds);
  - powered selection for the headroom check (256 agents per setting and regime; 64 x 4 switching lives).
The E2b regime table is built here (src unchanged). No modulator is tuned here.
  register   register the spec (= the commitment; commit the yaml before running anything else)
  calibrate  per-regime ceiling calibration on development seeds -> stageE2b_0/calibration.json
  validate   numpy vs torch (learner tab) in E2b, TOST +-0.01 (as E2-0)
  headroom   regime-informed reference vs best fixed vs non-learner in E2b (the gate) + contrast profile
  posctrl    the same powered headroom protocol in the original world (positive control; reported, not gating)
  all        register, calibrate, validate, headroom, posctrl, then the gate summary
Resumable (per-job cache, atomic writes). Run: python stage_e2b_0.py <step> [--workers N]
Smoke test: E2B_SMOKE=<scratch dir> python stage_e2b_0.py all   (tiny sizes, no registration)"""
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _pool import NWORK

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _resume import _atomic_write, cached_map
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

LOGS = ROOT / "results" / "logs" / "single"
OUT = LOGS / "stageE2b_0"
DECAY_E2B = 0.0026                                       # (d + 0.002) x 9.78 = (0.005 + 0.002) x 6.43
CAL_GRID = {"danger": ("p_zone", [0.10, 0.15, 0.20, 0.25, 0.30]),
            "uncontrollable": ("p_random", [0.003, 0.004, 0.005, 0.006]),
            "scarce": ("regrow", [60, 75, 90, 105, 120]),
            "volatile": ("switch_every", [300, 450, 600, 900])}
E2_0_VALUE = {"danger": 0.30, "uncontrollable": 0.006, "scarce": 105, "volatile": 450}     # tie-break
GRID = list(itertools.product([0.1, 0.3, 0.6, 0.9], [0.02, 0.1, 0.2, 0.35, 0.5], [0.5, 1.0, 2.0, 4.0]))
SEEDS = dict(cal_bank_e2=3_990_000, cal_bank_orig=3_995_000, cal_sim=980_000,
             val_sched=410_000, val_bank=3_600_000, val_np=280_000, val_torch=290_000,
             head_train=3_450_000, head_sw=47_000, head_test=47_100, head_bank=3_500_000, head_sim=205_000,
             pos_train=3_700_000, pos_bank=3_710_000)
REPS, CHUNK = 256, 64                  # stationary agents per setting and regime, in chunks of 64 (memory)
NSW, KSW, NTEST, KTEST = 64, 4, 64, 8
T_ST = {"orig": 3000, "e2b": 4500}     # stationary life (as stage A / E2-0)
NS_VAL, K_VAL, CH = 4096, 4, 1024
SMOKE = os.environ.get("E2B_SMOKE", "").strip()
if SMOKE:                              # code check only: tiny sizes, scratch output, nothing registered
    OUT = Path(SMOKE)
    GRID = GRID[::20]
    REPS, CHUNK, NSW, NTEST, NS_VAL, CH = 8, 4, 2, 2, 16, 32
    T_ST = {"orig": 300, "e2b": 450}
    CAL_GRID = {k: (p, v[:2]) for k, (p, v) in CAL_GRID.items()}
    register = lambda spec: "smoke-no-registration"
NCH = REPS // CHUNK

SPEC = {
    "id": "single_stageE2b_0_gate",
    "registered": ("after the E2-0 gate failed (5.4% < 10%) and a diagnosis on development seeds (2026-10-29), before "
                   "any E2b-0 run; smoke tests on scratch output only"),
    "attempt": ("SECOND AND LAST attempt at a second environment. If E2b-0 fails, C4 is not run and both gates (E2-0 "
                "and E2b-0) are reported in the Supplement as negatives"),
    "changes_after_E2_0": ("declared: (1) the calibration invariant (E2-0 matched the survival of the four stage-B "
                           "bases, which were tuned in the original world; E2b matches each regime's ceiling = "
                           "survival of the tabular learner at the best grid setting in stationary lives); (2) energy "
                           "decay fixed a priori by distance scaling instead of being calibrated; (3) selection sample "
                           "sizes 4x (stationary 256 agents, switching 64 x 4) because development replicates showed "
                           "that the stage A / E2-0 protocol has large selection noise in both worlds (headroom 8-30% "
                           "in the original world, 2-42% in E2). Test size and pass criterion unchanged"),
    "environment": ("E2 layouts (12 x 12, 3 food >= 6 apart, 2 threat clusters of 7), schedules 800-2000 steps, life "
                    "9000, juvenile 750, normal regrowth 38, injury 0.4, move cost and food gain as original. Energy "
                    "decay 0.0026 (energy cost of a shelter-food trip as in the original world: (d + 0.002) x 9.78 = "
                    "(0.005 + 0.002) x 6.43). Regime parameters from the calibration rule"),
    "calibrate": ("development seeds only: E2 layout bank 3990000-3991999, original bank 3995000-3996999, simulator "
                  "980000 + 1000 * regime + 100 * value index + chunk (original world: + 500). Tabular learner "
                  "(controller fixed), 80 settings (alpha {0.1, 0.3, 0.6, 0.9} x eps {0.02, 0.1, 0.2, 0.35, 0.5} x "
                  "safety weight {0.5, 1, 2, 4}) x 256 agents, stationary lives 3000 (original) / 4500 (E2) steps. "
                  "For each regime, s = survival at the setting with the best mean score. Candidates: danger p_zone "
                  "{0.10, 0.15, 0.20, 0.25, 0.30}; uncontrollable p_random {0.003, 0.004, 0.005, 0.006}; scarce "
                  "regrowth {60, 75, 90, 105, 120}; volatile switch interval {300, 450, 600, 900}; safe: none. "
                  "Choose per regime the value minimising |s_E2 - s_orig|; ties -> the E2-0 value (0.30, 0.006, "
                  "105, 450); grid-edge choices are reported. Contrast profile (cross-loss, knob sensitivity) of "
                  "E2b vs the original world reported, not used for selection"),
    "validate": ("as E2-0 with the E2b values: tabular VDBE numpy (vdbe_generic) vs torch (tab), 8 configurations "
                 "(stage-B VDBE bases r0-3, stage-B3 generic layers r0-3), 4096 schedules (seeds 410000-414095) x 4, "
                 "layout bank 3600000-3600999, independent noise (numpy 280000 + 100 config + chunk, torch 290000 + "
                 "config); paired TOST over schedules, margin +-0.01, alpha 0.05, pooled score and survival; both "
                 "must pass"),
    "headroom": ("as stage A / E2-0 with powered selection, in E2b: per-regime best of the 80 settings from stationary "
                 "4500-step lives, 256 agents per setting and regime (training bank 3450000-3451999, simulator "
                 "203000 + 10 regime + chunk); best fixed from 64 switching schedules (seeds 47000-47063) x 4 agents "
                 "(simulator 206000 + schedule index); test on 64 new schedules (seeds 47100-47163) x 8 agents, "
                 "layout bank 3500000-3500999, simulator 205000 + schedule: best fixed, regime-informed reference "
                 "and non-learner (alpha 0, eps 0); headroom = reference - best fixed, 95% percentile bootstrap over "
                 "schedules (4000 resamples, seed 0)"),
    "positive_control": ("reported, does not gate: the same powered protocol in the original world (decay 0.005, "
                         "original regime table), training bank 3700000-3701999, test bank 3710000-3710999, the same "
                         "schedule seed formulas with original-world schedules, the same simulator formulas + 1"),
    "gate": ("E2b passes if (1) both validation TOSTs pass and (2) the headroom CI excludes 0 and the headroom is at "
             "least 10% of the range between the non-learner and the reference (as stage A and E2-0). Only then is "
             "C4 run (separately registered)"),
}

_b = {}


def bank(name):
    """E2 banks (build_e2) unless the name starts with 'orig'."""
    if name not in _b:
        lo, n, e2 = {"cal_e2": (SEEDS["cal_bank_e2"], 2000, True), "cal_orig": (SEEDS["cal_bank_orig"], 2000, False),
                     "val": (SEEDS["val_bank"], 1000, True), "head_train": (SEEDS["head_train"], 2000, True),
                     "head_test": (SEEDS["head_bank"], 1000, True), "orig_train": (SEEDS["pos_train"], 2000, False),
                     "orig_test": (SEEDS["pos_bank"], 1000, False)}[name]
        _b[name] = LayoutBank.build_e2(range(lo, lo + n)) if e2 else LayoutBank.build(range(lo, lo + n))
    return _b[name]


def reg_tab(over=None):
    """E2b regime table (p_zone, p_random, regrow, switch_every), rows in NAMES order. over: {regime: value} for the
    calibrated parameter of each regime; missing regimes take the E2-0 values."""
    v = {**E2_0_VALUE, **(over or {})}
    tab = {"safe": (0.0, 0.0, R.REGROW_E2, 0), "danger": (v["danger"], 0.0, R.REGROW_E2, 0),
           "uncontrollable": (0.0, v["uncontrollable"], R.REGROW_E2, 0), "scarce": (0.0, 0.0, v["scarce"], 0),
           "volatile": (0.0, 0.0, R.REGROW_E2, v["volatile"])}
    return np.array([tab[n] for n in R.NAMES], float)


def world_kw(w, over=None):
    """simulate kwargs: 'e2b' (calibrated or candidate table) or 'orig' (defaults)."""
    if w == "orig":
        return {}
    return dict(reg_tab=reg_tab(over), decay=DECAY_E2B, juvenile=R.JUVENILE_E2)


def calib():
    return json.loads((OUT / "calibration.json").read_text())


def knobs(cfgs, reps):
    c = np.repeat(np.array(cfgs, float), reps, 0)
    return A.params(len(c), alpha0=c[:, 0], eps0=c[:, 1], s0=c[:, 2])


def stationary_chunk(w, regime, bk, seed, over, chunk):
    """Mean score and survival per setting (80,) for one chunk of CHUNK agents per setting."""
    b = bank(bk)
    T = T_ST["orig" if w == "orig" else "e2b"]
    B = len(GRID) * CHUNK
    M = np.full((T, B), R.NAMES.index(regime), int)
    lay = np.tile((np.arange(CHUNK) + CHUNK * chunk) * 13 % len(b), len(GRID))
    o = A.simulate("fixed", knobs(GRID, CHUNK), M, b, lay, seed=seed, **world_kw(w, over))
    return o["score"].reshape(-1, CHUNK).mean(1), o["survival"].reshape(-1, CHUNK).mean(1)


# ---------------------------------------------------------------- calibration
def cal_jobs():
    jobs = [("orig", r, -1, None, c) for r in R.NAMES for c in range(NCH)]
    jobs += [("e2b", "safe", -1, None, c) for c in range(NCH)]
    jobs += [("e2b", r, i, v, c) for r, (_, vals) in CAL_GRID.items() for i, v in enumerate(vals) for c in range(NCH)]
    return jobs


def cal_job(job):
    w, r, i, v, c = job
    seed = SEEDS["cal_sim"] + 1000 * R.NAMES.index(r) + 100 * max(i, 0) + c + (500 if w == "orig" else 0)
    return stationary_chunk(w, r, "cal_orig" if w == "orig" else "cal_e2", seed, None if v is None else {r: v}, c)


def profile(score, surv):
    """per-regime ceiling, cross-loss and knob sensitivity from (regime -> (80,) score) and survival dicts."""
    G = pd.DataFrame(GRID, columns=["alpha", "eps", "safety"])
    best = {r: int(np.argmax(score[r])) for r in score}
    rows = []
    for r in score:
        G["s"] = score[r]
        other = [score[r][best[q]] for q in score if q != r]
        rows.append(dict(regime=r, best=str(GRID[best[r]]), best_score=score[r][best[r]], surv_best=surv[r][best[r]],
                         mean_cross_loss=score[r][best[r]] - np.mean(other),
                         **{f"sens_{k}": float(G.groupby(k).s.mean().pipe(lambda x: x.max() - x.min()))
                            for k in ("alpha", "eps", "safety")}))
    return pd.DataFrame(rows)


def calibrate():
    print("registered", register(SPEC)[:12], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = cal_jobs()
    keys = ["cal_%s_%s_%d_%d" % (w, r, i, c) for w, r, i, v, c in jobs]
    with Pool(NWORK) as p:
        res = cached_map(p, cal_job, jobs, keys, OUT / "cal_cache")
    acc = {}
    for (w, r, i, v, c), (sc, sv) in zip(jobs, res):
        a = acc.setdefault((w, r, i, v), [0, 0])
        a[0] = a[0] + sc / NCH
        a[1] = a[1] + sv / NCH
    rows = []
    for (w, r, i, v), (sc, sv) in acc.items():
        b = int(np.argmax(sc))
        rows.append(dict(world=w, regime=r, value=v, best=str(GRID[b]), best_score=sc[b], surv_best=sv[b]))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "calibration_grid.csv", index=False)
    ref = D[D.world == "orig"].set_index("regime").surv_best
    choice, edge = {}, []
    for r, (par, vals) in CAL_GRID.items():
        g = D[(D.world == "e2b") & (D.regime == r)].copy()
        g["loss"] = (g.surv_best - ref[r]).abs()
        g["tie"] = (g.value != E2_0_VALUE[r]).astype(int)
        v = g.sort_values(["loss", "tie"]).iloc[0].value
        choice[r] = float(v) if r in ("danger", "uncontrollable") else int(v)
        if v in (vals[0], vals[-1]):
            edge.append(r)
    out = dict(decay=DECAY_E2B, values=choice, reg_tab=reg_tab(choice).tolist(), on_grid_edge=edge,
               ref_surv_best={r: float(ref[r]) for r in R.NAMES})
    _atomic_write(OUT / "calibration.json", json.dumps(out, indent=1).encode())
    # contrast profile at the chosen values (reported only)
    sc_e, sv_e, sc_o, sv_o = {}, {}, {}, {}
    for (w, r, i, v), (sc, sv) in acc.items():
        if w == "orig":
            sc_o[r], sv_o[r] = sc, sv
        elif r == "safe" or v == choice.get(r):
            sc_e[r], sv_e[r] = sc, sv
    order = lambda d: {r: d[r] for r in R.NAMES}
    P = pd.concat([profile(order(sc_o), order(sv_o)).assign(world="orig"),
                   profile(order(sc_e), order(sv_e)).assign(world="e2b")])
    P.to_csv(OUT / "calibration_profile.csv", index=False)
    pd.set_option("display.width", 220)
    print(D.round(4).to_string())
    print(P.round(4).to_string())
    print("choice", out, flush=True)


# ---------------------------------------------------------------- validation (as E2-0)
def val_configs():
    out = [(f"base_r{r}", json.loads((LOGS / "stageB_opt" / f"vdbe_r{r}.json").read_text())["params"]) for r in range(4)]
    out += [(f"generic_r{r}", json.loads((LOGS / "stageB3_opt" / f"generic_r{r}.json").read_text())["params"])
            for r in range(4)]
    return out[:1] if SMOKE else out


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
    name, prm = val_configs()[ci]
    sl = slice(c * CH, (c + 1) * CH)
    o = A.simulate("vdbe_generic", A.params(CH, **prm), _vc["M"][:, sl], bank("val"), _vc["lay"][sl],
                   seed=SEEDS["val_np"] + 100 * ci + c, **world_kw("e2b", calib()["values"]))
    return o["score"], o["survival"]


def tost_paired(d, margin=0.01):
    se = d.std(ddof=1) / np.sqrt(len(d))
    df = len(d) - 1
    p = max(1 - stats.t.cdf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df))
    return float(p), float(d.mean() - 1.645 * se), float(d.mean() + 1.645 * se)


def validate():
    print("registered", register(SPEC)[:12], flush=True)
    kw = world_kw("e2b", calib()["values"])
    C = val_configs()
    nchunk = NS_VAL * K_VAL // CH
    jobs = [(ci, c) for ci in range(len(C)) for c in range(nchunk)]
    with Pool(NWORK) as p:
        res = cached_map(p, val_np_job, jobs, [f"np_{ci}_{c}" for ci, c in jobs], OUT / "val_cache")
    from bio_rl.torchsim import world as W
    M, lay = val_setup()
    rows, dsc, dsv = [], [], []
    for ci, (name, prm) in enumerate(C):
        f = OUT / "val_cache" / f"torch_{ci}.pkl"
        if not f.exists():
            t = time.time()
            prm_t = {k: v for k, v in prm.items() if k in W.params(1)}
            o = W.simulate("tab", W.params(NS_VAL * K_VAL, **prm_t), M, bank("val"), lay, seed=SEEDS["val_torch"] + ci,
                           **kw)
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
    print(T.round(5).to_string(), flush=True)


# ---------------------------------------------------------------- headroom (gate) and positive control
def sched_fn(w):
    return R.schedule if w == "orig" else R.schedule_e2


def t_life(w):
    return R.T_LIFE if w == "orig" else R.T_LIFE_E2


def head_stat_job(job):
    w, regime, c = job
    over = None if w == "orig" else calib()["values"]
    seed = SEEDS["head_sim"] - 2000 + 10 * R.NAMES.index(regime) + c + (w == "orig")
    return stationary_chunk(w, regime, "orig_train" if w == "orig" else "head_train", seed, over, c)


def head_sw_job(job):
    w, i = job
    s = SEEDS["head_sw"] + i
    b = bank("orig_train" if w == "orig" else "head_train")
    B = len(GRID) * KSW
    M = A.regime_matrix([sched_fn(w)(np.random.default_rng(s))] * B, t_life(w))
    lay = np.tile((np.arange(KSW) * 31 + s * 7) % len(b), len(GRID))
    o = A.simulate("fixed", knobs(GRID, KSW), M, b, lay, seed=SEEDS["head_sim"] + 1000 + i + (w == "orig"),
                   **world_kw(w, None if w == "orig" else calib()["values"]))
    return o["score"].reshape(-1, KSW).mean(1)


def head_test_job(job):
    w, i, fixed, oracle = job
    s = SEEDS["head_test"] + i
    b = bank("orig_test" if w == "orig" else "head_test")
    M = A.regime_matrix([sched_fn(w)(np.random.default_rng(s))] * KTEST, t_life(w))
    lay = (np.arange(KTEST) * 37 + s * 11) % len(b)
    kw = dict(world_kw(w, None if w == "orig" else calib()["values"]), record=("wellbeing",))
    seed = SEEDS["head_sim"] + s + (w == "orig")
    res = {"best_fixed": A.simulate("fixed", knobs([fixed], KTEST), M, b, lay, seed=seed, **kw),
           "floor": A.simulate("fixed", A.params(KTEST, alpha0=0.0, eps0=0.0), M, b, lay, seed=seed, **kw),
           "reference": A.simulate("oracle", A.params(KTEST), M, b, lay, seed=seed,
                                   oracle_table=np.array([oracle[r] for r in R.NAMES], float), **kw)}
    rows = []
    for kind, o in res.items():
        for a in range(KTEST):
            row = dict(schedule=s, agent=a, kind=kind, score=float(o["score"][a]), survival=float(o["survival"][a]))
            for j, rn in enumerate(R.NAMES):
                m = M[:, a] == j
                row[f"score_{rn}"] = float(o["wellbeing"][m, a].mean()) if m.any() else np.nan
            rows.append(row)
    return rows


def headroom_world(w):
    """Selection (stationary + switching) and test for world w ('e2b' gate, 'orig' positive control)."""
    tag = "head" if w == "e2b" else "pos"
    jobs = [(w, r, c) for r in R.NAMES for c in range(NCH)]
    with Pool(NWORK) as p:
        st = cached_map(p, head_stat_job, jobs, [f"{tag}_st_{r}_{c}" for _, r, c in jobs], OUT / f"{tag}_cache")
        sw = cached_map(p, head_sw_job, [(w, i) for i in range(NSW)], [f"{tag}_sw_{i}" for i in range(NSW)],
                        OUT / f"{tag}_cache")
    score = {r: np.mean([st[jobs.index((w, r, c))][0] for c in range(NCH)], 0) for r in R.NAMES}
    surv = {r: np.mean([st[jobs.index((w, r, c))][1] for c in range(NCH)], 0) for r in R.NAMES}
    sw = np.mean(sw, axis=0)
    grid = pd.DataFrame(GRID, columns=["alpha", "epsilon", "safety_mult"])
    for r in R.NAMES:
        grid[f"stationary_{r}"] = score[r]
        grid[f"surv_{r}"] = surv[r]
    grid["switching"] = sw
    grid.to_csv(OUT / f"{tag}_grid.csv", index=False)
    profile(score, surv).to_csv(OUT / f"{tag}_profile.csv", index=False)
    fixed = GRID[int(np.argmax(sw))]
    oracle = {r: GRID[int(np.argmax(score[r]))] for r in R.NAMES}
    _atomic_write(OUT / f"{tag}_choices.json", json.dumps(dict(best_fixed=fixed, reference=oracle), indent=1).encode())
    f = OUT / f"{tag}_test_raw.csv"
    if not f.exists():
        with Pool(NWORK) as p:
            rows = [r for rr in p.map(head_test_job, [(w, i, fixed, oracle) for i in range(NTEST)]) for r in rr]
        pd.DataFrame(rows).to_csv(f, index=False)
    df = pd.read_csv(f)
    per = df.groupby(["schedule", "kind"]).score.mean().unstack()
    d = (per["reference"] - per["best_fixed"]).values
    boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rel = d.mean() / (per["reference"].mean() - per["floor"].mean())
    S = dict(world=w, score_reference=per["reference"].mean(), score_best_fixed=per["best_fixed"].mean(),
             score_floor=per["floor"].mean(), headroom=d.mean(), ci_lo=lo, ci_hi=hi, relative_headroom=rel,
             headroom_pass=bool(lo > 0 and rel >= 0.10))
    for rn in R.NAMES:
        pr = df.groupby(["schedule", "kind"])[f"score_{rn}"].mean().unstack()
        S[f"headroom_{rn}"] = float((pr["reference"] - pr["best_fixed"]).mean())
    S.update({f"survival_{k}": float(v) for k, v in df.groupby("kind").survival.mean().items()})
    print(w, "best fixed", fixed, "reference", oracle)
    print(pd.Series(S).to_string(), flush=True)
    return S


def headroom():
    print("registered", register(SPEC)[:12], flush=True)
    pd.Series(headroom_world("e2b")).to_csv(OUT / "headroom_summary.csv")


def posctrl():
    print("registered", register(SPEC)[:12], flush=True)
    pd.Series(headroom_world("orig")).to_csv(OUT / "posctrl_summary.csv")


def gate():
    S = pd.read_csv(OUT / "headroom_summary.csv", index_col=0).iloc[:, 0].to_dict()
    V = pd.read_csv(OUT / "validation.csv")
    vp = V[V.family == "primary"]
    S["validation_pass"] = bool((vp.verdict == "PASS").all())
    S["headroom_pass"] = str(S["headroom_pass"]) == "True"
    S["gate"] = "PASS" if (S["validation_pass"] and S["headroom_pass"]) else "FAIL"
    pc = OUT / "posctrl_summary.csv"
    if pc.exists():
        P = pd.read_csv(pc, index_col=0).iloc[:, 0]
        S.update({f"posctrl_{k}": P[k] for k in ("headroom", "ci_lo", "ci_hi", "relative_headroom", "headroom_pass")})
    pd.Series(S).to_csv(OUT / "gate_summary.csv")
    print(pd.Series(S).to_string(), flush=True)


def reg():
    print("registered", register(SPEC)[:12], flush=True)


def all_():
    for step in (reg, calibrate, validate, headroom, posctrl, gate):
        print("=== step", step.__name__, time.strftime("%H:%M:%S"), flush=True)
        step()
    print("=== E2b-0 finished", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    {"register": reg, "calibrate": calibrate, "validate": validate, "headroom": headroom, "posctrl": posctrl,
     "gate": gate, "all": all_}[sys.argv[1]]()
