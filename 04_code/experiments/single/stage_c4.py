"""Stage C4 (second environment E2b; only after the E2b-0 gate passed): the C3 design in E2b with 8 new view-learner
bases and 10 arms (C3, C3b and C3c arms together), one add-on run per arm and base; base = unit.
  register_base  register the C4-0 spec (commit the yaml before running)
  base           C4-0: tune 8 new dqnv bases in E2b (C2-0/C3-0 budget, random starts)
  register_arms  register the C4 arms spec (after C4-0 is frozen; commit the yaml before running the arms)
  arms [a,b,..]  C4 arms on the frozen bases (subset: run arms in parallel processes; results do not depend on the split)
  test           all arms on a new E2b test set
  report         tests (stage_b10.family), summary, deaths by regime, parameters on bounds
World: E2 layouts (build_e2), schedule_e2, T_LIFE_E2, JUVENILE_E2, view radius 3 (7 x 7), energy decay and regime table
from stageE2b_0/calibration.json. The script refuses to run unless stageE2b_0/gate_summary.csv says gate = PASS.
Bases per GPU call: env C4_BPC (default 4 = two calls, as C3). The value is written into both specs, so it must be
chosen before registering and kept for the whole stage (a different value changes the batch composition and the
noise, and register() refuses a changed spec).
Run: python stage_c4.py <step>   Smoke: C4_SMOKE=<scratch dir> [C4_E2B=<dir with calibration.json>] python stage_c4.py all"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b10 as B10
import stage_c as SC
import stage_c2 as C2
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.fast.popsim import LayoutBank
from bio_rl.torchsim import world as W

LOGS = ROOT / "results" / "logs" / "single"
E2B = Path(os.environ.get("C4_E2B", "").strip() or LOGS / "stageE2b_0")
NB, POP, NAG, SIG = 8, 12, 48, 0.15
BGENS, AGENS = 30, 20
BPC = int(os.environ.get("C4_BPC", "4") or 4)
TCUT = R.T_LIFE_E2
H_BIG = 105                        # ff_big: 272 H + 10 = 28,570 weights vs GRU 28,650 (input 261, 7 x 7 view)
TRAIN_BANK = 3_810_000
TEST = dict(sched=48000, bank=3_800_000, sim=260_000)
NS, NA_ = 128, 4
SEED = dict(base_cma=36000, base_start=36100, base_gen=72_000_000, arm_cma=37000, arm_gen=73_000_000)
log = SC.log
SMOKE = os.environ.get("C4_SMOKE", "").strip()
if SMOKE:                          # code check only: dev seeds, short lives, tiny batches, scratch output, no registration
    LOGS = Path(SMOKE)
    BGENS, AGENS, POP, NAG, TCUT, NS = 1, 1, 4, 4, 600, 8
    TRAIN_BANK = 3_990_000
    TEST = dict(sched=98500, bank=3_991_000, sim=99_000)
    register = lambda spec: "smoke-no-registration"
BASE_SPACE = SC.BASE_SPACE
DQN = {**SC.DQN, "view": 3}
ARMS = {  # name: (learner, space, dqn cfg extra, start)
    "control": ("dqnv", BASE_SPACE, {}, "base"),
    "control_ra": ("dqnv", BASE_SPACE, {"ra": True}, "base"),
    "aug": ("dqnv", SC.XS, {}, 0.5),
    "aug_ra": ("dqnv", {**BASE_SPACE, **SC.XS}, {"ra": True}, "base+0.5"),
    "film": ("dqnv", C2.FS, {}, 0.5),
    "generic": ("dqnv", SC.GW, {}, 0.5),
    "bias": ("dqnv", C2.ARMS["bias"], {}, 0.5),
    "gru": ("gru", BASE_SPACE, {}, "base"),
    "gru_aug": ("gru", {**BASE_SPACE, **SC.XS}, {}, "base+0.5"),
    "ff_big": ("dqnv", BASE_SPACE, {"ra": True, "H": H_BIG}, "base"),
}
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
BASE_OUT, ARM_OUT, CACHE = LOGS / "stageC4_base", LOGS / "stageC4_opt", LOGS / "stageC4_cache"

SPEC_BASE = {
    "id": "single_stageC4_base",
    "registered": "after the E2b-0 gate passed (single_stageE2b_0_gate), before any C4 run; smoke tests only",
    "world": ("E2b: E2 layouts (12 x 12, 3 food, 2 threat clusters of 7), schedules 800-2000 steps, life 9000, juvenile "
              "750, energy decay and regime table from stageE2b_0/calibration.json; view radius 3 (7 x 7 x 5 inputs)"),
    "learner": "dqnv of stage C2/C3 (hidden 32, replay 1000, minibatch 8, C = 1) with view radius 3",
    "space": {k: list(v) for k, v in BASE_SPACE.items()},
    "budget": ("as C3-0 for each of 8 NEW bases k = 0..7: CMA-ES 30 generations x 12 x 48, sigma 0.15; start u ~ "
               "U(0.1, 0.9) from default_rng(36100 + k); CMA seed 36000 + k; generation seeds 72000000 + 1000 k + g "
               "(training schedules and layouts, E2 bank 3810000-3811999); simulator seed 72000000 + 10 g + c "
               f"(c = call index; {BPC} bases per call)"),
    "bases_per_call": BPC,
    "check": "parameters within 2% of a bound reported",
}
SPEC_ARMS = {
    "id": "single_stageC4_arms_e2b",
    "registered": "after C4-0 (bases frozen), before any C4 arm run",
    "question": ("Do the C3-C3c findings hold in a second, larger environment (E2b): do the appraisal signals help as "
                 "network input, does a recurrent learner beat a feed-forward one with the same inputs and with "
                 "matched capacity, and do the signals help a learner that sees its last reward and action?"),
    "arms": {
        "control": "no signals: re-tune the 5 base parameters from the base",
        "control_ra": "control + last felt reward and last action as input (as C3b); re-tune the 5 base parameters",
        "aug": "signals as network input, scales xs (0.03, 30) log, start ~1; 5 params",
        "aug_ra": "control_ra + signals as input; 5 base parameters (start at the base) + xs (start ~1); 10 params",
        "film": "FiLM of the hidden layer by the signals (C2), scales fs (0.03, 30) log, start ~1; 5 params",
        "generic": "knob-level global layer, 15 weights (-6, 6), start 0",
        "bias": "3 constant knob offsets (-6, 6), start 0 (constant-offset control)",
        "gru": "GRU (32 units) fed view, coordinates, needs, last felt reward and last action, as C3; 5 base parameters",
        "gru_aug": "gru + the 5 signals as input; 5 base parameters + xs; 10 params",
        "ff_big": (f"control_ra with hidden width {H_BIG} (272 H + 10 = 28,570 weights vs 28,650 for the GRU; input 261; "
                   "unused FiLM matrices excluded); replay scheme not matched (as C3c); 5 base parameters")},
    "budget": ("each arm on each base: CMA-ES 20 x 12 x 48, sigma 0.15; CMA seed 37000 + 100 * arm index + k (arm "
               "index in the order above); generation seeds 73000000 + 1000 k + g and simulator seed 73000000 + 10 g "
               f"+ c shared by all arms (common random numbers); {BPC} bases per call"),
    "bases_per_call": BPC,
    "test": ("new independent set, never used: 128 E2b schedules (seeds 48000-48127) x 4 agents per base, E2 layout bank "
             "3800000-3800999, simulator seed 260000 + c (c = call index), identical for all arms"),
    "predictions": {"P1": "aug > control", "P2": "gru > control_ra", "P3": "gru > ff_big (memory beyond capacity)",
                    "P4": "aug_ra > control_ra"},
    "secondary": {"S1": "film vs control", "S2": "film vs aug", "S3": "generic vs aug", "S4": "aug vs bias",
                  "S5": "film vs bias", "S6": "gru_aug vs gru", "S7": "ff_big vs control_ra", "S8": "gru vs aug",
                  "note": "all two-sided, Holm within the secondary family",
                  "deaths": "survival, wellbeing while alive, deaths by regime (descriptive)",
                  "bounds": "parameters within 2% of a bound reported for every base and arm"},
    "inference": ("as C3/B10: d = difference of agent-averaged score per (base, schedule); unit = base (n = 8): "
                  "one-sided t-test on base means for P1-P4, Holm over P1-P4; two-sided for S1-S8, Holm within; 95% "
                  "three-level bootstrap CI (stage_b10.ci3); a claim needs p_Holm < 0.05 AND the CI excluding 0. "
                  "Checks reported: exact Wilcoxon on base means, schedule-level Wilcoxon with hierarchical CI"),
    "interpretation": ("P1-P4 replicate (or not) the C3, C3b and C3c findings of the original world in E2b. A failure "
                       "is reported as a failure to replicate in E2b, not as evidence of no effect. No detectable "
                       "difference is not equivalence."),
}


def world():
    """(regime table, decay) of E2b; refuses unless the E2b-0 gate passed (smoke: gate not checked)."""
    if not SMOKE:
        if not (E2B / "gate_summary.csv").exists():
            raise SystemExit(f"{E2B / 'gate_summary.csv'} missing: run E2b-0 first; C4 must not be run without a PASS")
        g = pd.read_csv(E2B / "gate_summary.csv", index_col=0).iloc[:, 0]
        if g.get("gate") != "PASS":
            raise SystemExit(f"E2b-0 gate is not PASS ({g.get('gate')}): C4 must not be run (see RUNBOOK_fa.md)")
    c = json.loads((E2B / "calibration.json").read_text())
    return np.array(c["reg_tab"], float), float(c["decay"])


_BANKS = {}


def bank(lo, n=1000):
    if lo not in _BANKS:
        _BANKS[lo] = LayoutBank.build_e2(range(lo, lo + n))
    return _BANKS[lo]


def calls():
    return [list(range(i, min(i + BPC, NB))) for i in range(0, NB, BPC)]


def sim(learner, cfg, P, M, bk, lay, seed, record=()):
    tab, dec = world()
    return W.simulate(learner, P, M, bk, lay, seed=seed, dqn={**DQN, **cfg}, record=record, juvenile=R.JUVENILE_E2,
                      reg_tab=tab, decay=dec)


def stack_params(prms, nag):
    d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
    keys = set(d0) | set().union(*prms)
    return {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nag) for k_ in keys}


def score_fn(learner, cfg, seed_base, tag):
    def score_all(items, g):
        blocks = {}
        for run, X in items:
            rng = np.random.default_rng(seed_base + 1000 * run["r"] + g)
            sch = [R.schedule_e2(rng) for _ in range(NAG)]
            lay = rng.integers(len(bank(TRAIN_BANK, 2000)), size=NAG)
            blocks[run["r"]] = ([run["make"](x) for x in X], sch, lay)
        t = time.time()
        out = {}
        for c, ks in enumerate(calls()):
            ks = [k for k in ks if k in blocks]
            if not ks:
                continue
            prms = [p for k in ks for p in blocks[k][0]]
            sched = [s for k in ks for _ in blocks[k][0] for s in blocks[k][1]]
            lays = np.concatenate([blocks[k][2] for k in ks for _ in blocks[k][0]])
            M = A.regime_matrix(sched, R.T_LIFE_E2)[:TCUT]
            o = sim(learner, cfg, stack_params(prms, NAG), M, bank(TRAIN_BANK, 2000), lays, seed_base + 10 * g + c)
            i = 0
            for k in ks:
                n = len(blocks[k][0]) * NAG
                out[k] = o["score"][i:i + n].reshape(-1, NAG).mean(1)
                i += n
        log(f"{tag} gen {g}: {sum(len(b[0]) for b in blocks.values()) * NAG} agents in {time.time() - t:.0f}s")
        return [out[run["r"]] for run, _ in items]
    return score_all


def on_bound(u):
    return [i for i, ui in enumerate(u) if ui < 0.02 or ui > 0.98]


def reg_base():
    world()
    log("registered", register(SPEC_BASE)[:12])


def base():
    reg_base()
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    runs = [dict(out_json=BASE_OUT / f"dqnv_r{k}.json",
                 x0=np.random.default_rng(SEED["base_start"] + k).uniform(0.1, 0.9, 5), seed=SEED["base_cma"] + k, r=k,
                 make=lambda u: {"gamma": 0.9, **decode_space(BASE_SPACE, u)}, meta=dict(arm="dqnv_base", run=k))
            for k in range(NB)]
    res = cma_lockstep(runs, BGENS, score_fn("dqnv", {}, SEED["base_gen"], "C4-0"), BASE_OUT / "base.ckpt", POP, SIG)
    for k, j in enumerate(res):
        log("base", k, {kk: round(j["params"][kk], 5) for kk in BASE_SPACE}, "on bound:",
            [list(BASE_SPACE)[i] for i in on_bound(j["u"])], "last", round(j["history"][-1], 4))


def load_base(k):
    return json.loads((BASE_OUT / f"dqnv_r{k}.json").read_text())["params"]


def arm_runs(arm):
    learner, space, cfg, start = ARMS[arm]
    runs = []
    for k in range(NB):
        b = load_base(k)
        if start == "base":
            x0 = encode(space, b)
            make = (lambda sp: (lambda u: {"gamma": 0.9, **decode_space(sp, u)}))(space)
        elif start == "base+0.5":
            x0 = np.concatenate([encode(BASE_SPACE, b), np.full(len(space) - len(BASE_SPACE), 0.5)])
            make = (lambda sp: (lambda u: {"gamma": 0.9, **decode_space(sp, u)}))(space)
        else:
            x0 = np.full(len(space), start)
            make = (lambda sp, bb: (lambda u: {**bb, **decode_space(sp, u)}))(space, b)
        runs.append(dict(out_json=ARM_OUT / f"{arm}_r{k}.json", x0=x0, seed=SEED["arm_cma"] + 100 * ARM_IDX[arm] + k,
                         r=k, make=make, meta=dict(arm=arm, run=k, learner=learner, cfg=cfg)))
    return runs


def reg_arms():
    world()
    assert all((BASE_OUT / f"dqnv_r{k}.json").exists() for k in range(NB)), "C4-0 bases missing: run 'base' first"
    log("registered", register(SPEC_ARMS)[:12])


def arms(names=None):
    reg_arms()
    ARM_OUT.mkdir(parents=True, exist_ok=True)
    for arm, (learner, _, cfg, _) in ARMS.items():
        if names and arm not in names.split(","):
            continue
        cma_lockstep(arm_runs(arm), AGENS, score_fn(learner, cfg, SEED["arm_gen"], f"C4 {arm}"),
                     ARM_OUT / f"{arm}.ckpt", POP, SIG)
        log(f"C4 arm {arm} done")


def arm_load(arm, k):
    return json.loads((ARM_OUT / f"{arm}_r{k}.json").read_text())


def evaluate(name, arm, M, lay):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_{arm}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    learner, _, cfg, _ = ARMS[arm]
    out = {}
    nb = M.shape[1]
    for c, ks in enumerate(calls()):
        prms = [arm_load(arm, k)["params"] for k in ks]
        t = time.time()
        o = sim(learner, cfg, stack_params(prms, nb), np.tile(M, (1, len(ks))), bank(TEST["bank"]),
                np.tile(lay, len(ks)), TEST["sim"] + c, record=("wellbeing",))
        for i, k in enumerate(ks):
            out[k] = {kk: (v[i * nb:(i + 1) * nb] if v.ndim == 1 else v[:, i * nb:(i + 1) * nb]) for kk, v in o.items()}
        log(f"{name} {arm} bases {ks}: {len(ks) * nb} agents, {time.time() - t:.0f}s")
    _atomic_write(f, pickle.dumps(out))
    return out


def frame(arm, k, o, M):
    T_ = M.shape[0]
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
    df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
    life = np.round(o["survival"] * T_).astype(int)
    died = life < T_
    rd = M[np.clip(life - 1, 0, T_ - 1), np.arange(M.shape[1])]
    for j, rn in enumerate(R.NAMES):
        m = M == j
        df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
        df[f"died_{rn}"] = (died & (rd == j)).astype(float)
    df["arm"], df["run"] = arm, k
    return df


def test():
    reg_arms()
    sch = [R.schedule_e2(np.random.default_rng(TEST["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE_E2)[:TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    D = pd.concat([frame(a, k, o, M) for a in ARMS for k, o in evaluate("test", a, M, lay).items()])
    D.to_csv(LOGS / "stageC4_test_raw.csv", index=False)
    log("test raw written", len(D))


PRIMARY = [("P1", "aug", "control", "greater"), ("P2", "gru", "control_ra", "greater"),
           ("P3", "gru", "ff_big", "greater"), ("P4", "aug_ra", "control_ra", "greater")]
SECONDARY = [("S1", "film", "control", "two-sided"), ("S2", "film", "aug", "two-sided"),
             ("S3", "generic", "aug", "two-sided"), ("S4", "aug", "bias", "two-sided"),
             ("S5", "film", "bias", "two-sided"), ("S6", "gru_aug", "gru", "two-sided"),
             ("S7", "ff_big", "control_ra", "two-sided"), ("S8", "gru", "aug", "two-sided")]


def report():
    D = pd.read_csv(LOGS / "stageC4_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, PRIMARY, "primary"), B10.family(piv, SECONDARY, "secondary")], ignore_index=True)
    T.to_csv(LOGS / "stageC4_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC4_summary.csv")
    bm = D.groupby(["arm", "run"]).score.mean().unstack(0)
    bm.to_csv(LOGS / "stageC4_base_means.csv")
    rows = []
    for a, (_, sp, _, _) in ARMS.items():
        for k in range(NB):
            j = arm_load(a, k)
            for (kk, _), ui in zip(sp.items(), j["u"]):
                rows.append(dict(arm=a, base=k, param=kk, value=j["params"][kk], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    for k in range(NB):
        j = json.loads((BASE_OUT / f"dqnv_r{k}.json").read_text())
        for (kk, _), ui in zip(BASE_SPACE.items(), j["u"]):
            rows.append(dict(arm="base", base=k, param=kk, value=j["params"][kk], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageC4_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S.round(4).T.to_string())
    print(bm.round(3).to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (base, arms, test, report):
        log("=== step", step.__name__)
        step()
    log("=== C4 finished")


if __name__ == "__main__":
    if sys.argv[1] == "arms" and len(sys.argv) > 2:
        arms(sys.argv[2])
    else:
        {"register_base": reg_base, "base": base, "register_arms": reg_arms, "arms": arms, "test": test,
         "report": report, "all": all_}[sys.argv[1]]()
