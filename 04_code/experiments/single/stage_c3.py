"""Stage C3 (after the 2026-10-21 review): the view learner of C2 (world.py, dqnv) with 8 NEW independent bases, a
recurrent baseline (GRU) and an online meta-gradient baseline. One add-on run per arm and base; base = unit.
C3-0  tune 8 new dqnv bases (C2-0 budget, random start points, new seeds).
C3    arms on the frozen bases; test on a new set; savings protocol (stage D) as secondary.
Every evaluation of an arm runs the 8 bases in two calls of 4 bases (common schedules, layouts and world noise
across arms; the world and agent random streams are separate in world.py). Resumable (per-arm lockstep CMA with a
checkpoint after every generation; evaluations cached).
Run: python stage_c3.py [base|arms [arm,arm,...]|test|savings|report|all]  (see 04_code/SERVER_RUN.md)"""
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
import stage_b3 as B3
import stage_b10 as B10
import stage_c as SC
import stage_c2 as C2
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

NB, POP, NAG, SIG = 8, 12, 48, 0.15
BGENS, AGENS = 30, 20
TCUT = R.T_LIFE                       # steps per life (smoke test only: shorter)
LOGS = B3.LOGS
DQN = SC.DQN
log = SC.log
BASE_SPACE = SC.BASE_SPACE
TEST = dict(sched=22000, bank=2_200_000, sim=110_000)
SAV = dict(bank=2_300_000, sim=120_000)
NS, NA_ = 128, 4
if os.environ.get("C3_SMOKE", "").strip():     # code check only: dev seeds, short lives, tiny batches, scratch output, no registration
    LOGS = Path(os.environ["C3_SMOKE"].strip())
    BGENS, AGENS, POP, NAG, TCUT, NS = 1, 1, 4, 4, 600, 8
    TEST = dict(sched=97000, bank=1_960_000, sim=130_000)
    SAV = dict(bank=1_960_000, sim=131_000)
    register = lambda spec: "smoke-no-registration"

META_SPACE = {"m_a": (-3.0, 3.0, "lin"), "m_e": (-6.0, 6.0, "lin"), "m_s": (-6.0, 6.0, "lin"),
              "mg_lr": (1e-6, 0.3, "log")}
ARMS = {  # name: (learner, space, dqn cfg extra, start)
    "control": ("dqnv", BASE_SPACE, {}, "base"),
    "aug": ("dqnv", SC.XS, {}, 0.5),
    "film": ("dqnv", C2.FS, {}, 0.5),
    "generic": ("dqnv", SC.GW, {}, 0.5),
    "bias": ("dqnv", C2.ARMS["bias"], {}, 0.5),
    "gru": ("gru", BASE_SPACE, {}, "base"),
    "gru_aug": ("gru", {**BASE_SPACE, **SC.XS}, {}, "base+0.5"),
    "metagrad": ("dqnv", META_SPACE, {"meta": True}, [0.5, 0.5, 0.5, 0.45]),
}
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
SPEC_BASE = {
    "id": "single_stageC3_base",
    "registered": "after the 2026-10-21 review and stage B10/B11, before any C3 run; dev smoke tests only",
    "learner": "dqnv of stage C2 (unchanged; the dqnv path of world.py is bit-identical to C2 after the C3 additions)",
    "space": {k: list(v) for k, v in BASE_SPACE.items()},
    "budget": ("C2-0 budget for each of 8 NEW bases k = 0..7: CMA-ES 30 generations x 12 x 48, sigma 0.15; start u ~ "
               "U(0.1, 0.9) from default_rng(26100 + k) (random starts for independence; C2-0 started all at 0.5); CMA "
               "seed 26000 + k; generation seeds 60000000 + 1000 k + g (training schedules and layouts, bank "
               "400000-401999); simulator seed 60000000 + 10 g + c (c = call, bases 0-3 / 4-7)"),
    "check": "parameters within 2% of a bound reported (C-stage convention)",
}
SPEC_ARMS = {
    "id": "single_stageC3_independent_bases_gru_metagrad",
    "registered": "after C3-0 (bases frozen), before any C3 arm run",
    "question": ("With 8 independent view-learner bases: do the appraisal signals help as network input or FiLM, do they "
                 "help beyond a recurrent network that can build its own memory of recent experience (GRU), and how "
                 "does an online meta-gradient on the knobs compare?"),
    "arms": {
        "control": "no signals: re-tune the 5 base parameters from the base",
        "aug": "signals as network input, scales xs (0.03, 30) log, start ~1; 5 params",
        "film": "FiLM of the hidden layer by the signals (C2), scales fs (0.03, 30) log, start ~1; 5 params",
        "generic": "knob-level global layer, 15 weights (-6, 6), start 0",
        "bias": "3 constant knob offsets (-6, 6), start 0",
        "gru": ("hidden layer replaced by a GRU (32 units) fed the view, coordinates, needs, the last felt reward and the "
                "last action; no appraisal signals; stored-state sequence replay, 2 sequences x 4 steps per update (8 "
                "transitions, as dqnv), gradients by autograd through the 4 steps; re-tune the 5 base parameters from "
                "the dqnv base"),
        "gru_aug": "gru plus the 5 signals as input: 5 base parameters (start at the base) + xs (start ~1); 10 params",
        "metagrad": ("online one-step meta-gradient of the learning rate through the differentiable update (Xu et al. "
                     "2018; as in STAC, Zahavy et al. 2020): after each Adam step, d L_val / d log-lr offset = "
                     "-<grad L_val(net'), lr * direction> on an independent replayed minibatch, RMS-normalised, step "
                     "mg_lr; exploration and safety weight have no gradient through the value loss and get constant "
                     "offsets. CMA tunes m_a (-3, 3) initial lr offset, m_e, m_s (-6, 6), mg_lr (1e-6, 0.3) log; start "
                     "offsets 0 and mg_lr ~3e-4 at u = 0.45 (near-zero effect, away from the bound; a dev smoke start at u = 0.1 "
                     "hit the lower bound in 3 of 8 bases); 4 params")},
    "budget": ("each arm on each base: CMA-ES 20 x 12 x 48, sigma 0.15; CMA seed 27000 + 100 * arm index + k; generation "
               "seeds 61000000 + 1000 k + g and simulator seed 61000000 + 10 g + c shared by all arms (common random "
               "numbers)"),
    "test": ("new independent set: 128 schedules (seeds 22000-22127) x 4 agents per base, layout bank 2200000-2200999, "
             "simulator seed 110000 + c (c = 0 for bases 0-3, 1 for bases 4-7), identical for all arms"),
    "predictions": {"P1": "film > control", "P2": "aug > control", "P3": "film > gru", "P4": "aug > gru",
                    "P5": "generic < aug (tested as aug - generic > 0)"},
    "secondary": {"S1": "gru_aug > gru (one-sided)", "S2": "metagrad vs control (two-sided)",
                  "S3": "gru vs control (two-sided)",
                  "bias": "generic, aug, film, metagrad vs bias (two-sided, Holm within this family)",
                  "savings": ("stage-D savings protocol (death disabled; safe / X three times, X = danger, "
                              "uncontrollable, scarce, volatile), layouts 2300000-2300999, simulator 120000 + 1000 X + c; "
                              "re-exposure R and savings S = R - e1 per arm, differences to control, CI only"),
                  "deaths": "survival, wellbeing while alive, deaths by regime on the test set (descriptive)",
                  "bounds": "parameters within 2% of a bound reported for every base and arm"},
    "inference": ("as B10: d = difference of agent-averaged score per (base, schedule); unit = base (n = 8): one-sided "
                  "t-test on base means, Holm over P1-P5 (and within each secondary family), 95% three-level bootstrap "
                  "CI (reanalysis_runlevel.ci3); a claim needs p_Holm < 0.05 AND the CI excluding 0. Registered checks "
                  "reported beside it: exact Wilcoxon on base means and the schedule-level one-sided Wilcoxon with the "
                  "hierarchical CI (stage_b3.hier_ci), Holm over P1-P5"),
    "interpretation": ("P3/P4 pass: the signals add information a recurrent learner does not build for itself from "
                       "rewards and actions within the budget. P3/P4 fail with gru >= film/aug: the C2 gain may be "
                       "replaceable by memory. P5 replicates the C2 finding that the knob layer is worse than input. S1 "
                       "asks whether the signals still help a recurrent learner. Caveat fixed in advance: gru also "
                       "receives the last reward and action, which the feed-forward arms do not."),
}
BASE_OUT, ARM_OUT, CACHE = LOGS / "stageC3_base", LOGS / "stageC3_opt", LOGS / "stageC3_cache"


def calls(nbases):
    """bases per simulator call: two calls of 4 bases (memory of the GRU learner on a 4 GB GPU)"""
    h = (nbases + 1) // 2
    return [list(range(0, h)), list(range(h, nbases))]


def simulate_blocks(learner, cfg, blocks, seed_of_call, record=()):
    """blocks: list of (base k, params list, schedules, layouts) in base order; runs calls of 4 bases."""
    out = {}
    for c, ks in enumerate(calls(NB)):
        bl = [b for b in blocks if b[0] in ks]
        if not bl:
            continue
        sched, lays, prms = [], [], []
        for k, plist, sch, lay in bl:
            for p in plist:
                sched += sch
                lays.append(lay)
                prms.append(p)
        nag = len(bl[0][2])
        d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nag) for k_ in keys}
        M = A.regime_matrix(sched, R.T_LIFE)[:TCUT]
        o = W.simulate(learner, P, M, B3.bank(), np.concatenate(lays), seed=seed_of_call(c), dqn={**DQN, **cfg},
                       record=record)
        i = 0
        for k, plist, sch, lay in bl:
            n = len(plist) * nag
            out[k] = {kk: (v[i:i + n] if v.ndim == 1 else v[:, i:i + n]) for kk, v in o.items()}
            i += n
    return out


_BANKS = {}


def score_fn(learner, cfg, seed_base, tag):
    def score_all(items, g):
        blocks = []
        for run, X in items:
            rng = np.random.default_rng(seed_base + 1000 * run["r"] + g)
            sch = [R.schedule(rng) for _ in range(NAG)]
            lay = rng.integers(len(B3.bank()), size=NAG)
            blocks.append((run["r"], [run["make"](x) for x in X], sch, lay))
        t = time.time()
        o = simulate_blocks(learner, cfg, blocks, lambda c: seed_base + 10 * g + c)
        log(f"{tag} gen {g}: {sum(len(b[1]) for b in blocks) * NAG} agents in {time.time() - t:.0f}s")
        return [o[run["r"]]["score"].reshape(-1, NAG).mean(1) for run, _ in items]
    return score_all


def base():
    log("registered", register(SPEC_BASE)[:12])
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    runs = [dict(out_json=BASE_OUT / f"dqnv_r{k}.json", x0=np.random.default_rng(26100 + k).uniform(0.1, 0.9, 5),
                 seed=26000 + k, r=k, make=lambda u: {"gamma": 0.9, **decode_space(BASE_SPACE, u)},
                 meta=dict(arm="dqnv_base", run=k)) for k in range(NB)]
    res = cma_lockstep(runs, BGENS, score_fn("dqnv", {}, 60_000_000, "C3-0"), BASE_OUT / "base.ckpt", POP, SIG)
    for k, j in enumerate(res):
        log("base", k, {kk: round(j["params"][kk], 5) for kk in BASE_SPACE}, "on bound:",
            [kk for kk, ui in zip(BASE_SPACE, j["u"]) if ui < 0.02 or ui > 0.98], "last", round(j["history"][-1], 4))


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
            x0 = np.asarray(start, float) if isinstance(start, list) else np.full(len(space), start)
            make = (lambda sp, bb: (lambda u: {**bb, **decode_space(sp, u)}))(space, b)
        runs.append(dict(out_json=ARM_OUT / f"{arm}_r{k}.json", x0=x0, seed=27000 + 100 * ARM_IDX[arm] + k, r=k,
                         make=make, meta=dict(arm=arm, run=k, learner=learner, cfg=cfg)))
    return runs


def arms(names=None):
    """names: comma-separated subset (e.g. 'gru,gru_aug') to run arms in parallel processes; results do not depend
    on the split (each arm has its own seeds, checkpoint and outputs)."""
    log("registered", register(SPEC_ARMS)[:12])
    ARM_OUT.mkdir(parents=True, exist_ok=True)
    for arm, (learner, space, cfg, start) in ARMS.items():
        if names and arm not in names.split(","):
            continue
        cma_lockstep(arm_runs(arm), AGENS, score_fn(learner, cfg, 61_000_000, f"C3 {arm}"), ARM_OUT / f"{arm}.ckpt",
                     POP, SIG)
        log(f"C3 arm {arm} done")


def arm_load(arm, k):
    return json.loads((ARM_OUT / f"{arm}_r{k}.json").read_text())


def evaluate(name, arm, M, lay, bank_lo, seed, juvenile=500):
    """All 8 bases of one arm on schedules M (T, nb) and layouts lay; cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_{arm}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    learner, _, cfg, _ = ARMS[arm]
    key = f"bank{bank_lo}"
    if key not in _BANKS:
        _BANKS[key] = SC._bank(bank_lo)
    out = {}
    for c, ks in enumerate(calls(NB)):
        prms = [arm_load(arm, k)["params"] for k in ks]
        nb = M.shape[1]
        d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nb) for k_ in keys}
        t = time.time()
        rec = ("wellbeing", "alpha") if arm == "metagrad" else ("wellbeing",)
        o = W.simulate(learner, P, np.tile(M, (1, len(ks))), _BANKS[key], np.tile(lay, len(ks)), seed=seed + c,
                       dqn={**DQN, **cfg}, record=rec, juvenile=juvenile)
        for i, k in enumerate(ks):
            out[k] = {kk: (v[i * nb:(i + 1) * nb] if v.ndim == 1 else v[:, i * nb:(i + 1) * nb]) for kk, v in o.items()}
        log(f"{name} {arm} bases {ks}: {len(ks) * nb} agents, {time.time() - t:.0f}s")
    _atomic_write(f, pickle.dumps(out))
    return out


def test():
    log("registered", register(SPEC_ARMS)[:12])
    sch = [R.schedule(np.random.default_rng(TEST["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    rows = []
    for arm in ARMS:
        for k, o in evaluate("test", arm, M, lay, TEST["bank"], TEST["sim"]).items():
            wb = o["wellbeing"]
            df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
            df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
            life = np.round(o["survival"] * TCUT).astype(int)
            died = life < TCUT
            rd = M[np.clip(life - 1, 0, TCUT - 1), np.arange(M.shape[1])]
            for j, rn in enumerate(R.NAMES):
                m = M == j
                df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
                df[f"died_{rn}"] = (died & (rd == j)).astype(float)
            if "alpha" in o:
                df["mean_log_alpha"] = np.log(np.maximum(o["alpha"], 1e-12)).mean(0)
            df["arm"], df["run"] = arm, k
            rows.append(df)
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC3_test_raw.csv", index=False)
    log("test raw written", len(D))


XS_REG = ["danger", "uncontrollable", "scarce", "volatile"]
SEG = [(0, 1200, "safe"), (1200, 1800, "X"), (1800, 2400, "safe"), (2400, 3000, "X"), (3000, 3600, "safe"),
       (3600, 4200, "X")]
EXPO = [1200, 2400, 3600]


def savings():
    rows = []
    NL = NS
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NL)])
    T_ = min(4200, TCUT) if os.environ.get("C3_SMOKE", "").strip() else 4200
    for xi, X in enumerate(XS_REG):
        M = np.zeros((4200, NL * NA_), int)
        for a_, b_, nm in SEG:
            M[a_:b_] = R.NAMES.index(X if nm == "X" else nm)
        if os.environ.get("C3_SMOKE", "").strip():
            M = M[[i % 4200 for i in range(1150, 1150 + T_)]]
        for arm in ARMS:
            for k, o in evaluate(f"savings_{X}", arm, M, lay, SAV["bank"], SAV["sim"] + 1000 * xi,
                                 juvenile=M.shape[0] + 1).items():
                wb = o["wellbeing"]
                d = dict(arm=arm, run=k, regime=X, layout=np.repeat(np.arange(NL), NA_))
                for i, e in enumerate(EXPO):
                    e_ = e if not os.environ.get("C3_SMOKE", "").strip() else 50
                    d[f"e{i + 1}"] = wb[e_:e_ + 200].mean(0)
                rows.append(pd.DataFrame(d))
    D = pd.concat(rows)
    D["R"] = (D.e2 + D.e3) / 2
    D["S"] = D.R - D.e1
    D.to_csv(LOGS / "stageC3_savings_raw.csv", index=False)
    log("savings raw written", len(D))


PRIMARY = [("P1", "film", "control", "greater"), ("P2", "aug", "control", "greater"), ("P3", "film", "gru", "greater"),
           ("P4", "aug", "gru", "greater"), ("P5", "aug", "generic", "greater")]
SECONDARY = [("S1", "gru_aug", "gru", "greater"), ("S2", "metagrad", "control", "two-sided"),
             ("S3", "gru", "control", "two-sided")]
BIAS = [("B1", "generic", "bias", "two-sided"), ("B2", "aug", "bias", "two-sided"), ("B3", "film", "bias", "two-sided"),
        ("B4", "metagrad", "bias", "two-sided")]


def report():
    D = pd.read_csv(LOGS / "stageC3_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, PRIMARY, "primary"), B10.family(piv, SECONDARY, "secondary"),
                   B10.family(piv, BIAS, "secondary_bias")], ignore_index=True)
    T.to_csv(LOGS / "stageC3_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC3_summary.csv")
    bm = D.groupby(["arm", "run"]).score.mean().unstack(0)
    bm.to_csv(LOGS / "stageC3_base_means.csv")
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
    Pm.to_csv(LOGS / "stageC3_params.csv", index=False)
    sv = LOGS / "stageC3_savings_raw.csv"
    Sv = None
    if sv.exists():
        Sv = pd.read_csv(sv)
        srows = []
        for meas in ("R", "S"):
            pv = Sv.groupby(["arm", "run", "layout"])[meas].mean()
            for a in ARMS:
                if a != "control":
                    d = (pv[a] - pv["control"]).unstack()
                    lo, hi = B10.ci3(d.values, np.asarray(d.index))
                    srows.append(dict(measure=meas, comparison=f"{a} - control", diff=d.values.mean(), ci3_lo=lo,
                                      ci3_hi=hi, bases_better=int((d.mean(1) > 0).sum())))
        pd.DataFrame(srows).to_csv(LOGS / "stageC3_savings_tests.csv", index=False)
        Sv.groupby(["arm", "regime"])[["e1", "e2", "e3", "R", "S"]].mean().to_csv(LOGS / "stageC3_savings_summary.csv")
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S.round(4).T.to_string())
    print(bm.round(3).to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())
    if Sv is not None:
        print(pd.read_csv(LOGS / "stageC3_savings_tests.csv").round(4).to_string())


def all_():
    for step in (base, arms, test, savings, report):
        log("=== step", step.__name__)
        step()
    log("=== C3 finished")


if __name__ == "__main__":
    if sys.argv[1] == "arms" and len(sys.argv) > 2:
        arms(sys.argv[2])
    else:
        {"base": base, "arms": arms, "test": test, "savings": savings, "report": report, "all": all_}[sys.argv[1]]()
