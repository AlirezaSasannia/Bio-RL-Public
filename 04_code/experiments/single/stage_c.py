"""Phase 2 (roadmap step 4a): function approximation on the GPU.
C0  tune the base hyperparameters of a small per-agent DQN-style learner with need-based utility (one value head per
    need, felt need change as reward, per-cell VDBE exploration) in the regime-switching world, as stage B did for VDBE.
C1  arms on the tuned base (registered after C0, before any C1 run).
World: bio_rl.torchsim.world (validated against numpy: single_torch_world_validation). All CMA runs of a stage advance
in lockstep (one GPU call per generation); checkpoint after every generation; evaluations cached per job.
Run: python stage_c.py [c0|c1|test|speed|report|all]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
from _resume import cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

LOGS = B3.LOGS
DQN = dict(H=32, M=1000, m=8, C=1, needs_in=1.0)
POP, NAG, SIG = 12, 48, 0.15
BASE_SPACE = {"alpha0": (1e-4, 0.1, "log"), "eps0": (0.0, 0.5, "lin"), "s0": (0.05, 20.0, "log"),
              "vdbe_sigma": (1e-3, 100.0, "log"), "q_init": (0.0, 1.0, "lin")}
SPEC_C0 = {
    "id": "single_stageC0_dqn_base",
    "registered": "after the torch world validation passed, before any C0 run; design choices from dev seeds only",
    "learner": ("per-agent MLP (input: one-hot cell 64, cell coordinates 2, own need levels 2, 5 appraisal-signal "
                "slots scaled by xs (0 here); hidden 32 ReLU; output 2 needs x 5 actions); action value = need "
                "weights (as in the table, safety weight s) . per-need values; reward = felt need change; replay "
                "1000 transitions per agent, minibatch 8 incl. the newest, every step; double-DQN argmax with no "
                "separate target network (C = 1; better than C = 50/200 on dev seeds); Adam direction scaled by the "
                "learning-rate knob (cap 0.1); output bias initialised at q_init (optimism); per-cell VDBE exploration "
                "driven by |w . delta| / vdbe_sigma"),
    "space": {k: list(v) for k, v in BASE_SPACE.items()},
    "budget": ("as stage B: 4 restarts x 30 generations x 12 candidates x 48 agents, sigma 0.15, start u = 0.5, "
               "CMA seeds 15000 + r, generation seed 30000000 + 1000 r + g, training layout bank 400000-401999"),
    "output": "the CMA mean of each restart = the 4 frozen DQN bases for C1",
    "check": "parameters within 2% of a bound are reported; if any base sits on a bound, widen and re-run under an amendment",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def score_all(items, g, seed_base, tag=""):
    """items: list of (run, X). Each run's candidates are scored on NAG agents with the run's own training schedules."""
    sched_blocks, lay_blocks, prm_blocks, sizes = [], [], [], []
    for run, X in items:
        rng = np.random.default_rng(seed_base + 1000 * run["r"] + g)
        sch = [R.schedule(rng) for _ in range(NAG)]
        lay = rng.integers(len(B3.bank()), size=NAG)
        cands = [run["make"](x) for x in X]
        for c in cands:
            sched_blocks += sch
            lay_blocks.append(lay)
            prm_blocks.append(c)
        sizes.append(len(cands))
    B = len(sched_blocks)
    d0 = {k: float(v[0]) for k, v in W.params(1).items()}
    keys = set(d0) | set().union(*prm_blocks)
    P = {k: np.repeat([c.get(k, d0.get(k, 0.0)) for c in prm_blocks], NAG) for k in keys}
    M = A.regime_matrix(sched_blocks, R.T_LIFE)
    t = time.time()
    o = W.simulate("dqn", P, M, B3.bank(), np.concatenate(lay_blocks), seed=seed_base + g, dqn=DQN)
    log(f"{tag} gen {g}: {B} agents in {time.time() - t:.0f}s")
    sc = o["score"].reshape(-1, NAG).mean(1)
    out, i = [], 0
    for s in sizes:
        out.append(sc[i:i + s])
        i += s
    return out


def c0():
    log("registered", register(SPEC_C0)[:12])
    out = LOGS / "stageC0_opt"
    out.mkdir(exist_ok=True)
    runs = [dict(out_json=out / f"dqn_r{r}.json", x0=np.full(len(BASE_SPACE), 0.5), seed=15000 + r, r=r,
                 make=lambda u: {**DQN_KNOBS0, **decode_space(BASE_SPACE, u)}, meta=dict(arm="dqn_base", run=r))
            for r in range(4)]
    res = cma_lockstep(runs, 30, lambda items, g: score_all(items, g, 30_000_000, "C0"), out / "c0.ckpt", POP, SIG)
    for r, j in enumerate(res):
        u = np.array(j["u"])
        log("base", r, {k: round(j["params"][k], 5) for k in BASE_SPACE}, "on bound:",
            [k for k, ui in zip(BASE_SPACE, u) if ui < 0.02 or ui > 0.98], "last", round(j["history"][-1], 4))


DQN_KNOBS0 = {"gamma": 0.9}

# ------------------------------------------------------------------ C1: arms on the tuned DQN bases
XS = {f"xs_{i}": (0.03, 30.0, "log") for i in range(5)}                 # start u = 0.5 -> scale ~0.95
GW = {f"w_{i}_{k}": (-6.0, 6.0, "lin") for i in range(5) for k in "aes"}
C1_ARMS = {"control": BASE_SPACE, "aug": XS, "generic": GW,
           "bias": {k: (-6.0, 6.0, "lin") for k in ("b_a", "b_e", "b_s")}, "aug_generic": {**XS, **GW}}
SPEC_C1 = {
    "id": "single_stageC1_fa_arms",
    "registered": "after C0 (bases frozen), before any C1 run",
    "question": ("With a learner that can generalise (DQN-style, C0), do the appraisal signals help more as network "
                 "input (state augmentation) or as a global modulator of learning rate, exploration and safety weight?"),
    "arms": {"control": "no signals: re-tune the 5 base parameters from the base (C0 space)",
             "aug": ("signals as network input: the 5 appraisal signals (clipped 0-3, /3) enter the network with "
                     "per-input scales xs in (0.03, 30) log, start ~1 (signals on); 5 params"),
             "generic": "generic global layer on the 3 knobs: 15 weights in (-6, 6), start 0",
             "bias": "3 constant knob offsets in (-6, 6), start 0",
             "aug_generic": "signals as input and global layer: 20 params, start xs ~1, weights 0"},
    "budget": ("each arm 8 runs on DQN base r % 4, CMA-ES 20 generations x 12 x 48, sigma 0.15; CMA seeds 16000 + "
               "100 * arm index + r; generation seed 31000000 + 1000 r + g (shared by all arms of run r)"),
    "test": ("new independent set: 128 schedules (seeds 10000-10127) x 4 agents, layout bank 1300000-1300999, "
             "simulator seeds 83000 + run; common schedules and layouts across arms"),
    "predictions": {"G1": "generic > aug", "G2": "generic > bias", "G3": "generic > control",
                    "G4": "aug > control", "G5": "aug_generic > aug"},
    "inference": ("one-sided Wilcoxon over schedules of the run-averaged difference, Holm across G1-G5, plus "
                  "hierarchical bootstrap 95% CI (runs, then schedules); a claim needs both"),
    "secondary": {"two_sided": "aug - generic two-sided (if aug matches or beats generic, that is the finding)",
                  "decomposition": "survival, wellbeing while alive, per-regime score, deaths by regime",
                  "speed": ("B6 switch protocol, death disabled, layouts 1400000-1400999, simulator seeds 93000 + run; "
                            "arms - control per switch and window, CI only"),
                  "bounds": "parameters within 2% of a bound reported"},
    "interpretation": ("G1 passes: the tabular finding (modulator > state) holds with function approximation. G1 fails "
                       "with aug >= generic: with a learner that can generalise, giving the signals as input is as "
                       "good, and the global layer is not needed for the information. G2 tests signal dependence"),
}
ARM_IDX = {a: i for i, a in enumerate(C1_ARMS)}
C1_OUT = LOGS / "stageC1_opt"
C_CACHE = LOGS / "stageC_cache"
NS, NA_ = 128, 4
_BANKS = {}


def _bank(lo):
    if lo not in _BANKS:
        from bio_rl.fast.popsim import LayoutBank
        _BANKS[lo] = LayoutBank.build(range(lo, lo + 1000))
    return _BANKS[lo]


def load_base(r):
    return json.loads((LOGS / "stageC0_opt" / f"dqn_r{r % 4}.json").read_text())["params"]


def c1():
    log("registered", register(SPEC_C1)[:12])
    C1_OUT.mkdir(exist_ok=True)
    runs = []
    for arm, space in C1_ARMS.items():
        for r in range(8):
            base = load_base(r)
            if arm == "control":
                x0 = encode(space, base)
                make = (lambda sp: (lambda u: {**DQN_KNOBS0, **decode_space(sp, u)}))(space)
            else:
                x0 = np.full(len(space), 0.5)
                make = (lambda sp, b: (lambda u: {**b, **decode_space(sp, u)}))(space, base)
            runs.append(dict(out_json=C1_OUT / f"{arm}_r{r}.json", x0=x0, seed=16000 + 100 * ARM_IDX[arm] + r, r=r,
                             make=make, meta=dict(arm=arm, run=r, learner="dqn")))
    # two lockstep groups (runs 0-3, 4-7) to bound GPU memory and the length of a single job
    for lo in (0, 4):
        sub = [x for x in runs if lo <= x["r"] < lo + 4]
        cma_lockstep(sub, 20, lambda items, g: score_all(items, g, 31_000_000, f"C1 runs {lo}-{lo + 3}"),
                     C1_OUT / f"c1_{lo}.ckpt", POP, SIG)
    log("C1 optimisation done")


def c1_load(arm, r):
    return json.loads((C1_OUT / f"{arm}_r{r}.json").read_text())


def _run_all_arms(cache_name, r, M, lay, bank, seed, **kw):
    """All C1 arms of run r in one simulate call (same world noise for every arm); cached per (cache_name, r)."""
    import pickle
    from _resume import _atomic_write
    C_CACHE.mkdir(exist_ok=True)
    f = C_CACHE / f"{cache_name}_r{r}.pkl"
    arms = list(C1_ARMS)
    if not f.exists():
        prms = [c1_load(a, r)["params"] for a in arms]
        d0 = {k: float(v[0]) for k, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        nb = M.shape[1]
        P = {k: np.repeat([p.get(k, d0.get(k, 0.0)) for p in prms], nb) for k in keys}
        t = time.time()
        o = W.simulate("dqn", P, np.tile(M, (1, len(arms))), bank, np.tile(lay, len(arms)), seed=seed, dqn=DQN,
                       record=("wellbeing",), **kw)
        res = {a: dict(score=o["score"][i * nb:(i + 1) * nb], survival=o["survival"][i * nb:(i + 1) * nb],
                       wellbeing=o["wellbeing"][:, i * nb:(i + 1) * nb]) for i, a in enumerate(arms)}
        _atomic_write(f, pickle.dumps(res))
        log(f"{cache_name} run {r}: {len(arms) * nb} agents, {time.time() - t:.0f}s")
    return pickle.loads(f.read_bytes())


def test():
    sch = [R.schedule(np.random.default_rng(10000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    rows = []
    for r in range(8):
        res = _run_all_arms("test", r, M, lay, _bank(1_300_000), 83_000 + r)
        for a, o in res.items():
            wb = o["wellbeing"]
            df = pd.DataFrame(dict(schedule=np.repeat(np.arange(NS), NA_), score=o["score"], survival=o["survival"]))
            df["alive_wellbeing"] = df.score / np.maximum(df.survival, 1e-9)
            life = np.round(o["survival"] * R.T_LIFE).astype(int)
            died = life < R.T_LIFE
            rd = M[np.clip(life - 1, 0, R.T_LIFE - 1), np.arange(M.shape[1])]
            for j, rn in enumerate(R.NAMES):
                m = M == j
                df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
                df[f"died_{rn}"] = (died & (rd == j)).astype(float)
            df["arm"], df["run"] = a, r
            rows.append(df)
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC1_test_raw.csv", index=False)
    log("test raw written", len(D))


def speed():
    import stage_b6 as B6
    rows, curves = [], []
    for a_, b_ in B6.SW:
        B = B6.NL * NA_
        M = np.zeros((B6.PRE + B6.POST, B), int)
        M[:B6.PRE], M[B6.PRE:] = R.NAMES.index(a_), R.NAMES.index(b_)
        lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(B6.NL)])
        for r in range(8):
            res = _run_all_arms(f"speed_{a_}-{b_}", r, M, lay, _bank(1_400_000), 93_000 + r,
                                juvenile=B6.PRE + B6.POST + 1)
            for arm, o in res.items():
                wb = o["wellbeing"]
                d = dict(arm=arm, run=r, switch=f"{a_}->{b_}", layout=np.repeat(np.arange(B6.NL), NA_),
                         pre=wb[B6.PRE - 200:B6.PRE].mean(0))
                for lo, hi in B6.WIN:
                    d[f"w{lo}_{hi}"] = wb[B6.PRE + lo:B6.PRE + hi].mean(0)
                rows.append(pd.DataFrame(d))
                cv = wb[B6.PRE - 100:].reshape(-1, 10, wb.shape[1]).mean(1).mean(1)
                curves.append(dict(arm=arm, run=r, switch=f"{a_}->{b_}", **{f"b{i}": v for i, v in enumerate(cv)}))
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC1_speed_raw.csv", index=False)
    pd.DataFrame(curves).to_csv(LOGS / "stageC1_speed_curves.csv", index=False)
    T = []
    for s in D.switch.unique():
        X = D[D.switch == s]
        for col in ("pre", "w0_100", "w100_300", "w0_300", "w300_1000"):
            pv = X.groupby(["arm", "run", "layout"])[col].mean()
            for a in C1_ARMS:
                if a != "control":
                    d = (pv[a] - pv["control"]).unstack()
                    lo, hi = B3.hier_ci(d.values, n=2000)
                    T.append(dict(switch=s, window=col, comparison=f"{a} - control", diff=d.values.mean(),
                                  ci_lo=lo, ci_hi=hi))
    pd.DataFrame(T).to_csv(LOGS / "stageC1_speed_tests.csv", index=False)
    log("speed tests written")


def report():
    import stage_b5 as B5
    from bio_rl.analysis.stats import holm
    D = pd.read_csv(LOGS / "stageC1_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    prim = [("G1", "generic", "aug"), ("G2", "generic", "bias"), ("G3", "generic", "control"),
            ("G4", "aug", "control"), ("G5", "aug_generic", "aug")]
    T = pd.DataFrame([dict(pred=k, **B5.comp(piv, a, b)) for k, a, b in prim])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL")
    T["family"] = "primary"
    sec = pd.DataFrame([B5.comp(piv, a, b, "two-sided") for a, b in
                        (("aug", "generic"), ("bias", "control"), ("aug_generic", "generic"),
                         ("aug_generic", "control"), ("aug", "bias"))])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(LOGS / "stageC1_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC1_summary.csv")
    rows = []
    for a, sp in C1_ARMS.items():
        for r in range(8):
            j = c1_load(a, r)
            for (k, _), ui in zip(sp.items(), j["u"]):
                rows.append(dict(arm=a, run=r, param=k, value=j["params"][k], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageC1_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(S.round(4).T.to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_c1():
    for step in (c1, test, report, speed):
        log("=== step", step.__name__)
        step()
    log("=== C1 finished")


if __name__ == "__main__":
    {"c0": c0, "c1": c1, "test": test, "speed": speed, "report": report, "all": all_c1}[sys.argv[1]]()
