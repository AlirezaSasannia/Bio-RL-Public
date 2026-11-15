"""Path 1 (after paper v0.7): a network learner with an egocentric view, and modulation of computation (FiLM) versus
signals as input. Learner 'dqnv' of bio_rl.torchsim.world (5x5 view: ready food, regrowing food, shelter, outside
grid, own threat memory; cell coordinates and need levels; no one-hot cell).
C2-0  tune the base (5 params) as C0.
C2    arms on the tuned bases (registered after C2-0, before any C2 run).
Resumable: lockstep CMA with per-generation checkpoints; evaluations cached per (protocol, run).
Run: python stage_c2.py [base|arms|test|speed|report|all]"""
import json
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
import stage_c as SC
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

LOGS = B3.LOGS
DQN = SC.DQN
POP, NAG, SIG = 12, 48, 0.15
BASE_SPACE = SC.BASE_SPACE
log = SC.log
SPEC_BASE = {
    "id": "single_stageC2_base",
    "registered": "after paper v0.7 (C1), before any C2 run; dev smoke tests only (seeds 99970+, 99000+)",
    "learner": ("dqnv: as the C0 learner but the input is a 5x5 egocentric view (ready food, regrowing food, shelter, "
                "outside the grid, own remembered threat T_map; 125 inputs) + cell coordinates + need levels + 5 "
                "signal slots (scale xs, 0 here); no one-hot cell; optional FiLM of the hidden layer (scale fs, 0 here). "
                "Hidden 32, replay 1000, minibatch 8 incl. newest, no separate target net, Adam scaled by the "
                "learning-rate knob, optimistic output bias q_init, per-cell VDBE. Replayed views use stored layout "
                "and food readiness and the current threat memory"),
    "space": {k: list(v) for k, v in BASE_SPACE.items()},
    "budget": ("as C0: 4 restarts x 30 generations x 12 x 48, sigma 0.15, start u = 0.5, CMA seeds 17000 + r, "
               "generation seed 32000000 + 1000 r + g, training layout bank 400000-401999"),
    "check": "parameters within 2% of a bound reported; a bound equal to a simulator cap (eps0 0.5) is not widened",
}
FS = {f"fs_{i}": (0.03, 30.0, "log") for i in range(5)}
ARMS = {"control": BASE_SPACE, "aug": SC.XS, "film": FS, "film_yoked": FS, "generic": SC.GW,
        "bias": {k: (-6.0, 6.0, "lin") for k in ("b_a", "b_e", "b_s")}}
SPEC_ARMS = {
    "id": "single_stageC2_film_vs_input",
    "registered": "after C2-0 (bases frozen), before any C2 arm run",
    "question": ("With a learner that can generalise across layouts (egocentric view), do the appraisal signals help "
                 "more as additive network input or as multiplicative modulation of the network's computation (FiLM, "
                 "neuromodulation-like gain and bias of hidden units, learned within the life)? Does the knob-level "
                 "global layer still help?"),
    "arms": {"control": "no signals: re-tune the 5 base parameters from the base",
             "aug": "signals as network input, per-input scales xs (0.03, 30) log, start ~1; 5 params",
             "film": ("signals modulate the hidden layer: pre' = pre * (1 + tanh(xf G)) + xf Fb, xf = scaled signals, "
                      "G and Fb start at 0 and are learned within the life by the TD gradient; CMA tunes the input "
                      "scales fs (0.03, 30) log, start ~1; 5 params"),
             "film_yoked": ("as film, but the FiLM is driven by another agent's signals (partner in the same "
                            "evaluation block: +24 mod 48 in training, +256 mod 512 at test); capacity control"),
             "generic": "knob-level global layer (lr, exploration, safety weight): 15 weights (-6, 6), start 0",
             "bias": "3 constant knob offsets (-6, 6), start 0"},
    "budget": ("each arm 8 runs on base r % 4, CMA-ES 20 x 12 x 48, sigma 0.15; CMA seeds 18000 + 100 * arm index + r; "
               "generation seed 33000000 + 1000 r + g shared by all arms of run r; run in lockstep groups of 2 runs"),
    "test": ("new independent set: 128 schedules (seeds 11000-11127) x 4 agents, layout bank 1500000-1500999, "
             "simulator seeds 84000 + run; all arms of a run in one call (common schedules, layouts and world noise)"),
    "predictions": {"P1": "film > aug", "P2": "film > film_yoked", "P3": "film > control", "P4": "aug > control",
                    "P5": "generic > control"},
    "inference": ("one-sided Wilcoxon over schedules of the run-averaged difference, Holm across P1-P5, plus "
                  "hierarchical bootstrap 95% CI (runs, then schedules); a claim needs both"),
    "secondary": {"two_sided": "aug - film, generic - aug, generic - bias, film - bias, aug - film_yoked",
                  "decomposition": "survival, wellbeing while alive, per-regime score, deaths by regime",
                  "speed": ("B6 switch protocol, death disabled, layouts 1600000-1600999, simulator seeds 94000 + run; "
                            "arms - control per switch and window, CI only"),
                  "bounds": "parameters within 2% of a bound reported"},
    "interpretation": ("P1 passes: modulating computation beats adding the same signals as input (the modulator-vs-state "
                       "point in a generalising learner). P1 fails with aug >= film: additive input is enough; P2 tests "
                       "whether film needs the agent's own signals (information) or only extra capacity"),
}
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
BASE_OUT, ARM_OUT, CACHE = LOGS / "stageC2_base", LOGS / "stageC2_opt", LOGS / "stageC2_cache"


def score_all(items, g, seed_base, tag):
    """As stage_c.score_all with learner dqnv and per-agent yoke partners (yoked arms: +24 mod 48 in the block)."""
    sched, lays, prms, yok, sizes = [], [], [], [], []
    off = 0
    for run, X in items:
        rng = np.random.default_rng(seed_base + 1000 * run["r"] + g)
        sch = [R.schedule(rng) for _ in range(NAG)]
        lay = rng.integers(len(B3.bank()), size=NAG)
        for x in X:
            sched += sch
            lays.append(lay)
            prms.append(run["make"](x))
            ii = np.arange(NAG)
            yok.append(off + ((ii + 24) % NAG if run.get("yoked") else ii))
            off += NAG
        sizes.append(len(X))
    d0 = {k: float(v[0]) for k, v in W.params(1).items()}
    keys = set(d0) | set().union(*prms)
    P = {k: np.repeat([c.get(k, d0.get(k, 0.0)) for c in prms], NAG) for k in keys}
    P["yoke"] = np.concatenate(yok)
    t = time.time()
    o = W.simulate("dqnv", P, A.regime_matrix(sched, R.T_LIFE), B3.bank(), np.concatenate(lays), seed=seed_base + g,
                   dqn=DQN)
    log(f"{tag} gen {g}: {len(sched)} agents in {time.time() - t:.0f}s")
    sc = o["score"].reshape(-1, NAG).mean(1)
    out, i = [], 0
    for s in sizes:
        out.append(sc[i:i + s])
        i += s
    return out


def base():
    log("registered", register(SPEC_BASE)[:12])
    BASE_OUT.mkdir(exist_ok=True)
    runs = [dict(out_json=BASE_OUT / f"dqnv_r{r}.json", x0=np.full(len(BASE_SPACE), 0.5), seed=17000 + r, r=r,
                 make=lambda u: {"gamma": 0.9, **decode_space(BASE_SPACE, u)}, meta=dict(arm="dqnv_base", run=r))
            for r in range(4)]
    res = cma_lockstep(runs, 30, lambda items, g: score_all(items, g, 32_000_000, "C2-0"), BASE_OUT / "base.ckpt",
                       POP, SIG)
    for r, j in enumerate(res):
        log("base", r, {k: round(j["params"][k], 5) for k in BASE_SPACE}, "on bound:",
            [k for k, ui in zip(BASE_SPACE, j["u"]) if ui < 0.02 or ui > 0.98], "last", round(j["history"][-1], 4))


def load_base(r):
    return json.loads((BASE_OUT / f"dqnv_r{r % 4}.json").read_text())["params"]


def arms():
    log("registered", register(SPEC_ARMS)[:12])
    ARM_OUT.mkdir(exist_ok=True)
    runs = []
    for arm, space in ARMS.items():
        for r in range(8):
            b = load_base(r)
            if arm == "control":
                x0, make = encode(space, b), (lambda sp: (lambda u: {"gamma": 0.9, **decode_space(sp, u)}))(space)
            else:
                x0 = np.full(len(space), 0.5)
                make = (lambda sp, bb: (lambda u: {**bb, **decode_space(sp, u)}))(space, b)
            runs.append(dict(out_json=ARM_OUT / f"{arm}_r{r}.json", x0=x0, seed=18000 + 100 * ARM_IDX[arm] + r, r=r,
                             make=make, yoked=arm == "film_yoked", meta=dict(arm=arm, run=r, learner="dqnv")))
    for lo in (0, 2, 4, 6):
        sub = [x for x in runs if lo <= x["r"] < lo + 2]
        cma_lockstep(sub, 20, lambda items, g: score_all(items, g, 33_000_000, f"C2 runs {lo}-{lo + 1}"),
                     ARM_OUT / f"arms_{lo}.ckpt", POP, SIG)
    log("C2 arm optimisation done")


def arm_load(arm, r):
    return json.loads((ARM_OUT / f"{arm}_r{r}.json").read_text())


def run_all_arms(name, r, M, lay, bank, seed, **kw):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}_r{r}.pkl"
    names = list(ARMS)
    if not f.exists():
        prms = [arm_load(a, r)["params"] for a in names]
        d0 = {k: float(v[0]) for k, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        nb = M.shape[1]
        P = {k: np.repeat([p.get(k, d0.get(k, 0.0)) for p in prms], nb) for k in keys}
        ii = np.arange(nb)
        P["yoke"] = np.concatenate([i * nb + ((ii + nb // 2) % nb if a == "film_yoked" else ii)
                                    for i, a in enumerate(names)])
        t = time.time()
        o = W.simulate("dqnv", P, np.tile(M, (1, len(names))), bank, np.tile(lay, len(names)), seed=seed, dqn=DQN,
                       record=("wellbeing",), **kw)
        res = {a: dict(score=o["score"][i * nb:(i + 1) * nb], survival=o["survival"][i * nb:(i + 1) * nb],
                       wellbeing=o["wellbeing"][:, i * nb:(i + 1) * nb]) for i, a in enumerate(names)}
        _atomic_write(f, pickle.dumps(res))
        log(f"{name} run {r}: {len(names) * nb} agents, {time.time() - t:.0f}s")
    return pickle.loads(f.read_bytes())


NS, NA_ = 128, 4


def test():
    sch = [R.schedule(np.random.default_rng(11000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    rows = []
    for r in range(8):
        for a, o in run_all_arms("test", r, M, lay, SC._bank(1_500_000), 84_000 + r).items():
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
    D.to_csv(LOGS / "stageC2_test_raw.csv", index=False)
    log("test raw written", len(D))


def speed():
    import stage_b6 as B6
    rows = []
    for a_, b_ in B6.SW:
        B = B6.NL * NA_
        M = np.zeros((B6.PRE + B6.POST, B), int)
        M[:B6.PRE], M[B6.PRE:] = R.NAMES.index(a_), R.NAMES.index(b_)
        lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(B6.NL)])
        for r in range(8):
            for arm, o in run_all_arms(f"speed_{a_}-{b_}", r, M, lay, SC._bank(1_600_000), 94_000 + r,
                                       juvenile=B6.PRE + B6.POST + 1).items():
                wb = o["wellbeing"]
                d = dict(arm=arm, run=r, switch=f"{a_}->{b_}", layout=np.repeat(np.arange(B6.NL), NA_),
                         pre=wb[B6.PRE - 200:B6.PRE].mean(0))
                for lo, hi in B6.WIN:
                    d[f"w{lo}_{hi}"] = wb[B6.PRE + lo:B6.PRE + hi].mean(0)
                rows.append(pd.DataFrame(d))
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC2_speed_raw.csv", index=False)
    T = []
    for s in D.switch.unique():
        X = D[D.switch == s]
        for col in ("pre", "w0_100", "w100_300", "w0_300", "w300_1000"):
            pv = X.groupby(["arm", "run", "layout"])[col].mean()
            for a in ARMS:
                if a != "control":
                    d = (pv[a] - pv["control"]).unstack()
                    lo, hi = B3.hier_ci(d.values, n=2000)
                    T.append(dict(switch=s, window=col, comparison=f"{a} - control", diff=d.values.mean(),
                                  ci_lo=lo, ci_hi=hi))
    pd.DataFrame(T).to_csv(LOGS / "stageC2_speed_tests.csv", index=False)
    log("speed tests written")


def report():
    import stage_b5 as B5
    from bio_rl.analysis.stats import holm
    D = pd.read_csv(LOGS / "stageC2_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    prim = [("P1", "film", "aug"), ("P2", "film", "film_yoked"), ("P3", "film", "control"), ("P4", "aug", "control"),
            ("P5", "generic", "control")]
    T = pd.DataFrame([dict(pred=k, **B5.comp(piv, a, b)) for k, a, b in prim])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL")
    T["family"] = "primary"
    sec = pd.DataFrame([B5.comp(piv, a, b, "two-sided") for a, b in
                        (("aug", "film"), ("generic", "aug"), ("generic", "bias"), ("film", "bias"),
                         ("aug", "film_yoked"), ("bias", "control"), ("film_yoked", "control"))])
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(LOGS / "stageC2_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"score_{r}" for r in R.NAMES] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC2_summary.csv")
    rows = []
    for a, sp in ARMS.items():
        for r in range(8):
            j = arm_load(a, r)
            for (k, _), ui in zip(sp.items(), j["u"]):
                rows.append(dict(arm=a, run=r, param=k, value=j["params"][k], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageC2_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(S.round(4).T.to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (base, arms, test, report, speed):
        log("=== step", step.__name__)
        step()
    log("=== C2 finished")


if __name__ == "__main__":
    {"base": base, "arms": arms, "test": test, "speed": speed, "report": report, "all": all_}[sys.argv[1]]()
