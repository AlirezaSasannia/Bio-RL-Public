"""Path 2: hormone-keyed memory. Can appraisal ('hormonal') state make an agent re-adapt faster when a regime it has
met before returns (savings), something knob tuning cannot do? Learner dqnv (C2-0 bases) with a 3000-step replay
buffer; replay samples are drawn preferentially from moments whose hormonal state resembles the present one
(state-dependent memory). World and learner: bio_rl.torchsim.world (mem_mode). Resumable (lockstep CMA with
checkpoints; evaluations cached). Run: python stage_d.py [arms|savings|test|report|all]"""
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
import stage_c2 as C2
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

LOGS = B3.LOGS
DQN = {**SC.DQN, "M": 3000, "mem_mode": True, "K": 64}
POP, NAG, SIG = 12, 48, 0.15
log = SC.log
KS = {f"ks_{i}": (0.01, 30.0, "log") for i in range(5)}
KV = {f"kv_{i}": (0.01, 30.0, "log") for i in range(4)}
ARMS = {"short": C2.BASE_SPACE, "uniform": C2.BASE_SPACE, "keyed": KS, "keyed_yoked": KS, "statekeyed": KV}
MEMW = {"short": 1000, "uniform": 3000, "keyed": 3000, "keyed_yoked": 3000, "statekeyed": 3000}
SPEC = {
    "id": "single_stageD_hormone_keyed_memory",
    "registered": "after C2 results, before any D run; dev smoke tests only (seeds 99980+)",
    "question": ("Does replay keyed on the agent's appraisal ('hormonal') state let a network learner re-adapt faster "
                 "when a previously met regime returns (savings), compared with the same memory sampled uniformly, "
                 "keyed on another agent's signals, or keyed on the agent's location and needs?"),
    "learner": ("dqnv with the frozen C2-0 bases; replay buffer 3000 steps; per step the newest transition plus 7 drawn "
                "from 64 uniform candidates within the agent's window by Gumbel-top-k on -sum((scale * (key - key_now))^2); "
                "hormonal key = 5 appraisal signals (clip 0-3, /3), state key = (row/7, col/7, energy, safety)"),
    "arms": {"short": "window 1000, uniform sampling (as the C2 learner); re-tune the 5 base parameters",
             "uniform": "window 3000, uniform sampling; re-tune the 5 base parameters (memory-capacity control)",
             "keyed": "window 3000, hormonal key scales ks (0.01, 30) log, start u = 0.2 (~0.05, near uniform); 5 params",
             "keyed_yoked": ("as keyed, but the key (stored and current) uses another agent's signals (partner +24 mod 48 "
                             "in training, +256 mod 512 at test); information control"),
             "statekeyed": "window 3000, state key scales kv (0.01, 30) log, start u = 0.2; 4 params (non-hormonal context)"},
    "budget": ("each arm 8 runs on C2-0 base r % 4, CMA-ES 20 x 12 x 48, sigma 0.15; CMA seeds 19000 + 100 * arm "
               "index + r; generation seed 34000000 + 1000 r + g shared by all arms of run r; lockstep groups of 2 runs"),
    "savings_protocol": ("death disabled; 4200 steps: safe [0,1200), X [1200,1800), safe [1800,2400), X [2400,3000), "
                         "safe [3000,3600), X [3600,4200), X in danger, uncontrollable, scarce, volatile; 128 layouts "
                         "(bank 1800000-1800999) x 4 agents, simulator seeds 95000 + run (+ 1000 * X index); "
                         "e_k = mean wellbeing in the first 200 steps of exposure k; re-exposure R = mean(e2, e3), "
                         "averaged over the 4 X; savings S = R - e1"),
    "test": ("new independent set: 128 schedules (seeds 12000-12127) x 4 agents, layout bank 1700000-1700999, simulator "
             "seeds 85000 + run"),
    "predictions": {"Q1": "keyed > uniform on R", "Q2": "keyed > keyed_yoked on R", "Q3": "keyed > statekeyed on R",
                    "Q4": "keyed > uniform on test score", "Q5": "keyed > short on test score"},
    "inference": ("one-sided Wilcoxon over layouts (R) or schedules (score) of the run-averaged difference, Holm across "
                  "Q1-Q5, plus hierarchical bootstrap 95% CI (runs, then layouts/schedules); a claim needs both"),
    "secondary": {"savings": "S per arm and differences; per X; e1, e2, e3",
                  "decomposition": "survival, wellbeing while alive, deaths by regime on the test set",
                  "two_sided": "statekeyed - keyed, uniform - short, statekeyed - uniform",
                  "bounds": "parameters within 2% of a bound reported"},
    "interpretation": ("Q1-Q3 pass: hormonal state is a useful retrieval key, beyond memory size, beyond information-free "
                       "keys and beyond location/need similarity. statekeyed >= keyed: a non-hormonal context key is as "
                       "good, and the benefit is retrieval by context in general, not hormonal state"),
}
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
OUT, CACHE = LOGS / "stageD_opt", LOGS / "stageD_cache"


def score_all(items, g, seed_base, tag):
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


def arms():
    log("registered", register(SPEC)[:12])
    OUT.mkdir(exist_ok=True)
    runs = []
    for arm, space in ARMS.items():
        for r in range(8):
            b = {**C2.load_base(r), "mem": float(MEMW[arm])}
            if arm in ("short", "uniform"):
                x0 = encode(space, b)
                make = (lambda sp, mw: (lambda u: {"gamma": 0.9, "mem": mw, **decode_space(sp, u)}))(space, float(MEMW[arm]))
            else:
                x0 = np.full(len(space), 0.2)
                make = (lambda sp, bb: (lambda u: {**bb, **decode_space(sp, u)}))(space, b)
            runs.append(dict(out_json=OUT / f"{arm}_r{r}.json", x0=x0, seed=19000 + 100 * ARM_IDX[arm] + r, r=r,
                             make=make, yoked=arm == "keyed_yoked", meta=dict(arm=arm, run=r, learner="dqnv_mem")))
    for lo in (0, 2, 4, 6):
        sub = [x for x in runs if lo <= x["r"] < lo + 2]
        cma_lockstep(sub, 20, lambda items, g: score_all(items, g, 34_000_000, f"D runs {lo}-{lo + 1}"),
                     OUT / f"arms_{lo}.ckpt", POP, SIG)
    log("D optimisation done")


def arm_load(arm, r):
    return json.loads((OUT / f"{arm}_r{r}.json").read_text())


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
        P["yoke"] = np.concatenate([i * nb + ((ii + nb // 2) % nb if a == "keyed_yoked" else ii)
                                    for i, a in enumerate(names)])
        t = time.time()
        o = W.simulate("dqnv", P, np.tile(M, (1, len(names))), bank, np.tile(lay, len(names)), seed=seed, dqn=DQN,
                       record=("wellbeing",), **kw)
        res = {a: dict(score=o["score"][i * nb:(i + 1) * nb], survival=o["survival"][i * nb:(i + 1) * nb],
                       wellbeing=o["wellbeing"][:, i * nb:(i + 1) * nb]) for i, a in enumerate(names)}
        _atomic_write(f, pickle.dumps(res))
        log(f"{name} run {r}: {len(names) * nb} agents, {time.time() - t:.0f}s")
    return pickle.loads(f.read_bytes())


NS, NA_, NL = 128, 4, 128
XS_REG = ["danger", "uncontrollable", "scarce", "volatile"]
SEG = [(0, 1200, "safe"), (1200, 1800, "X"), (1800, 2400, "safe"), (2400, 3000, "X"), (3000, 3600, "safe"),
       (3600, 4200, "X")]
EXPO = [1200, 2400, 3600]


def savings():
    rows = []
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NL)])
    for xi, X in enumerate(XS_REG):
        M = np.zeros((4200, NL * NA_), int)
        for a_, b_, nm in SEG:
            M[a_:b_] = R.NAMES.index(X if nm == "X" else nm)
        for r in range(8):
            res = run_all_arms(f"savings_{X}", r, M, lay, SC._bank(1_800_000), 95_000 + r + 1000 * xi, juvenile=4201)
            for arm, o in res.items():
                wb = o["wellbeing"]
                d = dict(arm=arm, run=r, regime=X, layout=np.repeat(np.arange(NL), NA_))
                for k, e in enumerate(EXPO):
                    d[f"e{k + 1}"] = wb[e:e + 200].mean(0)
                    d[f"full{k + 1}"] = wb[e:e + 600].mean(0)
                rows.append(pd.DataFrame(d))
    D = pd.concat(rows)
    D["R"] = (D.e2 + D.e3) / 2
    D["S"] = D.R - D.e1
    D.to_csv(LOGS / "stageD_savings_raw.csv", index=False)
    log("savings raw written", len(D))


def test():
    sch = [R.schedule(np.random.default_rng(12000 + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    rows = []
    for r in range(8):
        for a, o in run_all_arms("test", r, M, lay, SC._bank(1_700_000), 85_000 + r).items():
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
    D.to_csv(LOGS / "stageD_test_raw.csv", index=False)
    log("test raw written", len(D))


def _comp(piv, a, b, alt="greater"):
    from scipy import stats
    d = (piv[a] - piv[b]).unstack()
    lo, hi = B3.hier_ci(d.values)
    p = float(stats.wilcoxon(d.mean(0).values, alternative=alt).pvalue)
    return dict(comparison=f"{a} - {b}", diff=d.values.mean(), ci_lo=lo, ci_hi=hi,
                runs_better=int((d.mean(1) > 0).sum()), p=p)


def report():
    from bio_rl.analysis.stats import holm
    Sv = pd.read_csv(LOGS / "stageD_savings_raw.csv")
    D = pd.read_csv(LOGS / "stageD_test_raw.csv")
    pR = Sv.groupby(["arm", "run", "layout"]).R.mean()
    pS = D.groupby(["arm", "run", "schedule"]).score.mean()
    prim = [("Q1", pR, "keyed", "uniform"), ("Q2", pR, "keyed", "keyed_yoked"), ("Q3", pR, "keyed", "statekeyed"),
            ("Q4", pS, "keyed", "uniform"), ("Q5", pS, "keyed", "short")]
    T = pd.DataFrame([dict(pred=k, measure="R" if pv is pR else "score", **_comp(pv, a, b)) for k, pv, a, b in prim])
    adj = holm(dict(enumerate(T.p)))
    T["p_holm"] = [adj[i] for i in range(len(T))]
    T["verdict"] = np.where((T.p_holm < 0.05) & (T.ci_lo > 0), "PASS", "FAIL")
    T["family"] = "primary"
    pSS = Sv.groupby(["arm", "run", "layout"]).S.mean()
    sec = [dict(measure="R", **_comp(pR, a, b, "two-sided")) for a, b in
           (("statekeyed", "keyed"), ("uniform", "short"), ("statekeyed", "uniform"), ("keyed_yoked", "uniform"))]
    sec += [dict(measure="S", **_comp(pSS, a, b, "two-sided")) for a, b in
            (("keyed", "uniform"), ("statekeyed", "uniform"), ("keyed", "keyed_yoked"), ("uniform", "short"))]
    sec += [dict(measure="score", **_comp(pS, a, b, "two-sided")) for a, b in
            (("statekeyed", "keyed"), ("uniform", "short"), ("statekeyed", "short"), ("keyed_yoked", "short"))]
    sec = pd.DataFrame(sec)
    sec["family"] = "secondary"
    T = pd.concat([T, sec], ignore_index=True)
    T.to_csv(LOGS / "stageD_tests.csv", index=False)
    S1 = Sv.groupby(["arm", "regime"])[["e1", "e2", "e3", "R", "S"]].mean()
    S1.to_csv(LOGS / "stageD_savings_summary.csv")
    cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
    S2 = D.groupby("arm")[cols].mean()
    S2.to_csv(LOGS / "stageD_summary.csv")
    rows = []
    for a, sp in ARMS.items():
        for r in range(8):
            j = arm_load(a, r)
            for (k, _), ui in zip(sp.items(), j["u"]):
                rows.append(dict(arm=a, run=r, param=k, value=j["params"][k], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageD_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(Sv.groupby("arm")[["e1", "e2", "e3", "R", "S"]].mean().round(4).to_string())
    print(S2.round(4).T.to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (arms, savings, test, report):
        log("=== step", step.__name__)
        step()
    log("=== D finished")


if __name__ == "__main__":
    {"arms": arms, "savings": savings, "test": test, "report": report, "all": all_}[sys.argv[1]]()
