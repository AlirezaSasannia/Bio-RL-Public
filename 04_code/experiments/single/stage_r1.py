"""Stage R1 (optional robustness, original world): do the C3 findings depend on the network size or the view?
Two variants of the C3 view learner on the 8 frozen C3 bases: H64 (hidden width 64, also the GRU's units) and view3
(view radius 3 = 7 x 7 in the 8 x 8 world). Arms per variant: control, aug, film, gru and the constant-offset control
bias; one add-on run per arm, variant and base, C3 add-on budget; base = unit.
  register   register the spec (commit the yaml before running)
  arms [a@v,..]  arms (subset like 'gru@H64,gru@view3' for parallel processes; results do not depend on the split)
  test       all arm-variants on a new test set
  report     tests (stage_b10.family), summary, bounds
Run: python stage_r1.py <step>   Smoke: R1_SMOKE=<scratch dir> python stage_r1.py all"""
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = os.environ.get("R1_SMOKE", "").strip()
if SMOKE:                       # reuse the C3 smoke sizes (stage_c3 reads C3_SMOKE at import); copy the frozen bases
    os.environ["C3_SMOKE"] = SMOKE
    (Path(SMOKE) / "stageC3_base").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "results" / "logs" / "single" / "stageC3_base").glob("dqnv_r*.json"):
        shutil.copy(f, Path(SMOKE) / "stageC3_base" / f.name)

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b10 as B10
import stage_c as SC
import stage_c3 as C3
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.torchsim import world as W

LOGS, NB, NS, NA_ = C3.LOGS, C3.NB, C3.NS, C3.NA_
log = SC.log
TEST = dict(sched=49000, bank=2_600_000, sim=270_000)
SEED = dict(cma=38000, gen=74_000_000)
if SMOKE:
    TEST = dict(sched=97400, bank=1_960_000, sim=135_000)
    register = lambda spec: "smoke-no-registration"
VARIANTS = {"H64": {"H": 64}, "view3": {"view": 3}}
ARMS = {a: C3.ARMS[a] for a in ("control", "aug", "film", "gru", "bias")}
ARM_IDX = {a: i for i, a in enumerate(ARMS)}
VAR_IDX = {v: i for i, v in enumerate(VARIANTS)}
OUT, CACHE = LOGS / "stageR1_opt", LOGS / "stageR1_cache"
SPEC = {
    "id": "single_stageR1_size_view_robustness",
    "registered": "optional robustness stage, before any R1 run; smoke tests only",
    "question": ("Do the C3 results (signals as input or FiLM help the view learner; the GRU) depend on the hidden "
                 "width or on the view size?"),
    "variants": {"H64": "hidden width 64 for every arm (GRU: 64 units)",
                 "view3": "view radius 3 (7 x 7) in the 8 x 8 world, all arms"},
    "arms": {"control": "re-tune the 5 base parameters from the C3 base", "aug": "signals as input (xs), as C3",
             "film": "FiLM (fs), as C3", "gru": "GRU, as C3 (re-tune the 5 base parameters)",
             "bias": "3 constant knob offsets (-6, 6), start 0 (constant-offset control)"},
    "bases": "the 8 frozen C3 bases (single_stageC3_base, 1317a0fe), tuned with H = 32 and view radius 2",
    "budget": ("each arm, variant and base: CMA-ES 20 x 12 x 48, sigma 0.15; CMA seed 38000 + 1000 * variant index + "
               "100 * arm index + k (orders as listed); generation seeds 74000000 + 1000 k + g and simulator seed "
               "74000000 + 10 g + c shared by all arms and variants (common random numbers); training bank "
               "400000-401999 (as C3)"),
    "test": ("new independent set, never used: 128 schedules (seeds 49000-49127) x 4 agents per base, layout bank "
             "2600000-2600999, simulator seed 270000 + c (c = 0 bases 0-3, 1 bases 4-7), identical for all arms"),
    "predictions": {"R1_H64": "aug > control (H64)", "R2_H64": "film > control (H64)",
                    "R1_view3": "aug > control (view3)", "R2_view3": "film > control (view3)",
                    "note": "one-sided, Holm over the 4"},
    "secondary": {"per variant": "gru vs control, aug vs gru, aug vs bias, film vs bias (two-sided, Holm within the "
                                 "8 tests)", "bounds": "parameters within 2% of a bound reported"},
    "inference": ("as C3: unit = base (n = 8), t-test on base means, Holm, 95% three-level bootstrap CI; a claim needs "
                  "p_Holm < 0.05 AND the CI excluding 0; exact Wilcoxon on base means and schedule-level checks"),
    "interpretation": ("passes in both variants: the C3 input/FiLM gains are not specific to H = 32 and the 5 x 5 view. "
                       "A failure limits the claim to the original configuration. No detectable difference is not "
                       "equivalence. The bases were tuned for H = 32 / view 2 (registered caveat)."),
}


def cfg_of(arm, var):
    return {**ARMS[arm][2], **VARIANTS[var]}


def name(arm, var):
    return f"{arm}@{var}"


def runs_of(arm, var):
    runs = C3.arm_runs(arm)
    for r in runs:
        r["out_json"] = OUT / f"{arm}_{var}_r{r['r']}.json"
        r["seed"] = SEED["cma"] + 1000 * VAR_IDX[var] + 100 * ARM_IDX[arm] + r["r"]
        r["meta"] = dict(r["meta"], variant=var, cfg=cfg_of(arm, var))
    return runs


def reg():
    log("registered", register(SPEC)[:12])


def arms(names=None):
    reg()
    OUT.mkdir(parents=True, exist_ok=True)
    for var in VARIANTS:
        for arm in ARMS:
            if names and name(arm, var) not in names.split(","):
                continue
            cma_lockstep(runs_of(arm, var), C3.AGENS, C3.score_fn(ARMS[arm][0], cfg_of(arm, var), SEED["gen"],
                                                                  f"R1 {name(arm, var)}"),
                         OUT / f"{arm}_{var}.ckpt", C3.POP, C3.SIG)
            log(f"R1 {name(arm, var)} done")


def evaluate(arm, var, M, lay):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"test_{arm}_{var}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    bank = SC._bank(TEST["bank"])
    out = {}
    nb = M.shape[1]
    for c, ks in enumerate(C3.calls(NB)):
        prms = [json.loads((OUT / f"{arm}_{var}_r{k}.json").read_text())["params"] for k in ks]
        d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nb) for k_ in keys}
        t = time.time()
        o = W.simulate(ARMS[arm][0], P, np.tile(M, (1, len(ks))), bank, np.tile(lay, len(ks)), seed=TEST["sim"] + c,
                       dqn={**C3.DQN, **cfg_of(arm, var)}, record=("wellbeing",))
        for i, k in enumerate(ks):
            out[k] = {kk: (v[i * nb:(i + 1) * nb] if v.ndim == 1 else v[:, i * nb:(i + 1) * nb]) for kk, v in o.items()}
        log(f"test {name(arm, var)} bases {ks}: {len(ks) * nb} agents, {time.time() - t:.0f}s")
    _atomic_write(f, pickle.dumps(out))
    return out


def test():
    import stage_c3c as C3C
    reg()
    sch = [R.schedule(np.random.default_rng(TEST["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:C3.TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    D = pd.concat([C3C.frame(name(a, v), k, o, M) for v in VARIANTS for a in ARMS
                   for k, o in evaluate(a, v, M, lay).items()])
    D.to_csv(LOGS / "stageR1_test_raw.csv", index=False)
    log("test raw written", len(D))


PRIMARY = [(f"R{i}_{v}", name(a, v), name("control", v), "greater") for v in VARIANTS
           for i, a in ((1, "aug"), (2, "film"))]
SECONDARY = [(f"S_{a}-{b}_{v}", name(a, v), name(b, v), "two-sided") for v in VARIANTS
             for a, b in (("gru", "control"), ("aug", "gru"), ("aug", "bias"), ("film", "bias"))]


def report():
    D = pd.read_csv(LOGS / "stageR1_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, PRIMARY, "primary"), B10.family(piv, SECONDARY, "secondary")], ignore_index=True)
    T.to_csv(LOGS / "stageR1_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageR1_summary.csv")
    rows = []
    for v in VARIANTS:
        for a, (_, sp, _, _) in ARMS.items():
            for k in range(NB):
                j = json.loads((OUT / f"{a}_{v}_r{k}.json").read_text())
                for (kk, _), ui in zip(sp.items(), j["u"]):
                    rows.append(dict(arm=name(a, v), base=k, param=kk, value=j["params"][kk], u=ui,
                                     on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageR1_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S.round(4).T.to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (arms, test, report):
        log("=== step", step.__name__)
        step()
    log("=== R1 finished")


if __name__ == "__main__":
    if sys.argv[1] == "arms" and len(sys.argv) > 2:
        arms(sys.argv[2])
    else:
        {"register": reg, "arms": arms, "test": test, "report": report, "all": all_}[sys.argv[1]]()
