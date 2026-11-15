"""Stage C3c (registered after C3b, before any C3c run): capacity-matched feed-forward control for the GRU.
In C3b the GRU's advantage was not explained by its last-reward/last-action inputs, but the GRU had 3.5x the weights
of the feed-forward net (17,130 vs 4,874). ff_big is the C3b control_ra learner (feed-forward view net with last felt
reward and last action) with hidden width 113, so that its weight count (17,186) matches the GRU.
Arm on the 8 frozen C3 bases, C3 add-on budget and generation seeds (common random numbers with C3/C3b).
Test: a new set (primary) with the C3/C3b solutions of gru, control_ra, aug_ra and aug re-evaluated; ff_big also on the
C3b set (secondary, compared with the logged C3b results).
Run: python stage_c3c.py [arms|test|report|all]"""
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
import stage_c3 as C3
import stage_c3b as C3B
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

LOGS, NB, NS, NA_ = C3.LOGS, C3.NB, C3.NS, C3.NA_
log = SC.log
H_BIG = 113
TEST = dict(sched=45000, bank=2_500_000, sim=150_000)
OLD = dict(sched=44000, bank=2_400_000, sim=140_000)          # C3b set (secondary)
if os.environ.get("C3_SMOKE", "").strip():      # code check (dev seeds; see stage_c3.py)
    TEST = dict(sched=97000, bank=1_960_000, sim=133_000)
    OLD = dict(sched=97200, bank=1_960_000, sim=134_000)
    register = lambda spec: "smoke-no-registration"
ARM = ("dqnv", C3.BASE_SPACE, {"ra": True, "H": H_BIG})
REUSED = ["gru", "control_ra", "aug_ra", "aug"]
OUT, CACHE = LOGS / "stageC3c_opt", LOGS / "stageC3c_cache"
SPEC = {
    "id": "single_stageC3c_capacity_control",
    "registered": "after the C3b results (2026-10-24) and a dev timing benchmark, before any C3c run",
    "question": ("Is the recurrent network's advantage memory, or capacity? A feed-forward net with the same inputs as "
                 "the GRU (view, needs, last felt reward, last action; no appraisal signals) and the same weight count."),
    "arm": {"ff_big": ("dqnv with last reward and action (as C3b control_ra), hidden width 113 (17,186 weights vs "
                       "17,130 for the GRU; all input weights counted, unused FiLM matrices excluded); re-tune the 5 base "
                       "parameters from the C3 base; learns from 8 single replayed transitions (the GRU from 2 x 4-step "
                       "sequences: replay scheme not matched)")},
    "reused": "C3 solutions of gru and aug, C3b solutions of control_ra and aug_ra (same bases, budget, generation seeds)",
    "bases": "the 8 frozen C3 bases (single_stageC3_base, 1317a0fe)",
    "budget": ("CMA-ES 20 x 12 x 48, sigma 0.15; CMA seed 28200 + k; generation seeds 61000000 + 1000 k + g and simulator "
               "seed 61000000 + 10 g + c, identical to C3 and C3b"),
    "test": ("primary: new set, never used: 128 schedules (seeds 45000-45127) x 4 agents per base, layout bank "
             "2500000-2500999, simulator seed 150000 + c (c = 0 bases 0-3, 1 bases 4-7), identical for ff_big, gru, "
             "control_ra, aug_ra, aug. Secondary: ff_big on the C3b test set (schedules 44000-44127, layouts 2400000+, "
             "simulator 140000 + c), compared with the logged C3b results of the other arms"),
    "predictions": {"Q1": "gru > ff_big (memory beyond capacity; one-sided)"},
    "secondary": {"S1": "ff_big vs control_ra (capacity alone; two-sided)", "S2": "ff_big vs aug_ra (two-sided)",
                  "S3": "gru vs aug_ra (two-sided)", "C3b_set": "gru - ff_big and ff_big - control_ra on the C3b set",
                  "deaths": "deaths by regime (descriptive)", "bounds": "ff_big parameters within 2% of a bound"},
    "inference": ("as C3: unit = base (n = 8); one-sided t-test on base means for Q1, two-sided for secondary (Holm "
                  "within family); 95% three-level bootstrap CI; a claim needs p_Holm < 0.05 AND the CI excluding 0; "
                  "exact Wilcoxon on base means and schedule-level Wilcoxon reported as checks"),
    "interpretation": ("Q1 pass: the GRU advantage needs recurrence beyond matched capacity (replay scheme still differs). "
                       "Q1 fail with ff_big >= gru: capacity explains it. No detectable difference is not equivalence."),
}


def arm_runs():
    learner, space, cfg = ARM
    make = lambda u: {"gamma": 0.9, **decode_space(space, u)}
    return [dict(out_json=OUT / f"ff_big_r{k}.json", x0=encode(space, C3.load_base(k)), seed=28200 + k, r=k, make=make,
                 meta=dict(arm="ff_big", run=k, learner=learner, cfg=cfg)) for k in range(NB)]


def arms():
    log("registered", register(SPEC)[:12])
    OUT.mkdir(parents=True, exist_ok=True)
    learner, _, cfg = ARM
    cma_lockstep(arm_runs(), C3.AGENS, C3.score_fn(learner, cfg, 61_000_000, "C3c ff_big"), OUT / "ff_big.ckpt",
                 C3.POP, C3.SIG)
    log("C3c arm ff_big done")


def spec_of(arm):
    if arm == "ff_big":
        return ARM[0], ARM[2], OUT
    return C3B.spec_of(arm)


def evaluate(name, arm, ts, M, lay):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_{arm}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    learner, cfg, d = spec_of(arm)
    bank = SC._bank(ts["bank"])
    out = {}
    for c, ks in enumerate(C3.calls(NB)):
        prms = [json.loads((d / f"{arm}_r{k}.json").read_text())["params"] for k in ks]
        nb = M.shape[1]
        d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nb) for k_ in keys}
        t = time.time()
        o = W.simulate(learner, P, np.tile(M, (1, len(ks))), bank, np.tile(lay, len(ks)), seed=ts["sim"] + c,
                       dqn={**C3.DQN, **cfg}, record=("wellbeing",))
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


def test_set(ts):
    sch = [R.schedule(np.random.default_rng(ts["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:C3.TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    return M, lay


def test():
    log("registered", register(SPEC)[:12])
    M, lay = test_set(TEST)
    rows = [frame(a, k, o, M) for a in ["ff_big"] + REUSED for k, o in evaluate("test", a, TEST, M, lay).items()]
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC3c_test_raw.csv", index=False)
    M2, lay2 = test_set(OLD)
    D2 = pd.concat([frame("ff_big", k, o, M2) for k, o in evaluate("c3bset", "ff_big", OLD, M2, lay2).items()])
    D2.to_csv(LOGS / "stageC3c_c3bset_raw.csv", index=False)
    log("test raw written", len(D), len(D2))


PRIMARY = [("Q1", "gru", "ff_big", "greater")]
SECONDARY = [("S1", "ff_big", "control_ra", "two-sided"), ("S2", "ff_big", "aug_ra", "two-sided"),
             ("S3", "gru", "aug_ra", "two-sided")]
C3BSET = [("C1", "gru", "ff_big", "two-sided"), ("C2", "ff_big", "control_ra", "two-sided")]


def report():
    D = pd.read_csv(LOGS / "stageC3c_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, PRIMARY, "primary"), B10.family(piv, SECONDARY, "secondary")], ignore_index=True)
    old = pd.read_csv(LOGS / "stageC3b_test_raw.csv") if not os.environ.get("C3_SMOKE", "").strip() else D.iloc[0:0]
    D2 = pd.concat([old, pd.read_csv(LOGS / "stageC3c_c3bset_raw.csv")])
    if set(D2.arm) >= {"gru", "control_ra", "ff_big"}:
        piv2 = D2.groupby(["arm", "run", "schedule"]).score.mean()
        T = pd.concat([T, B10.family(piv2, C3BSET, "secondary_c3bset")], ignore_index=True)
    T.to_csv(LOGS / "stageC3c_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC3c_summary.csv")
    bm = D.groupby(["arm", "run"]).score.mean().unstack(0)
    bm.to_csv(LOGS / "stageC3c_base_means.csv")
    rows = []
    for k in range(NB):
        j = json.loads((OUT / f"ff_big_r{k}.json").read_text())
        for (kk, _), ui in zip(C3.BASE_SPACE.items(), j["u"]):
            rows.append(dict(arm="ff_big", base=k, param=kk, value=j["params"][kk], u=ui,
                             on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageC3c_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S.round(4).T.to_string())
    print(bm.round(3).to_string())
    print(Pm.groupby("param").agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (arms, test, report):
        log("=== step", step.__name__)
        step()
    log("=== C3c finished")


if __name__ == "__main__":
    {"arms": arms, "test": test, "report": report, "all": all_}[sys.argv[1]]()
