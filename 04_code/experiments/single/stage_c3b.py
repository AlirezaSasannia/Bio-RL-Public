"""Stage C3b (registered after the C3 results, before any C3b run): is the GRU's advantage in C3 memory, or only its
two extra inputs (last felt reward, last action)?
Arms on the 8 frozen C3 bases, C3 add-on budget and generation seeds (common random numbers with the C3 arms):
  control_ra  feed-forward dqnv + last felt reward and last action as input (encoding as the gru input); re-tune the
              5 base parameters from the base (as gru)
  aug_ra      control_ra + the 5 appraisal signals as input; 5 base parameters + xs (as gru_aug)
C3 solutions of control, aug and gru are reused and re-tested on a new independent set.
Run: python stage_c3b.py [arms|test|report|all]"""
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
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W

LOGS, NB, NS, NA_ = C3.LOGS, C3.NB, C3.NS, C3.NA_
log = SC.log
TEST = dict(sched=44000, bank=2_400_000, sim=140_000)
if os.environ.get("C3_SMOKE", "").strip():      # code check (dev seeds; see stage_c3.py)
    TEST = dict(sched=97000, bank=1_960_000, sim=132_000)
    register = lambda spec: "smoke-no-registration"
NEW = {"control_ra": ("dqnv", C3.BASE_SPACE, {"ra": True}, "base"),
       "aug_ra": ("dqnv", {**C3.BASE_SPACE, **SC.XS}, {"ra": True}, "base+0.5")}
REUSED = ["control", "aug", "gru"]
ARM_IDX = {a: i for i, a in enumerate(NEW)}
OUT, CACHE = LOGS / "stageC3b_opt", LOGS / "stageC3b_cache"
SPEC = {
    "id": "single_stageC3b_gru_confound",
    "registered": "after the C3 results (2026-10-24), before any C3b run; dev smoke test only",
    "question": ("Is the C3 advantage of the GRU learner memory, or the two inputs it alone received (last felt reward, "
                 "last action)? Do the appraisal signals still add information when those inputs are given?"),
    "arms": {"control_ra": ("feed-forward dqnv + last felt reward (2) and last action (one-hot 5) as input, encoded as "
                            "in the gru input; re-tune the 5 base parameters from the C3 base (as gru); 5 params"),
             "aug_ra": ("control_ra + the 5 appraisal signals as input (scales xs (0.03, 30) log, start ~1); 5 base "
                        "parameters from the base + xs, as gru_aug; 10 params"),
             "reused": "C3 solutions of control, aug and gru (same bases, budget and generation seeds), re-tested"},
    "bases": "the 8 frozen C3 bases (single_stageC3_base, 1317a0fe)",
    "budget": ("as C3 arms: CMA-ES 20 x 12 x 48, sigma 0.15; CMA seed 28000 + 100 * arm index + k; generation seeds "
               "61000000 + 1000 k + g and simulator seed 61000000 + 10 g + c, identical to the C3 arms"),
    "test": ("new independent set, never used: 128 schedules (seeds 44000-44127) x 4 agents per base, layout bank "
             "2400000-2400999, simulator seed 140000 + c (c = 0 for bases 0-3, 1 for bases 4-7), identical for all "
             "five arms"),
    "predictions": {"P1": "gru > control_ra (memory beyond the two extra inputs)",
                    "P2": "aug_ra > control_ra (signals add information when last reward and action are given)"},
    "secondary": {"S1": "control_ra vs control (two-sided)", "S2": "aug_ra vs gru (two-sided)",
                  "S3": "gru vs aug (two-sided; re-test of C3 P4 on the new set)",
                  "deaths": "survival, wellbeing while alive, deaths by regime (descriptive)",
                  "bounds": "parameters within 2% of a bound reported for every base and new arm"},
    "inference": ("as C3/B10: unit = base (n = 8); one-sided t-test on base means, Holm over P1-P2 (and over S1-S3), "
                  "95% three-level bootstrap CI; a claim needs p_Holm < 0.05 AND the CI excluding 0. Checks: exact "
                  "Wilcoxon on base means, schedule-level Wilcoxon with hierarchical CI"),
    "interpretation": ("P1 pass: the GRU's advantage needs memory, not only the two inputs. P1 fail with control_ra >= "
                       "gru: the two inputs explain it. P2 pass: the signals still help a feed-forward learner that "
                       "sees its last reward and action. No detectable difference is not equivalence."),
}


def arm_runs(arm):
    learner, space, cfg, start = NEW[arm]
    runs = []
    for k in range(NB):
        b = C3.load_base(k)
        x0 = encode(C3.BASE_SPACE, b)
        if start == "base+0.5":
            x0 = np.concatenate([x0, np.full(len(space) - len(C3.BASE_SPACE), 0.5)])
        make = (lambda sp: (lambda u: {"gamma": 0.9, **decode_space(sp, u)}))(space)
        runs.append(dict(out_json=OUT / f"{arm}_r{k}.json", x0=x0, seed=28000 + 100 * ARM_IDX[arm] + k, r=k,
                         make=make, meta=dict(arm=arm, run=k, learner=learner, cfg=cfg)))
    return runs


def arms():
    log("registered", register(SPEC)[:12])
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, (learner, _, cfg, _) in NEW.items():
        cma_lockstep(arm_runs(arm), C3.AGENS, C3.score_fn(learner, cfg, 61_000_000, f"C3b {arm}"),
                     OUT / f"{arm}.ckpt", C3.POP, C3.SIG)
        log(f"C3b arm {arm} done")


def spec_of(arm):
    if arm in NEW:
        learner, _, cfg, _ = NEW[arm]
        return learner, cfg, OUT
    learner, _, cfg, _ = C3.ARMS[arm]
    return learner, cfg, C3.ARM_OUT


def evaluate(arm, M, lay):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"test_{arm}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    learner, cfg, d = spec_of(arm)
    if "bank" not in C3._BANKS:
        C3._BANKS["bank"] = SC._bank(TEST["bank"])
    out = {}
    for c, ks in enumerate(C3.calls(NB)):
        prms = [json.loads((d / f"{arm}_r{k}.json").read_text())["params"] for k in ks]
        nb = M.shape[1]
        d0 = {k_: float(v[0]) for k_, v in W.params(1).items()}
        keys = set(d0) | set().union(*prms)
        P = {k_: np.repeat([p.get(k_, d0.get(k_, 0.0)) for p in prms], nb) for k_ in keys}
        t = time.time()
        o = W.simulate(learner, P, np.tile(M, (1, len(ks))), C3._BANKS["bank"], np.tile(lay, len(ks)),
                       seed=TEST["sim"] + c, dqn={**C3.DQN, **cfg}, record=("wellbeing",))
        for i, k in enumerate(ks):
            out[k] = {kk: (v[i * nb:(i + 1) * nb] if v.ndim == 1 else v[:, i * nb:(i + 1) * nb]) for kk, v in o.items()}
        log(f"test {arm} bases {ks}: {len(ks) * nb} agents, {time.time() - t:.0f}s")
    _atomic_write(f, pickle.dumps(out))
    return out


def test():
    log("registered", register(SPEC)[:12])
    sch = [R.schedule(np.random.default_rng(TEST["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:C3.TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    T_ = M.shape[0]
    rows = []
    for arm in REUSED + list(NEW):
        for k, o in evaluate(arm, M, lay).items():
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
            rows.append(df)
    D = pd.concat(rows)
    D.to_csv(LOGS / "stageC3b_test_raw.csv", index=False)
    log("test raw written", len(D))


PRIMARY = [("P1", "gru", "control_ra", "greater"), ("P2", "aug_ra", "control_ra", "greater")]
SECONDARY = [("S1", "control_ra", "control", "two-sided"), ("S2", "aug_ra", "gru", "two-sided"),
             ("S3", "gru", "aug", "two-sided")]


def report():
    D = pd.read_csv(LOGS / "stageC3b_test_raw.csv")
    piv = D.groupby(["arm", "run", "schedule"]).score.mean()
    T = pd.concat([B10.family(piv, PRIMARY, "primary"), B10.family(piv, SECONDARY, "secondary")], ignore_index=True)
    T.to_csv(LOGS / "stageC3b_tests.csv", index=False)
    cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
    S = D.groupby("arm")[cols].mean()
    S.to_csv(LOGS / "stageC3b_summary.csv")
    bm = D.groupby(["arm", "run"]).score.mean().unstack(0)
    bm.to_csv(LOGS / "stageC3b_base_means.csv")
    rows = []
    for a, (_, sp, _, _) in NEW.items():
        for k in range(NB):
            j = json.loads((OUT / f"{a}_r{k}.json").read_text())
            for (kk, _), ui in zip(sp.items(), j["u"]):
                rows.append(dict(arm=a, base=k, param=kk, value=j["params"][kk], u=ui, on_bound=bool(ui < 0.02 or ui > 0.98)))
    Pm = pd.DataFrame(rows)
    Pm.to_csv(LOGS / "stageC3b_params.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["alternative"]).round(4).to_string())
    print(S.round(4).T.to_string())
    print(bm.round(3).to_string())
    print(Pm.groupby(["arm", "param"]).agg(median=("value", "median"), on_bound=("on_bound", "sum")).to_string())


def all_():
    for step in (arms, test, report):
        log("=== step", step.__name__)
        step()
    log("=== C3b finished")


if __name__ == "__main__":
    {"arms": arms, "test": test, "report": report, "all": all_}[sys.argv[1]]()
