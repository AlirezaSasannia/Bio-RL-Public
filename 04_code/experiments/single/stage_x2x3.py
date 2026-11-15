"""Revision X2/X3 (family single_revision_v1_3, see revision_v13.py), GPU.
X2: ff_big_seq = C3c ff_big trained with the GRU's 2 x 4 sequence replay (C3c budget, C3 bases, same generation seeds);
    tested with gru, ff_big and control_ra on a new set (schedules 54000+).
X3: evaluation only: every frozen C3/C3b/C3c solution on the used test sets it lacks; pooled random-effects analysis.
Run: python stage_x2x3.py [x3|arms|x2test|report|family|all]
Smoke (dev seeds, tiny sizes, scratch output, nothing registered): set X2_SMOKE=<scratch dir>."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_c as SC
import stage_c3 as C3
import stage_c3b as C3B
import stage_c3c as C3C
from _resume import _atomic_write, cma_lockstep
from bio_rl.battery.prereg import register
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.adaptive_search import decode_space, encode
from bio_rl.torchsim import world as W
from reanalysis_runlevel import ci3, holm
from revision_v13 import SPEC

LOGS = Path(__file__).resolve().parents[2] / "results" / "logs" / "single"
OUT_W = LOGS / "revX2X3"                  # everything this script writes
log = SC.log
NB, NA_ = 8, 4
NS = 128
CFG_SEQ = {"ra": True, "H": C3C.H_BIG, "seq": True}
X2SET = dict(sched=54000, bank=5_300_000, sim=330_000)
SETS = {"c3": dict(sched=22000, bank=2_200_000, sim=110_000), "c3b": dict(sched=44000, bank=2_400_000, sim=140_000),
        "c3c": dict(sched=45000, bank=2_500_000, sim=150_000)}
LOGGED = {"c3": ["stageC3_test_raw.csv"], "c3b": ["stageC3b_test_raw.csv", "stageC3c_c3bset_raw.csv"],
          "c3c": ["stageC3c_test_raw.csv"]}
ALL_ARMS = ["control", "aug", "film", "generic", "bias", "gru", "gru_aug", "metagrad", "control_ra", "aug_ra", "ff_big"]
SMOKE = os.environ.get("X2_SMOKE", "").strip()
if SMOKE:     # code check only: dev seeds, 1 generation, tiny batches, short lives, scratch output
    OUT_W = Path(SMOKE)
    C3.AGENS, C3.POP, C3.NAG, C3.TCUT = 1, 4, 4, 300
    NS = 4
    ALL_ARMS = ["control", "gru", "control_ra", "aug_ra", "ff_big", "aug"]     # every learner type once
    X2SET = dict(sched=97300, bank=1_960_000, sim=136_000)
    SETS = {"c3": dict(sched=97400, bank=1_960_000, sim=137_000), "c3b": dict(sched=97500, bank=1_960_000, sim=138_000),
            "c3c": dict(sched=97600, bank=1_960_000, sim=139_000)}
    register = lambda spec: "smoke-no-registration"
OPT_SEQ, CACHE = OUT_W / "ff_big_seq_opt", OUT_W / "cache"


def spec_of(arm):
    """learner, dqn cfg, directory of the frozen solutions"""
    if arm == "ff_big_seq":
        return "dqnv", CFG_SEQ, OPT_SEQ
    return C3C.spec_of(arm)


def schedules(ts):
    sch = [R.schedule(np.random.default_rng(ts["sched"] + i)) for i in range(NS)]
    M = A.regime_matrix([s for s in sch for _ in range(NA_)], R.T_LIFE)[:C3.TCUT]
    lay = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
    return M, lay


def evaluate(name, arm, ts):
    """All 8 bases of one frozen arm on test set ts (two calls of 4 bases, simulator seed ts['sim'] + c); cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_{arm}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    M, lay = schedules(ts)
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


def logged_arms(sname):
    if SMOKE:
        return set()
    return set().union(*[set(pd.read_csv(LOGS / f, usecols=["arm"]).arm) for f in LOGGED[sname]])


# ------------------------------------------------------------------ X3: evaluation only on the used sets
def x3():
    log("registered", register(SPEC)[:12])
    OUT_W.mkdir(parents=True, exist_ok=True)
    for sname, ts in SETS.items():
        M, _ = schedules(ts)
        have = logged_arms(sname)
        rows = [frame(a, k, o, M) for a in ALL_ARMS if a not in have for k, o in evaluate(f"x3_{sname}", a, ts).items()]
        pd.concat(rows).to_csv(OUT_W / f"x3_{sname}_raw.csv", index=False)
        log(f"X3 {sname}: evaluated {sorted(set(ALL_ARMS) - have)}")
    if not SMOKE:     # determinism check: control on the C3b set must reproduce the logged C3b scores
        M, _ = schedules(SETS["c3b"])
        o = evaluate("check_c3b", "control", SETS["c3b"])
        raw = pd.read_csv(LOGS / "stageC3b_test_raw.csv")
        worst = max(float(np.abs(raw[(raw.arm == "control") & (raw.run == k)].score.to_numpy() - o[k]["score"]).max())
                    for k in range(NB))
        log("X3 determinism check (control, C3b set), max |diff| =", worst)
        (OUT_W / "x3_check.txt").write_text(f"max_abs_diff {worst}\n")
        if worst > 1e-6:
            raise SystemExit("re-evaluation does not reproduce the logged C3b scores")


# ------------------------------------------------------------------ X2
def arms():
    log("registered", register(SPEC)[:12])
    OPT_SEQ.mkdir(parents=True, exist_ok=True)
    space = C3.BASE_SPACE
    make = lambda u: {"gamma": 0.9, **decode_space(space, u)}
    runs = [dict(out_json=OPT_SEQ / f"ff_big_seq_r{k}.json", x0=encode(space, C3.load_base(k)), seed=28300 + k, r=k,
                 make=make, meta=dict(arm="ff_big_seq", run=k, learner="dqnv", cfg=CFG_SEQ)) for k in range(NB)]
    cma_lockstep(runs, C3.AGENS, C3.score_fn("dqnv", CFG_SEQ, 61_000_000, "X2 ff_big_seq"),
                 OPT_SEQ / "ff_big_seq.ckpt", C3.POP, C3.SIG)
    log("X2 arm ff_big_seq done")


X2_ARMS = ["ff_big_seq", "gru", "ff_big", "control_ra"]


def x2test():
    log("registered", register(SPEC)[:12])
    M, _ = schedules(X2SET)
    D = pd.concat([frame(a, k, o, M) for a in X2_ARMS for k, o in evaluate("x2", a, X2SET).items()])
    D.to_csv(OUT_W / "x2_test_raw.csv", index=False)
    log("X2 test raw written", len(D))


# ------------------------------------------------------------------ report
def tint(bm):
    h = stats.t.ppf(0.975, len(bm) - 1) * bm.std(ddof=1) / np.sqrt(len(bm))
    return bm.mean() - h, bm.mean() + h


def compare(piv, a, b, name):
    d = (piv[a] - piv[b]).unstack()                       # bases x schedules
    bm = d.mean(1).to_numpy()
    lo, hi = ci3(d.to_numpy(), np.asarray(d.index))
    tl, th = tint(bm)
    return dict(test=name, comparison=f"{a} - {b}", diff=float(d.to_numpy().mean()), bases_better=int((bm > 0).sum()),
                n_bases=len(bm), p_t=float(stats.ttest_1samp(bm, 0).pvalue),
                p_wilcoxon=float(stats.wilcoxon(bm, method="exact").pvalue), ci3_lo=lo, ci3_hi=hi, t_lo=tl, t_hi=th)


X3_COMP = [("gru", "control_ra"), ("aug_ra", "control_ra"), ("control_ra", "control"), ("aug_ra", "gru"),
           ("gru", "aug"), ("gru", "ff_big"), ("ff_big", "control_ra"), ("ff_big", "aug_ra"), ("gru", "aug_ra"),
           ("film", "control"), ("aug", "control"), ("aug", "generic"), ("film", "gru"), ("aug", "gru"),
           ("gru", "control"), ("gru_aug", "gru"), ("metagrad", "control"), ("aug", "bias"), ("film", "bias"),
           ("generic", "bias"), ("metagrad", "bias")]


def pooled_data():
    parts = {}
    for sname in SETS:
        fs = ([] if SMOKE else [LOGS / f for f in LOGGED[sname]]) + [OUT_W / f"x3_{sname}_raw.csv"]
        D = pd.concat([pd.read_csv(f) for f in fs])
        D = D.drop_duplicates(["arm", "run", "schedule", "score", "survival"])
        parts[sname] = D
    return parts


def dersimonian_laird(est, se):
    w = 1 / se ** 2
    mu_f = (w * est).sum() / w.sum()
    q = (w * (est - mu_f) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = (ws * est).sum() / ws.sum()
    s = np.sqrt(1 / ws.sum())
    return mu, mu - 1.96 * s, mu + 1.96 * s, tau2, q


def report():
    rows = []
    if (OUT_W / "x2_test_raw.csv").exists():
        D = pd.read_csv(OUT_W / "x2_test_raw.csv")
        piv = D.groupby(["arm", "run", "schedule"]).score.mean()
        rows = [compare(piv, "gru", "ff_big_seq", "X2-P1"), compare(piv, "ff_big_seq", "ff_big", "X2-S1"),
                compare(piv, "ff_big_seq", "control_ra", "X2-S2"), compare(piv, "gru", "ff_big", "X2-S3")]
        pd.DataFrame(rows).to_csv(OUT_W / "x2_tests_raw.csv", index=False)
        cols = ["score", "survival", "alive_wellbeing"] + [f"died_{r}" for r in R.NAMES]
        D.groupby("arm")[cols].mean().to_csv(OUT_W / "x2_summary.csv")
        prm = []
        for k in range(NB):
            j = json.loads((OPT_SEQ / f"ff_big_seq_r{k}.json").read_text())
            for (kk, _), ui in zip(C3.BASE_SPACE.items(), j["u"]):
                prm.append(dict(arm="ff_big_seq", base=k, param=kk, value=j["params"][kk], u=ui,
                                on_bound=bool(ui < 0.02 or ui > 0.98)))
        pd.DataFrame(prm).to_csv(OUT_W / "x2_params.csv", index=False)
        print(pd.DataFrame(rows).round(4).to_string())
        print(D.groupby("arm")[cols].mean().round(4).to_string())
    if all((OUT_W / f"x3_{s}_raw.csv").exists() for s in SETS):
        parts = pooled_data()
        out = []
        for a, b in X3_COMP:
            per = {}
            for sname, D in parts.items():
                if not {a, b} <= set(D.arm):
                    continue
                piv = D.groupby(["arm", "run", "schedule"]).score.mean()
                d = (piv[a] - piv[b]).unstack()
                bm = d.mean(1).to_numpy()
                # schedule variance component of the difference (two-way base x schedule decomposition)
                resid = d.to_numpy() - d.mean(1).to_numpy()[:, None] - d.mean(0).to_numpy()[None, :] + d.to_numpy().mean()
                ms_s = NB * d.mean(0).var(ddof=1)
                ms_e = (resid ** 2).sum() / ((NB - 1) * (d.shape[1] - 1))
                per[sname] = dict(est=bm.mean(), se=bm.std(ddof=1) / np.sqrt(NB), d=d,
                                  var_sched=max(0.0, (ms_s - ms_e) / NB))
            if len(per) < 3:
                continue
            dall = pd.concat([v["d"] for v in per.values()], axis=1, ignore_index=True)   # bases x 384 schedules
            bm = dall.mean(1).to_numpy()
            lo, hi = ci3(dall.to_numpy(), np.asarray(dall.index))
            tl, th = tint(bm)
            est = np.array([v["est"] for v in per.values()])
            se = np.array([v["se"] for v in per.values()])
            mu, rlo, rhi, tau2, q = dersimonian_laird(est, se)
            out.append(dict(comparison=f"{a} - {b}", pooled_diff=float(dall.to_numpy().mean()),
                            bases_better=int((bm > 0).sum()), p_t=float(stats.ttest_1samp(bm, 0).pvalue),
                            ci3_lo=lo, ci3_hi=hi, t_lo=tl, t_hi=th, re_mu=mu, re_lo=rlo, re_hi=rhi, tau2=tau2, Q=q,
                            **{f"est_{s}": v["est"] for s, v in per.items()},
                            **{f"varsched_{s}": v["var_sched"] for s, v in per.items()}))
        X3 = pd.DataFrame(out)
        X3["p_t_holm"] = holm(X3.p_t.to_numpy())
        X3.to_csv(OUT_W / "x3_pooled.csv", index=False)
        pd.set_option("display.width", 250)
        print(X3.round(4).to_string())


def family():
    """Holm over the registered primary family (X1-P1..P6, X2-P1) and the secondary family (X1-S1..S7, X2-S1..S3)."""
    x1f = LOGS / "revX1" / "x1_tests_raw.csv"
    x2f = OUT_W / "x2_tests_raw.csv"
    if not (x1f.exists() and x2f.exists()):
        log("family: X1 or X2 tests missing; run stage_x1.py report and this report first")
        return
    T = pd.concat([pd.read_csv(x1f), pd.read_csv(x2f)], ignore_index=True)
    T = T[T.test.str.contains(r"-P|-S")].copy()
    T["family"] = np.where(T.test.str.contains("-P"), "primary", "secondary")
    T["p_t_holm"] = np.nan
    for fam, idx in T.groupby("family").groups.items():
        T.loc[idx, "p_t_holm"] = holm(T.loc[idx, "p_t"].to_numpy())
    excl = ((T.ci3_lo > 0) | (T.ci3_hi < 0)) & ((T.t_lo > 0) | (T.t_hi < 0))
    T["verdict"] = np.where((T.p_t_holm < 0.05) & excl, "PASS", "n.e.")
    out = LOGS / "revision_v13_tests.csv"
    T.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(T[["family", "test", "comparison", "diff", "bases_better", "p_t", "p_t_holm", "ci3_lo", "ci3_hi", "t_lo",
             "t_hi", "verdict"]].round(4).to_string())


def all_():
    for step in (x3, arms, x2test, report, family):
        log("=== step", step.__name__)
        step()
    log("=== X2/X3 finished")


if __name__ == "__main__":
    {"x3": x3, "arms": arms, "x2test": x2test, "report": report, "family": family, "all": all_}[sys.argv[1]]()
