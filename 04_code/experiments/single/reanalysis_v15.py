"""Re-analyses for paper v1.5 (logs already seen, no new test set; listed in the revision_v1.5 response matrix).
(a) every independent-base test (B10, B11, C3, C3b, C3c, X1, X2): exact sign-flip permutation p (two-sided and in the
    registered direction) beside the t-test, t interval (primary interval), d_z, effect as share of the stage-A
    headroom (tabular only), what the t interval excludes, and the minimum detectable effect at n = 8;
(b) registered primary predictions with and without B4 (the input-subset stage the paper interprets as exploratory);
(c) B10 test set: restricted mean survival time (tau = 6000) and Kaplan-Meier curves per arm, base as cluster;
(d) training-fitness slope of the B10 bases over their last five generations (25-29; the bases ran 30 generations);
(e) whether CMA-ES means per generation were saved (budget curves).
Run: python reanalysis_v15.py -> results/logs/single/reanalysis_v15/*.csv"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reanalysis_runlevel import holm

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "logs" / "single"
OUT = LOGS / "reanalysis_v15"
T_LIFE = 6000
HEADROOM = float(pd.read_csv(LOGS / "stageA_summary.csv", index_col=0).iloc[:, 0]["headroom"])
TAB = {"B10", "B11", "X1"}


def signflip(bm, alternative="two-sided"):
    bm = np.asarray(bm, float)
    s = np.array(list(itertools.product([1, -1], repeat=len(bm))))
    m = (s * bm).mean(1)
    obs = bm.mean()
    if alternative == "greater":
        return float((m >= obs - 1e-12).mean())
    if alternative == "less":
        return float((m <= obs + 1e-12).mean())
    return float((np.abs(m) >= abs(obs) - 1e-12).mean())


def mde_dz(n=8, alpha=0.05, power=0.8):
    """smallest standardised paired effect d_z detectable with the given power, two-sided one-sample t-test"""
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    f = lambda d: stats.nct.sf(tc, n - 1, d * np.sqrt(n)) + stats.nct.cdf(-tc, n - 1, d * np.sqrt(n)) - power
    lo, hi = 0.01, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


MDE = mde_dz()


def row(stage, pred, family, comp, alt, bm, verdict):
    bm = np.asarray(bm, float)
    n = len(bm)
    sd = bm.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    alt = alt if alt in ("greater", "less") else "two-sided"
    return dict(stage=stage, pred=pred, family=family, comparison=comp, registered_alternative=alt, diff=bm.mean(),
                t_lo=bm.mean() - h, t_hi=bm.mean() + h, bases_better=int((bm > 0).sum()), n_bases=n,
                p_t_two=float(stats.ttest_1samp(bm, 0).pvalue),
                p_t_registered=float(stats.ttest_1samp(bm, 0, alternative=alt).pvalue),
                p_signflip_two=signflip(bm), p_signflip_registered=signflip(bm, alt),
                d_z=bm.mean() / sd if sd > 0 else np.nan, share_headroom=bm.mean() / HEADROOM if stage in TAB else np.nan,
                mde=MDE * sd, verdict_registered=verdict)


def tests():
    rows = []
    STAGES = {"B10": ("stageB10_tests.csv", ["stageB10_test_raw.csv"]),
              "B11": ("stageB11_tests.csv", ["stageB11_test_raw.csv"]),
              "C3": ("stageC3_tests.csv", ["stageC3_test_raw.csv"]),
              "C3b": ("stageC3b_tests.csv", ["stageC3b_test_raw.csv"]),
              "C3c": ("stageC3c_tests.csv", ["stageC3c_test_raw.csv"])}
    D2 = pd.concat([pd.read_csv(LOGS / "stageC3b_test_raw.csv"), pd.read_csv(LOGS / "stageC3c_c3bset_raw.csv")])
    for st, (tf, raws) in STAGES.items():
        Tt = pd.read_csv(LOGS / tf)
        D = pd.concat([pd.read_csv(LOGS / f) for f in raws])
        for _, r in Tt.iterrows():
            src = D2 if r.family == "secondary_c3bset" else D
            piv = src.groupby(["arm", "run", "schedule"]).score.mean()
            a, b = [s.strip() for s in r.comparison.split(" - ")]
            bm = (piv[a] - piv[b]).groupby("run").mean().to_numpy()
            rows.append(row(st, r.pred, r.family, r.comparison, r.alternative, bm, r.verdict))
    # X1 (revision v1.3; two-sided by registration); base means as in stage_x1.report
    G = pd.read_csv(LOGS / "revX1" / "x1_groups.csv")
    S = G.pivot_table(index=["cond", "base", "group"], columns="arm", values="score_post")
    sw = G[G.cond == "switching"].pivot_table(index=["base", "group"], columns="arm", values="score")
    names = ["safe", "danger", "uncontrollable", "scarce", "volatile"]
    st_ = {rn: S.loc[rn] for rn in names}
    pooled = sum(st_.values()) / len(names)
    bmf = lambda d: d.groupby("base").mean().to_numpy()
    fam = pd.read_csv(LOGS / "revision_v13_tests.csv")
    vd = dict(zip(fam.test, fam.verdict))
    for c, rn in enumerate(names):
        rows.append(row("X1", f"X1-P{c + 1}", "primary", f"generic - bias, stationary {rn}", "two-sided",
                        bmf(st_[rn].generic - st_[rn].bias), vd.get(f"X1-P{c + 1}")))
    rows.append(row("X1", "X1-P6", "primary", "generic - bias, pooled stationary", "two-sided",
                    bmf(pooled.generic - pooled.bias), vd.get("X1-P6")))
    for c, rn in enumerate(names):
        rows.append(row("X1", f"X1-S{c + 1}", "secondary", f"generic - control, stationary {rn}", "two-sided",
                        bmf(st_[rn].generic - st_[rn].control), vd.get(f"X1-S{c + 1}")))
    rows.append(row("X1", "X1-S6", "secondary", "bias - control, pooled stationary", "two-sided",
                    bmf(pooled.bias - pooled.control), vd.get("X1-S6")))
    rows.append(row("X1", "X1-S7", "secondary", "(generic - bias) switching - pooled stationary", "two-sided",
                    bmf(sw.generic - sw.bias) - bmf(pooled.generic - pooled.bias), vd.get("X1-S7")))
    # X2
    X2 = pd.read_csv(LOGS / "revX2X3" / "x2_test_raw.csv")
    piv = X2.groupby(["arm", "run", "schedule"]).score.mean()
    for pred, fam_, (a, b) in (("X2-P1", "primary", ("gru", "ff_big_seq")), ("X2-S1", "secondary", ("ff_big_seq", "ff_big")),
                               ("X2-S2", "secondary", ("ff_big_seq", "control_ra")),
                               ("X2-S3", "secondary", ("gru", "ff_big"))):
        rows.append(row("X2", pred, fam_, f"{a} - {b}", "two-sided", (piv[a] - piv[b]).groupby("run").mean().to_numpy(),
                        vd.get(pred)))
    A = pd.DataFrame(rows)
    A["p_t_two_holm"] = np.nan
    for _, idx in A.groupby(["stage", "family"]).groups.items():
        A.loc[idx, "p_t_two_holm"] = holm(A.loc[idx, "p_t_two"].to_numpy())
    A["t_excludes_0"] = (A.t_lo > 0) | (A.t_hi < 0)
    A["excludes"] = [("excludes 0" if r.t_excludes_0 else "") + f"; excludes < {r.t_lo:+.3f} and > {r.t_hi:+.3f}"
                     for r in A.itertuples()]
    # registered one-sided passes that would not pass two-sided at Holm 0.05 (t interval must also exclude 0)
    A["pass_two_sided"] = (A.p_t_two_holm < 0.05) & A.t_excludes_0
    A.to_csv(OUT / "tests_v15.csv", index=False)
    return A


def b4_count():
    P = pd.read_csv(LOGS / "reanalysis_v13" / "bh_primary.csv")
    rows = [dict(scope="all registered primary predictions", n=len(P), passed=int(P.registered_pass.sum())),
            dict(scope="without B4 (input subsets; interpreted as exploratory: arms differ in parameter count)",
                 n=int((P.stage != "B4").sum()), passed=int(P[P.stage != "B4"].registered_pass.sum()))]
    pd.DataFrame(rows).to_csv(OUT / "b4_count.csv", index=False)
    return rows


def survival():
    D = pd.read_csv(LOGS / "stageB10_test_raw.csv")
    D["life"] = np.round(D.survival * T_LIFE)
    grid = np.arange(0, T_LIFE + 1, 100)
    km, rm = [], []
    tint = lambda v: (v.mean() - stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)),
                      v.mean() + stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    for arm, sub in D.groupby("arm"):
        per = np.array([[(g.life.to_numpy() > t).mean() for t in grid] for _, g in sub.groupby("run")])
        ci = [tint(per[:, i]) if per[:, i].std() > 0 else (per[:, i].mean(), per[:, i].mean()) for i in range(len(grid))]
        km.append(pd.DataFrame(dict(arm=arm, t=grid, surv=per.mean(0), lo=[c[0] for c in ci], hi=[c[1] for c in ci])))
        r = sub.groupby("run").life.mean().to_numpy()
        lo, hi = tint(r)
        rm.append(dict(arm=arm, rmst=r.mean(), lo=lo, hi=hi, died_share=float((sub.life < T_LIFE).mean())))
    life = D.groupby(["arm", "run"]).life.mean().unstack("arm")
    for a, b in (("generic", "bias"), ("generic", "control"), ("bias", "control"), ("generic", "bio"),
                 ("generic", "yoked")):
        d = (life[a] - life[b]).to_numpy()
        lo, hi = tint(d)
        rm.append(dict(arm=f"{a} - {b}", rmst=d.mean(), lo=lo, hi=hi, p_t=float(stats.ttest_1samp(d, 0).pvalue),
                       p_signflip=signflip(d)))
    pd.concat(km).to_csv(OUT / "b10_km.csv", index=False)
    pd.DataFrame(rm).to_csv(OUT / "b10_rmst.csv", index=False)
    return pd.DataFrame(rm)


def base_slopes():
    rows = []
    for k in range(8):
        h = json.loads((LOGS / "stageB10_opt" / f"base_r{k}.json").read_text())["history"]
        rows.append(dict(base=k, n_gens=len(h), g19=h[19], g29=h[-1],
                         slope_15_19=np.polyfit(np.arange(5), h[15:20], 1)[0],
                         slope_last5=np.polyfit(np.arange(5), h[-5:], 1)[0]))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "b10_base_slopes.csv", index=False)
    return B


def cma_means_saved():
    """budget curves need the CMA mean at intermediate generations; the run JSONs keep only the final mean and the
    population-mean training fitness per generation (history), and checkpoints are deleted at the end of a run"""
    keys = set()
    for f in (LOGS / "stageB10_opt").glob("*.json"):
        keys |= set(json.loads(f.read_text()))
    ck = list(LOGS.rglob("*.ckpt"))
    note = dict(json_keys=" ".join(sorted(keys)), checkpoints_left=len(ck),
                per_generation_means_saved=False if not ck and "means" not in keys else True)
    pd.DataFrame([note]).to_csv(OUT / "cma_means_note.csv", index=False)
    return note


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    A = tests()
    print(A[["stage", "pred", "comparison", "registered_alternative", "diff", "t_lo", "t_hi", "p_t_two", "p_signflip_two",
             "p_signflip_registered", "d_z", "share_headroom", "mde", "verdict_registered", "pass_two_sided"]]
          .round(4).to_string())
    print("MDE d_z (n = 8, two-sided 0.05, power 0.8):", round(MDE, 3), "headroom", round(HEADROOM, 4))
    print(b4_count())
    print(survival().round(3).to_string())
    print(base_slopes().round(4).to_string())
    print(cma_means_saved())
