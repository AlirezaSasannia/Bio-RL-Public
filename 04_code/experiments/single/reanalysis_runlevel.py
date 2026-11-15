"""Run- and base-level re-analysis of all registered test comparisons (no new simulation).

Why: every add-on run sits on one of four frozen base learners (base = run % 4), and the registered inference
(one-sided Wilcoxon over schedules of the run-averaged difference + hierarchical CI over runs, then schedules)
treats schedules as the replication unit and ignores variation between bases. In stage C1 the two criteria
disagreed (p_Holm < 0.05 while the CI included 0). This script recomputes every logged comparison with the base
learner as the top-level unit, to show which claims survive a stricter, base-aware inference.

Registered AFTER all results were seen (exploratory sensitivity re-analysis); the criterion below is fixed before
the new statistics are computed. It does not replace the registered verdicts; it is reported beside them.

Run: python reanalysis_runlevel.py register   (writes the spec to the registry; needs pyyaml)
     python reanalysis_runlevel.py analyse    (needs numpy, pandas, scipy; writes reanalysis_runlevel.csv)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "logs" / "single"

SPEC = {
    "id": "single_reanalysis_runlevel",
    "registered": "after all stage results (B3-D) were seen, before computing any base-level statistic",
    "status": "exploratory sensitivity re-analysis of existing test logs; no new simulation, no new data",
    "question": ("Which registered comparisons survive inference that treats the frozen base learner (base = run % 4) "
                 "as the top-level replication unit?"),
    "data": ("stageB3 (all four sets), B5, B7, B8, B9, B9b, C1, C2 and D (score measure only) *_test_raw.csv; every "
             "comparison row of the matching *_tests.csv"),
    "statistics": {
        "d": "difference of agent-averaged score per (run, schedule), arms paired by run index and schedule",
        "run_level": "t-test on run means (n = runs), one-sided in the registered direction, two-sided for secondary rows",
        "base_level": ("t-test on base means (runs averaged within base; n = 4, df = 3), same sidedness; Holm over the "
                       "rows of the original family (stage x family or set)"),
        "ci3": ("95% three-level bootstrap CI, 4000 draws: resample bases, then runs within each drawn base, then "
                "schedules (shared across runs, as in stage_b3.hier_ci); seed 0"),
    },
    "criterion": ("a comparison holds under the strict reading if base-level p_Holm < 0.05 AND the three-level CI "
                  "excludes 0 in the tested direction"),
    "interpretation": ("rows that pass the logged criterion but fail the strict one are reported as dependent on the "
                       "four shared bases; with n = 4 bases the strict test has low power, so a strict failure is not "
                       "evidence of no effect. Primary use: decide which claims need independent bases (new stage)."),
}

STAGES = ["B3", "B5", "B7", "B8", "B9", "B9b", "C1", "C2", "D"]


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out


def ci3(d, base_of_run, n=4000, seed=0):
    """d: (runs, schedules). Resample bases, runs within base, schedules."""
    rng = np.random.default_rng(seed)
    bases = np.unique(base_of_run)
    runs_by_base = {b: np.where(base_of_run == b)[0] for b in bases}
    R, S = d.shape
    out = np.empty(n)
    for k in range(n):
        bb = rng.choice(bases, size=len(bases))
        rr = np.concatenate([rng.choice(runs_by_base[b], size=len(runs_by_base[b])) for b in bb])
        ss = rng.integers(S, size=S)
        out[k] = d[np.ix_(rr, ss)].mean()
    return np.percentile(out, [2.5, 97.5])


def analyse():
    import pandas as pd
    from scipy import stats

    rows = []
    for st in STAGES:
        raw = pd.read_csv(LOGS / f"stage{st}_test_raw.csv")
        tests = pd.read_csv(LOGS / f"stage{st}_tests.csv")
        if "set" not in raw.columns:
            raw["set"] = "main"
        if "set" not in tests.columns:
            tests["set"] = "main"
        if "measure" in tests.columns:
            tests = tests[tests["measure"].astype(str).str.contains("score")]
        g = raw.groupby(["set", "arm", "run", "schedule"])["score"].mean()
        for _, t in tests.iterrows():
            comp = str(t["comparison"])
            if " - " not in comp:
                continue
            a, b = [x.strip() for x in comp.split(" - ", 1)]
            s = t["set"]
            try:
                A, B = g.loc[(s, a)].unstack("schedule"), g.loc[(s, b)].unstack("schedule")
            except KeyError:
                continue
            runs = sorted(set(A.index) & set(B.index))
            scheds = sorted(set(A.columns) & set(B.columns))
            d = A.loc[runs, scheds].to_numpy() - B.loc[runs, scheds].to_numpy()
            base_of_run = np.array(runs) % 4
            run_means = d.mean(axis=1)
            base_means = np.array([run_means[base_of_run == bb].mean() for bb in np.unique(base_of_run)])
            fam = str(t["family"]) if "family" in t and isinstance(t["family"], str) else "registered"
            secondary = "secondary" in fam.lower()
            alt = "two-sided" if secondary else "greater"
            p_run = stats.ttest_1samp(run_means, 0, alternative=alt).pvalue
            p_base = stats.ttest_1samp(base_means, 0, alternative=alt).pvalue
            lo, hi = ci3(d, base_of_run)
            lp = t.get("p_holm", np.nan)
            if pd.isna(lp):
                lp = t.get("p", np.nan)
            if "verdict" in t and isinstance(t["verdict"], str) and t["verdict"]:
                vlog = t["verdict"]
            else:
                ok = (t["ci_lo"] > 0 or (secondary and t["ci_hi"] < 0)) and lp < 0.05
                vlog = "PASS" if ok else "FAIL"
            rows.append(dict(stage=st, set=s, family=fam, pred=t.get("pred", ""), comparison=comp,
                             diff=d.mean(), diff_logged=t["diff"], n_runs=len(runs), n_bases=len(base_means),
                             runs_better=int((run_means > 0).sum()), bases_better=int((base_means > 0).sum()),
                             p_run=p_run, p_base=p_base, ci3_lo=lo, ci3_hi=hi,
                             ci_logged_lo=t["ci_lo"], ci_logged_hi=t["ci_hi"], p_holm_logged=lp,
                             verdict_logged=vlog, two_sided=secondary))
    R = pd.DataFrame(rows)
    R["p_base_holm"] = np.nan
    for _, idx in R.groupby(["stage", "set", "family"]).groups.items():
        R.loc[idx, "p_base_holm"] = holm(R.loc[idx, "p_base"].to_numpy())
    excl = np.where(R["two_sided"], (R["ci3_lo"] > 0) | (R["ci3_hi"] < 0), R["ci3_lo"] > 0)
    R["verdict_strict"] = np.where((R["p_base_holm"] < 0.05) & excl, "PASS", "FAIL")
    R["changed"] = (R["verdict_logged"].str.upper().str.startswith("PASS")) & (R["verdict_strict"] == "FAIL")
    R.to_csv(LOGS / "reanalysis_runlevel.csv", index=False)
    print(R[["stage", "set", "comparison", "diff", "bases_better", "p_base_holm", "ci3_lo", "ci3_hi",
             "verdict_logged", "verdict_strict"]].to_string())


def do_register():
    sys.path.insert(0, str(ROOT / "src"))
    from bio_rl.battery.prereg import register
    print(register(SPEC))


if __name__ == "__main__":
    {"register": do_register, "analyse": analyse}[sys.argv[1]]()
