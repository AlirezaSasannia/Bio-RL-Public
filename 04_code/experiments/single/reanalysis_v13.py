"""Re-analyses for paper v1.3 (listed in the revision_v1.3 registration; all on logs already seen, no new test set).
(a) t-based (df = 7) intervals and two-sided p for every independent-base test (B10, B11, C3, C3b, C3c);
(b) Benjamini-Hochberg sensitivity check over all registered primary predictions of the single-agent study;
(c) project-level table of every registration with its outcome;
(d) learned constant offsets of the B10 'bias' arm; (e) training fitness at CMA generations 0, 10, 19 per arm.
Run: python reanalysis_v13.py   -> results/logs/single/reanalysis_v13_*.csv"""
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
OUT = LOGS / "reanalysis_v13"


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------ (a)
STAGES = {"B10": ("stageB10_tests.csv", ["stageB10_test_raw.csv"]),
          "B11": ("stageB11_tests.csv", ["stageB11_test_raw.csv"]),
          "C3": ("stageC3_tests.csv", ["stageC3_test_raw.csv"]),
          "C3b": ("stageC3b_tests.csv", ["stageC3b_test_raw.csv"]),
          "C3c": ("stageC3c_tests.csv", ["stageC3c_test_raw.csv"])}


def tintervals():
    rows = []
    for st, (tf, raws) in STAGES.items():
        T = pd.read_csv(LOGS / tf)
        D = pd.concat([pd.read_csv(LOGS / f) for f in raws])
        D2 = pd.concat([pd.read_csv(LOGS / "stageC3b_test_raw.csv"), pd.read_csv(LOGS / "stageC3c_c3bset_raw.csv")])
        for _, r in T.iterrows():
            src = D2 if r.family == "secondary_c3bset" else D
            piv = src.groupby(["arm", "run", "schedule"]).score.mean()
            a, b = [s.strip() for s in r.comparison.split(" - ")]
            bm = (piv[a] - piv[b]).groupby("run").mean().to_numpy()
            n = len(bm)
            h = stats.t.ppf(0.975, n - 1) * bm.std(ddof=1) / np.sqrt(n)
            rows.append(dict(stage=st, pred=r.pred, family=r.family, comparison=r.comparison,
                             registered_alternative=r.alternative, diff=bm.mean(), diff_logged=r["diff"],
                             ci3_lo=r.ci3_lo, ci3_hi=r.ci3_hi, t_lo=bm.mean() - h, t_hi=bm.mean() + h,
                             p_t_registered=r.p_t, p_t_holm_registered=r.p_t_holm,
                             p_two_sided=float(stats.ttest_1samp(bm, 0).pvalue), verdict_registered=r.verdict))
    A = pd.DataFrame(rows)
    A["p_two_sided_holm"] = np.nan
    for _, idx in A.groupby(["stage", "family"]).groups.items():
        A.loc[idx, "p_two_sided_holm"] = holm(A.loc[idx, "p_two_sided"].to_numpy())
    A["t_excludes_0"] = (A.t_lo > 0) | (A.t_hi < 0)
    A["ci3_excludes_0"] = (A.ci3_lo > 0) | (A.ci3_hi < 0)
    A.to_csv(OUT / "tintervals.csv", index=False)
    return A


# ------------------------------------------------------------------ (b)
def primaries():
    rd = lambda f: pd.read_csv(LOGS / f)
    rows = []
    add = lambda st, test, comp, p, verdict: rows.append(dict(stage=st, test=test, comparison=comp, p=float(p),
                                                              verdict=str(verdict)))
    for _, r in rd("stageB_tests.csv").iterrows():
        add("B", r.test, r.comparison, r.p, r.verdict)
    for _, r in rd("stageB2_tests.csv").iterrows():
        add("B2", r.test, f"{r.hormones} - {r.control}", r.p, r.verdict)
    for _, r in rd("stageB3_tests.csv").query("set == 'main'").iterrows():
        add("B3", "", r.comparison, r.p, r.verdict)
    for _, r in rd("stageB4_tests.csv").dropna(subset=["p_holm"]).iterrows():
        add("B4", "", r.comparison, r.p, "PASS" if r.p_holm < 0.05 and r.ci_lo > 0 else "FAIL")
    for st in ("B5", "B7", "B8"):
        for _, r in rd(f"stage{st}_tests.csv").iterrows():
            if str(r.family).startswith("primary"):
                add(st, "", r.comparison, r.p, "PASS" if r.p_holm < 0.05 and r.ci_lo > 0 else "FAIL")
    for _, r in rd("stageB6_tests.csv").dropna(subset=["p_holm"]).iterrows():
        add("B6", r.switch, r.comparison, r.p, "PASS" if r.p_holm < 0.05 and r.ci_lo > 0 else "FAIL")
    for st in ("B9", "B9b", "C1", "C2", "D"):
        for _, r in rd(f"stage{st}_tests.csv").query("family == 'primary'").iterrows():
            add(st, r.pred, r.comparison, r.p, r.verdict)
    for st in ("B10", "B11", "C3", "C3b", "C3c"):
        for _, r in rd(f"stage{st}_tests.csv").query("family == 'primary'").iterrows():
            add(st, r.pred, r.comparison, r.p_t, r.verdict)
    for _, r in rd("stageB11_weight_tests.csv").iterrows():
        add("B11", r.pred, "sign of split weight", r.p_sign, r.verdict)
    P = pd.DataFrame(rows)
    P["p_bh"] = bh(P.p.to_numpy())
    P["bh_reject"] = P.p_bh < 0.05
    P["registered_pass"] = P.verdict.str.upper().str.startswith("PASS")
    P["agree"] = P.bh_reject == P.registered_pass
    P.to_csv(OUT / "bh_primary.csv", index=False)
    return P


# ------------------------------------------------------------------ (c)
NOTE = {
    "battery_v1": ("stress-phenomenon battery, reactive architecture", "companion paper"),
    "sensitivity_and_calibration_v1": ("battery sensitivity / calibration", "companion paper"),
    "predict_v1": ("held-out phenomenon predictions", "companion paper"),
    "recovery_v1": ("recovery phenomena", "companion paper"),
    "recovery_v1b_offgrid": ("recovery phenomena, off-grid amendment", "companion paper"),
    "sensitivity_v2": ("battery sensitivity v2", "companion paper"),
    "battery_v2": ("stress-phenomenon battery v2", "companion paper"),
    "evolution_v1": ("evolution in fixed-regime worlds, study 1", "fail: endocrine survived less (-0.061)"),
    "evolution_v1_amendment1": ("study 1 search-range amendment", "amendment"),
    "evolution_v1_followup_invasion": ("study 1 invasion follow-up", "no channel positive"),
    "evolution_v2": ("evolution study 2 (lethal injuries)", "fail: endocrine -0.032 overall"),
    "single_stageA_headroom": ("headroom gate, this world", "gate pass (24% of range)"),
    "single_stageB3_amendment1": ("B3 TOST margin and sets", "amendment"),
    "single_stageB5_amendment1": ("B5 augtab0 start", "amendment"),
    "single_stageB7_addendum_bias": ("B7 constant-offset arm", "amendment"),
    "single_clamp_exploratory": ("clamping one input", "exploratory"),
    "single_b9_dose_exploratory": ("SPSA dose diagnostic", "exploratory"),
    "single_torch_world_validation": ("GPU world equivalence (TOST)", "pass"),
    "single_stageC0_dqn_base": ("cell-input network bases", "bases tuned"),
    "single_stageC0_amendment1": ("C0 exploration-cap amendment", "amendment"),
    "single_stageC2_base": ("view-network bases (shared)", "bases tuned"),
    "single_stageC3_base": ("view-network bases (independent)", "bases tuned"),
    "single_reanalysis_runlevel": ("base-level re-analysis (after results seen)", "sensitivity analysis"),
    "single_stageB11_dev_exploratory": ("split-surprise development", "exploratory"),
    "single_stageE2_0_gate": ("second-world gate 1", "gate fail (5.4%)"),
    "single_stageE2b_0_gate": ("second-world gate 2 (last)", "gate fail (8.7%)"),
    "single_revision_v1_3": ("revision experiments X1-X3", "pending"),
}
STAGE_OF = {"single_stageB_controllers": "B", "single_stageB2_addon": "B2", "single_stageB3_placebo": "B3",
            "single_stageB4_inputs": "B4", "single_stageB5_info_vs_modulation": "B5",
            "single_stageB6_adaptation_speed": "B6", "single_stageB7_more_runs": "B7",
            "single_stageB8_constant_shift": "B8", "single_stageB9_modern_baselines": "B9",
            "single_stageB9b_warmstart": "B9b", "single_stageC1_fa_arms": "C1", "single_stageC2_film_vs_input": "C2",
            "single_stageD_hormone_keyed_memory": "D", "single_stageB10_independent_bases": "B10",
            "single_stageB11_surprise_split": "B11", "single_stageC3_independent_bases_gru_metagrad": "C3",
            "single_stageC3b_gru_confound": "C3b", "single_stageC3c_capacity_control": "C3c"}


def registrations(P):
    R = pd.read_csv(ROOT / "preregistrations" / "registry.csv")
    out = []
    for _, r in R.iterrows():
        st = STAGE_OF.get(r["id"])
        if st:
            sub = P[P.stage == st]
            what, outcome = f"stage {st}", f"{int(sub.registered_pass.sum())} of {len(sub)} primary passed"
        elif r["id"] == "single_revision_v1_3" and (LOGS / "revision_v13_tests.csv").exists():
            X = pd.read_csv(LOGS / "revision_v13_tests.csv")
            pr = X[X.family == "primary"]
            what = "revision experiments X1-X3"
            outcome = (f"{int((pr.verdict == 'PASS').sum())} of {len(pr)} primary passed (two-sided); "
                       f"{int((X[X.family == 'secondary'].verdict == 'PASS').sum())} of "
                       f"{int((X.family == 'secondary').sum())} secondary")
        else:
            what, outcome = NOTE.get(r["id"], ("", "see decision log"))
        out.append(dict(id=r["id"], registered=r["registered_utc"][:10], sha12=r["sha256"][:12], content=what,
                        outcome=outcome, in_paper="no (companion paper)" if outcome == "companion paper" else "yes"))
    D = pd.DataFrame(out)
    D.to_csv(OUT / "registrations.csv", index=False)
    return D


# ------------------------------------------------------------------ (d, e)
def offsets_and_curves():
    rows = []
    for k in range(8):
        p = json.loads((LOGS / "stageB10_opt" / f"bias_r{k}.json").read_text())["params"]
        rows.append(dict(base=k, o_alpha=p["b_a"], o_eps=p["b_e"], o_s=p["b_s"], eps0=p.get("eps0"),
                         alpha0=p.get("alpha0"), s0=p.get("s0")))
    pd.DataFrame(rows).to_csv(OUT / "b10_bias_offsets.csv", index=False)
    cur = []
    for d in ("stageB10_opt", "stageC3_opt", "stageC3b_opt", "stageC3c_opt"):
        for f in sorted((LOGS / d).glob("*_r[0-9].json")):
            j = json.loads(f.read_text())
            h = j.get("history")
            if not h or len(h) < 20:
                continue
            cur.append(dict(stage=d.replace("_opt", ""), arm=j.get("arm", f.stem.rsplit("_r", 1)[0]), run=j.get("run"),
                            g0=h[0], g10=h[10], g19=h[19], gain_10_19=h[19] - h[10],
                            slope_last5=np.polyfit(np.arange(5), h[15:20], 1)[0]))
    C = pd.DataFrame(cur)
    C.to_csv(OUT / "training_curves.csv", index=False)
    return C.groupby(["stage", "arm"])[["g0", "g10", "g19", "gain_10_19", "slope_last5"]].mean()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    A = tintervals()
    print(A[["stage", "pred", "family", "comparison", "diff", "ci3_lo", "ci3_hi", "t_lo", "t_hi", "p_two_sided",
             "p_two_sided_holm", "verdict_registered"]].round(4).to_string())
    P = primaries()
    print(P.round(5).to_string())
    print("primary predictions:", len(P), "registered pass:", int(P.registered_pass.sum()),
          "BH reject:", int(P.bh_reject.sum()), "disagree:", int((~P.agree).sum()))
    print(registrations(P).to_string())
    print(offsets_and_curves().round(4).to_string())
    print(pd.read_csv(OUT / "b10_bias_offsets.csv").round(3).to_string())
