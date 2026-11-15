"""Stage B, step 2: pre-registered tests H1-H4 and secondary analyses on held-out schedules."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.analysis.stats import holm
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank

LOGS = ROOT / "results" / "logs" / "single"
OPT = LOGS / "stageB_opt"
CTRLS = ["fixed", "reactive", "fuzzy", "plastic", "vdbe", "changept", "pearcehall"]
NS, NA_, RESTARTS = 64, 8, 4
BANK = LayoutBank.build(range(700000, 701000))
SCHED = [R.schedule(np.random.default_rng(2000 + i)) for i in range(NS)]
M = A.regime_matrix([s for s in SCHED for _ in range(NA_)], R.T_LIFE)
LAY = np.concatenate([(np.arange(NA_) * 37 + i * 11) % 1000 for i in range(NS)])
SCHED_ID = np.repeat(np.arange(NS), NA_)
ch = json.load(open(LOGS / "stageA_choices.json"))
ORACLE = np.array([ch["oracle"][n] for n in R.NAMES], float)
REC = ("wellbeing",)


def load(c, r):
    return json.loads((OPT / f"{c}_r{r}.json").read_text())["params"]


def run(ctrl, prm, r, tag=None):
    o = A.simulate(ctrl, A.params(len(LAY), **prm), M, BANK, LAY, seed=31_000 + r, oracle_table=ORACLE, record=REC)
    wb = o["wellbeing"]
    df = pd.DataFrame(dict(schedule=SCHED_ID, agent=np.tile(np.arange(NA_), NS), score=o["score"],
                           survival=o["survival"]))
    df["controller"], df["restart"], df["variant"] = ctrl, r, tag or ctrl
    for j, rn in enumerate(R.NAMES):            # score accrued inside each regime
        m = (M == j)
        df[f"score_{rn}"] = np.where(m.any(0), (wb * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
    # adaptation: mean wellbeing in the first 300 steps after each switch (after the juvenile period)
    post = np.zeros(len(LAY)); cnt = np.zeros(len(LAY))
    for b, sch in enumerate([s for s in SCHED for _ in range(NA_)]):
        for st, _ in sch[1:]:
            post[b] += wb[st:st + 300, b].mean(); cnt[b] += 1
    df["post_switch"] = post / np.maximum(cnt, 1)
    return df


def sched_means(D, col="score"):
    return D.groupby("schedule")[col].mean()


def wilcox_greater(x, y):
    d = (x - y).values
    return float(stats.wilcoxon(d, alternative="greater").pvalue), float(d.mean())


if __name__ == "__main__":
    frames = [run("oracle", {}, 0)]
    for c in CTRLS:
        for r in range(RESTARTS):
            frames.append(run(c, load(c, r), r))
    D = pd.concat(frames)
    D.to_csv(LOGS / "stageB_test_raw.csv", index=False)
    sm = {c: sched_means(D[D.variant == c]) for c in CTRLS + ["oracle"]}
    head = sm["oracle"] - sm["fixed"]
    rows = []
    for c in CTRLS + ["oracle"]:
        d = (sm[c] - sm["fixed"]).values
        boot = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
        rows.append(dict(controller=c, score=sm[c].mean(), survival=D[D.variant == c].survival.mean(),
                         gain_vs_fixed=d.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                         headroom_captured=d.mean() / head.mean(),
                         post_switch=D[D.variant == c].post_switch.mean(),
                         **{f"captured_{rn}": (D[D.variant == c][f"score_{rn}"].mean() - D[D.variant == "fixed"][f"score_{rn}"].mean())
                            / (D[D.variant == "oracle"][f"score_{rn}"].mean() - D[D.variant == "fixed"][f"score_{rn}"].mean())
                            for rn in R.NAMES}))
    S = pd.DataFrame(rows)
    S.to_csv(LOGS / "stageB_summary.csv", index=False)

    tests = []
    for o in ("vdbe", "changept", "pearcehall"):
        p, d = wilcox_greater(sm["plastic"], sm[o])
        tests.append(dict(test="H1", comparison=f"plastic > {o}", diff=d, p=p))
    for t, a, b in (("H2", "plastic", "fuzzy"), ("H3", "fuzzy", "reactive"), ("H4", "plastic", "fixed")):
        p, d = wilcox_greater(sm[a], sm[b])
        tests.append(dict(test=t, comparison=f"{a} > {b}", diff=d, p=p))
    Tt = pd.DataFrame(tests)
    Tt["p_holm"] = np.nan
    for fam in (["H1"], ["H2", "H3", "H4"]):
        idx = Tt.index[Tt.test.isin(fam)]
        adj = holm({i: Tt.loc[i, "p"] for i in idx})
        for i in idx:
            Tt.loc[i, "p_holm"] = adj[i]
    Tt["verdict"] = np.where(Tt.p_holm < 0.05, "PASS", "FAIL")
    Tt.to_csv(LOGS / "stageB_tests.csv", index=False)

    # secondary: hormone lesions in fuzzy and plastic agents
    L = []
    for c in ("fuzzy", "plastic"):
        base = D[D.variant == c]
        for h in A.HORMONES if False else ("ne", ("mr", "gr"), "gh"):
            hs = h if isinstance(h, tuple) else (h,)
            name = "cortisol" if isinstance(h, tuple) else {"ne": "NE", "gh": "ghrelin"}[h]
            fr = []
            for r in range(RESTARTS):
                p = dict(load(c, r))
                for hh in hs:
                    for k in A.KNOBS:
                        p[f"g_{hh}_{k}"] = 0.0
                fr.append(run(c, p, r, tag=f"{c}-no{name}"))
            Ld = pd.concat(fr)
            row = dict(controller=c, lesion=name, diff=(sched_means(base) - sched_means(Ld)).mean(),
                       p=float(stats.wilcoxon((sched_means(base) - sched_means(Ld)).values).pvalue))
            for rn in R.NAMES:
                row[f"diff_{rn}"] = base[f"score_{rn}"].mean() - Ld[f"score_{rn}"].mean()
            L.append(row)
    pd.DataFrame(L).to_csv(LOGS / "stageB_lesions.csv", index=False)

    pd.set_option("display.width", 250)
    print(S.round(4).to_string())
    print(Tt.round(4).to_string())
    print(pd.DataFrame(L).round(4).to_string())
