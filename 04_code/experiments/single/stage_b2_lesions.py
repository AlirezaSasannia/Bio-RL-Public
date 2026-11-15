"""Secondary (exploratory): which hormone carries the add-on gain? Lesion each hormone's gains in the
optimised vdbe+hormones and fixed+hormones agents, on the stage-B held-out schedules."""
import json
import sys
from multiprocessing import Pool

from _pool import NWORK
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b_test as T
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R

OPT = ROOT / "results" / "logs" / "single" / "stageB2_opt"
LES = {"none": (), "NE": ("ne",), "cortisol": ("mr", "gr"), "ghrelin": ("gh",)}


def job(a):
    base, les, r = a
    d = json.loads((OPT / f"{base}_hormones_r{r}.json").read_text())
    p = dict(d["params"])
    for h in LES[les]:
        for k in A.KNOBS:
            p[f"g_{h}_{k}"] = 0.0
    return T.run(d["controller"], p, r, tag=f"{base}-{les}")


if __name__ == "__main__":
    with Pool(NWORK) as pool:
        D = pd.concat(pool.map(job, [(b, l, r) for b in ("fixed", "vdbe") for l in LES for r in range(4)]))
    rows = []
    for b in ("fixed", "vdbe"):
        full = D[D.variant == f"{b}-none"]
        for l in ("NE", "cortisol", "ghrelin"):
            x = D[D.variant == f"{b}-{l}"]
            d = (full.groupby("schedule").score.mean() - x.groupby("schedule").score.mean()).values
            rows.append(dict(base=b, lesion=l, loss=d.mean(), p=float(stats.wilcoxon(d).pvalue),
                             **{f"loss_{rn}": full[f"score_{rn}"].mean() - x[f"score_{rn}"].mean() for rn in R.NAMES}))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "logs" / "single" / "stageB2_lesions.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())
