"""Equivalence (TOST) for the B3 placebo claims. Margin +/-0.033 = half of the B3 bio - control gain on the
fresh set (registered in SPEC_B5 before computing). Equivalence if the 90% hierarchical CI lies inside the margin."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_b3 as B3
from stage_b5 import _boot
M = 0.033
D = pd.read_csv(B3.LOGS / "stageB3_test_raw.csv")
rows = []
for kind in ("main", "short", "long", "lethal"):
    piv = D[D.set == kind].groupby(["arm", "run", "schedule"]).score.mean()
    for o in ("randrules", "nomem", "generic"):
        d = (piv["bio"] - piv[o]).unstack().values
        lo, hi = np.percentile(_boot(d), [5, 95])
        rows.append(dict(set=kind, comparison=f"bio - {o}", diff=d.mean(), ci90_lo=lo, ci90_hi=hi, margin=M,
                         equivalent=bool(lo > -M and hi < M)))
T = pd.DataFrame(rows); T.to_csv(B3.LOGS / "stageB3_tost.csv", index=False); print(T.round(4).to_string())
