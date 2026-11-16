"""Check: the adaptive simulator's 'fixed' controller is statistically equivalent to popsim
(stage A) in the switching world (same knobs, same schedules and layouts, independent
random streams). Pre-specified: TOST margin 0.3 SD on score and survival."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.analysis.stats import compare
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank, param_arrays, simulate

bank = LayoutBank.build(range(300000, 301000))
knobs = (0.3, 0.1, 2.0)
ref_s, ref_v, fa_s, fa_v = [], [], [], []
scheds = [R.schedule(np.random.default_rng(5000 + i)) for i in range(48)]
for i, sch in enumerate(scheds):
    lay = (np.arange(8) * 41 + i * 3) % 1000
    o = simulate("needs", param_arrays(8, alpha=knobs[0], epsilon=knobs[1], safety_mult=knobs[2]), R.eco("safe", 8),
                 bank, lay, R.T_LIFE, seed=i, phases=R.phases(sch, 8), grace=100, injury_grace=1, social_need=False)
    ref_s += list(o["score"]); ref_v += list(o["survival"])
M = A.regime_matrix([s for s in scheds for _ in range(8)], R.T_LIFE)
lay = np.concatenate([(np.arange(8) * 41 + i * 3) % 1000 for i in range(48)])
o = A.simulate("fixed", A.params(len(lay), alpha0=knobs[0], eps0=knobs[1], s0=knobs[2]), M, bank, lay, seed=77)
rows = []
for k, x, y in (("score", o["score"], ref_s), ("survival", o["survival"], ref_v)):
    c = compare(np.array(x), np.array(y), margin_sd=0.3)
    rows.append(dict(measure=k, adaptive=c["mean_x"], popsim=c["mean_y"], g=c["g"], p_diff=c["p"], p_equiv=c["p_equiv"]))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "logs" / "single" / "validate_adaptive.csv", index=False)
print(df.round(4).to_string())
