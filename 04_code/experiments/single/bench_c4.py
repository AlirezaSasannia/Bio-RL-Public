"""C4 timing benchmark in E2 (development seeds: schedules 98600+, E2 layouts 3992000+, simulator 98700; no result is
used). Run it on the target machine before costing C4 or choosing C4_BPC.
Usage: python bench_c4.py <T steps> <agents per call, comma list> [arms]
  e.g. python bench_c4.py 300 2304,4608 dqnv,gru,ff_big   (ms per 9000-step agent-life is printed)
Agents per call in C4 = C4_BPC x 12 x 48 in training (BPC 4 -> 2304) and C4_BPC x 512 in the test."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R
from bio_rl.fast.popsim import LayoutBank
from bio_rl.torchsim import world as W

T = int(sys.argv[1])
Bs = [int(b) for b in sys.argv[2].split(",")]
arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dqnv", "gru", "ff_big"]
bank = LayoutBank.build_e2(range(3_992_000, 3_992_400))
base = json.loads((ROOT / "results" / "logs" / "single" / "stageC3_base" / "dqnv_r0.json").read_text())["params"]
DQN = dict(H=32, M=1000, m=8, C=1, needs_in=1.0, view=3)
EXTRA = {"dqnv": ("dqnv", {}, {}), "aug": ("dqnv", {f"xs_{i}": 1.0 for i in range(5)}, {}),
         "film": ("dqnv", {f"fs_{i}": 1.0 for i in range(5)}, {}), "ra": ("dqnv", {}, {"ra": True}),
         "ff_big": ("dqnv", {}, {"ra": True, "H": 105}), "gru": ("gru", {}, {}),
         "gru_aug": ("gru", {f"xs_{i}": 1.0 for i in range(5)}, {})}
for B in Bs:
    sch = [R.schedule_e2(np.random.default_rng(98600 + i)) for i in range(B)]
    M = A.regime_matrix(sch, R.T_LIFE_E2)[:T]
    lay = np.arange(B) % len(bank)
    for a in arms:
        learner, kw, cfg = EXTRA[a]
        P = W.params(B, **{**base, **kw})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        o = W.simulate(learner, P, M, bank, lay, seed=98_700, dqn={**DQN, **cfg}, juvenile=min(R.JUVENILE_E2, T),
                       decay=0.0026)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"{a:8s} B={B:5d} T={T}: {dt:7.1f}s -> {dt * R.T_LIFE_E2 / T / B * 1e3:6.1f} ms per agent-life "
              f"(9000 steps); peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
