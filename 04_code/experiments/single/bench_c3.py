"""C3 timing benchmark on development seeds (schedules 97000+, layouts 1960000+, simulator 97500); no result is used.
Usage: python bench_c3.py <T steps> <agents per call, comma list> [arms]
  e.g. python bench_c3.py 300 2304,4608 dqnv,gru,metagrad   (ms per 6000-step agent-life is printed)"""
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
arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dqnv", "gru", "metagrad"]
bank = LayoutBank.build(range(1_960_000, 1_960_400))
base = json.loads((ROOT / "results" / "logs" / "single" / "stageC2_base" / "dqnv_r0.json").read_text())["params"]
DQN = dict(H=32, M=1000, m=8, C=1, needs_in=1.0)
EXTRA = {"dqnv": ("dqnv", {}, {}), "film": ("dqnv", {f"fs_{i}": 1.0 for i in range(5)}, {}),
         "aug": ("dqnv", {f"xs_{i}": 1.0 for i in range(5)}, {}),
         "generic": ("dqnv", {f"w_{i}_{k}": 0.3 for i in range(5) for k in "aes"}, {}),
         "gru": ("gru", {}, {}), "gru_aug": ("gru", {f"xs_{i}": 1.0 for i in range(5)}, {}),
         "metagrad": ("dqnv", {"mg_lr": 0.01}, {"meta": True})}
for B in Bs:
    sch = [R.schedule(np.random.default_rng(97000 + i)) for i in range(B)]
    M = A.regime_matrix(sch, R.T_LIFE)[:T]
    lay = np.arange(B) % len(bank)
    for a in arms:
        learner, kw, cfg = EXTRA[a]
        P = W.params(B, **{**base, **kw})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        o = W.simulate(learner, P, M, bank, lay, seed=97_500, dqn={**DQN, **cfg}, juvenile=min(500, T))
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"{a:9s} B={B:5d} T={T}: {dt:7.1f}s  -> {dt * 6000 / T / B * 1e3:6.2f} ms per agent-life "
              f"(6000 steps); peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB; score {o['score'].mean():.3f}",
              flush=True)
