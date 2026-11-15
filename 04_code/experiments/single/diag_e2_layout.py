"""Layout geometry of the original world vs E2 (development layout banks of diag_e2.py; exploratory): shelter-food
distance, detour forced by threat cells (BFS avoiding threats minus plain distance) and the share of shelter-food
pairs whose every shortest path crosses a threat cell. Run: python diag_e2_layout.py"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bio_rl.fast.popsim import LayoutBank


def bfs(size, src, block):
    d = np.full(size * size, -1)
    d[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        r, c = divmod(u, size)
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            v = rr * size + cc
            if 0 <= rr < size and 0 <= cc < size and d[v] < 0 and not block[v]:
                d[v] = d[u] + 1
                q.append(v)
    return d


rows = []
for w, bank in (("orig", LayoutBank.build(range(3_980_000, 3_982_000))),
                ("e2", LayoutBank.build_e2(range(3_970_000, 3_972_000)))):
    size = bank.SIZE
    for i in range(len(bank)):
        sh, th = bank.shelter[i], bank.threat[i]
        free = bfs(size, sh, np.zeros(size * size, bool))
        avoid = bfs(size, sh, th)
        for f in bank.food[i]:
            rows.append(dict(world=w, dist=free[f], detour=(avoid[f] - free[f]) if avoid[f] >= 0 else np.nan,
                             forced=avoid[f] != free[f], threat_share=th.mean()))
D = pd.DataFrame(rows)
print(D.groupby("world").agg(dist=("dist", "mean"), detour=("detour", "mean"), forced=("forced", "mean"),
                             threat_share=("threat_share", "mean")).round(3).to_string())
