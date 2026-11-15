"""Search spaces for stage B (unit cube -> named parameters; 'log' = log-uniform)."""
import numpy as np

from bio_rl.fast import adaptive as A
from bio_rl.fast import regimes as R

BASE = {"alpha0": (0.02, 0.9, "lin"), "eps0": (0.0, 0.5, "lin"), "s0": (0.25, 8.0, "log")}
GAINS = {f"g_{h}_{k}": (-3.0, 3.0, "lin") for h in A.HORMONES for k in A.KNOBS}
FUZZY = {**BASE, "c_thr": (0.005, 0.5, "log"), "c_k": (0.1, 0.9, "lin"), "c_sur": (0.005, 1.0, "log"),
         "c_intake": (0.002, 0.05, "log"), **GAINS}
SPACES = {
    "fixed": dict(BASE),
    "reactive": {**BASE, "c_to_safety": (0, 6, "lin"), "ne_to_alpha": (0, 4, "lin"), "ne_to_temp": (0, 6, "lin"),
                 "c_to_temp": (0, 4, "lin"), "kp_explore": (0, 4, "lin"), "kp_effort": (0, 4, "lin"),
                 "kappa_c": (0.005, 0.3, "log")},
    "fuzzy": FUZZY,
    "plastic": {**FUZZY, "eta": (0.0, 2.0, "lin"), "sigma": (0.0, 0.5, "lin")},
    "vdbe": {**BASE, "vdbe_sigma": (0.001, 1.0, "log")},
    "changept": {**BASE, "ph_lambda": (0.0005, 0.05, "log"), "ph_h": (0.05, 5.0, "log"), "boost_a": (0, 5, "lin"),
                 "boost_e": (0, 0.5, "lin"), "boost_tau": (10, 1000, "log")},
    "pearcehall": {**BASE, "ph_rate": (0.005, 0.3, "lin")},
}


def decode(ctrl, u):
    u = np.clip(np.asarray(u, float), 0, 1)
    out = {}
    for ui, (k, (lo, hi, sc)) in zip(u, SPACES[ctrl].items()):
        out[k] = float(lo * (hi / lo) ** ui) if sc == "log" else float(lo + ui * (hi - lo))
    return out


def evaluate(ctrl, cands, n_agents, bank, gen_seed):
    rng = np.random.default_rng(gen_seed)
    scheds = [R.schedule(rng) for _ in range(n_agents)]
    lay1 = rng.integers(len(bank), size=n_agents)
    m = len(cands)
    M = A.regime_matrix(scheds * m, R.T_LIFE)
    keys = set().union(*cands)
    P = A.params(m * n_agents, **{k: np.repeat([c[k] for c in cands], n_agents) for k in keys})
    out = A.simulate(ctrl, P, M, bank, np.tile(lay1, m), seed=gen_seed + 1)
    return out["score"].reshape(m, n_agents).mean(1)

HORMONE_ONLY = {k: v for k, v in FUZZY.items() if k not in BASE}     # 4 centres + 12 gains


def encode(space: dict, prm: dict) -> np.ndarray:
    """Inverse of decode for a given space."""
    u = []
    for k, (lo, hi, sc) in space.items():
        x = prm[k]
        u.append(np.log(x / lo) / np.log(hi / lo) if sc == "log" else (x - lo) / (hi - lo))
    return np.clip(np.array(u), 0, 1)


def decode_space(space: dict, u) -> dict:
    u = np.clip(np.asarray(u, float), 0, 1)
    return {k: float(lo * (hi / lo) ** ui) if sc == "log" else float(lo + ui * (hi - lo))
            for ui, (k, (lo, hi, sc)) in zip(u, space.items())}

GENERIC = {f"w_{i}_{k}": (-3.0, 3.0, "lin") for i in range(5) for k in ("a", "e", "s")}   # 15 parameters
