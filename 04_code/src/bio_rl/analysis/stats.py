"""One statistical toolkit for every comparison, so all claims are made the same way.

* effect size: Hedges' g (bias-corrected standardized mean difference)
* uncertainty: bootstrap 95% CI of the mean difference and of g
* difference test: Welch t-test
* "no effect / no harm" claims: TOST equivalence test with a margin in SD units
* multiple comparisons: Holm-Bonferroni
* power: seeds per group for a two-sided Welch test at a smallest effect of interest
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def hedges_g(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    d = (x.mean() - y.mean()) / sp
    return float(d * (1 - 3 / (4 * (nx + ny) - 9)))


def bootstrap_ci(x, y, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of mean(x) - mean(y)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    bx = rng.choice(x, (n_boot, len(x))).mean(1)
    by = rng.choice(y, (n_boot, len(y))).mean(1)
    lo, hi = np.percentile(bx - by, [2.5, 97.5])
    return float(lo), float(hi)


def welch(x, y) -> float:
    return float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def tost(x, y, margin_sd: float = 0.5) -> float:
    """Equivalence p-value: is |mean(x)-mean(y)| < margin_sd * pooled SD? Small p = equivalent."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    m = margin_sd * sp
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    if se == 0:
        return 0.0 if abs(diff) < m else 1.0
    df = (se ** 4) / ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
    p_lower = 1 - stats.t.cdf((diff + m) / se, df)   # H0: diff <= -m
    p_upper = stats.t.cdf((diff - m) / se, df)       # H0: diff >= +m
    return float(max(p_lower, p_upper))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (keys preserved)."""
    pvals = {k: (1.0 if (p is None or p != p) else p) for k, p in pvals.items()}   # NaN -> 1
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def seeds_for_power(g: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Seeds per group for a two-sided test to detect standardized effect g."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * (za + zb) ** 2 / g ** 2)) + 1


def compare(x, y, margin_sd: float = 0.5) -> dict:
    lo, hi = bootstrap_ci(x, y)
    return dict(mean_x=float(np.mean(x)), mean_y=float(np.mean(y)), diff=float(np.mean(x) - np.mean(y)),
                ci_lo=lo, ci_hi=hi, g=hedges_g(x, y), p=welch(x, y), p_equiv=tost(x, y, margin_sd),
                n_x=len(x), n_y=len(y))
