"""Point-and-interval marker used by the figure scripts in place of bars on a truncated axis (v1.4).
point_ci(ax, x, vals, color): mean of vals (runs or bases) as a filled marker with its 95% t interval
(df = n - 1) as a vertical line. The interval treats the plotted units as independent; for runs on shared bases it is
descriptive only. The individual units are drawn by the calling script as small dots."""
import numpy as np
from scipy import stats


def point_ci(ax, x, vals, color, label=None, size=7, lw=1.6):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    ax.vlines(x, m - h, m + h, color=color, lw=lw, zorder=4)
    ax.plot([x], [m], marker="o", ms=size, color=color, mec="black", mew=0.6, zorder=5, ls="none", label=label)
    return m, h
