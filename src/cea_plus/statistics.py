from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SignificanceResult:
    mean_delta: float
    ttest_p: float
    wilcoxon_p: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


def paired_significance(baseline: np.ndarray, proposed: np.ndarray, seed: int = 0, bootstraps: int = 3000) -> SignificanceResult:
    baseline = np.asarray(baseline, dtype=np.float64)
    proposed = np.asarray(proposed, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline and proposed must have same shape")
    if baseline.size < 2:
        raise ValueError("at least two paired samples are required")

    delta = proposed - baseline
    ttest_p = float(stats.ttest_rel(proposed, baseline).pvalue)
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(bootstraps, delta.size))
    means = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return SignificanceResult(
        mean_delta=float(delta.mean()),
        ttest_p=ttest_p,
        wilcoxon_p=wilcoxon_p,
        bootstrap_ci_low=float(low),
        bootstrap_ci_high=float(high),
    )
