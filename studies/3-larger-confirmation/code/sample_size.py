"""Sample size of Study 3, computed from the saved Phase 2 variance before any Study 3 run.

The primary endpoint is the per-family difference D (independent minus joint history) in unchanged-skill error, one
per architecture. Each architecture gets a two-sided Student t interval at 97.5%, so the two intervals together have
at least 95% simultaneous coverage (Bonferroni). Equivalence at margin m = 0.05 is established when the interval
lies inside (-m, +m). The number of task families n is the smallest for which, if the true difference is zero and the
family differences are normal with the larger Phase 2 standard deviation, the interval falls inside (-m, +m) with
probability at least 0.8. The probability is computed exactly by integrating over the distribution of the sample
standard deviation:

    P(n) = integral over x ~ chi2(n-1) of  P(|mean| < m - h(x)),  h(x) = q * sigma * sqrt(x / (n-1)) / sqrt(n),

with q the 0.9875 quantile of Student t with n-1 degrees of freedom and mean ~ Normal(0, sigma^2 / n).

Reads ../../2-controlled-histories/analysis/confirmation/summary.json; writes ../protocol/sample-size.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import integrate, stats

STUDY = Path(__file__).resolve().parents[1]
PHASE2 = STUDY.parent / "2-controlled-histories/analysis/confirmation/summary.json"
MARGIN, POWER, INTERVALS, ALPHA = 0.05, 0.8, 2, 0.05


def quantile(n: int) -> float:
    return float(stats.t.ppf(1 - ALPHA / (2 * INTERVALS), n - 1))


def probability_inside(n: int, sigma: float, margin: float = MARGIN) -> float:
    q, df, se = quantile(n), n - 1, sigma / math.sqrt(n)
    x_max = df * (margin * math.sqrt(n) / (q * sigma)) ** 2

    def integrand(x):
        h = q * sigma * math.sqrt(x / df) / math.sqrt(n)
        return stats.chi2.pdf(x, df) * (2 * stats.norm.cdf((margin - h) / se) - 1)

    return float(integrate.quad(integrand, 0, x_max, limit=200)[0])


def families_needed(sigma: float) -> int:
    lo, hi = 4, 8
    while probability_inside(hi, sigma) < POWER:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if probability_inside(mid, sigma) < POWER else (lo, mid)
    return hi


def main() -> None:
    summary = json.loads(PHASE2.read_text())
    sd = {a: summary["primary"][a]["effects"]["stable"]["sd"] for a in ("linear", "leaky")}
    sigma = max(sd.values())
    n = families_needed(sigma)
    out = dict(
        source="studies/2-controlled-histories/analysis/confirmation/summary.json (Phase 2 confirmation, 64 families)",
        phase2_family_sd=sd, sigma_used=sigma, margin=MARGIN, power=POWER, primary_intervals=INTERVALS,
        interval_confidence=1 - ALPHA / INTERVALS, familywise_coverage=1 - ALPHA, families=n,
        t_quantile_at_n=quantile(n), expected_half_width_at_n=quantile(n) * sigma / math.sqrt(n),
        probability_inside_at_n={a: probability_inside(n, v) for a, v in sd.items()},
        probability_inside_at_n_minus_1=probability_inside(n - 1, sigma),
        probability_both_inside_if_independent=math.prod(probability_inside(n, v) for v in sd.values()))
    path = STUDY / "protocol/sample-size.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
