"""Post hoc (9 October 2026): what the frozen sample-size rule implied, and how many task families a future
confirmation would need to establish equivalence. Planning arithmetic for future work, not a result.

1. The frozen rule (studies/2-controlled-histories/protocol/CONFIRMATION.md): n = ceil((2.3 s / 0.025)^2), with s
   the larger development standard deviation of the family-level history differences, bounded to 32-64 families.
   The primary intervals instead use the Student t quantile for six simultaneous 95% intervals with 63 degrees of
   freedom (about 2.72).
2. The expected half-width of the primary interval at the 64-family cap, under the development and the confirmation
   standard deviations.
3. Families needed for the corrected interval to fall inside +-m with probability 0.8 when the true difference is
   zero (m = 0.05 and 0.025), using the larger confirmation standard deviation. The probability is computed exactly
   for normally distributed family differences by integrating over the sample standard deviation. Reported for a
   two-way correction (only the unchanged-skill endpoint, both architectures; the post hoc choice argued in
   analysis/posthoc_endpoints.py) and for the frozen six-way correction.

Writes analysis/results/posthoc-sample-size.json.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, stats

from common import ARCHS, CONF, P2, read, write

POWER = 0.8


def quantile(intervals: int, n: int) -> float:
    return float(stats.t.ppf(1 - 0.05 / (2 * intervals), n - 1))


def probability_inside(n: int, sigma: float, margin: float, intervals: int) -> float:
    """P(|mean| + q s / sqrt(n) < margin) for n normal differences with mean 0 and SD sigma."""
    q = quantile(intervals, n)
    se = sigma / math.sqrt(n)
    df = n - 1
    x_max = df * (margin * math.sqrt(n) / (q * sigma)) ** 2           # chi-square values with half-width < margin
    if x_max <= 0:
        return 0.0
    def integrand(x):
        h = q * sigma * math.sqrt(x / df) / math.sqrt(n)
        return stats.chi2.pdf(x, df) * (2 * stats.norm.cdf((margin - h) / se) - 1)
    value, _ = integrate.quad(integrand, 0, x_max, limit=200)
    return float(value)


def families_needed(sigma: float, margin: float, intervals: int) -> int:
    lo, hi = 4, 8
    while probability_inside(hi, sigma, margin, intervals) < POWER:
        hi *= 2
    while hi - lo > 1:                                                     # probability increases with n
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if probability_inside(mid, sigma, margin, intervals) < POWER else (lo, mid)
    return hi


def half_width_families(sigma: float, margin: float, intervals: int) -> int:
    n = 4
    while quantile(intervals, n) * sigma / math.sqrt(n) > margin:
        n += 1
    return n


def main() -> None:
    dev_sd = {"linear": read(P2 / "analysis/development/summary.json")["primary"]["linear"]["effects"]["stable"]["sd"],
              "leaky": read(P2 / "analysis/amendment/summary.json")["primary"]["leaky"]["effects"]["stable"]["sd"]}
    config = read(P2 / "protocol/confirmation.json")
    s_dev = max(dev_sd.values())
    rule_raw = (2.3 * s_dev / config["confirmation_target_half_width"]) ** 2
    rule = math.ceil(rule_raw)
    n = len(config["task_seeds"])
    q6 = quantile(6, n)
    summary = read(CONF / "summary.json")
    conf_sd = {a: summary["primary"][a]["effects"]["stable"]["sd"] for a in ARCHS}
    realised = {a: (summary["primary"][a]["effects"]["stable"]["high"] - summary["primary"][a]["effects"]["stable"]["low"]) / 2
                for a in ARCHS}
    sigma = max(conf_sd.values())
    needed = {}
    for margin in (0.05, 0.025):
        needed[str(margin)] = {
            f"{k}_way": dict(power_0_8=families_needed(sigma, margin, k),
                             expected_half_width_equals_margin=half_width_families(sigma, margin, k))
            for k in (2, 6)}
    out = dict(post_hoc=True, created="2026-10-09", power=POWER,
               development_sd=dev_sd, rule_sd=s_dev, rule_multiplier=2.3,
               rule_target_half_width=config["confirmation_target_half_width"],
               rule_families_uncapped=rule, rule_families_uncapped_exact=rule_raw,
               rule_bounds=[config["confirmation_minimum_families"], config["confirmation_maximum_families"]],
               families_used=n, t_quantile_six_way=q6, t_quantile_two_way=quantile(2, n),
               rule_families_with_t_quantile=math.ceil((q6 * s_dev / config["confirmation_target_half_width"]) ** 2),
               expected_half_width_at_cap_development_sd=q6 * s_dev / math.sqrt(n),
               expected_half_width_at_cap_confirmation_sd=q6 * sigma / math.sqrt(n),
               confirmation_sd=conf_sd, realised_half_width=realised, sigma_used=sigma,
               probability_inside_0_05_at_64_two_way=probability_inside(n, sigma, 0.05, 2),
               probability_inside_0_05_at_64_six_way=probability_inside(n, sigma, 0.05, 6),
               probability_inside_two_way={f"{m}@{k}": probability_inside(k, sigma, m, 2)
                                           for m, k in ((0.05, 110), (0.05, 150), (0.025, 350), (0.025, 600))},
               families_needed=needed)
    path = write("posthoc-sample-size.json", out)
    print(f"frozen rule: (2.3*{s_dev:.4f}/0.025)^2 = {rule_raw:.1f} -> {rule} families, capped at {n}; t quantile {q6:.3f}")
    print(f"expected half-width at {n}: {out['expected_half_width_at_cap_development_sd']:.3f} (development SD), "
          f"{out['expected_half_width_at_cap_confirmation_sd']:.3f} (confirmation SD)")
    for m, v in needed.items():
        print(f"margin {m}: {v}")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
