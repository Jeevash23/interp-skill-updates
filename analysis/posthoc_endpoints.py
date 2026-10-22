"""Post hoc (9 October 2026): how informative were the three primary endpoints, and do the conclusions change if
only the unchanged-skill endpoint is corrected for (two intervals instead of six)?

Reads the saved confirmation analysis (studies/2-controlled-histories/analysis/confirmation); writes
analysis/results/posthoc-endpoints.json.

1. Unseen-combination error versus unchanged-skill error under unrestricted updates: after an update that learns the
   changed function, an additive network's error on a withheld pair (a, b) is the drift of the unchanged function at
   its argument, and each argument occurs twice among the 16 withheld pairs, so the unseen-combination error
   (normalised by sqrt 2) equals the unchanged-skill error divided by sqrt 2 up to the changed-skill residual.
2. The changed-skill error sits at its floor (about 5e-4) in both histories.
3. Two-way corrected (97.5%) intervals for the unchanged-skill endpoint in the two architectures, with the frozen
   decision rule applied to them.
"""
from __future__ import annotations

import numpy as np

from common import ARCHS, CONF, interval, read, write

MARGIN = 0.05


def decision(low: float, high: float) -> str:
    """The frozen preservation rule (studies/2-controlled-histories/code/analyze.py), unchanged-skill endpoint only."""
    if high < -MARGIN:
        return "practically_useful_independent_history_benefit"
    if low > -MARGIN and high < MARGIN:
        return "preservation_equivalent_within_0.05"
    if low > MARGIN:
        return "independent_history_practically_worse"
    if low > -MARGIN:
        return "practical_independent_history_benefit_ruled_out"
    return "inconclusive"


def main() -> None:
    summary = read(CONF / "summary.json")
    records = read(CONF / "records.json")
    out = dict(post_hoc=True, created="2026-10-09", source="studies/2-controlled-histories/analysis/confirmation",
               margin=MARGIN, architectures={})
    for arch in ARCHS:
        eff = summary["primary"][arch]["effects"]
        stable = np.array(list(eff["stable"]["family_differences"].values()))
        transfer = np.array(list(eff["transfer"]["family_differences"].values()))
        six = interval(stable, 1 - 0.05 / 6)
        assert abs(six["low"] - eff["stable"]["low"]) < 1e-12 and abs(six["high"] - eff["stable"]["high"]) < 1e-12
        two = interval(stable, 1 - 0.05 / 2)
        rec = [r for r in records if r["architecture"] == arch and r["arm"] == "unrestricted"]
        st = np.array([r["stable"] for r in rec])
        tr = np.array([r["transfer"] for r in rec])
        ch = np.array([r["changed"] for r in rec])
        slope, intercept = np.polyfit(stable, transfer, 1)
        changed = eff["changed"]
        out["architectures"][arch] = dict(
            six_way=dict(low=six["low"], high=six["high"], confidence=six["confidence"],
                         decision=summary["primary"][arch]["decision"]),
            two_way=dict(low=two["low"], high=two["high"], confidence=two["confidence"], mean=two["mean"],
                         decision=decision(two["low"], two["high"])),
            unseen_vs_unchanged=dict(trajectories=len(rec), correlation=float(np.corrcoef(st, tr)[0, 1]),
                                     max_abs_deviation_from_unchanged_over_sqrt2=float(np.max(np.abs(tr - st / np.sqrt(2)))),
                                     mean_ratio=float(np.mean(tr / (st / np.sqrt(2)))),
                                     family_difference_correlation=float(np.corrcoef(stable, transfer)[0, 1]),
                                     family_difference_slope=float(slope)),
            changed_floor=dict(mean_error=float(ch.mean()), max_error=float(ch.max()),
                               difference_mean=changed["mean"], difference_low=changed["low"],
                               difference_high=changed["high"],
                               interval_width_over_noninferiority_margin=(changed["high"] - changed["low"]) / 0.02))
    out["conclusions_unchanged"] = all(
        (out["architectures"][a]["two_way"]["decision"] == out["architectures"][a]["six_way"]["decision"]) for a in ARCHS)
    path = write("posthoc-endpoints.json", out)
    for arch in ARCHS:
        a = out["architectures"][arch]
        print(f"{arch:6s} two-way [{a['two_way']['low']:+.4f}, {a['two_way']['high']:+.4f}] {a['two_way']['decision']}; "
              f"unseen = unchanged/sqrt2: r={a['unseen_vs_unchanged']['correlation']:.5f}, "
              f"max dev {a['unseen_vs_unchanged']['max_abs_deviation_from_unchanged_over_sqrt2']:.4f}")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
