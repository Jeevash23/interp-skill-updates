"""Post hoc (9 October 2026): the history effect split by which function the update changes.

The two schedules are not symmetric in f and g. Counting transitions in the saved schedules
(studies/2-controlled-histories/data/confirmation.json): the independent history changes f alone more often than
g alone, while the joint history changes each alone equally rarely. This script reports those counts and the
independent-minus-joint difference in unchanged-skill error separately for f-updates and g-updates, with
per-direction 95% family intervals and the f-minus-g interaction. The frozen display code
(studies/2-controlled-histories/code/plot_results.py) reported the two direction means without intervals.

Writes analysis/results/posthoc-direction.json.
"""
from __future__ import annotations

import numpy as np

from common import ARCHS, CONF, P2, family_differences, interval, read, write


def transition_counts(order):
    counts = {"f_only": 0, "g_only": 0, "both": 0}
    for old, new in zip(order, order[1:]):
        f, g = old[0] != new[0], old[1] != new[1]
        counts["both" if f and g else "f_only" if f else "g_only"] += 1
    return counts


def main() -> None:
    schedules = read(P2 / "data/confirmation.json")["schedules"]
    counts = {h: transition_counts(order) for h, order in schedules.items()}
    suffix = 4                                   # the last four transitions (five environments) are shared
    counts_before_suffix = {h: transition_counts(order[:-suffix]) for h, order in schedules.items()}
    records = read(CONF / "records.json")
    out = dict(post_hoc=True, created="2026-10-09", transition_counts=counts,
               transition_counts_before_shared_suffix=counts_before_suffix, effects={})
    for arch in ARCHS:
        out["effects"][arch] = {}
        for arm in ("unrestricted", "normalized"):
            by = {}
            for branch in ("f", "g"):
                d = family_differences(records, arch, arm, "stable", lambda r, b=branch: r["branch"] == b)
                by[branch] = interval(list(d.values()))
            f = family_differences(records, arch, arm, "stable", lambda r: r["branch"] == "f")
            g = family_differences(records, arch, arm, "stable", lambda r: r["branch"] == "g")
            by["f_minus_g"] = interval([f[k] - g[k] for k in f])
            out["effects"][arch][arm] = by
    sens = read(CONF / "descriptive-sensitivity.json")["results"]
    for arch in ARCHS:
        for branch in ("f", "g"):
            frozen = sens[arch][f"direction_{branch}"]["mean"]
            assert abs(frozen - out["effects"][arch]["unrestricted"][branch]["mean"]) < 1e-12
    path = write("posthoc-direction.json", out)
    print("transitions", counts)
    for arch in ARCHS:
        e = out["effects"][arch]["unrestricted"]
        print(f"{arch:6s} f-updates {e['f']['mean']:+.4f} [{e['f']['low']:+.4f}, {e['f']['high']:+.4f}]  "
              f"g-updates {e['g']['mean']:+.4f} [{e['g']['low']:+.4f}, {e['g']['high']:+.4f}]  "
              f"f-g {e['f_minus_g']['mean']:+.4f} [{e['f_minus_g']['low']:+.4f}, {e['f_minus_g']['high']:+.4f}]")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
