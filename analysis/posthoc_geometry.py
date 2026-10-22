"""Post hoc (9 October 2026): do the two histories leave the networks in different places, and does simple weight
geometry predict how much an update damages the unchanged skill?

Reads the saved confirmation base models (runs/confirmation/<model>/base.npz and initial.npz) and the saved
per-update records; writes analysis/results/posthoc-geometry.json.

1. Paired distance: for each task family, initialization and architecture, the distance between the independent-
   and joint-history base networks (all 544 weights), divided by the joint network's weight norm, and divided by
   the mean distance each network travelled from their shared initial weights.
2. Geometry of each base network, with H_f and H_g the 8 x 32 hidden codes of the f-only and g-only inputs and w
   the readout: readout norm |w|; code overlap |H_g H_f^T| / (|H_g| |H_f|) (Frobenius norms); and the share of the
   unchanged function's code energy along the readout direction. Each is correlated (Pearson) with the
   unchanged-skill error after the unrestricted 200-step update, across the 512 networks x 2 update directions, and
   the paired (independent minus joint) differences are correlated across families. Two-sided p-values for the
   paired correlations (Student t with n - 2 degrees of freedom) are reported without correction for the six of them
   (3 measures x 2 architectures).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from common import ARCHS, CONF, P2, read, write

RUNS = P2 / "runs/confirmation"


def flat(z):
    return np.concatenate([z["0.weight"].ravel(), z["2.weight"].ravel()])


def codes(W, arch):
    act = (lambda x: x) if arch == "linear" else (lambda x: np.where(x > 0, x, 0.9 * x))
    return act(W[:, :8]).T, act(W[:, 8:]).T              # f-only and g-only hidden codes, 8 x 32


def main() -> None:
    manifest = read(RUNS / "manifest.json")
    records = read(CONF / "records.json")
    damage = {(r["task_seed"], r["model_seed"], r["architecture"], r["history"], r["branch"]): r["stable"]
              for r in records if r["arm"] == "unrestricted"}
    models = {}
    for job in manifest["jobs"]:
        key = (job["task_seed"], job["model_seed"], job["architecture"], job["history"])
        d = RUNS / "t{}-s{}-{}-{}".format(*key)
        with np.load(d / "base.npz") as b, np.load(d / "initial.npz") as i:
            W, w = b["0.weight"], b["2.weight"][0]
            models[key] = dict(theta=flat(b), init=flat(i), W=W, w=w)
    out = dict(post_hoc=True, created="2026-10-09", networks=len(models), architectures={})
    for arch in ARCHS:
        rel_norm, rel_travel = [], []
        rows = []
        for (task, seed, a, hist), m in models.items():
            if a != arch:
                continue
            if hist == "joint":
                o = models[(task, seed, a, "independent")]
                dist = np.linalg.norm(o["theta"] - m["theta"])
                rel_norm.append(dist / np.linalg.norm(m["theta"]))
                travel = 0.5 * (np.linalg.norm(o["theta"] - o["init"]) + np.linalg.norm(m["theta"] - m["init"]))
                rel_travel.append(dist / travel)
            Hf, Hg = codes(m["W"], arch)
            what = m["w"] / np.linalg.norm(m["w"])
            overlap = np.linalg.norm(Hg @ Hf.T) / (np.linalg.norm(Hg) * np.linalg.norm(Hf))
            for branch, H in (("f", Hg), ("g", Hf)):           # an f-update damages g, whose code is Hg
                rows.append(dict(task=task, seed=seed, hist=hist, branch=branch,
                                 readout_norm=float(np.linalg.norm(m["w"])), code_overlap=float(overlap),
                                 readout_share=float(np.sum((H @ what) ** 2) / np.sum(H ** 2)),
                                 damage=damage[(task, seed, arch, hist, branch)]))
        y = np.array([r["damage"] for r in rows])
        measures = ("readout_norm", "code_overlap", "readout_share")
        corr = {k: float(np.corrcoef([r[k] for r in rows], y)[0, 1]) for k in measures}
        paired = {}
        index = {(r["task"], r["seed"], r["hist"], r["branch"]): r for r in rows}
        for k in measures:
            dx, dy = [], []
            for task in sorted({r["task"] for r in rows}):
                pick = lambda h, f: np.mean([f(index[(task, s, h, b)]) for s in (2701, 2702) for b in ("f", "g")])
                dx.append(pick("independent", lambda r: r[k]) - pick("joint", lambda r: r[k]))
                dy.append(pick("independent", lambda r: r["damage"]) - pick("joint", lambda r: r["damage"]))
            paired[k] = float(np.corrcoef(dx, dy)[0, 1])
            families = len(dx)
        out["architectures"][arch] = dict(
            pairs=len(rel_norm),
            distance_over_norm=dict(median=float(np.median(rel_norm)), min=float(np.min(rel_norm)), max=float(np.max(rel_norm))),
            distance_over_travel=dict(median=float(np.median(rel_travel)), min=float(np.min(rel_travel)), max=float(np.max(rel_travel))),
            correlation_with_damage=corr, paired_difference_correlation=paired, families=families,
            paired_difference_p={k: float(2 * stats.t.sf(abs(r) * np.sqrt((families - 2) / (1 - r * r)), families - 2))
                                 for k, r in paired.items()},
            max_abs_correlation=max(abs(v) for v in list(corr.values()) + list(paired.values())),
            observations=len(rows))
    path = write("posthoc-geometry.json", out)
    for arch in ARCHS:
        a = out["architectures"][arch]
        print(f"{arch:6s} distance/norm median {a['distance_over_norm']['median']:.3f} "
              f"[{a['distance_over_norm']['min']:.3f}, {a['distance_over_norm']['max']:.3f}]; "
              f"distance/travel {a['distance_over_travel']['median']:.3f}; r {a['correlation_with_damage']}; "
              f"paired r {a['paired_difference_correlation']}")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
