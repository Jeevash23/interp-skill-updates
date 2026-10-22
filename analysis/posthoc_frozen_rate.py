"""Post hoc (9 October 2026): what sets the speed of frozen-readout updates?

With the readout w frozen, an update on the eight queries of f changes only the f slot's input weights W_a. For the
linear network the answer to query a is w^T W_a e_a, and one gradient step on the mean squared error multiplies each
residual by (1 - lr |w|^2 / 4), the same factor for every a. The changed-skill error therefore falls geometrically,
and the first step at which it reaches a level e is predicted exactly by

    t(e) = ceil( log(e / e0) / log(1 - lr |w|^2 / 4) ),

where e0 is the error before the update and |w| the readout norm of the trained network. This script compares that
prediction with the first-passage steps recorded by the 1,000-step frozen-readout run
(studies/2-controlled-histories/runs/duration-control/records.json) for all linear networks, and, as an
approximation, for the leaky networks with |w|^2 replaced by the mean over the eight f (or g) queries of
sum_i w_i^2 phi'(z_ai)^2 at the start of the update. For comparison it also reports, for the linear networks, the mean
squared norm of the changed function's hidden codes, tr(H H^T)/8: in an unrestricted update the readout adds the code
Gram matrix H H^T to the |w|^2 I that the input weights contribute to the per-step change of the answers.

Writes analysis/results/posthoc-frozen-rate.json.
"""
from __future__ import annotations

import math

import numpy as np

from common import ARCHS, P2, read, write

RUN = P2 / "runs/duration-control"
BASE = P2 / "runs/confirmation"


def main() -> None:
    lr = read(P2 / "protocol/confirmation.json")["update_learning_rate"]
    records = read(RUN / "records.json")
    out = dict(post_hoc=True, created="2026-10-09", learning_rate=lr, architectures={})
    for arch in ARCHS:
        diffs, norms, codes = [], [], []
        for r in (x for x in records if x["architecture"] == arch):
            name = f"t{r['task_seed']}-s{r['model_seed']}-{arch}-{r['history']}"
            with np.load(BASE / name / "base.npz") as b:
                W, w = b["0.weight"], b["2.weight"][0]
            with np.load(RUN / f"{name}.npz") as z:
                bi = 0 if r["branch"] == "f" else 1
                idx = np.arange(8) if bi == 0 else np.arange(8, 16)
                e0 = float(np.sqrt(np.mean((z["predictions"][bi, 0, idx] - z["targets"][bi, idx]) ** 2)))
            if arch == "linear":
                gain = float(w @ w)
                H = (W[:, :8] if bi == 0 else W[:, 8:]).T
                codes.append(float(np.trace(H @ H.T) / 8))
            else:
                cols = W[:, :8] if bi == 0 else W[:, 8:]
                slope = np.where(cols > 0, 1.0, 0.9)                       # leaky ReLU derivative at the start
                gain = float(np.mean(((w[:, None] * slope) ** 2).sum(0)))
            factor = 1 - lr * gain / 4
            norms.append(gain)
            pred = [math.ceil(math.log(e / e0) / math.log(factor)) for e in (0.1, 0.01)]
            diffs.append((pred[0] - r["first_0_10"], pred[1] - r["first_0_01"]))
        d = np.array(diffs)
        out["architectures"][arch] = dict(
            trajectories=len(d), exact_0_10=int((d[:, 0] == 0).sum()), exact_0_01=int((d[:, 1] == 0).sum()),
            max_abs_error_0_10=int(np.abs(d[:, 0]).max()), max_abs_error_0_01=int(np.abs(d[:, 1]).max()),
            mean_abs_error_0_10=float(np.abs(d[:, 0]).mean()), mean_gain=float(np.mean(norms)),
            gain_range=[float(np.min(norms)), float(np.max(norms))],
            mean_code_norm_squared=float(np.mean(codes)) if codes else None,
            code_to_readout_ratio=float(np.mean(codes) / np.mean(norms)) if codes else None)
    path = write("posthoc-frozen-rate.json", out)
    for arch, v in out["architectures"].items():
        print(f"{arch:6s} exact first passage at 0.10: {v['exact_0_10']}/{v['trajectories']}, at 0.01: {v['exact_0_01']}; "
              f"max error {v['max_abs_error_0_10']}/{v['max_abs_error_0_01']} steps; mean gain {v['mean_gain']:.3f}")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
