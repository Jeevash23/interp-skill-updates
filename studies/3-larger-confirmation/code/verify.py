"""Independent verification of Study 3 from the saved raw outputs (frozen with the protocol before any run).

This script imports none of the study code (`study.py`, `analyze.py`, `analyze_study3.py`). It rebuilds the task
labels and the evaluation inputs from `data/tasks.json`, re-implements the networks in NumPy, and from every
network's saved arrays recomputes:

- competence: the base network's predictions recomputed from its saved weights, and its errors on the f and g
  queries, observed pairs and withheld pairs; the paired networks' identical initial weights;
- every update: the targets, the errors after 200 steps under each of the nine conditions, the final predictions
  recomputed from the saved final weights, the readout identity at every step, the frozen readout's exact
  preservation, and the first step at which the changed-skill error reaches 0.10;
- every family-level result: the primary differences and 97.5% intervals and their decisions, and the secondary
  intervals, condition means and pooled estimate.

Each result is compared with `results/summary.json`; numbers must agree to 1e-9 and decisions exactly. It also
checks that every frozen file still has its hash in `FREEZE.json`, that the run started after the freeze with the
frozen sources, and that the run is complete. Writes results/verification.json; exits nonzero on any failure.

    python -B code/verify.py --run runs/confirmation --summary results/summary.json --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
PHASE2_SUMMARY = STUDY.parent / "2-controlled-histories/analysis/confirmation/summary.json"
TOL = 1e-9


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs():
    rows, x = [], []
    for a in range(8):
        z = np.zeros(16); z[a] = 1; x.append(z); rows.append(("f", a, None))
    for b in range(8):
        z = np.zeros(16); z[8 + b] = 1; x.append(z); rows.append(("g", None, b))
    for a in range(8):
        for b in range(8):
            z = np.zeros(16); z[a] = 1; z[8 + b] = 1; x.append(z); rows.append(("c", a, b))
    held = np.array([i for i, r in enumerate(rows) if r[0] == "c" and (r[2] - r[1]) % 8 in (0, 1)])
    seen = np.array([i for i, r in enumerate(rows) if r[0] == "c" and (r[2] - r[1]) % 8 not in (0, 1)])
    return np.array(x), held, seen


X, HELD, SEEN = inputs()


def labels(task, env):
    f, g = np.array(task["f"][int(env[0])]), np.array(task["g"][int(env[1])])
    return np.concatenate([f, g, (f[:, None] + g[None, :]).reshape(-1)])


def forward(W, w, arch):
    h = X @ W.T
    if arch == "leaky":
        h = np.where(h > 0, h, 0.9 * h)
    return (h @ w.T).reshape(-1)


def rmse(d, idx, scale=1.0):
    return float(np.sqrt(np.mean(d[idx] ** 2) / scale))


def check_job(args):
    folder, job, task, arms, thresholds = args
    folder, arch = Path(folder), job["architecture"]
    out = dict(job=job)
    status = json.loads((folder / "status.json").read_text())
    out["complete"] = status["status"] == "complete"
    with np.load(folder / "initial.npz") as z:
        h = hashlib.sha256()
        for k in sorted(z.files):
            h.update(k.encode()); h.update(z[k].tobytes())
        out["initial"] = h.hexdigest()
    with np.load(folder / "base.npz") as z:
        pred = forward(z["0.weight"], z["2.weight"], arch)
        out["base_forward_error"] = float(np.max(np.abs(pred - z["predictions"])))
        target = labels(task, "00")
        out["base_target_error"] = float(np.max(np.abs(target - z["targets"])))
        d = z["predictions"] - target
    out["base_errors"] = dict(f=rmse(d, np.arange(8)), g=rmse(d, np.arange(8, 16)), combined_seen=rmse(d, SEEN, 2),
                              combined_unseen=rmse(d, HELD, 2))
    e = out["base_errors"]
    out["qualified"] = (max(e["f"], e["g"]) <= thresholds[0] and e["combined_seen"] <= thresholds[1]
                        and e["combined_unseen"] <= thresholds[2])
    with np.load(folder / "updates.npz") as z, np.load(folder / "endpoints.npz") as ends:
        p, targets, drift_pred, paths = z["predictions"], z["targets"], z["readout_predicted_drift"], z["path_length"]
        out["target_error"] = float(max(np.max(np.abs(targets[0] - labels(task, "40"))),
                                        np.max(np.abs(targets[1] - labels(task, "04")))))
        identity, endpoint_err, freeze_exact = 0.0, 0.0, True
        res = {}
        for bi, branch in enumerate(("f", "g")):
            changed = np.arange(8) if branch == "f" else np.arange(8, 16)
            stable = np.arange(8, 16) if branch == "f" else np.arange(8)
            for ai, arm in enumerate(arms):
                v = p[bi, ai]
                sq = (v - targets[bi]) ** 2
                ce = np.sqrt(sq[:, changed].mean(1)); se = np.sqrt(sq[:, stable].mean(1))
                te = np.sqrt(sq[:, HELD].mean(1) / 2)
                drift = v[:, stable] - v[0, stable]
                identity = max(identity, float(np.max(np.abs(drift - drift_pred[bi, ai]))))
                final = forward(ends[f"{branch}__{arm}__0.weight"], ends[f"{branch}__{arm}__2.weight"], arch)
                endpoint_err = max(endpoint_err, float(np.max(np.abs(final - v[-1]))))
                if arm == "freeze_readout":
                    freeze_exact &= bool(np.array_equal(v[:, stable], np.broadcast_to(v[0, stable], v[:, stable].shape)))
                hit = np.flatnonzero(ce <= 0.1)
                res[(branch, arm)] = dict(changed=float(ce[-1]), stable=float(se[-1]), transfer=float(te[-1]),
                                          drift=float(np.sqrt((drift[-1] ** 2).mean())), path=float(paths[bi, ai, -1]),
                                          first=int(hit[0]) if len(hit) else None,
                                          stable_at_first=float(se[hit[0]]) if len(hit) else None)
        out["identity_error"], out["endpoint_forward_error"], out["freeze_exact"] = identity, endpoint_err, freeze_exact
        out["results"] = {f"{b}|{a}": v for (b, a), v in res.items()}
    return out


def interval(values, confidence):
    a = np.asarray(list(values), dtype=float)
    n, mean, sd = len(a), float(a.mean()), float(a.std(ddof=1))
    half = float(student_t.ppf(1 - (1 - confidence) / 2, n - 1) * sd / np.sqrt(n))
    return dict(mean=mean, sd=sd, low=mean - half, high=mean + half)


def decision(iv, competent, margin=0.05):
    if not competent:
        return "competence_not_met"
    if -margin < iv["low"] and iv["high"] < margin:
        return "equivalence_established"
    if iv["high"] < 0 or iv["low"] > 0:
        side = "independent_lower" if iv["high"] < 0 else "independent_higher"
        beyond = iv["high"] < -margin or iv["low"] > margin
        return f"difference_established_{side}" + ("_beyond_margin" if beyond else "")
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run, summary_path = a.run.resolve(), a.summary.resolve()
    failures, compared = [], []

    def same(label, mine, theirs):
        compared.append(label)
        if isinstance(mine, str) or isinstance(theirs, str) or isinstance(mine, bool):
            if mine != theirs:
                failures.append(f"{label}: recomputed {mine!r}, summary {theirs!r}")
        elif abs(mine - theirs) > TOL:
            failures.append(f"{label}: recomputed {mine}, summary {theirs}")

    # Freeze and run provenance.
    freeze = json.loads((STUDY / "FREEZE.json").read_text())
    changed = [rel for rel, digest in freeze["files"].items() if sha(STUDY / rel) != digest]
    if changed:
        failures.append(f"frozen files changed: {changed}")
    manifest = json.loads((run / "manifest.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if not (status["status"] == "complete" and not status["failures"]):
        failures.append("run is not complete")
    if not datetime.fromisoformat(freeze["created_utc"]) < datetime.fromisoformat(manifest["created_utc"]):
        failures.append("run started before the freeze")
    for rel, digest in manifest["source_hashes"].items():
        if rel in freeze["files"] and freeze["files"][rel] != digest:
            failures.append(f"run source {rel} differs from the frozen one")
    config = json.loads((STUDY / "protocol/config.json").read_text())
    if manifest["config"] != config:
        failures.append("run configuration differs from protocol/config.json")
    tasks = {t["seed"]: t for t in json.loads((STUDY / "data/tasks.json").read_text())["tasks"]}
    if sorted(tasks) != sorted(config["task_seeds"]):
        failures.append("task data does not match the configured seeds")
    jobs = manifest["jobs"]
    expected = [dict(task_seed=t, model_seed=s, architecture=ar, history=h) for t in config["task_seeds"]
                for s in config["initialization_seeds"] for ar in config["architectures"] for h in config["histories"]]
    if jobs != expected:
        failures.append("run jobs differ from the planned design")
    thresholds = (config["base_primitive_nrmse_max"], config["base_seen_nrmse_max"], config["base_unseen_nrmse_max"])
    work = [(str(run / f"t{j['task_seed']}-s{j['model_seed']}-{j['architecture']}-{j['history']}"), j,
             tasks[j["task_seed"]], config["arms"], thresholds) for j in jobs]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        checked = list(pool.map(check_job, work, chunksize=4))
    maxima = {k: max(c[k] for c in checked) for k in ("base_forward_error", "base_target_error", "target_error",
                                                      "identity_error", "endpoint_forward_error")}
    for k, v in maxima.items():
        if v > 1e-9:
            failures.append(f"{k} {v}")
    if not all(c["complete"] for c in checked):
        failures.append("a network folder is not complete")
    if not all(c["freeze_exact"] for c in checked):
        failures.append("a frozen-readout update changed the unchanged answers")
    by = {(c["job"]["task_seed"], c["job"]["model_seed"], c["job"]["architecture"], c["job"]["history"]): c for c in checked}
    for (t, s, ar, h), c in by.items():
        if h == "independent" and c["initial"] != by[(t, s, ar, "joint")]["initial"]:
            failures.append(f"paired initial weights differ: {t} {s} {ar}")
    summary = json.loads(summary_path.read_text())
    p2 = json.loads(PHASE2_SUMMARY.read_text())["primary"]
    recomputed = {}
    for ar in config["architectures"]:
        pairs = defaultdict(list)
        for (t, s, a_, h), c in by.items():
            if a_ == ar:
                pairs[(t, s)].append(c["qualified"])
        comp = sum(all(v) for v in pairs.values())
        same(f"{ar}.competent_pairs", comp, summary["competence"][ar]["competent_pairs"])
        competent = comp / len(pairs) >= 0.9

        def fam(arm, metric, branch=None, history="independent"):
            out = defaultdict(list)
            for (t, s, a_, h), c in by.items():
                if a_ == ar and h == history:
                    for b in (("f", "g") if branch is None else (branch,)):
                        out[t].append(c["results"][f"{b}|{arm}"][metric])
            return {k: float(np.mean(v)) for k, v in out.items()}

        def diff(arm, metric, branch=None):
            i, j = fam(arm, metric, branch), fam(arm, metric, branch, "joint")
            return {k: i[k] - j[k] for k in sorted(i)}

        d = diff("unrestricted", "stable")
        iv = interval(d.values(), 0.975)
        prim = summary["primary"][ar]
        for k in ("mean", "sd", "low", "high"):
            same(f"{ar}.primary.{k}", iv[k], prim[k])
        same(f"{ar}.primary.decision", decision(iv, competent), prim["decision"])
        same(f"{ar}.primary.families_favouring_independent", sum(v < 0 for v in d.values()),
             prim["families_favouring_independent"])
        sec = summary["secondary"][ar]
        for metric in ("changed", "transfer"):
            ivm = interval(diff("unrestricted", metric).values(), 0.95)
            for k in ("mean", "low", "high"):
                same(f"{ar}.{metric}.{k}", ivm[k], sec[metric][k])
        df, dg = diff("unrestricted", "stable", "f"), diff("unrestricted", "stable", "g")
        for name, vals in (("f", df.values()), ("g", dg.values()), ("f_minus_g", [df[k] - dg[k] for k in df])):
            ivd = interval(vals, 0.95)
            for k in ("mean", "low", "high"):
                same(f"{ar}.direction.{name}.{k}", ivd[k], sec["direction"][name][k])
        for metric in ("changed", "stable"):
            ivn = interval(diff("normalized", metric).values(), 0.95)
            for k in ("mean", "low", "high"):
                same(f"{ar}.fixed_length.{metric}.{k}", ivn[k], sec["fixed_length"][metric][k])
        for arm in config["arms"]:
            vals = [c["results"][f"{b}|{arm}"] for (t, s, a_, h), c in by.items() if a_ == ar for b in ("f", "g")]
            for metric in ("changed", "stable", "transfer"):
                same(f"{ar}.{arm}.{metric}", float(np.mean([v[metric] for v in vals])), sec["conditions"][arm][metric])
            same(f"{ar}.{arm}.reached_0_10", sum(v["first"] is not None for v in vals), sec["conditions"][arm]["reached_0_10"])
        unr = [c["results"][f"{b}|unrestricted"] for (t, s, a_, h), c in by.items() if a_ == ar for b in ("f", "g")]
        hit = [v for v in unr if v["first"] is not None]
        same(f"{ar}.progress.mean_step", float(np.mean([v["first"] for v in hit])), sec["progress_unrestricted"]["mean_step"])
        pooled = interval(list(p2[ar]["effects"]["stable"]["family_differences"].values()) + list(d.values()), 0.975)
        for k in ("mean", "low", "high"):
            same(f"{ar}.pooled.{k}", pooled[k], sec["pooled_with_phase2"][k])
        recomputed[ar] = dict(mean=iv["mean"], low=iv["low"], high=iv["high"], decision=decision(iv, competent))
    out = dict(status="passed" if not failures else "failed", created_utc=datetime.now(timezone.utc).isoformat(),
               networks=len(checked), trajectories=len(checked) * 2 * len(config["arms"]),
               comparisons=len(compared), failures=failures, maxima=maxima, primary=recomputed,
               freeze_created_utc=freeze["created_utc"], run_created_utc=manifest["created_utc"],
               source_sha256=sha(Path(__file__)))
    (summary_path.parent / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"} | dict(failures=failures[:20]), indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
