"""Prespecified analysis of Study 3 (frozen with the protocol before any run).

1. The Phase 2 analysis (`analyze.py`, an unchanged copy) is run on the Study 3 run folder. It extracts one record per
   network, update direction and update condition (errors after 200 steps, progress to 0.10, the readout identity),
   with Phase 2's definitions, into `results/records/`. Its own decision, made under Phase 2's six-interval rule, is
   kept there as a secondary view.
2. From those records this script computes the Study 3 primary result for each architecture: the per-family
   difference D (independent minus joint history) in unchanged-skill error after 200 unrestricted steps, averaging the
   two initialization seeds and the two update directions within each family; its mean and a two-sided Student t
   interval at 97.5% (two primary intervals, Bonferroni, 95% simultaneous coverage); and the decision:
     - competence not met: fewer than 90% of the architecture's pairs competent (no conclusion);
     - equivalence established: the interval lies inside (-0.05, +0.05);
     - difference established: otherwise, the interval excludes zero (direction recorded; "beyond the margin" when it
       lies wholly beyond -0.05 or +0.05);
     - inconclusive: otherwise.
3. Secondary results (descriptive 95% intervals, no multiplicity correction): the changed-skill and withheld-pair
   differences (with the +0.02 noninferiority margin), the update-direction split, fixed-length steps, the update
   conditions, progress to 0.10, and the estimate pooled with Phase 2's 64 families (97.5%).
4. Each numeric prediction in `protocol/predictions.json` is evaluated.

Writes results/summary.json. `python -B code/analyze_study3.py --run runs/confirmation --output results`
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
sys.path.insert(0, str(CODE))
import analyze as phase2_analysis  # noqa: E402  (Phase 2 analysis, unchanged copy)

MARGIN, NONINFERIORITY, PRIMARY_CONFIDENCE, SECONDARY_CONFIDENCE = 0.05, 0.02, 0.975, 0.95
PHASE2_SUMMARY = STUDY.parent / "2-controlled-histories/analysis/confirmation/summary.json"


def interval(values, confidence):
    a = np.asarray(list(values), dtype=float)
    n, mean, sd = len(a), float(a.mean()), float(a.std(ddof=1))
    half = float(student_t.ppf(1 - (1 - confidence) / 2, n - 1) * sd / np.sqrt(n))
    return dict(n=n, mean=mean, sd=sd, low=mean - half, high=mean + half, half_width=half, confidence=confidence)


def family_means(records, select, metric):
    groups = defaultdict(list)
    for r in records:
        if select(r):
            groups[r["task_seed"]].append(r[metric])
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def difference(records, arch, arm, metric, extra=lambda r: True):
    ind = family_means(records, lambda r: r["architecture"] == arch and r["history"] == "independent"
                       and r["arm"] == arm and extra(r), metric)
    joint = family_means(records, lambda r: r["architecture"] == arch and r["history"] == "joint"
                         and r["arm"] == arm and extra(r), metric)
    assert set(ind) == set(joint)
    return {k: ind[k] - joint[k] for k in ind}, ind, joint


def decide(iv, competent):
    if not competent:
        return "competence_not_met"
    if iv["low"] > -MARGIN and iv["high"] < MARGIN:
        return "equivalence_established"
    if iv["high"] < 0 or iv["low"] > 0:
        beyond = iv["high"] < -MARGIN or iv["low"] > MARGIN
        side = "independent_lower" if iv["high"] < 0 else "independent_higher"
        return f"difference_established_{side}" + ("_beyond_margin" if beyond else "")
    return "inconclusive"


def evaluate(prediction, value):
    if "equals" in prediction:
        return dict(prediction, observed=value, held=value == prediction["equals"])
    lo, hi = prediction.get("low", -np.inf), prediction.get("high", np.inf)
    return dict(prediction, observed=value, held=bool(lo <= value <= hi))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    run, out = a.run.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phase2_analysis.analyze(run, out / "records")
    records = json.loads((out / "records/records.json").read_text())
    base = json.loads((out / "records/base-records.json").read_text())
    p2rule = json.loads((out / "records/summary.json").read_text())
    config = json.loads((run / "manifest.json").read_text())["config"]
    archs = config["architectures"]
    res = dict(study="AI-07 Study 3 (larger confirmation)", run=str(run.relative_to(STUDY)),
               families=len(config["task_seeds"]), task_seeds=[config["task_seeds"][0], config["task_seeds"][-1]],
               initialization_seeds=config["initialization_seeds"], networks=len(base), trajectories=len(records),
               updates_per_condition=len(records) // len(config["arms"]),
               margin=MARGIN, primary_confidence=PRIMARY_CONFIDENCE, secondary_confidence=SECONDARY_CONFIDENCE,
               competence={}, primary={}, secondary={}, phase2_rule=dict(
                   note="Phase 2's own decision rule (three endpoints, six 99.17% intervals), secondary",
                   decisions={k: p2rule["primary"][k]["decision"] for k in archs}),
               identity_max_error=p2rule["identity_max_error"],
               normalized_zero_gradient_steps=p2rule["normalized_zero_gradient_steps"])
    for arch in archs:
        pairs = defaultdict(list)
        for b in base:
            if b["architecture"] == arch:
                pairs[(b["task_seed"], b["model_seed"])].append(b["qualified"])
        competent = sum(all(v) for v in pairs.values())
        errs = {k: dict(mean=float(np.mean([b["errors"][k] for b in base if b["architecture"] == arch])),
                        maximum=float(max(b["errors"][k] for b in base if b["architecture"] == arch)))
                for k in ("f", "g", "combined_seen", "combined_unseen")}
        res["competence"][arch] = dict(competent_pairs=competent, pairs=len(pairs), fraction=competent / len(pairs),
                                       rule_met=competent / len(pairs) >= 0.9, base_errors=errs)
        d, ind, joint = difference(records, arch, "unrestricted", "stable")
        iv = interval(d.values(), PRIMARY_CONFIDENCE)
        res["primary"][arch] = dict(**iv, independent_mean=float(np.mean(list(ind.values()))),
                                    joint_mean=float(np.mean(list(joint.values()))),
                                    families_favouring_independent=sum(v < 0 for v in d.values()),
                                    family_min=min(d.values()), family_max=max(d.values()),
                                    decision=decide(iv, res["competence"][arch]["rule_met"]), family_differences=d)
        sec = {}
        for metric in ("changed", "transfer"):
            dm, i_m, j_m = difference(records, arch, "unrestricted", metric)
            sec[metric] = dict(**interval(dm.values(), SECONDARY_CONFIDENCE),
                               independent_mean=float(np.mean(list(i_m.values()))),
                               joint_mean=float(np.mean(list(j_m.values()))))
        sec["changed"]["noninferior_at_0_02"] = sec["changed"]["high"] < NONINFERIORITY
        sec["transfer"]["noninferior_at_0_02"] = sec["transfer"]["high"] < NONINFERIORITY
        fd = {b: difference(records, arch, "unrestricted", "stable", lambda r, b=b: r["branch"] == b)[0] for b in ("f", "g")}
        sec["direction"] = dict(f=interval(fd["f"].values(), SECONDARY_CONFIDENCE),
                                g=interval(fd["g"].values(), SECONDARY_CONFIDENCE),
                                f_minus_g=interval([fd["f"][k] - fd["g"][k] for k in fd["f"]], SECONDARY_CONFIDENCE))
        sec["fixed_length"] = {m: interval(difference(records, arch, "normalized", m)[0].values(), SECONDARY_CONFIDENCE)
                               for m in ("changed", "stable")}
        arms = {}
        for arm in config["arms"]:
            sel = [r for r in records if r["architecture"] == arch and r["arm"] == arm]
            arms[arm] = dict(changed=float(np.mean([r["changed"] for r in sel])),
                             stable=float(np.mean([r["stable"] for r in sel])),
                             transfer=float(np.mean([r["transfer"] for r in sel])),
                             stable_drift_max=float(max(r["stable_drift"] for r in sel)),
                             path=float(np.mean([r["path"] for r in sel])),
                             reached_0_10=sum(r["progress_reached"] for r in sel), total=len(sel))
        reached = [r for r in records if r["architecture"] == arch and r["arm"] == "unrestricted" and r["progress_reached"]]
        sec["conditions"] = arms
        sec["progress_unrestricted"] = dict(reached=len(reached), mean_step=float(np.mean([r["progress_step"] for r in reached])),
                                            mean_stable=float(np.mean([r["progress_stable"] for r in reached])))
        unr = [r for r in records if r["architecture"] == arch and r["arm"] == "unrestricted"]
        sec["withheld_vs_damage_correlation"] = float(np.corrcoef([r["stable"] for r in unr], [r["transfer"] for r in unr])[0, 1])
        p2 = json.loads(PHASE2_SUMMARY.read_text())["primary"][arch]["effects"]["stable"]["family_differences"]
        pooled = list(p2.values()) + list(d.values())
        sec["pooled_with_phase2"] = dict(**interval(pooled, PRIMARY_CONFIDENCE), phase2_families=len(p2))
        res["secondary"][arch] = sec
    preds = json.loads((STUDY / "protocol/predictions.json").read_text())["predictions"]
    observed = {}
    for arch in archs:
        p, s = res["primary"][arch], res["secondary"][arch]
        observed.update({
            f"{arch}.primary.decision": p["decision"], f"{arch}.primary.mean": p["mean"], f"{arch}.primary.sd": p["sd"],
            f"{arch}.primary.half_width": p["half_width"],
            f"{arch}.competence.fraction": res["competence"][arch]["fraction"],
            f"{arch}.damage.independent": p["independent_mean"], f"{arch}.damage.joint": p["joint_mean"],
            f"{arch}.changed.independent": s["changed"]["independent_mean"], f"{arch}.changed.joint": s["changed"]["joint_mean"],
            f"{arch}.changed.noninferior": s["changed"]["noninferior_at_0_02"],
            f"{arch}.withheld_vs_damage_correlation": s["withheld_vs_damage_correlation"],
            f"{arch}.freeze_readout.stable_drift_max": s["conditions"]["freeze_readout"]["stable_drift_max"],
            f"{arch}.random_minus_unrestricted": float(np.mean([s["conditions"][f"random_freeze_{i}"]["stable"] for i in range(3)])
                                                       - s["conditions"]["unrestricted"]["stable"]),
            f"{arch}.readout_only.stable": s["conditions"]["readout_only"]["stable"],
            f"{arch}.fixed_length.changed.high": s["fixed_length"]["changed"]["high"],
            f"{arch}.direction.f.mean": s["direction"]["f"]["mean"], f"{arch}.direction.g.mean": s["direction"]["g"]["mean"],
            f"{arch}.direction.f_minus_g.mean": s["direction"]["f_minus_g"]["mean"],
            f"{arch}.progress.mean_step": s["progress_unrestricted"]["mean_step"],
            f"{arch}.freeze_readout.reached_0_10": s["conditions"]["freeze_readout"]["reached_0_10"],
            f"{arch}.pooled.decision": decide(s["pooled_with_phase2"], True)})
    observed["identity_max_error"] = res["identity_max_error"]
    res["predictions"] = [evaluate(p, observed[p["quantity"]]) for p in preds]
    res["predictions_held"] = sum(p["held"] for p in res["predictions"])
    res["predictions_total"] = len(res["predictions"])
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(dict(primary={k: {x: res["primary"][k][x] for x in ("mean", "low", "high", "decision")} for k in archs},
                          predictions_held=f"{res['predictions_held']}/{res['predictions_total']}"), indent=2))


if __name__ == "__main__":
    main()
