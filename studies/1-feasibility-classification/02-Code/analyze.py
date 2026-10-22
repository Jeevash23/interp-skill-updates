"""Recompute descriptive feasibility tables; no model execution or hypothesis search."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def average(values):
    return statistics.mean(values) if values else None


def table(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest, status = read(run / "manifest.json"), read(run / "status.json")
    config = manifest["configuration"]
    gate = config["gate"]
    base, endpoints, curves, transitions, failures = [], [], [], [], []
    for job in manifest["jobs"]:
        folder = run / job["id"]
        state = read(folder / "status.json") if (folder / "status.json").exists() else {"status": "missing"}
        if state["status"] != "complete":
            failures.append(dict(job=job["id"], **state))
            continue
        training, updates = read(folder / "training.json"), read(folder / "updates.json")
        ident = {k: job[k] for k in ("task_seed", "model_seed", "history")}
        m = training["blocks"][-1]["metrics"]
        row = dict(ident, **{k: v["accuracy"] for k, v in m.items()})
        row["competence_qualified"] = bool(row["f"] >= gate["base_primitive_accuracy"] and row["g"] >= gate["base_primitive_accuracy"] and row["combined_seen"] >= gate["base_seen_combined_accuracy"] and row["combined_heldout"] >= gate["base_heldout_combined_accuracy"])
        base.append(row)
        for block in training["blocks"][1:]:
            transitions.append(dict(ident, block=block["block"], environment=block["environment"],
                                    target_disagreement=block["training_target_disagreement"],
                                    loss_jump=block["transition_loss_jump"], gradient_norm=block["initial_gradient_norm"]))
        for trajectory in updates["trajectories"]:
            branch = trajectory["branch"]
            changed, stable = ("f", "g") if branch == "update_f" else ("g", "f")
            for point in trajectory["curve"]:
                m = point["updated_target"]
                r = dict(ident, branch=branch, condition=trajectory["condition"],
                         optimizer_mode=trajectory["optimizer_mode"], step=point["step"],
                         changed_primitive=m[changed]["accuracy"], unchanged_primitive=m[stable]["accuracy"],
                         combined_seen=m["combined_seen"]["accuracy"], combined_heldout=m["combined_heldout"]["accuracy"],
                         old_combined_seen=point["old_target"]["combined_seen"]["accuracy"],
                         old_combined_heldout=point["old_target"]["combined_heldout"]["accuracy"],
                         parameter_delta_l2=point["parameter_delta_l2"])
                curves.append(r)
                if point["step"] == max(config["evaluation_steps"]):
                    endpoints.append(r)
    if not base:
        raise ValueError("No complete base jobs")
    table(destination / "base.csv", base)
    table(destination / "endpoints.csv", endpoints)
    table(destination / "curves.csv", curves)
    table(destination / "transitions.csv", transitions)

    paired = []
    for task in config["task_seeds"]:
        for seed in config["initialization_seeds"]:
            group = [r for r in base if r["task_seed"] == task and r["model_seed"] == seed]
            paired.append(dict(task_seed=task, model_seed=seed, histories=len(group),
                               qualified=len(group) == len(config["histories"]) and all(r["competence_qualified"] for r in group)))
    qualified_by_task = {str(task): sum(r["qualified"] for r in paired if r["task_seed"] == task) for task in config["task_seeds"]}
    completeness = len(base) == len(config["task_seeds"]) * len(config["initialization_seeds"]) * len(config["histories"]) and not failures
    competence = completeness and all(n >= gate["minimum_qualified_initializations_per_task"] for n in qualified_by_task.values())
    family = []
    for task in config["task_seeds"]:
        for history in config["histories"]:
            for mode in config["optimizer_modes"] + ["none"]:
                for condition in config["update_conditions"]:
                    group = [r for r in endpoints if r["task_seed"] == task and r["history"] == history and r["optimizer_mode"] == mode and r["condition"] == condition]
                    if group:
                        family.append(dict(task_seed=task, history=history, optimizer_mode=mode, condition=condition,
                                           trajectories=len(group), **{k: average([r[k] for r in group]) for k in
                                               ("changed_primitive", "unchanged_primitive", "combined_seen", "combined_heldout", "parameter_delta_l2")}))
    table(destination / "family-endpoints.csv", family)
    effects, effects_by_mode = [], {}
    for mode in config["optimizer_modes"]:
        for task in config["task_seeds"]:
            differences = []
            for seed in config["initialization_seeds"]:
                values = {}
                for history in ("factorwise", "shuffled"):
                    rows = [r for r in endpoints if (r["task_seed"], r["model_seed"], r["history"], r["optimizer_mode"], r["condition"]) == (task, seed, history, mode, "changed_primitive")]
                    if len(rows) == 2:
                        values[history] = average([r["combined_heldout"] for r in rows])
                if len(values) == 2:
                    difference = values["factorwise"] - values["shuffled"]
                    effects.append(dict(task_seed=task, model_seed=seed, optimizer_mode=mode, factorwise_minus_shuffled=difference))
                    differences.append(difference)
            if differences:
                effects_by_mode.setdefault(mode, []).append(dict(task_seed=task, paired_initializations=len(differences),
                                                               mean=average(differences), sd=statistics.stdev(differences) if len(differences)>1 else None))
    table(destination / "paired-effects.csv", effects)
    primary = effects_by_mode.get(config["primary_optimizer_mode"], [])
    mean_effect = average([r["mean"] for r in primary])
    sds = [r["sd"] for r in primary if r["sd"] is not None]
    variability = (average([s*s for s in sds]) ** .5) if sds else None
    enough_effect = completeness and sum(r["mean"] >= gate["minimum_factorwise_advantage"] for r in primary) >= gate["minimum_positive_families"] and all(r["mean"] >= -gate["maximum_allowed_family_reversal"] for r in primary)
    exceeds_variability = mean_effect is not None and variability is not None and mean_effect > variability
    target_rows = [r for r in family if r["history"] == "factorwise" and r["condition"] == "changed_primitive" and r["optimizer_mode"] == config["primary_optimizer_mode"]]
    useful_adaptation = completeness and all(r["changed_primitive"] >= gate["adapted_primitive_accuracy"] and r["unchanged_primitive"] >= gate["unchanged_primitive_accuracy"] and r["combined_heldout"] >= gate["adapted_heldout_combined_accuracy"] for r in target_rows)
    controls = []
    for r in target_rows:
        for condition, mode in (("no_update", "none"), ("old_primitive", config["primary_optimizer_mode"]), ("old_environment", config["primary_optimizer_mode"])):
            candidates = [q for q in family if q["task_seed"] == r["task_seed"] and q["history"] == r["history"] and q["condition"] == condition and q["optimizer_mode"] == mode]
            if candidates:
                controls.append(dict(task_seed=r["task_seed"], comparison=condition,
                                     combined_heldout_gain=r["combined_heldout"]-candidates[0]["combined_heldout"]))
    control_support = completeness and len(controls) == 3*len(config["task_seeds"]) and all(r["combined_heldout_gain"]>0 for r in controls)
    retained = effects_by_mode.get("retained", [])
    retained_consistency = completeness and sum(r["mean"]>0 for r in retained)>=gate["minimum_positive_families"] and all(r["mean"]>=-gate["maximum_allowed_family_reversal"] for r in retained)
    criteria = dict(complete_coverage=completeness, matched_base_competence=competence,
                    useful_selective_adaptation=useful_adaptation, history_effect=enough_effect,
                    exceeds_initialization_variability=exceeds_variability,
                    improvement_over_controls=control_support, retained_optimizer_direction_consistent=retained_consistency)
    result = dict(stage=manifest["stage"], run=run.name, planned_jobs=len(manifest["jobs"]), completed_jobs=len(base), failures=failures,
                  paired_initializations=paired, qualified_pairs_by_task=qualified_by_task,
                  gate_criteria=criteria, gate_passed=all(criteria.values()),
                  decision="transition_controlled_development_required" if all(criteria.values()) else "stop_mechanism_stage",
                  mean_primary_history_effect=mean_effect, within_family_initialization_sd=variability,
                  history_effects_by_mode=effects_by_mode, factorwise_vs_controls=controls,
                  family_endpoints=family, base_history_means={h:{k:average([r[k] for r in base if r["history"]==h]) for k in ("f","g","combined_seen","combined_heldout")} for h in config["histories"]},
                  all_results_development=True, independent_task_families=len(config["task_seeds"]),
                  transition_confound_resolved=False, causal_claim_allowed=False,
                  run_seconds=status.get("elapsed_seconds"), maximum_rss_kib=status.get("maximum_rss_kib"),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(destination / "summary.json", result)
    print(json.dumps({k:result[k] for k in ("run","completed_jobs","gate_criteria","gate_passed","base_history_means","mean_primary_history_effect","within_family_initialization_sd","decision")},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
