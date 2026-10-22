"""Prepare finite task definitions and job plans; never train or load a model."""
from __future__ import annotations

from collections import Counter
import random

from support import preparation_main


def distinct_tables(rng, size):
    tables = []
    while len(tables) < 3:
        table = rng.sample(range(size), size)
        if table not in tables:
            tables.append(table)
    return tables


def make_examples(size, f, g, heldout_offsets, environment):
    rows = []
    for kind, table in (("f", f), ("g", g)):
        for argument in range(size):
            rows.append(dict(id=f"{environment}:{kind}:{argument}", query=kind,
                             a=argument if kind == "f" else None,
                             b=argument if kind == "g" else None,
                             target=table[argument], split="train"))
    for a in range(size):
        for b in range(size):
            split = "heldout_pairs" if (b - a) % size in heldout_offsets else "train"
            rows.append(dict(id=f"{environment}:combine:{a}:{b}", query="combine",
                             a=a, b=b, target=(f[a] + g[b]) % size, split=split))
    return rows


def build(config):
    size = config["domain_size"]
    if size < 3 or config["model_execution_enabled"]:
        raise ValueError("Preparation requires domain >= 3 and model execution disabled")
    if config["history_names"] != ["factorwise", "mixed_change", "shuffled"]:
        raise ValueError("History names must match the draft design")
    if config["common_final_environment"] != "E00":
        raise ValueError("The draft common final environment is E00")
    offsets = config["heldout_pair_offsets"]
    if len(set(offsets)) != len(offsets) or not 0 < len(offsets) < size:
        raise ValueError("Specify distinct held-out offsets leaving some training pairs")
    if any(not 0 <= offset < size for offset in offsets):
        raise ValueError("Held-out offset is outside the domain")
    visits = config["visits_per_environment"]
    if visits < 1:
        raise ValueError("Every environment needs at least one visit")
    tasks, schedules, jobs = [], [], []
    versions = {"E00": (0, 0), "E10": (1, 0), "E11": (1, 1), "E01": (0, 1),
                "update_f": (2, 0), "update_g": (0, 2)}
    for task_seed in config["task_seeds"]:
        rng = random.Random(task_seed)
        f_tables, g_tables = distinct_tables(rng, size), distinct_tables(rng, size)
        environments = {}
        for name, (fi, gi) in versions.items():
            environments[name] = dict(f_version=fi, g_version=gi,
                examples=make_examples(size, f_tables[fi], g_tables[gi], offsets, name))
        tasks.append(dict(task_seed=task_seed, domain_size=size, f_tables=f_tables,
                          g_tables=g_tables, environments=environments))
        factorwise = ["E00", "E10", "E11", "E01"] * visits
        mixed = ["E00", "E11", "E10", "E01"] * visits
        shuffled = factorwise.copy()
        shuffled.remove("E00")
        random.Random(task_seed + 707).shuffle(shuffled)
        shuffled.insert(0, "E00")
        histories = dict(factorwise=factorwise, mixed_change=mixed, shuffled=shuffled)
        for history, order in histories.items():
            order = order + [config["common_final_environment"]]
            transitions = []
            for before, after in zip(order, order[1:]):
                old_rows = environments[before]["examples"]
                new_rows = environments[after]["examples"]
                changed = {}
                for query in ("f", "g", "combine"):
                    pairs = [(a, b) for a, b in zip(old_rows, new_rows)
                             if a["query"] == query and a["split"] == "train"]
                    changed[query] = sum(a["target"] != b["target"] for a, b in pairs) / len(pairs)
                transitions.append(dict(before=before, after=after,
                    changed_functions=sum(a != b for a, b in zip(versions[before], versions[after])),
                    training_target_disagreement=changed))
            schedules.append(dict(task_seed=task_seed, history=history, order=order,
                                  counts=dict(Counter(order)), transitions=transitions))
            for model_seed in config["initialization_seeds"]:
                jobs.append(dict(id=f"task{task_seed}-init{model_seed}-{history}",
                    status="planned_not_started", task_seed=task_seed, model_seed=model_seed,
                    history=history, environment_order=order,
                    base_optimizer_steps=len(order) * config["training"]["steps_per_visit"],
                    adaptation_branches=config["adaptation"]["branches"],
                    controls="no_update_and_continued_old_environment_to_be_implemented",
                    runner_implemented=False))
    counts = dict(project_id="AI-07", status="symbolic_preparation_only",
                  task_families=len(tasks), paired_initializations_per_family=len(config["initialization_seeds"]),
                  histories=len(config["history_names"]), planned_base_training_jobs=len(jobs),
                  planned_update_branches=len(jobs) * len(config["adaptation"]["branches"]),
                  model_runs_completed=0, gpu_used=False, runtime_estimate=None)
    return {"task-families.json": tasks, "training-schedules.json": schedules,
            "job-plan.json": jobs, "counts.json": counts}


if __name__ == "__main__":
    preparation_main(build)
