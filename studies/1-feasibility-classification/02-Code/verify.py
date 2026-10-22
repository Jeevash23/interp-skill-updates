"""Independent semantic checks of prepared inputs; no model experiment."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    read = lambda name: json.loads((directory / name).read_text())
    config = json.loads((ROOT / "03-Data/config/pilot.json").read_text())
    tasks, schedules, jobs = read("task-families.json"), read("training-schedules.json"), read("job-plan.json")
    require(len({t["task_seed"] for t in tasks}) == len(tasks), "Duplicate task family")
    require({t["task_seed"] for t in tasks} == set(config["task_seeds"]), "Task seed inventory differs")
    label_checks = 0
    for task in tasks:
        size = task["domain_size"]
        domain = set(range(size))
        for key in ("f_tables", "g_tables"):
            require(len(task[key]) == 3 and len({tuple(t) for t in task[key]}) == 3, "Version collision")
            require(all(set(table) == domain and len(table) == size for table in task[key]), "Invalid permutation")
        heldout_sets = []
        for env, spec in task["environments"].items():
            f, g = task["f_tables"][spec["f_version"]], task["g_tables"][spec["g_version"]]
            if env.startswith("E"):
                require(spec["f_version"] < 2 and spec["g_version"] < 2, "Reserved function leaked into history")
            elif env == "update_f":
                require((spec["f_version"], spec["g_version"]) == (2, 0), "f update changed g")
            elif env == "update_g":
                require((spec["f_version"], spec["g_version"]) == (0, 2), "g update changed f")
            else:
                raise ValueError("Unknown environment")
            rows = spec["examples"]
            require(len({r["id"] for r in rows}) == len(rows), "Duplicate example")
            combined, heldout, training = set(), set(), set()
            primitives = {"f": set(), "g": set()}
            for row in rows:
                a, b, query = row["a"], row["b"], row["query"]
                if query == "f":
                    require(a in domain and b is None, "Malformed f input")
                    expected = f[a]
                    primitives["f"].add(a)
                    require(row["split"] == "train", "Primitive split wrong")
                elif query == "g":
                    require(b in domain and a is None, "Malformed g input")
                    expected = g[b]
                    primitives["g"].add(b)
                    require(row["split"] == "train", "Primitive split wrong")
                elif query == "combine":
                    require(a in domain and b in domain, "Combined input out of range")
                    expected = (f[a] + g[b]) % size
                    combined.add((a, b))
                    expected_split = "heldout_pairs" if (b-a) % size in config["heldout_pair_offsets"] else "train"
                    require(row["split"] == expected_split, "Incorrect held-out membership")
                    (heldout if row["split"] == "heldout_pairs" else training).add((a, b))
                else:
                    raise ValueError("Unknown query")
                require(row["target"] == expected, f"Wrong symbolic target: {row['id']}")
                label_checks += 1
            require(combined == {(a, b) for a in domain for b in domain}, "Combined-domain coverage missing")
            require(not heldout & training, "Train/evaluation overlap")
            require(all(values == domain for values in primitives.values()), "Primitive coverage incomplete")
            require(set(Counter(a for a, b in heldout).values()) == {len(config["heldout_pair_offsets"])}, "Unbalanced a coverage")
            require(set(Counter(b for a, b in heldout).values()) == {len(config["heldout_pair_offsets"])}, "Unbalanced b coverage")
            heldout_sets.append(heldout)
        require(all(s == heldout_sets[0] for s in heldout_sets), "Evaluation membership changes across environments")
        histories = [s for s in schedules if s["task_seed"] == task["task_seed"]]
        require({s["history"] for s in histories} == set(config["history_names"]), "Missing history")
        expected_counts = Counter({e: config["visits_per_environment"] for e in ("E00", "E10", "E11", "E01")})
        expected_counts["E00"] += 1
        for history in histories:
            order = history["order"]
            require(order[0] == order[-1] == "E00", "Start/end environments differ")
            require(Counter(order) == expected_counts == Counter(history["counts"]), "Unequal example exposure")
            require(len(history["transitions"]) == len(order)-1, "Missing transitions")
            if history["history"] == "factorwise":
                require(all(sum(x != y for x, y in zip(a[1:], b[1:])) == 1 for a, b in zip(order, order[1:])), "Factorwise schedule changes multiple functions")
    expected_jobs = {(t, i, h) for t in config["task_seeds"] for i in config["initialization_seeds"] for h in config["history_names"]}
    require({(j["task_seed"], j["model_seed"], j["history"]) for j in jobs} == expected_jobs, "Unpaired jobs")
    require(len(jobs) == len(expected_jobs), "Duplicate jobs")
    require(all(j["status"] == "planned_not_started" and not j["runner_implemented"] for j in jobs), "Preparation claims an executed job")
    counts = read("counts.json")
    require(counts["planned_base_training_jobs"] == len(jobs) and counts["model_runs_completed"] == 0, "Incorrect summary counts")
    return dict(project_id="AI-07", check="symbolic_semantics", status="passed",
                exact_label_checks=label_checks, matched_history_jobs=len(jobs), model_execution=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "03-Data/prepared")
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir), indent=2))
