"""Check that Study 3's seeds and tasks are fresh, and that its design matches Phase 2's (run before the freeze).

1. Every seed used earlier in AI-07 is collected: the task and initialization seeds of every Phase 1 and Phase 2
   configuration, the fixture seeds of both phases' tests, and the bootstrap seed of Phase 2's descriptive code. Study
   3's task seeds, initialization seeds and the seeds of its timing pilot must be disjoint from all of them and from
   each other. The random masks are seeded by task_seed * 100000 + model_seed * 10 + draw (+ 70000000 for g updates);
   those seeds are compared between Phase 2's confirmation and Study 3 as well.
2. Study 3's task functions (data/tasks.json) must differ from every Phase 2 task function.
3. Study 3's configuration must equal Phase 2's confirmation configuration in every key except the stage, the task
   seeds and the initialization seeds, and its schedules must equal Phase 2's (so the independent history again changes
   f alone 14 times and g alone 8 times).

Writes ../protocol/seed-check.json and exits nonzero on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
STUDIES = STUDY.parent
P1, P2 = STUDIES / "1-feasibility-classification", STUDIES / "2-controlled-histories"
PILOT = dict(task_seeds=[84001, 84002], initialization_seeds=[3901])


def seeds_in(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "seed" in k.lower():
                if isinstance(v, int):
                    out.add(v)
                elif isinstance(v, list):
                    out.update(x for x in v if isinstance(x, int))
            seeds_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            seeds_in(v, out)


def masks(task_seeds, model_seeds):
    return {t * 100000 + m * 10 + d + (0 if b == "f" else 70000000)
            for t in task_seeds for m in model_seeds for d in range(3) for b in ("f", "g")}


def transitions(order):
    c = Counter()
    for old, new in zip(order, order[1:]):
        f, g = old[0] != new[0], old[1] != new[1]
        c["both" if f and g else "f_only" if f else "g_only"] += 1
    return dict(c)


def main() -> None:
    earlier = set()
    configs = sorted((P1 / "03-Data/config").glob("*.json")) + sorted((P2 / "protocol").glob("*.json"))
    for path in configs:
        seeds_in(json.loads(path.read_text()), earlier)
    tests = [P1 / "02-Code/test_experiment.py", P2 / "code/test_study.py", P2 / "code/test_analysis.py"]
    for path in tests:
        text = path.read_text()
        for a, b in re.findall(r"range\((\d+),\s*(\d+)\)", text):
            if int(a) > 1000:
                earlier.update(range(int(a), int(b)))
        earlier.update(int(v) for v in re.findall(r"make_task\((\d+)\)", text))
        earlier.update(int(v) for v in re.findall(r"network\((\d+),", text))
    config = json.loads((STUDY / "protocol/config.json").read_text())
    p2 = json.loads((P2 / "protocol/confirmation.json").read_text())
    new_tasks, new_inits = config["task_seeds"], config["initialization_seeds"]
    groups = dict(study3_tasks=set(new_tasks), study3_initializations=set(new_inits),
                  pilot_tasks=set(PILOT["task_seeds"]), pilot_initializations=set(PILOT["initialization_seeds"]))
    failures = []
    for name, group in groups.items():
        overlap = sorted(group & earlier)
        if overlap:
            failures.append(f"{name} overlaps earlier seeds: {overlap[:10]}")
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] & groups[b]:
                failures.append(f"{a} overlaps {b}")
    if len(set(new_tasks)) != len(new_tasks):
        failures.append("duplicate Study 3 task seeds")
    mask_overlap = masks(new_tasks, new_inits) & masks(p2["task_seeds"], p2["initialization_seeds"])
    if mask_overlap:
        failures.append(f"mask seeds overlap Phase 2: {sorted(mask_overlap)[:5]}")
    # Task functions: Study 3 against Phase 2 (development and confirmation).
    s3 = json.loads((STUDY / "data/tasks.json").read_text())
    old = [json.loads((P2 / f"data/{n}.json").read_text()) for n in ("development", "confirmation")]
    old_vectors = [np.array(t[k]) for b in old for t in b["tasks"] for k in ("f", "g")]
    closest = min(float(np.min(np.abs(np.array(t[k])[:, None, :] - v[None, :, :]).max(-1)))
                  for t in s3["tasks"] for k in ("f", "g") for v in old_vectors)
    if closest < 1e-6:
        failures.append("a Study 3 task function equals a Phase 2 task function")
    # Design identity with Phase 2.
    differing = sorted(k for k in set(config) | set(p2)
                       if k not in ("stage", "task_seeds", "initialization_seeds") and config.get(k) != p2.get(k))
    if differing:
        failures.append(f"configuration differs from Phase 2 confirmation in {differing}")
    p2_data = json.loads((P2 / "data/confirmation.json").read_text())
    if s3["schedules"] != p2_data["schedules"]:
        failures.append("schedules differ from Phase 2")
    counts = {h: transitions(o) for h, o in s3["schedules"].items()}
    if counts["independent"] != {"f_only": 14, "g_only": 8, "both": 6}:
        failures.append(f"unexpected transition counts {counts}")
    if s3["rows"] != p2_data["rows"]:
        failures.append("evaluation inputs differ from Phase 2")
    out = dict(status="passed" if not failures else "failed", failures=failures,
               earlier_seed_count=len(earlier), earlier_seed_range=[min(earlier), max(earlier)],
               earlier_configurations=[str(p.relative_to(STUDIES)) for p in configs],
               earlier_test_files=[str(p.relative_to(STUDIES)) for p in tests],
               study3=dict(task_seeds=[new_tasks[0], new_tasks[-1], len(new_tasks)], initialization_seeds=new_inits),
               pilot=PILOT, closest_task_function_distance_to_phase2=closest,
               configuration_keys_differing_from_phase2=["stage", "task_seeds", "initialization_seeds"],
               transition_counts=counts)
    (STUDY / "protocol/seed-check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
