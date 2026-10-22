"""Bounded CPU execution for AI-07; never runs on import or modifies old studies."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_guard(config, directory):
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    directory = Path(directory)
    while not directory.exists():
        directory = directory.parent
    disk = shutil.disk_usage(directory).free
    if available < config["minimum_available_memory_gib"] * 2**30:
        raise RuntimeError("Insufficient available memory before model allocation")
    if disk < config["minimum_free_disk_gib"] * 2**30:
        raise RuntimeError("Insufficient disk before model allocation")
    return dict(available_memory_bytes=available, free_disk_bytes=disk)


def initialize():
    global torch, np
    import torch
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)


def network(seed, sizes=(64, 64)):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(19, sizes[0]), torch.nn.ReLU(),
                               torch.nn.Linear(sizes[0], sizes[1]), torch.nn.ReLU(),
                               torch.nn.Linear(sizes[1], 8))


def encode(rows):
    x = torch.zeros((len(rows), 19), dtype=torch.float32)
    for i, r in enumerate(rows):
        x[i, {"f": 0, "g": 1, "combine": 2}[r["query"]]] = 1
        if r["a"] is not None:
            x[i, 3 + r["a"]] = 1
        if r["b"] is not None:
            x[i, 11 + r["b"]] = 1
    return x, torch.tensor([r["target"] for r in rows], dtype=torch.long)


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parameter_delta(model, reference):
    return float(sum((p.detach() - reference[k]).square().sum().item()
                     for k, p in model.named_parameters()) ** 0.5)


def metrics(logits, rows):
    predictions = logits.argmax(-1)
    targets = np.array([r["target"] for r in rows])
    result = {}
    for label, choose in (
        ("f", lambda r: r["query"] == "f"),
        ("g", lambda r: r["query"] == "g"),
        ("combined_seen", lambda r: r["query"] == "combine" and r["split"] == "train"),
        ("combined_heldout", lambda r: r["query"] == "combine" and r["split"] == "heldout_pairs"),
    ):
        indices = np.array([i for i, r in enumerate(rows) if choose(r)], dtype=int)
        result[label] = dict(correct=int((predictions[indices] == targets[indices]).sum()),
                             total=len(indices), accuracy=float((predictions[indices] == targets[indices]).mean()))
    return result


def predict(model, rows):
    with torch.no_grad():
        logits = model(encode(rows)[0]).numpy().copy()
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite evaluation logits")
    return logits


def update_batch(task, branch, condition):
    primitive = "f" if branch == "update_f" else "g"
    environment = branch if condition == "changed_primitive" else "E00"
    rows = task["environments"][environment]["examples"]
    if condition == "no_update":
        return []
    if condition == "old_environment":
        return [r for r in rows if r["split"] == "train"]
    if condition not in ("changed_primitive", "old_primitive"):
        raise ValueError(condition)
    return [r for r in rows if r["query"] == primitive]


def clone_branch(model, optimizer, config, mode):
    clone = copy.deepcopy(model)
    if mode == "none":
        return clone, None
    opt = torch.optim.Adam(clone.parameters(), lr=config["learning_rate"])
    if mode == "retained":
        opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    elif mode != "reset":
        raise ValueError(mode)
    return clone, opt


def step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    grad = sum(p.grad.detach().square().sum().item() for p in model.parameters()) ** 0.5
    optimizer.step()
    return float(loss.detach()), float(grad)


def emit(handle, common, rows, logits, step_number, elapsed):
    for row, values in zip(rows, logits):
        record = dict(common, step=step_number, elapsed_seconds=elapsed, split=row["split"],
                      query=row["query"], example_id=row["id"], a=row["a"], b=row["b"],
                      predicted_class=int(values.argmax()), target_class=row["target"],
                      logits=[float(v) for v in values], status="ok")
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def train_base(task, job, config, folder, provenance):
    started = time.perf_counter()
    model = network(job["model_seed"], config["hidden_sizes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    initial_hash = state_hash(model)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), initial_hash=initial_hash), folder / "initial.pt")
    logs = []
    previous_rows = None
    global_step = 0
    for block, environment in enumerate(job["environment_order"]):
        rows = task["environments"][environment]["examples"]
        train_rows = [r for r in rows if r["split"] == "train"]
        x, y = encode(train_rows)
        with torch.no_grad():
            new_loss = float(torch.nn.functional.cross_entropy(model(x), y))
            old_loss = None if previous_rows is None else float(torch.nn.functional.cross_entropy(model(encode(previous_rows)[0]), encode(previous_rows)[1]))
        target_disagreement = None if previous_rows is None else sum(a["target"] != b["target"] for a, b in zip(previous_rows, train_rows)) / len(train_rows)
        nsteps = config["steps_per_environment"] + (config["final_extra_steps"] if block == len(job["environment_order"]) - 1 else 0)
        checkpoints = []
        first_gradient = None
        for s in range(1, nsteps + 1):
            loss, gradient = step(model, optimizer, x, y)
            global_step += 1
            if s == 1:
                first_gradient = gradient
            if s == 1 or s % 25 == 0 or s == nsteps:
                checkpoints.append(dict(step=s, global_step=global_step, training_loss_before_step=loss, gradient_norm=gradient))
        final_logits = predict(model, rows)
        logs.append(dict(block=block, environment=environment, steps=nsteps,
                         previous_environment_loss=old_loss, new_environment_loss=new_loss,
                         transition_loss_jump=None if old_loss is None else new_loss-old_loss,
                         training_target_disagreement=target_disagreement, initial_gradient_norm=first_gradient,
                         curve=checkpoints, metrics=metrics(final_logits, rows)))
        previous_rows = train_rows
    final_hash = state_hash(model)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), initial_hash=initial_hash,
                    final_hash=final_hash, provenance=provenance), folder / "base.pt")
    write(folder / "training.json", dict(initial_hash=initial_hash, final_hash=final_hash,
                                         base_optimizer_steps=global_step, blocks=logs,
                                         elapsed_seconds=time.perf_counter()-started))
    rows = task["environments"]["E00"]["examples"]
    with gzip.open(folder / "base-observations.jsonl.gz", "wt") as handle:
        emit(handle, dict(provenance, environment="E00", branch="base", condition="base", optimizer_mode="retained"),
             rows, predict(model, rows), global_step, time.perf_counter()-started)
    return model, optimizer


def run_updates(task, model, optimizer, config, folder, provenance):
    started = time.perf_counter()
    base_hash = state_hash(model)
    reference = {k: p.detach().clone() for k, p in model.named_parameters()}
    summaries = []
    old_rows = task["environments"]["E00"]["examples"]
    with gzip.open(folder / "update-observations.jsonl.gz", "wt") as handle:
        for branch in ("update_f", "update_g"):
            eval_rows = task["environments"][branch]["examples"]
            for condition in config["update_conditions"]:
                modes = ["none"] if condition == "no_update" else config["optimizer_modes"]
                for mode in modes:
                    branch_start = time.perf_counter()
                    clone, opt = clone_branch(model, optimizer, config, mode)
                    if state_hash(clone) != base_hash:
                        raise AssertionError("Branch does not start from common checkpoint")
                    batch = update_batch(task, branch, condition)
                    if batch:
                        x, y = encode(batch)
                    common = dict(provenance, environment=branch, branch=branch, condition=condition,
                                  optimizer_mode=mode, training_example_ids=[r["id"] for r in batch])
                    curve = []
                    initial_logits = predict(clone, eval_rows)
                    for s in range(max(config["evaluation_steps"]) + 1):
                        if s > 0 and opt is not None:
                            loss, grad = step(clone, opt, x, y)
                        else:
                            loss, grad = None, None
                        if s in config["evaluation_steps"]:
                            logits = predict(clone, eval_rows)
                            elapsed = time.perf_counter()-branch_start
                            if condition == "no_update" and not np.array_equal(logits, initial_logits):
                                raise AssertionError("No-update outputs changed")
                            emit(handle, common, eval_rows, logits, s, elapsed)
                            curve.append(dict(step=s, updated_target=metrics(logits, eval_rows),
                                              old_target=metrics(logits, old_rows), parameter_delta_l2=parameter_delta(clone, reference),
                                              training_loss_before_step=loss, gradient_norm=grad,
                                              elapsed_seconds=elapsed))
                    checkpoint = f"{branch}-{condition}-{mode}.pt"
                    torch.save(dict(model=clone.state_dict(), optimizer=None if opt is None else opt.state_dict(),
                                    common=common, base_hash=base_hash, final_hash=state_hash(clone)), folder / checkpoint)
                    summaries.append(dict(branch=branch, condition=condition, optimizer_mode=mode,
                                          training_examples_per_step=len(batch), optimizer_steps=0 if opt is None else 100,
                                          start_hash=base_hash, final_hash=state_hash(clone), checkpoint=checkpoint, curve=curve))
                    if state_hash(model) != base_hash:
                        raise AssertionError("Updating a branch changed the parent model")
    write(folder / "updates.json", dict(trajectories=summaries, elapsed_seconds=time.perf_counter()-started))


def select_jobs(config, qualification):
    jobs = read(ROOT / "03-Data/prepared/job-plan.json")
    chosen = [j for j in jobs if j["task_seed"] in config["task_seeds"] and j["model_seed"] in config["initialization_seeds"] and j["history"] in config["histories"]]
    if qualification:
        chosen = [j for j in chosen if j["task_seed"] == min(config["task_seeds"]) and j["model_seed"] == min(config["initialization_seeds"])]
    if not chosen:
        raise ValueError("No matching jobs")
    return chosen


def execute(config_path, output, qualification=False):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read(config_path)
    if config["device"] != "cpu" or config["threads"] != 1 or config["dtype"] != "float32":
        raise ValueError("Only qualified one-thread float32 CPU execution is supported")
    if config["evaluation_steps"] != [0, 1, 5, 10, 25, 50, 100]:
        raise ValueError("Evaluation schedule differs from execution contract")
    resources = resource_guard(config, output.parent)
    output.mkdir(parents=True, exist_ok=False)
    initialize()
    tasks = {t["task_seed"]: t for t in read(ROOT / "03-Data/prepared/task-families.json")}
    jobs = select_jobs(config, qualification)
    inputs = [ROOT / "03-Data/prepared/task-families.json", ROOT / "03-Data/prepared/training-schedules.json",
              ROOT / "03-Data/prepared/job-plan.json", ROOT / "03-Data/config/pilot.json", config_path,
              Path(__file__).resolve(), ROOT / "06-Study-History/EXECUTION-2026-10-08.md"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    source = output / "snapshot"
    source.mkdir()
    for p in inputs:
        dest = source / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    manifest = dict(created_utc=now(), study=config["study"], stage="qualification" if qualification else config["stage"],
                    configuration=config, input_hashes=hashes, jobs=jobs, resources=resources,
                    environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
                                     platform=platform.platform(), cpu_threads=torch.get_num_threads(), gpu_used=False))
    write(output / "manifest.json", manifest)
    write(output / "status.json", dict(status="running", completed=0, planned=len(jobs), started_utc=now()))
    began = time.perf_counter()
    completed, failed = [], []
    for j in jobs:
        jobdir = output / j["id"]
        jobdir.mkdir()
        start = time.perf_counter()
        provenance = dict(run_id=output.name, task_seed=j["task_seed"], model_seed=j["model_seed"], history=j["history"], input_hashes=hashes)
        try:
            resource_guard(config, output)
            model, optimizer = train_base(tasks[j["task_seed"]], j, config, jobdir, provenance)
            run_updates(tasks[j["task_seed"]], model, optimizer, config, jobdir, provenance)
            elapsed = time.perf_counter()-start
            write(jobdir / "status.json", dict(status="complete", seconds=elapsed, finished_utc=now()))
            completed.append(j["id"])
            print(json.dumps(dict(job=j["id"], status="complete", seconds=round(elapsed, 2), done=len(completed), total=len(jobs))), flush=True)
            del model, optimizer
        except Exception as exc:
            failure = dict(job=j["id"], status="failed", error=str(exc), traceback=traceback.format_exc(), utc=now())
            write(jobdir / "status.json", failure)
            failed.append(failure)
            print(json.dumps(failure), flush=True)
            if isinstance(exc, (MemoryError, RuntimeError)):
                break
        write(output / "status.json", dict(status="running", completed=completed, failures=failed,
                                            elapsed_seconds=time.perf_counter()-began, updated_utc=now()))
    final = dict(status="complete" if len(completed) == len(jobs) and not failed else "incomplete",
                 completed=completed, failures=failed, planned=len(jobs), elapsed_seconds=time.perf_counter()-began,
                 maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, finished_utc=now())
    write(output / "status.json", final)
    print(json.dumps(final), flush=True)
    if final["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "03-Data/config/execution-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    execute(args.config, args.output, args.qualification)
