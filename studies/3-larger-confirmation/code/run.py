"""Study 3 runner: train and update every planned network with the Phase 2 engine, in parallel, resumably.

The science code is Phase 2's, unchanged: `study.py` (task construction, base training, the nine update conditions)
and `amended_study.py` (the leaky ReLU network) are byte-identical copies of
`../../2-controlled-histories/code/` (hashes in `../FREEZE.json`). This file only replaces the engine's sequential
`execute` loop: it runs the same jobs, in the same order of definition, in worker processes, each job writing the
same files as Phase 2 into its own folder. A job's result depends only on its task, seed, architecture and history,
so the order and the number of workers do not change any output.

Resuming: a job whose folder holds a `status.json` with status `complete` is kept; any other existing job folder is
moved, unchanged, to `<output>/_incomplete/` and the job is run again. The run's `manifest.json` must match the
current configuration, tasks and sources, or the runner refuses to continue.

    python -B code/run.py --config protocol/config.json --tasks data/tasks.json --output runs/confirmation \
        --workers 6 --freeze FREEZE.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
sys.path.insert(0, str(CODE))
import study as s  # noqa: E402  (Phase 2 engine, unchanged copy)
import amended_study  # noqa: E402  (adds the leaky ReLU network)

SOURCES = ["code/study.py", "code/amended_study.py", "code/run.py", "PROTOCOL.md", "PREDICTIONS.md"]
_STATE: dict = {}


def job_id(job: dict) -> str:
    return f"t{job['task_seed']}-s{job['model_seed']}-{job['architecture']}-{job['history']}"


def plan(config: dict, tasks: list) -> list:
    """The engine's job list, in its order."""
    return [dict(task_seed=t["seed"], model_seed=seed, architecture=arch, history=h)
            for t in tasks for seed in config["initialization_seeds"]
            for arch in config["architectures"] for h in config["histories"]]


def _worker_init(config_path: str, tasks_path: str) -> None:
    s.initialize(True)
    s.network = amended_study.network          # as amended_study.py does in run mode
    config, bundle = s.read(config_path), s.read(tasks_path)
    rows, values = s.rows_and_inputs()
    _STATE.update(config=config, tasks={t["seed"]: t for t in bundle["tasks"]}, x=s.torch.tensor(values),
                  train=s.np.array([r["split"] == "train" for r in rows]))


def _run_job(job: dict, output: str) -> tuple:
    config, folder = _STATE["config"], Path(output) / job_id(job)
    folder.mkdir()
    begin = time.perf_counter()
    try:
        s.guard(config, Path(output))
        task = _STATE["tasks"][job["task_seed"]]
        model = s.train_base(task, job["model_seed"], job["architecture"], job["history"], config, folder,
                             _STATE["x"], _STATE["train"])
        s.updates(task, job["model_seed"], model, config, folder, _STATE["x"])
        record = dict(status="complete", job=job, seconds=time.perf_counter() - begin, finished_utc=s.now())
    except Exception as exc:  # recorded, never hidden
        record = dict(status="failed", job=job, error=str(exc), traceback=traceback.format_exc(), utc=s.now())
    s.write(folder / "status.json", record)
    return job_id(job), record


def check_freeze(freeze_path: Path) -> None:
    freeze = s.read(freeze_path)
    changed = [rel for rel, digest in freeze["files"].items() if s.sha(STUDY / rel) != digest]
    if changed:
        sys.exit(f"refusing to run: frozen files changed since {freeze['created_utc']}: {changed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--freeze", type=Path, help="refuse to run unless every file it lists has its recorded hash")
    a = ap.parse_args()
    config_path, tasks_path, output = a.config.resolve(), a.tasks.resolve(), a.output.resolve()
    if a.freeze:
        check_freeze(a.freeze.resolve())
    s.initialize(True)
    config, bundle = s.read(config_path), s.read(tasks_path)
    tasks = bundle["tasks"]
    assert bundle["configuration_sha256"] == s.sha(config_path)
    assert [t["seed"] for t in tasks] == config["task_seeds"]
    for task in tasks:
        s.check_design(task, config)
    jobs = plan(config, tasks)
    sources = [STUDY / p for p in SOURCES] + [config_path, tasks_path] + ([a.freeze.resolve()] if a.freeze else [])
    hashes = {str(p.relative_to(STUDY)): s.sha(p) for p in sources}
    resources = s.guard(config, output.parent if not output.exists() else output)
    if output.exists():
        manifest = s.read(output / "manifest.json")
        if manifest["jobs"] != jobs or manifest["source_hashes"] != hashes or manifest["config"] != config:
            sys.exit("refusing to resume: the existing run was started with different jobs, sources or configuration")
    else:
        output.mkdir(parents=True)
        for p in sources:
            dest = output / "snapshot" / p.relative_to(STUDY)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        s.write(output / "manifest.json", dict(
            stage=config["stage"], created_utc=s.now(), config=config, source_hashes=hashes, jobs=jobs,
            resources=resources, workers=a.workers,
            environment=dict(python=sys.version, executable=sys.executable, numpy=s.np.__version__,
                             torch=s.torch.__version__, dtype="float64", threads=1, gpu_used=False)))
    (STUDY / "logs").mkdir(exist_ok=True)
    (STUDY / "logs/run.pid").write_text(f"{os.getpid()}\n")
    done, failed, todo = {}, [], []
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for job in jobs:
        folder = output / job_id(job)
        if (folder / "status.json").exists() and s.read(folder / "status.json")["status"] == "complete":
            done[job_id(job)] = True
            continue
        if folder.exists():
            (output / "_incomplete").mkdir(exist_ok=True)
            folder.rename(output / "_incomplete" / f"{job_id(job)}-{stamp}")
        todo.append(job)
    print(json.dumps(dict(event="start", utc=s.now(), planned=len(jobs), already_complete=len(done),
                          to_run=len(todo), workers=a.workers, pid=os.getpid())), flush=True)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=context, initializer=_worker_init,
                             initargs=(str(config_path), str(tasks_path))) as pool:
        pending = {pool.submit(_run_job, job, str(output)) for job in todo}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                jid, record = fut.result()
                if record["status"] == "complete":
                    done[jid] = True
                else:
                    failed.append(record)
                elapsed = time.perf_counter() - started
                rate = (len(done) - (len(jobs) - len(todo))) / elapsed if elapsed > 0 else 0
                remaining = len(jobs) - len(done) - len(failed)
                print(json.dumps(dict(job=jid, status=record["status"], done=len(done), failed=len(failed),
                                      total=len(jobs), elapsed_min=round(elapsed / 60, 1),
                                      eta_min=round(remaining / rate / 60, 1) if rate else None)), flush=True)
                s.write(output / "status.json", dict(status="running", completed=[job_id(j) for j in jobs
                                                                                 if job_id(j) in done],
                                                     failures=failed, planned=len(jobs),
                                                     elapsed_seconds=elapsed))
    final = dict(status="complete" if len(done) == len(jobs) and not failed else "incomplete",
                 completed=[job_id(j) for j in jobs if job_id(j) in done], failures=failed, planned=len(jobs),
                 seconds=time.perf_counter() - started, workers=a.workers,
                 job_seconds=sum(s.read(output / job_id(j) / "status.json")["seconds"] for j in jobs if job_id(j) in done),
                 maximum_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, finished_utc=s.now())
    s.write(output / "status.json", final)
    print(json.dumps(dict(event="finish", **{k: v for k, v in final.items() if k != "completed"})), flush=True)
    if final["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
