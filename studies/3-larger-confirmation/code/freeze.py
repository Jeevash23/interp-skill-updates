"""Freeze Study 3 before any run: write FREEZE.json with the SHA-256 of the protocol, predictions, configuration,
sample-size and seed records, task data and every code file, plus the seeds and the UTC time. Refuses to run twice.

    python -B code/freeze.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PHASE2 = STUDY.parent / "2-controlled-histories"
FILES = ["PROTOCOL.md", "PREDICTIONS.md", "protocol/predictions.json", "protocol/config.json",
         "protocol/sample-size.json", "protocol/seed-check.json", "data/tasks.json"]
COPIED = ["study.py", "amended_study.py", "analyze.py"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out = STUDY / "FREEZE.json"
    if out.exists():
        sys.exit("FREEZE.json exists: Study 3 is already frozen")
    if any((STUDY / "runs").glob("*")) if (STUDY / "runs").exists() else False:
        sys.exit("a run folder already exists; the freeze must come first")
    files = FILES + sorted(str(p.relative_to(STUDY)) for p in (STUDY / "code").glob("*.py")) + ["code/pipeline.sh"]
    config = json.loads((STUDY / "protocol/config.json").read_text())
    tasks = json.loads((STUDY / "data/tasks.json").read_text())
    sample = json.loads((STUDY / "protocol/sample-size.json").read_text())
    seeds_ok = json.loads((STUDY / "protocol/seed-check.json").read_text())["status"] == "passed"
    phase2_freeze = json.loads((PHASE2 / "protocol/CONFIRMATION-FREEZE.json").read_text())["files"]
    copied = {f"code/{f}": dict(sha256=sha(STUDY / "code" / f), phase2_path=f"studies/2-controlled-histories/code/{f}",
                                phase2_frozen_sha256=phase2_freeze[f"code/{f}"]) for f in COPIED}
    if not all(v["sha256"] == v["phase2_frozen_sha256"] for v in copied.values()):
        sys.exit("a copied Phase 2 file differs from its frozen version")
    if not seeds_ok:
        sys.exit("protocol/seed-check.json does not pass")
    if [t["seed"] for t in tasks["tasks"]] != config["task_seeds"]:
        sys.exit("task data does not match the configured seeds")
    record = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        study="AI-07 Study 3: larger confirmation of Phase 2",
        outcomes_observed=False, runs_started=False,
        families=len(config["task_seeds"]), task_seeds=dict(first=config["task_seeds"][0], last=config["task_seeds"][-1],
                                                            count=len(config["task_seeds"])),
        initialization_seeds=config["initialization_seeds"], networks=len(config["task_seeds"]) * len(
            config["initialization_seeds"]) * len(config["architectures"]) * len(config["histories"]),
        primary=dict(endpoint="independent minus joint unchanged-skill error after 200 unrestricted steps, per family",
                     interval="two-sided Student t, 97.5% per architecture (two intervals, Bonferroni)",
                     margin=0.05, sample_size=sample["families"], sample_size_power=sample["probability_inside_at_n"]),
        copied_phase2_code=copied,
        environment=dict(python=sys.version.split()[0], platform=platform.platform()),
        files={rel: sha(STUDY / rel) for rel in files})
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "files"} | dict(files=len(record["files"])), indent=2))


if __name__ == "__main__":
    main()
