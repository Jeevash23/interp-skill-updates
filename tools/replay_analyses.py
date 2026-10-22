"""Rerun the saved-output analyses and compare their outputs with the shipped files, byte for byte.

    python tools/replay_analyses.py          # after tools/unpack.py; NumPy and SciPy only; exit status 1 on a difference

Reruns phase 1's frozen analysis on its pilot and longer-training runs (from the run records) and the six post hoc and
descriptive analyses of 9 October 2026 (from phase 2's analyses, run records, the confirmation networks' initial and
base parameters and the frozen-readout runs). Each runs in a temporary copy, so the release itself is never written.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
P1 = "studies/1-feasibility-classification"
POSTHOC = [("posthoc_endpoints", "posthoc-endpoints"), ("posthoc_direction", "posthoc-direction"),
           ("posthoc_sample_size", "posthoc-sample-size"), ("posthoc_geometry", "posthoc-geometry"),
           ("posthoc_frozen_rate", "posthoc-frozen-rate"), ("duration_curves", "duration-curves")]
COPY = ["studies", "analysis"]


def steps(tmp: Path):
    """(command run from the copy's root, [(produced file, shipped file)])."""
    out = []
    for name in ("pilot", "long-final"):
        saved = RELEASE / P1 / f"04-Results/{name}-analysis-v1"
        dest = tmp / f"phase1-{name}"
        out.append(([f"{P1}/02-Code/analyze.py", "--run", f"{P1}/04-Results/{name}-v1", "--output", str(dest)],
                    [(dest / p.name, saved / p.name) for p in sorted(saved.iterdir())]))
    for script, result in POSTHOC:
        out.append(([f"analysis/{script}.py"],
                    [(tmp / "copy" / f"analysis/results/{result}.json", RELEASE / f"analysis/results/{result}.json")]))
    return out


def main():
    if not (RELEASE / P1 / "04-Results/pilot-v1").is_dir():
        sys.exit("saved outputs missing; run tools/unpack.py (make unpack) first")
    failed = n = 0
    with tempfile.TemporaryDirectory(prefix="replay-") as t:
        tmp = Path(t)
        work = tmp / "copy"
        for rel in COPY:
            shutil.copytree(RELEASE / rel, work / rel, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
        for cmd, pairs in steps(tmp):
            n += 1
            run = subprocess.run([sys.executable, "-B", *cmd], cwd=work, capture_output=True, text=True)
            if run.returncode != 0:
                failed += 1
                print(f"FAIL  {' '.join(cmd[:1])} exited with {run.returncode}")
                print("      " + (run.stdout + run.stderr).strip().replace("\n", "\n      ")[-2000:])
                continue
            same = [p for p, s in pairs if p.is_file() and p.read_bytes() == s.read_bytes()]
            failed += len(same) != len(pairs)
            label = cmd[0] + (" " + cmd[2] if len(cmd) > 2 else "")
            print(f"{'identical' if len(same) == len(pairs) else 'DIFFERENT'}  {len(same)}/{len(pairs)} output "
                  f"file(s)  ({label})")
    print(f"{n} analyses rerun from the saved records: "
          + ("every output identical to the shipped files" if not failed else f"{failed} problem(s)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
