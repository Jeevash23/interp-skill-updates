"""Descriptive summary (9 October 2026) of the 1,000-step fixed-readout updates (the duration control, declared and
run after the confirmation results were seen).

Reads the saved predictions (every 20 steps) in studies/2-controlled-histories/runs/duration-control/*.npz and the
per-trajectory records, and the unrestricted first-passage times from the confirmation analysis. Writes
analysis/results/duration-curves.json: mean changed-skill, unchanged-skill and unseen-combination errors at each
recorded step for each architecture and history, and the steps needed to reach changed-skill error 0.10 and 0.01
under the fixed readout compared with unrestricted updates.
"""
from __future__ import annotations

import numpy as np

from common import ARCHS, CONF, HISTORIES, P2, read, write

RUN = P2 / "runs/duration-control"


def heldout_rows():
    rows = [("combine", a, b) for a in range(8) for b in range(8)]
    return np.array([16 + i for i, (_, a, b) in enumerate(rows) if (b - a) % 8 in (0, 1)])


def main() -> None:
    held = heldout_rows()
    assert len(held) == 16
    records = read(RUN / "records.json")
    summary = read(RUN / "summary.json")
    progress = read(CONF / "summary.json")["matched_progress"]
    steps = None
    curves = {}
    for arch in ARCHS:
        for hist in HISTORIES:
            acc = []
            for r in records:
                if r["architecture"] != arch or r["history"] != hist or r["branch"] != "f":
                    continue
                name = f"t{r['task_seed']}-s{r['model_seed']}-{arch}-{hist}.npz"
                with np.load(RUN / name) as z:
                    p, targets = z["predictions"], z["targets"]
                    steps = z["steps"].tolist() if steps is None else steps
                for bi in range(2):
                    changed = np.arange(8) if bi == 0 else np.arange(8, 16)
                    stable = np.arange(8, 16) if bi == 0 else np.arange(8)
                    sq = (p[bi] - targets[bi]) ** 2
                    acc.append(np.stack([np.sqrt(sq[:, changed].mean(1)), np.sqrt(sq[:, stable].mean(1)),
                                         np.sqrt(sq[:, held].mean(1) / 2)]))
            acc = np.array(acc)
            curves[f"{arch}/{hist}"] = dict(trajectories=len(acc), changed=acc[:, 0].mean(0).tolist(),
                                            stable=acc[:, 1].mean(0).tolist(), transfer=acc[:, 2].mean(0).tolist())
    passage = {}
    for arch in ARCHS:
        sub = [r for r in records if r["architecture"] == arch]
        f10 = [r["first_0_10"] for r in sub]
        f01 = [r["first_0_01"] for r in sub]
        unrestricted = [progress[arch][h]["unrestricted"]["step_mean"] for h in HISTORIES]
        passage[arch] = dict(trajectories=len(sub), reached_0_10=sum(v is not None for v in f10),
                             reached_0_01=sum(v is not None for v in f01),
                             mean_steps_0_10=float(np.mean(f10)), max_steps_0_10=int(np.max(f10)),
                             mean_steps_0_01=float(np.mean(f01)), max_steps_0_01=int(np.max(f01)),
                             unrestricted_mean_steps_0_10=float(np.mean(unrestricted)),
                             slowdown_0_10=float(np.mean(f10) / np.mean(unrestricted)),
                             endpoint_changed_mean=float(np.mean([r["changed"] for r in sub])),
                             endpoint_changed_max=float(np.max([r["changed"] for r in sub])),
                             endpoint_transfer_max=float(np.max([r["transfer"] for r in sub])),
                             endpoint_stable_drift_max=float(np.max([r["stable_drift"] for r in sub])))
    # Agreement with the duration control's own summary.
    for arch in ARCHS:
        for hist in HISTORIES:
            row = summary["descriptive"][arch][hist]
            sub = [r for r in records if r["architecture"] == arch and r["history"] == hist]
            assert abs(row["progress"]["first_0_10"]["mean_steps"] - np.mean([r["first_0_10"] for r in sub])) < 1e-9
            assert abs(curves[f"{arch}/{hist}"]["changed"][-1] - row["endpoint"]["changed"]["mean"]) < 1e-12
    out = dict(descriptive=True, post_confirmation=True, created="2026-10-09", steps=steps, curves=curves,
               first_passage=passage, all_trajectories=len(records))
    path = write("duration-curves.json", out)
    for arch, v in passage.items():
        print(f"{arch:6s} 0.10 reached {v['reached_0_10']}/{v['trajectories']} mean {v['mean_steps_0_10']:.1f} max {v['max_steps_0_10']}; "
              f"0.01 mean {v['mean_steps_0_01']:.1f} max {v['max_steps_0_01']}; unrestricted {v['unrestricted_mean_steps_0_10']:.1f}; "
              f"slowdown {v['slowdown_0_10']:.2f}")
    print("wrote", path.relative_to(path.parents[2]))


if __name__ == "__main__":
    main()
