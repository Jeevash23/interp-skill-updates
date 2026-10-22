"""Figures of the manuscript, drawn from saved outputs only (paper/figures/*.pdf).

fig-design.pdf         the two training histories (from the saved schedules) and the network
fig-families.pdf       per-family history differences in unchanged-skill error in both confirmations, with intervals
fig-interventions.pdf  unchanged- and changed-skill error after 200 update steps, for every update condition
fig-trajectories.pdf   mean error during unrestricted updates (200 steps) and readout-frozen updates (1,000 steps)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "build/mpl"))
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

P2 = PROJECT / "studies/2-controlled-histories"
CONF = P2 / "analysis/confirmation"
AR = PROJECT / "analysis/results"
OUT = PAPER / "figures"
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#eeede9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (light surface)
ARCHS = ("linear", "leaky")
CONFIG = json.loads((P2 / "protocol/confirmation.json").read_text())
UPDATE_STEPS = CONFIG["update_steps"]
ARCH_LABEL = {"linear": "Linear", "leaky": "Leaky ReLU (slope 0.9)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "mathtext.fontset": "stix",
})


def read(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ design
def design():
    sched = read(P2 / "data/confirmation.json")["schedules"]
    fig = plt.figure(figsize=(6.5, 2.15))
    ax = fig.add_axes([0.0, 0.0, 0.67, 1.0])
    n = len(sched["joint"]) - 1
    ax.set_xlim(-5.6, n + 0.3)
    ax.set_ylim(-1.55, 3.0)
    ax.axis("off")
    colors = {"f": BLUE, "g": ORANGE, "fg": "#b9b8b2"}
    ax.text(-5.6, 2.95, "a", fontsize=10, fontweight="bold", va="top")
    for row, (hist, name) in enumerate((("independent", "Independent"), ("joint", "Joint"))):
        y = 1.75 - 1.6 * row
        order = sched[hist]
        counts = defaultdict(int)
        ax.text(-0.4, y, name, ha="right", va="center", fontsize=8.5)
        for i, (old, new) in enumerate(zip(order, order[1:])):
            kind = "fg" if old[0] != new[0] and old[1] != new[1] else "f" if old[0] != new[0] else "g"
            counts[kind] += 1
            ax.add_patch(FancyBboxPatch((i + 0.08, y - 0.3), 0.84, 0.6, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=colors[kind], ec="white", lw=0.8))
            if kind != "fg":
                ax.text(i + 0.5, y, kind, ha="center", va="center", fontsize=8.5, color="white", style="italic")
        ax.text(0.05, y - 0.45, f"$f$ alone {counts['f']},  $g$ alone {counts['g']},  both {counts['fg']}",
                ha="left", va="top", fontsize=8.5, color=INK2)
    ax.plot([n - 4 + 0.08, n - 0.08], [2.25, 2.25], color=INK2, lw=0.8)
    ax.text(n - 2, 2.32, "shared", ha="center", va="bottom", fontsize=8.5, color=INK2)
    for kind, x, text in (("f", 0.1, "$f$ changes"), ("g", 6.6, "$g$ changes"), ("fg", 13.1, "both change")):
        ax.add_patch(FancyBboxPatch((x, -1.42), 0.6, 0.34, boxstyle="round,pad=0,rounding_size=0.08", fc=colors[kind], ec="none"))
        ax.text(x + 0.85, -1.25, text, va="center", fontsize=8.5)

    ax = fig.add_axes([0.68, 0.0, 0.32, 1.0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.0, 9.85, "b", fontsize=10, fontweight="bold", va="top")
    inputs_f = [(1.3, 8.3 - 0.55 * i) for i in range(4)]
    inputs_g = [(1.3, 5.4 - 0.55 * i) for i in range(4)]
    hidden = [(5.0, 8.5 - 0.88 * i) for i in range(6)]
    out = (8.6, 6.3)
    for group, color in ((inputs_f, BLUE), (inputs_g, ORANGE)):
        for (x0, y0) in group:
            for (x1, y1) in hidden:
                ax.plot([x0, x1], [y0, y1], color=color, lw=0.35, alpha=0.6)
    for (x1, y1) in hidden:
        ax.plot([x1, out[0]], [y1, out[1]], color=INK, lw=1.3)
    for group, color in ((inputs_f, BLUE), (inputs_g, ORANGE)):
        for (x, y) in group:
            ax.add_patch(Circle((x, y), 0.2, fc=color, ec="white", lw=0.6))
    for (x, y) in hidden:
        ax.add_patch(Circle((x, y), 0.24, fc="white", ec=INK2, lw=0.7))
    ax.add_patch(Circle(out, 0.3, fc="white", ec=INK, lw=0.9))
    ax.text(0.55, 7.47, "$a$", ha="center", va="center", fontsize=8.5)
    ax.text(0.55, 4.57, "$b$", ha="center", va="center", fontsize=8.5)
    ax.text(5.0, 9.05, f"{CONFIG['width']} hidden units", ha="center", fontsize=8.5)
    ax.text(9.15, 6.3, "$\\hat y$", ha="left", va="center", fontsize=8.5)
    ax.text(7.2, 8.05, "readout $w$", ha="center", fontsize=8.5)
    ax.text(0.2, 0.1, "$\\hat y = w^{\\top}\\phi(W_a x_a + W_b x_b)$.\nAn update on $f$ labels leaves $W_b$\nunchanged, so the $g$ outputs move\nonly through $w$: $\\Delta \\hat g = H_g\\,\\Delta w$.",
            ha="left", va="bottom", fontsize=8.5, color=INK2, linespacing=1.35)
    save(fig, "fig-design.pdf")


# ------------------------------------------------------------------------------------------------ families
def families():
    s = read(CONF / "summary.json")
    post = read(AR / "posthoc-endpoints.json")["architectures"]
    s3 = read(PROJECT / "studies/3-larger-confirmation/results/summary.json")
    dev = {"linear": read(P2 / "analysis/development/summary.json")["primary"]["linear"]["effects"]["stable"]["mean"],
           "leaky": read(P2 / "analysis/amendment/summary.json")["primary"]["leaky"]["effects"]["stable"]["mean"]}
    fig, ax = plt.subplots(figsize=(6.5, 3.0), layout="constrained")
    margin = CONFIG["preservation_effect_margin"]
    ax.axvspan(-margin, margin, color=BAND, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
    rng = np.random.default_rng(7)
    labels = []
    for i, arch in enumerate(ARCHS):
        color = [BLUE, ORANGE][i]
        for j, (study, n) in enumerate((("first", s3["secondary"][arch]["pooled_with_phase2"]["phase2_families"]),
                                        ("larger", s3["families"]))):
            row = 2 * i + j
            if study == "first":
                e = s["primary"][arch]["effects"]["stable"]
                vals, low, high, m = np.array(list(e["family_differences"].values())), e["low"], e["high"], e["mean"]
            else:
                e = s3["primary"][arch]
                vals, low, high, m = np.array(list(e["family_differences"].values())), e["low"], e["high"], e["mean"]
            y = row - 0.12 + rng.uniform(-0.14, 0.14, len(vals))
            ax.scatter(vals, y, s=8, color=color, alpha=0.8 if study == "first" else 0.55, lw=0, zorder=2)
            yc = row + 0.22
            ax.plot([low, high], [yc] * 2, color=INK, lw=1.2, zorder=3)
            ax.plot([m], [yc], "o", color=INK, ms=3.5, zorder=4)
            if study == "first":
                ax.plot([post[arch]["two_way"]["low"], post[arch]["two_way"]["high"]], [yc + 0.11] * 2, color=MUTED,
                        lw=2.2, solid_capstyle="butt", zorder=3)
                ax.plot([dev[arch]], [yc], marker="v", ms=5, mfc="white", mec=INK, mew=0.8, ls="none", zorder=4)
            labels.append(f"{ARCH_LABEL[arch]}\n{'first' if study == 'first' else 'larger'} confirmation ({n})")
    ax.set_yticks(range(4), labels)
    ax.set_ylim(3.48, -0.45)
    ax.set_xlabel("Independent minus joint history: unchanged-skill error after the update (negative favours independent)")
    ax.tick_params(axis="y", length=0)
    ax.text(margin + 0.006, -0.42, f"margin $\\pm${margin}", ha="left", va="top", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=INK, lw=1.2, marker="o", ms=3.5, label="mean and prespecified interval"),
               plt.Line2D([], [], color=MUTED, lw=2.2, label="two-interval correction (97.5%, post hoc)"),
               plt.Line2D([], [], marker="v", ms=5, mfc="white", mec=INK, ls="none", label="development estimate")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=7.5, handlelength=1.6)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig-families.pdf")


# ------------------------------------------------------------------------------------------------ interventions
def family_interval(records, arch, select, metric):
    groups = defaultdict(list)
    for r in records:
        if r["architecture"] == arch and select(r["arm"]):
            groups[r["task_seed"]].append(r[metric])
    v = np.array([np.mean(x) for x in groups.values()])
    return v.mean(), student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def interventions():
    records = read(CONF / "records.json")
    arms = [("unrestricted", "Unrestricted"), ("freeze_readout", "Readout frozen"),
            ("random", f"Random {CONFIG['width']}-weight masks (3)"), ("readout_only", "Readout only"),
            ("normalized", "Fixed-length steps"), ("no_update", "No update"), ("old_primitive", "Old values")]
    sel = {a: ((lambda x: x.startswith("random_freeze_")) if a == "random" else (lambda x, a=a: x == a)) for a, _ in arms}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3), sharey=True, layout="constrained")
    for ax, metric, title in zip(axes, ("stable", "changed"),
                                 (f"a  Unchanged skill: error after {UPDATE_STEPS} steps",
                                  f"b  Changed skill: error after {UPDATE_STEPS} steps")):
        for i, arch in enumerate(ARCHS):
            ys = np.arange(len(arms)) + (i - 0.5) * 0.3
            for y, (arm, _) in zip(ys, arms):
                m, h = family_interval(records, arch, sel[arm], metric)
                ax.errorbar([m], [y], xerr=[h], fmt="o", ms=3.6, color=[BLUE, ORANGE][i], elinewidth=1, capsize=0,
                            label=ARCH_LABEL[arch] if y == ys[0] else None)
        ax.set_yticks(range(len(arms)), [a for _, a in arms])
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Root-mean-square error (functions have RMS 1)")
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.05, 1.45)
    axes[1].set_xlim(-0.05, 1.55)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "fig-interventions.pdf")


# ------------------------------------------------------------------------------------------------ trajectories
def trajectories():
    curves = read(CONF / "curves.json")
    dur = read(AR / "duration-curves.json")
    steps = np.array(dur["steps"])
    last = int(steps[-1])
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 2.9), sharex=True, layout="constrained")
    for row, arch in enumerate(ARCHS):
        un = np.mean([np.array(c["mean"]) for c in curves if c["architecture"] == arch and c["arm"] == "unrestricted"], axis=0)
        fr = np.mean([np.array(c["mean"]) for c in curves if c["architecture"] == arch and c["arm"] == "freeze_readout"], axis=0)
        long = {k: np.mean([dur["curves"][f"{arch}/{h}"][k] for h in ("independent", "joint")], axis=0)
                for k in ("changed", "stable")}
        assert np.max(np.abs(long["changed"][:11] - fr[0][::20])) < 1e-8
        for col, (k, idx) in enumerate((("changed", 0), ("stable", 1))):
            ax = axes[row, col]
            x200 = np.arange(len(un[idx]))
            ax.plot(x200, un[idx], color=BLUE, lw=1.6, label=f"unrestricted ({UPDATE_STEPS} steps)")
            ax.plot(steps, long[k], color=ORANGE, lw=1.6, label=f"readout frozen ({last:,} steps)")
            ax.axvline(UPDATE_STEPS, color=MUTED, lw=0.6, ls=(0, (3, 2)))
            if k == "changed":
                ax.set_yscale("log")
                ax.set_ylim(5e-5, 3)
                ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
                ax.axhline(0.1, color=MUTED, lw=0.6)
                ax.axhline(0.01, color=MUTED, lw=0.6)
                ax.text(last, 0.115, "0.10", ha="right", va="bottom", fontsize=6.5, color=INK2)
                ax.text(last, 0.0115, "0.01", ha="right", va="bottom", fontsize=6.5, color=INK2)
            else:
                ax.set_ylim(-0.03, 0.8)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{'ab'[col] if row == 0 else 'cd'[col]}  {ARCH_LABEL[arch]}: "
                         f"{'changed' if k == 'changed' else 'unchanged'}-skill error")
            if row == 1:
                ax.set_xlabel("Update step")
    axes[0, 1].legend(loc="center right")
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE (log scale)")
    for ax in axes[:, 1]:
        ax.set_ylabel("RMSE")
    axes[1, 0].set_xlim(0, last)
    save(fig, "fig-trajectories.pdf")


if __name__ == "__main__":
    design()
    families()
    interventions()
    trajectories()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
