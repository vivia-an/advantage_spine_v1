#!/usr/bin/env python3
"""One cross-referenced evidence plate for concentration and support identity."""
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

import figstyle

figstyle.apply()
P = figstyle.PALETTE


def panel_title(ax, letter, title):
    ax.set_title(rf"$\bf{{{letter}}}$  {title}", loc="left", fontsize=8.6,
                 fontweight="bold", pad=6)


specificity_rows = list(csv.DictReader(
    open("audited_specificity_results.csv", encoding="utf-8")))
control_rows = list(csv.DictReader(
    open("audited_mask_controls.csv", encoding="utf-8")))

cross = [r for r in specificity_rows if r["comparison"] == "spine_dense"]
dense = [float(r["jaccard"]) for r in specificity_rows
         if r["comparison"] == "dense_dense"]
random_j = next(float(r["jaccard"]) for r in specificity_rows
                if r["comparison"] == "random")
seeds = [42, 43, 44]
matrix = np.array([
    [next(float(r["jaccard"]) for r in cross
          if int(r["left_seed"]) == s and int(r["right_seed"]) == d)
     for d in seeds]
    for s in seeds
])
matched = np.diag(matrix)
mismatched = matrix[~np.eye(3, dtype=bool)]
specificity = np.array([
    matrix[i, i] - np.delete(matrix[i], i).mean() for i in range(3)
])

fig = plt.figure(figsize=(7.35, 4.65))
gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.38],
                      height_ratios=[0.84, 1.0], wspace=.52, hspace=.56)

# A: concentration. Equal-length tracks make the retained-vs-residual split
# comparable without using two unrelated axes.
ax = fig.add_subplot(gs[0, 0])
labels = ["Coordinates", "Update energy"]
kept = [.393, .850]
colors = ["#c99e5a", P["accent"]]
for y, (label, value, color) in enumerate(zip(labels, kept, colors)):
    ax.barh(y, value, color=color, height=.42)
    ax.barh(y, 1-value, left=value, color="#e8ebef", height=.42)
    ax.text(value/2, y, f"{100*value:.1f}%", ha="center", va="center",
            color="white", fontsize=7.6, fontweight="bold")
    ax.text(value+(1-value)/2, y, f"{100*(1-value):.1f}%",
            ha="center", va="center", color=P["muted"], fontsize=6.7)
ax.set_yticks([0, 1], labels)
ax.set_xlim(0, 1)
ax.set_xlabel("fraction retained")
ax.invert_yaxis()
panel_title(ax, "A", "A small support carries most update energy")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

# B: full cross-seed matrix. The amber diagonal is the hypothesis test readers
# should see before looking at aggregate means.
ax = fig.add_subplot(gs[0, 1])
cmap = LinearSegmentedColormap.from_list(
    "paperd_overlap", [P["raw"], "#b7cbe1", P["blue"]])
ax.imshow(matrix, vmin=.25, vmax=.62, cmap=cmap, aspect="equal")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=7.5, color="white" if matrix[i, j] > .53 else P["body"],
                fontweight="bold" if i == j else "normal")
    ax.add_patch(Rectangle((i-.48, i-.48), .96, .96, fill=False,
                           edgecolor=P["accent"], linewidth=1.6))
ax.set_xticks(range(3), [f"Dense {s}" for s in seeds], rotation=20, ha="right")
ax.set_yticks(range(3), [f"Spine {s}" for s in seeds])
ax.tick_params(length=0)
panel_title(ax, "B", "Matched seeds form a clean diagonal")
for spine in ax.spines.values():
    spine.set_visible(False)

# C: aggregate structured nulls with raw pairwise values retained.
ax = fig.add_subplot(gs[1, 0])
groups = [matched, mismatched, np.array(dense), np.array([random_j])]
labels = ["Matched\nSpine–Dense", "Mismatched\nSpine–Dense",
          "Dense–Dense", "Independent\nrandom"]
colors = [P["accent"], P["blue"], P["green"], P["muted"]]
offsets = [np.array([-.07, 0, .07]), np.linspace(-.10, .10, 6),
           np.array([-.07, 0, .07]), np.array([0.])]
for i, (vals, color, off) in enumerate(zip(groups, colors, offsets)):
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0
    if len(vals) > 1:
        ax.errorbar(i, mu, yerr=sd, fmt="o", ms=6.2, capsize=3,
                    color=color, zorder=3)
    else:
        ax.scatter(i, mu, marker="D", s=30, color=color, zorder=3)
    ax.scatter(i + off, vals, s=17, color=color, alpha=.48, zorder=4)
    text = f"{mu:.3f}±{sd:.3f}" if len(vals) > 1 else f"{mu:.3f}"
    ax.text(i, mu+.032, text, ha="center", fontsize=6.5, color=color)
ax.text(.03, .965,
        f"matched − mismatched = {st.mean(specificity):.3f}±{st.stdev(specificity):.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
        color=P["edge"], fontweight="bold")
ax.set_xticks(range(4), labels)
ax.set_ylim(.22, .75)
ax.set_ylabel("Jaccard")
ax.grid(axis="y", ls=":", alpha=.32)
panel_title(ax, "C", "Alignment exceeds structured nulls")

# D: outcome controls in a compact dot-range chart.
ax = fig.add_subplot(gs[1, 1])
control_labels = [r["control"] for r in control_rows][::-1]
control_values = [float(r["mean4"]) for r in control_rows][::-1]
control_colors = [P["muted"], P["red"], P["blue"], P["green"],
                  "#c49a5a", P["accent"]]
y = range(len(control_labels))
ax.hlines(y, .33, control_values, color=control_colors, lw=2.2)
ax.scatter(control_values, y, color=control_colors, s=35,
           edgecolors=P["edge"], linewidths=.4, zorder=3)
for yi, value in zip(y, control_values):
    ax.text(value+.0025, yi, f"{value:.3f}", va="center", fontsize=6.8)
ax.axvline(.426, color=P["muted"], ls="--", lw=.9)
ax.text(.4245, 4.55, "Dense 1× reference", ha="right", va="center",
        fontsize=6.2, color=P["muted"])
ax.set_yticks(list(y), control_labels)
ax.set_xlim(.33, .463)
ax.set_xlabel("Mean4 at 20× (reported aggregate)")
ax.grid(axis="x", ls=":", alpha=.32)
panel_title(ax, "D", "Dynamic support beats arbitrary controls")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

figstyle.save(fig, "fig_support_evidence", pad=.025)
