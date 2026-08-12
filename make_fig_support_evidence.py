#!/usr/bin/env python3
"""Independent, LaTeX-composed panels for support evidence (Figure 4)."""
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

# Concentration. Equal-length tracks compare retained and residual fractions.
fig, ax = plt.subplots(figsize=(3.25, 1.72))
labels = ["Adaptive support", "Direction energy"]
kept = [.393, .850]
colors = ["#c99e5a", P["accent"]]
for y, (label, value, color) in enumerate(zip(labels, kept, colors)):
    ax.barh(y, value, color=color, height=.42)
    ax.barh(y, 1-value, left=value, color="#e8ebef", height=.42)
    ax.text(value/2, y, f"{100*value:.1f}%", ha="center", va="center",
            color="white", fontsize=7.7, fontweight="bold")
    ax.text(value+(1-value)/2, y, f"{100*(1-value):.1f}%",
            ha="center", va="center", color=P["muted"], fontsize=7.0)
ax.set_yticks([0, 1], labels)
ax.set_xlim(0, 1)
ax.set_xlabel("Fraction retained")
ax.invert_yaxis()
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=.45)
figstyle.save(fig, "fig_support_concentration", pad=.015)
plt.close(fig)

# Full cross-seed matrix. The diagonal is the seed-specific identity test.
fig, ax = plt.subplots(figsize=(3.25, 1.72))
cmap = LinearSegmentedColormap.from_list(
    "paperd_overlap", [P["raw"], "#b7cbe1", P["blue"]])
ax.imshow(matrix, vmin=.25, vmax=.62, cmap=cmap, aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=8.0, color="white" if matrix[i, j] > .53 else P["body"],
                fontweight="bold" if i == j else "normal")
    ax.add_patch(Rectangle((i-.48, i-.48), .96, .96, fill=False,
                           edgecolor=P["accent"], linewidth=1.6))
ax.set_xticks(range(3), [f"Dense {s}" for s in seeds])
ax.set_yticks(range(3), [f"Spine {s}" for s in seeds])
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout(pad=.35)
figstyle.save(fig, "fig_support_matrix", pad=.015)
plt.close(fig)

# Structured nulls with all pairwise values visible.
fig, ax = plt.subplots(figsize=(3.25, 2.05))
groups = [matched, mismatched, np.array(dense), np.array([random_j])]
labels = ["Matched", "Mismatched", "Dense–Dense", "Random"]
colors = [P["accent"], P["blue"], P["green"], P["muted"]]
offsets = [np.array([-.07, 0, .07]), np.linspace(-.10, .10, 6),
           np.array([-.07, 0, .07]), np.array([0.])]
for i, (vals, color, off) in enumerate(zip(groups, colors, offsets)):
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0
    if len(vals) > 1:
        ax.errorbar(i, mu, yerr=sd, fmt="o", ms=6.0, capsize=3,
                    color=color, zorder=3)
    else:
        ax.scatter(i, mu, marker="D", s=30, color=color, zorder=3)
    ax.scatter(i + off, vals, s=17, color=color, alpha=.48, zorder=4)
    text = f"{mu:.3f}±{sd:.3f}" if len(vals) > 1 else f"{mu:.3f}"
    ax.text(i, mu+.030, text, ha="center", fontsize=7.0, color=color)
ax.text(.03, .97,
        f"matched − mismatched = {st.mean(specificity):.3f}±{st.stdev(specificity):.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
        color=P["edge"], fontweight="bold")
ax.set_xticks(range(4), labels)
ax.set_ylim(.22, .75)
ax.set_ylabel("Jaccard")
ax.grid(axis="y", ls=":", alpha=.32)
fig.tight_layout(pad=.45)
figstyle.save(fig, "fig_support_nulls", pad=.015)
plt.close(fig)

# Density-matched outcome controls.
fig, ax = plt.subplots(figsize=(3.25, 2.05))
control_labels = [r["control"] for r in control_rows][::-1]
control_values = [float(r["mean"]) for r in control_rows][::-1]
control_sds = [float(r["sample_std"]) for r in control_rows][::-1]
control_colors = [P["muted"], P["red"], P["blue"], P["green"],
                  "#c49a5a", P["accent"]]
y = range(len(control_labels))
ax.hlines(y, .33, control_values, color=control_colors, lw=2.2)
ax.scatter(control_values, y, color=control_colors, s=34,
           edgecolors=P["edge"], linewidths=.4, zorder=3)
for yi, value, sd, color in zip(y, control_values, control_sds, control_colors):
    ax.errorbar(value, yi, xerr=sd, fmt="none", ecolor=color,
                capsize=2.5, lw=.8, zorder=2)
    ax.text(value+.0025, yi, f"{value:.3f}", va="center", fontsize=7.0)
ax.axvline(.426, color=P["muted"], ls="--", lw=.9)
ax.text(.4245, 4.55, "Dense 1×", ha="right", va="center",
        fontsize=6.7, color=P["muted"])
ax.set_yticks(list(y), control_labels)
ax.set_xlim(.33, .463)
ax.set_xlabel("Mean accuracy at 20×")
ax.grid(axis="x", ls=":", alpha=.32)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout(pad=.45)
figstyle.save(fig, "fig_support_controls", pad=.015)
plt.close(fig)
