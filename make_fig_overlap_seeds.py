#!/usr/bin/env python3
"""Seed-specific Spine/dense-emergent alignment and structured nulls."""
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
rows = list(csv.DictReader(open("audited_specificity_results.csv", encoding="utf-8")))
cross = [r for r in rows if r["comparison"] == "spine_dense"]
dense = [float(r["jaccard"]) for r in rows if r["comparison"] == "dense_dense"]
random_j = next(float(r["jaccard"]) for r in rows if r["comparison"] == "random")
seeds = [42, 43, 44]
matrix = np.array([[next(float(r["jaccard"]) for r in cross
                         if int(r["left_seed"]) == s and int(r["right_seed"]) == d)
                    for d in seeds] for s in seeds])
matched = np.diag(matrix)
mismatched = matrix[~np.eye(3, dtype=bool)]
specificity = np.array([matrix[i, i] - np.delete(matrix[i], i).mean()
                        for i in range(3)])

def fmt3(value):
    """Three decimals with conventional half-up behavior for audited labels."""
    return f"{float(value) + 1e-10:.3f}"

fig = plt.figure(figsize=(6.8, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.48], wspace=.42)

# A: the complete cross-seed matrix makes the diagonal separation visible.
ax = fig.add_subplot(gs[0, 0])
cmap = LinearSegmentedColormap.from_list("paperd_overlap", [P["raw"], "#b7cbe1", P["blue"]])
ax.imshow(matrix, vmin=.25, vmax=.62, cmap=cmap, aspect="equal")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                fontsize=7.4, color="white" if matrix[i,j] > .53 else P["body"],
                fontweight="bold" if i == j else "normal")
    ax.add_patch(Rectangle((i-.48, i-.48), .96, .96, fill=False,
                           edgecolor=P["accent"], linewidth=1.5))
ax.set_xticks(range(3), [f"Dense {s}" for s in seeds], rotation=25, ha="right")
ax.set_yticks(range(3), [f"Spine {s}" for s in seeds])
ax.tick_params(length=0)
ax.set_title("A  Full cross-seed Jaccard", loc="left", fontsize=8.4, fontweight="bold")
for spine in ax.spines.values(): spine.set_visible(False)

# B: structured baselines, rather than the random reference alone.
ax = fig.add_subplot(gs[0, 1])
groups = [matched, mismatched, np.array(dense), np.array([random_j])]
labels = ["Matched\nSpine–Dense", "Mismatched\nSpine–Dense", "Dense–Dense", "Independent\nrandom"]
colors = [P["accent"], P["blue"], P["green"], P["muted"]]
offsets = [np.array([-.07, 0, .07]), np.linspace(-.10,.10,6), np.array([-.07,0,.07]), np.array([0.])]
for i, (vals, color, off) in enumerate(zip(groups, colors, offsets)):
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0
    if len(vals) > 1:
        ax.errorbar(i, mu, yerr=sd, fmt="o", ms=6.5, capsize=3,
                    color=color, zorder=3)
    else:
        ax.scatter(i, mu, marker="D", s=30, color=color, zorder=3)
    ax.scatter(i + off, vals, s=18, color=color, alpha=.48, zorder=4)
    label = f"{fmt3(mu)}±{fmt3(sd)}" if len(vals) > 1 else fmt3(mu)
    ax.text(i, mu+.032, label, ha="center", va="bottom", fontsize=6.5, color=color)

ax.plot([0, 0, 1, 1], [.672, .682, .682, .672], color=P["edge"], lw=.7)
ax.text(.5, .687, f"specificity gain {fmt3(st.mean(specificity))}±{fmt3(st.stdev(specificity))}",
        ha="center", va="bottom", fontsize=6.2, color=P["edge"], fontweight="bold")
ax.set_xticks(range(4), labels)
ax.set_ylim(.22,.72)
ax.set_ylabel("Jaccard")
ax.set_title("B  Matched alignment exceeds structured nulls", loc="left",
             fontsize=8.4, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=.32)
for s in ("top", "right"): ax.spines[s].set_visible(False)

figstyle.save(fig, "fig_overlap_seeds", pad=.025)
