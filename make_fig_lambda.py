#!/usr/bin/env python3
"""Density-matched mask controls reported by the audited cluster."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply()
P = figstyle.PALETTE
rows = list(csv.DictReader(open("audited_mask_controls.csv", encoding="utf-8")))
labels = [r["control"] for r in rows][::-1]
values = [float(r["mean"]) for r in rows][::-1]
sds = [float(r["sample_std"]) for r in rows][::-1]
colors = [P["muted"], P["red"], P["blue"], P["green"], "#c49a5a", P["accent"]]

fig, ax = plt.subplots(figsize=(5.5, 2.9))
y = range(len(labels))
ax.hlines(y, 0.33, values, color=colors, lw=2.3)
ax.scatter(values, y, color=colors, s=38, edgecolors=P["edge"], linewidths=.4, zorder=3)
for yi, value, sd, color in zip(y, values, sds, colors):
    ax.errorbar(value, yi, xerr=sd, fmt="none", ecolor=color,
                capsize=2.5, lw=.8, zorder=2)
for yi, v in zip(y, values):
    ax.text(v + .003, yi, f"{v:.3f}", va="center", fontsize=7.2)
ax.axvline(.426, color=P["muted"], ls="--", lw=.9)
ax.text(.4245, 4.58, "Dense 1× reference", ha="right", va="center",
        fontsize=6.2, color=P["muted"])
ax.set_yticks(list(y)); ax.set_yticklabels(labels)
ax.set_xlim(.33, .463); ax.set_xlabel("Mean4 (three-seed mean)")
ax.set_title("The selected coordinates are not an arbitrary 40% subset",
             fontsize=8.7, fontweight="bold")
ax.grid(axis="x", ls=":", alpha=.35)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
figstyle.save(fig, "fig_lambda")
