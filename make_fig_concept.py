#!/usr/bin/env python3
"""Opening schematic for the audited Advantage-Spine experiment.

The coordinate locations are deliberately illustrative.  Every number in the
outcome panel is copied from (and mechanically checked against) the audited
CSV bundle; no per-coordinate observation is implied by the drawing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle

# Canonical audited callouts. verify_numbers.py checks these against the CSVs.
DENSE20_MEAN4 = 0.340
SPINE20_MEAN4 = 0.445
GAIN_VS_DENSE20 = 0.105
NONZERO_RATIO = 0.393
ENERGY_KEPT = 0.850
JACCARD_MEAN = 0.600
JACCARD_SD = 0.009643651

figstyle.apply()
P = figstyle.PALETTE
INK, BODY, MUTED = P["edge"], P["body"], P["muted"]
RED, BLUE, AMBER, GREEN = P["red"], P["blue"], P["accent"], P["green"]
rng = np.random.default_rng(17)

fig, ax = plt.subplots(figsize=(7.45, 2.62))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")


def matrix_stack(x0, selected=False):
    """Draw four illustrative Transformer parameter matrices."""
    names = ["QKV", "Attn out", "MLP up", "MLP down"]
    for row, name in enumerate(names):
        y0 = 27.0 - row * 5.25
        ax.text(x0 - 1.0, y0 + 1.45, name, ha="right", va="center",
                fontsize=5.5, color=MUTED)
        ax.add_patch(FancyBboxPatch(
            (x0, y0), 17.8, 3.05,
            boxstyle="round,pad=0.05,rounding_size=0.35",
            facecolor="#f7f9fb", edgecolor=P["grid"], linewidth=0.55))

        # Use shared, seeded locations across the dense and selected panels.
        local = np.random.default_rng(100 + row)
        px = x0 + 0.75 + 16.3 * local.random(18)
        py = y0 + 0.45 + 2.15 * local.random(18)
        if selected:
            keep = np.argsort(local.random(18))[:7]
            drop = np.setdiff1d(np.arange(18), keep)
            ax.scatter(px[drop], py[drop], s=5.0, c=P["grid"], alpha=.55,
                       linewidths=0, zorder=3)
            ax.scatter(px[keep], py[keep], s=10.0, c=AMBER,
                       edgecolors=INK, linewidths=.25, zorder=4)
        else:
            ax.scatter(px, py, s=6.0, c=BLUE, alpha=.62,
                       linewidths=0, zorder=3)
            conflict = local.choice(np.arange(18), size=4, replace=False)
            ax.scatter(px[conflict], py[conflict], s=13.0, c=RED,
                       marker="x", linewidths=.75, zorder=4)


def arrow(x0, x1, label, sublabel):
    y = 19.2
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color=INK))
    ax.text((x0 + x1) / 2, y + 2.0, label, ha="center", va="bottom",
            fontsize=6.3, fontweight="bold", color=GREEN)
    ax.text((x0 + x1) / 2, y - 1.8, sublabel, ha="center", va="top",
            fontsize=5.7, color=MUTED)


# A — dense high-rate update.
ax.text(2.0, 36.1, "A", fontsize=8.2, fontweight="bold", color=INK,
        bbox=dict(boxstyle="circle,pad=.18", fc=P["raw"], ec=INK, lw=.6))
ax.text(6.0, 36.2, r"Dense update at $20\times$", fontsize=8.1,
        fontweight="bold", color=INK, va="center")
matrix_stack(5.2, selected=False)
ax.text(14.1, 4.2, "all coordinates receive the step",
        ha="center", fontsize=6.1, color=BODY)
ax.text(14.1, 1.45, f"Mean4 {DENSE20_MEAN4:.3f}  |  collapse: yes",
        ha="center", fontsize=6.4, color=RED, fontweight="bold")

arrow(24.4, 34.0, "deconflict", r"$\lambda=0.1$")

# B — dynamic coordinate selector.
ax.text(34.8, 36.1, "B", fontsize=8.2, fontweight="bold", color=INK,
        bbox=dict(boxstyle="circle,pad=.18", fc=P["spine"], ec=INK, lw=.6))
ax.text(38.8, 36.2, "Dynamic Advantage Spine", fontsize=8.1,
        fontweight="bold", color=INK, va="center")
matrix_stack(38.0, selected=True)
ax.text(46.9, 5.2, "per-tensor top 40% of pre-mask AdamW update",
        ha="center", fontsize=6.1, color=BODY)
ax.text(46.9, 2.6, "support refreshed every optimization step",
        ha="center", fontsize=6.1, color=AMBER, fontweight="bold")

arrow(57.3, 66.0, "evaluate", "matched seeds")

# C — measured outcomes only.
ax.text(66.8, 36.1, "C", fontsize=8.2, fontweight="bold", color=INK,
        bbox=dict(boxstyle="circle,pad=.18", fc=P["decon"], ec=INK, lw=.6))
ax.text(70.8, 36.2, "Observed at the tested limit", fontsize=8.1,
        fontweight="bold", color=INK, va="center")

metric_rows = [
    ("coordinates retained", f"{100*NONZERO_RATIO:.1f}%"),
    ("update energy retained", f"{100*ENERGY_KEPT:.1f}%"),
    ("support Jaccard", rf"${JACCARD_MEAN:.3f}\,\pm\,{JACCARD_SD:.3f}$"),
]
for i, (label, value) in enumerate(metric_rows):
    y = 29.2 - i * 5.1
    ax.text(70.0, y, label, ha="left", va="center", fontsize=6.2, color=MUTED)
    ax.text(96.5, y, value, ha="right", va="center", fontsize=7.7,
            color=AMBER if i < 2 else BLUE, fontweight="bold")
    ax.plot([70.0, 96.5], [y - 2.05, y - 2.05], color=P["grid"], lw=.45)

ax.text(70.0, 12.0, r"Dense $20\times$", ha="left", va="center",
        fontsize=6.2, color=MUTED)
ax.text(84.0, 12.0, f"{DENSE20_MEAN4:.3f}", ha="right", va="center",
        fontsize=7.3, color=RED, fontweight="bold")
ax.add_patch(FancyArrowPatch((85.0, 12.0), (89.0, 12.0), arrowstyle="-|>",
                             mutation_scale=8, lw=.9, color=GREEN))
ax.text(96.5, 12.0, f"{SPINE20_MEAN4:.3f}", ha="right", va="center",
        fontsize=8.8, color=GREEN, fontweight="bold")
ax.text(83.2, 8.2, rf"Mean4 gain vs dense $20\times$: +{GAIN_VS_DENSE20:.3f}",
        ha="center", va="center", fontsize=6.5, color=GREEN,
        fontweight="bold")

ax.text(50, -0.15,
        "SCHEMATIC — dot positions are illustrative, not measured coordinates",
        ha="center", va="bottom", fontsize=5.8, color=MUTED, style="italic")

figstyle.save(fig, "fig_concept", pad=0.025)
