#!/usr/bin/env python3
"""Independent panels for audited channel overlap and sign conflict (Figure 2)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply()
P = figstyle.PALETTE

rows = list(csv.DictReader(open(
    "audited_channel_overlap_results.csv", encoding="utf-8")))
metric = {(r["metric"], r["record_type"]): float(r["value"]) for r in rows
          if r["record_type"] != "seed"}
independent = metric[("independent_overlap_reference", "aggregate_product")]
overlap = metric[("measured_overlap_fraction", "mean")]
enrichment = metric[("overlap_enrichment", "ratio_of_aggregate_quantities")]
same_sign = metric[("same_sign_rate", "mean")]
conflict = metric[("conflict_rate", "mean")]

# Size each source at its final subfigure width so labels remain readable after
# LaTeX placement.  Panel titles and letters belong to LaTeX, not the artwork.
plt.rcParams.update({
    "font.size": 7.8,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
})

fig, ax = plt.subplots(figsize=(1.72, 1.58))
vals = [100 * independent, 100 * overlap]
ax.bar([0, 1], vals, color=[P["muted"], P["accent"]], width=.58)
ax.set_xticks([0, 1], ["independent", "measured"])
ax.set_ylim(0, 2.05)
ax.set_ylabel("Support overlap (%)")
for x, value, color in zip([0, 1], vals, [P["muted"], P["accent"]]):
    ax.text(x, value + .08, f"{value:.2f}%", ha="center", va="bottom",
            fontsize=7.1, color=color, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=.32)
fig.tight_layout(pad=.35)
figstyle.save(fig, "fig_contam_overlap", pad=.015)
plt.close(fig)

fig, ax = plt.subplots(figsize=(1.72, 1.58))
vals = [100 * same_sign, 100 * conflict]
ax.bar([0, 1], vals, color=[P["blue"], P["red"]], width=.58)
ax.set_xticks([0, 1], ["same sign", "conflict"])
ax.set_ylim(0, 60)
ax.set_ylabel("Shared coordinates (%)")
for x, value, color in zip([0, 1], vals, [P["blue"], P["red"]]):
    ax.text(x, value + 1.5, f"{value:.0f}%", ha="center", va="bottom",
            fontsize=7.1, color=color, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=.32)
fig.tight_layout(pad=.35)
figstyle.save(fig, "fig_contam_sign", pad=.015)
plt.close(fig)
