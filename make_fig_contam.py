#!/usr/bin/env python3
"""Audited positive/negative channel support and tail concentration (Figure 2)."""
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

# Local upsizing for two-column readability (Figure 2).
plt.rcParams.update({
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
})

fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.85))
a.bar([0, 1], [independent, overlap],
      color=[P["muted"], P["accent"]], width=.58)
a.set_xticks([0, 1])
a.set_xticklabels(["independent\nreference", "measured\noverlap"])
a.set_ylabel("shared-support fraction")
a.set_title(f"Support overlap is {enrichment:.1f}× enriched",
            fontsize=11.0, fontweight="bold")
a.text(1, overlap + .0012, f"{enrichment:.1f}×", ha="center",
       fontsize=10.0, color=P["accent"], fontweight="bold")

b.bar([0, 1], [same_sign, conflict],
      color=[P["blue"], P["red"]], width=.58)
b.set_xticks([0, 1])
b.set_xticklabels(["same sign", "conflicting sign"])
b.set_ylim(0, .60)
b.set_ylabel("share on common support")
b.set_title("The common support is conflict-prone", fontsize=11.0, fontweight="bold")

for ax in (a, b):
    ax.grid(axis="y", ls=":", alpha=.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
figstyle.save(fig, "fig_contam")
