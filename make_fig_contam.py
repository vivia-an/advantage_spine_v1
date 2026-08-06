#!/usr/bin/env python3
"""Audited positive/negative channel support and tail concentration (Figure 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply()
P = figstyle.PALETTE
# Local upsizing for two-column readability (Figure 2).
plt.rcParams.update({
    "font.size": 10.0,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
})

fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.85))
a.bar([0, 1], [.0023, .017], color=[P["muted"], P["accent"]], width=.58)
a.set_xticks([0, 1])
a.set_xticklabels(["independent\nreference", "measured\noverlap"])
a.set_ylabel("shared-support fraction")
a.set_title("Support overlap is 7.4× enriched", fontsize=11.0, fontweight="bold")
a.text(1, .0182, "7.4×", ha="center", fontsize=10.0, color=P["accent"], fontweight="bold")

b.bar([0, 1], [.48, .52], color=[P["blue"], P["red"]], width=.58)
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
