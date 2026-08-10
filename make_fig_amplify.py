#!/usr/bin/env python3
"""Audited 2x2 factorial over mask, negative-channel weight, and LR."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply()
P = figstyle.PALETTE

rows = list(csv.DictReader(open("audited_factorial_results.csv", encoding="utf-8")))
specs = [
    (("Dense", "1.0"), "Dense, $\\lambda=1$", P["red"], "--", "X"),
    (("Dense", "0.1"), "Dense, $\\lambda=0.1$", P["blue"], ":", "s"),
    (("Top-0.4", "1.0"), "Top-0.4, $\\lambda=1$", P["green"], "-.", "^"),
    (("Top-0.4", "0.1"), "Spine recipe", P["accent"], "-", "o"),
]

fig, ax = plt.subplots(figsize=(5.65, 3.25))
for (mask, lam), label, color, ls, marker in specs:
    rr = [r for r in rows if r["mask"] == mask and r["lambda"] == lam]
    x = [float(r["lr_multiplier"]) for r in rr]
    y = [float(r["mean"]) for r in rr]
    sd = [float(r["sample_std"]) for r in rr]
    ax.errorbar(x, y, yerr=sd, label=label, color=color, ls=ls, marker=marker,
                lw=1.8, ms=5.7, capsize=2.5, zorder=3)

ax.axhline(0.426, color=P["muted"], lw=0.9, ls=(0, (3, 2)), zorder=1)
ax.text(1.05, 0.429, "Dense 1× reference", fontsize=6.5, color=P["muted"])
ax.annotate("+0.019\n95% paired CI\n[0.0124, 0.0256]",
            xy=(20, 0.445), xytext=(11.0, 0.466), ha="center", va="bottom",
            fontsize=7.2, color=P["accent"],
            arrowprops=dict(arrowstyle="->", color=P["accent"], lw=0.8))
ax.annotate("−0.086", xy=(20, 0.340), xytext=(12.7, 0.357), ha="center",
            fontsize=7.2, color=P["red"],
            arrowprops=dict(arrowstyle="->", color=P["red"], lw=0.8))
ax.set_xscale("log")
ax.set_xticks([1, 5, 10, 20])
ax.set_xticklabels(["1×", "5×", "10×", "20×"])
ax.set_xlim(0.86, 24)
ax.set_ylim(0.315, 0.485)
ax.set_xlabel("learning-rate multiplier")
ax.set_ylabel("Mean4 (mean ± sample SD; $n=3$)")
ax.set_title("Only the joint recipe improves at 20×",
             fontsize=8.6, fontweight="bold")
ax.grid(axis="y", ls=":", alpha=0.38)
ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
figstyle.save(fig, "fig_amplify")
