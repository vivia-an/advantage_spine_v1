#!/usr/bin/env python3
"""Audited coordinate and energy concentration without causal overclaim."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

figstyle.apply(); P=figstyle.PALETTE
fig, ax = plt.subplots(figsize=(5.2,2.35))
labels=["Coordinate share", "Update-energy share"]
kept=[.393,.850]
colors=["#c49a5a",P["accent"]]
for i,(lab,val,col) in enumerate(zip(labels,kept,colors)):
    ax.barh(i,val,color=col,height=.38,label=lab)
    ax.barh(i,1-val,left=val,color="#e8ebef",height=.38)
    ax.text(val/2,i,f"{100*val:.1f}% kept",ha="center",va="center",fontsize=7.4,
            color="white",fontweight="bold")
    ax.text(val+(1-val)/2,i,f"{100*(1-val):.1f}% residual",ha="center",va="center",fontsize=6.8,color=P["muted"])
ax.set_yticks([0,1]); ax.set_yticklabels(labels); ax.set_xlim(0,1)
ax.set_xlabel("fraction")
ax.set_title("39.3% support retains 85.0% of adaptive-direction energy",fontsize=8.8,fontweight="bold")
for s in ("top","right","left"): ax.spines[s].set_visible(False)
fig.tight_layout(); figstyle.save(fig,"fig_energy")
