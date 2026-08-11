#!/usr/bin/env python3
"""Graphical abstract for the PaperD submission."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle
figstyle.apply(); P=figstyle.PALETTE
fig,ax=plt.subplots(figsize=(7.4,3.2)); ax.set_xlim(0,100); ax.set_ylim(0,44); ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.6,rounding_size=2',fc=color,ec=P['edge'],lw=1))
    ax.text(x+w/2,y+h-4,title,ha='center',fontweight='bold',fontsize=9)
    ax.text(x+w/2,y+h/2-2,body,ha='center',va='center',fontsize=7.3,linespacing=1.35)
def arrow(x0,x1,y): ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=14,lw=1.5,color=P['edge']))
ax.text(50,41.5,'Dynamic coordinate selection extends the tested high-rate regime',ha='center',fontweight='bold',fontsize=11)
box(1,14,27,21,'Dense high-rate RLVR','$1\\times$: 0.426\n$20\\times$: 0.340\n$-0.086$ Mean4',P['raw'])
box(36.5,14,27,21,'Advantage Spine','$\\lambda=0.1$\nper-step top 40%\ndynamic support',P['decon'])
box(72,14,27,21,'Joint recipe at 20×','Mean4 0.445\npaired gain +0.0195\n95% CI [0.0131, 0.0258]',P['spine'])
arrow(28.5,36,24.5); arrow(64,71.5,24.5)
ax.text(50,7,'39.3% adaptive support  •  85.0% direction energy  •  Jaccard 0.600±0.010',ha='center',fontsize=8.5,fontweight='bold',color=P['accent'])
ax.text(50,2.5,'Complete 16-cell factorial; three matched seeds per cell',ha='center',fontsize=7.7,color=P['body'])
fig.tight_layout(); fig.savefig('graphical_abstract.pdf',bbox_inches='tight',pad_inches=.05); fig.savefig('graphical_abstract.png',dpi=300,bbox_inches='tight',pad_inches=.05)
