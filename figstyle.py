#!/usr/bin/env python3
"""Shared figure style for Paper D (Advantage Spine).

One visual system across all four figures, matching top-tier science-of-DL
house style: serif typography that echoes the Elsevier body font (STIX, a
Computer-Modern/Times-like face available without a full TeX install), a
muted colourblind-safe palette, thin de-cluttered axes, and dual vector
output (PDF for the manuscript + SVG for alignment/replication work).

Usage:
    import figstyle
    figstyle.apply()
    ...
    figstyle.save(fig, "fig_concept")   # writes fig_concept.pdf AND .svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- muted, colourblind-safe palette (one source of truth) -------------------
PALETTE = {
    "raw":    "#e7edf4",   # neutral slate — raw / baseline
    "decon":  "#d7e5d3",   # muted sage — deconflict stage
    "spine":  "#f0e0c4",   # warm sand — the Spine (highlight object)
    "edge":   "#37475a",   # ink for outlines / titles
    "body":   "#243040",   # box body text
    "accent": "#8a5a1e",   # amplification / arrow-label accent
    "muted":  "#6a7787",   # footnotes, secondary
    "blue":   "#3d6aa2",   # data series A
    "red":    "#b0503f",   # data series B (collapse / negative)
    "green":  "#4f8055",   # data series C
    "grid":   "#c9d2dc",
}


def apply():
    """Install the shared rcParams. Call once before building a figure."""
    plt.rcParams.update({
        "font.family": "serif",
        # STIX first (Times/CM-like, ships with matplotlib), graceful fallbacks.
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.edgecolor": PALETTE["edge"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,     # embed real (Type-42) fonts — no Type-3 warnings
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text in SVG for editable alignment
    })


def save(fig, stem, pad=0.02):
    """Save both a PDF (manuscript) and an SVG (alignment/replication)."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=pad)
    print(f"wrote {stem}.pdf + {stem}.svg")


# --- coordinate-field primitives (scheme-① scientific schematics) ------------
def coord_field(ax, x0, y0, w, h, rng, *, n_bg=48, n_spine=0, n_conflict=0,
                spine_s=11, bg_s=5.5, conflict_s=7.0, alpha_bg=0.35,
                frame=True):
    """Draw a soft coordinate cloud: gray residual, amber Spine, red conflict.

    No filled stage boxes — the field *is* the visual object.
    """
    import numpy as np
    P = PALETTE
    # faint frame (hairline, not a PPT card)
    if frame:
        ax.plot([x0, x0 + w, x0 + w, x0, x0],
                [y0, y0, y0 + h, y0 + h, y0],
                color=P["grid"], lw=0.55, zorder=1)
    # background residual (diffuse)
    if n_bg > 0:
        bx = x0 + 0.08 * w + 0.84 * w * rng.random(n_bg)
        by = y0 + 0.08 * h + 0.84 * h * rng.random(n_bg)
        ax.scatter(bx, by, s=bg_s, c=P["muted"], alpha=alpha_bg,
                   linewidths=0, zorder=2, rasterized=True)
    # conflicted shared coords (red ×)
    if n_conflict > 0:
        cx = x0 + 0.15 * w + 0.70 * w * rng.random(n_conflict)
        cy = y0 + 0.15 * h + 0.70 * h * rng.random(n_conflict)
        ax.scatter(cx, cy, s=conflict_s, c=P["red"], marker="x",
                   linewidths=0.7, alpha=0.75, zorder=3)
    # Spine (kept / locked)
    if n_spine > 0:
        # slightly clustered toward center-left for "concentration" read
        sx = x0 + 0.18 * w + 0.45 * w * rng.random(n_spine)
        sy = y0 + 0.18 * h + 0.64 * h * rng.random(n_spine)
        ax.scatter(sx, sy, s=spine_s, c=P["accent"], alpha=0.92,
                   edgecolors=P["edge"], linewidths=0.25, zorder=4,
                   rasterized=True)
    return


def thin_arrow(ax, x0, y0, x1, y1, color=None, lw=1.15):
    """Hairline arrow between coordinate fields."""
    from matplotlib.patches import FancyArrowPatch
    if color is None:
        color = PALETTE["edge"]
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=5))
