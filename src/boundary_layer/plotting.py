"""Shared figure style: a validated categorical palette and one-hue ramps."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e4e3df"

# Categorical slots, fixed order (CVD-validated, see README "Figuras")
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("bl_blue", BLUE_RAMP)

# Diverging blue <-> red through a neutral gray (signed quantities)
DIVERGING = LinearSegmentedColormap.from_list(
    "bl_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4b2a8", "#e34948", "#9c1f22"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": INK_3,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "text.color": INK,
    })


def label_end(ax, x, y, text, dx=6, color=INK_2, **kw):
    """Direct label at a line end, in text ink (never the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", color=color, fontsize=9, **kw)
