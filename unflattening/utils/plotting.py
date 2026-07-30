"""Shared plotting setup: writes PGF plus a PNG preview. 
Figures go to the repo-root ``figures/`` directory; numeric
outputs to the repo-root ``data/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # PNG previews; PGF written per-save via backend="pgf"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402

_CODE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = _CODE_DIR / "figures"
DATA_DIR = _CODE_DIR / "data"
FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Manuscript palette: R ``COLOURS.LIST`` exact hexes (from the sibling
# pulse-level paper's ``layout.r``), mapped onto the semantics fixed by Fig. 1
# (fig_a_overview): teal = the favourable / trainable "good" path, orange = the
# barren-plateau "bad" path; blue and navy are neutral accents.  GOOD/BAD are
# reserved strictly for the good/bad path so the figures read consistently with
# the Fig. 1 schematic.
TEAL = "#009371"    # COLOURS.LIST[4] teal; usually the good path
ORANGE = "#E69F00"     # COLOURS.LIST[2] orange; usually the bad path
ACCENT = "#1f78b4"  # COLOURS.LIST[7] blue; usually a secondary good/bad path
NAVY = "#002D4C"  # COLOURS.LIST[8] navy; structure/reference/empirical data
_PALETTE = [TEAL, ACCENT, NAVY, ORANGE]

# R ``theme_paper_base()`` (theme_bw, base_size 8.5): white panel with a black
# four-sided frame, grey92 major+minor gridlines (no minor tick marks), thin
# lines/markers, and compact frameless legends placed on top (see ``top_legend``).
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        # theme_bw frame: thin black box on all four spines.
        "axes.edgecolor": "black",
        "axes.linewidth": 0.5,
        # grey92 major + minor gridlines; minor lines but no minor tick marks.
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 0.0,
        "ytick.minor.size": 0.0,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.prop_cycle": plt.cycler("color", _PALETTE),
        # lighter geometry, leaning toward R LINE.SIZE/POINT.SIZE.
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        # compact frameless legend (R theme_paper_base legend.position="top").
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "figure.constrained_layout.use": True,
        # PGF: don't embed fonts, inherit the document's; avoid usetex at save.
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "text.usetex": False,
    }
)

# IEEE single / double column widths in inches.
COL = 3.4
WIDE = 7.0

# Neutral greys for non-data elements (kept out of the colour cycle).
GREY_FILL = "0.85"      # proven-range / band fill_between
GREY_REF = "#999999"    # COLOURS.LIST[3] grey -- reference / guide lines

# Sequential cool ramp (teal -> blue -> navy) for an *ordinal* sweep -- a doping
# count t, a qubit count n, a depth.  Monotone-cool so it reads as an order and
# stays colourblind / greyscale legible, and keeps BAD=orange reserved for the
# barren / "bad path".  Categorical (unordered) series should use the palette.
_ORDINAL_CMAP = LinearSegmentedColormap.from_list("polar_ordinal", [TEAL, ACCENT, NAVY])


def ordinal_colors(k):
    """``k`` sequential colours for an ordered sweep (not unordered categories)."""
    return [_ORDINAL_CMAP(i / (k - 1) if k > 1 else 0.0) for i in range(k)]


def unify_grid(ax, *twins):
    """Consistent grid for the paper's log panels: major gridlines only (one per decade
    on a log axis, at the major ticks on a linear axis) and no minor gridlines, on the
    primary ``ax``.  Any twin/secondary axes get ticks and labels but no gridlines: the
    secondary scale is not a fixed rescaling of the primary (Var/P differs per curve and
    per n, and |Omega| is unrelated to the purity), so a shared gridline would imply a
    correspondence that does not exist.

    The log y-axis uses an every-other-decade locator so the horizontal-line density is
    the same on every panel regardless of its span (the default locator auto-thins by
    axis height, which makes narrow panels denser than wide ones)."""
    ax.minorticks_off()
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_locator(LogLocator(base=100.0))
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    for tw in twins:
        tw.minorticks_off()
        tw.grid(False)


def top_legend(ax, handles=None, labels=None, ncol=None, **kw):
    """Compact, frameless legend above the panel (R theme_paper_base)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if ncol is None:
        ncol = len(labels)
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, frameon=False, **kw)


def save(fig, name: str) -> None:
    """Save ``fig`` as ``figures/<name>.pgf`` (+ a .png preview)."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    try:
        fig.savefig(FIG_DIR / f"{name}.pgf", backend="pgf")
    except Exception as e:  # pragma: no cover - latex may be unavailable
        print(f"  [warn] PGF export of {name} failed ({e}); PNG written.")
    plt.close(fig)
    print(f"  wrote figures/{name}.pgf (+ .png)")
