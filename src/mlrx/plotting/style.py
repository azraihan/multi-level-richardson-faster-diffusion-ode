"""Figure style and the save contract.

House rules for this project, applied everywhere:

* **No titles and no captions inside the figure.**  Every figure is a standalone
  PDF destined for a report where the caption is set in the document, not baked
  into the artwork.  Axis labels and legends stay; titles do not.
* **Vector PDF with embedded, editable text** (``pdf.fonttype = 42``), so the
  figures scale losslessly and the text remains selectable and searchable.
* **Every figure ships its data.**  :func:`save_figure` writes a CSV alongside
  the PDF containing exactly the numbers that were plotted, so the figure can
  be restyled later -- different palette, different aspect, different tool --
  without re-running any experiment.
* **Colour-blind safe palette, and never colour alone.**  Series are separated
  by marker and dash pattern as well as hue, so the figures survive greyscale
  printing.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ["apply_style", "save_figure", "PALETTE", "MARKERS", "DASHES",
           "series_style", "figsize"]

# Okabe-Ito: eight hues distinguishable under the common forms of colour
# vision deficiency.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = [
    (None, None), (5, 1.6), (1.4, 1.4), (7, 1.6, 1.4, 1.6),
    (3.5, 1.4), (8, 2), (1.4, 1.2, 4, 1.2), (2.5, 1.2),
]


def figsize(kind="single"):
    """Sizes chosen for a two-column report at 100% scale."""
    return {
        "single": (3.5, 2.6),
        "wide": (7.0, 2.8),
        "square": (3.4, 3.2),
        "tall": (3.5, 4.2),
        "panel": (7.0, 5.0),
    }[kind]


def apply_style():
    """Install the project's rcParams.  Idempotent."""
    mpl.rcParams.update({
        # vector output with real, editable text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.labelpad": 3.0,
        "axes.axisbelow": True,

        "grid.color": "#D8DCE0",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.85,

        "lines.linewidth": 1.45,
        "lines.markersize": 3.9,
        "lines.markeredgewidth": 0.0,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,

        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.32,
        "legend.borderaxespad": 0.3,

        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def series_style(i, marker=True):
    """Consistent colour + dash + marker for series ``i``."""
    s = {
        "color": PALETTE[i % len(PALETTE)],
        "dashes": DASHES[i % len(DASHES)],
    }
    if marker:
        s["marker"] = MARKERS[i % len(MARKERS)]
    return s


def save_figure(fig, outdir, name, data=None, formats=("pdf",), close=True):
    """Save a figure and, crucially, the data behind it.

    Parameters
    ----------
    fig : matplotlib Figure
    outdir : path
        Figures are written to ``outdir``; their data to ``outdir/data``.
    name : str
        Base name, no extension.
    data : DataFrame, list of dicts, or dict of those
        The numbers plotted.  A dict writes one CSV per entry, suffixed with
        the key, so a multi-panel figure can ship one table per panel.
    formats : iterable of str

    Returns
    -------
    list of str
        Paths written.
    """
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)
    datadir = os.path.join(outdir, "data")
    os.makedirs(datadir, exist_ok=True)

    written = []
    for fmt in formats:
        path = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
        written.append(path)

    def _write(obj, suffix=""):
        if obj is None:
            return
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        p = os.path.join(datadir, f"{name}{suffix}.csv")
        df.to_csv(p, index=False)
        written.append(p)

    if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
        for key, obj in data.items():
            _write(obj, f"__{key}")
    else:
        _write(data)

    if close:
        plt.close(fig)
    return written
