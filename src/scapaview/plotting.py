"""Shared plotting utilities for scAPAview."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def draw_gene_model(
    ax: plt.Axes,
    gene_row: pd.Series,
    exons: pd.DataFrame,
    strand: str,
    color: str = "steelblue",
    y: float = 0.0,
    height: float = 0.4,
) -> None:
    """Draw a gene model (exon boxes + intron line) on *ax*.

    Parameters
    ----------
    ax       : matplotlib Axes
    gene_row : Series with start/end (0-based)
    exons    : DataFrame of exon records for this gene
    strand   : '+' or '-'
    color    : fill colour for exon boxes
    y        : vertical centre of the gene model
    height   : height of exon boxes
    """
    start_col = "start" if "start" in gene_row.index else "Start"
    end_col = "end" if "end" in gene_row.index else "End"
    gene_start = float(gene_row[start_col])
    gene_end = float(gene_row[end_col])

    # Backbone line
    ax.hlines(y, gene_start, gene_end, colors="black", linewidth=1, zorder=1)

    # Exon boxes
    ex_start = "start" if "start" in exons.columns else "Start"
    ex_end = "end" if "end" in exons.columns else "End"
    for _, exon in exons.iterrows():
        rect = matplotlib.patches.Rectangle(
            (float(exon[ex_start]), y - height / 2),
            float(exon[ex_end]) - float(exon[ex_start]),
            height,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(rect)

    # Strand arrow
    mid = (gene_start + gene_end) / 2
    dx = (gene_end - gene_start) * 0.05
    if strand == "-":
        dx = -dx
    ax.annotate(
        "",
        xy=(mid + dx, y),
        xytext=(mid, y),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )


def draw_pas_sites(
    ax: plt.Axes,
    pas_sites: pd.DataFrame,
    y_level: float,
    color_by: str = "source",
    marker: str = "v",
    markersize: float = 8,
) -> None:
    """Draw PAS site marks on *ax*.

    Parameters
    ----------
    pas_sites : DataFrame with start column
    y_level   : y position for markers
    color_by  : column to colour markers by (default 'source')
    """
    start_col = "start" if "start" in pas_sites.columns else "Start"
    palette = {"sierra": "royalblue", "scapture": "darkorange", "scpolaseq": "green"}

    for _, row in pas_sites.iterrows():
        color = palette.get(str(row.get(color_by, "")), "grey")
        ax.plot(
            float(row[start_col]),
            y_level,
            marker=marker,
            color=color,
            markersize=markersize,
            zorder=3,
        )


def draw_apa_lollipops(
    ax: plt.Axes,
    apa_events: pd.DataFrame,
    y_level: float = 0.0,
    delta_col: str = "delta_pdui",
    site_pos_col: str = "start",
    max_height: float = 1.0,
) -> None:
    """Lollipop plot of ΔPDUI values on *ax*.

    Stems are coloured red (lengthening) or blue (shortening).
    """
    if apa_events is None or apa_events.empty:
        return
    cmap = make_colormap_delta_pdui()
    for _, row in apa_events.iterrows():
        delta = float(row.get(delta_col, 0))
        pos = float(row.get(site_pos_col, 0))
        norm_delta = (delta + 1) / 2  # map [-1,1] → [0,1]
        color = cmap(norm_delta)
        ax.vlines(pos, y_level, y_level + delta * max_height, colors=color, linewidth=1.5)
        ax.plot(pos, y_level + delta * max_height, "o", color=color, markersize=5)


def draw_coverage(
    ax: plt.Axes,
    positions: np.ndarray,
    coverage: np.ndarray,
    label: str | None = None,
    color: str | None = None,
    alpha: float = 0.7,
) -> None:
    """Fill-under plot of bigWig coverage on *ax*."""
    ax.fill_between(positions, coverage, alpha=alpha, color=color or "steelblue", label=label)
    ax.set_ylim(bottom=0)


def draw_splice_arcs(
    ax: plt.Axes,
    junctions: pd.DataFrame,
    y_base: float = 0.0,
    max_linewidth: float = 5.0,
) -> None:
    """Draw curved arcs for splice junctions on *ax*.

    Expects junctions DataFrame with columns: start, end, count (or score).
    Arc height is proportional to span; linewidth proportional to count.
    """
    if junctions is None or junctions.empty:
        return
    start_col = "start" if "start" in junctions.columns else "Start"
    end_col = "end" if "end" in junctions.columns else "End"
    count_col = "count" if "count" in junctions.columns else "score"

    max_count = float(junctions[count_col].max()) if count_col in junctions.columns else 1.0

    for _, row in junctions.iterrows():
        x0 = float(row[start_col])
        x1 = float(row[end_col])
        span = x1 - x0
        height = span * 0.3
        count = float(row.get(count_col, 1))
        lw = (count / max_count) * max_linewidth if max_count > 0 else 1.0

        # Bezier arc via Path
        import matplotlib.path as mpath
        import matplotlib.patches as mpatches

        verts = [
            (x0, y_base),
            (x0 + span * 0.25, y_base + height),
            (x0 + span * 0.75, y_base + height),
            (x1, y_base),
        ]
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
        ]
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(
            path, facecolor="none", edgecolor="purple", linewidth=lw, alpha=0.7
        )
        ax.add_patch(patch)


def make_colormap_delta_pdui(
    vmin: float = -1, vmax: float = 1
) -> matplotlib.colors.Colormap:
    """Return a diverging colormap for ΔPDUI values (blue→white→red)."""
    return matplotlib.cm.RdBu_r


def save_figure(fig: plt.Figure, output: str | Path, dpi: int = 150) -> None:
    """Save figure to file, detecting format from extension."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", output)
