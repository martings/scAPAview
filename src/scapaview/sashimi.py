"""Sashimi-style APA plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import (
    draw_coverage,
    draw_gene_model,
    draw_pas_sites,
    draw_apa_lollipops,
    draw_splice_arcs,
    save_figure,
)

logger = logging.getLogger(__name__)


def plot_apa_sashimi_like(
    gene_name: str,
    gtf: pd.DataFrame,
    pas_sites: pd.DataFrame,
    apa_events: pd.DataFrame | None = None,
    bigwig_tracks: dict | None = None,
    junctions: pd.DataFrame | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    flank: int = 500,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot a sashimi-like APA visualisation for a single gene.

    If *junctions* are absent: plots coverage + gene model + PAS sites + APA events.
    If *junctions* are provided: additionally draws arc annotations with
    linewidth proportional to junction support.

    Parameters
    ----------
    gene_name     : gene symbol or gene_id
    gtf           : standardised GTF DataFrame (0-based)
    pas_sites     : PAS sites DataFrame
    apa_events    : APA events DataFrame (optional)
    bigwig_tracks : dict mapping label → bigWig path
    junctions     : junction DataFrame with start/end/count columns (optional)
    group_a/b     : group labels for display
    flank         : bp extension around gene
    output        : output file path
    show          : call plt.show()

    Returns
    -------
    (fig, axes) tuple
    """
    feature_col = "Feature" if "Feature" in gtf.columns else "feature"
    start_col = "Start" if "Start" in gtf.columns else "start"
    end_col = "End" if "End" in gtf.columns else "end"
    strand_col = "Strand" if "Strand" in gtf.columns else "strand"
    chrom_col = "Chromosome" if "Chromosome" in gtf.columns else "chrom"
    gene_id_col = "gene_id" if "gene_id" in gtf.columns else "GeneID"
    gene_name_col = "gene_name" if "gene_name" in gtf.columns else None

    # Find gene
    gene_mask = gtf[feature_col].str.lower() == "gene"
    if gene_name_col:
        name_match = gtf[gene_name_col] == gene_name
        gene_rows = gtf[gene_mask & name_match]
    else:
        gene_rows = gtf[gene_mask & (gtf[gene_id_col] == gene_name)]

    if gene_rows.empty:
        gene_rows = gtf[gtf[gene_id_col] == gene_name]

    gene_row = gene_rows.iloc[0] if not gene_rows.empty else pd.Series(dtype=object)

    if gene_row.empty:
        g_start, g_end, strand, chrom = 0, 1000, "+", "chr1"
        gene_id_val = gene_name
    else:
        g_start = int(gene_row[start_col]) - flank
        g_end = int(gene_row[end_col]) + flank
        strand = gene_row[strand_col]
        chrom = gene_row[chrom_col]
        gene_id_val = gene_row.get(gene_id_col, gene_name)

    exon_mask = gtf[feature_col].str.lower() == "exon"
    if not gene_row.empty:
        exon_mask = exon_mask & (gtf[gene_id_col] == gene_id_val)
    exons = gtf[exon_mask]

    pas_gene_col = "gene_id" if "gene_id" in pas_sites.columns else gene_id_col
    pas_sub = pas_sites[pas_sites[pas_gene_col] == gene_id_val] if not gene_row.empty else pas_sites

    apa_sub: pd.DataFrame | None = None
    if apa_events is not None and not apa_events.empty:
        apa_gene_col = "gene_id" if "gene_id" in apa_events.columns else gene_id_col
        apa_sub = apa_events[apa_events[apa_gene_col] == gene_id_val]

    # Panels: coverage + (junctions over it) + gene model + PAS/lollipop
    n_bw = len(bigwig_tracks) if bigwig_tracks else 1
    n_panels = n_bw + 2
    height_ratios = [3] * n_bw + [1, 1]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 2.5 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    x_range = np.arange(max(0, g_start), g_end)
    panel_idx = 0

    # Coverage + junction arcs
    if bigwig_tracks:
        from .coverage import extract_bigwig_interval
        for label, bw_path in bigwig_tracks.items():
            ax = axes[panel_idx]
            try:
                cov = extract_bigwig_interval(bw_path, chrom, max(0, g_start), g_end)
                draw_coverage(ax, x_range, cov, label=label)
            except Exception as exc:
                logger.warning("Could not load bigwig '%s': %s", bw_path, exc)
                ax.text(0.5, 0.5, f"[coverage unavailable: {label}]",
                        transform=ax.transAxes, ha="center", va="center")
            if junctions is not None:
                draw_splice_arcs(ax, junctions, y_base=0.0)
            ax.set_ylabel(label, fontsize=8)
            panel_idx += 1
    else:
        ax = axes[panel_idx]
        ax.text(0.5, 0.5, "[no bigWig tracks provided]",
                transform=ax.transAxes, ha="center", va="center", color="grey")
        if junctions is not None:
            draw_splice_arcs(ax, junctions, y_base=0.0)
        panel_idx += 1

    # Gene model
    ax_gene = axes[panel_idx]
    panel_idx += 1
    if not gene_row.empty:
        draw_gene_model(ax_gene, gene_row, exons, strand)
    ax_gene.set_ylabel("Gene model", fontsize=8)
    ax_gene.set_yticks([])

    # PAS + lollipops
    ax_pas = axes[panel_idx]
    if not pas_sub.empty:
        draw_pas_sites(ax_pas, pas_sub, y_level=0.0)
    if apa_sub is not None and not apa_sub.empty:
        if "start" in pas_sub.columns and "site_id" in apa_sub.columns and "site_id" in pas_sub.columns:
            apa_with_pos = apa_sub.merge(
                pas_sub[["site_id", "start"]], on="site_id", how="left"
            )
            draw_apa_lollipops(ax_pas, apa_with_pos, y_level=0.2)
    ax_pas.set_ylabel("PAS / ΔPDUI", fontsize=8)
    ax_pas.set_yticks([])

    title_parts = [gene_name]
    if group_a and group_b:
        title_parts.append(f"{group_a} vs {group_b}")
    fig.suptitle(" · ".join(title_parts), fontsize=11)
    axes[-1].set_xlabel(f"{chrom} (0-based)", fontsize=9)

    plt.tight_layout()
    if output:
        save_figure(fig, output)
    if show:
        plt.show()

    return fig, list(axes)
