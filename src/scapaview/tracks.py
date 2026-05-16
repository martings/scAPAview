"""Gene-level APA track plots."""

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
    save_figure,
)

logger = logging.getLogger(__name__)


def plot_gene_apa_tracks(
    gene_name: str,
    gtf: pd.DataFrame,
    pas_sites: pd.DataFrame,
    apa_events: pd.DataFrame | None = None,
    bigwig_tracks: dict | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    celltype: str | None = None,
    flank: int = 1000,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot gene-level APA tracks.

    Panels (from top to bottom):
    1. Coverage tracks (one per bigwig group, or empty placeholder)
    2. Gene model with exons and intron backbone
    3. PAS site marks coloured by source
    4. APA event lollipops coloured by ΔPDUI

    Parameters
    ----------
    gene_name     : HGNC symbol or gene_id to look up
    gtf           : standardised GTF DataFrame (0-based)
    pas_sites     : PAS sites DataFrame
    apa_events    : APA events DataFrame (optional)
    bigwig_tracks : dict mapping label → bw_path (or dict with group/celltype keys)
    group_a/b     : group labels for display
    celltype      : restrict to a specific cell type bigwig
    flank         : bp to extend left/right of gene
    output        : path to save figure (PNG/PDF)
    show          : call plt.show() if True

    Returns
    -------
    (fig, axes) tuple
    """
    # Identify gene in GTF
    gene_name_col = "gene_name" if "gene_name" in gtf.columns else None
    gene_id_col = "gene_id" if "gene_id" in gtf.columns else "GeneID"
    feature_col = "Feature" if "Feature" in gtf.columns else "feature"

    gene_mask = (gtf[feature_col].str.lower() == "gene") & (
        (gtf.get(gene_name_col, pd.Series(dtype=str)) == gene_name)
        if gene_name_col
        else (gtf[gene_id_col] == gene_name)
    )
    gene_rows = gtf[gene_mask]
    if gene_rows.empty:
        gene_rows = gtf[gtf[gene_id_col] == gene_name]
    if gene_rows.empty:
        logger.warning("Gene '%s' not found in GTF; producing empty plot.", gene_name)

    gene_row = gene_rows.iloc[0] if not gene_rows.empty else pd.Series(dtype=object)

    start_col = "Start" if "Start" in gtf.columns else "start"
    end_col = "End" if "End" in gtf.columns else "end"
    strand_col = "Strand" if "Strand" in gtf.columns else "strand"
    chrom_col = "Chromosome" if "Chromosome" in gtf.columns else "chrom"

    # Gene coordinates
    if gene_row.empty:
        g_start, g_end, strand, chrom = 0, 1000, "+", "chr1"
    else:
        g_start = int(gene_row[start_col]) - flank
        g_end = int(gene_row[end_col]) + flank
        strand = gene_row[strand_col]
        chrom = gene_row[chrom_col]

    # Exons for this gene
    exon_mask = (gtf[feature_col].str.lower() == "exon")
    if not gene_row.empty and gene_id_col in gtf.columns:
        gene_id_val = gene_row.get(gene_id_col)
        if gene_id_val is not None:
            exon_mask = exon_mask & (gtf[gene_id_col] == gene_id_val)
    exons = gtf[exon_mask]

    # PAS sites for this gene
    pas_gene_col = "gene_id" if "gene_id" in pas_sites.columns else gene_id_col
    pas_sub = pas_sites
    if not gene_row.empty and pas_gene_col in pas_sites.columns:
        gene_id_val = gene_row.get(gene_id_col)
        pas_sub = pas_sites[pas_sites[pas_gene_col] == gene_id_val]

    # APA events
    apa_sub = None
    if apa_events is not None and not apa_events.empty:
        apa_gene_col = "gene_id" if "gene_id" in apa_events.columns else gene_id_col
        if not gene_row.empty and apa_gene_col in apa_events.columns:
            gene_id_val = gene_row.get(gene_id_col)
            apa_sub = apa_events[apa_events[apa_gene_col] == gene_id_val]

    # Build panel list
    n_bw = len(bigwig_tracks) if bigwig_tracks else 1
    n_panels = n_bw + 2  # coverage panels + gene model + PAS+lollipop

    height_ratios = [2] * n_bw + [1, 1]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 2 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]

    x_range = np.arange(max(0, g_start), g_end)

    # Coverage panels
    panel_idx = 0
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
            ax.set_ylabel(label, fontsize=8)
            panel_idx += 1
    else:
        ax = axes[panel_idx]
        ax.text(0.5, 0.5, "[no bigWig tracks provided]",
                transform=ax.transAxes, ha="center", va="center", color="grey")
        panel_idx += 1

    # Gene model panel
    ax_gene = axes[panel_idx]
    panel_idx += 1
    if not gene_row.empty:
        draw_gene_model(ax_gene, gene_row, exons, strand)
    ax_gene.set_ylabel("Gene model", fontsize=8)
    ax_gene.set_yticks([])

    # PAS + lollipop panel
    ax_pas = axes[panel_idx]
    if not pas_sub.empty:
        draw_pas_sites(ax_pas, pas_sub, y_level=0.0)
    if apa_sub is not None and not apa_sub.empty:
        # Try to join pas position to apa events
        if "start" in pas_sub.columns and "site_id" in apa_sub.columns and "site_id" in pas_sub.columns:
            apa_with_pos = apa_sub.merge(
                pas_sub[["site_id", "start"]],
                on="site_id", how="left",
            )
            draw_apa_lollipops(ax_pas, apa_with_pos, y_level=0.2)
    ax_pas.set_ylabel("PAS / ΔPDUI", fontsize=8)
    ax_pas.set_yticks([])

    title_parts = [gene_name]
    if group_a and group_b:
        title_parts.append(f"{group_a} vs {group_b}")
    if celltype:
        title_parts.append(f"[{celltype}]")
    fig.suptitle(" · ".join(title_parts), fontsize=11)
    axes[-1].set_xlabel(f"{chrom} (0-based)", fontsize=9)

    plt.tight_layout()

    if output:
        save_figure(fig, output)
    if show:
        plt.show()

    return fig, list(axes)
