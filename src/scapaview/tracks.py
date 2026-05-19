"""Gene-level APA track plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coverage import extract_bigwig_interval, select_strand_bigwig
from .io import gene_id_base
from .plotting import draw_apa_lollipops, draw_coverage, draw_gene_model, draw_pas_sites, save_figure

logger = logging.getLogger(__name__)


def _feature_col(gtf: pd.DataFrame) -> str:
    return "Feature" if "Feature" in gtf.columns else "feature"


def _find_gene(gtf: pd.DataFrame, gene_name: str) -> pd.Series:
    feature_col = _feature_col(gtf)
    genes = gtf[gtf[feature_col].astype(str).str.lower() == "gene"].copy()
    if genes.empty:
        return pd.Series(dtype=object)
    masks = []
    if "gene_name" in genes.columns:
        masks.append(genes["gene_name"].astype(str) == gene_name)
    if "gene_id" in genes.columns:
        masks.append(genes["gene_id"].astype(str) == gene_name)
        masks.append(genes["gene_id"].map(gene_id_base) == gene_id_base(gene_name))
    if "gene_id_base" in genes.columns:
        masks.append(genes["gene_id_base"].astype(str) == gene_id_base(gene_name))
    for mask in masks:
        hit = genes[mask]
        if not hit.empty:
            return hit.iloc[0]
    return pd.Series(dtype=object)


def _subset_by_gene(df: pd.DataFrame, gene_row: pd.Series) -> pd.DataFrame:
    if df is None or df.empty or gene_row.empty:
        return pd.DataFrame()
    gid = gene_row.get("gene_id")
    gid_base = gene_row.get("gene_id_base", gene_id_base(gid))
    if "gene_id_base" in df.columns:
        return df[df["gene_id_base"].astype(str) == str(gid_base)]
    if "gene_id" in df.columns:
        return df[df["gene_id"].map(gene_id_base) == str(gid_base)]
    return pd.DataFrame()


def _save_png_pdf(fig: plt.Figure, output: str | Path) -> None:
    output = Path(output)
    save_figure(fig, output)
    if output.suffix.lower() != ".pdf":
        save_figure(fig, output.with_suffix(".pdf"))


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
    """Plot coverage, gene model, PAS sites, and delta-PDUI events for one gene."""
    gene_row = _find_gene(gtf, gene_name)
    start_col = "Start" if "Start" in gtf.columns else "start"
    end_col = "End" if "End" in gtf.columns else "end"
    strand_col = "Strand" if "Strand" in gtf.columns else "strand"
    chrom_col = "Chromosome" if "Chromosome" in gtf.columns else "chrom"
    feature_col = _feature_col(gtf)

    if gene_row.empty:
        chrom, strand, g_start, g_end = "chr1", "+", 0, 1000
        exons = pd.DataFrame()
        logger.warning("Gene '%s' not found in GTF; generating empty plot.", gene_name)
    else:
        chrom = gene_row[chrom_col]
        strand = gene_row[strand_col]
        g_start = max(0, int(gene_row[start_col]) - flank)
        g_end = int(gene_row[end_col]) + flank
        gid_base = gene_row.get("gene_id_base", gene_id_base(gene_row.get("gene_id")))
        exons = gtf[(gtf[feature_col].astype(str).str.lower() == "exon") & (gtf.get("gene_id_base", gtf.get("gene_id")).map(gene_id_base) == gid_base)]

    pas_sub = _subset_by_gene(pas_sites, gene_row)
    apa_sub = _subset_by_gene(apa_events, gene_row) if apa_events is not None else pd.DataFrame()

    tracks = bigwig_tracks or {}
    n_bw = max(len(tracks), 1)
    n_panels = n_bw + 2
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, max(6, 1.8 * n_panels)),
        sharex=True,
        gridspec_kw={"height_ratios": [2] * n_bw + [1, 1.2]},
    )
    axes = list(np.atleast_1d(axes))
    x_range = np.arange(g_start, g_end)

    panel_idx = 0
    if tracks:
        for label, track in tracks.items():
            ax = axes[panel_idx]
            bw_path = select_strand_bigwig(track, strand=strand)
            try:
                cov = extract_bigwig_interval(bw_path, chrom, g_start, g_end)
                draw_coverage(ax, x_range[: len(cov)], cov, label=label)
                ax.legend(loc="upper right", fontsize=8)
            except Exception as exc:
                logger.warning("Could not load bigWig for %s: %s", label, exc)
                ax.text(0.5, 0.5, f"coverage unavailable: {label}", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel(label, fontsize=8)
            panel_idx += 1
    else:
        axes[panel_idx].text(0.5, 0.5, "no bigWig tracks", transform=axes[panel_idx].transAxes, ha="center", va="center", color="grey")
        panel_idx += 1

    ax_gene = axes[panel_idx]
    panel_idx += 1
    if not gene_row.empty:
        draw_gene_model(ax_gene, gene_row, exons, strand)
    ax_gene.set_ylabel("gene", fontsize=8)
    ax_gene.set_yticks([])

    ax_pas = axes[panel_idx]
    if not pas_sub.empty:
        draw_pas_sites(ax_pas, pas_sub, y_level=0.0)
    else:
        ax_pas.text(0.02, 0.75, "no PAS sites for gene", transform=ax_pas.transAxes, fontsize=8, color="grey")
    if not apa_sub.empty and not pas_sub.empty and {"site_id", "start"}.issubset(pas_sub.columns):
        site_pos = pas_sub[["site_id", "start"]].copy()
        site_pos["site_id"] = site_pos["site_id"].astype(str)
        apa_plot = apa_sub.copy()
        apa_plot["site_id"] = apa_plot["site_id"].astype(str)
        apa_plot = apa_plot.merge(site_pos, on="site_id", how="left")
        if apa_plot["start"].notna().any():
            draw_apa_lollipops(ax_pas, apa_plot.dropna(subset=["start"]), y_level=0.1, max_height=0.8)
    elif apa_sub.empty:
        ax_pas.text(0.02, 0.55, "no APA events for selected comparison/gene", transform=ax_pas.transAxes, fontsize=8, color="grey")
    ax_pas.axhline(0, color="black", linewidth=0.5)
    ax_pas.set_ylabel("PAS / dPDUI", fontsize=8)
    ax_pas.set_yticks([])

    title = [gene_name]
    if group_a and group_b:
        title.append(f"{group_a} vs {group_b}")
    if celltype:
        title.append(celltype)
    fig.suptitle(" | ".join(title), fontsize=12)
    axes[-1].set_xlabel(f"{chrom} (0-based)", fontsize=9)
    axes[-1].set_xlim(g_start, g_end)
    plt.tight_layout()

    if output:
        _save_png_pdf(fig, output)
    if show:
        plt.show()
    return fig, axes
