"""Metagene analysis and plotting utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import draw_coverage, save_figure

logger = logging.getLogger(__name__)


def compute_pas_density_metagene(
    pas_sites: pd.DataFrame,
    regions: pd.DataFrame,
    n_bins: int = 100,
    source_col: str | None = None,
) -> np.ndarray:
    """Compute PAS site density across n_bins for a metagene plot.

    For each region, maps PAS sites falling within the region to a bin
    (strand-aware), then averages across all regions.

    Parameters
    ----------
    pas_sites  : PAS site DataFrame with chrom/start/strand columns
    regions    : DataFrame with chrom/start/end/strand columns
    n_bins     : number of bins
    source_col : if provided, filter PAS sites to this source value

    Returns
    -------
    np.ndarray of shape (n_bins,) with mean PAS density per bin
    """
    if pas_sites.empty or regions.empty:
        return np.zeros(n_bins)

    if source_col is not None and source_col in pas_sites.columns:
        pas_sites = pas_sites[pas_sites["source"] == source_col]
    if pas_sites.empty:
        return np.zeros(n_bins)

    chrom_col_r = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col_r = "start" if "start" in regions.columns else "Start"
    end_col_r = "end" if "end" in regions.columns else "End"
    strand_col_r = "strand" if "strand" in regions.columns else "Strand"

    chrom_col_p = "chrom" if "chrom" in pas_sites.columns else "Chromosome"
    start_col_p = "start" if "start" in pas_sites.columns else "Start"
    strand_col_p = "strand" if "strand" in pas_sites.columns else "Strand"

    density_sum = np.zeros(n_bins)
    n_regions = 0

    for _, reg in regions.iterrows():
        r_chrom = reg[chrom_col_r]
        r_start = int(reg[start_col_r])
        r_end = int(reg[end_col_r])
        r_strand = reg[strand_col_r]
        span = r_end - r_start
        if span <= 0:
            continue

        pas_in = pas_sites[
            (pas_sites[chrom_col_p] == r_chrom)
            & (pas_sites[start_col_p] >= r_start)
            & (pas_sites[start_col_p] < r_end)
        ]

        region_density = np.zeros(n_bins)
        for _, pas_row in pas_in.iterrows():
            pos = int(pas_row[start_col_p])
            rel = (pos - r_start) / span
            if r_strand == "-":
                rel = 1.0 - rel
            bin_idx = min(int(rel * n_bins), n_bins - 1)
            region_density[bin_idx] += 1

        density_sum += region_density
        n_regions += 1

    if n_regions == 0:
        return np.zeros(n_bins)
    return density_sum / n_regions


def plot_metagene_3utr(
    bw_paths_a: list,
    bw_paths_b: list | None,
    regions: pd.DataFrame,
    pas_sites: pd.DataFrame | None = None,
    apa_events: pd.DataFrame | None = None,
    n_bins: int = 100,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot metagene coverage across 3' UTR regions.

    Parameters
    ----------
    bw_paths_a  : list of bigWig paths for group A
    bw_paths_b  : list of bigWig paths for group B (optional)
    regions     : 3' UTR regions DataFrame
    pas_sites   : optional PAS density overlay
    apa_events  : optional APA event overlay
    n_bins      : metagene resolution
    output      : save path
    show        : call plt.show()
    """
    x = np.linspace(0, 1, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))

    _plot_metagene_coverage(ax, bw_paths_a, regions, n_bins, x, label="Group A", color="steelblue")
    if bw_paths_b:
        _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label="Group B", color="tomato")

    if pas_sites is not None and not pas_sites.empty:
        density = compute_pas_density_metagene(pas_sites, regions, n_bins=n_bins)
        ax2 = ax.twinx()
        ax2.plot(x, density, color="black", linestyle="--", linewidth=1, alpha=0.6, label="PAS density")
        ax2.set_ylabel("PAS density (per region)", fontsize=9)

    ax.set_xlabel("Relative position (5' → 3')", fontsize=10)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.set_title("Metagene: 3' UTR", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if output:
        save_figure(fig, output)
    if show:
        plt.show()
    return fig, ax


def _plot_metagene_coverage(
    ax: plt.Axes,
    bw_paths: list,
    regions: pd.DataFrame,
    n_bins: int,
    x: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Helper: aggregate coverage and draw on ax."""
    from .coverage import aggregate_metagene_coverage

    try:
        cov = aggregate_metagene_coverage(bw_paths, regions, n_bins=n_bins)
        draw_coverage(ax, x, cov, label=label, color=color)
    except Exception as exc:
        logger.warning("Could not compute metagene coverage for '%s': %s", label, exc)
        ax.text(0.5, 0.5, f"[coverage unavailable: {label}]",
                transform=ax.transAxes, ha="center", va="center")


def plot_metagene_gene_body(
    bw_paths_a: list,
    bw_paths_b: list | None,
    regions: pd.DataFrame,
    n_bins: int = 100,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot metagene coverage across scaled gene bodies."""
    x = np.linspace(0, 1, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))

    _plot_metagene_coverage(ax, bw_paths_a, regions, n_bins, x, label="Group A", color="steelblue")
    if bw_paths_b:
        _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label="Group B", color="tomato")

    ax.set_xlabel("Relative position (TSS → TES)", fontsize=10)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.set_title("Metagene: Gene body", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if output:
        save_figure(fig, output)
    if show:
        plt.show()
    return fig, ax


def plot_metagene_splice_sites(
    bw_paths_a: list,
    bw_paths_b: list | None,
    regions: pd.DataFrame,
    n_bins: int = 100,
    flank: int = 500,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot metagene coverage centred on splice sites."""
    x = np.linspace(-flank, flank, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))

    _plot_metagene_coverage(ax, bw_paths_a, regions, n_bins, x, label="Group A", color="steelblue")
    if bw_paths_b:
        _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label="Group B", color="tomato")

    ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.8)
    ax.set_xlabel("Distance from splice site (bp)", fontsize=10)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.set_title("Metagene: Splice sites", fontsize=11)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output:
        save_figure(fig, output)
    if show:
        plt.show()
    return fig, ax
