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
    """Compute strand-aware PAS density across scaled regions."""
    if pas_sites is None or pas_sites.empty or regions is None or regions.empty:
        return np.zeros(n_bins)

    pas = pas_sites.copy()
    if source_col is not None and "source" in pas.columns:
        pas = pas[pas["source"].astype(str).str.contains(source_col, na=False)]
    if pas.empty:
        return np.zeros(n_bins)

    chrom_col_r = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col_r = "start" if "start" in regions.columns else "Start"
    end_col_r = "end" if "end" in regions.columns else "End"
    strand_col_r = "strand" if "strand" in regions.columns else "Strand"
    chrom_col_p = "chrom" if "chrom" in pas.columns else "Chromosome"
    start_col_p = "start" if "start" in pas.columns else "Start"

    density_sum = np.zeros(n_bins)
    n_regions = 0
    pas_by_chrom = {chrom: grp for chrom, grp in pas.groupby(chrom_col_p, observed=True)}

    for _, reg in regions.iterrows():
        r_start = int(reg[start_col_r])
        r_end = int(reg[end_col_r])
        span = r_end - r_start
        if span <= 0:
            continue
        pas_in = pas_by_chrom.get(reg[chrom_col_r], pd.DataFrame())
        if not pas_in.empty:
            pas_in = pas_in[(pas_in[start_col_p] >= r_start) & (pas_in[start_col_p] < r_end)]
        region_density = np.zeros(n_bins)
        for _, pas_row in pas_in.iterrows():
            rel = (int(pas_row[start_col_p]) - r_start) / span
            if reg[strand_col_r] == "-":
                rel = 1.0 - rel
            bin_idx = min(max(int(rel * n_bins), 0), n_bins - 1)
            region_density[bin_idx] += 1
        density_sum += region_density
        n_regions += 1

    return density_sum / n_regions if n_regions else np.zeros(n_bins)


def _save_png_pdf(fig: plt.Figure, output: str | Path) -> None:
    output = Path(output)
    save_figure(fig, output)
    if output.suffix.lower() != ".pdf":
        save_figure(fig, output.with_suffix(".pdf"))


def _plot_metagene_coverage(
    ax: plt.Axes,
    bw_paths: list,
    regions: pd.DataFrame,
    n_bins: int,
    x: np.ndarray,
    label: str,
    color: str,
) -> np.ndarray:
    """Aggregate coverage and draw a line/fill metagene trace."""
    from .coverage import aggregate_metagene_coverage

    try:
        cov = aggregate_metagene_coverage([p for p in bw_paths if p], regions, n_bins=n_bins)
        draw_coverage(ax, x, cov, label=label, color=color, alpha=0.25)
        ax.plot(x, cov, color=color, linewidth=1.4)
        return cov
    except Exception as exc:
        logger.warning("Could not compute metagene coverage for '%s': %s", label, exc)
        ax.text(0.5, 0.5, f"coverage unavailable: {label}", transform=ax.transAxes, ha="center", va="center")
        return np.zeros(n_bins)


def plot_metagene_3utr(
    bw_paths_a: list,
    bw_paths_b: list | None,
    regions: pd.DataFrame,
    pas_sites: pd.DataFrame | None = None,
    apa_events: pd.DataFrame | None = None,
    n_bins: int = 100,
    output: str | Path | None = None,
    show: bool = True,
    label_a: str = "Group A",
    label_b: str = "Group B",
    comparison_label: str | None = None,
    celltype: str | None = None,
    gene_set_name: str | None = None,
    plot_delta: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot terminal-exon/3' region metagene coverage with optional PAS density and delta."""
    x = np.linspace(0, 1, n_bins)
    has_delta = bool(plot_delta and bw_paths_b)
    if has_delta:
        fig, (ax, ax_delta) = plt.subplots(
            2,
            1,
            figsize=(8, 5.4),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax_delta = None

    cov_a = _plot_metagene_coverage(ax, bw_paths_a or [], regions, n_bins, x, label=label_a, color="steelblue")
    cov_b = None
    if bw_paths_b:
        cov_b = _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label=label_b, color="tomato")

    if pas_sites is not None and not pas_sites.empty:
        density = compute_pas_density_metagene(pas_sites, regions, n_bins=n_bins)
        ax2 = ax.twinx()
        ax2.plot(x, density, color="black", linestyle="--", linewidth=1, alpha=0.7, label="PAS density")
        ax2.set_ylabel("PAS / region", fontsize=9)

    title_parts = ["Metagene: terminal exon / 3' region"]
    for part in (comparison_label, celltype, gene_set_name):
        if part:
            title_parts.append(part)
    ax.set_title(" | ".join(title_parts), fontsize=11)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)

    if ax_delta is not None and cov_b is not None:
        delta = cov_b - cov_a
        delta_label = f"{label_b} - {label_a} coverage"
        ax_delta.axhline(0, color="0.25", linewidth=0.8)
        ax_delta.fill_between(x, delta, 0, where=delta >= 0, color="tomato", alpha=0.35, interpolate=True)
        ax_delta.fill_between(x, delta, 0, where=delta < 0, color="steelblue", alpha=0.35, interpolate=True)
        ax_delta.plot(x, delta, color="0.2", linewidth=1.0, label=delta_label)
        ax_delta.set_ylabel(f"{label_b} - {label_a}", fontsize=9)
        ax_delta.legend(fontsize=8, loc="upper left")
        ax_delta.set_xlabel("Relative position (5' to 3')", fontsize=10)
    else:
        ax.set_xlabel("Relative position (5' to 3')", fontsize=10)

    plt.tight_layout()
    if output:
        _save_png_pdf(fig, output)
    if show:
        plt.show()
    return fig, ax


def plot_metagene_gene_body(
    bw_paths_a: list,
    bw_paths_b: list | None,
    regions: pd.DataFrame,
    n_bins: int = 100,
    output: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot scaled gene-body metagene coverage."""
    x = np.linspace(0, 1, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_metagene_coverage(ax, bw_paths_a or [], regions, n_bins, x, label="Group A", color="steelblue")
    if bw_paths_b:
        _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label="Group B", color="tomato")
    ax.set_xlabel("Relative position (TSS to TES)", fontsize=10)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.set_title("Metagene: gene body", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    if output:
        _save_png_pdf(fig, output)
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
    """Plot metagene coverage centred on splice-site windows."""
    x = np.linspace(-flank, flank, n_bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_metagene_coverage(ax, bw_paths_a or [], regions, n_bins, x, label="Group A", color="steelblue")
    if bw_paths_b:
        _plot_metagene_coverage(ax, bw_paths_b, regions, n_bins, x, label="Group B", color="tomato")
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Distance from splice site (bp)", fontsize=10)
    ax.set_ylabel("Mean coverage", fontsize=10)
    ax.set_title("Metagene: splice sites", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    if output:
        _save_png_pdf(fig, output)
    if show:
        plt.show()
    return fig, ax
