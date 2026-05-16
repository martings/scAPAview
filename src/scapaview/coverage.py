"""BigWig coverage extraction utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _open_bigwig(bw_path: str | Path):
    """Open a bigWig file, raising helpful errors if unavailable."""
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyBigWig is required for bigWig operations. "
            "Install it with: pip install pyBigWig"
        ) from exc

    bw_path = Path(bw_path)
    if not bw_path.exists():
        raise FileNotFoundError(f"BigWig file not found: {bw_path}")
    return pyBigWig.open(str(bw_path))


def extract_bigwig_interval(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    bins: int | None = None,
    fillna: float = 0.0,
) -> np.ndarray:
    """Extract coverage from a bigWig file for a genomic interval.

    Parameters
    ----------
    bw_path : path to bigWig file
    chrom   : chromosome name
    start   : 0-based start position
    end     : 0-based exclusive end position
    bins    : if provided, scale to this many bins
    fillna  : value for missing data (default 0.0)
    """
    bw = _open_bigwig(bw_path)
    try:
        if bins is not None:
            vals = bw.stats(chrom, start, end, nBins=bins, type="mean")
        else:
            vals = bw.values(chrom, start, end)
    finally:
        bw.close()

    arr = np.array(vals, dtype=float)
    arr = np.where(np.isnan(arr), fillna, arr)
    return arr


def extract_gene_coverage(
    bw_path: str | Path,
    gene_row: pd.Series,
    flank: int = 1000,
    bins: int | None = None,
) -> np.ndarray:
    """Extract coverage for a gene region with flanking sequence.

    Expects gene_row to have: chrom/Chromosome, start/Start, end/End columns.
    """
    chrom = gene_row.get("chrom") or gene_row.get("Chromosome")
    start = int(gene_row.get("start") or gene_row.get("Start"))
    end = int(gene_row.get("end") or gene_row.get("End"))
    return extract_bigwig_interval(
        bw_path, chrom, max(0, start - flank), end + flank, bins=bins
    )


def extract_scaled_region_coverage(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    n_bins: int = 100,
) -> np.ndarray:
    """Extract and scale coverage to n_bins for a region.

    For − strand genes the array is reversed so index 0 = TSS.
    """
    arr = extract_bigwig_interval(bw_path, chrom, start, end, bins=n_bins)
    if strand == "-":
        arr = arr[::-1]
    return arr


def aggregate_metagene_coverage(
    bw_paths: list[str | Path],
    regions: pd.DataFrame,
    n_bins: int = 100,
) -> np.ndarray:
    """Aggregate coverage across many regions for metagene analysis.

    Returns an array of shape (n_bins,) with mean coverage across all regions
    and all bigWig files provided.

    Parameters
    ----------
    bw_paths : list of bigWig file paths
    regions  : DataFrame with columns chrom/start/end/strand (0-based)
    n_bins   : number of bins for scaling
    """
    chrom_col = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col = "start" if "start" in regions.columns else "Start"
    end_col = "end" if "end" in regions.columns else "End"
    strand_col = "strand" if "strand" in regions.columns else "Strand"

    all_arrays: list[np.ndarray] = []
    for bw_path in bw_paths:
        for _, row in regions.iterrows():
            try:
                arr = extract_scaled_region_coverage(
                    bw_path,
                    row[chrom_col],
                    int(row[start_col]),
                    int(row[end_col]),
                    row[strand_col],
                    n_bins=n_bins,
                )
                all_arrays.append(arr)
            except Exception as exc:
                logger.warning("Skipping region %s:%s-%s due to error: %s",
                               row[chrom_col], row[start_col], row[end_col], exc)

    if not all_arrays:
        return np.zeros(n_bins)
    return np.nanmean(np.vstack(all_arrays), axis=0)
