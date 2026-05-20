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
        raise ImportError("pyBigWig is required for bigWig operations. Install it with pip/conda.") from exc
    bw_path = Path(bw_path)
    if not bw_path.exists():
        raise FileNotFoundError(f"BigWig file not found: {bw_path}")
    return pyBigWig.open(str(bw_path))


def _resolve_chrom(bw, chrom: str) -> str:
    chroms = bw.chroms()
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in chroms:
        return alt
    raise KeyError(f"Chromosome '{chrom}' not found in bigWig. Available examples: {list(chroms)[:5]}")


def extract_bigwig_interval(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    bins: int | None = None,
    fillna: float = 0.0,
) -> np.ndarray:
    """Extract coverage from a bigWig interval."""
    bw = _open_bigwig(bw_path)
    try:
        bw_chrom = _resolve_chrom(bw, chrom)
        chrom_len = bw.chroms()[bw_chrom]
        start = max(0, int(start))
        end = min(int(end), chrom_len)
        if end <= start:
            return np.zeros(bins or 0, dtype=float)
        vals = bw.stats(bw_chrom, start, end, nBins=bins, type="mean") if bins else bw.values(bw_chrom, start, end)
    finally:
        bw.close()
    arr = np.array(vals, dtype=float)
    return np.where(np.isnan(arr), fillna, arr)


def select_strand_bigwig(track: str | Path | dict, strand: str | None = None) -> str | Path:
    """Select a strand-specific bigWig path from a track mapping."""
    if not isinstance(track, dict):
        return track
    if strand == "-" and track.get("rev"):
        return track["rev"]
    if strand == "+" and track.get("fwd"):
        return track["fwd"]
    return track.get("all") or track.get("fwd") or track.get("rev")


def extract_gene_coverage(
    bw_path: str | Path,
    gene_row: pd.Series,
    flank: int = 1000,
    bins: int | None = None,
) -> np.ndarray:
    """Extract coverage for a gene region with flanking sequence."""
    chrom = gene_row.get("chrom") or gene_row.get("Chromosome")
    start = int(gene_row.get("start") if "start" in gene_row else gene_row.get("Start"))
    end = int(gene_row.get("end") if "end" in gene_row else gene_row.get("End"))
    return extract_bigwig_interval(bw_path, chrom, max(0, start - flank), end + flank, bins=bins)


def extract_scaled_region_coverage(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    n_bins: int = 100,
) -> np.ndarray:
    """Extract coverage scaled to n_bins, reversed for minus-strand regions."""
    arr = extract_bigwig_interval(bw_path, chrom, start, end, bins=n_bins)
    return arr[::-1] if strand == "-" else arr


def aggregate_metagene_coverage(
    bw_paths: list[str | Path],
    regions: pd.DataFrame,
    n_bins: int = 100,
) -> np.ndarray:
    """Aggregate mean coverage across many regions and bigWigs."""
    if not bw_paths or regions.empty:
        return np.zeros(n_bins)
    chrom_col = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col = "start" if "start" in regions.columns else "Start"
    end_col = "end" if "end" in regions.columns else "End"
    strand_col = "strand" if "strand" in regions.columns else "Strand"
    arrays: list[np.ndarray] = []
    for bw_path in bw_paths:
        if not bw_path:
            continue
        for _, row in regions.iterrows():
            try:
                arr = extract_scaled_region_coverage(
                    bw_path, row[chrom_col], int(row[start_col]), int(row[end_col]), row[strand_col], n_bins=n_bins
                )
                if len(arr) == n_bins:
                    arrays.append(arr)
            except Exception as exc:
                logger.warning("Skipping region %s:%s-%s for %s: %s", row[chrom_col], row[start_col], row[end_col], bw_path, exc)
    if not arrays:
        return np.zeros(n_bins)
    return np.nanmean(np.vstack(arrays), axis=0)
