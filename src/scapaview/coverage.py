"""Coverage extraction utilities for bigWig and bedGraph files."""

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
        raise FileNotFoundError(f"Coverage file not found: {bw_path}")
    return pyBigWig.open(str(bw_path))


def _resolve_chrom(bw, chrom: str) -> str:
    chroms = bw.chroms()
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in chroms:
        return alt
    raise KeyError(f"Chromosome '{chrom}' not found in bigWig. Available examples: {list(chroms)[:5]}")


def _extract_bedgraph_interval(path: Path, chrom: str, start: int, end: int, bins: int | None, fillna: float) -> np.ndarray:
    length = max(0, end - start)
    if length == 0:
        return np.zeros(bins or 0, dtype=float)
    if bins is None:
        arr = np.full(length, fillna, dtype=float)
        for chunk in pd.read_csv(path, sep="	", header=None, names=["chrom", "start", "end", "value"], chunksize=500_000):
            sub = chunk[(chunk["chrom"] == chrom) & (chunk["end"] > start) & (chunk["start"] < end)]
            for _, row in sub.iterrows():
                s = max(start, int(row["start"]))
                e = min(end, int(row["end"]))
                arr[s - start:e - start] = float(row["value"])
        return arr

    sums = np.zeros(bins, dtype=float)
    widths = np.zeros(bins, dtype=float)
    scale = bins / length
    for chunk in pd.read_csv(path, sep="	", header=None, names=["chrom", "start", "end", "value"], chunksize=500_000):
        sub = chunk[(chunk["chrom"] == chrom) & (chunk["end"] > start) & (chunk["start"] < end)]
        for _, row in sub.iterrows():
            s = max(start, int(row["start"]))
            e = min(end, int(row["end"]))
            if e <= s:
                continue
            b0 = min(int((s - start) * scale), bins - 1)
            b1 = min(int(np.ceil((e - start) * scale)), bins)
            for b in range(b0, b1):
                bs = start + int(b / scale)
                be = start + int((b + 1) / scale)
                ov = max(0, min(e, be) - max(s, bs))
                if ov:
                    sums[b] += float(row["value"]) * ov
                    widths[b] += ov
    return np.divide(sums, widths, out=np.full(bins, fillna, dtype=float), where=widths > 0)


def extract_bigwig_interval(
    bw_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    bins: int | None = None,
    fillna: float = 0.0,
) -> np.ndarray:
    """Extract coverage from a bigWig or bedGraph interval."""
    path = Path(bw_path)
    if path.suffix.lower() in {".bedgraph", ".bdg"}:
        return _extract_bedgraph_interval(path, chrom, max(0, int(start)), int(end), bins, fillna)

    bw = _open_bigwig(path)
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
    """Select a strand-specific coverage path from a track mapping."""
    if not isinstance(track, dict):
        return track
    if strand == "-" and track.get("rev"):
        return _prefer_nonempty_coverage(track["rev"])
    if strand == "+" and track.get("fwd"):
        return _prefer_nonempty_coverage(track["fwd"])
    return _prefer_nonempty_coverage(track.get("all") or track.get("fwd") or track.get("rev"))


def _prefer_nonempty_coverage(path: str | Path | None) -> str | Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return p
    if p.suffix == ".bw":
        bg = p.with_suffix(".bedGraph")
        if bg.exists() and bg.stat().st_size > 0:
            return bg
    return p


def extract_gene_coverage(bw_path: str | Path, gene_row: pd.Series, flank: int = 1000, bins: int | None = None) -> np.ndarray:
    """Extract coverage for a gene region with flanking sequence."""
    chrom = gene_row.get("chrom") or gene_row.get("Chromosome")
    start = int(gene_row.get("start") if "start" in gene_row else gene_row.get("Start"))
    end = int(gene_row.get("end") if "end" in gene_row else gene_row.get("End"))
    return extract_bigwig_interval(bw_path, chrom, max(0, start - flank), end + flank, bins=bins)


def extract_scaled_region_coverage(bw_path: str | Path, chrom: str, start: int, end: int, strand: str, n_bins: int = 100) -> np.ndarray:
    """Extract coverage scaled to n_bins, reversed for minus-strand regions."""
    arr = extract_bigwig_interval(bw_path, chrom, start, end, bins=n_bins)
    return arr[::-1] if strand == "-" else arr


def _aggregate_bedgraph_coverage(path: Path, regions: pd.DataFrame, n_bins: int) -> list[np.ndarray]:
    chrom_col = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col = "start" if "start" in regions.columns else "Start"
    end_col = "end" if "end" in regions.columns else "End"
    strand_col = "strand" if "strand" in regions.columns else "Strand"
    regs = regions.reset_index(drop=True).copy()
    sums = np.zeros((len(regs), n_bins), dtype=float)
    widths = np.zeros((len(regs), n_bins), dtype=float)
    by_chrom = {chrom: grp.index.to_numpy() for chrom, grp in regs.groupby(chrom_col, observed=True)}

    for chunk in pd.read_csv(path, sep="	", header=None, names=["chrom", "start", "end", "value"], chunksize=500_000):
        for chrom, idxs in by_chrom.items():
            sub_chunk = chunk[chunk["chrom"] == chrom]
            if sub_chunk.empty:
                continue
            for i in idxs:
                reg = regs.loc[i]
                r_start = int(reg[start_col]); r_end = int(reg[end_col]); span = r_end - r_start
                if span <= 0:
                    continue
                overlaps = sub_chunk[(sub_chunk["end"] > r_start) & (sub_chunk["start"] < r_end)]
                scale = n_bins / span
                for _, row in overlaps.iterrows():
                    s = max(r_start, int(row["start"])); e = min(r_end, int(row["end"]))
                    if e <= s:
                        continue
                    b0 = min(int((s - r_start) * scale), n_bins - 1)
                    b1 = min(int(np.ceil((e - r_start) * scale)), n_bins)
                    for b in range(b0, b1):
                        bs = r_start + int(b / scale); be = r_start + int((b + 1) / scale)
                        ov = max(0, min(e, be) - max(s, bs))
                        if ov:
                            target_b = n_bins - 1 - b if reg[strand_col] == "-" else b
                            sums[i, target_b] += float(row["value"]) * ov
                            widths[i, target_b] += ov
    arrays = [np.divide(sums[i], widths[i], out=np.zeros(n_bins), where=widths[i] > 0) for i in range(len(regs))]
    return arrays


def aggregate_metagene_coverage(bw_paths: list[str | Path], regions: pd.DataFrame, n_bins: int = 100) -> np.ndarray:
    """Aggregate mean coverage across many regions and coverage files."""
    if not bw_paths or regions.empty:
        return np.zeros(n_bins)
    chrom_col = "chrom" if "chrom" in regions.columns else "Chromosome"
    start_col = "start" if "start" in regions.columns else "Start"
    end_col = "end" if "end" in regions.columns else "End"
    strand_col = "strand" if "strand" in regions.columns else "Strand"
    arrays: list[np.ndarray] = []
    for raw_path in bw_paths:
        if not raw_path:
            continue
        path = Path(_prefer_nonempty_coverage(raw_path))
        try:
            if path.suffix.lower() in {".bedgraph", ".bdg"}:
                arrays.extend(_aggregate_bedgraph_coverage(path, regions, n_bins=n_bins))
                continue
            for _, row in regions.iterrows():
                arr = extract_scaled_region_coverage(path, row[chrom_col], int(row[start_col]), int(row[end_col]), row[strand_col], n_bins=n_bins)
                if len(arr) == n_bins:
                    arrays.append(arr)
        except Exception as exc:
            logger.warning("Skipping coverage file %s: %s", path, exc)
    if not arrays:
        return np.zeros(n_bins)
    return np.nanmean(np.vstack(arrays), axis=0)
