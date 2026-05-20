"""Genomic annotation utilities for scAPAview.

Internal coordinates are 0-based half-open. GTF inputs are converted from
1-based inclusive by subtracting 1 from Start and leaving End unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .io import add_gene_id_columns, gene_id_base, normalize_chromosome

logger = logging.getLogger(__name__)

PAS_CONTEXT_CATEGORIES = [
    "5UTR", "CDS", "intron", "last_intron", "terminal_exon", "3UTR",
    "downstream_TES", "intergenic", "unknown",
]


def _col(df: pd.DataFrame, lower: str, upper: str) -> str:
    if lower in df.columns:
        return lower
    if upper in df.columns:
        return upper
    raise KeyError(f"Expected column '{lower}' or '{upper}'")


def standardize_gtf(gtf: pd.DataFrame) -> pd.DataFrame:
    """Convert GTF coordinates to 0-based half-open and add helper columns."""
    out = gtf.copy()
    if "Start" in out.columns:
        out["Start"] = pd.to_numeric(out["Start"], errors="coerce") - 1
        out["Start"] = out["Start"].astype(int)
    elif "start" in out.columns:
        out["start"] = pd.to_numeric(out["start"], errors="coerce") - 1
        out["start"] = out["start"].astype(int)
    if "End" in out.columns:
        out["End"] = pd.to_numeric(out["End"], errors="coerce").astype(int)
    elif "end" in out.columns:
        out["end"] = pd.to_numeric(out["end"], errors="coerce").astype(int)
    chrom_col = "Chromosome" if "Chromosome" in out.columns else "chrom" if "chrom" in out.columns else None
    if chrom_col:
        out[chrom_col] = out[chrom_col].map(normalize_chromosome)
    if "gene_id" in out.columns:
        out = add_gene_id_columns(out)
    return out


def build_gene_table(gtf: pd.DataFrame) -> pd.DataFrame:
    """Extract gene-level records from a GTF DataFrame."""
    feature_col = _col(gtf, "feature", "Feature")
    genes = gtf.loc[gtf[feature_col].astype(str).str.lower() == "gene"].copy()
    if "gene_id" in genes.columns and "gene_id_base" not in genes.columns:
        genes = add_gene_id_columns(genes)
    return genes.reset_index(drop=True)


def build_exon_table(gtf: pd.DataFrame) -> pd.DataFrame:
    """Extract exon-level records from a GTF DataFrame."""
    feature_col = _col(gtf, "feature", "Feature")
    exons = gtf.loc[gtf[feature_col].astype(str).str.lower() == "exon"].copy()
    if "gene_id" in exons.columns and "gene_id_base" not in exons.columns:
        exons = add_gene_id_columns(exons)
    return exons.reset_index(drop=True)


def derive_terminal_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """Return terminal exons per transcript, strand-aware."""
    if exons.empty:
        return pd.DataFrame(columns=exons.columns)
    start_col = _col(exons, "start", "Start")
    strand_col = _col(exons, "strand", "Strand")
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"
    results = []
    for (_, _, strand), grp in exons.groupby([gene_col, tx_col, strand_col], observed=True):
        idx = grp[start_col].idxmax() if strand == "+" else grp[start_col].idxmin()
        results.append(grp.loc[idx])
    return pd.DataFrame(results).reset_index(drop=True) if results else pd.DataFrame(columns=exons.columns)


def derive_introns(exons: pd.DataFrame) -> pd.DataFrame:
    """Derive intron intervals from exon coordinates per transcript."""
    start_col = _col(exons, "start", "Start")
    end_col = _col(exons, "end", "End")
    chrom_col = _col(exons, "chrom", "Chromosome")
    strand_col = _col(exons, "strand", "Strand")
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"
    introns = []
    for (gene, tx, chrom, strand), grp in exons.groupby([gene_col, tx_col, chrom_col, strand_col], observed=True):
        grp = grp.sort_values(start_col)
        starts = grp[start_col].to_numpy()
        ends = grp[end_col].to_numpy()
        for i in range(len(starts) - 1):
            if starts[i + 1] > ends[i]:
                introns.append({gene_col: gene, tx_col: tx, chrom_col: chrom, strand_col: strand, start_col: int(ends[i]), end_col: int(starts[i + 1]), "Feature": "intron"})
    return pd.DataFrame(introns)


def derive_splice_sites(exons: pd.DataFrame) -> pd.DataFrame:
    """Extract donor/acceptor splice-site positions from exon coordinates."""
    if exons.empty:
        return pd.DataFrame(columns=["gene_id", "transcript_id", "chrom", "strand", "position", "splice_site_type"])
    start_col = _col(exons, "start", "Start")
    end_col = _col(exons, "end", "End")
    chrom_col = _col(exons, "chrom", "Chromosome")
    strand_col = _col(exons, "strand", "Strand")
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"
    sites = []
    for (gene, tx, chrom, strand), grp in exons.groupby([gene_col, tx_col, chrom_col, strand_col], observed=True):
        grp = grp.sort_values(start_col)
        starts = grp[start_col].to_numpy()
        ends = grp[end_col].to_numpy()
        for i in range(len(starts)):
            if i < len(starts) - 1:
                sites.append({gene_col: gene, tx_col: tx, chrom_col: chrom, strand_col: strand, "position": int(ends[i] if strand == "+" else starts[i]), "splice_site_type": "donor"})
            if i > 0:
                sites.append({gene_col: gene, tx_col: tx, chrom_col: chrom, strand_col: strand, "position": int(starts[i] if strand == "+" else ends[i]), "splice_site_type": "acceptor"})
    return pd.DataFrame(sites).drop_duplicates().reset_index(drop=True)


def annotate_intervals_with_gene_context(intervals: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    """Annotate intervals with overlapping gene metadata using a midpoint heuristic."""
    return annotate_pas_context(intervals, gtf)


def annotate_pas_context(
    pas_sites: pd.DataFrame,
    gtf: pd.DataFrame,
    terminal_exons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Annotate PAS sites with gene-region context.

    The heuristic prioritizes explicit UTR/CDS features from GTF, then pipeline
    terminal exons, then gene overlap as intronic/intergenic context.
    """
    pas = pas_sites.copy()
    if pas.empty:
        pas["pas_context"] = []
        return pas
    if "gene_id_base" not in pas.columns and "gene_id" in pas.columns:
        pas["gene_id_base"] = pas["gene_id"].map(gene_id_base)

    feature_col = _col(gtf, "feature", "Feature")
    chrom_col = _col(gtf, "chrom", "Chromosome")
    start_col = _col(gtf, "start", "Start")
    end_col = _col(gtf, "end", "End")
    strand_col = _col(gtf, "strand", "Strand")
    gene_col = "gene_id_base" if "gene_id_base" in gtf.columns else "gene_id"

    features = gtf[[chrom_col, start_col, end_col, strand_col, feature_col, gene_col]].copy()
    features[feature_col] = features[feature_col].astype(str).str.lower()
    genes = features[features[feature_col] == "gene"]
    utr3 = features[features[feature_col].isin(["three_prime_utr", "3utr"])]
    utr5 = features[features[feature_col].isin(["five_prime_utr", "5utr"])]
    cds = features[features[feature_col] == "cds"]

    terminal = None
    if terminal_exons is not None and not terminal_exons.empty:
        terminal = terminal_exons.copy()
        if "gene_id_base" not in terminal.columns and "gene_id" in terminal.columns:
            terminal["gene_id_base"] = terminal["gene_id"].map(gene_id_base)

    contexts = []
    for _, row in pas.iterrows():
        chrom = row.get("chrom") or row.get("Chromosome")
        strand = row.get("strand") or row.get("Strand")
        pos = int(row.get("start", row.get("Start")))
        gid = row.get("gene_id_base", gene_id_base(row.get("gene_id", "")))
        contexts.append(_classify_position(chrom, pos, strand, gid, genes, utr3, utr5, cds, terminal))
    pas["pas_context"] = contexts
    return pas


def _overlaps(df: pd.DataFrame, chrom: str, pos: int, strand: str, gene: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    chrom_col = "chrom" if "chrom" in df.columns else "Chromosome"
    start_col = "start" if "start" in df.columns else "Start"
    end_col = "end" if "end" in df.columns else "End"
    strand_col = "strand" if "strand" in df.columns else "Strand"
    mask = (df[chrom_col] == chrom) & (df[start_col] <= pos) & (df[end_col] >= pos) & (df[strand_col] == strand)
    if gene and "gene_id_base" in df.columns:
        mask &= df["gene_id_base"] == gene
    return df[mask]


def _classify_position(chrom, pos, strand, gid, genes, utr3, utr5, cds, terminal) -> str:
    if not _overlaps(utr3, chrom, pos, strand, gid).empty:
        return "3UTR"
    if not _overlaps(utr5, chrom, pos, strand, gid).empty:
        return "5UTR"
    if not _overlaps(cds, chrom, pos, strand, gid).empty:
        return "CDS"
    if terminal is not None and not _overlaps(terminal, chrom, pos, strand, gid).empty:
        return "terminal_exon"
    if not _overlaps(genes, chrom, pos, strand, gid).empty:
        return "intron"
    return "intergenic"


def distance_to_nearest_splice_site(
    pas_sites: pd.DataFrame,
    splice_sites: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Compute distance from each PAS site to the nearest splice site.

    Uses sorted per chromosome/strand splice positions, so it scales to full
    scPolASeq PAS catalogs without all-vs-all distance scans.
    """
    pas = pas_sites.copy()
    pas["nearest_splice_site_type"] = None
    pas["nearest_splice_site_distance"] = np.nan
    pas["is_splice_proximal"] = False
    if splice_sites.empty or pas.empty:
        return pas

    chrom_col_pas = "chrom" if "chrom" in pas.columns else "Chromosome"
    start_col_pas = "start" if "start" in pas.columns else "Start"
    strand_col_pas = "strand" if "strand" in pas.columns else "Strand"
    chrom_col_ss = "chrom" if "chrom" in splice_sites.columns else "Chromosome"
    strand_col_ss = "strand" if "strand" in splice_sites.columns else "Strand"

    grouped: dict[tuple[object, object], tuple[np.ndarray, np.ndarray]] = {}
    for key, grp in splice_sites.groupby([chrom_col_ss, strand_col_ss], observed=True):
        ordered = grp.sort_values("position")
        grouped[key] = (ordered["position"].to_numpy(dtype=int), ordered["splice_site_type"].astype(str).to_numpy())

    for key, idxs in pas.groupby([chrom_col_pas, strand_col_pas], observed=True).groups.items():
        if key not in grouped:
            continue
        ss_pos, ss_type = grouped[key]
        if len(ss_pos) == 0:
            continue
        positions = pas.loc[idxs, start_col_pas].to_numpy(dtype=int)
        insert = np.searchsorted(ss_pos, positions)
        best_dist = np.full(len(positions), np.iinfo(np.int32).max, dtype=int)
        best_type = np.array([None] * len(positions), dtype=object)

        right_mask = insert < len(ss_pos)
        right_dist = np.abs(ss_pos[np.clip(insert, 0, len(ss_pos) - 1)] - positions)
        best_dist[right_mask] = right_dist[right_mask]
        best_type[right_mask] = ss_type[np.clip(insert[right_mask], 0, len(ss_pos) - 1)]

        left_insert = insert - 1
        left_mask = left_insert >= 0
        left_dist = np.abs(ss_pos[np.clip(left_insert, 0, len(ss_pos) - 1)] - positions)
        take_left = left_mask & (left_dist < best_dist)
        best_dist[take_left] = left_dist[take_left]
        best_type[take_left] = ss_type[left_insert[take_left]]

        pas.loc[idxs, "nearest_splice_site_distance"] = best_dist.astype(float)
        pas.loc[idxs, "nearest_splice_site_type"] = best_type
        pas.loc[idxs, "is_splice_proximal"] = best_dist <= window
    return pas

def compute_relative_position(site_pos: int, region_start: int, region_end: int, strand: str) -> float:
    """Compute relative position [0, 1] of a site within a region, strand-aware."""
    length = region_end - region_start
    if length <= 0:
        return float("nan")
    if strand == "+":
        return (site_pos - region_start) / length
    return (region_end - site_pos) / length


def terminal_regions_for_genes(terminal_exons: pd.DataFrame, genes: list[str], gene_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return terminal exon regions matching gene symbols or gene ids."""
    regions = terminal_exons.copy()
    if regions.empty:
        return regions
    if "gene_id_base" not in regions.columns and "gene_id" in regions.columns:
        regions["gene_id_base"] = regions["gene_id"].map(gene_id_base)
    candidates = set(genes)
    if gene_table is not None and "gene_name" in gene_table.columns:
        gt = gene_table.copy()
        if "gene_id_base" not in gt.columns and "gene_id" in gt.columns:
            gt["gene_id_base"] = gt["gene_id"].map(gene_id_base)
        candidates.update(gt.loc[gt["gene_name"].isin(genes), "gene_id_base"].dropna().astype(str))
    return regions[regions["gene_id_base"].isin(candidates) | regions.get("gene_id", pd.Series(dtype=str)).isin(candidates)].copy()
