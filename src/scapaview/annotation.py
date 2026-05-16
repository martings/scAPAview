"""Genomic annotation utilities for scAPAview.

Coordinate convention
---------------------
All internal coordinates are **0-based half-open** (BED-style).  GTF files
use 1-based inclusive coordinates, so when reading a GTF we subtract 1 from
the *Start* column to convert to 0-based:

    gtf_start_0based = gtf_Start - 1   # (Start was 1-based inclusive)
    gtf_end_0based   = gtf_End          # (End was 1-based inclusive → same as 0-based exclusive)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PAS context categories (ordered from 5' to 3')
PAS_CONTEXT_CATEGORIES = [
    "5UTR",
    "CDS",
    "intron",
    "last_intron",
    "terminal_exon",
    "3UTR",
    "downstream_TES",
    "intergenic",
    "unknown",
]


# ---------------------------------------------------------------------------
# GTF standardisation
# ---------------------------------------------------------------------------


def standardize_gtf(gtf: pd.DataFrame) -> pd.DataFrame:
    """Convert GTF DataFrame to 0-based half-open coordinates.

    GTF Start (1-based inclusive) → subtract 1.
    GTF End   (1-based inclusive) → keep (becomes 0-based exclusive).
    """
    gtf = gtf.copy()
    if "Start" in gtf.columns:
        gtf["Start"] = gtf["Start"] - 1
    elif "start" in gtf.columns:
        gtf["start"] = gtf["start"] - 1
    return gtf


def build_gene_table(gtf: pd.DataFrame) -> pd.DataFrame:
    """Extract gene-level records from a GTF DataFrame."""
    feature_col = "Feature" if "Feature" in gtf.columns else "feature"
    mask = gtf[feature_col].str.lower() == "gene"
    return gtf.loc[mask].reset_index(drop=True)


def build_exon_table(gtf: pd.DataFrame) -> pd.DataFrame:
    """Extract exon-level records from a GTF DataFrame."""
    feature_col = "Feature" if "Feature" in gtf.columns else "feature"
    mask = gtf[feature_col].str.lower() == "exon"
    return gtf.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derived annotation tables
# ---------------------------------------------------------------------------


def derive_terminal_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """Return the terminal (last) exon for each transcript, strand-aware.

    For + strand the terminal exon has the largest *Start*.
    For − strand the terminal exon has the smallest *Start*.
    """
    start_col = "Start" if "Start" in exons.columns else "start"
    strand_col = "Strand" if "Strand" in exons.columns else "strand"
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"

    results = []
    for (gene, tx, strand), grp in exons.groupby([gene_col, tx_col, strand_col], observed=True):
        if strand == "+":
            idx = grp[start_col].idxmax()
        else:
            idx = grp[start_col].idxmin()
        results.append(grp.loc[idx])

    if not results:
        return pd.DataFrame(columns=exons.columns)
    return pd.DataFrame(results).reset_index(drop=True)


def derive_introns(exons: pd.DataFrame) -> pd.DataFrame:
    """Derive intron coordinates from exon coordinates per transcript.

    Intron [i] spans from end of exon[i] to start of exon[i+1]
    (in genomic order regardless of strand).
    """
    start_col = "Start" if "Start" in exons.columns else "start"
    end_col = "End" if "End" in exons.columns else "end"
    chrom_col = "Chromosome" if "Chromosome" in exons.columns else "chrom"
    strand_col = "Strand" if "Strand" in exons.columns else "strand"
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"

    introns = []
    for (gene, tx, chrom, strand), grp in exons.groupby(
        [gene_col, tx_col, chrom_col, strand_col], observed=True
    ):
        grp_sorted = grp.sort_values(start_col)
        starts = grp_sorted[start_col].values
        ends = grp_sorted[end_col].values
        for i in range(len(starts) - 1):
            introns.append(
                {
                    gene_col: gene,
                    tx_col: tx,
                    chrom_col: chrom,
                    strand_col: strand,
                    start_col: ends[i],
                    end_col: starts[i + 1],
                    "Feature": "intron",
                }
            )

    if not introns:
        return pd.DataFrame(
            columns=[gene_col, tx_col, chrom_col, strand_col, start_col, end_col, "Feature"]
        )
    return pd.DataFrame(introns).reset_index(drop=True)


def derive_splice_sites(exons: pd.DataFrame) -> pd.DataFrame:
    """Extract 5' and 3' splice site positions from exon coordinates.

    Returns a DataFrame with columns:
    gene_id, transcript_id, chrom, strand, position, splice_site_type
    """
    start_col = "Start" if "Start" in exons.columns else "start"
    end_col = "End" if "End" in exons.columns else "end"
    chrom_col = "Chromosome" if "Chromosome" in exons.columns else "chrom"
    strand_col = "Strand" if "Strand" in exons.columns else "strand"
    gene_col = "gene_id" if "gene_id" in exons.columns else "GeneID"
    tx_col = "transcript_id" if "transcript_id" in exons.columns else "TranscriptID"

    sites = []
    for (gene, tx, chrom, strand), grp in exons.groupby(
        [gene_col, tx_col, chrom_col, strand_col], observed=True
    ):
        grp_sorted = grp.sort_values(start_col)
        starts = grp_sorted[start_col].values
        ends = grp_sorted[end_col].values
        n = len(starts)
        for i in range(n):
            # Donor (5' splice site) and acceptor (3' splice site) depend on strand
            if i < n - 1:  # not last exon → has a downstream intron
                if strand == "+":
                    sites.append(
                        {gene_col: gene, tx_col: tx, chrom_col: chrom,
                         strand_col: strand, "position": int(ends[i]),
                         "splice_site_type": "donor"}
                    )
                else:
                    sites.append(
                        {gene_col: gene, tx_col: tx, chrom_col: chrom,
                         strand_col: strand, "position": int(starts[i]),
                         "splice_site_type": "donor"}
                    )
            if i > 0:  # not first exon → has an upstream intron
                if strand == "+":
                    sites.append(
                        {gene_col: gene, tx_col: tx, chrom_col: chrom,
                         strand_col: strand, "position": int(starts[i]),
                         "splice_site_type": "acceptor"}
                    )
                else:
                    sites.append(
                        {gene_col: gene, tx_col: tx, chrom_col: chrom,
                         strand_col: strand, "position": int(ends[i]),
                         "splice_site_type": "acceptor"}
                    )

    if not sites:
        return pd.DataFrame(
            columns=[gene_col, tx_col, chrom_col, strand_col, "position", "splice_site_type"]
        )
    return pd.DataFrame(sites).drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# PAS annotation
# ---------------------------------------------------------------------------


def annotate_pas_context(
    pas_sites: pd.DataFrame, gtf: pd.DataFrame
) -> pd.DataFrame:
    """Annotate each PAS site with a genomic context region.

    Adds a ``pas_context`` column with one of the PAS_CONTEXT_CATEGORIES.
    This is a simplified heuristic annotation based on overlap with GTF features.
    """
    pas = pas_sites.copy()
    pas["pas_context"] = "intergenic"

    feature_col = "Feature" if "Feature" in gtf.columns else "feature"
    chrom_col_gtf = "Chromosome" if "Chromosome" in gtf.columns else "chrom"
    start_col_gtf = "Start" if "Start" in gtf.columns else "start"
    end_col_gtf = "End" if "End" in gtf.columns else "end"
    strand_col_gtf = "Strand" if "Strand" in gtf.columns else "strand"

    chrom_col_pas = "chrom" if "chrom" in pas.columns else "Chromosome"
    start_col_pas = "start" if "start" in pas.columns else "Start"
    end_col_pas = "end" if "end" in pas.columns else "End"
    strand_col_pas = "strand" if "strand" in pas.columns else "Strand"

    for idx, row in pas.iterrows():
        chrom = row[chrom_col_pas]
        pos = int(row[start_col_pas])
        strand = row[strand_col_pas]
        context = _classify_context(
            chrom, pos, strand, gtf,
            feature_col, chrom_col_gtf, start_col_gtf, end_col_gtf, strand_col_gtf
        )
        pas.at[idx, "pas_context"] = context

    return pas


def _classify_context(
    chrom: str,
    pos: int,
    strand: str,
    gtf: pd.DataFrame,
    feature_col: str,
    chrom_col: str,
    start_col: str,
    end_col: str,
    strand_col: str,
) -> str:
    """Classify a single genomic position into a PAS context."""
    region = gtf[
        (gtf[chrom_col] == chrom)
        & (gtf[start_col] <= pos)
        & (gtf[end_col] > pos)
        & (gtf[strand_col] == strand)
    ]
    if region.empty:
        return "intergenic"

    features = set(region[feature_col].str.lower().unique())
    if "three_prime_utr" in features or "3utr" in features:
        return "3UTR"
    if "five_prime_utr" in features or "5utr" in features:
        return "5UTR"
    if "cds" in features:
        return "CDS"
    if "exon" in features:
        return "terminal_exon"
    if "gene" in features:
        return "intron"
    return "unknown"


def distance_to_nearest_splice_site(
    pas_sites: pd.DataFrame,
    splice_sites: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """Compute distance from each PAS site to the nearest splice site.

    Adds columns:
    - nearest_splice_site_type: 'donor' | 'acceptor' | None
    - nearest_splice_site_distance: int
    - is_splice_proximal: bool (True if distance <= window)
    """
    pas = pas_sites.copy()
    pas["nearest_splice_site_type"] = None
    pas["nearest_splice_site_distance"] = np.nan
    pas["is_splice_proximal"] = False

    if splice_sites.empty:
        return pas

    chrom_col_pas = "chrom" if "chrom" in pas.columns else "Chromosome"
    start_col_pas = "start" if "start" in pas.columns else "Start"
    chrom_col_ss = "chrom" if "chrom" in splice_sites.columns else "Chromosome"

    for idx, row in pas.iterrows():
        chrom = row[chrom_col_pas]
        pos = int(row[start_col_pas])
        same_chrom = splice_sites[splice_sites[chrom_col_ss] == chrom]
        if same_chrom.empty:
            continue
        dists = np.abs(same_chrom["position"].values - pos)
        min_i = int(np.argmin(dists))
        min_dist = int(dists[min_i])
        pas.at[idx, "nearest_splice_site_distance"] = min_dist
        pas.at[idx, "nearest_splice_site_type"] = same_chrom.iloc[min_i]["splice_site_type"]
        pas.at[idx, "is_splice_proximal"] = min_dist <= window

    return pas


# ---------------------------------------------------------------------------
# Relative position
# ---------------------------------------------------------------------------


def compute_relative_position(
    site_pos: int,
    region_start: int,
    region_end: int,
    strand: str,
) -> float:
    """Compute relative position [0, 1] of a site within a region, strand-aware.

    For + strand: 0 = region_start (5' end), 1 = region_end (3' end).
    For − strand: 0 = region_end   (5' end), 1 = region_start (3' end).

    Returns float in [0, 1] or np.nan if region has zero length.
    """
    length = region_end - region_start
    if length <= 0:
        return float("nan")
    if strand == "+":
        return (site_pos - region_start) / length
    else:
        return (region_end - site_pos) / length
