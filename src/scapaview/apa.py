"""APA event classification and PAS site merging utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# APA event classification
# ---------------------------------------------------------------------------


def classify_apa_events(
    events: pd.DataFrame,
    fdr_col: str = "adj_p_value",
    delta_col: str = "delta_pdui",
    fdr_cutoff: float = 0.05,
    delta_cutoff: float = 0.15,
) -> pd.DataFrame:
    """Classify APA events and add classification columns.

    Added columns
    -------------
    is_fdr_significant : bool
    is_delta_candidate : bool
    is_fdr_and_delta   : bool
    abs_delta_pdui     : float
    direction          : 'lengthening' | 'shortening' | 'none'
    priority_class     : 'fdr_and_delta' | 'fdr_only' | 'delta_only' | 'not_significant'
    """
    df = events.copy()
    if fdr_col not in df.columns:
        raise ValueError(f"FDR column '{fdr_col}' not found in events DataFrame.")
    if delta_col not in df.columns:
        raise ValueError(f"Delta column '{delta_col}' not found in events DataFrame.")

    df["abs_delta_pdui"] = df[delta_col].abs()
    df["is_fdr_significant"] = df[fdr_col] < fdr_cutoff
    df["is_delta_candidate"] = df["abs_delta_pdui"] >= delta_cutoff
    df["is_fdr_and_delta"] = df["is_fdr_significant"] & df["is_delta_candidate"]

    def _direction(val: float) -> str:
        if val > 0:
            return "lengthening"
        if val < 0:
            return "shortening"
        return "none"

    df["direction"] = df[delta_col].apply(_direction)

    def _priority(row: pd.Series) -> str:
        if row["is_fdr_and_delta"]:
            return "fdr_and_delta"
        if row["is_fdr_significant"]:
            return "fdr_only"
        if row["is_delta_candidate"]:
            return "delta_only"
        return "not_significant"

    df["priority_class"] = df.apply(_priority, axis=1)
    logger.info(
        "Classified %d APA events: %d fdr_and_delta, %d fdr_only, %d delta_only, %d not_significant",
        len(df),
        (df["priority_class"] == "fdr_and_delta").sum(),
        (df["priority_class"] == "fdr_only").sum(),
        (df["priority_class"] == "delta_only").sum(),
        (df["priority_class"] == "not_significant").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# PAS site merging
# ---------------------------------------------------------------------------


def merge_sierra_scapture_sites(
    sierra_sites: pd.DataFrame,
    scapture_sites: pd.DataFrame,
    window: int = 25,
) -> pd.DataFrame:
    """Merge PAS sites from Sierra Quant and scapture using coordinate proximity.

    Sites within ``window`` bp are considered the same site.  The merged site
    adopts the Sierra site_id; scapture-only sites are appended.
    """
    start_col = "start" if "start" in sierra_sites.columns else "Start"
    chrom_col = "chrom" if "chrom" in sierra_sites.columns else "Chromosome"
    gene_col = "gene_id" if "gene_id" in sierra_sites.columns else "GeneID"
    strand_col = "strand" if "strand" in sierra_sites.columns else "Strand"

    merged = sierra_sites.copy()
    merged["source"] = "sierra"

    unmatched = []
    for _, sc_row in scapture_sites.iterrows():
        chrom = sc_row.get(chrom_col)
        pos = sc_row.get(start_col)
        gene = sc_row.get(gene_col)
        same = merged[
            (merged[chrom_col] == chrom)
            & (merged[gene_col] == gene)
            & ((merged[start_col] - pos).abs() <= window)
        ]
        if same.empty:
            row_copy = sc_row.copy()
            row_copy["source"] = "scapture"
            unmatched.append(row_copy)
        else:
            idx = same.index[0]
            existing_src = merged.at[idx, "source"]
            if "scapture" not in str(existing_src):
                merged.at[idx, "source"] = existing_src + ",scapture"

    if unmatched:
        unmatched_df = pd.DataFrame(unmatched)
        merged = pd.concat([merged, unmatched_df], ignore_index=True)

    logger.info("Merged %d sierra + %d scapture → %d unified sites",
                len(sierra_sites), len(scapture_sites), len(merged))
    return merged


# ---------------------------------------------------------------------------
# PAS site ranking and support
# ---------------------------------------------------------------------------


def rank_pas_within_gene(pas_sites: pd.DataFrame) -> pd.DataFrame:
    """Rank PAS sites within each gene by position, strand-aware.

    For + strand: rank 1 = most 5' site (lowest start).
    For − strand: rank 1 = most 5' site (highest end or start for − genes,
    i.e. largest genomic coordinate).

    Adds column ``pas_rank_in_gene``.
    """
    df = pas_sites.copy()
    start_col = "start" if "start" in df.columns else "Start"
    gene_col = "gene_id" if "gene_id" in df.columns else "GeneID"
    strand_col = "strand" if "strand" in df.columns else "Strand"

    df["pas_rank_in_gene"] = 0
    for (gene, strand), grp in df.groupby([gene_col, strand_col]):
        if strand == "+":
            order = grp[start_col].rank(method="first").astype(int)
        else:
            order = grp[start_col].rank(method="first", ascending=False).astype(int)
        df.loc[grp.index, "pas_rank_in_gene"] = order.values

    return df


def summarize_pas_support(pas_sites: pd.DataFrame) -> pd.DataFrame:
    """Summarize how many sources support each PAS site.

    Expects a ``source`` column with comma-separated source names.
    Adds ``n_sources`` column.
    """
    df = pas_sites.copy()
    if "source" not in df.columns:
        df["n_sources"] = 1
        return df
    df["n_sources"] = df["source"].apply(
        lambda s: len(str(s).split(",")) if pd.notna(s) else 0
    )
    return df


def build_unified_pas_table(
    scpolaseq_sites: pd.DataFrame | None = None,
    sierra_sites: pd.DataFrame | None = None,
    scapture_sites: pd.DataFrame | None = None,
    window: int = 25,
) -> pd.DataFrame:
    """Build a unified PAS site table from multiple sources.

    Merges whatever combination of sources is provided.
    """
    frames: list[pd.DataFrame] = []

    if scpolaseq_sites is not None and not scpolaseq_sites.empty:
        sp = scpolaseq_sites.copy()
        if "source" not in sp.columns:
            sp["source"] = "scpolaseq"
        frames.append(sp)

    if sierra_sites is not None and scapture_sites is not None:
        merged = merge_sierra_scapture_sites(sierra_sites, scapture_sites, window=window)
        frames.append(merged)
    elif sierra_sites is not None:
        s = sierra_sites.copy()
        if "source" not in s.columns:
            s["source"] = "sierra"
        frames.append(s)
    elif scapture_sites is not None:
        sc = scapture_sites.copy()
        if "source" not in sc.columns:
            sc["source"] = "scapture"
        frames.append(sc)

    if not frames:
        return pd.DataFrame()

    unified = pd.concat(frames, ignore_index=True)
    unified = summarize_pas_support(unified)
    unified = rank_pas_within_gene(unified)
    logger.info("Built unified PAS table with %d sites", len(unified))
    return unified
