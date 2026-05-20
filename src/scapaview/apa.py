"""APA event classification and PAS site merging utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .io import add_gene_id_columns, gene_id_base, parse_site_id

logger = logging.getLogger(__name__)


def classify_apa_events(
    events: pd.DataFrame,
    fdr_col: str = "adj_p_value",
    delta_col: str = "delta_pdui",
    fdr_cutoff: float = 0.05,
    delta_cutoff: float = 0.15,
) -> pd.DataFrame:
    """Classify APA events by FDR and absolute delta-PDUI thresholds."""
    df = events.copy()
    if df.empty:
        return df
    if fdr_col not in df.columns:
        raise ValueError(f"FDR column '{fdr_col}' not found in events DataFrame.")
    if delta_col not in df.columns:
        raise ValueError(f"Delta column '{delta_col}' not found in events DataFrame.")

    df = add_gene_id_columns(df)
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df["abs_delta_pdui"] = df[delta_col].abs()
    df["is_fdr_significant"] = df[fdr_col] < fdr_cutoff
    df["is_delta_candidate"] = df["abs_delta_pdui"] >= delta_cutoff
    df["is_fdr_and_delta"] = df["is_fdr_significant"] & df["is_delta_candidate"]

    df["direction"] = np.select(
        [df[delta_col] > 0, df[delta_col] < 0],
        ["lengthening", "shortening"],
        default="none",
    )
    df["priority_class"] = np.select(
        [
            df["is_fdr_and_delta"],
            df["is_fdr_significant"],
            df["is_delta_candidate"],
        ],
        ["fdr_and_delta", "fdr_only", "delta_only"],
        default="not_significant",
    )
    return df


def _ensure_pas_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if "site_id" not in out.columns:
        raise ValueError(f"PAS table for {source} lacks site_id")
    parsed = pd.DataFrame([parse_site_id(sid) for sid in out["site_id"]])
    for col in ("gene_id", "gene_id_base", "chrom", "start", "end", "strand"):
        if col not in out.columns and col in parsed.columns:
            out[col] = parsed[col]
    if "gene_id_base" not in out.columns and "gene_id" in out.columns:
        out["gene_id_base"] = out["gene_id"].map(gene_id_base)
    if "source" not in out.columns:
        out["source"] = source
    required = ["site_id", "gene_id", "chrom", "start", "end", "strand", "gene_id_base"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"PAS table for {source} missing columns after parsing: {missing}")
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["start", "end"]).copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["source"] = out["source"].fillna(source).astype(str)
    return out


def _collapse_values(values: pd.Series) -> object:
    vals = [str(v) for v in values.dropna().astype(str) if str(v) and str(v) != "nan"]
    unique = sorted(set(vals))
    if not unique:
        return pd.NA
    if len(unique) == 1:
        return unique[0]
    return ",".join(unique)


def _merge_group(group: pd.DataFrame, window: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: list[int] = []
    current_center: float | None = None

    for idx, row in group.sort_values("start").iterrows():
        center = (int(row["start"]) + int(row["end"])) / 2
        if current and current_center is not None and abs(center - current_center) > window:
            rows.append(_collapse_cluster(group.loc[current]))
            current = []
            current_center = None
        current.append(idx)
        if current_center is None:
            current_center = center
        else:
            current_center = float(np.mean([(group.loc[i, "start"] + group.loc[i, "end"]) / 2 for i in current]))
    if current:
        rows.append(_collapse_cluster(group.loc[current]))
    return rows


def _collapse_cluster(cluster: pd.DataFrame) -> dict[str, object]:
    first = cluster.iloc[0]
    sources = sorted(set(",".join(cluster["source"].astype(str)).split(",")))
    row: dict[str, object] = {
        "site_id": _collapse_values(cluster["site_id"]),
        "gene_id": _collapse_values(cluster["gene_id"]),
        "gene_id_base": first["gene_id_base"],
        "chrom": first["chrom"],
        "start": int(cluster["start"].min()),
        "end": int(cluster["end"].max()),
        "strand": first["strand"],
        "source": ",".join(s for s in sources if s),
        "n_sources": len([s for s in sources if s]),
    }
    for col in cluster.columns:
        if col in row or col in {"gene_id_base", "chrom", "start", "end", "strand"}:
            continue
        if pd.api.types.is_numeric_dtype(cluster[col]):
            if col.endswith("count") or col.startswith("sierra_"):
                row[col] = cluster[col].sum(skipna=True)
            else:
                row[col] = cluster[col].dropna().iloc[0] if cluster[col].notna().any() else pd.NA
        elif col in {"site_source", "reference_source", "site_class", "priming_flag", "sierra_groups", "scapture_sample", "original_gene_name"}:
            row[col] = _collapse_values(cluster[col])
    return row


def merge_sierra_scapture_sites(
    sierra_sites: pd.DataFrame,
    scapture_sites: pd.DataFrame,
    window: int = 25,
) -> pd.DataFrame:
    """Merge Sierra and scapture PAS sites by gene/chrom/strand and coordinate proximity."""
    return build_unified_pas_table(sierra_sites=sierra_sites, scapture_sites=scapture_sites, window=window)


def rank_pas_within_gene(pas_sites: pd.DataFrame) -> pd.DataFrame:
    """Rank PAS sites within each gene by transcript-strand direction."""
    df = pas_sites.copy()
    if df.empty:
        df["pas_rank_in_gene"] = []
        return df
    gene_col = "gene_id_base" if "gene_id_base" in df.columns else "gene_id"
    df["pas_rank_in_gene"] = 0
    for (_, strand), grp in df.groupby([gene_col, "strand"], observed=True):
        if strand == "+":
            order = grp["start"].rank(method="first").astype(int)
        else:
            order = grp["start"].rank(method="first", ascending=False).astype(int)
        df.loc[grp.index, "pas_rank_in_gene"] = order.values
    return df


def summarize_pas_support(pas_sites: pd.DataFrame) -> pd.DataFrame:
    """Add or refresh source-count support columns."""
    df = pas_sites.copy()
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["n_sources"] = df["source"].apply(lambda s: len([x for x in str(s).split(",") if x]))
    return df


def build_unified_pas_table(
    scpolaseq_sites: pd.DataFrame | None = None,
    sierra_sites: pd.DataFrame | None = None,
    scapture_sites: pd.DataFrame | None = None,
    window: int = 25,
) -> pd.DataFrame:
    """Build a coordinate-merged PAS table from scPolASeq, Sierra, and scapture sources."""
    frames: list[pd.DataFrame] = []
    for source, frame in (
        ("scpolaseq", scpolaseq_sites),
        ("sierra", sierra_sites),
        ("scapture", scapture_sites),
    ):
        if frame is not None and not frame.empty:
            frames.append(_ensure_pas_columns(frame, source))
    if not frames:
        return pd.DataFrame()

    all_sites = pd.concat(frames, ignore_index=True, sort=False)
    if len(frames) == 1:
        unified = summarize_pas_support(all_sites)
        unified = rank_pas_within_gene(unified)
        logger.info("Built unified PAS table with %d unmerged single-source sites", len(unified))
        return unified

    merged_rows: list[dict[str, object]] = []
    group_cols = ["gene_id_base", "chrom", "strand"]
    for _, group in all_sites.groupby(group_cols, observed=True, dropna=False):
        merged_rows.extend(_merge_group(group, window=window))
    unified = pd.DataFrame(merged_rows)
    unified = summarize_pas_support(unified)
    unified = rank_pas_within_gene(unified)
    logger.info("Built unified PAS table with %d merged sites from %d source rows", len(unified), len(all_sites))
    return unified
