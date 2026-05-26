"""Transcript landmark QC and architecture diagrams.

This module is intentionally separate from metagene plotting. It helps audit
whether transcript landmarks are positioned correctly before aggregating them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .io import gene_id_base
from .plotting import save_figure

logger = logging.getLogger(__name__)

LANDMARK_COLUMNS = [
    "gene_id", "gene_name", "transcript_id", "chrom", "position", "strand",
    "landmark_type", "feature_source",
]

WINDOW_LANDMARKS = {"TES", "PAS", "donor_splice_site", "acceptor_splice_site"}
PAS_COLORS = {
    "known": "0.55",
    "known_pas": "0.55",
    "pas_reference": "0.55",
    "sierra": "royalblue",
    "scapture": "darkorange",
    "scpolaseq": "seagreen",
}
APA_EDGE = {
    "fdr_only": "goldenrod",
    "delta_only": "mediumpurple",
    "fdr_and_delta": "crimson",
}
SUMMARY_LANDMARK_ORDER = [
    "TSS",
    "donor_splice_site",
    "acceptor_splice_site",
    "translation_start_site",
    "translation_end_site",
    "TES",
    "PAS",
]


def _feature_col(gtf: pd.DataFrame) -> str:
    return "Feature" if "Feature" in gtf.columns else "feature"


def _start_col(df: pd.DataFrame) -> str:
    return "Start" if "Start" in df.columns else "start"


def _end_col(df: pd.DataFrame) -> str:
    return "End" if "End" in df.columns else "end"


def _chrom_col(df: pd.DataFrame) -> str:
    return "Chromosome" if "Chromosome" in df.columns else "chrom"


def _strand_col(df: pd.DataFrame) -> str:
    return "Strand" if "Strand" in df.columns else "strand"


def _gene_matches(df: pd.DataFrame, gene_name: str) -> pd.Series:
    masks = []
    if "gene_name" in df.columns:
        masks.append(df["gene_name"].astype(str) == gene_name)
    if "gene_id" in df.columns:
        masks.append(df["gene_id"].astype(str) == gene_name)
        masks.append(df["gene_id"].map(gene_id_base) == gene_id_base(gene_name))
    if "gene_id_base" in df.columns:
        masks.append(df["gene_id_base"].astype(str) == gene_id_base(gene_name))
    if not masks:
        return pd.Series(False, index=df.index)
    mask = masks[0].copy()
    for extra in masks[1:]:
        mask |= extra
    return mask


def _select_gene(gtf: pd.DataFrame, gene_name: str) -> pd.Series:
    feature_col = _feature_col(gtf)
    genes = gtf[gtf[feature_col].astype(str).str.lower() == "gene"].copy()
    hits = genes[_gene_matches(genes, gene_name)]
    if hits.empty:
        raise ValueError(f"Gene not found in GTF: {gene_name}")
    return hits.iloc[0]


def _select_transcript(gtf: pd.DataFrame, gene_row: pd.Series, transcript_id: str | None = None) -> tuple[str | None, bool]:
    feature_col = _feature_col(gtf)
    if "transcript_id" not in gtf.columns:
        return None, False
    gid = gene_row.get("gene_id")
    gid_base = gene_row.get("gene_id_base", gene_id_base(gid))
    gene_gtf = gtf[gtf.get("gene_id", pd.Series(index=gtf.index, dtype=str)).map(gene_id_base) == gid_base]
    tx_rows = gene_gtf[gene_gtf["transcript_id"].notna()]
    if transcript_id and transcript_id != "auto":
        if transcript_id not in set(tx_rows["transcript_id"].astype(str)):
            raise ValueError(f"Transcript '{transcript_id}' not found for gene {gene_row.get('gene_name', gid)}")
        return transcript_id, False
    tx_ids = sorted(set(tx_rows["transcript_id"].astype(str)))
    if not tx_ids:
        return None, False
    exon_rows = gene_gtf[gene_gtf[feature_col].astype(str).str.lower() == "exon"].copy()
    if exon_rows.empty:
        return tx_ids[0], len(tx_ids) > 1
    start_col = _start_col(exon_rows)
    end_col = _end_col(exon_rows)
    lengths = (
        exon_rows.assign(_len=exon_rows[end_col].astype(int) - exon_rows[start_col].astype(int))
        .groupby("transcript_id", observed=True)["_len"]
        .sum()
        .sort_values(ascending=False)
    )
    return str(lengths.index[0]), len(tx_ids) > 1


def _transcript_rows(gtf: pd.DataFrame, gene_row: pd.Series, transcript_id: str | None) -> pd.DataFrame:
    gid = gene_row.get("gene_id")
    gid_base = gene_row.get("gene_id_base", gene_id_base(gid))
    rows = gtf[gtf.get("gene_id", pd.Series(index=gtf.index, dtype=str)).map(gene_id_base) == gid_base].copy()
    if transcript_id and "transcript_id" in rows.columns:
        tx_specific = rows[(rows["transcript_id"].astype(str) == transcript_id) | rows["transcript_id"].isna()]
        if not tx_specific.empty:
            rows = tx_specific
    return rows


def _exons_for_transcript(gtf: pd.DataFrame, gene_row: pd.Series, transcript_id: str | None) -> pd.DataFrame:
    feature_col = _feature_col(gtf)
    rows = _transcript_rows(gtf, gene_row, transcript_id)
    exons = rows[rows[feature_col].astype(str).str.lower() == "exon"].copy()
    return exons.sort_values(_start_col(exons)) if not exons.empty else exons


def _terminal_exon(exons: pd.DataFrame, strand: str) -> pd.Series | None:
    if exons.empty:
        return None
    start_col = _start_col(exons)
    idx = exons[start_col].idxmax() if strand == "+" else exons[start_col].idxmin()
    return exons.loc[idx]


def _splice_landmarks(exons: pd.DataFrame) -> pd.DataFrame:
    if exons.empty:
        return pd.DataFrame(columns=["position", "landmark_type"])
    start_col = _start_col(exons)
    end_col = _end_col(exons)
    strand_col = _strand_col(exons)
    strand = str(exons.iloc[0][strand_col])
    ordered = exons.sort_values(start_col).reset_index(drop=True)
    rows = []
    if strand == "+":
        for i in range(len(ordered) - 1):
            rows.append({"position": int(ordered.loc[i, end_col]), "landmark_type": "donor_splice_site"})
            rows.append({"position": int(ordered.loc[i + 1, start_col]), "landmark_type": "acceptor_splice_site"})
    else:
        for i in range(1, len(ordered)):
            rows.append({"position": int(ordered.loc[i, start_col]), "landmark_type": "donor_splice_site"})
            rows.append({"position": int(ordered.loc[i - 1, end_col]), "landmark_type": "acceptor_splice_site"})
    return pd.DataFrame(rows)


def build_transcript_landmarks(
    gene_name: str,
    gtf: pd.DataFrame,
    transcript_id: str | None = None,
) -> pd.DataFrame:
    """Build transcript landmarks from GTF for one gene/transcript."""
    gene = _select_gene(gtf, gene_name)
    tx_id, auto_ambiguous = _select_transcript(gtf, gene, transcript_id)
    rows = _transcript_rows(gtf, gene, tx_id)
    exons = _exons_for_transcript(gtf, gene, tx_id)
    start_col = _start_col(gtf)
    end_col = _end_col(gtf)
    chrom_col = _chrom_col(gtf)
    strand_col = _strand_col(gtf)
    feature_col = _feature_col(gtf)
    chrom = gene[chrom_col]
    strand = gene[strand_col]
    gene_id = gene.get("gene_id", gene_name)
    symbol = gene.get("gene_name", gene_name)
    g_start = int(gene[start_col])
    g_end = int(gene[end_col])
    tss = g_start if strand == "+" else g_end
    tes = g_end if strand == "+" else g_start

    out = [
        {"landmark_type": "TSS", "position": tss, "feature_source": "gtf_gene"},
        {"landmark_type": "TES", "position": tes, "feature_source": "gtf_gene"},
    ]
    terminal = _terminal_exon(exons, strand)
    if terminal is not None:
        out.extend([
            {"landmark_type": "terminal_exon_start", "position": int(terminal[start_col]), "feature_source": "gtf_exon"},
            {"landmark_type": "terminal_exon_end", "position": int(terminal[end_col]), "feature_source": "gtf_exon"},
        ])

    utr3 = rows[rows[feature_col].astype(str).str.lower().isin(["three_prime_utr", "3utr"])]
    if not utr3.empty:
        out.extend([
            {"landmark_type": "3UTR_start", "position": int(utr3[start_col].min()), "feature_source": "gtf_utr"},
            {"landmark_type": "3UTR_end", "position": int(utr3[end_col].max()), "feature_source": "gtf_utr"},
        ])

    cds = rows[rows[feature_col].astype(str).str.lower() == "cds"]
    if not cds.empty:
        if strand == "+":
            out.append({"landmark_type": "translation_start_site", "position": int(cds[start_col].min()), "feature_source": "gtf_cds"})
            out.append({"landmark_type": "translation_end_site", "position": int(cds[end_col].max()), "feature_source": "gtf_cds"})
        else:
            out.append({"landmark_type": "translation_start_site", "position": int(cds[end_col].max()), "feature_source": "gtf_cds"})
            out.append({"landmark_type": "translation_end_site", "position": int(cds[start_col].min()), "feature_source": "gtf_cds"})

    splice = _splice_landmarks(exons)
    for _, row in splice.iterrows():
        out.append({"landmark_type": row["landmark_type"], "position": int(row["position"]), "feature_source": "derived_exon"})

    landmarks = pd.DataFrame(out)
    landmarks["gene_id"] = gene_id
    landmarks["gene_name"] = symbol
    landmarks["transcript_id"] = tx_id or transcript_id or "auto"
    landmarks["chrom"] = chrom
    landmarks["strand"] = strand
    landmarks["auto_transcript_ambiguous"] = auto_ambiguous
    return landmarks[LANDMARK_COLUMNS + ["auto_transcript_ambiguous"]]


def _filter_landmarks(landmarks: pd.DataFrame, gene_name: str | None, transcript_id: str | None) -> pd.DataFrame:
    out = landmarks.copy()
    if gene_name:
        masks = []
        if "gene_name" in out.columns:
            masks.append(out["gene_name"].astype(str) == gene_name)
        if "gene_id" in out.columns:
            masks.append(out["gene_id"].astype(str) == gene_name)
            masks.append(out["gene_id"].map(gene_id_base) == gene_id_base(gene_name))
        if masks:
            mask = masks[0]
            for extra in masks[1:]:
                mask |= extra
            out = out[mask]
    if transcript_id and transcript_id != "auto" and "transcript_id" in out.columns:
        out = out[out["transcript_id"].astype(str) == transcript_id]
    return out


def _filter_pas(pas_sites: pd.DataFrame | None, gene_row: pd.Series, transcript_landmarks: pd.DataFrame) -> pd.DataFrame:
    if pas_sites is None or pas_sites.empty:
        return pd.DataFrame()
    pas = pas_sites.copy()
    gene_id = gene_row.get("gene_id")
    gid_base = gene_row.get("gene_id_base", gene_id_base(gene_id))
    masks = []
    if "gene_id_base" in pas.columns:
        masks.append(pas["gene_id_base"].astype(str) == str(gid_base))
    if "gene_id" in pas.columns:
        masks.append(pas["gene_id"].map(gene_id_base) == str(gid_base))
    if "gene_name" in pas.columns:
        masks.append(pas["gene_name"].astype(str) == str(gene_row.get("gene_name", "")))
    if masks:
        mask = masks[0]
        for extra in masks[1:]:
            mask |= extra
        pas = pas[mask]
    chrom = transcript_landmarks["chrom"].iloc[0] if not transcript_landmarks.empty else gene_row[_chrom_col(pd.DataFrame([gene_row]))]
    if "chrom" in pas.columns:
        pas = pas[pas["chrom"] == chrom]
    return pas


def _landmark_positions(landmarks: pd.DataFrame, landmark_type: str) -> list[int]:
    if landmarks.empty or "landmark_type" not in landmarks.columns:
        return []
    vals = landmarks.loc[landmarks["landmark_type"] == landmark_type, "position"]
    return [int(v) for v in vals.dropna()]


def _transcript_rank(pos: int, gene_start: int, gene_end: int, strand: str) -> int:
    return pos if strand == "+" else gene_end - (pos - gene_start)


def qc_landmark_consistency(
    gtf: pd.DataFrame,
    landmarks: pd.DataFrame,
    pas_sites: pd.DataFrame | None = None,
    gene_name: str | None = None,
    transcript_id: str | None = None,
) -> pd.DataFrame:
    """Return QC records for transcript landmark consistency."""
    lm = _filter_landmarks(landmarks, gene_name, transcript_id)
    rows: list[dict[str, object]] = []

    def add(ltype, position, status, issue, severity, fix):
        tx = lm["transcript_id"].iloc[0] if not lm.empty and "transcript_id" in lm.columns else transcript_id
        gname = lm["gene_name"].iloc[0] if not lm.empty and "gene_name" in lm.columns else gene_name
        rows.append({
            "gene_name": gname,
            "transcript_id": tx,
            "landmark_type": ltype,
            "position": position,
            "status": status,
            "issue": issue,
            "severity": severity,
            "suggested_fix": fix,
        })

    if lm.empty:
        add("all", pd.NA, "fail", "No landmarks found for requested gene/transcript", "error", "Build or pass landmarks for this gene/transcript")
        return pd.DataFrame(rows)

    gene = _select_gene(gtf, str(gene_name or lm["gene_name"].iloc[0]))
    tx_id, auto_ambiguous = _select_transcript(gtf, gene, transcript_id)
    if transcript_id in (None, "auto") and auto_ambiguous:
        add("transcript_id", pd.NA, "warn", "transcript_id is ambiguous and was selected automatically", "warning", "Pass an explicit transcript_id")
    exons = _exons_for_transcript(gtf, gene, tx_id if transcript_id in (None, "auto") else transcript_id)
    start_col = _start_col(gtf)
    end_col = _end_col(gtf)
    chrom_col = _chrom_col(gtf)
    strand_col = _strand_col(gtf)
    g_start = int(gene[start_col])
    g_end = int(gene[end_col])
    chrom = gene[chrom_col]
    strand = gene[strand_col]

    for ltype in ["TSS", "TES", "terminal_exon_start", "terminal_exon_end"]:
        positions = _landmark_positions(lm, ltype)
        if not positions:
            add(ltype, pd.NA, "fail", f"Missing {ltype}", "error", f"Derive {ltype} from transcript annotation")
        else:
            add(ltype, positions[0], "pass", "", "info", "")

    for ltype in ["TSS", "TES"]:
        positions = _landmark_positions(lm, ltype)
        if len(positions) > 1:
            add(ltype, ",".join(map(str, positions)), "warn", f"Multiple {ltype} landmarks for one transcript", "warning", "Deduplicate transcript landmarks")

    bad_pos = lm[pd.to_numeric(lm["position"], errors="coerce") < 0]
    for _, row in bad_pos.iterrows():
        add(row["landmark_type"], row["position"], "fail", "Negative landmark position", "error", "Drop or correct invalid coordinate")

    wrong_chrom = lm[lm["chrom"] != chrom]
    for _, row in wrong_chrom.iterrows():
        add(row["landmark_type"], row["position"], "fail", "Landmark chromosome does not match gene chromosome", "error", f"Use chromosome {chrom}")

    expected_splice = _splice_landmarks(exons)
    for ltype in ["donor_splice_site", "acceptor_splice_site"]:
        observed = set(_landmark_positions(lm, ltype))
        expected = set(expected_splice.loc[expected_splice["landmark_type"] == ltype, "position"].astype(int)) if not expected_splice.empty else set()
        missing = expected - observed
        unexpected = observed - expected
        for pos in sorted(missing):
            add(ltype, pos, "fail", f"Missing expected {ltype}", "error", "Recompute splice landmarks from exon boundaries and strand")
        for pos in sorted(unexpected):
            add(ltype, pos, "fail", f"Unexpected {ltype} for exon structure/strand", "error", "Check donor/acceptor strand logic")
        if not missing and not unexpected:
            add(ltype, pd.NA, "pass", "", "info", "")

    pas = _filter_pas(pas_sites, gene, lm) if pas_sites is not None else pd.DataFrame()
    if not pas.empty:
        pas_start = "start" if "start" in pas.columns else "Start"
        pas_end = "end" if "end" in pas.columns else "End"
        for _, row in pas.iterrows():
            pos = int(row[pas_start])
            if pos < g_start or pos > g_end:
                add("PAS", pos, "fail", "PAS outside gene bounds", "error", "Check PAS-gene assignment or coordinate system")
        te_start = _landmark_positions(lm, "terminal_exon_start")
        te_end = _landmark_positions(lm, "terminal_exon_end")
        if te_start and te_end:
            low, high = min(te_start[0], te_end[0]), max(te_start[0], te_end[0])
            terminal_pas = pas[pas.get("site_class", pd.Series("", index=pas.index)).astype(str).str.contains("terminal|3utr|known", case=False, na=False)]
            for _, row in terminal_pas.iterrows():
                pos = int(row[pas_start])
                if not (low <= pos <= high):
                    add("PAS", pos, "warn", "Terminal/3UTR-like PAS outside terminal exon", "warning", "Inspect terminal exon/3UTR assignment")

    starts = _landmark_positions(lm, "translation_start_site")
    stops = _landmark_positions(lm, "translation_end_site")
    if starts and stops:
        start_rank = _transcript_rank(starts[0], g_start, g_end, strand)
        stop_rank = _transcript_rank(stops[0], g_start, g_end, strand)
        if start_rank > stop_rank:
            add("translation_start_site", starts[0], "fail", "translation_start occurs after translation_end in transcript orientation", "error", "Check CDS/start/stop landmark derivation")
        else:
            add("translation_start_site", starts[0], "pass", "", "info", "")
            add("translation_end_site", stops[0], "pass", "", "info", "")

    return pd.DataFrame(rows)


def _pas_position(row: pd.Series) -> int:
    if "start" in row.index:
        return int(row["start"])
    if "Start" in row.index:
        return int(row["Start"])
    if "position" in row.index:
        return int(row["position"])
    raise KeyError("PAS row must include start/Start/position")


def _source_color(source: object) -> str:
    src = str(source).lower()
    for key, color in PAS_COLORS.items():
        if key in src:
            return color
    return "0.55"


def _event_class_for_site(site_id: object, apa_events: pd.DataFrame | None) -> str | None:
    if apa_events is None or apa_events.empty or "site_id" not in apa_events.columns:
        return None
    hits = apa_events[apa_events["site_id"].astype(str) == str(site_id)]
    if hits.empty:
        return None
    if "priority_class" in hits.columns:
        priorities = list(hits["priority_class"].dropna().astype(str))
        for cls in ["fdr_and_delta", "fdr_only", "delta_only"]:
            if cls in priorities:
                return cls
    return None



def _source_key(source: object) -> str:
    src = str(source).lower()
    for key in PAS_COLORS:
        if key in src:
            if key in {"known_pas", "pas_reference"}:
                return "known"
            return key
    return "known"


def _source_from_pas_row(row: pd.Series) -> object:
    return row.get("source", row.get("site_source", row.get("reference_source", "known")))


def _segments_for_transcript(
    exons: pd.DataFrame,
    gene_start: int,
    gene_end: int,
    strand: str,
) -> list[dict[str, object]]:
    """Return exon/intron segments in transcript order."""
    start_col = _start_col(exons)
    end_col = _end_col(exons)
    genomic_exons = [
        {"kind": "exon", "start": int(row[start_col]), "end": int(row[end_col]), "terminal": False}
        for _, row in exons.sort_values(start_col).iterrows()
    ]
    terminal = _terminal_exon(exons, strand)
    if terminal is not None:
        t_start = int(terminal[start_col])
        t_end = int(terminal[end_col])
        for seg in genomic_exons:
            seg["terminal"] = seg["start"] == t_start and seg["end"] == t_end

    genomic: list[dict[str, object]] = []
    if not genomic_exons:
        genomic = [{"kind": "gene", "start": gene_start, "end": gene_end, "terminal": False}]
    else:
        if gene_start < genomic_exons[0]["start"]:
            genomic.append({"kind": "flank", "start": gene_start, "end": genomic_exons[0]["start"], "terminal": False})
        for i, exon in enumerate(genomic_exons):
            genomic.append(exon)
            if i < len(genomic_exons) - 1 and exon["end"] < genomic_exons[i + 1]["start"]:
                genomic.append({"kind": "intron", "start": exon["end"], "end": genomic_exons[i + 1]["start"], "terminal": False})
        if genomic_exons[-1]["end"] < gene_end:
            genomic.append({"kind": "flank", "start": genomic_exons[-1]["end"], "end": gene_end, "terminal": False})

    ordered = genomic if strand == "+" else list(reversed(genomic))
    return [seg for seg in ordered if int(seg["end"]) > int(seg["start"])]


def _collapsed_width(seg: dict[str, object], exon_width: float, intron_width: float, terminal_scale: float) -> float:
    length = max(1, int(seg["end"]) - int(seg["start"]))
    if seg["kind"] == "exon" and bool(seg.get("terminal")):
        return max(exon_width, length * terminal_scale)
    if seg["kind"] == "exon":
        return exon_width
    return intron_width


def _build_collapsed_mapper(
    exons: pd.DataFrame,
    gene_start: int,
    gene_end: int,
    strand: str,
    exon_width: float = 260.0,
    intron_width: float = 90.0,
    terminal_scale: float = 0.20,
    gap_width: float = 8.0,
):
    """Build a compact transcript-view mapper.

    The compact view intentionally collapses all non-terminal exonic/intronic
    sequence into one transcript-body block, then draws the terminal exon as a
    separate length-scaled block. This avoids one rectangle per internal exon
    for long multi-exon genes while preserving transcript order and terminal
    3' architecture.
    """
    terminal = _terminal_exon(exons, strand)
    if terminal is None:
        width = max(exon_width, 1.0)
        mapped = [{"kind": "exon_body", "start": gene_start, "end": gene_end, "terminal": False, "x0": 0.0, "x1": width}]

        def map_pos_no_terminal(pos: int | float) -> float:
            p = min(max(float(pos), gene_start), gene_end)
            frac = (p - gene_start) / max(1.0, gene_end - gene_start)
            if strand == "-":
                frac = 1.0 - frac
            return frac * width

        return map_pos_no_terminal, mapped

    start_col = _start_col(exons)
    end_col = _end_col(exons)
    t_start = int(terminal[start_col])
    t_end = int(terminal[end_col])
    terminal_len = max(1, t_end - t_start)
    body_width = max(exon_width, 1.0)
    connector_width = max(intron_width, 1.0)
    terminal_width = min(max(exon_width * 0.85, terminal_len * terminal_scale), exon_width * 1.8)
    terminal_x0 = body_width + connector_width
    terminal_x1 = terminal_x0 + terminal_width

    if strand == "+":
        body_start, body_end = gene_start, t_start
    else:
        body_start, body_end = t_end, gene_end

    mapped = [
        {"kind": "exon_body", "start": min(body_start, body_end), "end": max(body_start, body_end), "terminal": False, "x0": 0.0, "x1": body_width},
        {"kind": "connector", "start": min(body_start, body_end), "end": max(body_start, body_end), "terminal": False, "x0": body_width, "x1": terminal_x0},
        {"kind": "exon", "start": t_start, "end": t_end, "terminal": True, "x0": terminal_x0, "x1": terminal_x1},
    ]

    def map_pos(pos: int | float) -> float:
        p = min(max(float(pos), gene_start), gene_end)
        if strand == "+":
            if p < t_start:
                frac = (p - gene_start) / max(1.0, t_start - gene_start)
                return frac * body_width
            frac = (p - t_start) / max(1.0, t_end - t_start)
            return terminal_x0 + frac * terminal_width
        if p > t_end:
            frac = (gene_end - p) / max(1.0, gene_end - t_end)
            return frac * body_width
        frac = (t_end - p) / max(1.0, t_end - t_start)
        return terminal_x0 + frac * terminal_width

    return map_pos, mapped

def summarize_landmark_signal_windows(
    landmarks: pd.DataFrame,
    pas_sites: pd.DataFrame | None = None,
    apa_events: pd.DataFrame | None = None,
    flank: int = 500,
) -> pd.DataFrame:
    """Summarize PAS/event signal falling inside windows around transcript landmarks."""
    lm = landmarks.copy()
    rows: list[dict[str, object]] = []
    pas = pas_sites.copy() if pas_sites is not None else pd.DataFrame()
    if not pas.empty:
        pas["_position"] = pas.apply(_pas_position, axis=1)
        pas["_source_key"] = pas.apply(lambda row: _source_key(_source_from_pas_row(row)), axis=1)
    for _, row in lm.iterrows():
        ltype = row["landmark_type"]
        pos = int(row["position"])
        rec: dict[str, object] = {
            "gene_name": row.get("gene_name"),
            "transcript_id": row.get("transcript_id"),
            "landmark_type": ltype,
            "position": pos,
            "window_start": max(0, pos - flank),
            "window_end": pos + flank,
            "n_pas": 0,
            "n_apa_events": 0,
            "mean_pas_per_window": 0.0,
        }
        for source in ["known", "sierra", "scapture", "scpolaseq"]:
            rec[f"n_pas_{source}"] = 0
        for cls in ["fdr_only", "delta_only", "fdr_and_delta"]:
            rec[f"n_apa_{cls}"] = 0
        if not pas.empty:
            hits = pas[(pas["_position"] >= rec["window_start"]) & (pas["_position"] <= rec["window_end"])]
            rec["n_pas"] = int(len(hits))
            rec["mean_pas_per_window"] = float(len(hits))
            for source, count in hits["_source_key"].value_counts().items():
                rec[f"n_pas_{source}"] = int(count)
            if apa_events is not None and not apa_events.empty and "site_id" in hits.columns:
                for site_id in hits["site_id"].dropna().astype(str):
                    cls = _event_class_for_site(site_id, apa_events)
                    if cls:
                        rec["n_apa_events"] = int(rec["n_apa_events"]) + 1
                        rec[f"n_apa_{cls}"] = int(rec[f"n_apa_{cls}"]) + 1
        rows.append(rec)
    return pd.DataFrame(rows)

def _pas_as_landmarks(pas: pd.DataFrame, lm: pd.DataFrame) -> pd.DataFrame:
    if pas.empty or lm.empty:
        return pd.DataFrame(columns=lm.columns)
    template = lm.iloc[0].to_dict()
    rows = []
    for _, row in pas.iterrows():
        rec = {col: template.get(col, pd.NA) for col in lm.columns}
        rec["position"] = _pas_position(row)
        rec["landmark_type"] = "PAS"
        rec["feature_source"] = _source_from_pas_row(row)
        if "site_id" in row.index:
            rec["site_id"] = row["site_id"]
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_landmark_signal_summary(
    ax: plt.Axes,
    landmarks: pd.DataFrame,
    pas: pd.DataFrame,
    apa_events: pd.DataFrame | None,
    flank: int,
) -> pd.DataFrame:
    summary_input = landmarks.copy()
    pas_lm = _pas_as_landmarks(pas, landmarks)
    if not pas_lm.empty:
        summary_input = pd.concat([summary_input, pas_lm], ignore_index=True, sort=False)
    summary = summarize_landmark_signal_windows(summary_input, pas_sites=pas, apa_events=apa_events, flank=flank)
    if summary.empty:
        ax.text(0.5, 0.5, "No landmark signal windows", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return summary

    grouped = summary.groupby("landmark_type", observed=True).agg(mean_pas=("n_pas", "mean"), n_windows=("n_pas", "size")).reset_index()
    present = set(grouped["landmark_type"].astype(str))
    order = [name for name in SUMMARY_LANDMARK_ORDER if name in present]
    order += sorted(present - set(order))
    grouped["_order"] = grouped["landmark_type"].map({name: i for i, name in enumerate(order)})
    grouped = grouped.sort_values("_order")
    xs = np.arange(len(grouped))
    ax.bar(xs, grouped["mean_pas"], color="#8aa6c1", edgecolor="0.25", linewidth=0.5)
    for x, (_, row) in zip(xs, grouped.iterrows()):
        ax.text(x, float(row["mean_pas"]) + 0.03, f"n={int(row['n_windows'])}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(grouped["landmark_type"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean PAS/window", fontsize=8)
    ax.set_title(f"Landmark-window PAS signal (±{flank} bp)", fontsize=9, loc="left")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ymax = max(1.0, float(grouped["mean_pas"].max()) * 1.25)
    ax.set_ylim(0, ymax)
    return summary


def plot_transcript_landmark_diagram(
    gene_name: str,
    gtf: pd.DataFrame,
    landmarks: pd.DataFrame,
    pas_sites: pd.DataFrame | None = None,
    transcript_id: str | None = None,
    apa_events: pd.DataFrame | None = None,
    flank: int = 500,
    output: str | Path | None = None,
    show: bool = True,
    collapsed: bool = False,
    show_landmark_signal: bool | None = None,
    collapsed_exon_width: float = 260.0,
    collapsed_intron_width: float = 90.0,
    terminal_exon_scale: float = 0.20,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a transcript architecture diagram with landmarks and PAS sites.

    When ``collapsed=True``, internal exons/introns are summarized as one
    transcript-body block while the terminal exon keeps a length-scaled
    representation. This keeps landmark ordering auditable without drawing one
    box per internal exon.
    """
    gene = _select_gene(gtf, gene_name)
    tx_id, _ = _select_transcript(gtf, gene, transcript_id)
    if transcript_id and transcript_id != "auto":
        tx_id = transcript_id
    lm = _filter_landmarks(landmarks, gene_name, tx_id)
    if lm.empty and transcript_id in (None, "auto"):
        lm = _filter_landmarks(landmarks, gene_name, None)
    exons = _exons_for_transcript(gtf, gene, tx_id)
    if exons.empty:
        raise ValueError(f"No exons found for {gene_name} / {tx_id}")

    start_col = _start_col(gtf)
    end_col = _end_col(gtf)
    chrom_col = _chrom_col(gtf)
    strand_col = _strand_col(gtf)
    gene_start = int(gene[start_col])
    gene_end = int(gene[end_col])
    chrom = gene[chrom_col]
    strand = gene[strand_col]
    pas = _filter_pas(pas_sites, gene, lm)
    show_signal = collapsed if show_landmark_signal is None else show_landmark_signal

    if collapsed:
        map_x, mapped_segments = _build_collapsed_mapper(
            exons,
            gene_start,
            gene_end,
            strand,
            exon_width=collapsed_exon_width,
            intron_width=collapsed_intron_width,
            terminal_scale=terminal_exon_scale,
        )
        x_min = 0.0
        x_max = float(mapped_segments[-1]["x1"]) if mapped_segments else 1.0
    else:
        map_x = lambda pos: float(pos)
        mapped_segments = []
        x_min = max(0, min(gene_start, int(exons[_start_col(exons)].min())) - flank)
        x_max = max(gene_end, int(exons[_end_col(exons)].max())) + flank

    if show_signal:
        fig, (ax, signal_ax) = plt.subplots(
            2,
            1,
            figsize=(13, 6.1),
            gridspec_kw={"height_ratios": [4.8, 1.3], "hspace": 0.15},
            sharex=False,
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 4.8))
        signal_ax = None

    y_arch = 0.0
    ax.hlines(y_arch, x_min, x_max, color="0.25", linewidth=1.0, zorder=1)

    terminal = _terminal_exon(exons, strand)
    if collapsed:
        for seg in mapped_segments:
            if seg["kind"] not in {"exon", "gene", "exon_body"}:
                continue
            is_terminal = bool(seg.get("terminal"))
            face = "#4c8bb8" if not is_terminal else "#f2c14e"
            ax.add_patch(Rectangle((float(seg["x0"]), y_arch - 0.16), float(seg["x1"]) - float(seg["x0"]), 0.32, facecolor=face, edgecolor="black", linewidth=0.7, zorder=3))
            if seg["kind"] == "exon_body":
                ax.text((float(seg["x0"]) + float(seg["x1"])) / 2, y_arch, "collapsed internal exons", ha="center", va="center", fontsize=7, color="white", zorder=5)
    else:
        for _, exon in exons.iterrows():
            x0 = int(exon[_start_col(exons)])
            x1 = int(exon[_end_col(exons)])
            is_terminal = terminal is not None and x0 == int(terminal[_start_col(exons)]) and x1 == int(terminal[_end_col(exons)])
            face = "#4c8bb8" if not is_terminal else "#f2c14e"
            ax.add_patch(Rectangle((x0, y_arch - 0.16), x1 - x0, 0.32, facecolor=face, edgecolor="black", linewidth=0.7, zorder=3))

    utr_start = _landmark_positions(lm, "3UTR_start")
    utr_end = _landmark_positions(lm, "3UTR_end")
    if utr_start and utr_end:
        x0, x1 = sorted([map_x(utr_start[0]), map_x(utr_end[0])])
        ax.add_patch(Rectangle((x0, y_arch - 0.17), max(0.6, x1 - x0), 0.34, facecolor="#88c999", edgecolor="black", linewidth=0.5, alpha=0.85, zorder=4))

    if collapsed:
        arrow_start, arrow_end = x_min, x_max
        label_x = (x_min + x_max) / 2
    else:
        arrow_start = gene_start if strand == "+" else gene_end
        arrow_end = gene_end if strand == "+" else gene_start
        label_x = (gene_start + gene_end) / 2
    ax.annotate("", xy=(arrow_end, 0.44), xytext=(arrow_start, 0.44), arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(label_x, 0.52, "Transcript orientation: 5' -> 3'", ha="center", va="bottom", fontsize=9)

    splice_positions: dict[str, list[float]] = {"donor_splice_site": [], "acceptor_splice_site": []}
    for _, row in lm.iterrows():
        ltype = row["landmark_type"]
        pos = int(row["position"])
        x = map_x(pos)
        if collapsed and ltype in splice_positions:
            splice_positions[ltype].append(x)
            continue
        if ltype in WINDOW_LANDMARKS:
            wx0, wx1 = sorted([map_x(max(0, pos - flank)), map_x(pos + flank)])
            ax.axvspan(wx0, wx1, color="gold", alpha=0.08, zorder=0)
        if ltype == "TSS":
            ax.vlines(x, -0.65, 0.75, color="purple", linewidth=1.2, zorder=5)
            ax.text(x, 0.8, "TSS", rotation=90, va="bottom", ha="center", fontsize=8, color="purple")
        elif ltype == "TES":
            ax.vlines(x, -0.65, 0.75, color="black", linewidth=1.2, zorder=5)
            ax.text(x, 0.8, "TES", rotation=90, va="bottom", ha="center", fontsize=8, color="black")
        elif ltype == "translation_start_site":
            ax.vlines(x, -0.48, 0.48, color="seagreen", linewidth=1.1, zorder=5)
            ax.text(x, -0.58, "START", rotation=90, va="top", ha="center", fontsize=8, color="seagreen")
        elif ltype == "translation_end_site":
            ax.vlines(x, -0.48, 0.48, color="crimson", linewidth=1.1, zorder=5)
            ax.text(x, -0.58, "STOP", rotation=90, va="top", ha="center", fontsize=8, color="crimson")
        elif ltype == "donor_splice_site":
            ax.plot(x, 0.26, marker="^", color="navy", markersize=6, zorder=6)
        elif ltype == "acceptor_splice_site":
            ax.plot(x, -0.26, marker="v", color="darkred", markersize=6, zorder=6)

    if collapsed:
        donor_x = splice_positions.get("donor_splice_site", [])
        acceptor_x = splice_positions.get("acceptor_splice_site", [])
        if donor_x:
            x = float(np.median(donor_x))
            ax.plot(x, 0.26, marker="^", color="navy", markersize=7, zorder=6)
            ax.text(x, 0.34, f"SD x{len(donor_x)}", ha="center", va="bottom", fontsize=8, color="navy")
        if acceptor_x:
            x = float(np.median(acceptor_x))
            ax.plot(x, -0.26, marker="v", color="darkred", markersize=7, zorder=6)
            ax.text(x, -0.34, f"SA x{len(acceptor_x)}", ha="center", va="top", fontsize=8, color="darkred")

    for _, row in pas.iterrows():
        pos = _pas_position(row)
        x = map_x(pos)
        wx0, wx1 = sorted([map_x(max(0, pos - flank)), map_x(pos + flank)])
        ax.axvspan(wx0, wx1, color="teal", alpha=0.05, zorder=0)
        src = _source_from_pas_row(row)
        cls = _event_class_for_site(row.get("site_id", ""), apa_events)
        edge = APA_EDGE.get(cls, "black")
        lw = 1.8 if cls else 0.4
        ax.plot(x, -0.52, marker="o", markersize=6.5, markerfacecolor=_source_color(src), markeredgecolor=edge, markeredgewidth=lw, zorder=7)

    ax.text(x_min, 1.05, f"Genomic strand: {strand}", fontsize=9, ha="left")
    coord_label = "Coordinates: collapsed transcript view; underlying landmarks are 0-based genomic" if collapsed else "Coordinates: 0-based genomic"
    ax.text(x_min, 0.92, coord_label, fontsize=9, ha="left")
    if collapsed:
        ax.text(x_min, -0.82, "Compact view: all internal exons/introns collapsed into one transcript body; terminal exon length-scaled", fontsize=8, ha="left", color="0.35")
    title_tx = tx_id or (lm["transcript_id"].iloc[0] if not lm.empty and "transcript_id" in lm.columns else "auto")
    title_prefix = "Collapsed transcript landmark diagram" if collapsed else "Transcript landmark diagram"
    ax.set_title(f"{title_prefix}: {gene_name} | {title_tx}", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.9, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Collapsed transcript architecture" if collapsed else f"{chrom} (0-based genomic)")

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#4c8bb8", edgecolor="black", label="collapsed internal exons" if collapsed else "exon"),
        Rectangle((0, 0), 1, 1, facecolor="#f2c14e", edgecolor="black", label="terminal exon"),
        Rectangle((0, 0), 1, 1, facecolor="#88c999", edgecolor="black", label="3'UTR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.55", markeredgecolor="black", label="PAS", markersize=6),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if signal_ax is not None:
        _plot_landmark_signal_summary(signal_ax, lm, pas, apa_events, flank)

    if signal_ax is not None:
        fig.subplots_adjust(hspace=0.35, bottom=0.16, top=0.90)
    else:
        plt.tight_layout()
    if output:
        save_figure(fig, output)
    if show:
        plt.show()
    return fig, ax
