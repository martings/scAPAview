"""Tests for scapaview.apa module."""

import pandas as pd
import pytest

from scapaview.apa import (
    classify_apa_events,
    merge_sierra_scapture_sites,
    rank_pas_within_gene,
    build_unified_pas_table,
)


# ---------------------------------------------------------------------------
# classify_apa_events
# ---------------------------------------------------------------------------


def _make_events(**kwargs) -> pd.DataFrame:
    defaults = dict(
        gene_id=["G1", "G2", "G3", "G4"],
        site_id=["s1", "s2", "s3", "s4"],
        group_a=["A"] * 4,
        group_b=["B"] * 4,
        pdui_a=[0.5, 0.5, 0.5, 0.5],
        pdui_b=[0.5, 0.5, 0.5, 0.5],
        delta_pdui=[0.20, 0.20, 0.05, 0.05],
        p_value=[0.001, 0.10, 0.001, 0.50],
        adj_p_value=[0.002, 0.10, 0.002, 0.50],
    )
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_classify_fdr_and_delta():
    events = _make_events()
    classified = classify_apa_events(events)
    # Row 0: fdr significant (0.002 < 0.05) AND delta >= 0.15 → fdr_and_delta
    assert classified.loc[0, "priority_class"] == "fdr_and_delta"
    assert bool(classified.loc[0, "is_fdr_and_delta"]) is True


def test_classify_delta_only():
    events = _make_events()
    classified = classify_apa_events(events)
    # Row 1: NOT fdr significant (0.10 >= 0.05) but delta >= 0.15 → delta_only
    assert classified.loc[1, "priority_class"] == "delta_only"
    assert bool(classified.loc[1, "is_delta_candidate"]) is True
    assert bool(classified.loc[1, "is_fdr_significant"]) is False


def test_classify_fdr_only():
    events = _make_events()
    classified = classify_apa_events(events)
    # Row 2: fdr significant (0.002 < 0.05) but delta < 0.15 → fdr_only
    assert classified.loc[2, "priority_class"] == "fdr_only"
    assert bool(classified.loc[2, "is_fdr_significant"]) is True
    assert bool(classified.loc[2, "is_delta_candidate"]) is False


def test_classify_not_significant():
    events = _make_events()
    classified = classify_apa_events(events)
    # Row 3: not fdr significant AND delta < 0.15 → not_significant
    assert classified.loc[3, "priority_class"] == "not_significant"


def test_classify_direction_lengthening():
    events = pd.DataFrame(
        dict(
            gene_id=["G1"], site_id=["s1"], group_a=["A"], group_b=["B"],
            pdui_a=[0.3], pdui_b=[0.7],
            delta_pdui=[0.4], p_value=[0.01], adj_p_value=[0.01],
        )
    )
    classified = classify_apa_events(events)
    assert classified.loc[0, "direction"] == "lengthening"


def test_classify_direction_shortening():
    events = pd.DataFrame(
        dict(
            gene_id=["G1"], site_id=["s1"], group_a=["A"], group_b=["B"],
            pdui_a=[0.7], pdui_b=[0.3],
            delta_pdui=[-0.4], p_value=[0.01], adj_p_value=[0.01],
        )
    )
    classified = classify_apa_events(events)
    assert classified.loc[0, "direction"] == "shortening"


# ---------------------------------------------------------------------------
# merge_sierra_scapture_sites
# ---------------------------------------------------------------------------


def _make_sites(starts, gene="G1", source_col=None, strand="+") -> pd.DataFrame:
    df = pd.DataFrame(
        dict(
            site_id=[f"{gene}_s{i}" for i in range(len(starts))],
            gene_id=[gene] * len(starts),
            chrom=["chr1"] * len(starts),
            start=starts,
            end=[s + 1 for s in starts],
            strand=[strand] * len(starts),
        )
    )
    if source_col:
        df["source"] = source_col
    return df


def test_merge_within_window():
    sierra = _make_sites([1000, 2000])
    # scapture site at 1010 is within default window=25 of 1000
    scapture = _make_sites([1010, 3000], gene="G1")
    merged = merge_sierra_scapture_sites(sierra, scapture, window=25)
    # 1010 merges into 1000; 3000 is new → total should be 3
    assert len(merged) == 3


def test_merge_outside_window():
    sierra = _make_sites([1000])
    scapture = _make_sites([2000])
    merged = merge_sierra_scapture_sites(sierra, scapture, window=25)
    assert len(merged) == 2


def test_merge_source_annotation():
    sierra = _make_sites([1000])
    scapture = _make_sites([1010])
    merged = merge_sierra_scapture_sites(sierra, scapture, window=25)
    assert "scapture" in merged.loc[0, "source"]


# ---------------------------------------------------------------------------
# rank_pas_within_gene
# ---------------------------------------------------------------------------


def test_rank_plus_strand():
    sites = pd.DataFrame(
        dict(
            site_id=["s1", "s2", "s3"],
            gene_id=["G1", "G1", "G1"],
            chrom=["chr1"] * 3,
            start=[3000, 1000, 2000],
            end=[3001, 1001, 2001],
            strand=["+", "+", "+"],
        )
    )
    ranked = rank_pas_within_gene(sites)
    # Rank 1 = lowest start = 1000
    rank_map = dict(zip(ranked["start"], ranked["pas_rank_in_gene"]))
    assert rank_map[1000] == 1
    assert rank_map[2000] == 2
    assert rank_map[3000] == 3


def test_rank_minus_strand():
    sites = pd.DataFrame(
        dict(
            site_id=["s1", "s2", "s3"],
            gene_id=["G1", "G1", "G1"],
            chrom=["chr1"] * 3,
            start=[1000, 2000, 3000],
            end=[1001, 2001, 3001],
            strand=["-", "-", "-"],
        )
    )
    ranked = rank_pas_within_gene(sites)
    # For - strand, rank 1 = highest genomic start (most 5' when reading 3'→5')
    rank_map = dict(zip(ranked["start"], ranked["pas_rank_in_gene"]))
    assert rank_map[3000] == 1
    assert rank_map[2000] == 2
    assert rank_map[1000] == 3


# ---------------------------------------------------------------------------
# build_unified_pas_table
# ---------------------------------------------------------------------------


def test_build_unified_scpolaseq_only(toy_pas_sites):
    unified = build_unified_pas_table(scpolaseq_sites=toy_pas_sites)
    assert len(unified) == len(toy_pas_sites)
    assert "pas_rank_in_gene" in unified.columns


def test_build_unified_sierra_scapture():
    sierra = _make_sites([1000, 2000])
    scapture = _make_sites([1010, 3000])
    unified = build_unified_pas_table(sierra_sites=sierra, scapture_sites=scapture)
    assert "source" in unified.columns
    assert len(unified) == 3  # 1010 merges into 1000


def test_build_unified_empty():
    result = build_unified_pas_table()
    assert result.empty


def test_build_unified_all_sources(toy_pas_sites):
    sierra = _make_sites([500])
    scapture = _make_sites([600])
    unified = build_unified_pas_table(
        scpolaseq_sites=toy_pas_sites,
        sierra_sites=sierra,
        scapture_sites=scapture,
    )
    assert len(unified) >= len(toy_pas_sites)
