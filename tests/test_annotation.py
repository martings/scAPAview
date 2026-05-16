"""Tests for scapaview.annotation module."""

import math

import pandas as pd
import pytest

from scapaview.annotation import (
    compute_relative_position,
    derive_terminal_exons,
    derive_introns,
    derive_splice_sites,
    distance_to_nearest_splice_site,
    standardize_gtf,
    build_exon_table,
    build_gene_table,
)


# ---------------------------------------------------------------------------
# standardize_gtf
# ---------------------------------------------------------------------------


def test_standardize_gtf_converts_start():
    """GTF 1-based Start should become 0-based after standardisation."""
    df = pd.DataFrame({"Start": [1001, 2001], "End": [1500, 2800], "Feature": ["gene", "exon"]})
    result = standardize_gtf(df)
    assert result["Start"].tolist() == [1000, 2000]
    assert result["End"].tolist() == [1500, 2800]  # End unchanged


def test_standardize_gtf_lowercase_col():
    df = pd.DataFrame({"start": [101, 201], "end": [200, 300]})
    result = standardize_gtf(df)
    assert result["start"].tolist() == [100, 200]


def test_standardize_gtf_end_unchanged():
    df = pd.DataFrame({"Start": [100], "End": [500]})
    result = standardize_gtf(df)
    assert result["End"].iloc[0] == 500


# ---------------------------------------------------------------------------
# build_gene_table / build_exon_table (integration with toy GTF)
# ---------------------------------------------------------------------------


def test_build_gene_table(toy_gtf_df):
    genes = build_gene_table(toy_gtf_df)
    assert len(genes) == 2
    assert set(genes["gene_id"]) == {"GENE1", "GENE2"}


def test_build_exon_table(toy_gtf_df):
    exons = build_exon_table(toy_gtf_df)
    assert len(exons) == 6  # 3 exons × 2 genes


# ---------------------------------------------------------------------------
# derive_terminal_exons
# ---------------------------------------------------------------------------


def _make_exons(gene_id, transcript_id, starts, ends, strand, chrom="chr1"):
    return pd.DataFrame(
        dict(
            gene_id=[gene_id] * len(starts),
            transcript_id=[transcript_id] * len(starts),
            Chromosome=[chrom] * len(starts),
            Strand=[strand] * len(starts),
            Start=starts,
            End=ends,
            Feature=["exon"] * len(starts),
        )
    )


def test_derive_terminal_exons_plus():
    exons = _make_exons("G1", "T1", [1000, 2000, 3000], [1500, 2500, 3500], "+")
    terminal = derive_terminal_exons(exons)
    assert len(terminal) == 1
    assert terminal.iloc[0]["Start"] == 3000


def test_derive_terminal_exons_minus():
    exons = _make_exons("G1", "T1", [1000, 2000, 3000], [1500, 2500, 3500], "-")
    terminal = derive_terminal_exons(exons)
    assert len(terminal) == 1
    # For - strand, terminal exon has the smallest Start (most 5' in genomic coords = 3' end of transcript)
    assert terminal.iloc[0]["Start"] == 1000


def test_derive_terminal_exons_multi_gene(toy_exons):
    terminal = derive_terminal_exons(toy_exons)
    assert len(terminal) == 2  # one terminal exon per transcript/gene


# ---------------------------------------------------------------------------
# derive_introns
# ---------------------------------------------------------------------------


def test_derive_introns_basic():
    exons = _make_exons("G1", "T1", [1000, 2000, 3000], [1500, 2500, 3500], "+")
    introns = derive_introns(exons)
    # 3 exons → 2 introns
    assert len(introns) == 2
    # First intron spans from end of exon1 to start of exon2
    assert introns.iloc[0]["Start"] == 1500
    assert introns.iloc[0]["End"] == 2000


def test_derive_introns_single_exon():
    exons = _make_exons("G1", "T1", [1000], [1500], "+")
    introns = derive_introns(exons)
    assert len(introns) == 0


def test_derive_introns_toy(toy_exons):
    introns = derive_introns(toy_exons)
    # Each gene has 3 exons → 2 introns per gene → 4 total
    assert len(introns) == 4


# ---------------------------------------------------------------------------
# derive_splice_sites
# ---------------------------------------------------------------------------


def test_derive_splice_sites_count():
    exons = _make_exons("G1", "T1", [1000, 2000, 3000], [1500, 2500, 3500], "+")
    ss = derive_splice_sites(exons)
    # 3 exons: middle exon has donor and acceptor; first has donor only; last has acceptor only
    assert len(ss) == 4


# ---------------------------------------------------------------------------
# distance_to_nearest_splice_site
# ---------------------------------------------------------------------------


def test_distance_to_nearest_splice_site():
    pas = pd.DataFrame(
        dict(
            site_id=["s1"],
            gene_id=["G1"],
            chrom=["chr1"],
            start=[1550],
            end=[1551],
            strand=["+"],
        )
    )
    splice_sites = pd.DataFrame(
        dict(
            gene_id=["G1"],
            transcript_id=["T1"],
            chrom=["chr1"],
            strand=["+"],
            position=[1500],
            splice_site_type=["donor"],
        )
    )
    result = distance_to_nearest_splice_site(pas, splice_sites)
    assert result.loc[0, "nearest_splice_site_distance"] == 50
    assert result.loc[0, "nearest_splice_site_type"] == "donor"
    assert bool(result.loc[0, "is_splice_proximal"]) is True  # 50 <= 100


def test_distance_proximal_window():
    pas = pd.DataFrame(
        dict(site_id=["s1"], gene_id=["G1"], chrom=["chr1"], start=[1700], end=[1701], strand=["+"]),
    )
    splice_sites = pd.DataFrame(
        dict(gene_id=["G1"], transcript_id=["T1"], chrom=["chr1"], strand=["+"],
             position=[1500], splice_site_type=["donor"])
    )
    result = distance_to_nearest_splice_site(pas, splice_sites, window=100)
    assert bool(result.loc[0, "is_splice_proximal"]) is False  # 200 > 100


# ---------------------------------------------------------------------------
# compute_relative_position
# ---------------------------------------------------------------------------


def test_relative_position_plus_start():
    assert compute_relative_position(1000, 1000, 2000, "+") == pytest.approx(0.0)


def test_relative_position_plus_end():
    assert compute_relative_position(2000, 1000, 2000, "+") == pytest.approx(1.0)


def test_relative_position_plus_mid():
    assert compute_relative_position(1500, 1000, 2000, "+") == pytest.approx(0.5)


def test_relative_position_minus_start():
    # For - strand: site at region_end → relative 0 (5' end of transcript)
    assert compute_relative_position(2000, 1000, 2000, "-") == pytest.approx(0.0)


def test_relative_position_minus_end():
    # For - strand: site at region_start → relative 1 (3' end of transcript)
    assert compute_relative_position(1000, 1000, 2000, "-") == pytest.approx(1.0)


def test_relative_position_minus_mid():
    assert compute_relative_position(1500, 1000, 2000, "-") == pytest.approx(0.5)


def test_relative_position_zero_length():
    result = compute_relative_position(1000, 1000, 1000, "+")
    assert math.isnan(result)
