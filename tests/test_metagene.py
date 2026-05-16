"""Tests for scapaview.metagene module."""

import numpy as np
import pandas as pd
import pytest

from scapaview.metagene import compute_pas_density_metagene, plot_metagene_3utr


# ---------------------------------------------------------------------------
# compute_pas_density_metagene
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_regions():
    return pd.DataFrame(
        dict(
            chrom=["chr1"],
            start=[1000],
            end=[2000],
            strand=["+"],
        )
    )


@pytest.fixture
def simple_pas(simple_regions):
    # PAS site at position 1500, which is midpoint (0.5) of [1000, 2000)
    return pd.DataFrame(
        dict(
            site_id=["s1"],
            gene_id=["G1"],
            chrom=["chr1"],
            start=[1500],
            end=[1501],
            strand=["+"],
        )
    )


def test_compute_pas_density_returns_n_bins(simple_regions, simple_pas):
    density = compute_pas_density_metagene(simple_pas, simple_regions, n_bins=100)
    assert density.shape == (100,)


def test_compute_pas_density_correct_bin(simple_regions, simple_pas):
    density = compute_pas_density_metagene(simple_pas, simple_regions, n_bins=100)
    # PAS at 1500 in [1000,2000) → rel=0.5 → bin 50
    assert density[50] == 1.0


def test_compute_pas_density_empty_pas(simple_regions):
    empty_pas = pd.DataFrame(columns=["site_id", "gene_id", "chrom", "start", "end", "strand"])
    density = compute_pas_density_metagene(empty_pas, simple_regions, n_bins=50)
    assert density.shape == (50,)
    assert np.all(density == 0)


def test_compute_pas_density_empty_regions(simple_pas):
    empty_regions = pd.DataFrame(columns=["chrom", "start", "end", "strand"])
    density = compute_pas_density_metagene(simple_pas, empty_regions, n_bins=50)
    assert density.shape == (50,)
    assert np.all(density == 0)


def test_compute_pas_density_minus_strand():
    regions = pd.DataFrame(dict(chrom=["chr1"], start=[1000], end=[2000], strand=["-"]))
    # PAS at 1500: raw rel=(1500-1000)/1000=0.5; for - strand: 1-0.5=0.5 → bin 50
    pas = pd.DataFrame(
        dict(site_id=["s1"], gene_id=["G1"], chrom=["chr1"],
             start=[1500], end=[1501], strand=["-"])
    )
    density = compute_pas_density_metagene(pas, regions, n_bins=100)
    assert density[50] == 1.0


# ---------------------------------------------------------------------------
# plot_metagene_3utr
# ---------------------------------------------------------------------------


def test_plot_metagene_3utr_returns_fig_ax(simple_regions, monkeypatch):
    """plot_metagene_3utr should return (fig, ax) even with empty bigwig paths."""
    import matplotlib
    matplotlib.use("Agg")

    # Patch aggregate_metagene_coverage to return zeros (no actual bigWig needed)
    import scapaview.metagene as meta_mod

    def mock_aggregate(*args, **kwargs):
        n_bins = kwargs.get("n_bins", 100)
        return np.zeros(n_bins)

    monkeypatch.setattr("scapaview.coverage.aggregate_metagene_coverage", mock_aggregate)

    fig, ax = plot_metagene_3utr(
        bw_paths_a=[],
        bw_paths_b=None,
        regions=simple_regions,
        n_bins=100,
        show=False,
    )
    import matplotlib.pyplot as plt

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
