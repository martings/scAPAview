"""Tests for gene-level track plotting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scapaview.tracks import plot_gene_apa_tracks


def _toy_gtf():
    return pd.DataFrame(
        {
            "Feature": ["gene", "exon"],
            "gene_id": ["GENE1", "GENE1"],
            "gene_id_base": ["GENE1", "GENE1"],
            "gene_name": ["G", "G"],
            "Chromosome": ["chr1", "chr1"],
            "Start": [100, 120],
            "End": [200, 180],
            "Strand": ["+", "+"],
        }
    )


def test_plot_gene_apa_tracks_shared_coverage_ylim(monkeypatch):
    calls = []

    def fake_extract(path, chrom, start, end):
        calls.append(path)
        if path == "df.bw":
            return np.array([0.0, 10.0, 5.0])
        return np.array([0.0, 100.0, 20.0])

    monkeypatch.setattr("scapaview.tracks.extract_bigwig_interval", fake_extract)
    fig, axes = plot_gene_apa_tracks(
        gene_name="G",
        gtf=_toy_gtf(),
        pas_sites=pd.DataFrame(),
        apa_events=None,
        bigwig_tracks={"DF": "df.bw", "DHF": "dhf.bw"},
        show=False,
    )

    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert axes[0].get_ylim()[1] == 105.0
    assert "shared coverage y-axis" in axes[0].texts[0].get_text()
    plt.close(fig)
