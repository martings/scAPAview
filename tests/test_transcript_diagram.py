"""Tests for transcript landmark diagrams and QC."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scapaview.transcript_diagram import (
    build_transcript_landmarks,
    plot_transcript_landmark_diagram,
    qc_landmark_consistency,
)


def _gtf(strand="+"):
    chrom = "chr1"
    if strand == "+":
        gene_start, gene_end = 1000, 5000
        exons = [(1000, 1500), (2000, 2800), (3500, 5000)]
        utr = (4500, 5000)
        cds = (1200, 4300)
        gene = "GENEP"
        tx = "GENEP.T1"
    else:
        gene_start, gene_end = 7000, 12000
        exons = [(7000, 8000), (9000, 10000), (11000, 12000)]
        utr = (7000, 7600)
        cds = (7800, 11500)
        gene = "GENEM"
        tx = "GENEM.T1"
    rows = [
        {"Feature": "gene", "gene_id": gene, "gene_id_base": gene, "gene_name": gene, "transcript_id": pd.NA, "Chromosome": chrom, "Start": gene_start, "End": gene_end, "Strand": strand},
        {"Feature": "transcript", "gene_id": gene, "gene_id_base": gene, "gene_name": gene, "transcript_id": tx, "Chromosome": chrom, "Start": gene_start, "End": gene_end, "Strand": strand},
    ]
    for i, (start, end) in enumerate(exons, start=1):
        rows.append({"Feature": "exon", "gene_id": gene, "gene_id_base": gene, "gene_name": gene, "transcript_id": tx, "Chromosome": chrom, "Start": start, "End": end, "Strand": strand, "exon_number": str(i)})
    rows.append({"Feature": "three_prime_utr", "gene_id": gene, "gene_id_base": gene, "gene_name": gene, "transcript_id": tx, "Chromosome": chrom, "Start": utr[0], "End": utr[1], "Strand": strand})
    rows.append({"Feature": "CDS", "gene_id": gene, "gene_id_base": gene, "gene_name": gene, "transcript_id": tx, "Chromosome": chrom, "Start": cds[0], "End": cds[1], "Strand": strand})
    return pd.DataFrame(rows)


def test_diagram_generates_for_plus_gene():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    pas = pd.DataFrame({"site_id": ["pas1"], "gene_id": ["GENEP"], "gene_name": ["GENEP"], "chrom": ["chr1"], "start": [4700], "end": [4701], "strand": ["+"], "source": ["sierra"], "site_class": ["known_pas"]})
    fig, ax = plot_transcript_landmark_diagram("GENEP", gtf, landmarks, pas_sites=pas, transcript_id="GENEP.T1", show=False)
    assert fig is not None
    plt.close(fig)


def test_diagram_generates_for_minus_gene():
    gtf = _gtf("-")
    landmarks = build_transcript_landmarks("GENEM", gtf, transcript_id="GENEM.T1")
    fig, ax = plot_transcript_landmark_diagram("GENEM", gtf, landmarks, transcript_id="GENEM.T1", show=False)
    assert fig is not None
    plt.close(fig)


def test_donor_acceptor_change_by_strand():
    plus = build_transcript_landmarks("GENEP", _gtf("+"), transcript_id="GENEP.T1")
    minus = build_transcript_landmarks("GENEM", _gtf("-"), transcript_id="GENEM.T1")
    assert set(plus.loc[plus["landmark_type"] == "donor_splice_site", "position"]) == {1500, 2800}
    assert set(plus.loc[plus["landmark_type"] == "acceptor_splice_site", "position"]) == {2000, 3500}
    assert set(minus.loc[minus["landmark_type"] == "donor_splice_site", "position"]) == {9000, 11000}
    assert set(minus.loc[minus["landmark_type"] == "acceptor_splice_site", "position"]) == {8000, 10000}


def test_qc_detects_pas_outside_gene():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    pas = pd.DataFrame({"site_id": ["bad"], "gene_id": ["GENEP"], "gene_name": ["GENEP"], "chrom": ["chr1"], "start": [9000], "end": [9001], "strand": ["+"], "source": ["sierra"], "site_class": ["known_pas"]})
    qc = qc_landmark_consistency(gtf, landmarks, pas_sites=pas, gene_name="GENEP", transcript_id="GENEP.T1")
    assert ((qc["landmark_type"] == "PAS") & qc["issue"].str.contains("outside gene bounds", na=False)).any()


def test_qc_detects_translation_order_error():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    landmarks.loc[landmarks["landmark_type"] == "translation_start_site", "position"] = 4600
    landmarks.loc[landmarks["landmark_type"] == "translation_end_site", "position"] = 1300
    qc = qc_landmark_consistency(gtf, landmarks, gene_name="GENEP", transcript_id="GENEP.T1")
    assert qc["issue"].str.contains("translation_start occurs after translation_end", na=False).any()


def test_qc_detects_wrong_chromosome():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    landmarks.loc[landmarks["landmark_type"] == "TES", "chrom"] = "chr2"
    qc = qc_landmark_consistency(gtf, landmarks, gene_name="GENEP", transcript_id="GENEP.T1")
    assert qc["issue"].str.contains("chromosome does not match", na=False).any()


def test_figure_includes_required_landmark_labels_and_terminal_exon():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    pas = pd.DataFrame({"site_id": ["pas1"], "gene_id": ["GENEP"], "gene_name": ["GENEP"], "chrom": ["chr1"], "start": [4700], "end": [4701], "strand": ["+"], "source": ["sierra"], "site_class": ["known_pas"]})
    fig, ax = plot_transcript_landmark_diagram("GENEP", gtf, landmarks, pas_sites=pas, transcript_id="GENEP.T1", show=False)
    labels = " ".join(text.get_text() for text in ax.texts)
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "TSS" in labels
    assert "TES" in labels
    assert "START" in labels
    assert "STOP" in labels
    assert "terminal exon" in legend_labels
    assert any(line.get_marker() == "^" for line in ax.lines)
    assert any(line.get_marker() == "v" for line in ax.lines)
    assert any(line.get_marker() == "o" for line in ax.lines)
    plt.close(fig)



def test_collapsed_diagram_shortens_internal_structure_and_adds_signal_panel():
    gtf = _gtf("+")
    landmarks = build_transcript_landmarks("GENEP", gtf, transcript_id="GENEP.T1")
    pas = pd.DataFrame({
        "site_id": ["pas1", "pas2"],
        "gene_id": ["GENEP", "GENEP"],
        "gene_name": ["GENEP", "GENEP"],
        "chrom": ["chr1", "chr1"],
        "start": [4700, 2100],
        "end": [4701, 2101],
        "strand": ["+", "+"],
        "source": ["sierra", "scpolaseq"],
        "site_class": ["known_pas", "internal"],
    })
    fig, ax = plot_transcript_landmark_diagram(
        "GENEP",
        gtf,
        landmarks,
        pas_sites=pas,
        transcript_id="GENEP.T1",
        flank=300,
        collapsed=True,
        show=False,
    )
    assert len(fig.axes) == 2
    assert "Collapsed transcript" in ax.get_title()
    assert "Collapsed transcript architecture" in ax.get_xlabel()
    assert ax.get_xlim()[1] < 1000
    exon_widths = [patch.get_width() for patch in ax.patches if abs(patch.get_y() + 0.16) < 1e-6 and abs(patch.get_height() - 0.32) < 1e-6]
    assert len(exon_widths) == 2
    assert any("collapsed internal exons" in text.get_text() for text in ax.texts)
    signal_labels = [label.get_text() for label in fig.axes[1].get_xticklabels()]
    assert "PAS" in signal_labels
    plt.close(fig)


def test_collapsed_diagram_keeps_minus_strand_transcript_orientation_left_to_right():
    gtf = _gtf("-")
    landmarks = build_transcript_landmarks("GENEM", gtf, transcript_id="GENEM.T1")
    fig, ax = plot_transcript_landmark_diagram("GENEM", gtf, landmarks, transcript_id="GENEM.T1", collapsed=True, show=False)
    labels = " ".join(text.get_text() for text in ax.texts)
    assert "Transcript orientation: 5' -> 3'" in labels
    assert ax.get_xlim()[0] == 0
    assert ax.get_xlim()[1] < 1000
    plt.close(fig)
