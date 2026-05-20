"""Tests for real scPolASeq/Sierra/scapture helper behavior."""

from pathlib import Path

import pandas as pd

from scapaview.genesets import summarize_gene_set_apa_burden
from scapaview.io import add_gene_id_columns, parse_site_id


def test_parse_site_id_normalizes_chr_and_base_gene():
    parsed = parse_site_id("ENSG00000000419.14:20:50957965-50957965:-")
    assert parsed["gene_id"] == "ENSG00000000419.14"
    assert parsed["gene_id_base"] == "ENSG00000000419"
    assert parsed["chrom"] == "chr20"
    assert parsed["start"] == 50957965
    assert parsed["end"] == 50957965
    assert parsed["strand"] == "-"


def test_add_gene_id_columns_keeps_versioned_and_base_ids():
    df = add_gene_id_columns(pd.DataFrame({"gene_id": ["ENSG1.5", "ENSG2"]}))
    assert df["gene_id"].tolist() == ["ENSG1.5", "ENSG2"]
    assert df["gene_id_base"].tolist() == ["ENSG1", "ENSG2"]


def test_gene_set_burden_maps_symbols_to_gene_ids():
    gene_table = pd.DataFrame(
        {"gene_id": ["ENSG1.1", "ENSG2.1"], "gene_id_base": ["ENSG1", "ENSG2"], "gene_name": ["CXCL10", "PAF1"]}
    )
    pas = pd.DataFrame({"gene_id": ["ENSG1.1", "ENSG2.1"], "gene_id_base": ["ENSG1", "ENSG2"], "site_id": ["s1", "s2"]})
    apa = pd.DataFrame(
        {
            "gene_id": ["ENSG1.1", "ENSG2.1"],
            "gene_id_base": ["ENSG1", "ENSG2"],
            "is_fdr_and_delta": [True, False],
            "is_fdr_significant": [True, False],
        }
    )
    burden = summarize_gene_set_apa_burden({"immune": ["CXCL10"]}, pas, apa, gene_table=gene_table)
    row = burden.iloc[0]
    assert row["n_genes_with_pas"] == 1
    assert row["n_pas_sites"] == 1
    assert row["n_apa_events"] == 1
    assert row["n_fdr_and_delta_events"] == 1
