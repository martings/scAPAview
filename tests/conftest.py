"""Shared pytest fixtures for scAPAview tests."""

from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def toy_pas_sites() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "toy_pas_sites.tsv", sep="\t")


@pytest.fixture
def toy_apa_events() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "toy_apa_events.tsv", sep="\t")


@pytest.fixture
def toy_gtf_path() -> Path:
    return DATA_DIR / "toy.gtf"


@pytest.fixture
def toy_gene_sets_path() -> Path:
    return DATA_DIR / "toy_gene_sets.yaml"


@pytest.fixture
def toy_gtf_df(toy_gtf_path) -> pd.DataFrame:
    """Load the toy GTF as a DataFrame via pyranges, then standardize."""
    import pyranges as pr
    from scapaview.annotation import standardize_gtf

    gtf = pr.read_gtf(str(toy_gtf_path), as_df=True)
    return standardize_gtf(gtf)


@pytest.fixture
def toy_exons(toy_gtf_df) -> pd.DataFrame:
    from scapaview.annotation import build_exon_table

    return build_exon_table(toy_gtf_df)
