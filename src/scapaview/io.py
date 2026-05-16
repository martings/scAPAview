"""Input/output functions for reading all scAPAview file types."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column definitions
# ---------------------------------------------------------------------------

_APA_EVENTS_COLS = [
    "gene_id", "site_id", "group_a", "group_b",
    "pdui_a", "pdui_b", "delta_pdui", "p_value", "adj_p_value",
]

_PAS_SITES_COLS = ["site_id", "gene_id", "chrom", "start", "end", "strand"]

_CELL_LABELS_COLS = ["cell_id", "barcode_raw", "group", "cluster_id", "celltype_corrected"]

_CONFIG_REQUIRED_SECTIONS = ["project", "reference"]


def _check_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {missing}. "
            f"Found: {list(df.columns)}"
        )


def read_apa_events(path: str | Path) -> pd.DataFrame:
    """Read APA events TSV.

    Expected columns: gene_id, site_id, group_a, group_b, pdui_a, pdui_b,
    delta_pdui, p_value, adj_p_value
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"APA events file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    _check_columns(df, _APA_EVENTS_COLS, str(path))
    logger.info("Loaded %d APA events from %s", len(df), path)
    return df


def read_pas_sites(path: str | Path) -> pd.DataFrame:
    """Read PAS sites TSV.

    Expected columns: site_id, gene_id, chrom, start, end, strand
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PAS sites file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    _check_columns(df, _PAS_SITES_COLS, str(path))
    logger.info("Loaded %d PAS sites from %s", len(df), path)
    return df


def read_pdui_matrix(path: str | Path) -> pd.DataFrame:
    """Read PDUI matrix (sites × cells or sites × groups).

    Assumes first column is site_id index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDUI matrix file not found: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("Loaded PDUI matrix (%d sites × %d columns) from %s", *df.shape, path)
    return df


def read_cell_labels(path: str | Path) -> pd.DataFrame:
    """Read cell label table.

    Expected columns: cell_id, barcode_raw, group, cluster_id, celltype_corrected
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cell labels file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    _check_columns(df, _CELL_LABELS_COLS, str(path))
    logger.info("Loaded %d cell labels from %s", len(df), path)
    return df


def read_gtf(path: str | Path) -> "pd.DataFrame":
    """Read GTF file using pyranges and return as a DataFrame.

    Returns pyranges.PyRanges object (behaves as DataFrame).
    """
    import pyranges as pr  # type: ignore

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GTF file not found: {path}")
    gtf = pr.read_gtf(str(path), as_df=True)
    logger.info("Loaded GTF with %d records from %s", len(gtf), path)
    return gtf


def read_bed(path: str | Path) -> pd.DataFrame:
    """Read a BED file into a DataFrame with standard column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")
    col_names = ["chrom", "start", "end", "name", "score", "strand"]
    df = pd.read_csv(path, sep="\t", header=None, names=col_names[: None])
    # Trim to columns actually present
    df.columns = col_names[: len(df.columns)]
    logger.info("Loaded %d BED records from %s", len(df), path)
    return df


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and validate required sections.

    Returns the config dict.  Raises ValueError if required sections are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        config = yaml.safe_load(fh)
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)
    return config


def validate_config(config: dict) -> list[str]:
    """Validate config dict structure.

    Returns a list of warning/error strings.  Does *not* raise exceptions so
    that the CLI can surface all issues at once.
    """
    issues: list[str] = []
    if not isinstance(config, dict):
        issues.append("Config must be a YAML mapping at the top level.")
        return issues

    for section in _CONFIG_REQUIRED_SECTIONS:
        if section not in config:
            issues.append(f"Missing required config section: '{section}'")

    project = config.get("project", {})
    for key in ("name", "output_dir"):
        if key not in project:
            issues.append(f"Missing 'project.{key}' in config.")

    return issues
