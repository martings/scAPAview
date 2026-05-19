"""Input/output functions for reading scAPAview file types."""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_APA_EVENTS_COLS = [
    "gene_id", "site_id", "group_a", "group_b",
    "pdui_a", "pdui_b", "delta_pdui", "p_value", "adj_p_value",
]

_PAS_SITES_COLS = ["site_id", "gene_id", "chrom", "start", "end", "strand"]
_CONFIG_REQUIRED_SECTIONS = ["project", "reference"]

_SITE_RE = re.compile(
    r"^(?P<gene>[^:]+):(?P<chrom>[^:]+):(?P<start>\d+)(?:-(?P<end>\d+))?:(?P<strand>[+-])$"
)


def _check_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {missing}. Found: {list(df.columns)}"
        )


def gene_id_base(value: object) -> str:
    """Return an Ensembl-style gene id without a version suffix."""
    if pd.isna(value):
        return ""
    return str(value).split(".", 1)[0]


def normalize_chromosome(chrom: object, add_chr: bool = True) -> str:
    """Normalize chromosome names to the common ``chr`` style used by GTF/bigWig."""
    value = str(chrom)
    if add_chr and value and not value.startswith("chr"):
        return f"chr{value}"
    return value


def parse_site_id(site_id: object) -> dict[str, object]:
    """Parse scPolASeq/Sierra-style site ids into coordinate fields when possible."""
    sid = str(site_id)
    match = _SITE_RE.match(sid)
    if match is None:
        return {}
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    return {
        "gene_id": match.group("gene"),
        "gene_id_base": gene_id_base(match.group("gene")),
        "chrom": normalize_chromosome(match.group("chrom")),
        "start": min(start, end),
        "end": max(start, end),
        "strand": match.group("strand"),
    }


def add_gene_id_columns(df: pd.DataFrame, gene_col: str = "gene_id") -> pd.DataFrame:
    """Add version-aware gene id helper columns if a gene id column is present."""
    out = df.copy()
    if gene_col in out.columns:
        out[gene_col] = out[gene_col].astype(str)
        out["gene_id_base"] = out[gene_col].map(gene_id_base)
    return out


def _coerce_pas_schema(df: pd.DataFrame, source: str, source_label: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "site_id" not in out.columns and "pas_reference_id" in out.columns:
        out = out.rename(columns={"pas_reference_id": "site_id"})

    if not set(_PAS_SITES_COLS).issubset(out.columns) and "site_id" in out.columns:
        parsed = pd.DataFrame([parse_site_id(sid) for sid in out["site_id"]])
        for col in _PAS_SITES_COLS:
            if col not in out.columns and col in parsed.columns:
                out[col] = parsed[col]
        if "gene_id_base" not in out.columns and "gene_id_base" in parsed.columns:
            out["gene_id_base"] = parsed["gene_id_base"]

    _check_columns(out, _PAS_SITES_COLS, source)
    out = add_gene_id_columns(out)
    out["chrom"] = out["chrom"].map(normalize_chromosome)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").astype("Int64")
    out["end"] = pd.to_numeric(out["end"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["start", "end"]).copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["source"] = source_label or source
    return out


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tab-separated table with consistent path validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")
    return pd.read_csv(path, sep="	", **kwargs)


def read_apa_events(path: str | Path) -> pd.DataFrame:
    """Read APA events TSV and add gene id helper columns."""
    df = read_tsv(path)
    _check_columns(df, _APA_EVENTS_COLS, str(path))
    df = add_gene_id_columns(df)
    logger.info("Loaded %d APA events from %s", len(df), path)
    return df


def read_pas_sites(path: str | Path, source: str = "scpolaseq") -> pd.DataFrame:
    """Read PAS sites from site catalog or PAS reference TSV."""
    df = _coerce_pas_schema(read_tsv(path), str(path), source_label=source)
    logger.info("Loaded %d PAS sites from %s", len(df), path)
    return df


def read_pdui_matrix(path: str | Path) -> pd.DataFrame:
    """Read PDUI matrix with ``gene_id``/``site_id`` retained when present."""
    df = read_tsv(path)
    if "gene_id" in df.columns:
        df = add_gene_id_columns(df)
    logger.info("Loaded PDUI matrix-like table %s from %s", df.shape, path)
    return df


def read_cell_labels(path: str | Path) -> pd.DataFrame:
    """Read cell labels from current or older scPolASeq label schemas."""
    df = read_tsv(path)
    if "cell_id" not in df.columns and {"sample_id", "library_id", "barcode_corrected"}.issubset(df.columns):
        df["cell_id"] = (
            df["sample_id"].astype(str) + ":" + df["library_id"].astype(str) + ":" + df["barcode_corrected"].astype(str)
        )
    if "group" not in df.columns:
        if "sample_id" in df.columns:
            df["group"] = df["sample_id"]
        elif "library_id" in df.columns:
            df["group"] = df["library_id"]
    if "celltype_corrected" not in df.columns:
        if "cell_type" in df.columns:
            df["celltype_corrected"] = df["cell_type"]
        else:
            df["celltype_corrected"] = "unknown"
    required = ["cell_id", "barcode_raw", "group", "cluster_id", "celltype_corrected"]
    _check_columns(df, required, str(path))
    logger.info("Loaded %d cell labels from %s", len(df), path)
    return df


def read_gtf(path: str | Path) -> pd.DataFrame:
    """Read a GTF file using pyranges and return a DataFrame."""
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
    df = pd.read_csv(path, sep="	", header=None)
    df.columns = col_names[: len(df.columns)]
    logger.info("Loaded %d BED records from %s", len(df), path)
    return df


def read_gene_info(path: str | Path) -> pd.DataFrame:
    """Read STAR ``geneInfo.tab`` or a generic gene info TSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gene info file not found: {path}")
    df = pd.read_csv(path, sep="	", header=None)
    if df.shape[1] >= 5:
        cols = ["gene_id", "chrom", "start", "end", "strand"] + [f"extra_{i}" for i in range(df.shape[1] - 5)]
        df.columns = cols[: df.shape[1]]
        df = add_gene_id_columns(df)
    return df


def load_terminal_exons(path: str | Path) -> pd.DataFrame:
    """Load pipeline terminal exon regions and normalize coordinates."""
    df = _coerce_pas_schema(read_tsv(path).rename(columns={"site_id": "_site_id"}), str(path), source_label="terminal_exon") if False else read_tsv(path)
    _check_columns(df, ["gene_id", "chrom", "start", "end", "strand"], str(path))
    df = add_gene_id_columns(df)
    df["chrom"] = df["chrom"].map(normalize_chromosome)
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype(int)
    return df


def load_sierra_quant(path_or_dir: str | Path, max_files: int | None = None) -> pd.DataFrame:
    """Load and aggregate Sierra Quant long UMI tables into PAS support rows."""
    path = Path(path_or_dir)
    if path.is_dir():
        files = sorted(path.glob("*.sierra_quant.tsv"))
    else:
        files = [path]
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No Sierra Quant TSV files found at {path}")

    chunks = []
    for file_path in files:
        df = read_tsv(file_path)
        _check_columns(df, ["library_id", "group_level", "group_id", "site_id", "cell_barcode", "umi_count"], str(file_path))
        df["_sierra_file"] = file_path.name
        chunks.append(df)
    raw = pd.concat(chunks, ignore_index=True)
    raw["umi_count"] = pd.to_numeric(raw["umi_count"], errors="coerce").fillna(0)
    raw["sierra_group"] = raw["library_id"].astype(str) + "." + raw["group_id"].astype(str)

    grouped = raw.groupby("site_id", observed=True).agg(
        sierra_umi_count=("umi_count", "sum"),
        sierra_n_cells=("cell_barcode", "nunique"),
        sierra_groups=("sierra_group", lambda s: ",".join(sorted(set(map(str, s))))),
    ).reset_index()
    parsed = pd.DataFrame([parse_site_id(sid) for sid in grouped["site_id"]])
    out = pd.concat([grouped, parsed], axis=1)
    out = _coerce_pas_schema(out, str(path), source_label="sierra")
    return out


def load_scapture_sites(path_or_dir: str | Path, gene_name_map: str | Path | None = None) -> pd.DataFrame:
    """Load one or more scapture site catalogs."""
    path = Path(path_or_dir)
    files = sorted(path.glob("*.scapture.site_catalog.tsv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No scapture site catalogs found at {path}")
    frames = []
    for file_path in files:
        df = _coerce_pas_schema(read_tsv(file_path), str(file_path), source_label="scapture")
        df["scapture_sample"] = file_path.name.split(".scapture.site_catalog.tsv")[0]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["site_id"])
    if gene_name_map is not None:
        gmap = load_scapture_gene_name_map(gene_name_map)
        out = out.merge(gmap, on="gene_id", how="left")
    return out


def load_scapture_gene_name_map(path_or_dir: str | Path) -> pd.DataFrame:
    """Load scapture gene-name maps."""
    path = Path(path_or_dir)
    files = sorted(path.glob("*.scapture.gene_name_map.tsv")) if path.is_dir() else [path]
    frames = [read_tsv(file_path) for file_path in files]
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    if "gene_id" in out.columns:
        out = add_gene_id_columns(out)
    return out


def load_scapture_umis(path: str | Path, max_cells: int | None = None) -> pd.DataFrame:
    """Load scapture UMI matrix/table, optionally limiting cell columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"scapture UMI file not found: {path}")
    df = pd.read_csv(path, sep="	", index_col=0)
    if max_cells is not None:
        df = df.iloc[:, :max_cells]
    return df


def read_bigwig_paths(config: dict) -> dict[str, dict[str, str]]:
    """Return configured bigWig track paths as a nested dictionary."""
    return {str(k): dict(v) for k, v in config.get("bigwig_tracks", {}).items()}


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and log validation warnings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        config = yaml.safe_load(fh)
    for warning in validate_config(config):
        logger.warning(warning)
    return config


def validate_config(config: dict) -> list[str]:
    """Validate config dict structure without opening data files."""
    issues: list[str] = []
    if not isinstance(config, dict):
        return ["Config must be a YAML mapping at the top level."]
    for section in _CONFIG_REQUIRED_SECTIONS:
        if section not in config:
            issues.append(f"Missing required config section: '{section}'")
    project = config.get("project", {})
    for key in ("name", "output_dir"):
        if key not in project:
            issues.append(f"Missing 'project.{key}' in config.")
    return issues


load_pdui_matrix = read_pdui_matrix
load_apa_events = read_apa_events
load_cell_labels = read_cell_labels
