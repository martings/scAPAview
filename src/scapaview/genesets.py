"""Built-in gene sets and gene-set utilities for scAPAview."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


DEFAULT_GENE_SETS: dict[str, list[str]] = {
    "Dengue_ISG": [
        "CXCL10", "MX1", "ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3",
        "OAS1", "OAS2", "DDX58", "IFIH1", "IRF7", "STAT1", "STAT2",
    ],
    "PAF1_NS5_axis": [
        "PAF1", "LEO1", "CTR9", "CDC73", "RELB", "TRIM22", "HLA-F", "BHLHE40",
    ],
    "Antigen_presentation": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1",
        "B2M", "TAP1", "TAP2",
    ],
    "Monocyte_activation": [
        "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "LST1",
        "FCGR3A", "MS4A7", "CTSS",
    ],
    "Cytotoxic_program": [
        "NKG7", "GNLY", "GZMB", "PRF1", "CCL5", "GZMK", "KLRD1", "KLRF1",
    ],
    "RNA_processing_splicing": [
        "HNRNPA1", "HNRNPC", "HNRNPH1", "HNRNPH3", "SRSF1",
        "SRSF3", "U2AF1", "U2AF2", "SF3B1",
    ],
}


def load_gene_sets_yaml(path: str | Path) -> dict[str, list[str]]:
    """Load gene sets from a YAML file.

    Expected format::

        gene_sets:
          SetName1:
            - GENE_A
            - GENE_B
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gene sets YAML not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if "gene_sets" in data:
        return data["gene_sets"]
    return data


def load_gene_sets_gmt(path: str | Path) -> dict[str, list[str]]:
    """Load gene sets from a GMT (Gene Matrix Transposed) file.

    GMT format: each row is ``set_name \\t description \\t gene1 \\t gene2 ...``
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {path}")
    gene_sets: dict[str, list[str]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            set_name = parts[0]
            genes = [g for g in parts[2:] if g]
            gene_sets[set_name] = genes
    return gene_sets


def filter_gene_sets_to_annotation(
    gene_sets: dict[str, list[str]],
    gene_table: pd.DataFrame,
) -> dict[str, list[str]]:
    """Filter gene sets to only include genes present in the annotation.

    Parameters
    ----------
    gene_sets  : dict mapping set name → list of gene symbols
    gene_table : DataFrame with a ``gene_id`` or ``gene_name`` column
    """
    name_col = None
    for c in ("gene_name", "gene_id", "GeneID"):
        if c in gene_table.columns:
            name_col = c
            break
    if name_col is None:
        logger.warning("No gene name column found in gene_table; returning unfiltered sets.")
        return gene_sets

    annotated_genes = set(gene_table[name_col].dropna().unique())
    filtered: dict[str, list[str]] = {}
    for name, genes in gene_sets.items():
        kept = [g for g in genes if g in annotated_genes]
        if kept:
            filtered[name] = kept
        else:
            logger.warning("Gene set '%s' has no overlap with annotation; skipping.", name)
    return filtered


def _gene_symbol_to_ids(gene_table: pd.DataFrame | None) -> dict[str, set[str]]:
    """Build a gene-symbol to base-Ensembl-id map from annotation."""
    if gene_table is None or gene_table.empty or "gene_name" not in gene_table.columns:
        return {}
    table = gene_table.copy()
    if "gene_id_base" not in table.columns and "gene_id" in table.columns:
        table["gene_id_base"] = table["gene_id"].astype(str).str.split(".", n=1).str[0]
    mapping: dict[str, set[str]] = {}
    for _, row in table.dropna(subset=["gene_name", "gene_id_base"]).iterrows():
        mapping.setdefault(str(row["gene_name"]), set()).add(str(row["gene_id_base"]))
    return mapping


def _gene_targets(genes: list[str], symbol_map: dict[str, set[str]]) -> set[str]:
    targets = set(genes)
    for gene in genes:
        targets.update(symbol_map.get(gene, set()))
    targets.update(g.split(".", 1)[0] for g in list(targets))
    return targets


def summarize_gene_set_apa_burden(
    gene_sets: dict[str, list[str]],
    pas_sites: pd.DataFrame,
    apa_events: pd.DataFrame,
    gene_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize APA burden per gene set using gene symbols or Ensembl ids."""
    pas = pas_sites.copy() if pas_sites is not None else pd.DataFrame()
    apa = apa_events.copy() if apa_events is not None else pd.DataFrame()
    for df in (pas, apa):
        if not df.empty and "gene_id_base" not in df.columns and "gene_id" in df.columns:
            df["gene_id_base"] = df["gene_id"].astype(str).str.split(".", n=1).str[0]

    symbol_map = _gene_symbol_to_ids(gene_table)
    pas_gene_col = "gene_id_base" if "gene_id_base" in pas.columns else "gene_id" if "gene_id" in pas.columns else None
    apa_gene_col = "gene_id_base" if "gene_id_base" in apa.columns else "gene_id" if "gene_id" in apa.columns else None

    sig_col = next((c for c in ("is_fdr_and_delta", "is_fdr_significant", "adj_p_value") if c in apa.columns), None)
    rows = []
    for set_name, genes in gene_sets.items():
        targets = _gene_targets(genes, symbol_map)
        pas_sub = pas[pas[pas_gene_col].astype(str).isin(targets)] if pas_gene_col else pd.DataFrame()
        apa_sub = apa[apa[apa_gene_col].astype(str).isin(targets)] if apa_gene_col else pd.DataFrame()
        if sig_col == "adj_p_value":
            n_sig = int((pd.to_numeric(apa_sub[sig_col], errors="coerce") < 0.05).sum())
        elif sig_col is not None:
            n_sig = int(apa_sub[sig_col].fillna(False).astype(bool).sum())
        else:
            n_sig = 0
        rows.append(
            {
                "gene_set": set_name,
                "n_genes": len(genes),
                "n_gene_ids_matched": len(targets - set(genes)),
                "n_genes_with_pas": int(pas_sub[pas_gene_col].nunique()) if pas_gene_col and not pas_sub.empty else 0,
                "n_pas_sites": len(pas_sub),
                "n_apa_events": len(apa_sub),
                "n_significant_apa_events": n_sig,
                "n_fdr_and_delta_events": int(apa_sub.get("is_fdr_and_delta", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not apa_sub.empty else 0,
            }
        )
    return pd.DataFrame(rows)
