"""Report generation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_table(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    """Write a DataFrame to a TSV file in *output_dir*.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return out_path


def generate_summary_report(
    unified_pas: pd.DataFrame,
    apa_events: pd.DataFrame,
    gene_set_burden: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Generate a plain-text summary report of APA analysis results.

    Writes individual TSV tables and a human-readable summary text file.

    Returns the path to the summary text file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write individual tables
    write_table(unified_pas, output_dir, "unified_pas_sites.tsv")
    write_table(apa_events, output_dir, "apa_events_classified.tsv")
    write_table(gene_set_burden, output_dir, "gene_set_apa_burden.tsv")

    # Compute summary statistics
    n_pas = len(unified_pas)
    n_genes_with_pas = unified_pas["gene_id"].nunique() if "gene_id" in unified_pas.columns else "N/A"
    n_events = len(apa_events)

    sig_cols = [c for c in ("is_fdr_and_delta", "is_fdr_significant") if c in apa_events.columns]
    if sig_cols:
        n_sig = int(apa_events[sig_cols[0]].sum())
    elif "adj_p_value" in apa_events.columns:
        n_sig = int((apa_events["adj_p_value"] < 0.05).sum())
    else:
        n_sig = "N/A"

    summary_lines = [
        "scAPAview Analysis Summary",
        "=" * 40,
        f"Unified PAS sites:              {n_pas}",
        f"Genes with PAS sites:           {n_genes_with_pas}",
        f"Total APA events:               {n_events}",
        f"Significant APA events:         {n_sig}",
        "",
        "Gene set APA burden:",
    ]
    if not gene_set_burden.empty:
        for _, row in gene_set_burden.iterrows():
            summary_lines.append(
                f"  {row.get('gene_set', 'N/A'):40s} "
                f"n_sig={row.get('n_significant_apa_events', 0)}"
            )

    summary_path = output_dir / "summary_report.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    logger.info("Summary report written to %s", summary_path)
    return summary_path
