"""Command-line interface for scAPAview."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli() -> None:
    """scAPAview – visualise alternative polyadenylation from single-cell data."""


@cli.command("validate-config")
@click.argument("config_path")
def validate_config_cmd(config_path: str) -> None:
    """Validate a YAML config file without opening any data files."""
    from .io import validate_config
    import yaml

    path = Path(config_path)
    if not path.exists():
        click.echo(f"ERROR: Config file not found: {path}", err=True)
        sys.exit(1)

    with path.open() as fh:
        config = yaml.safe_load(fh)

    issues = validate_config(config)
    if issues:
        for issue in issues:
            click.echo(f"WARNING: {issue}")
        click.echo(f"\n{len(issues)} issue(s) found.")
    else:
        click.echo(f"✓ Config '{path.name}' is valid.")


@cli.command("build-pas-table")
@click.argument("config_path")
def build_pas_table_cmd(config_path: str) -> None:
    """Build a unified PAS site table from sources listed in config."""
    from .io import load_config, read_pas_sites
    from .apa import build_unified_pas_table
    from .report import write_table

    config = load_config(config_path)
    output_dir = Path(config["project"]["output_dir"])
    window = config.get("analysis", {}).get("pas_match_window", 25)

    frames: dict = {}
    scpolaseq_cfg = config.get("scpolaseq", {})
    if "pas_reference" in scpolaseq_cfg:
        frames["scpolaseq"] = read_pas_sites(scpolaseq_cfg["pas_reference"])

    sierra_cfg = config.get("sierra", {})
    if "quant_dir" in sierra_cfg:
        click.echo("Sierra quant directory detected; site loading not yet implemented.")

    scapture_cfg = config.get("scapture", {})
    if "site_catalog" in scapture_cfg:
        frames["scapture"] = read_pas_sites(scapture_cfg["site_catalog"])

    unified = build_unified_pas_table(
        scpolaseq_sites=frames.get("scpolaseq"),
        sierra_sites=frames.get("sierra"),
        scapture_sites=frames.get("scapture"),
        window=window,
    )
    out = write_table(unified, output_dir, "unified_pas_sites.tsv")
    click.echo(f"Unified PAS table written to {out}")


@cli.command("annotate-pas")
@click.argument("config_path")
def annotate_pas_cmd(config_path: str) -> None:
    """Annotate PAS sites with genomic context using GTF annotation."""
    from .io import load_config, read_gtf
    from .annotation import standardize_gtf, annotate_pas_context
    from .report import write_table
    import pandas as pd

    config = load_config(config_path)
    output_dir = Path(config["project"]["output_dir"])
    gtf_path = config.get("reference", {}).get("gtf")
    if not gtf_path:
        click.echo("ERROR: 'reference.gtf' not set in config.", err=True)
        sys.exit(1)

    gtf = read_gtf(gtf_path)
    gtf = standardize_gtf(gtf)

    pas_path = output_dir / "unified_pas_sites.tsv"
    if not pas_path.exists():
        click.echo(f"ERROR: {pas_path} not found. Run 'build-pas-table' first.", err=True)
        sys.exit(1)

    pas = pd.read_csv(pas_path, sep="\t")
    annotated = annotate_pas_context(pas, gtf)
    out = write_table(annotated, output_dir, "annotated_pas_sites.tsv")
    click.echo(f"Annotated PAS table written to {out}")


@cli.command("plot-gene")
@click.argument("config_path")
@click.option("--gene", required=True, help="Gene symbol or gene_id to plot.")
@click.option("--group-a", default=None, help="Group A label.")
@click.option("--group-b", default=None, help="Group B label.")
@click.option("--celltype", default=None, help="Cell type to select bigWig tracks for.")
def plot_gene_cmd(
    config_path: str,
    gene: str,
    group_a: str | None,
    group_b: str | None,
    celltype: str | None,
) -> None:
    """Generate a gene-level APA track plot."""
    from .io import load_config, read_gtf, read_pas_sites, read_apa_events
    from .annotation import standardize_gtf
    from .tracks import plot_gene_apa_tracks
    import pandas as pd

    config = load_config(config_path)
    output_dir = Path(config["project"]["output_dir"])

    gtf = standardize_gtf(read_gtf(config["reference"]["gtf"]))

    pas_path = output_dir / "annotated_pas_sites.tsv"
    if not pas_path.exists():
        pas_path = output_dir / "unified_pas_sites.tsv"
    pas = pd.read_csv(pas_path, sep="\t") if pas_path.exists() else pd.DataFrame()

    apa_path = Path(config.get("scpolaseq", {}).get("apa_events", ""))
    apa = read_apa_events(apa_path) if apa_path and apa_path.exists() else None

    # Build bigwig_tracks dict
    bw_cfg = config.get("bigwig_tracks", {})
    bw_tracks: dict = {}
    for group, tracks in bw_cfg.items():
        key = celltype if celltype and celltype in tracks else "all"
        bw_path = tracks.get(key)
        if bw_path:
            bw_tracks[f"{group} [{key}]"] = bw_path

    ga = group_a or config.get("analysis", {}).get("group_a")
    gb = group_b or config.get("analysis", {}).get("group_b")

    out_path = output_dir / "plots" / f"{gene}_apa_tracks.png"
    plot_gene_apa_tracks(
        gene_name=gene,
        gtf=gtf,
        pas_sites=pas,
        apa_events=apa,
        bigwig_tracks=bw_tracks or None,
        group_a=ga,
        group_b=gb,
        celltype=celltype,
        output=out_path,
        show=False,
    )
    click.echo(f"Plot saved to {out_path}")


@cli.command("plot-metagene")
@click.argument("config_path")
@click.option("--gene-set", required=True, help="Gene set name (from DEFAULT_GENE_SETS or config).")
@click.option(
    "--region",
    required=True,
    type=click.Choice(["3utr", "gene_body", "splice_sites"]),
    help="Genomic region to use for metagene.",
)
def plot_metagene_cmd(config_path: str, gene_set: str, region: str) -> None:
    """Generate a metagene coverage plot for a gene set."""
    from .io import load_config, read_gtf
    from .annotation import standardize_gtf, build_exon_table
    from .genesets import DEFAULT_GENE_SETS
    from .metagene import plot_metagene_3utr, plot_metagene_gene_body, plot_metagene_splice_sites
    import pandas as pd

    config = load_config(config_path)
    output_dir = Path(config["project"]["output_dir"])

    genes = DEFAULT_GENE_SETS.get(gene_set)
    if not genes:
        click.echo(f"ERROR: Gene set '{gene_set}' not found in DEFAULT_GENE_SETS.", err=True)
        sys.exit(1)

    gtf = standardize_gtf(read_gtf(config["reference"]["gtf"]))
    gene_name_col = "gene_name" if "gene_name" in gtf.columns else None
    if gene_name_col:
        regions = gtf[gtf[gene_name_col].isin(genes)]
    else:
        regions = gtf[gtf.get("gene_id", pd.Series()).isin(genes)]

    bw_cfg = config.get("bigwig_tracks", {})
    ga = config.get("analysis", {}).get("group_a")
    gb = config.get("analysis", {}).get("group_b")
    bw_a = [v.get("all") for k, v in bw_cfg.items() if k == ga and v.get("all")]
    bw_b = [v.get("all") for k, v in bw_cfg.items() if k == gb and v.get("all")]

    out_path = output_dir / "plots" / f"metagene_{gene_set}_{region}.png"
    fn_map = {
        "3utr": plot_metagene_3utr,
        "gene_body": plot_metagene_gene_body,
        "splice_sites": plot_metagene_splice_sites,
    }
    fn_map[region](bw_a, bw_b or None, regions, output=out_path, show=False)
    click.echo(f"Metagene plot saved to {out_path}")


@cli.command("summarize-burden")
@click.argument("config_path")
def summarize_burden_cmd(config_path: str) -> None:
    """Summarize APA burden per gene set."""
    from .io import load_config, read_apa_events
    from .genesets import DEFAULT_GENE_SETS, summarize_gene_set_apa_burden
    from .report import write_table
    import pandas as pd

    config = load_config(config_path)
    output_dir = Path(config["project"]["output_dir"])

    pas_path = output_dir / "unified_pas_sites.tsv"
    pas = pd.read_csv(pas_path, sep="\t") if pas_path.exists() else pd.DataFrame()

    apa_path = Path(config.get("scpolaseq", {}).get("apa_events", ""))
    apa = read_apa_events(apa_path) if apa_path and apa_path.exists() else pd.DataFrame()

    burden = summarize_gene_set_apa_burden(DEFAULT_GENE_SETS, pas, apa)
    out = write_table(burden, output_dir, "gene_set_apa_burden.tsv")
    click.echo(f"Gene set burden table written to {out}")
