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
    """scAPAview: visualize alternative polyadenylation from single-cell data."""


def _output_dir(config: dict) -> Path:
    out = Path(config["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def _load_gtf_gene_table(config: dict):
    from .annotation import build_gene_table, standardize_gtf
    from .io import read_gtf

    gtf = standardize_gtf(read_gtf(config["reference"]["gtf"]))
    return gtf, build_gene_table(gtf)


def _configured_bigwigs(config: dict, group_a: str | None = None, group_b: str | None = None) -> dict:
    bw_cfg = config.get("bigwig_tracks", {})
    ga = group_a or config.get("analysis", {}).get("group_a")
    gb = group_b or config.get("analysis", {}).get("group_b")
    selected = {}
    for group in (ga, gb):
        if group and group in bw_cfg:
            selected[group] = bw_cfg[group]
    return selected or bw_cfg


def _all_group_bw_paths(config: dict, group: str | None, stranded: bool = False) -> list[str]:
    if not group:
        return []
    tracks = config.get("bigwig_tracks", {}).get(group, {})
    if stranded:
        return [p for p in (tracks.get("fwd"), tracks.get("rev")) if p]
    return [tracks.get("all") or tracks.get("fwd") or tracks.get("rev")] if tracks else []


def _read_pas_for_plot(output_dir: Path):
    import pandas as pd

    for name in ("pas_site_context.tsv", "annotated_pas_sites.tsv", "unified_pas_sites.tsv"):
        path = output_dir / "tables" / name
        if path.exists():
            return pd.read_csv(path, sep="	")
        path = output_dir / name
        if path.exists():
            return pd.read_csv(path, sep="	")
    return pd.DataFrame()


def _read_classified_events(config: dict, output_dir: Path):
    import pandas as pd
    from .apa import classify_apa_events
    from .io import read_apa_events

    path = output_dir / "tables" / "apa_events_classified.tsv"
    if path.exists():
        return pd.read_csv(path, sep="	")
    raw = read_apa_events(config.get("scpolaseq", {}).get("apa_events"))
    return classify_apa_events(
        raw,
        fdr_cutoff=config.get("analysis", {}).get("fdr_cutoff", 0.05),
        delta_cutoff=config.get("analysis", {}).get("delta_pdui_cutoff", 0.15),
    )


@cli.command("validate-config")
@click.argument("config_path")
def validate_config_cmd(config_path: str) -> None:
    """Validate a YAML config file."""
    from .io import validate_config
    import yaml

    path = Path(config_path)
    if not path.exists():
        click.echo(f"ERROR: Config file not found: {path}", err=True)
        sys.exit(1)
    config = yaml.safe_load(path.read_text())
    issues = validate_config(config)
    if issues:
        for issue in issues:
            click.echo(f"WARNING: {issue}")
        click.echo(f"\n{len(issues)} issue(s) found.")
    else:
        click.echo(f"Config '{path.name}' is valid.")


@cli.command("build-pas-table")
@click.argument("config_path")
@click.option("--no-sierra", is_flag=True, help="Skip Sierra Quant aggregation.")
def build_pas_table_cmd(config_path: str, no_sierra: bool = False) -> None:
    """Build a unified PAS site table from configured sources."""
    import pandas as pd
    from .apa import build_unified_pas_table
    from .io import load_config, load_scapture_sites, load_sierra_quant, read_pas_sites
    from .report import write_table

    config = load_config(config_path)
    output_dir = _output_dir(config)
    window = config.get("analysis", {}).get("pas_match_window", 25)
    scp_cfg = config.get("scpolaseq", {})
    frames = []
    if scp_cfg.get("site_catalog"):
        frames.append(read_pas_sites(scp_cfg["site_catalog"], source="scpolaseq"))
    if scp_cfg.get("pas_reference"):
        frames.append(read_pas_sites(scp_cfg["pas_reference"], source="pas_reference"))
    scpolaseq_sites = pd.concat(frames, ignore_index=True, sort=False) if frames else None

    sierra_sites = None
    if config.get("sierra", {}).get("quant_dir") and not no_sierra:
        sierra_sites = load_sierra_quant(config["sierra"]["quant_dir"])

    scapture_sites = None
    scapture_cfg = config.get("scapture", {})
    if scapture_cfg.get("site_catalog"):
        gene_map = scapture_cfg.get("gene_name_map")
        scapture_sites = load_scapture_sites(scapture_cfg["site_catalog"], gene_name_map=gene_map)

    unified = build_unified_pas_table(
        scpolaseq_sites=scpolaseq_sites,
        sierra_sites=sierra_sites,
        scapture_sites=scapture_sites,
        window=window,
    )
    out = write_table(unified, output_dir / "tables", "unified_pas_sites.tsv")
    click.echo(f"Unified PAS table written to {out} ({len(unified):,} rows)")


@cli.command("annotate-pas")
@click.argument("config_path")
def annotate_pas_cmd(config_path: str) -> None:
    """Annotate PAS sites with terminal-exon context and splice proximity."""
    import pandas as pd
    import pyranges as pr
    from .annotation import build_exon_table, derive_splice_sites, distance_to_nearest_splice_site, standardize_gtf
    from .io import load_config, load_terminal_exons, read_gtf
    from .report import write_table

    config = load_config(config_path)
    output_dir = _output_dir(config)
    pas_path = output_dir / "tables" / "unified_pas_sites.tsv"
    if not pas_path.exists():
        click.echo("Unified PAS table not found; running build-pas-table first.")
        build_pas_table_cmd.callback(config_path, no_sierra=False)
    pas = pd.read_csv(pas_path, sep="	")
    pas["pas_context"] = "intergenic"
    pas["_pas_index"] = range(len(pas))

    terminal_path = config.get("reference", {}).get("terminal_exons")
    if terminal_path:
        terminal = load_terminal_exons(terminal_path)
        pas_pr = pr.PyRanges(
            pas.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})[
                ["Chromosome", "Start", "End", "Strand", "gene_id_base", "_pas_index"]
            ]
        )
        term_pr = pr.PyRanges(
            terminal.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End", "strand": "Strand"})[
                ["Chromosome", "Start", "End", "Strand", "gene_id_base"]
            ]
        )
        joined = pas_pr.join(term_pr, strandedness="same").df
        if not joined.empty:
            gene_match = joined["gene_id_base"] == joined["gene_id_base_b"] if "gene_id_base_b" in joined.columns else True
            hit_idx = joined.loc[gene_match, "_pas_index"].unique()
            pas.loc[hit_idx, "pas_context"] = "terminal_exon"

    gtf = standardize_gtf(read_gtf(config["reference"]["gtf"]))
    splice = derive_splice_sites(build_exon_table(gtf))
    annotated = distance_to_nearest_splice_site(
        pas.drop(columns=["_pas_index"]),
        splice,
        window=config.get("analysis", {}).get("splice_proximal_window", 100),
    )
    out = write_table(annotated, output_dir / "tables", "pas_site_context.tsv")
    click.echo(f"Annotated PAS table written to {out} ({len(annotated):,} rows)")


@cli.command("classify-events")
@click.argument("config_path")
def classify_events_cmd(config_path: str) -> None:
    """Classify APA events by FDR and delta-PDUI thresholds."""
    from .apa import classify_apa_events
    from .io import load_config, read_apa_events
    from .report import write_table

    config = load_config(config_path)
    output_dir = _output_dir(config)
    events = read_apa_events(config.get("scpolaseq", {}).get("apa_events"))
    classified = classify_apa_events(
        events,
        fdr_cutoff=config.get("analysis", {}).get("fdr_cutoff", 0.05),
        delta_cutoff=config.get("analysis", {}).get("delta_pdui_cutoff", 0.15),
    )
    out = write_table(classified, output_dir / "tables", "apa_events_classified.tsv")
    click.echo(f"Classified APA events written to {out} ({len(classified):,} rows)")


@cli.command("plot-gene")
@click.argument("config_path")
@click.option("--gene", required=True, help="Gene symbol or gene_id to plot.")
@click.option("--group-a", default=None, help="Group A label.")
@click.option("--group-b", default=None, help="Group B label.")
@click.option("--flank", default=1000, show_default=True, help="Flanking bp around the gene.")
def plot_gene_cmd(config_path: str, gene: str, group_a: str | None, group_b: str | None, flank: int) -> None:
    """Generate a gene-level APA track plot."""
    from .io import load_config
    from .tracks import plot_gene_apa_tracks

    config = load_config(config_path)
    output_dir = _output_dir(config)
    gtf, _ = _load_gtf_gene_table(config)
    pas = _read_pas_for_plot(output_dir)
    apa = _read_classified_events(config, output_dir)
    ga = group_a or config.get("analysis", {}).get("group_a")
    gb = group_b or config.get("analysis", {}).get("group_b")
    tracks = _configured_bigwigs(config, ga, gb)
    out = output_dir / "plots" / f"gene_{gene}_tracks.png"
    plot_gene_apa_tracks(gene, gtf, pas, apa, tracks, group_a=ga, group_b=gb, flank=flank, output=out, show=False)
    click.echo(f"Plot saved to {out}")


@cli.command("plot-metagene")
@click.argument("config_path")
@click.option("--gene-set", required=True, help="Built-in gene set name.")
@click.option("--region", default="3utr", type=click.Choice(["3utr", "gene_body"]), show_default=True)
def plot_metagene_cmd(config_path: str, gene_set: str, region: str) -> None:
    """Generate a metagene coverage plot for a gene set."""
    from .annotation import build_gene_table, terminal_regions_for_genes
    from .genesets import DEFAULT_GENE_SETS
    from .io import load_config, load_terminal_exons
    from .metagene import plot_metagene_3utr, plot_metagene_gene_body

    config = load_config(config_path)
    output_dir = _output_dir(config)
    genes = DEFAULT_GENE_SETS.get(gene_set)
    if not genes:
        click.echo(f"ERROR: Gene set '{gene_set}' not found.", err=True)
        sys.exit(1)
    gtf, gene_table = _load_gtf_gene_table(config)
    pas = _read_pas_for_plot(output_dir)
    ga = config.get("analysis", {}).get("group_a")
    gb = config.get("analysis", {}).get("group_b")
    bw_a = _all_group_bw_paths(config, ga)
    bw_b = _all_group_bw_paths(config, gb)
    if region == "3utr":
        terminal = load_terminal_exons(config["reference"]["terminal_exons"])
        regions = terminal_regions_for_genes(terminal, genes, gene_table)
        out = output_dir / "plots" / f"metagene_{gene_set}_3utr.png"
        plot_metagene_3utr(bw_a, bw_b, regions, pas_sites=pas, output=out, show=False)
    else:
        name_col = "gene_name" if "gene_name" in gene_table.columns else "gene_id"
        regions = gene_table[gene_table[name_col].isin(genes)]
        out = output_dir / "plots" / f"metagene_{gene_set}_gene_body.png"
        plot_metagene_gene_body(bw_a, bw_b, regions, output=out, show=False)
    click.echo(f"Metagene plot saved to {out} ({len(regions):,} regions)")


@cli.command("summarize-burden")
@click.argument("config_path")
def summarize_burden_cmd(config_path: str) -> None:
    """Summarize APA burden per built-in gene set."""
    from .genesets import DEFAULT_GENE_SETS, summarize_gene_set_apa_burden
    from .io import load_config
    from .report import write_table

    config = load_config(config_path)
    output_dir = _output_dir(config)
    _, gene_table = _load_gtf_gene_table(config)
    pas = _read_pas_for_plot(output_dir)
    apa = _read_classified_events(config, output_dir)
    burden = summarize_gene_set_apa_burden(DEFAULT_GENE_SETS, pas, apa, gene_table=gene_table)
    out = write_table(burden, output_dir / "tables", "gene_set_apa_burden.tsv")
    click.echo(f"Gene set burden table written to {out}")


@cli.command("demo-dengue")
@click.argument("config_path")
def demo_dengue_cmd(config_path: str) -> None:
    """Run the dengue MVP workflow and generate first-review immune figures."""
    from .annotation import terminal_regions_for_genes
    from .genesets import DEFAULT_GENE_SETS
    from .io import load_config, load_terminal_exons
    from .metagene import plot_metagene_3utr
    from .tracks import plot_gene_apa_tracks

    config = load_config(config_path)
    output_dir = _output_dir(config)
    if not (output_dir / "tables" / "unified_pas_sites.tsv").exists():
        build_pas_table_cmd.callback(config_path, no_sierra=False)
    if not (output_dir / "tables" / "pas_site_context.tsv").exists():
        annotate_pas_cmd.callback(config_path)
    if not (output_dir / "tables" / "apa_events_classified.tsv").exists():
        classify_events_cmd.callback(config_path)
    if not (output_dir / "tables" / "gene_set_apa_burden.tsv").exists():
        summarize_burden_cmd.callback(config_path)

    gtf, gene_table = _load_gtf_gene_table(config)
    pas = _read_pas_for_plot(output_dir)
    apa = _read_classified_events(config, output_dir)
    tracks = _configured_bigwigs(config)
    ga = config.get("analysis", {}).get("group_a")
    gb = config.get("analysis", {}).get("group_b")

    for gene in ["CXCL10", "ISG15", "STAT1", "STAT2", "PAF1", "CTR9"]:
        out = output_dir / "plots" / f"gene_{gene}_tracks.png"
        plot_gene_apa_tracks(gene, gtf, pas, apa, tracks, group_a=ga, group_b=gb, output=out, show=False)
        click.echo(f"Plot saved to {out}")

    terminal = load_terminal_exons(config["reference"]["terminal_exons"])
    bw_a = _all_group_bw_paths(config, ga)
    bw_b = _all_group_bw_paths(config, gb)
    for gene_set in ["Dengue_ISG", "PAF1_NS5_axis"]:
        genes = DEFAULT_GENE_SETS[gene_set]
        regions = terminal_regions_for_genes(terminal, genes, gene_table)
        out = output_dir / "plots" / f"metagene_{gene_set}_3utr.png"
        plot_metagene_3utr(bw_a, bw_b, regions, pas_sites=pas, output=out, show=False)
        click.echo(f"Metagene plot saved to {out} ({len(regions):,} regions)")

    _write_figure_index(output_dir)
    click.echo(f"Dengue demo complete. See {output_dir / 'plots'}")


def _write_figure_index(output_dir: Path) -> None:
    plots = sorted((output_dir / "plots").glob("*.png"))
    lines = ["# scAPAview Figure Index", ""]
    for plot in plots:
        rel = plot.relative_to(output_dir)
        lines.append(f"- [{plot.name}]({rel.as_posix()})")
    (output_dir / "figure_index.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    cli()
