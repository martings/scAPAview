# scAPAview

**Visualising alternative polyadenylation (APA) from single-cell data**

scAPAview is a Python toolkit for integrating, annotating and visualising polyadenylation site (PAS) usage from scPolASeq, Sierra Quant and scapture outputs.

## What is scAPAview?

Alternative polyadenylation (APA) allows a single gene to produce mRNAs with different 3' UTR lengths, affecting mRNA stability, translation and subcellular localisation. scAPAview provides:

- Unified PAS site tables from multiple detection tools
- APA event classification (FDR + ΔPDUI)
- Gene-level track plots (coverage, gene model, PAS sites, ΔPDUI lollipops)
- Metagene plots across 3' UTRs, gene bodies and splice sites
- Sashimi-style junction plots
- Built-in gene sets for dengue/viral infection and immune biology
- CLI for automated analysis pipelines

## Installation

```bash
git clone https://github.com/martings/scAPAview.git
cd scAPAview
pip install -e ".[dev]"

# With bigWig support
pip install -e ".[dev,bigwig]"
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate scapaview
```

## Quickstart

```python
from scapaview.io import read_apa_events, read_pas_sites
from scapaview.apa import classify_apa_events
from scapaview.tracks import plot_gene_apa_tracks

pas = read_pas_sites("unified_pas_sites.tsv")
events = read_apa_events("apa_events.tsv")
classified = classify_apa_events(events, fdr_cutoff=0.05, delta_cutoff=0.15)

fig, axes = plot_gene_apa_tracks(
    gene_name="MX1", gtf=gtf_df,
    pas_sites=pas, apa_events=classified, show=True,
)
```

## CLI

```bash
scapaview validate-config configs/example_pbmc1k.yaml
scapaview build-pas-table configs/example_pbmc1k.yaml
scapaview annotate-pas   configs/example_pbmc1k.yaml
scapaview plot-gene      configs/example_pbmc1k.yaml --gene MX1 --group-a B_cell --group-b T_cell
scapaview plot-metagene  configs/example_pbmc1k.yaml --gene-set Dengue_ISG --region 3utr
scapaview summarize-burden configs/example_pbmc1k.yaml
```

## Input Files

See [docs/data_contract.md](docs/data_contract.md).

## Coordinate Conventions

0-based half-open internally; GTF auto-converted. See [docs/coordinate_systems.md](docs/coordinate_systems.md).

## Gene-level APA Tracks

```python
from scapaview.tracks import plot_gene_apa_tracks
fig, axes = plot_gene_apa_tracks(
    "CXCL10", gtf, pas, events,
    bigwig_tracks={"DF": "DF.bw", "DHF": "DHF.bw"},
    output="CXCL10_tracks.png", show=False,
)
```

## Metagene Plots

```python
from scapaview.metagene import plot_metagene_3utr
fig, ax = plot_metagene_3utr(["DF.bw"], ["DHF.bw"], utr_regions, n_bins=100, show=False)
```

## scPolASeq Integration

See [docs/scpolaseq_integration.md](docs/scpolaseq_integration.md).

## Dengue / Viral Infection Use Case

Developed for studying DENV NS5–PAF1 axis disruption in CD14+ monocytes.
Built-in gene sets: `Dengue_ISG`, `PAF1_NS5_axis`, `Antigen_presentation`, `Monocyte_activation`, `Cytotoxic_program`, `RNA_processing_splicing`.

See [docs/biological_use_cases.md](docs/biological_use_cases.md).

## Limitations

- bigWig generation is outside scope (use deepTools).
- APA calling requires scPolASeq or equivalent upstream pipeline.

## Roadmap

- Direct AnnData integration
- Interactive HTML track viewer
- Transcript-model-aware PAS merging
- CLIP-seq / eCLIP overlap analysis

## License

See [LICENSE](LICENSE).
