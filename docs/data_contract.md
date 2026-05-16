# scAPAview Data Contract

This document describes all input file formats accepted by scAPAview.

## APA Events (`apa_events.tsv`)

Tab-separated, one row per APA event (gene × comparison).

| Column | Type | Description |
|--------|------|-------------|
| `gene_id` | str | Ensembl gene identifier |
| `site_id` | str | PAS site identifier |
| `group_a` | str | Reference group label |
| `group_b` | str | Comparison group label |
| `pdui_a` | float | PDUI (Proximal/Distal Usage Index) for group A |
| `pdui_b` | float | PDUI for group B |
| `delta_pdui` | float | `pdui_b - pdui_a`; positive = lengthening |
| `p_value` | float | Raw p-value |
| `adj_p_value` | float | FDR-adjusted p-value (BH) |

## PAS Sites (`pas_reference.tsv` or `unified_pas_sites.tsv`)

Tab-separated, one row per polyadenylation site.

| Column | Type | Description |
|--------|------|-------------|
| `site_id` | str | Unique site identifier |
| `gene_id` | str | Host gene identifier |
| `chrom` | str | Chromosome (e.g. `chr1`) |
| `start` | int | 0-based start position |
| `end` | int | 0-based exclusive end position |
| `strand` | str | `+` or `-` |
| `source` | str | Origin tool (`sierra`, `scapture`, `scpolaseq`) |

## PDUI Matrix (`pdui_usage_matrix.tsv`)

Tab-separated matrix, rows = PAS sites, columns = cell groups or individual cells.
First column is the site_id index.

## Cell Labels (`apa_ready_groups.tsv`)

Tab-separated, one row per cell.

| Column | Type | Description |
|--------|------|-------------|
| `cell_id` | str | Unique cell identifier |
| `barcode_raw` | str | Raw 10x barcode |
| `group` | str | Condition/disease group |
| `cluster_id` | str | Cluster assignment |
| `celltype_corrected` | str | Annotated cell type |

## GTF Annotation

Standard GENCODE/Ensembl GTF format (1-based inclusive coordinates).
Internally converted to 0-based half-open; see `docs/coordinate_systems.md`.

## BigWig Coverage Tracks

Standard bigWig format (`.bw` or `.bigwig`).
Requires `pyBigWig` optional dependency: `pip install scapaview[bigwig]`.

## YAML Config

See `configs/example_pbmc1k.yaml` for the canonical structure.
Required top-level sections: `project`, `reference`.
