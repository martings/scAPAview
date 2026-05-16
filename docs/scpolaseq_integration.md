# scPolASeq Integration

## What is scPolASeq?

scPolASeq (single-cell Poly-A Sequencing) is a protocol that captures the 3' ends of
transcripts in single cells, enabling measurement of polyadenylation site usage at
single-cell resolution.

## Expected Outputs from scPolASeq Pipeline

| File | Description |
|------|-------------|
| `site_catalog.tsv` | All detected PAS sites across all cells |
| `pas_reference.tsv` | Filtered/annotated reference PAS sites |
| `pdui_usage_matrix.tsv` | PDUI scores per site per group |
| `apa_events.tsv` | APA differential usage events |
| `apa_ready_groups.tsv` | Cell-to-group assignments |

## Config Setup

Point scAPAview to your scPolASeq outputs via the YAML config:

```yaml
scpolaseq:
  site_catalog: /path/to/apa/site_catalog.tsv
  pas_reference: /path/to/pas_reference/pas_reference.tsv
  pdui_matrix: /path/to/apa_calls/pdui_usage_matrix.tsv
  apa_events: /path/to/apa_calls/apa_events.tsv

cell_labels:
  apa_ready_groups: /path/to/apa_ready_groups.tsv
```

## Multi-tool Integration

scAPAview can merge PAS sites from:

1. **scPolASeq** – primary source, cell-type-specific PDUI
2. **Sierra Quant** – scRNA-seq based PAS detection
3. **scapture** – additional PAS capture

Sites within 25 bp (configurable via `pas_match_window`) are merged into a single
unified site. Source provenance is tracked in the `source` column.

## Bigwig Track Convention

bigWig files should be coverage-normalised (e.g. CPM or RPM).
Organise tracks in the config as:

```yaml
bigwig_tracks:
  GroupName:
    all: /path/to/GroupName.all_cells.bw
    CellType1: /path/to/GroupName.CellType1.bw
```

The `--celltype` option in `scapaview plot-gene` selects the matching track.
