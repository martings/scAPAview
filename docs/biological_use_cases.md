# Biological Use Cases

## 1. PBMC Immune Cell APA Profiling

**Goal**: Compare APA landscape between B cells and T cells in 10x PBMC data.

**Workflow**:
1. Align reads, generate bigWig tracks per cell type.
2. Run scPolASeq APA calling pipeline.
3. Configure `example_pbmc1k.yaml` with output paths.
4. `scapaview build-pas-table configs/example_pbmc1k.yaml`
5. `scapaview annotate-pas configs/example_pbmc1k.yaml`
6. `scapaview plot-gene configs/example_pbmc1k.yaml --gene MX1 --group-a B_cell --group-b T_cell`

**Key gene sets**: `Antigen_presentation`, `Cytotoxic_program`, `Monocyte_activation`

---

## 2. Dengue Viral Infection (scPolASeq)

**Goal**: Identify APA changes driven by dengue virus NS5 protein PAF1C disruption,
comparing dengue fever (DF) vs dengue hemorrhagic fever (DHF) CD14+ monocytes.

**Biological context**: DENV NS5 interacts with PAF1 complex members (PAF1, LEO1, CTR9,
CDC73), leading to global 3' UTR shortening in infected cells. scAPAview is designed to
visualise this effect at single-gene and metagene resolution.

**Workflow**:
1. Integrate scPolASeq + Sierra Quant + scapture outputs.
2. Configure `example_dengue_day1.yaml`.
3. `scapaview build-pas-table configs/example_dengue_day1.yaml`
4. `scapaview plot-gene configs/example_dengue_day1.yaml --gene CXCL10 --group-a DF --group-b DHF --celltype CD14_Monocyte`
5. `scapaview plot-metagene configs/example_dengue_day1.yaml --gene-set Dengue_ISG --region 3utr`
6. `scapaview summarize-burden configs/example_dengue_day1.yaml`

**Key gene sets**: `Dengue_ISG`, `PAF1_NS5_axis`, `RNA_processing_splicing`

---

## 3. Interferon-Stimulated Gene (ISG) APA

**Goal**: Characterise 3' UTR dynamics of ISGs during innate immune activation.

ISGs often show APA regulation after type-I interferon stimulation.
Metagene plots across 3' UTR regions of `Dengue_ISG` gene set reveal
global lengthening or shortening trends.

```bash
scapaview plot-metagene config.yaml --gene-set Dengue_ISG --region 3utr
```

---

## 4. RNA Processing Factor APA

**Goal**: Examine APA in splicing factors and hnRNPs.

The `RNA_processing_splicing` gene set contains core splicing regulators.
Changes in their APA can reflect feedback regulation of the splicing machinery.

---

## Limitations

- bigWig generation (e.g. with deepTools bamCoverage) is outside scAPAview scope.
- APA calling (PDUI computation) requires scPolASeq, Sierra Quant, or equivalent.
- Multi-sample comparisons currently require pre-merged group-level bigWig files.
- scapture Sierra merging is coordinate-based (window ± 25 bp); transcript-model
  matching is not yet implemented.

## Roadmap

- [ ] Direct integration with AnnData objects for cell-level PDUI
- [ ] Pseudobulk bigWig generation wrapper
- [ ] Interactive HTML track viewer
- [ ] Transcript-model-aware PAS merging
- [ ] CLIP-seq / eCLIP overlap for RBP-APA co-analysis
