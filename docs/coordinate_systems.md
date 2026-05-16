# Coordinate Systems in scAPAview

## Summary

| System | Start | End | Used by |
|--------|-------|-----|---------|
| 0-based half-open (BED) | inclusive | exclusive | scAPAview internally, bigWig, BED files |
| 1-based inclusive (GTF) | inclusive | inclusive | GTF/GFF files |
| 1-based inclusive (VCF) | inclusive | inclusive | VCF files (positions only) |

## Internal Convention

scAPAview uses **0-based half-open** coordinates throughout:

- `start` is the first included base (0-based).
- `end` is the first excluded base (exclusive).
- Length of a feature = `end - start`.

This matches Python string slicing and the BED format.

## GTF Conversion

When reading a GTF file with `read_gtf()`, the function calls `standardize_gtf()` which performs:

```python
gtf["Start"] = gtf["Start"] - 1   # 1-based → 0-based
# gtf["End"] unchanged             # 1-based inclusive == 0-based exclusive
```

**Example:**

| GTF Start | GTF End | 0-based Start | 0-based End | Length |
|-----------|---------|---------------|-------------|--------|
| 1001 | 5000 | 1000 | 5000 | 4000 bp |
| 7001 | 12000 | 7000 | 12000 | 5000 bp |

## Relative Positions

The `compute_relative_position()` function returns a value in [0, 1]:

- **`+` strand**: 0 = region start (5' end), 1 = region end (3' end)
- **`−` strand**: 0 = region end (5' end in transcript space), 1 = region start (3' end in transcript space)

This ensures metagene plots always display 5'→3' direction left→right.

## BigWig Queries

All bigWig queries use 0-based half-open intervals (matching pyBigWig convention):

```python
bw.values("chr1", 1000, 5000)   # returns 4000 values
bw.stats("chr1", 1000, 5000, nBins=100)
```
