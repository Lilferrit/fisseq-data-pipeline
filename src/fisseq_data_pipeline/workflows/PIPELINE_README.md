# FISSEQ Nextflow Pipeline

A Nextflow DSL2 workflow that runs the full FISSEQ data processing pipeline, from raw cell parquets through QC, normalization, batch-effect assessment, variant classification, and feature selection.

---

## Prerequisites

- [Nextflow](https://www.nextflow.io/) ≥ 23.10
- The `fisseq-data-pipeline` Python package installed and available on `PATH` (or configured via a profile — see below)

---

## Directory layout

The pipeline expects a single root directory (`--input_dir`) with the following structure:

```
<input_dir>/
  input/          ← place your raw .parquet cell files here (one per batch)
```

All outputs are written back into `<input_dir>` alongside the `input/` folder:

```
<input_dir>/
  input/
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  permanova/
    wildtype/permanova.parquet
    synonymous/permanova.parquet
  ovwt_batchwise/<batch>/
    results.csv
    models.pkl
  ovwt_global/
    results.csv
    models.pkl
  feature_select_batchwise/<batch>/
    <batch>.parquet
    feature_correlations.parquet
  feature_select_global/
    global.parquet
    feature_correlations.parquet
```

---

## Quickstart

After running `fisseq-env-init`, the pipeline file and config live inside the experiment directory. Run from there:

```bash
cd /path/to/experiment
nextflow run fisseq_pipeline.nf --input_dir .
```

Nextflow writes pipeline logs and per-process work directories to `./work/` by default. Point `-w` at a scratch filesystem on shared clusters:

```bash
cd /path/to/experiment
nextflow run fisseq_pipeline.nf --input_dir . -w /scratch/$USER/nf-work
```

---

## Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--input_dir` | **required** | Root directory containing `input/*.parquet` batch files. |
| `--bc_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_bc_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--minimum_correlation` | `0.5` | Minimum pseudo-replicate Pearson *r* for feature selection. |
| `--permanova_n_bootstraps` | `200` | Bootstrap iterations for PERMANOVA. |
| `--permanova_sample_size` | `1000` | Rows per PERMANOVA bootstrap sample. |
| `--ovwt_min_cells` | `250` | Minimum cells required per variant for OvWT classification. |
| `--aggregator` | `"multi"` | Feature aggregation method (see `fisseq-aggregate` docs). |

Override any parameter on the command line:

```bash
nextflow run fisseq_pipeline.nf \
    --input_dir . \
    --bc_threshold 20 \
    --minimum_correlation 0.7
```

Or collect overrides in a params file (`params.yml`):

```yaml
input_dir: /path/to/experiment
bc_threshold: 20
minimum_correlation: 0.7
```

```bash
nextflow run fisseq_pipeline.nf -params-file params.yml
```

---

## Environment / execution profiles

Edit `nextflow.config` to configure how the pipeline finds your Python environment and where it runs jobs.

### Local (venv)

Uncomment the `venv` block in `nextflow.config` and set the path to your virtual environment:

```groovy
venv {
    process.beforeScript = "source /path/to/.venv/bin/activate"
}
```

Run with:

```bash
nextflow run fisseq_pipeline.nf -profile venv --input_dir .
```

### SGE cluster

Uncomment the `sge` block in `nextflow.config`, set your queue name, and adjust resource limits:

```groovy
sge {
    process {
        executor       = 'sge'
        queue          = 'all.q'
        clusterOptions = '-V'
        cpus           = 4
        memory         = '16 GB'
        time           = '4h'
        beforeScript   = "source /path/to/.venv/bin/activate"
    }
}
```

Run with:

```bash
nextflow run fisseq_pipeline.nf -profile sge --input_dir .
```

The `-V` flag in `clusterOptions` forwards your current shell environment to each SGE job, so any `module load` commands or environment variables set before launching Nextflow are inherited by the cluster jobs.

### Other executors

Nextflow supports SLURM, PBS, LSF, AWS Batch, Google Cloud Batch, and more. Add a profile block with `executor = '<name>'` and the relevant settings. See the [Nextflow executor docs](https://www.nextflow.io/docs/latest/executor.html) for the full list.

---

## Resuming a run

Nextflow caches completed tasks. If a run is interrupted, resume from the last successful step:

```bash
nextflow run fisseq_pipeline.nf -resume --input_dir .
```

---

## Pipeline DAG

```
input/*.parquet
     │
     ▼
QC_FILTER  (per batch)
     │
     ▼
NORMALIZE  (per batch)
     │
     ├──► PERMANOVA_WT            (global — waits for all batches)
     ├──► PERMANOVA_SYN           (global — waits for all batches)
     ├──► OVWT_BATCHWISE          (per batch)
     ├──► OVWT_GLOBAL             (global — waits for all batches)
     ├──► FEATURE_SELECT_BATCHWISE (per batch)
     └──► FEATURE_SELECT_GLOBAL   (global — waits for all batches)
```
