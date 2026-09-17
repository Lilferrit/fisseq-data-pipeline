# Configuration

Every pipeline parameter lives in **`params.yaml`** at the repo root. It must be
passed explicitly:

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
```

`nextflow.config` carries executor, profile and container settings **only** —
never parameter defaults.

## Precedence

1. A bare CLI flag (`--ovwt_min_cells 500`) — highest.
2. `-params-file params.yaml`.
3. The Python Hydra dataclass default for that stage.

Nextflow accepts only one `-params-file` at a time, so to run an alternate
parameter set, copy `params.yaml` and edit the copy.

!!! warning "Do not add a `params { }` block to `nextflow.config`"
    Not even an empty one. Combined with `-params-file`, Nextflow's
    `ConfigBuilder` before 26.04.6 misattributes every `params.yaml` key into the
    `process{}` scope and fails with `Unknown config attribute 'params'`.

## Declaring experiments

`experiments:` is a list of maps, one per experiment. This replaces the former
mandatory `<pipeline_dir>/configs/*.yaml` directory.

```yaml
experiments:
  - batch_stem: plate1
    input_paths:
      - /data/plate1/cellprofiler_features.csv
      - /data/plate1/barcode_calls.parquet
    global_channel: cohort_a
  - batch_stem: plate2
    input_paths: [/data/plate2/cellprofiler_features.csv]
    global_channel: [cohort_a, cohort_b]   # may belong to several
    csv_schema_scan_rows: null             # per-experiment override
  - batch_stem: plate3
    input_paths: [/data/plate3/features.parquet]
    # no global_channel -- processed batchwise only
```

An entry may set **only** these keys:

| Key | Required | Meaning |
| --- | -------- | ------- |
| `batch_stem` | yes | Unique experiment id. Names every output subdirectory. Must be unique across the list. |
| `input_paths` | yes | Raw CSV/parquet source files that INPUT merges. No pipeline-wide default exists — a list of raw files is inherently per-experiment. |
| `global_channel` | no | String or list of strings naming the channel(s) this experiment belongs to. |
| `feature_allowlist_file` | no | INPUT-stage; falls back to the pipeline-wide default. |
| `feature_blocklist_file` | no | Likewise. |
| `csv_schema_scan_rows` | no | Likewise. |

Any other key is **rejected with an error naming it**, rather than silently
ignored. If you want to change a QC threshold or an OvWT hyperparameter, set it
at the top level of `params.yaml` — it applies pipeline-wide. This is a
deliberate simplification: per-experiment override of arbitrary parameters was
removed for the release.

The workflow also fails fast on an empty `experiments:` list, a non-map entry, a
missing or blank `batch_stem`, a missing or empty `input_paths`, and duplicate
`batch_stem` values.

## Global channels

An experiment joins a channel via its `global_channel` key;
`params.global_channels` lists which channels actually run.

```yaml
global_channels: [cohort_a, cohort_b]
```

Each active channel gets its own `GLOBAL_OVWT` and `GLOBAL_FEATURE_SELECT` run,
scoped to that channel's member experiments and published under
`global/<channel>/`.

- `global_channels: null` or `[]` (the default) — no global stage runs at all.
- An experiment naming no channel is still processed batchwise, just excluded
  from both global stages.
- An experiment in several active channels contributes to each independently.
- A channel named in `global_channels` with no member experiments logs a
  warning.

!!! note "GLOBAL_OVWT needs at least two synonymous variants per experiment"
    It fits a per-experiment normalizer on that experiment's synonymous rows. A
    single synonymous variant makes the standard deviation (ddof=1) undefined,
    nulling every score for that experiment.

## The one random seed

```yaml
random_seed: 0
```

This is the only seed in the pipeline. Every stochastic step derives from it:
QC pseudo-variant downsampling, the feature-selection bootstrap splits and their
wildtype subsampling, OvWT's fold shuffle / inner calibration split / XGBoost
`seed`, PCA's solver, and UMAP's fit. Changing it moves all of them coherently.

Stages that must differ from one another derive a fixed offset rather than
owning a seed of their own (`GENERATE_SPLIT` uses `random_seed + bootstrap_idx`;
`AGGREGATE_HALF` uses `random_seed + bootstrap_idx * 2 + half_num`), so
bootstrap replicates still draw independent subsamples.

There is deliberately no stage-local `random_state` anywhere, and
`tests/unit/test_config.py` fails if one reappears.

## Run gates

All pipeline-wide.

| Parameter | Default | Effect when `false` |
| --------- | ------- | ------------------- |
| `run_ovwt` | `true` | Skips `OVWT_BATCHWISE` and, with it, `GLOBAL_OVWT`. |
| `run_feature_selection` | `true` | Skips the whole batchwise feature-selection chain and `GLOBAL_FEATURE_SELECT`. |
| `run_pca` | `false` | (Enable to add PCA to the feature-selection outputs.) |
| `run_umap` | `false` | (Enable to add UMAP.) |

## Parameter reference

### Required, no default

| Parameter | Meaning |
| --------- | ------- |
| `pipeline_dir` | Root output directory. Also settable as `--pipeline_dir`. |
| `container_image` | Image every process runs in. Pin to `:<short-sha>` for a reproducible run rather than floating on `:latest`. |
| `experiments` | See above. |

### Shared

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `random_seed` | `0` | The one seed. |
| `global_channels` | `null` | Which channels run the global stages. |
| `filter_label_column` | `"meta_aa_changes"` | Variant label column, threaded to every stage that reads one. |

### INPUT

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `feature_allowlist_file` | `null` | Restrict to these feature columns. |
| `feature_blocklist_file` | `null` | Drop these feature columns. |
| `csv_schema_scan_rows` | `100` | Rows scanned to infer CSV dtypes; `null` scans every row. No effect on parquet sources. |

### QC_FILTER

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `barcode_count_threshold` | `10` | Minimum cells per barcode. |
| `variant_barcode_count_threshold` | `4` | Minimum barcodes per variant. |
| `edit_distance_threshold` | `1` | Maximum barcode edit distance. |
| `qc_n_variants` | `null` | Cap the number of distinct variants in `qc_variant_downsample_classes`. |
| `qc_variant_downsample_classes` | `["Single Missense"]` | Classes eligible for that cap. |
| `qc_variant_downsample_mode` | `"top"` | `"top"` (highest cell count) or `"random"` (seeded by `random_seed`). |
| `qc_downsample_amounts` | `null` | Float in (0,1] or int, or a list of them: pseudo-variant downsampling per label group. Each amount gets its own `:downsample-{amount}` tag. |
| `qc_downsample_classes` | `["Synonymous", "Single Missense"]` | Classes eligible for pseudo-variant generation. |

### OVWT_BATCHWISE

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `ovwt_wt_label` | `"WT"` | Label identifying wildtype cells. |
| `ovwt_cv_mode` | `"kfold"` | Fold scheme: `"kfold"` or `"barcode_holdout"`. |
| `ovwt_n_folds` | `5` | Cross-validation folds per variant. Under `"kfold"` the fold count outright; under `"barcode_holdout"` a cap — the variant's barcodes are packed into at most this many cell-count-balanced groups, one held out per fold. `null` = one fold per barcode, and is an error under `"kfold"`. |
| `ovwt_calibrate` | `true` | Per-fold sigmoid (Platt) calibration. |
| `ovwt_min_cells` | `250` | Minimum cells for a variant to be scored; wildtype always kept. `null` disables. |
| `ovwt_downsample_wt` | `true` | Barcode-proportional wildtype downsampling to the largest remaining variant group. |

See [One-vs-WT](cli/ovwt.md) for what these actually do.

### Feature selection

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `feature_select_types` | `["mean","median","MAD","std","KS","QQ","AUROC"]` | Aggregators to compute and correlate. |
| `feature_select_passthrough_types` | `[]` | Aggregators computed and joined onto the final per-variant table but excluded from every selection step — no bootstrap, no blocklist, no pycytominer filters, no normalization. Intended for the p-value statistics (`KSnegLogP`, `AUROCnegLogP`). Must not overlap `feature_select_types`. |
| `feature_select_bootstrap_reps` | `10` | Bootstrap replicates per feature type. |
| `feature_select_downsample_wt` | `null` | Optional wildtype downsampling during aggregation. |
| `feature_select_min_correlation` | `0.5` | Median-`r` threshold for a feature to pass. |
| `global_feature_select_min_batches_ok` | `null` | Minimum member experiments that must mark a feature ok. `null` = all that report on it. |

### Dimensionality reduction

PCA and UMAP are computed independently of each other, both on the same final
selected/normalized feature matrix — UMAP does **not** run on PCA output.

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `pca_n_components` | `10` | Must be ≤ `min(n_rows, n_retained_features)` after all-null columns are dropped, or the run fails. |
| `umap_n_components` | `2` | Embedding dimensionality. |
| `umap_n_neighbors` | `10` | Local neighborhood size. |
| `umap_metric` | `"cosine"` | Distance metric. |
| `umap_min_dist` | `0.1` | Minimum embedded distance between points. |

!!! note "UMAP is now always seeded"
    It previously had its own nullable `umap_random_state` (default `42`), where
    `null` opted into faster nondeterministic multithreaded fitting. That knob is
    gone — UMAP now reads `random_seed` like everything else, so UMAP output from
    this release will not match a pre-release run's.

## Passing list values on the CLI

List-valued parameters cannot be expressed as a bare CLI flag:
`--global_channels foo,bar` arrives as the single string `"foo,bar"`, and
Groovy's `as List<String>` then splits it into individual characters. Set list
parameters in `params.yaml` (or a copy of it) instead.
