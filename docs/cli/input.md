# Input

`python -m fisseq_data_pipeline.input` (Nextflow process `INPUT`, optional — gated by `params.yaml_config_dir`)
reads a hand-authored YAML config describing one or more raw cell-score files
(CSV or Parquet) and merges them into a single `input/`-ready cell-level
Parquet file.

Variant-class/count-based restriction (previously done here via
`top_n_missense`) now happens downstream, in `QC_FILTER`'s `n_variants` /
`variant_downsample_classes` / `variant_downsample_mode` (see
[CLI Reference: qcfilter](qcfilter.md)), so it applies uniformly to every
batch rather than only to batches routed through this optional stage.

## Config fields

Extends the common `output_dir` / `output_root` / `log_level` fields (see
[Common config fields](#common-config-fields) below).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `config_path` | **required** | Path to a separate YAML file (see below) describing the input files and variant selection behavior. Parsed independently of the Hydra CLI config. |

### `config_path` YAML schema

```yaml
input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
feature_allowlist_file: null      # optional, default null (no allowlist)
feature_blocklist_file: null      # optional, default null (no blocklist)
```

- `input_paths` — one or more raw cell-score files (CSV or Parquet), concatenated.
  **Required, and batch-YAML-only** — there is no pipeline-wide default for a
  per-batch list of raw data files (see
  [Per-batch parameter overrides](../nextflow.md#per-batch-parameter-overrides)).
- `feature_allowlist_file` / `feature_blocklist_file` — optional paths to plain
  text files, one fnmatch-style glob pattern per line (e.g.
  `Cells_AreaShape_*`), matched against feature column names. If an allowlist
  is given, only feature columns matching at least one of its patterns are
  kept; if a blocklist is also given, matching columns are then dropped from
  what remains (allowlist is applied first). Identity columns (`upBarcode`,
  `editDistance`, `aaChanges`) and metadata columns are unaffected.

Except for `input_paths`, every field above is also a plain `nextflow.config`
pipeline-wide default (`params.feature_allowlist_file`,
`params.feature_blocklist_file`) — set one on the command line or in
`nextflow.config` to apply it to every batch, and/or override it for a
specific batch in that batch's YAML. See
[Per-batch parameter overrides](../nextflow.md#per-batch-parameter-overrides)
for the full mechanism.

## Output files

Written to `output_dir`, prefixed `{output_root}.` when `output_root` is set:

- `output.parquet` — the selected/filtered cells, ready to be placed in
  `<input_dir>/input/`

## Example

```bash
uv run python -m fisseq_data_pipeline.input \
    output_dir=./out \
    config_path=configs/batch1.yaml
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |

See [API Reference: input](../api/input.md) for full function documentation.
