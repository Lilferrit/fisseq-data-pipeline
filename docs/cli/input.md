# Input

`python -m fisseq_data_pipeline.input` (Nextflow process `INPUT`, optional — gated by `params.yaml_config_dir`)
reads a hand-authored YAML config describing one or more raw cell-score files
(CSV or Parquet), classifies each row's variant, and restricts to a fixed set of
variant classes plus the most common missense variants. Writes a single
`input/`-ready cell-level Parquet file.

## Config fields

Extends the common `output_dir` / `output_root` / `log_level` fields (see
[Common config fields](#common-config-fields) below).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `config_path` | **required** | Path to a separate YAML file (see below) describing the input files and variant selection behavior. Parsed independently of the Hydra CLI config. |

### `config_path` YAML schema

```yaml
input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
top_n_missense: null              # optional, default null (keep all Single Missense variants)
feature_allowlist_file: null      # optional, default null (no allowlist)
feature_blocklist_file: null      # optional, default null (no blocklist)
convert_first: false              # optional, default false (see below)
temp_dir: null                    # optional, default $TMPDIR or the system temp dir
```

- `input_paths` — one or more raw cell-score files (CSV or Parquet), concatenated.
  **Required, and batch-YAML-only** — there is no pipeline-wide default for a
  per-batch list of raw data files (see
  [Per-batch parameter overrides](../nextflow.md#per-batch-parameter-overrides)).
- `top_n_missense` — if set, the number of Single Missense variants (by cell
  count) to keep, alongside Synonymous, WT, and Frameshift variants. Omit or
  set to `null` (the default) to keep all Single Missense variants without
  any top-N restriction.
- `feature_allowlist_file` / `feature_blocklist_file` — optional paths to plain
  text files, one fnmatch-style glob pattern per line (e.g.
  `Cells_AreaShape_*`), matched against feature column names. If an allowlist
  is given, only feature columns matching at least one of its patterns are
  kept; if a blocklist is also given, matching columns are then dropped from
  what remains (allowlist is applied first). Identity columns (`upBarcode`,
  `editDistance`, `aaChanges`) and metadata columns are unaffected.
- `convert_first` — set to `true` to merge all `input_paths` into a single
  Parquet file up front (written to `temp_dir`, deleted once the run
  finishes) before variant classification, instead of re-scanning/
  re-concatenating the original files on every downstream pass. Only takes
  effect when `top_n_missense` is also set (that's what causes the extra
  pass it optimizes for); otherwise this is a no-op even if `true`.
- `temp_dir` — where `convert_first`'s merged file is written. Defaults to
  `$TMPDIR` if set, otherwise the system temp directory. Never read when
  `convert_first` is false (or a no-op per above).

Except for `input_paths`, every field above is also a plain `nextflow.config`
pipeline-wide default (`params.top_n_missense`, `params.convert_first`,
`params.temp_dir`, `params.feature_allowlist_file`,
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
