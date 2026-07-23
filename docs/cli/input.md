# Input

`fisseq-input` (Nextflow process `INPUT`, optional — gated by `params.config_dir`)
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
```

- `input_paths` — one or more raw cell-score files (CSV or Parquet), concatenated.
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

## Output files

Written to `output_dir`, prefixed `{output_root}.` when `output_root` is set:

- `output.parquet` — the selected/filtered cells, ready to be placed in
  `<input_dir>/input/`

## Example

```bash
uv run fisseq-input \
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
