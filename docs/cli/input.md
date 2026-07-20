# Input

`fisseq-input` (Nextflow process `INPUT`, optional — gated by `params.config_dir`)
reads a hand-authored YAML config describing one or more raw cell-score files
(CSV or Parquet), classifies each row's variant, restricts to a fixed set of
variant classes plus the most common missense variants, and optionally augments
the result with reproducibly-downsampled "pseudo variant" rows for QC/calibration
purposes. Writes a single `input/`-ready cell-level Parquet file.

## Config fields

Extends the common `output_dir` / `output_root` / `log_level` fields (see
[Common config fields](#common-config-fields) below).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `config_path` | **required** | Path to a separate YAML file (see below) describing the input files and downsampling behavior. Parsed independently of the Hydra CLI config. |

### `config_path` YAML schema

```yaml
input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
top_n_missense: 50                # optional, default 50
downsample_fraction: null         # optional, default null (disabled)
downsample_seed: 0                # optional, default 0
```

- `input_paths` — one or more raw cell-score files (CSV or Parquet), concatenated.
- `top_n_missense` — number of Single Missense variants (by cell count) to keep,
  alongside Synonymous, WT, and Frameshift variants.
- `downsample_fraction` — if set to a value in `(0, 1]`, generates reproducibly
  downsampled "pseudo variant" rows for the kept Synonymous and Single Missense
  variants, tagged `<aaChanges>:downsampled-half` (e.g. `V123A` ->
  `V123A:downsampled-half`). Disabled by default — omit or set to `null` to skip
  pseudo-variant generation entirely.
- `downsample_seed` — seed for the deterministic downsample selection; only used
  when `downsample_fraction` is set.

## Output files

Written to `output_dir`, prefixed `{output_root}.` when `output_root` is set:

- `output.parquet` — the selected/filtered (and optionally downsampled) cells,
  ready to be placed in `<input_dir>/input/`

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
