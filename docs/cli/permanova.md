# PERMANOVA

`fisseq-permanova` (Nextflow process `PERMANOVA`, run once against normalized
cells and once against batch-corrected cells) assesses batch effects in
cell-level data using a per-variant one-way PERMANOVA on cosine distances.
`input_file` is treated as a glob pattern; each matching file becomes one batch
(labeled by its filename stem). For each variant seen in more than one batch, all
unique pairwise cosine distances between its cells are computed (including
cross-batch pairs) and used to derive a pseudo-F statistic via the standard
sum-of-squares decomposition (Anderson 2001). An optional permutation test
(shuffling batch labels while holding distances fixed) yields a p-value per
variant.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern matching one or more batch parquet files. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `n_permutations` | `999` | Number of label permutations for the p-value. `0` skips the permutation test (`p_value` is `null`). |
| `seed` | `42` | Base random seed for label permutation; variant `i` (sorted order) uses `seed + i`. |

## Output files

- `{prefix}permanova.parquet` — one row per eligible variant (seen in more than
  one batch), with per-variant metadata, `f_statistic`, and `p_value`. `prefix` is
  `{output_root}.` when `output_root` is set, otherwise empty.

## Example

```bash
uv run fisseq-permanova \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    n_permutations=999
```

See [API Reference: permanova](../api/permanova.md) for full function
documentation.
