# Configuration YAML

The FISSEQ pipeline is configured with a YAML file that defines how to
interpret the dataset. A default configuration (`config.yaml`) ships with the
pipeline and is used if no custom configuration is provided.

## Default configuration

```yaml
# Regex or list to select feature columns
# (CellProfiler columns start with an uppercase and contain an underscore)
feature_cols: "^[A-Z][A-Za-z0-9]*_.*"

# SQL-like WHERE clause to select control samples.
# The query will be interpolated into:
#   SELECT * FROM self WHERE {control_sample_query}
control_sample_query: "variantClass = 'WT'"

# The name of the column containing the batch identifier
batch_col_name: "tile_experiment_well"

# Column containing biological labels
label_col_name: "aaChanges"
```

## Field descriptions

### `feature_cols`

- Type: `str` (regex pattern) or `list[str]` (explicit list).
- Default: `^[A-Z][A-Za-z0-9]*_.*`
- Defines which columns are treated as features.
- For CellProfiler outputs, this matches columns starting with an uppercase letter and containing an underscore.

### `control_sample_query`

- Type: `str` (SQL-like WHERE clause).
- Default: `variantClass = 'WT'`
- Used to flag control samples.
- Applied as a boolean mask when constructing metadata.

#### Example

```yaml
control_sample_query: "treatment = 'DMSO'"
```

### `batch_col_name`

- Type: `str`
- Default: `tile_experiment_well`
- Column containing batch identifiers (e.g., well, experiment, or run).
- Used for stratification and harmonization.

### `label_col_name`

- Type: `str`
- Default: `aaChanges`
- Column containing biological labels (e.g., variant information).
- Used for stratification during validation.
