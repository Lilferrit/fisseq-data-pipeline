# AppConfig

The `fisseq_data_pipeline.config` module defines `AppConfig`, the shared base configuration used by all pipeline entry points. It is a Hydra structured config implemented as a `dataclasses.dataclass`.

---

## Fields

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `output_dir` | `str` | **required** | Directory where output files and logs are written. |
| `output_root` | `str \| None` | `None` | Optional prefix for output file names. When set, files are named `{output_root}.{stem}.{ext}` rather than `{output_dir}/{filename}`. |
| `log_level` | `str` | `"info"` | Logging verbosity. One of: `debug`, `info`, `warning`, `error`, `critical`. |

---

## Usage

`AppConfig` is extended by each module's config:

- [`NormalizeConfig`](./normalize.md) — adds `input_file`, `control_sample_query`, `save_normalizer`
- [`AggregateConfig`](./aggregate.md) — adds `input_file`, `label_column`, `aggregator`, `save_normalizer`

---

## API reference

---

::: fisseq_data_pipeline.config.AppConfig

---
