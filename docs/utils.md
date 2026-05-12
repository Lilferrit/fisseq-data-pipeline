# Utilities

The `fisseq_data_pipeline.utils` module provides shared helpers used internally by the pipeline entry points.

---

## Overview

- **`setup_logging`** — Configure a timestamped file + console logger for a pipeline run.

---

## Example usage

```python
from fisseq_data_pipeline.config import AppConfig
from fisseq_data_pipeline.utils import setup_logging

cfg = AppConfig(output_dir="./out", log_level="debug")
setup_logging(cfg, "normalize")
```

Log files are written to `{output_dir}/{name}.log`, or `{output_dir}/{output_root}.{name}.log` when `output_root` is set.

---

## API reference

---

::: fisseq_data_pipeline.utils.setup_logging

---
