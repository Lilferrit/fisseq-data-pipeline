# Configuration utilities

The `fisseq_data_pipeline.utils.config` module provides a `Config` object for
managing pipeline configuration. It allows loading configuration values from
YAML files, Python dictionaries, or other `Config` objects, and ensures that
all values are validated against a default configuration.

## Overview

- **`Config`**: A wrapper around a validated configuration dictionary.  
  - Loads from a path, dictionary, `Config`, or falls back to the default
    `config.yaml`.  
  - Allows access via both attribute-style (`cfg.feature_cols`) and
    dictionary-style (`cfg["feature_cols"]`).  
  - Automatically fills in missing keys from the default configuration and
    removes invalid keys.

- **`DEFAULT_CFG_PATH`**: The path to the default configuration YAML file that
  ships with the pipeline.


## Example usage

```python
from fisseq_data_pipeline.utils.config import Config

# Load default configuration
cfg = Config(None)

# Load from a YAML file
cfg = Config("my_config.yaml")

# Load from a Python dict
cfg = Config({"feature_cols": ["f1", "f2"], "_batch": "batch"})

# Load from an existing Config
cfg2 = Config(cfg)

# Access values
print(cfg.feature_cols)
print(cfg["_batch"])
```

## Validation Behavior

When initializing a `Config`:

- Invalid keys not present in the default configuration are removed with a warning.
- Missing keys are filled with the default values from config.yaml.

This ensures that the configuration is always complete and consistent with the pipeline defaults.

## API Reference

---

::: fisseq_data_pipeline.utils.config.Config

---

::: fisseq_data_pipeline.utils.config.DEFAULT_CFG_PATH

---