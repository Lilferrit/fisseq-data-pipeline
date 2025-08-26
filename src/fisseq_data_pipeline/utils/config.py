import pathlib
from typing import Any

import yaml

from .types import PathLike

class Config:
    """Configuration loader with strict attribute access.

    Loads key/value pairs from a YAML file and exposes them
    as attributes. Accessing a missing attribute raises an
    AttributeError.
    """

    def __init__(self, yaml_path: PathLike):
        yaml_path = pathlib.Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open("r") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("YAML root must be a mapping (dict).")

        self._config = data

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(
            f"Config key '{name}' not found in YAML file. "
            "Available keys: " + ", ".join(self._config.keys())
        )

    def __getitem__(self, key: str) -> Any:
        # Allow dict-like access
        return self.__getattr__(key)

    def keys(self):
        return self._config.keys()

    def as_dict(self):
        return dict(self._config)

