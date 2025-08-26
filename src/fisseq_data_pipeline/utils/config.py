import logging
import pathlib
from typing import Any, Dict, Optional
from os import PathLike

import yaml


DEFAULT_CFG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"
ConfigDict = Dict[str, Any]


class Config:
    def __init__(self, config: Optional[PathLike | "Config"]):
        if config is None:
            logging.info("No config provided, using default config")
            config = DEFAULT_CFG_PATH

        if isinstance(config, Config):
            data = config._data
        else:
            config = pathlib.Path(config)
            with config.open("r") as f:
                data = yaml.safe_load(f)

            data = self._verify_config(data)

        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data[name]

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def _verify_config(self, cfg_data: ConfigDict) -> ConfigDict:
        with DEFAULT_CFG_PATH.open("r") as f:
            default_data = yaml.safe_load(f)

        for key in cfg_data.keys():
            if key in default_data:
                continue

            logging.warning(
                "Removing invalid config option %s from provided config", key
            )
            del cfg_data[key]

        for key in default_data.keys():
            if key in cfg_data:
                continue

            logging.warning(
                "Key %s not in provided config using default value of %s",
                key,
                default_data[key],
            )
            cfg_data[key] = default_data[key]

        return cfg_data
