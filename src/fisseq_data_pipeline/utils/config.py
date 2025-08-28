import logging
import pathlib
from os import PathLike
from typing import Any, Dict, Optional

import yaml

DEFAULT_CFG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"
ConfigDict = Dict[str, Any]


class Config:
    """
    A configuration object that wraps a dictionary of key-value pairs
    loaded from a provided path, dictionary, or another ``Config`` instance.
    If no configuration is provided, the default configuration file is used.

    Parameters
    ----------
    config : PathLike or dict or Config, optional
        - If ``None``, the default configuration file path is used.
        - If a ``dict``, the dictionary is validated and used directly.
        - If a ``PathLike``, the configuration is loaded from the YAML file.
        - If a ``Config``, the underlying configuration data is reused.
    """

    def __init__(self, config: Optional[PathLike | ConfigDict | "Config"]):
        """
        Retrieve a configuration value as an attribute.

        Parameters
        ----------
        name : str
            The name of the configuration option.

        Returns
        -------
        Any
            The value of the requested configuration option.

        Raises
        ------
        KeyError
            If the key does not exist in the configuration.
        """
        if config is None:
            logging.info("No config provided, using default config")
            config = DEFAULT_CFG_PATH

        if isinstance(config, Config):
            data = config._data
        else:
            if isinstance(config, dict):
                data = config
            else:
                config = pathlib.Path(config)
                with config.open("r") as f:
                    data = yaml.safe_load(f)

            data = self._verify_config(data)

        logging.debug("Using config %s", config)
        self._data = data

    def __getattr__(self, name: str) -> Any:
        """Retrieve a configuration value as an attribute."""
        return self._data[name]

    def __getitem__(self, key: str) -> Any:
        """Retrieve a configuration value using dictionary-style indexing."""
        return self.__getattr__(key)

    def _verify_config(self, cfg_data: ConfigDict) -> ConfigDict:
        """
        Verify the provided configuration against the default configuration.
        Invalid keys are removed, and missing keys are filled with defaults.

        Parameters
        ----------
        cfg_data : dict
            The configuration data to verify.

        Returns
        -------
        dict
            The validated configuration dictionary with defaults applied.
        """
        with DEFAULT_CFG_PATH.open("r") as f:
            default_data = yaml.safe_load(f)

        for key in list(cfg_data.keys()):
            if key in default_data:
                continue

            logging.warning(
                "Removing invalid config option %s from provided config", key
            )
            del cfg_data[key]

        for key in list(default_data.keys()):
            if key in cfg_data:
                continue

            logging.warning(
                "Key %s not in provided config using default value of %s",
                key,
                default_data[key],
            )
            cfg_data[key] = default_data[key]

        return cfg_data
