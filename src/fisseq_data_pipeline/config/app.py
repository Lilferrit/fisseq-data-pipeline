"""Base Hydra structured config shared by every pipeline entry point.

Defines :class:`AppConfig`, supplying ``output_dir``, ``output_root``, and
``log_level`` fields common to all Hydra CLIs in this package.
"""

import dataclasses
from typing import Optional

from omegaconf import MISSING


@dataclasses.dataclass
class AppConfig:
    """
    Shared application-level configuration.

    Attributes
    ----------
    output_dir : str
        Directory for outputs produced by the current run (e.g. per-experiment
        results, normalized data, model artifacts). Required.
    output_root : str or None
        If set, every output file is prefixed ``{output_root}.{name}`` instead
        of being placed under ``output_dir``. Optional, defaults to ``None``.
    log_level : str
        Logging verbosity. One of ``debug``, ``info``, ``warning``, ``error``,
        ``critical``. Defaults to ``info``.
    """

    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
