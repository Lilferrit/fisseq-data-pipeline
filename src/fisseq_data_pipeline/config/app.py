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
    output_root : str
        Root directory under which per-run output directories are created.
        Required.
    log_level : str
        Logging verbosity. One of ``debug``, ``info``, ``warning``, ``error``,
        ``critical``. Defaults to ``info``.
    """

    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
