"""Base Hydra structured config shared by every pipeline entry point.

Defines :class:`AppConfig`, supplying ``output_dir``, ``output_root``,
``log_level``, and ``random_seed`` fields common to all Hydra CLIs in this
package.
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
    random_seed : int
        The one seed every stochastic pipeline stage reads from -- QC
        pseudo-variant downsampling, the feature-selection bootstrap splits and
        their wildtype subsampling, OvWT's ``StratifiedKFold`` shuffle / inner
        calibration split / XGBoost ``seed``, PCA's solver, and UMAP's fit.
        Defaults to ``0``.

        Never add a stage-local ``random_state``/``seed`` field to a config
        that extends this one: a stage that must differ from its siblings
        derives a fixed offset from this seed instead (e.g. GENERATE_SPLIT uses
        ``random_seed + bootstrap_idx``), so changing this single value moves
        every stage coherently. ``tests/unit/test_config.py`` enforces this.
    """

    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
    random_seed: int = 0
