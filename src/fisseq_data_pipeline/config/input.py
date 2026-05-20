import dataclasses

from omegaconf import MISSING

from .app import AppConfig


@dataclasses.dataclass
class InputConfig(AppConfig):
    """
    Extends AppConfig with a required input file path.

    Attributes
    ----------
    input_file : str
        Path to the input file. Required.
    """

    input_file: str = MISSING


@dataclasses.dataclass
class LabeledInputConfig(InputConfig):
    """
    Extends InputConfig for steps that operate on variant-labeled data.

    Attributes
    ----------
    label_column : str
        Name of the column identifying variant labels. Defaults to
        ``"meta_aa_changes"``.
    """

    label_column: str = "meta_aa_changes"
