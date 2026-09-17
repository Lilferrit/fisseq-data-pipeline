"""Unit tests keeping ``workflows/fisseq.nf``'s hand-maintained allowlists in
step with the Python definitions they mirror.

``fisseq.nf`` validates ``params.feature_select_types`` and
``params.ovwt_cv_mode`` at workflow-construction time, before any task is
submitted. It has to: both values are interpolated straight into a process's
shell script, so a malformed entry breaks the generated ``.command.sh`` at the
bash level and the Python-side ``ValueError`` never gets a chance to fire.
Groovy cannot import the Python registries, so the valid sets are spelled out
twice -- and these tests are what stop the two copies from drifting.
"""

import pathlib
import re

import yaml

from fisseq_data_pipeline.aggregate import _AGGREGATORS
from fisseq_data_pipeline.ovwt import CV_MODES

WORKFLOW = pathlib.Path(__file__).parents[2] / "workflows" / "fisseq.nf"


def _groovy_string_list(function_name: str) -> set[str]:
    """
    Extract the quoted strings from a zero-argument Groovy function's body.

    Parameters
    ----------
    function_name : str
        Name of the ``def <name>() { ... }`` helper to read.

    Returns
    -------
    set[str]
        Every single- or double-quoted literal in that function's body.
    """
    source = WORKFLOW.read_text()
    match = re.search(
        rf"def {re.escape(function_name)}\(\) \{{(.*?)\n\}}", source, re.DOTALL
    )
    assert match is not None, f"{function_name}() not found in {WORKFLOW}"
    return set(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))


def test_aggregator_keys_match_python_registry():
    """A new aggregator must be added to fisseq.nf's allowlist too.

    Otherwise the pipeline rejects a feature type the Python side supports.
    """
    assert _groovy_string_list("aggregatorKeys") == set(_AGGREGATORS)


def test_ovwt_cv_modes_match_python_constants():
    assert _groovy_string_list("ovwtCvModes") == set(CV_MODES)


def test_params_yaml_feature_select_types_are_all_valid():
    """Guards the repo default against the mis-quoted-YAML failure mode.

    A missing opening quote in the flow sequence (``[median", "KS"]``) parses
    as the literal string ``median"``, which then breaks AGGREGATE_FEATURE_TYPE's
    generated shell script rather than failing in Python.
    """
    params = yaml.safe_load(
        (pathlib.Path(__file__).parents[2] / "params.yaml").read_text()
    )
    assert set(params["feature_select_types"]) <= set(_AGGREGATORS)
    assert params["ovwt_cv_mode"] in CV_MODES


def test_params_yaml_passthrough_types_are_all_valid():
    params = yaml.safe_load(
        (pathlib.Path(__file__).parents[2] / "params.yaml").read_text()
    )
    assert set(params["feature_select_passthrough_types"]) <= set(_AGGREGATORS)


def test_params_yaml_feature_select_lists_are_disjoint():
    """``fisseq.nf`` rejects an overlap, and ``featureselect.main`` raises on the
    resulting column collision; the repo default must not trip either."""
    params = yaml.safe_load(
        (pathlib.Path(__file__).parents[2] / "params.yaml").read_text()
    )
    assert not (
        set(params["feature_select_types"])
        & set(params["feature_select_passthrough_types"])
    )


def test_passthrough_validation_reuses_aggregator_keys():
    """Both lists are validated against the same Groovy helper — a new
    aggregator must not need a third hardcoded copy of the key set."""
    source = (pathlib.Path(__file__).parents[2] / "workflows" / "fisseq.nf").read_text()
    passthrough_block = source[source.index("feature_select_passthrough_types") :]
    assert "aggregatorKeys()" in passthrough_block
