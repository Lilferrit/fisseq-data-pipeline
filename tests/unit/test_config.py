"""Guards the single-shared-seed invariant across every stage config.

``AppConfig.random_seed`` is the one seed in this pipeline. A stage that needs
to differ from its siblings derives a fixed offset from it (e.g. GENERATE_SPLIT
uses ``random_seed + bootstrap_idx``) rather than declaring a seed of its own --
otherwise "set the seed and rerun" stops being a complete statement about
reproducibility, which is the whole point of the release.
"""

import dataclasses
import importlib
import pkgutil

import pytest

import fisseq_data_pipeline
from fisseq_data_pipeline.config import AppConfig

# Field names that would reintroduce a second, independent seed.
_BANNED_SEED_FIELDS = {"random_state", "seed", "downsample_seed", "umap_random_state"}


def _stage_config_classes() -> list[type]:
    """Every AppConfig subclass defined anywhere in the package."""
    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(
        fisseq_data_pipeline.__path__, prefix="fisseq_data_pipeline."
    ):
        module = importlib.import_module(info.name)
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, AppConfig)
                and dataclasses.is_dataclass(obj)
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return sorted(found.values(), key=lambda c: (c.__module__, c.__qualname__))


def test_discovery_finds_the_stage_configs():
    """Guards the guard: an import failure must not silently empty the sweep."""
    names = {c.__name__ for c in _stage_config_classes()}
    assert {"OvwtConfig", "QcFilterConfig", "GlobalOvwtConfig"} <= names


@pytest.mark.parametrize(
    "cfg_cls", _stage_config_classes(), ids=lambda c: f"{c.__module__}.{c.__name__}"
)
def test_no_stage_local_seed_field(cfg_cls):
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    offenders = fields & _BANNED_SEED_FIELDS
    assert not offenders, (
        f"{cfg_cls.__module__}.{cfg_cls.__name__} declares stage-local seed "
        f"field(s) {sorted(offenders)}. Use AppConfig.random_seed instead -- "
        f"derive a fixed offset from it if this stage must differ from its "
        f"siblings."
    )


@pytest.mark.parametrize(
    "cfg_cls", _stage_config_classes(), ids=lambda c: f"{c.__module__}.{c.__name__}"
)
def test_inherits_random_seed(cfg_cls):
    fields = {f.name for f in dataclasses.fields(cfg_cls)}
    assert "random_seed" in fields


def test_app_config_random_seed_default_is_zero():
    assert AppConfig.random_seed == 0
