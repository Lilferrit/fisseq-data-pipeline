import io
import pathlib

import pytest
import yaml

import fisseq_data_pipeline.utils.config as cfg_mod
from fisseq_data_pipeline.utils import Config


@pytest.fixture()
def temp_default_cfg(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """
    Create a temporary default config file and point DEFAULT_CFG_PATH to it.
    Returns the default data dict.
    """
    default_data = {"alpha": 1, "beta": "two", "gamma": True}
    default_path = tmp_path / "config.yaml"
    default_path.write_text(yaml.safe_dump(default_data))
    monkeypatch.setattr(cfg_mod, "DEFAULT_CFG_PATH", default_path, raising=True)
    return default_data, default_path


def test_init_none_uses_default_and_logs_info(temp_default_cfg, caplog):
    default_data, default_path = temp_default_cfg

    caplog.clear()
    with caplog.at_level("INFO"):
        c = Config(None)

    # It should log that no config was provided
    assert any("No config provided" in rec.getMessage() for rec in caplog.records)

    # Should load defaults
    assert c.alpha == default_data["alpha"]
    assert c["beta"] == default_data["beta"]
    assert c.gamma is True


def test_init_with_dict_verifies_and_mutates_in_place(temp_default_cfg, caplog):
    default_data, _ = temp_default_cfg
    provided = {
        "alpha": 10,
        "extra": 999,
    }  # 'extra' is invalid; 'beta' and 'gamma' missing

    caplog.clear()
    with caplog.at_level("WARNING"):
        c = Config(provided)

    # Invalid key removed; missing keys filled from defaults
    assert "extra" not in c._data
    assert c.alpha == 10
    assert c.beta == default_data["beta"]
    assert c.gamma == default_data["gamma"]

    # Verify warnings emitted for both the removal and missing fills
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Removing invalid config option extra" in m for m in msgs)
    # One warning per missing key
    assert any("Key beta not in provided config using default value" in m for m in msgs)
    assert any(
        "Key gamma not in provided config using default value" in m for m in msgs
    )

    # The provided dict is mutated in place by _verify_config
    assert "extra" not in provided
    assert provided["beta"] == default_data["beta"]
    assert provided["gamma"] == default_data["gamma"]


def test_init_with_path_reads_and_verifies(temp_default_cfg, tmp_path, caplog):
    default_data, _ = temp_default_cfg
    user_cfg_path = tmp_path / "user.yaml"
    user_cfg_path.write_text(yaml.safe_dump({"beta": "override", "junk": 123}))

    caplog.clear()
    with caplog.at_level("WARNING"):
        c = Config(user_cfg_path)

    # beta overridden, alpha/gamma filled from default, junk removed
    assert c.beta == "override"
    assert c.alpha == default_data["alpha"]
    assert c.gamma == default_data["gamma"]
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Removing invalid config option junk" in m for m in msgs)


def test_init_with_config_instance_copies_data_without_reverify(
    temp_default_cfg, monkeypatch, caplog
):
    # Build a Config from a dict first
    base = Config({"alpha": 7, "beta": "b", "gamma": False})
    caplog.clear()
    with caplog.at_level("WARNING"):
        # Now pass the Config instance to the constructor
        c2 = Config(base)

    # Data should match exactly (no changes)
    assert c2.alpha == 7
    assert c2.beta == "b"
    assert c2.gamma is False

    # No re-verification warnings should have been emitted
    assert not any(
        "Removing invalid config option" in r.getMessage() for r in caplog.records
    )
    assert not any("not in provided config" in r.getMessage() for r in caplog.records)


def test_attr_and_item_access(temp_default_cfg):
    _defaults, _ = temp_default_cfg
    c = Config({"alpha": 42, "beta": "bee", "gamma": True})

    # __getattr__
    assert c.alpha == 42
    assert c.beta == "bee"

    # __getitem__
    assert c["alpha"] == 42
    assert c["beta"] == "bee"
    assert c["gamma"] is True
