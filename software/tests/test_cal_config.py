"""Host unit tests for riscq.cal.Config: slash-path get/set/missing, YAML round-trip, deep copy."""

import pytest

from riscq.cal import Config


def test_get_set_slash_paths():
    cfg = Config()
    cfg["readout/0/freq"] = 5_000_000
    cfg["readout/0/amp"] = 0.3
    cfg["qubit/2/amp"] = 0.9
    assert cfg["readout/0/freq"] == 5_000_000
    assert cfg["readout/0/amp"] == 0.3
    assert cfg["qubit/2/amp"] == 0.9
    # intermediate nodes were created as nested dicts
    assert cfg["readout/0"] == {"freq": 5_000_000, "amp": 0.3}


def test_missing_key_raises_and_contains_get():
    cfg = Config({"readout": {"0": {"freq": 1}}})
    assert "readout/0/freq" in cfg
    assert "readout/1/freq" not in cfg
    assert cfg.get("readout/1/freq", 42) == 42
    with pytest.raises(KeyError):
        cfg["readout/1/freq"]
    with pytest.raises(KeyError):
        cfg[""]


def test_overwrite_scalar_with_subtree():
    cfg = Config()
    cfg["a/b"] = 1
    cfg["a/b/c"] = 2                     # b was a scalar; setting deeper replaces it with a dict
    assert cfg["a/b/c"] == 2


def test_yaml_round_trip(tmp_path):
    cfg = Config()
    cfg["readout/0/freq"] = 5_000_000
    cfg["qubit/0/x90/amp"] = 0.42
    cfg["qubit/0/x90/phase"] = 0.0
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    back = Config.load(path)
    assert back.to_dict() == cfg.to_dict()
    assert back["qubit/0/x90/amp"] == 0.42


def test_copy_is_deep():
    cfg = Config({"a": {"b": 1}})
    dup = cfg.copy()
    dup["a/b"] = 99
    assert cfg["a/b"] == 1               # original untouched
    assert dup["a/b"] == 99
