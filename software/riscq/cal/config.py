"""Config — the qcal-style YAML parameter tree the calibrations read and update (spec 06).

A nested dict addressed by slash paths: `cfg["readout/0/freq"]` walks `data["readout"]["0"]["freq"]`;
assignment creates the intermediate dicts. Keys are strings (path segments), so a YAML round-trip
preserves them. Minimal on purpose — the calibrations only need get / set / a working copy.
"""

from __future__ import annotations

import copy

import yaml


class Config:
    def __init__(self, data: dict | None = None):
        self._data: dict = copy.deepcopy(data) if data else {}

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f) or {})

    def save(self, path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=True)

    def copy(self) -> "Config":
        return Config(self._data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    @staticmethod
    def _keys(path) -> list[str]:
        keys = [k for k in str(path).split("/") if k != ""]
        if not keys:
            raise KeyError(f"empty config path {path!r}")
        return keys

    def __getitem__(self, path):
        node = self._data
        for k in self._keys(path):
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"no config entry {path!r} (missing {k!r})")
            node = node[k]
        return node

    def __setitem__(self, path, value) -> None:
        keys = self._keys(path)
        node = self._data
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    def __contains__(self, path) -> bool:
        try:
            self[path]
            return True
        except KeyError:
            return False

    def get(self, path, default=None):
        try:
            return self[path]
        except KeyError:
            return default
