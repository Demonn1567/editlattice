"""Backend settings: YAML config, overridable by environment variables on cloud jobs."""

from __future__ import annotations

import os
from typing import Any


def backend_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return {name, base_url, model}; EDITLATTICE_BASE_URL / EDITLATTICE_MODEL override."""
    settings = dict(config["backend"])
    if os.environ.get("EDITLATTICE_BASE_URL"):
        settings["base_url"] = os.environ["EDITLATTICE_BASE_URL"]
        settings["name"] = "vllm-kaggle"
    if os.environ.get("EDITLATTICE_MODEL"):
        settings["model"] = os.environ["EDITLATTICE_MODEL"]
    return settings


def proposer_settings(config: dict[str, Any]) -> dict[str, Any] | None:
    """Settings of a separate proposer model, or None if the proposer shares the solver's backend.

    Read from ``proposer_backend`` in the config; EDITLATTICE_PROPOSER_BASE_URL /
    EDITLATTICE_PROPOSER_MODEL override it on cloud jobs."""
    if "proposer_backend" not in config:
        return None
    settings = dict(config["proposer_backend"])
    if os.environ.get("EDITLATTICE_PROPOSER_BASE_URL"):
        settings["base_url"] = os.environ["EDITLATTICE_PROPOSER_BASE_URL"]
        settings["name"] = "vllm-kaggle"
    if os.environ.get("EDITLATTICE_PROPOSER_MODEL"):
        settings["model"] = os.environ["EDITLATTICE_PROPOSER_MODEL"]
    return settings
