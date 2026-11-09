"""Backend settings for a solver and a separate proposer model."""

from __future__ import annotations

from editlattice.llm.settings import backend_settings, proposer_settings

CONFIG = {
    "backend": {"name": "vllm-local", "base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3-4B"},
    "proposer_backend": {"name": "vllm-local", "base_url": "http://localhost:8100/v1", "model": "Qwen/Qwen3-14B-AWQ"},
}


def test_proposer_shares_the_solver_backend_by_default() -> None:
    assert proposer_settings({"backend": CONFIG["backend"]}) is None


def test_proposer_backend_is_read_from_the_config(monkeypatch) -> None:
    monkeypatch.delenv("EDITLATTICE_PROPOSER_BASE_URL", raising=False)
    monkeypatch.delenv("EDITLATTICE_PROPOSER_MODEL", raising=False)
    settings = proposer_settings(CONFIG)
    assert settings["model"] == "Qwen/Qwen3-14B-AWQ"
    assert settings["base_url"].endswith(":8100/v1")


def test_environment_overrides_solver_and_proposer_separately(monkeypatch) -> None:
    monkeypatch.setenv("EDITLATTICE_BASE_URL", "http://a:8000/v1")
    monkeypatch.setenv("EDITLATTICE_MODEL", "solver")
    monkeypatch.setenv("EDITLATTICE_PROPOSER_BASE_URL", "http://b:8100/v1")
    monkeypatch.setenv("EDITLATTICE_PROPOSER_MODEL", "proposer")
    assert backend_settings(CONFIG)["model"] == "solver"
    assert proposer_settings(CONFIG)["model"] == "proposer"
    assert proposer_settings(CONFIG)["base_url"] == "http://b:8100/v1"
