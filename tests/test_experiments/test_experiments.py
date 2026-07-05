"""Tests for environment capture, run recording, and deterministic seeding."""

import json

import torch

from zenith.experiments import capture_environment, record_run
from zenith.utils import set_deterministic


def test_capture_environment_has_expected_fields():
    env = capture_environment(seed=7)
    assert env["seed"] == 7
    assert "python_version" in env
    assert "torch_version" in env  # torch is installed in the test env


def test_record_run_writes_all_files(tmp_path):
    path = record_run(
        tmp_path / "run",
        config={"epochs": 3, "lr": 0.001},
        history=[{"train_loss": 2.0}, {"train_loss": 1.0}],
        samples=["hello", "world"],
        environment={"seed": 0},
    )
    for name in ("config.json", "metrics.json", "samples.txt", "environment.json", "run.json"):
        assert (path / name).exists()

    run = json.loads((path / "run.json").read_text())
    assert run["num_samples"] == 2
    assert run["config"] == {"epochs": 3, "lr": 0.001}
    assert len(run["metrics"]["history"]) == 2


def test_record_run_omits_optional_files(tmp_path):
    path = record_run(tmp_path / "run", config={}, history=[])
    assert (path / "run.json").exists()
    assert not (path / "samples.txt").exists()
    assert not (path / "environment.json").exists()


def test_set_deterministic_runs_and_resets():
    set_deterministic(0)
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)  # reset global state for other tests
