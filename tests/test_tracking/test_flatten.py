"""Tests for the MLflow tracker's config flattening (no mlflow import needed)."""

from zenith.tracking import flatten_config


def test_flatten_nested_dict():
    cfg = {"model": {"name": "polaris_encoder", "layers": 4}, "seed": 0}
    flat = flatten_config(cfg)
    assert flat == {"model.name": "polaris_encoder", "model.layers": 4, "seed": 0}


def test_flatten_handles_lists():
    cfg = {"peft": {"target_modules": ["attention", "head"]}}
    flat = flatten_config(cfg)
    assert flat == {"peft.target_modules.0": "attention", "peft.target_modules.1": "head"}


def test_flatten_empty():
    assert flatten_config({}) == {}
