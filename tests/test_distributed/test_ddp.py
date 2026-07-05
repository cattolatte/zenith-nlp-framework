"""Tests for the distributed helpers in the single-process (default) case."""

import torch
import torch.nn as nn

from zenith import distributed as dist


def test_single_process_is_not_distributed():
    assert dist.world_size() == 1
    assert not dist.is_distributed()
    assert dist.is_main_process()


def test_wrap_and_unwrap_are_identity_when_not_distributed():
    model = nn.Linear(4, 4)
    assert dist.wrap_model(model) is model
    assert dist.unwrap_model(model) is model


def test_sampler_is_none_when_not_distributed():
    assert dist.make_sampler([1, 2, 3]) is None


def test_resolve_device_returns_a_device():
    assert isinstance(dist.resolve_device(), torch.device)


def test_setup_and_cleanup_are_noops_single_process():
    dist.setup()
    dist.cleanup()
