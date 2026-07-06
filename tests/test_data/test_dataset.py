"""Tests for causal-LM dataset and corpus helpers."""

import torch
import pytest

from zenith.data import CausalLMDataset, encode_corpus, train_val_split
from zenith.tokenizers import ByteTokenizer


def test_getitem_is_input_shifted_by_one():
    ids = torch.arange(10)
    ds = CausalLMDataset(ids, block_size=4)
    x, y = ds[0]
    assert torch.equal(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4]))


def test_length_accounts_for_block_size():
    ds = CausalLMDataset(torch.arange(10), block_size=4)
    assert len(ds) == 6


def test_stride_yields_fewer_spaced_windows():
    ds = CausalLMDataset(torch.arange(20), block_size=4, stride=4)
    starts = [ds[i][0][0].item() for i in range(len(ds))]
    assert starts == [0, 4, 8, 12]  # non-overlapping window starts
    # default stride=1 is unchanged (backward compatible)
    assert len(CausalLMDataset(torch.arange(10), block_size=4)) == 6


def test_rejects_zero_stride():
    with pytest.raises(ValueError):
        CausalLMDataset(torch.arange(10), block_size=4, stride=0)


def test_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        CausalLMDataset(torch.arange(3), block_size=4)


def test_encode_corpus_returns_long_tensor():
    ids = encode_corpus("hello", ByteTokenizer())
    assert ids.dtype == torch.long and ids.numel() == 5


def test_train_val_split_is_contiguous():
    ids = torch.arange(100)
    train, val = train_val_split(ids, val_fraction=0.2)
    assert train.numel() == 80 and val.numel() == 20
    assert torch.equal(val, torch.arange(80, 100))
