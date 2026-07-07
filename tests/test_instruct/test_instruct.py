"""Instruction tuning: template, response-only loss masking, and EOS stopping."""

import json

import torch

from zenith.generation import Generator
from zenith.instruct import ChatTemplate, InstructionDataset, load_instructions
from zenith.instruct.dataset import IGNORE_INDEX
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _tok() -> ByteTokenizer:
    return ByteTokenizer()


# --- template -------------------------------------------------------------


def test_template_prompt_and_example():
    t = ChatTemplate()
    prompt = t.format_prompt("hi")
    assert prompt.endswith("### Response:\n")
    assert "hi" in prompt
    assert t.format_example("hi", "hello") == prompt + "hello"


# --- dataset masking (the crux of SFT) -----------------------------------


def test_only_response_tokens_are_supervised():
    tok = _tok()
    ds = InstructionDataset([("hi", "hello")], tok, max_length=64)
    inp, target = ds[0]
    assert inp.shape == target.shape  # fixed length, shifted pair

    supervised = target[target != IGNORE_INDEX]
    # The supervised tokens are exactly the response followed by EOS.
    expected = torch.tensor(tok.encode("hello") + [tok.eos_id])
    assert torch.equal(supervised, expected)


def test_prompt_and_padding_are_masked():
    tok = _tok()
    ds = InstructionDataset([("hi", "hello")], tok, max_length=64)
    _, target = ds[0]
    prompt_len = len(tok.encode(ChatTemplate().format_prompt("hi")))
    # Everything before the response, and all trailing padding, is ignored.
    assert (target[: prompt_len - 1] == IGNORE_INDEX).all()
    assert target[-1] == IGNORE_INDEX  # padding at the tail


def test_truncation_stays_within_max_length():
    tok = _tok()
    ds = InstructionDataset([("a very long instruction " * 20, "ok")], tok, max_length=32)
    inp, target = ds[0]
    assert inp.numel() == 31 and target.numel() == 31


def test_load_instructions(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"instruction": "hi", "response": "hello"}) + "\n\n"
        + json.dumps({"instruction": "bye", "response": "later"}) + "\n"
    )
    pairs = load_instructions(p)
    assert pairs == [("hi", "hello"), ("bye", "later")]


# --- EOS stopping ---------------------------------------------------------


def test_stop_ids_halts_generation():
    torch.manual_seed(0)
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=32, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    ).eval()
    gen = Generator(model, _tok())
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # Stopping on every possible token must yield nothing new.
    out = gen.generate_ids(x, max_new_tokens=20, temperature=0.0, stop_ids=range(259))
    assert out.size(1) == x.size(1)
