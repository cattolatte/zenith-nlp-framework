"""Grounded SFT: passages in the prompt, a cited answer or <abstain> in the target."""

import torch

from zenith.instruct import GroundedInstructionDataset, GroundedTemplate, InstructionDataset
from zenith.instruct.dataset import IGNORE_INDEX
from zenith.tokenizers import ByteTokenizer

PASSAGES = [("p1", "Aspirin reduces fever."), ("p2", "It also relieves pain.")]


def test_prompt_contains_passages_ids_and_question():
    prompt = GroundedTemplate().format_prompt("What does aspirin do?", PASSAGES)
    assert "p1" in prompt and "Aspirin reduces fever." in prompt
    assert "p2" in prompt and "What does aspirin do?" in prompt


def test_response_is_supervised_and_prompt_is_masked():
    tok = ByteTokenizer()
    answer = "Aspirin reduces fever [p1]."
    ds = GroundedInstructionDataset([("Q?", PASSAGES, answer)], tok, max_length=256)
    inp, target = ds[0]
    assert inp.shape == target.shape
    supervised = target[target != IGNORE_INDEX]
    assert torch.equal(supervised, torch.tensor(tok.encode(answer) + [tok.eos_id]))
    prompt_len = len(tok.encode(GroundedTemplate().format_prompt("Q?", PASSAGES)))
    assert (target[: prompt_len - 1] == IGNORE_INDEX).all()


def test_abstain_example_targets_the_abstain_token():
    tok = ByteTokenizer()
    ds = GroundedInstructionDataset([("Unanswerable?", PASSAGES, None)], tok, max_length=256)
    _, target = ds[0]
    supervised = target[target != IGNORE_INDEX]
    assert torch.equal(supervised, torch.tensor([tok.abstain_id, tok.eos_id]))


def test_fixed_length_and_deterministic():
    tok = ByteTokenizer()
    examples = [("Q?", PASSAGES, "A [p1]."), ("U?", PASSAGES, None)]
    a = GroundedInstructionDataset(examples, tok, max_length=128)
    b = GroundedInstructionDataset(examples, tok, max_length=128)
    for i in range(len(examples)):
        assert a[i][0].numel() == 127 and a[i][1].numel() == 127
        assert torch.equal(a[i][0], b[i][0]) and torch.equal(a[i][1], b[i][1])


def test_grounded_is_an_instruction_dataset():
    tok = ByteTokenizer()
    ds = GroundedInstructionDataset([("Q?", PASSAGES, "A.")], tok, max_length=64)
    assert isinstance(ds, InstructionDataset)  # reuses the base machinery
    assert len(ds) == 1
