"""Acceptance test — Meridian's Phase-7 usage sketch from zenith.md (section 8).

If this both runs offline and type-checks under ``mypy --strict``, the v1.1.0
contract Meridian consumes is satisfied. The imports and calls mirror the sketch;
only concrete tiny objects and a skipped training step differ (the trainer already
exists and is out of scope here).
"""

import torch

from zenith.generation import Generator
from zenith.generation.constraints import AllowedTokens
from zenith.instruct.grounded import GroundedInstructionDataset, GroundedTemplate
from zenith.models import DecoderConfig, DecoderLM
from zenith.peft.lora import LoraConfig, inject_lora
from zenith.tokenizers import ByteTokenizer


def test_meridian_phase7_usage() -> None:
    tokenizer = ByteTokenizer()
    examples = [
        ("What does aspirin do?", [("p1", "Aspirin reduces fever.")], "It reduces fever [p1]."),
        ("What is the capital of Mars?", [("p1", "Aspirin reduces fever.")], None),
    ]

    # grounded LoRA SFT
    dataset = GroundedInstructionDataset(
        examples, tokenizer, max_length=128, template=GroundedTemplate()
    )
    base_decoder = DecoderLM(
        DecoderConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=128,
            embed_dim=32,
            num_layers=2,
            num_heads=2,
            ff_dim=64,
            dropout=0.0,
        )
    )
    model = inject_lora(base_decoder, LoraConfig(rank=8))  # stays a DecoderLM
    # ... train on `dataset` with the existing trainer ...
    assert len(dataset) == 2

    # citation-constrained, abstain-aware decoding
    gen = Generator(model, tokenizer)
    citation_open_ids: set[int] = {ord("[")}
    valid_passage_ids: set[int] = {ord("p"), ord("1")}
    prompt_ids = torch.tensor([tokenizer.encode("### Answer:\n")])
    answer = gen.generate_ids(
        prompt_ids,
        max_new_tokens=8,
        logits_constraint=AllowedTokens(
            trigger_ids=citation_open_ids, allowed_ids=valid_passage_ids
        ),
        stop_ids={tokenizer.abstain_id},
    )
    assert answer.shape[1] >= prompt_ids.shape[1]
