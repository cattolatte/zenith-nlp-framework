"""Instruction fine-tune a base checkpoint into a mini chat model.

Supervised fine-tuning with response-only loss masking (the prompt is not
supervised — see zenith.instruct.InstructionDataset). Small data + a small model, so
this memorises more than it generalises; that is the honest scope (a demo of the SFT
pipeline, not a capable assistant).

    python scripts/finetune_instruct.py --base shakespeare-llama.pt \
        --data data/instructions.jsonl --out instruct-shakespeare.pt --epochs 50
"""

from __future__ import annotations

import argparse

from zenith.checkpoint import load_checkpoint
from zenith.instruct import InstructionDataset, load_instructions
from zenith.training import CausalLMTrainer, TrainingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base checkpoint to fine-tune.")
    ap.add_argument("--data", default="data/instructions.jsonl")
    ap.add_argument("--out", default="instruct-model.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    model, tokenizer = load_checkpoint(args.base)
    pairs = load_instructions(args.data)
    dataset = InstructionDataset(pairs, tokenizer, max_length=args.max_length)
    print(f"base {model.num_parameters() / 1e6:.1f}M  |  {len(dataset)} instruction pairs", flush=True)

    cfg = TrainingConfig(
        epochs=args.epochs, batch_size=16, learning_rate=args.lr, warmup_steps=50,
        log_samples=False, record_dir=None, save_path=args.out, tracking_enabled=False,
    )
    CausalLMTrainer(model, tokenizer, cfg).fit(dataset, None)
    print("INSTRUCT DONE ->", args.out, flush=True)


if __name__ == "__main__":
    main()
