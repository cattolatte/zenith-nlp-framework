"""Scaling study — bits/char vs model size on tiny-shakespeare.

Trains a sweep of Llama-style decoders (same recipe, only width/depth change),
then reports held-out bits/char for each. Produces the figure in BENCHMARKS.md.

    python scripts/scaling_study.py

Runs on whatever device Zenith picks (MPS/CUDA/CPU). ~30 min on an M-series Mac.
"""

from __future__ import annotations

import math
from pathlib import Path

from zenith.checkpoint import load_checkpoint
from zenith.data import CausalLMDataset, encode_corpus, train_val_split
from zenith.evaluation import evaluate
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer
from zenith.training import CausalLMTrainer, TrainingConfig

# (embed_dim, num_layers, num_heads, ff_dim) — a ~15x span of parameters.
SIZES = [
    dict(embed_dim=128, num_layers=3, num_heads=4, ff_dim=320),
    dict(embed_dim=192, num_layers=4, num_heads=6, ff_dim=512),
    dict(embed_dim=288, num_layers=5, num_heads=6, ff_dim=768),
    dict(embed_dim=384, num_layers=6, num_heads=6, ff_dim=1024),
]
BLOCK = 256
EPOCHS = 12
CORPUS = "data/tiny_shakespeare.txt"


def main() -> None:
    tokenizer = ByteTokenizer()
    text = Path(CORPUS).read_text(encoding="utf-8")
    ids = encode_corpus(text, tokenizer)
    train_ids, val_ids = train_val_split(ids, 0.1)
    train_ds = CausalLMDataset(train_ids, BLOCK, stride=128)
    val_ds = CausalLMDataset(val_ids, BLOCK, stride=BLOCK)

    results = []
    for size in SIZES:
        config = DecoderConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK, dropout=0.2, **size)
        model = DecoderLM(config)
        params = model.num_parameters()
        save_path = f"scale-{params}.pt"
        training = TrainingConfig(
            epochs=EPOCHS, batch_size=64, learning_rate=1e-3, warmup_steps=200,
            log_samples=False, record_dir=None, save_path=save_path, tracking_enabled=False,
        )
        CausalLMTrainer(model, tokenizer, training).fit(train_ds, val_ds)
        best, _ = load_checkpoint(save_path)  # best-by-val checkpoint
        metrics = evaluate(best, val_ds)
        bpc = metrics["loss"] / math.log(2)
        results.append((params, bpc))
        print(f"SCALE  params={params / 1e6:5.2f}M  bits/char={bpc:.3f}", flush=True)

    print("\n=== SCALING RESULTS (params -> bits/char) ===")
    for params, bpc in results:
        print(f"{params / 1e6:5.2f}M  {bpc:.3f}")


if __name__ == "__main__":
    main()
