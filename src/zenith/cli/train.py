"""Hydra training entrypoint — ``python -m zenith.cli.train``.

Composes a run from the ``configs/`` tree and trains a decoder language model.

Examples
--------
    python -m zenith.cli.train
    python -m zenith.cli.train training.epochs=50 model.embed_dim=384
    python -m zenith.cli.train -m training.learning_rate=1e-3,3e-4,1e-4   # sweep
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from ..data import CausalLMDataset, load_corpus_file, train_val_split
from ..models import DecoderConfig, DecoderLM
from ..tokenizers import ByteTokenizer
from ..training import CausalLMTrainer, TrainingConfig


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer = ByteTokenizer()

    ids = load_corpus_file(cfg.data.corpus_path, tokenizer)
    train_ids, val_ids = train_val_split(ids, cfg.data.val_fraction)

    block_size = int(cfg.model.block_size)
    train_dataset = CausalLMDataset(train_ids, block_size)
    val_dataset = (
        CausalLMDataset(val_ids, block_size) if val_ids.numel() > block_size + 1 else None
    )

    model = DecoderLM(
        DecoderConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=block_size,
            embed_dim=int(cfg.model.embed_dim),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            ff_dim=int(cfg.model.ff_dim),
            dropout=float(cfg.model.dropout),
        )
    )

    training_config = TrainingConfig(
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.batch_size),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        warmup_steps=int(cfg.training.warmup_steps),
        grad_clip=float(cfg.training.grad_clip),
        seed=int(cfg.seed),
        sample_prompt=str(cfg.training.sample_prompt),
        sample_tokens=int(cfg.training.sample_tokens),
        sample_temperature=float(cfg.training.sample_temperature),
        save_path=str(cfg.training.save_path),
        tracking_enabled=bool(cfg.tracking.enabled),
        experiment=str(cfg.tracking.experiment),
        tracking_uri=cfg.tracking.tracking_uri,
        run_name=cfg.tracking.run_name,
    )

    CausalLMTrainer(model, tokenizer, training_config).fit(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
