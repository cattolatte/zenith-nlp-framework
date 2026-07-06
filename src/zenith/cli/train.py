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
from omegaconf import DictConfig, OmegaConf

from ..data import CausalLMDataset, encode_corpus, train_val_split
from ..models import DecoderConfig, DecoderLM
from ..peft import LoraConfig
from ..tokenizers import BPETokenizer, ByteTokenizer
from ..training import CausalLMTrainer, TrainingConfig


def _build_tokenizer(tok_cfg, text: str):
    """Byte tokenizer, or a BPE tokenizer trained on the corpus."""
    if tok_cfg.get("type", "byte") == "bpe":
        return BPETokenizer().train([text], int(tok_cfg.get("vocab_size", 1024)))
    return ByteTokenizer()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from pathlib import Path

    text = Path(cfg.data.corpus_path).read_text(encoding="utf-8")
    tokenizer = _build_tokenizer(cfg.get("tokenizer", {}), text)
    ids = encode_corpus(text, tokenizer)
    train_ids, val_ids = train_val_split(ids, cfg.data.val_fraction)

    block_size = int(cfg.model.block_size)
    stride = int(cfg.data.get("stride", 1))
    train_dataset = CausalLMDataset(train_ids, block_size, stride=stride)
    # Evaluate on non-overlapping blocks (standard, fast).
    val_dataset = (
        CausalLMDataset(val_ids, block_size, stride=block_size)
        if val_ids.numel() > block_size + 1
        else None
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
            norm=str(cfg.model.get("norm", "rmsnorm")),
            positional=str(cfg.model.get("positional", "rope")),
            ffn=str(cfg.model.get("ffn", "swiglu")),
        )
    )

    peft = cfg.get("peft", {})
    lora = LoraConfig(
        rank=int(peft.get("rank", 8)),
        alpha=int(peft.get("alpha", 16)),
        dropout=float(peft.get("dropout", 0.0)),
        target_modules=tuple(peft.get("target_modules", ["attention"]) or ()),
    )

    training_config = TrainingConfig(
        epochs=int(cfg.training.epochs),
        batch_size=int(cfg.training.batch_size),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        warmup_steps=int(cfg.training.warmup_steps),
        grad_clip=float(cfg.training.grad_clip),
        seed=int(cfg.seed),
        grad_accum_steps=int(cfg.training.get("grad_accum_steps", 1)),
        amp=bool(cfg.training.get("amp", False)),
        amp_dtype=str(cfg.training.get("amp_dtype", "bf16")),
        use_lora=bool(peft.get("use_lora", False)),
        lora=lora,
        sample_prompt=str(cfg.training.sample_prompt),
        sample_tokens=int(cfg.training.sample_tokens),
        sample_temperature=float(cfg.training.sample_temperature),
        log_samples=bool(cfg.training.get("log_samples", True)),
        deterministic=bool(cfg.training.get("deterministic", False)),
        record_dir=cfg.training.get("record_dir", None),
        save_path=str(cfg.training.save_path),
        tracking_enabled=bool(cfg.tracking.enabled),
        experiment=str(cfg.tracking.experiment),
        tracking_uri=cfg.tracking.tracking_uri,
        run_name=cfg.tracking.run_name,
    )

    run_config = OmegaConf.to_container(cfg, resolve=True)
    trainer = CausalLMTrainer(model, tokenizer, training_config, run_config=run_config)
    trainer.fit(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
