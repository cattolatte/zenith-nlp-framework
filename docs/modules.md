# Module overview

One-way dependency direction: `utils` → `tokenizers` / `models` / `data` →
`generation` / `evaluation` → `training` → `serving` / `cli`, with `checkpoint`
tying model + tokenizer together. Everything is importable without heavy optional
deps until you use the module that needs them.

| Module | Purpose | Public surface |
| --- | --- | --- |
| `zenith.models` | From-scratch decoder-only transformer (causal attention, pre-norm blocks, tied embeddings) + KV-cache. | `DecoderLM`, `DecoderConfig`, `KVCache` |
| `zenith.tokenizers` | Byte-level tokenizer and a from-scratch byte-level BPE. | `ByteTokenizer`, `BPETokenizer` |
| `zenith.data` | Causal-LM dataset (next-token blocks) + corpus helpers. | `CausalLMDataset`, `encode_corpus`, `load_corpus_file`, `train_val_split` |
| `zenith.generation` | Decoding: greedy, temperature, top-k, nucleus, beam, streaming. | `Generator` |
| `zenith.training` | Single-loop trainer (warmup/cosine, AMP, grad accumulation, LoRA, DDP). | `CausalLMTrainer`, `TrainingConfig` |
| `zenith.evaluation` | Held-out loss and perplexity. | `evaluate`, `perplexity` |
| `zenith.peft` | LoRA adapters injected into `nn.Linear`. | `LoraConfig`, `inject_lora`, `LoRALinear`, … |
| `zenith.distributed` | `torchrun`-native DDP helpers (no-ops single-process). | `wrap_model`, `make_sampler`, `is_main_process`, … |
| `zenith.tracking` | Optional MLflow experiment tracking. | `MlflowTracker`, `get_tracker`, `flatten_config` |
| `zenith.experiments` | Environment capture + on-disk run records. | `capture_environment`, `record_run` |
| `zenith.serving` | FastAPI generation service (+ SSE streaming). | `create_app`, `serve` |
| `zenith.console` | Interactive `zenith >` REPL (banner + tunable decoding). | `ZenithConsole`, `run` |
| `zenith.checkpoint` | Self-describing save/load (model + tokenizer, LoRA-aware). | `save_checkpoint`, `load_checkpoint`, `load_pretrained` |
| `zenith.cli` | `zenith` CLI (`generate`, `chat`, `console`, `serve`, `eval`, `info`) + Hydra `train`. | `app` |

`zenith._incubating` is a frozen snapshot of the project's earlier general-NLP
code; it is not part of the supported surface (see its README).
