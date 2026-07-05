# Architecture Decision Records

Short records of the significant, hard-to-reverse decisions behind Zenith, and the
reasoning at the time. New major decisions get a new numbered ADR rather than
silently changing course.

| # | Decision |
|---|----------|
| [0001](0001-generative-identity.md) | Zenith is a generative (decoder) library; Polaris is an optional sibling, not a dependency |
| [0002](0002-byte-level-tokenizer.md) | Ship a byte-level tokenizer first; defer learned BPE |
| [0003](0003-kv-cache-and-decoding.md) | KV-cache is an opt-in path (training unchanged); decoding strategies live in the Generator |
| [0004](0004-scaling-opt-in.md) | Scaling (LoRA/AMP/accumulation/DDP) is opt-in and default-off; QLoRA/FSDP deferred |
| [0005](0005-run-records-and-reproducibility.md) | On-disk run records complement MLflow; determinism is best-effort; samples logged each epoch |
| [0006](0006-serving-and-streaming.md) | Serving is a thin layer over the Generator; streaming is SSE with UTF-8-complete chunks |
