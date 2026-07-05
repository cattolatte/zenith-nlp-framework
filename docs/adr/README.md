# Architecture Decision Records

Short records of the significant, hard-to-reverse decisions behind Zenith, and the
reasoning at the time. New major decisions get a new numbered ADR rather than
silently changing course.

| # | Decision |
|---|----------|
| [0001](0001-generative-identity.md) | Zenith is a generative (decoder) library; Polaris is an optional sibling, not a dependency |
| [0002](0002-byte-level-tokenizer.md) | Ship a byte-level tokenizer first; defer learned BPE |
