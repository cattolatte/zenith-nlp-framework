"""Speculative-decoding demo — measure acceptance rate and target-forward savings.

Greedy speculative decoding produces byte-for-byte the same text as greedy decoding
on the target, using a small draft model to cut the number of target forward passes.

    python scripts/speculative_demo.py --target shakespeare-llama.pt \
        --draft draft-shakespeare.pt --prompt "KING RICHARD III:"

Both checkpoints must share a vocabulary and context length.
"""

from __future__ import annotations

import argparse

import torch

from zenith.checkpoint import load_checkpoint
from zenith.generation import Generator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompt", default="KING RICHARD III:")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--lookahead", type=int, default=4)
    args = ap.parse_args()

    target, tok = load_checkpoint(args.target)
    draft, _ = load_checkpoint(args.draft)
    gen = Generator(target, tok)

    prompt_ids = tok.encode(args.prompt) or [tok.bos_id]
    x = torch.tensor([prompt_ids], dtype=torch.long)

    greedy = gen.generate_ids(x, max_new_tokens=args.tokens, temperature=0.0)
    spec, stats = gen.speculative_generate_ids(
        x, draft, max_new_tokens=args.tokens, lookahead=args.lookahead
    )
    identical = torch.equal(greedy[:, : spec.size(1)], spec)

    print(f"target {target.num_parameters() / 1e6:.1f}M  |  "
          f"draft {draft.num_parameters() / 1e6:.1f}M  |  lookahead {args.lookahead}")
    print(f"identical to greedy : {identical}")
    print(f"tokens generated    : {stats.tokens}")
    print(f"target forwards     : {stats.target_forwards}  (greedy needs {stats.tokens})")
    print(f"acceptance rate     : {stats.acceptance_rate:.1%}")
    print(f"speedup (fwds saved): {stats.speedup:.2f}x")
    print("--- sample ---")
    print(args.prompt + tok.decode(spec[0, len(prompt_ids):].tolist()))


if __name__ == "__main__":
    main()
