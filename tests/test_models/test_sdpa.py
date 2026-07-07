"""The SDPA fast path must be numerically equivalent to the eager attention."""

import copy

import torch

from zenith.models import DecoderConfig, DecoderLM, KVCache


def _pair() -> tuple[DecoderLM, DecoderLM]:
    """Eager and SDPA models sharing identical weights."""
    torch.manual_seed(0)
    cfg = DecoderConfig(vocab_size=259, block_size=32, embed_dim=48, num_layers=3,
                        num_heads=4, ff_dim=96, dropout=0.0)
    eager = DecoderLM(cfg).eval()
    sdpa = DecoderLM(DecoderConfig(**{**cfg.__dict__, "attention": "sdpa"})).eval()
    sdpa.load_state_dict(copy.deepcopy(eager.state_dict()))
    return eager, sdpa


def test_sdpa_matches_eager_parallel_forward():
    eager, sdpa = _pair()
    x = torch.randint(0, 259, (2, 20))
    with torch.no_grad():
        assert torch.allclose(eager(x), sdpa(x), atol=1e-5)


def test_sdpa_matches_eager_with_kv_cache_single_step():
    eager, sdpa = _pair()
    x = torch.randint(0, 259, (1, 10))
    with torch.no_grad():
        ce, cs = KVCache(eager.config.num_layers), KVCache(sdpa.config.num_layers)
        eager(x, cache=ce)
        sdpa(x, cache=cs)
        nxt = torch.randint(0, 259, (1, 1))
        assert torch.allclose(eager(nxt, cache=ce), sdpa(nxt, cache=cs), atol=1e-5)


def test_sdpa_matches_eager_multi_token_cache_step():
    """The speculative-verify path: several query tokens against a primed cache."""
    eager, sdpa = _pair()
    x = torch.randint(0, 259, (1, 8))
    with torch.no_grad():
        ce, cs = KVCache(eager.config.num_layers), KVCache(sdpa.config.num_layers)
        eager(x, cache=ce)
        sdpa(x, cache=cs)
        chunk = torch.randint(0, 259, (1, 4))
        assert torch.allclose(eager(chunk, cache=ce), sdpa(chunk, cache=cs), atol=1e-5)


def test_sdpa_generation_matches_eager():
    from zenith.generation import Generator
    from zenith.tokenizers import ByteTokenizer

    eager, sdpa = _pair()
    prompt = torch.tensor([[1, 5, 9, 13]])
    ge = Generator(eager, ByteTokenizer()).generate_ids(prompt, max_new_tokens=20, temperature=0.0)
    gs = Generator(sdpa, ByteTokenizer()).generate_ids(prompt, max_new_tokens=20, temperature=0.0)
    assert torch.equal(ge, gs)
