"""Tests for the interactive console's command methods (no cmdloop)."""

import random

from zenith.console.banner import render_banner
from zenith.console.repl import ZenithConsole, _parse_value
from zenith.generation import Generator
from zenith.models import DecoderConfig, DecoderLM
from zenith.tokenizers import ByteTokenizer


def _console_with_model() -> ZenithConsole:
    console = ZenithConsole()
    model = DecoderLM(
        DecoderConfig(vocab_size=259, block_size=16, embed_dim=32, num_layers=2,
                      num_heads=2, ff_dim=64, dropout=0.0)
    )
    console._generator = Generator(model, ByteTokenizer())
    return console


def test_parse_value_types():
    assert _parse_value("temperature", "0.7") == 0.7
    assert _parse_value("top_k", "40") == 40
    assert _parse_value("top_p", "none") is None


def test_set_updates_and_disables_settings():
    console = ZenithConsole()
    console.do_set("temperature 0.5")
    assert console.settings["temperature"] == 0.5
    console.do_set("top_k none")
    assert console.settings["top_k"] is None


def test_set_rejects_unknown_param(capsys):
    console = ZenithConsole()
    console.do_set("bogus 1")
    assert "unknown param" in capsys.readouterr().out


def test_generate_without_model_warns(capsys):
    ZenithConsole().default("hello there")
    assert "no model loaded" in capsys.readouterr().out


def test_generate_with_model_produces_output(capsys):
    console = _console_with_model()
    console.settings["max_new_tokens"] = 6
    console.default("hi")
    out = capsys.readouterr().out
    assert "no model loaded" not in out  # a model is loaded, so it generated


def test_exit_returns_true():
    assert ZenithConsole().do_exit("") is True


def test_banner_respects_no_color():
    plain = render_banner(random.Random(0), color=False)
    colored = render_banner(random.Random(0), color=True)
    assert "\033[" not in plain and "generative NLP library" in plain
    assert "\033[" in colored
