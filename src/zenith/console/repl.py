"""The ``zenith >`` interactive console.

A generation-focused REPL built on the stdlib ``cmd`` module (the same shape as
Metasploit's console): a prompt loop, ``do_<command>`` methods whose docstrings
become ``help``, and graceful handling of unknown input. The design principle:
*just type text and the model continues it* — commands are for loading a model and
tuning the decoding knobs.
"""

from __future__ import annotations

import cmd
from pathlib import Path
from typing import Any

__all__ = ["ZenithConsole", "run"]

_PARAMS = ("temperature", "top_k", "top_p", "repetition_penalty", "num_beams", "max_new_tokens")


class ZenithConsole(cmd.Cmd):
    """The interactive ``zenith >`` prompt.

    Load a checkpoint with ``load``, tune decoding with ``set``, then type any text
    to generate a continuation (streaming). ``help`` lists every command.
    """

    prompt = "zenith > "

    def __init__(self, intro: str | None = None) -> None:
        super().__init__()
        self.intro = intro
        self._generator: Any = None
        self.settings: dict[str, Any] = {
            "temperature": 0.8,
            "top_k": None,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "num_beams": None,
            "max_new_tokens": 200,
        }

    # -- commands -----------------------------------------------------------

    def do_load(self, arg: str) -> None:
        """load <path> — load a model checkpoint (.pt)."""
        path = arg.strip()
        if not path:
            print("usage: load <path-to-checkpoint>")
            return
        if not Path(path).is_file():
            print(f"no such file: {path}")
            return
        try:
            from ..checkpoint import load_pretrained

            self._generator = load_pretrained(path)
        except Exception as error:  # pragma: no cover - surfaced to the user
            print(f"could not load checkpoint: {error}")
            return
        print(f"loaded: {path}")

    def do_set(self, arg: str) -> None:
        """set <param> <value> — tune decoding. Params: temperature, top_k, top_p,
        repetition_penalty, num_beams, max_new_tokens. Use 'none' to disable a knob."""
        parts = arg.split()
        if len(parts) != 2:
            print("usage: set <param> <value>")
            return
        key, raw = parts
        if key not in _PARAMS:
            print(f"unknown param: {key!r} — one of: {', '.join(_PARAMS)}")
            return
        try:
            self.settings[key] = _parse_value(key, raw)
        except ValueError:
            print(f"invalid value for {key}: {raw!r}")
            return
        print(f"{key} = {self.settings[key]}")

    def do_show(self, _arg: str) -> None:
        """show — display the current settings and whether a model is loaded."""
        print(f"model: {'loaded' if self._generator is not None else 'none'}")
        for key, value in self.settings.items():
            print(f"  {key} = {value}")

    def do_generate(self, arg: str) -> None:
        """generate <prompt> — generate a continuation (or just type text directly)."""
        self._generate(arg)

    def do_info(self, _arg: str) -> None:
        """info — show Zenith / PyTorch versions."""
        from .. import __version__

        print(f"zenith {__version__}")
        try:
            import torch

            print(f"torch {torch.__version__}")
        except ImportError:  # pragma: no cover
            pass

    def do_exit(self, _arg: str) -> bool:
        """exit — leave the console."""
        return True

    def do_quit(self, _arg: str) -> bool:
        """quit — leave the console."""
        return True

    def do_EOF(self, _arg: str) -> bool:  # noqa: N802  (name fixed by cmd.Cmd)
        """Ctrl-D — leave the console."""
        print()
        return True

    # -- niceties -----------------------------------------------------------

    def emptyline(self) -> bool:
        """Do nothing on a blank line (cmd's default repeats the last command)."""
        return False

    def default(self, line: str) -> None:
        """Any non-command line is treated as a prompt to generate from."""
        self._generate(line)

    # -- internals ----------------------------------------------------------

    def _generate(self, prompt: str) -> None:
        if self._generator is None:
            print("no model loaded — use `load <path>` first")
            return
        s = self.settings
        if s["num_beams"] and s["num_beams"] > 1:
            # beam search isn't incremental; generate then print.
            print(self._generator.generate(
                prompt, max_new_tokens=s["max_new_tokens"], num_beams=s["num_beams"]
            ))
            return
        for chunk in self._generator.stream(
            prompt,
            max_new_tokens=s["max_new_tokens"],
            temperature=s["temperature"],
            top_k=s["top_k"],
            top_p=s["top_p"],
            repetition_penalty=s["repetition_penalty"],
        ):
            print(chunk, end="", flush=True)
        print()


def _parse_value(key: str, raw: str) -> Any:
    if raw.lower() in ("none", "off", "-"):
        return None
    if key in ("top_k", "num_beams", "max_new_tokens"):
        return int(raw)
    return float(raw)


def run(checkpoint: str | None = None) -> None:
    """Start the interactive console, optionally pre-loading a checkpoint."""
    from .banner import render_banner

    console = ZenithConsole(intro=render_banner())
    if checkpoint:
        console.do_load(checkpoint)
    try:
        console.cmdloop()
    except KeyboardInterrupt:
        print()
