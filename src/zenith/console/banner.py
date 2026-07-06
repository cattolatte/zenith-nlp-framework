"""The console banner: bundled ASCII art, randomly colored per launch.

Color is polite — suppressed when stdout is not a TTY or ``NO_COLOR`` is set — and
deterministic on demand (pass a seeded ``random.Random``).
"""

from __future__ import annotations

import os
import random
import sys

__all__ = ["render_banner", "COLOR_SCHEMES"]

_ART = r"""
███████╗███████╗███╗   ██╗██╗████████╗██╗  ██╗
╚══███╔╝██╔════╝████╗  ██║██║╚══██╔══╝██║  ██║
  ███╔╝ █████╗  ██╔██╗ ██║██║   ██║   ███████║
 ███╔╝  ██╔══╝  ██║╚██╗██║██║   ██║   ██╔══██║
███████╗███████╗██║ ╚████║██║   ██║   ██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝
""".strip("\n")

_TAGLINE = "a from-scratch generative NLP library — type `help` for commands"

# ANSI foreground codes for the art, chosen at random per launch.
COLOR_SCHEMES: tuple[int, ...] = (36, 35, 34, 32, 33, 31)  # cyan/magenta/blue/green/yellow/red


def _color_enabled(*, force: bool | None = None) -> bool:
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def render_banner(rng: random.Random | None = None, *, color: bool | None = None) -> str:
    """Render the ASCII banner, randomly colored (unless disabled)."""
    chooser = rng if rng is not None else random
    code = chooser.choice(COLOR_SCHEMES)
    if _color_enabled(force=color):
        art = f"\033[1;{code}m{_ART}\033[0m"
        tag = f"\033[2m{_TAGLINE}\033[0m"
    else:
        art, tag = _ART, _TAGLINE
    return f"\n{art}\n  {tag}\n"
