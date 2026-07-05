"""The ``zenith`` command-line entrypoint.

A small Typer app for the non-training commands. Training is a Hydra entrypoint
(``python -m zenith.cli.train``) because Hydra owns argv for config composition
and multirun sweeps.
"""

from __future__ import annotations

import typer

from .. import __version__

app = typer.Typer(help="Zenith — a from-scratch generative NLP library.")


@app.command()
def info() -> None:
    """Show Zenith / PyTorch (and optional Polaris) versions."""
    typer.echo(f"zenith {__version__}")
    try:
        import torch

        typer.echo(f"torch {torch.__version__}")
    except ImportError:
        typer.echo("torch not installed")
    try:
        import polaris

        typer.echo(f"polaris {getattr(polaris, '__version__', 'unknown')} (optional interop)")
    except ImportError:
        pass


@app.command()
def generate(
    prompt: str = typer.Argument("", help="Prompt to continue."),
    model: str = typer.Option(..., "--model", "-m", help="Path to a checkpoint (.pt)."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-n"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
) -> None:
    """Generate text from a trained checkpoint."""
    from ..checkpoint import load_pretrained

    generator = load_pretrained(model)
    text = generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    typer.echo(prompt + text)


@app.command(name="train")
def train_hint() -> None:
    """How to launch training (Hydra owns argv, so it has its own entrypoint)."""
    typer.echo("Train via the Hydra entrypoint, e.g.:")
    typer.echo("  python -m zenith.cli.train")
    typer.echo("  python -m zenith.cli.train -m training.learning_rate=1e-3,3e-4")


if __name__ == "__main__":
    app()
