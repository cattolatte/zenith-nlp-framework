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


@app.command()
def serve(
    model: str = typer.Option(..., "--model", "-m", help="Path to a checkpoint (.pt)."),
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Serve a checkpoint over HTTP (POST /generate, POST /generate/stream)."""
    from ..serving import serve as _serve

    _serve(model_path=model, host=host, port=port)


@app.command()
def chat(
    model: str = typer.Option(..., "--model", "-m", help="Path to a checkpoint (.pt)."),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-n"),
    temperature: float = typer.Option(0.8, "--temperature", "-t"),
    top_p: float = typer.Option(0.9, "--top-p"),
) -> None:
    """Interactive REPL: type a prompt, watch the model stream a continuation."""
    from ..checkpoint import load_pretrained

    generator = load_pretrained(model)
    typer.echo("Zenith chat — Ctrl-C or empty line to exit.\n")
    while True:
        try:
            prompt = typer.prompt("you", prompt_suffix=" > ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break
        for chunk in generator.stream(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        ):
            typer.echo(chunk, nl=False)
        typer.echo("\n")


@app.command()
def console(
    model: str = typer.Option(None, "--model", "-m", help="Checkpoint to load on start."),
) -> None:
    """Launch the interactive Zenith console (a generation REPL with a banner)."""
    from ..console import run

    run(model)


@app.command(name="eval")
def evaluate_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Path to a checkpoint (.pt)."),
    corpus: str = typer.Option(..., "--corpus", "-c", help="Held-out text file to score."),
    batch_size: int = typer.Option(32, "--batch-size"),
) -> None:
    """Report held-out loss and perplexity for a checkpoint."""
    from ..checkpoint import load_checkpoint
    from ..data import CausalLMDataset, load_corpus_file
    from ..evaluation import evaluate

    model_obj, tokenizer = load_checkpoint(model)
    ids = load_corpus_file(corpus, tokenizer)
    dataset = CausalLMDataset(ids, model_obj.config.block_size)
    metrics = evaluate(model_obj, dataset, batch_size=batch_size)
    typer.echo(f"loss: {metrics['loss']:.4f}  perplexity: {metrics['perplexity']:.2f}")


@app.command(name="train")
def train_hint() -> None:
    """How to launch training (Hydra owns argv, so it has its own entrypoint)."""
    typer.echo("Train via the Hydra entrypoint, e.g.:")
    typer.echo("  python -m zenith.cli.train")
    typer.echo("  python -m zenith.cli.train -m training.learning_rate=1e-3,3e-4")


if __name__ == "__main__":
    app()
