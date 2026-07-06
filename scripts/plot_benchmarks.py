"""Render the benchmark figures in ``assets/`` from the measured numbers.

Reproducible and self-contained: the values below are the results reported in
BENCHMARKS.md (final bits/char + the per-epoch validation curves from the scaling
and text8 runs). Figures are written on a white card so they read on any page
background (GitHub light *and* dark).

Requires matplotlib (``pip install matplotlib``); the generated PNGs are committed,
so consumers never need to regenerate them.

    python scripts/plot_benchmarks.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path("assets")
PURPLE = "#6E56CF"
TEAL = "#1D9E75"
GRAY = "#B4B2A9"
INK = "#1b1b2b"
MUTED = "#6b6b80"
GRID = "#e9e8f2"
CARD = "#ffffff"
SEQ = ["#C6B9EE", "#9E86E0", "#7A5FD0", "#4E3AA7"]  # light -> dark (small -> large)

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, color=INK, weight="bold", pad=12, loc="left")
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=200, bbox_inches="tight", pad_inches=0.35,
                facecolor=CARD, metadata={"Software": "zenith"})
    plt.close(fig)
    print("wrote", ASSETS / name)


def scaling_curve() -> None:
    params = [0.60, 1.82, 5.06, 10.73]
    bpc = [2.329, 2.137, 2.077, 2.066]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.plot(params, bpc, "-", color=PURPLE, lw=2.5, zorder=3)
    ax.scatter(params, bpc, color=PURPLE, s=70, zorder=4, edgecolor=CARD, linewidth=2)
    for x, y in zip(params, bpc):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9.5, color=INK)
    ax.axhline(2.066, color=MUTED, ls="--", lw=1, alpha=0.45)
    ax.annotate("data floor ≈ 2.07", (10.73, 2.066), textcoords="offset points",
                xytext=(-4, -15), ha="right", fontsize=9, color=MUTED)
    ax.set_xticks(params)
    ax.set_xticklabels(["0.6M", "1.8M", "5.1M", "10.7M"])
    ax.set_ylim(2.03, 2.37)
    _style(ax, "Scaling on tiny-shakespeare", "model parameters (log scale)",
           "bits / char  (lower is better)")
    _save(fig, "scaling_curve.png")


def convergence() -> None:
    epochs = list(range(1, 13))
    curves = {
        "0.6M": [3.547, 2.854, 2.668, 2.551, 2.503, 2.435, 2.395, 2.374, 2.355, 2.341, 2.338, 2.337],
        "1.8M": [3.159, 2.630, 2.421, 2.323, 2.274, 2.220, 2.198, 2.184, 2.158, 2.151, 2.149, 2.148],
        "5.1M": [2.908, 2.486, 2.328, 2.238, 2.197, 2.154, 2.131, 2.117, 2.101, 2.092, 2.090, 2.090],
        "10.7M": [2.800, 2.414, 2.270, 2.202, 2.164, 2.131, 2.105, 2.092, 2.083, 2.081, 2.083, 2.086],
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (label, vals), color in zip(curves.items(), SEQ):
        ax.plot(epochs, vals, color=color, lw=2.2, zorder=3, label=label)
        ax.scatter([epochs[-1]], [vals[-1]], color=color, s=30, zorder=4)
    ax.set_xticks(epochs)
    ax.set_xlim(1, 12.3)
    leg = ax.legend(title="parameters", loc="upper right", frameon=False,
                    fontsize=10, title_fontsize=10, labelcolor=INK, handlelength=1.4)
    leg.get_title().set_color(MUTED)
    _style(ax, "Convergence by model size", "epoch", "validation bits / char")
    _save(fig, "convergence.png")


def architecture() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 4.0))
    labels = ["GPT-2\nstyle", "Llama\nstyle"]
    colors = [GRAY, PURPLE]
    b1 = ax1.bar(labels, [2.11, 2.08], color=colors, width=0.6, zorder=3)
    ax1.set_ylim(0, 2.6)
    for bar, v in zip(b1, [2.11, 2.08]):
        ax1.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 5), ha="center",
                     fontsize=11, color=INK, weight="bold")
    _style(ax1, "Quality (nearly tied)", "", "bits / char")
    b2 = ax2.bar(labels, [17, 10], color=colors, width=0.6, zorder=3)
    ax2.set_ylim(0, 20)
    for bar, v in zip(b2, [17, 10]):
        ax2.annotate(f"{v}", (bar.get_x() + bar.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 5), ha="center",
                     fontsize=11, color=INK, weight="bold")
    _style(ax2, "Convergence (~2x faster)", "", "epochs to best")
    fig.suptitle("Architecture ablation  ·  same params, same recipe", x=0.06,
                 ha="left", fontsize=14, color=INK, weight="bold")
    fig.subplots_adjust(top=0.82, wspace=0.35)
    _save(fig, "architecture.png")


def text8_curve() -> None:
    epochs = list(range(1, 7))
    bpc = [2.150, 1.964, 1.880, 1.825, 1.787, 1.783]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epochs, bpc, "-", color=TEAL, lw=2.5, zorder=3)
    ax.scatter(epochs, bpc, color=TEAL, s=60, zorder=4, edgecolor=CARD, linewidth=2)
    for x, y in zip(epochs, bpc):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(epochs)
    ax.set_ylim(1.72, 2.22)
    _style(ax, "text8 subset (5 MB) — out-of-domain prose", "epoch",
           "validation bits / char")
    _save(fig, "text8_curve.png")


if __name__ == "__main__":
    scaling_curve()
    convergence()
    architecture()
    text8_curve()
