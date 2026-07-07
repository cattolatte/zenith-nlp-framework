"""Render a terminal-style demo GIF of the instruction-tuned model chatting.

Uses **real** model output (greedy, EOS-terminated) and draws it streaming into a
faux terminal with Pillow — no external screen-recording tools required, and fully
reproducible. The committed GIF is embedded in the README.

    python scripts/render_demo_gif.py --model instruct-shakespeare.pt --out assets/demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from zenith.checkpoint import load_checkpoint
from zenith.generation import Generator
from zenith.instruct import ChatTemplate

W, H, PAD, BAR_H, FSIZE, LINE_H = 680, 392, 22, 34, 17, 25
BG, BAR = (30, 30, 46), (43, 43, 61)
TXT, PROMPT, BOT, DIM = (205, 214, 244), (156, 143, 230), (94, 202, 165), (110, 114, 140)
DOTS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
QUESTIONS = ["Who are you?", "Tell me a joke.", "What is the capital of France?"]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
                 "/Library/Fonts/Menlo.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


FONT, SMALL = _font(FSIZE), _font(13)
CW = round(FONT.getlength("M"))
MAX_COLS = (W - 2 * PAD) // CW


def _transcript(model_path: str) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Real chat exchanges, wrapped into styled (char, color) display lines."""
    model, tok = load_checkpoint(model_path)
    gen = Generator(model, tok)
    tmpl = ChatTemplate()
    lines: list[list[tuple[str, tuple[int, int, int]]]] = []

    def styled(text, color):
        return [(c, color) for c in text]

    def wrap(spans):
        row: list[tuple[str, tuple[int, int, int]]] = []
        for ch, col in spans:
            row.append((ch, col))
            if len(row) >= MAX_COLS:
                brk = next((j for j in range(len(row) - 1, 0, -1) if row[j][0] == " "), None)
                if brk is None:  # no space to break on: hard wrap
                    lines.append(row)
                    row = []
                else:  # break at the last space, dropping it
                    lines.append(row[:brk])
                    row = row[brk + 1:]
        if row:
            lines.append(row)

    for q in QUESTIONS:
        wrap(styled("you ", PROMPT) + styled("> ", DIM) + styled(q, TXT))
        ids = tok.encode(tmpl.format_prompt(q))
        out = gen.generate_ids(torch.tensor([ids]), max_new_tokens=80, temperature=0.0,
                               stop_ids=[tok.eos_id])
        resp = tok.decode(out[0, len(ids):].tolist()).strip()
        wrap(styled("zenith ", BOT) + styled("» ", DIM) + styled(resp, TXT))
        lines.append([])  # spacer
    return lines


def _frame(lines, reveal: int, cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, BAR_H], fill=BAR)
    for i, color in enumerate(DOTS):
        cx = PAD + i * 20
        d.ellipse([cx - 5, BAR_H // 2 - 5, cx + 5, BAR_H // 2 + 5], fill=color)
    d.text((W // 2, BAR_H // 2), "zenith · chat", font=SMALL, fill=DIM, anchor="mm")

    y, consumed, cursor_xy = BAR_H + PAD - 6, 0, None
    for row in lines:
        take = max(0, min(len(row), reveal - consumed))
        x = PAD
        for ch, col in row[:take]:
            d.text((x, y), ch, font=FONT, fill=col)
            x += CW
        if take > 0:
            cursor_xy = (x, y)
        consumed += len(row)
        y += LINE_H
        if y > H - PAD:
            break
    if cursor and cursor_xy is not None:
        cx, cy = cursor_xy
        d.rectangle([cx + 1, cy + 2, cx + CW - 1, cy + FSIZE + 3], fill=TXT)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="instruct-shakespeare.pt")
    ap.add_argument("--out", default="assets/demo.gif")
    args = ap.parse_args()

    lines = _transcript(args.model)
    total = sum(len(r) for r in lines)
    frames, durations = [], []
    step = 3  # characters revealed per frame
    for n in range(0, total + 1, step):
        frames.append(_frame(lines, n, cursor=(n // step) % 2 == 0))
        durations.append(45)
    for _ in range(24):  # hold the finished screen
        frames.append(_frame(lines, total, cursor=len(frames) % 2 == 0))
        durations.append(90)

    Path(args.out).parent.mkdir(exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {args.out}  ({len(frames)} frames, {Path(args.out).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
