"""Download a public-domain corpus for real training/benchmarking.

Fetches the "tiny shakespeare" text (public domain) into ``data/``. Network access
required; this is a convenience script, not part of the tested library.

    python scripts/download_corpus.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path("data/tiny_shakespeare.txt")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)  # noqa: S310 - trusted public URL
    size = DEST.stat().st_size
    print(f"Wrote {DEST} ({size:,} bytes)")
    print("Train with:  python -m zenith.cli.train data.corpus_path=" + str(DEST))


if __name__ == "__main__":
    main()
