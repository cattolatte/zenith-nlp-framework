"""Download a text8 subset — a standard character-LM benchmark corpus.

text8 is the first 100 MB of cleaned English Wikipedia (lowercase a-z + space) —
the canonical character-level LM benchmark. Full text8 is a multi-hour train on a
laptop, so this writes a subset for an honest, out-of-domain second benchmark
(distinct from tiny-shakespeare). Network access required.

    python scripts/download_text8.py        # default 5 MB subset
    python scripts/download_text8.py 10      # 10 MB subset
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://mattmahoney.net/dc/text8.zip"
DEST = Path("data/text8_subset.txt")


def main() -> None:
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    raw = urllib.request.urlopen(URL, timeout=180).read()  # noqa: S310 - trusted public URL
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        data = zf.read("text8")
    subset = data[: mb * 1024 * 1024]
    DEST.write_bytes(subset)
    print(f"Wrote {DEST} ({len(subset):,} bytes — {mb} MB subset of text8)")
    print("Train with:  python -m zenith.cli.train data.corpus_path=" + str(DEST))


if __name__ == "__main__":
    main()
