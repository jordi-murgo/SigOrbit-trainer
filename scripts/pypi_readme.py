"""Rewrite README.md relative links to absolute GitHub URLs for PyPI.

PyPI renders the long description as HTML but has no filesystem context, so
relative links like ``docs/ARCHITECTURE.md`` or ``LICENSE`` resolve to broken
``pypi.org/project/sigorbit-trainer/docs/...`` URLs.  This script rewrites
in-place every relative markdown link or image to an absolute GitHub URL so
the same README works on both GitHub and PyPI.

Run it in CI immediately before ``python -m build``; never commit the
rewritten file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_URL = "https://github.com/jordi-murgo/sigorbit-trainer"
BLOB_BASE = f"{REPO_URL}/blob/main"
RAW_BASE = f"{REPO_URL}/raw/main"

# Matches [text](url) and ![alt](url)
_LINK_RE = re.compile(r"(?P<bang>!)?\[(?P<text>[^\]]*)\]\((?P<url>[^)]+)\)")


def _is_absolute(url: str) -> bool:
    return url.startswith(("http://", "https://", "#", "mailto:", "data:"))


def _rewrite(url: str, is_image: bool) -> str:
    if _is_absolute(url):
        return url
    # Strip optional anchor/query, preserve for reattachment
    anchor = ""
    if "#" in url:
        url, anchor = url.split("#", 1)
        anchor = f"#{anchor}"
    base = RAW_BASE if is_image else BLOB_BASE
    return f"{base}/{url}{anchor}"


def transform(text: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        bang = m.group("bang") or ""
        text_val = m.group("text")
        url = m.group("url").strip()
        is_image = bool(bang)
        return f"{bang}[{text_val}]({_rewrite(url, is_image)})"

    return _LINK_RE.sub(_sub, text)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: pypi_readme.py <README.md>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    original = path.read_text(encoding="utf-8")
    rewritten = transform(original)
    if rewritten == original:
        print("no relative links found; README unchanged", file=sys.stderr)
        return 0
    path.write_text(rewritten, encoding="utf-8")
    print(f"rewrote relative links in {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))