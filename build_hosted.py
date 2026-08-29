#!/usr/bin/env python3
"""
build_hosted.py -- bake the two CSVs into dashboard.html to produce a single
self-contained page (dashboard_hosted.html) that needs no server and no file
picking. Run it again after any rerun of project.py to refresh the baked data.

    python3 build_hosted.py

The output is a page *fragment* (title + style + content + script) rather than a
full document, which is the form the Artifact host expects; browsers render it
directly too.
"""

from __future__ import annotations

import json
import os
import re
import sys

DASHBOARD = "dashboard.html"
PROJ_CSV = os.path.join("data", "projections_2026_27.csv")
HIST_CSV = os.path.join("data", "skaters_3yr.csv")
OUT = "dashboard_hosted.html"


def read(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run fetch_nhl.py and project.py first")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def js_string(text: str) -> str:
    """JSON-encode for embedding in a <script> block (ASCII-only)."""
    return json.dumps(text, ensure_ascii=True).replace("</", "<\\/")


def ascii_safe(chunk: str, in_script: bool) -> str:
    """
    Escape every non-ASCII character so the page cannot be misread as
    windows-1252. The fragment has no <head> of its own to carry a charset
    declaration, and a mis-guessed encoding turns em dashes and the sort
    arrows into mojibake.
    """
    out = []
    for ch in chunk:
        if ord(ch) < 128:
            out.append(ch)
        elif in_script:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append("&#x%x;" % ord(ch))
    return "".join(out)


def escape_mixed(html: str) -> str:
    """Apply the right escape to each half of a chunk of markup + scripts."""
    parts = re.split(r"(<script>|</script>)", html)
    in_script = False
    out = []
    for part in parts:
        if part == "<script>":
            in_script = True
            out.append(part)
        elif part == "</script>":
            in_script = False
            out.append(part)
        else:
            out.append(ascii_safe(part, in_script))
    return "".join(out)


def main() -> int:
    src = read(DASHBOARD)
    proj = read(PROJ_CSV)
    hist = read(HIST_CSV)

    title = re.search(r"<title>.*?</title>", src, re.S)
    style = re.search(r"<style>.*?</style>", src, re.S)
    body = re.search(r"<body[^>]*>(.*)</body>", src, re.S)
    if not (title and style and body):
        sys.exit("could not find <title>, <style> and <body> in dashboard.html")

    data_script = (
        "<script>\n"
        "window.EMBEDDED_DATA = {\n"
        f"  proj: {js_string(proj)},\n"
        f"  hist: {js_string(hist)}\n"
        "};\n"
        "</script>"
    )

    out = "\n".join([
        '<meta charset="utf-8">',
        escape_mixed(title.group(0)),
        escape_mixed(style.group(0)),
        data_script,
        escape_mixed(body.group(1).strip()),
    ])
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(out)

    n_proj = proj.count("\n") - 1
    n_hist = hist.count("\n") - 1
    print(f"Wrote {OUT} ({len(out) / 1024:.0f} KB) "
          f"with {n_proj} projections and {n_hist} player-season rows baked in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
