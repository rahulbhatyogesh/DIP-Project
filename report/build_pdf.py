"""Render report.md to a print-ready PDF named after the BITS ID.

No pandoc or LaTeX needed: markdown-it-py produces the HTML, and headless Chrome
prints it. Both are already on this machine.

    python report/build_pdf.py                     # -> report/2023B4A31035P.pdf
    python report/build_pdf.py --html-only         # inspect the HTML first
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
BITS_ID = "2023B4A31035P"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 9.6pt; line-height: 1.38; color: #111; max-width: 100%;
}
h1 { font-size: 19pt; margin: 0 0 .2em; line-height: 1.25; }
h2 { font-size: 12.5pt; margin: 1.1em 0 .4em; padding-bottom: .12em;
     border-bottom: 1px solid #bbb; page-break-after: avoid; }
h3 { font-size: 10.6pt; margin: .85em 0 .3em; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: .6em 0;
        font-size: 8.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #c6c6c6; padding: 2.5px 5px; text-align: left; }
th { background: #f0f0f0; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
code { font-family: "Consolas", "Menlo", monospace; font-size: 8.8pt;
       background: #f2f2f2; padding: 1px 3px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; border-radius: 3px;
      padding: 8px 10px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.35; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 12px;
             color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.0em 0; }
img { max-width: 100%; max-height: 74mm; display: block; margin: .4em auto;
      page-break-inside: avoid; }
p > img + em, figure em { font-size: 8.6pt; }
hr + h2 { margin-top: .5em; }
strong { font-weight: 650; }
"""

# MathJax renders the LaTeX; Chrome is given a virtual-time budget below so it
# does not print before typesetting finishes.
MATHJAX = """
<script>
window.MathJax = {
  tex: {inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true},
  options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']},
  svg: {fontCache: 'global'}
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style>{mathjax}</head><body>{body}</body></html>"""


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    return shutil.which("chrome") or shutil.which("msedge")


def build_html(md_path: Path) -> str:
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    return HTML.format(
        title=BITS_ID, css=CSS, mathjax=MATHJAX,
        body=md.render(md_path.read_text(encoding="utf-8")),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(HERE / "report.md"))
    ap.add_argument("--out", default=str(HERE / f"{BITS_ID}.pdf"))
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args(argv)

    md_path = Path(args.md)
    if not md_path.exists():
        print(f"missing {md_path}", file=sys.stderr)
        return 1

    html_path = Path(args.out).with_suffix(".html")
    html_path.write_text(build_html(md_path), encoding="utf-8")
    print(f"wrote {html_path}")
    if args.html_only:
        return 0

    chrome = find_chrome()
    if not chrome:
        print("Chrome/Edge not found. Open the HTML and use Ctrl+P -> Save as PDF.")
        return 0

    out = Path(args.out)
    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            # let MathJax typeset before the page is printed
            "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={out}", html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    if out.exists():
        print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
        return 0
    print("Chrome produced no PDF; open the HTML and print manually.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
