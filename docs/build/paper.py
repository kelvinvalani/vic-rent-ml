"""Render docs/RESEARCH_PAPER.md to a typeset HTML page and a print-ready PDF.

Math spans are protected from the Markdown escaper, restored afterwards and typeset by
MathJax; figures are numbered and captioned; Chrome headless prints the paper.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS / "RESEARCH_PAPER.md"
HTML_OUT = DOCS / "vic-rent-ml-paper.html"
PDF_OUT = DOCS / "vic-rent-ml-paper.pdf"
MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
CHROME_CANDIDATES = ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable")

MATH_PATTERNS = (
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
)

STYLE = """
:root {
  --ink: #0f2740;
  --accent: #c2410c;
  --muted: #64748b;
  --rule: #d8e0ea;
}
* { box-sizing: border-box; }
body {
  margin: 0 auto;
  max-width: 46rem;
  padding: 3.5rem 1.5rem 4rem;
  font-family: "Charter", "Iowan Old Style", Georgia, "Times New Roman", serif;
  font-size: 11.2pt;
  line-height: 1.62;
  color: var(--ink);
  background: #fff;
  text-rendering: optimizeLegibility;
}
h1 {
  font-size: 2.05rem;
  line-height: 1.18;
  margin: 0 0 1.1rem;
  letter-spacing: -0.015em;
}
h2 {
  font-size: 1.32rem;
  margin: 2.6rem 0 0.8rem;
  padding-top: 0.9rem;
  border-top: 2px solid var(--ink);
  page-break-after: avoid;
}
h3 {
  font-size: 1.05rem;
  margin: 1.7rem 0 0.5rem;
  color: var(--accent);
  page-break-after: avoid;
}
p { margin: 0 0 0.85rem; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid #f1c9b4; }
code, pre {
  font-family: "SFMono-Regular", "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 0.86em;
}
code { background: #f1f5f9; padding: 0.08em 0.3em; border-radius: 3px; }
pre {
  background: #0f2740;
  color: #e2e8f0;
  padding: 0.9rem 1.1rem;
  border-radius: 6px;
  overflow-x: auto;
  line-height: 1.5;
  page-break-inside: avoid;
}
pre code { background: none; color: inherit; padding: 0; }
.masthead {
  border-left: 4px solid var(--accent);
  padding: 0.1rem 0 0.1rem 1rem;
  margin: 0 0 2.2rem;
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
}
.masthead strong { color: var(--ink); }
#abstract-block {
  background: #f6f9fc;
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 1.1rem 1.3rem 0.4rem;
  margin-bottom: 1.6rem;
  font-size: 0.97em;
}
#abstract-block h2 {
  border: 0;
  margin: 0 0 0.6rem;
  padding: 0;
  font-size: 0.82rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.1rem 0 1.4rem;
  font-size: 0.88em;
  page-break-inside: avoid;
}
th, td { padding: 0.42rem 0.55rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
thead th {
  text-align: left;
  border-bottom: 1.5px solid var(--ink);
  font-size: 0.88em;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--muted);
}
tbody tr:nth-child(even) { background: #f8fafc; }
figure { margin: 1.6rem 0; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid var(--rule); border-radius: 6px; }
figcaption {
  margin-top: 0.5rem;
  font-size: 0.83rem;
  line-height: 1.45;
  color: var(--muted);
}
figcaption b { color: var(--ink); }
blockquote {
  margin: 1.2rem 0;
  padding: 0.1rem 1rem;
  border-left: 3px solid var(--rule);
  color: var(--muted);
}
ul, ol { padding-left: 1.2rem; }
li { margin-bottom: 0.35rem; }
mjx-container[display="true"] { margin: 1.1rem 0 !important; }
@page { size: A4; margin: 17mm 16mm 19mm; }
@media print {
  body { padding: 0; max-width: none; font-size: 10.4pt; }
  a { border-bottom: 0; }
}
"""


def _protect_math(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def stash(match: re.Match[str]) -> str:
        store.append(match.group(0))
        return f"@@MATH{len(store) - 1}@@"

    for pattern in MATH_PATTERNS:
        text = pattern.sub(stash, text)
    return text, store


def _restore_math(html: str, store: list[str]) -> str:
    for index, chunk in enumerate(store):
        html = html.replace(f"@@MATH{index}@@", chunk)
    return html


def _number_figures(html: str) -> str:
    counter = {"n": 0}

    def replace(match: re.Match[str]) -> str:
        counter["n"] += 1
        alt, src = match.group(1), match.group(2)
        return (
            f'<figure><img src="{src}" alt="{alt}">'
            f'<figcaption><b>Figure {counter["n"]}.</b> {alt}</figcaption></figure>'
        )

    return re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', replace, html)


def _inline_images(html: str, base: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        src = match.group(1)
        path = (base / src).resolve()
        if not path.exists():
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{payload}"'

    return re.sub(r'src="((?!data:|https?:)[^"]+)"', replace, html)


def _split_front_matter(html: str) -> str:
    """Wrap the author/repo block and the abstract in their own presentation blocks."""
    html = re.sub(
        r"(<h1>.*?</h1>)\s*(<p>(?:(?!</p>).)*?<strong>Author:</strong>.*?</p>)",
        r'\1<div class="masthead">\2</div>',
        html,
        count=1,
        flags=re.DOTALL,
    )
    match = re.search(r"<h2>Abstract</h2>(.*?)(?=<h2>)", html, flags=re.DOTALL)
    if match:
        block = f'<section id="abstract-block"><h2>Abstract</h2>{match.group(1)}</section>'
        html = html[: match.start()] + block + html[match.end():]
    return html


def render_html(inline: bool) -> str:
    text, store = _protect_math(SOURCE.read_text(encoding="utf-8"))
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "smarty", "sane_lists"],
        output_format="html5",
    )
    body = _restore_math(body, store)
    body = _number_figures(body)
    body = _split_front_matter(body)
    if inline:
        body = _inline_images(body, DOCS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fair temporal model selection for Victorian rental medians</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }},
  options: {{ skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] }}
}};
</script>
<script id="MathJax-script" async src="{MATHJAX}"></script>
<style>{STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _chrome() -> str:
    for name in CHROME_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Chrome/Chromium binary found for PDF printing.")


def write_pdf(html: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "paper.html"
        source.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                _chrome(),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--virtual-time-budget=30000",
                "--run-all-compositor-stages-before-draw",
                "--no-pdf-header-footer",
                f"--print-to-pdf={PDF_OUT}",
                source.as_uri(),
            ],
            check=True,
            capture_output=True,
        )


def main() -> int:
    HTML_OUT.write_text(render_html(inline=False), encoding="utf-8")
    write_pdf(render_html(inline=True))
    print(f"wrote {HTML_OUT.relative_to(DOCS.parent)}")
    print(f"wrote {PDF_OUT.relative_to(DOCS.parent)} ({PDF_OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
