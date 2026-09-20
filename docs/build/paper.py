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
/* Typeset to imitate the LaTeX ``article'' class: Computer Modern, justified text,
   numbered sections, booktabs rules, captions below floats. */
body {
  margin: 0 auto;
  max-width: 40rem;
  padding: 3rem 1.5rem 4rem;
  font-family: "Latin Modern Roman", "LM Roman 10", "CMU Serif", "Computer Modern",
               "Liberation Serif", "Times New Roman", serif;
  font-size: 11pt;
  line-height: 1.42;
  color: #000;
  background: #fff;
  text-align: justify;
  hyphens: auto;
}
h1 {
  font-size: 1.62rem;
  font-weight: bold;
  line-height: 1.22;
  text-align: center;
  margin: 0 0 1.1rem;
}
h2 {
  font-size: 1.12rem;
  font-weight: bold;
  margin: 1.9rem 0 0.55rem;
  text-align: left;
  page-break-after: avoid;
}
h3 {
  font-size: 1rem;
  font-weight: bold;
  font-style: italic;
  margin: 1.2rem 0 0.4rem;
  text-align: left;
  page-break-after: avoid;
}
p { margin: 0 0 0.55rem; }
p + p { text-indent: 1.4em; margin-top: -0.3rem; }
p:has(mjx-container[display="true"]) { text-indent: 0; }
p:has(mjx-container[display="true"]) + p { text-indent: 0; }
a { color: #000; text-decoration: none; border-bottom: 0.4pt solid #999; }
code, pre {
  font-family: "Latin Modern Mono", "LM Mono 10", "Liberation Mono", Menlo, monospace;
  font-size: 0.88em;
  hyphens: none;
}
code { background: none; }
pre {
  background: #fbfbfb;
  border: 0.4pt solid #cfcfcf;
  color: #000;
  padding: 0.6rem 0.8rem;
  margin: 0.8rem 0 1rem;
  overflow-x: auto;
  line-height: 1.38;
  text-align: left;
  page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
.masthead {
  text-align: center;
  margin: 0 0 1.6rem;
  font-size: 0.96rem;
  line-height: 1.5;
}
.masthead strong { font-weight: normal; font-variant: small-caps; }
#abstract-block {
  margin: 0 auto 1.8rem;
  width: 88%;
  font-size: 0.95em;
  line-height: 1.38;
}
#abstract-block h2 {
  text-align: center;
  font-size: 0.95rem;
  margin: 0 0 0.45rem;
}
#abstract-block p + p { text-indent: 1.4em; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem auto 1.3rem;
  font-size: 0.87em;
  line-height: 1.32;
  page-break-inside: avoid;
}
th, td { padding: 0.3rem 0.5rem; vertical-align: top; text-align: left; hyphens: none; }
thead th {
  border-top: 1pt solid #000;
  border-bottom: 0.5pt solid #000;
  font-weight: bold;
}
tbody tr:last-child td { border-bottom: 1pt solid #000; }
figure { margin: 1.3rem 0; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figcaption {
  margin-top: 0.45rem;
  font-size: 0.84rem;
  line-height: 1.35;
  text-align: left;
}
figcaption b { font-weight: bold; }
blockquote {
  margin: 0.9rem 1.6rem;
  font-size: 0.95em;
}
ul, ol { padding-left: 1.4rem; margin: 0.5rem 0 0.8rem; }
li { margin-bottom: 0.18rem; }
mjx-container[display="true"] { margin: 0.9rem 0 !important; }
@page { size: A4; margin: 22mm 21mm 24mm; }
@media print {
  body { padding: 0; max-width: none; font-size: 10.5pt; }
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
            f'<figcaption><b>Figure {counter["n"]}:</b> {alt}</figcaption></figure>'
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
