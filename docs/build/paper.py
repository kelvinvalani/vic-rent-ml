"""Render docs/RESEARCH_PAPER.md to LaTeX and compile a typeset PDF.

The markdown stays the canonical source. This script translates the document's
markdown subset — sections, pipe tables, display/inline math, lists, links and
figures — into ``docs/vic-rent-ml-paper.tex`` and compiles it with the first TeX
engine found (Tectonic, XeLaTeX, LuaLaTeX or pdfLaTeX) into
``docs/vic-rent-ml-paper.pdf``. Tectonic is the recommended engine: a single
binary that fetches only the packages the document needs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS / "RESEARCH_PAPER.md"
TEX_OUT = DOCS / "vic-rent-ml-paper.tex"
PDF_OUT = DOCS / "vic-rent-ml-paper.pdf"

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{listings}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Measuring two-year forecast error for Victorian rental medians},
            pdfauthor={Kelvin Valani}}
\lstset{basicstyle=\small\ttfamily, breaklines=true, columns=fullflexible,
        keepspaces=true, frame=single, framerule=0.2pt, xleftmargin=1em}
\setlength{\emergencystretch}{3em}
"""

MATH_PATTERNS = (
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
)
FENCED = re.compile(r"^```[^\n]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
INLINE_CODE = re.compile(r"`([^`\n]+)`")

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}
UNICODE_MAP = {
    "—": "---",
    "–": "--",
    "±": r"$\pm$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
}


def _protect(text: str) -> tuple[str, dict[str, str]]:
    """Pull fenced code, math and inline code out of the text as placeholders."""
    store: dict[str, str] = {}

    def stash(kind: str, payload: str) -> str:
        token = f"@@{kind}{sum(1 for k in store if k.startswith('@@' + kind))}@@"
        store[token] = payload
        return token

    text = FENCED.sub(
        lambda m: stash("PRE", "\\begin{lstlisting}\n" + m.group(1) + "\\end{lstlisting}"), text)
    for pattern in MATH_PATTERNS:
        text = pattern.sub(lambda m: stash("MATH", m.group(0)), text)
    text = INLINE_CODE.sub(lambda m: stash("CODE", m.group(1)), text)
    return text, store


def _escape(text: str) -> str:
    out = []
    for char in text:
        out.append(LATEX_ESCAPES.get(char, UNICODE_MAP.get(char, char)))
    return "".join(out)


def _escape_code(text: str) -> str:
    """Escape code for ``\\texttt{}``; ``{-}`` stops -- ligatures inside tt."""
    return _escape(text).replace("-", "{-}")


def _restore_placeholders(text: str, store: dict[str, str]) -> str:
    for token, payload in store.items():
        if token.startswith("@@MATH"):
            text = text.replace(token, payload)
        elif token.startswith("@@CODE"):
            if "/" in payload or "\\\\" in payload:
                # \path sets the same tt text but may break at / and . — needed
                # for long file paths that \texttt would push into the margin.
                text = text.replace(token, rf"\path{{{payload}}}")
            else:
                text = text.replace(token, rf"\texttt{{{_escape_code(payload)}}}")
    return text


def inline(text: str, store: dict[str, str]) -> str:
    """Apply inline markdown → LaTeX on protected text, then restore spans."""
    text = _escape(text)
    text = re.sub(r'"([^"\n]{1,120})"', r"``\1''", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: rf"\href{{{m.group(2)}}}{{{m.group(1)}}}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\\emph{\1}", text)
    return _restore_placeholders(text, store)


def _strip_number(title: str) -> str:
    return re.sub(r"^\d+(\.\d+)*\.?\s+", "", title.strip())


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _column_spec(header: list[str], aligns: list[str], body: list[list[str]]) -> tuple[str, bool]:
    """Pick per-column specifiers; wide text columns become tabularx X columns."""
    def plain(cell: str) -> str:
        return re.sub(r"[*`]|@@\w+\d+@@", "", cell)

    spec, has_x = [], False
    for index, align in enumerate(aligns):
        head = len(plain(header[index])) if index < len(header) else 0
        longest = max([head] + [len(plain(row[index])) for row in body if index < len(row)])
        if longest > 30 or head > 12 or (align == "l" and longest > 24):
            spec.append(r">{\raggedright\arraybackslash}X")
            has_x = True
        else:
            spec.append(align)
    return "".join(spec), has_x


def _table_block(rows: list[str], store: dict[str, str]) -> str:
    header = _split_row(rows[0])
    aligns = []
    for marker in _split_row(rows[1]):
        marker = marker.strip()
        aligns.append("c" if marker.startswith(":") and marker.endswith(":")
                      else "r" if marker.endswith(":") else "l")
    body = [_split_row(row) for row in rows[2:]]
    spec, has_x = _column_spec(header, aligns, body)
    if has_x:
        begin, end = rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}{spec}@{{}}}}", r"\end{tabularx}"
    else:
        begin, end = rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\end{tabular}"
    lines = [r"\begin{center}", r"\small", begin, r"\toprule",
             " & ".join(inline(cell, store) for cell in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(inline(cell, store) for cell in row) + r" \\" for row in body]
    lines += [r"\bottomrule", end, r"\end{center}"]
    return "\n".join(lines)


def _figure_block(alt: str, src: str, store: dict[str, str]) -> str:
    return ("\n".join([
        r"\begin{figure}[!htbp]",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{src}}}",
        rf"\caption{{{inline(alt, store)}}}",
        r"\end{figure}"]))


def _list_block(lines: list[str], start: int, store: dict[str, str]) -> tuple[str, int]:
    """Consume an itemize/enumerate block, including indented continuations."""
    first = lines[start].strip()
    ordered = bool(re.match(r"\d+\.\s", first))
    env = "enumerate" if ordered else "itemize"
    items: list[str] = []
    i = start
    while i < len(lines):
        raw, stripped = lines[i], lines[i].strip()
        if not stripped:
            lookahead = i + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines) and re.match(r"(-|\d+\.)\s", lines[lookahead].strip()):
                i = lookahead
                continue
            break
        if re.match(r"(-|\d+\.)\s", stripped):
            items.append(re.sub(r"^(-|\d+\.)\s+", "", stripped))
        elif raw[:1] in {" ", "\t"} and items:
            items[-1] += " " + stripped
        else:
            break
        i += 1
    block = "\n".join([rf"\begin{{{env}}}"]
                      + [rf"\item {inline(item, store)}" for item in items]
                      + [rf"\end{{{env}}}"])
    return block, i


def _is_block_start(stripped: str) -> bool:
    return bool(
        re.match(r"(#{1,3}\s|\||@@PRE\d+@@|!\[|-\s|\d+\.\s)", stripped)
        or stripped.startswith("```")
    )


def _paragraphs(lines: list[str], store: dict[str, str]) -> tuple[str, int]:
    """Join consecutive non-block lines into LaTeX paragraphs."""
    out, buffer = [], []

    def flush() -> None:
        if buffer:
            out.append(inline(" ".join(buffer), store))
            buffer.clear()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("@@PRE") and stripped.endswith("@@"):
            flush()
            out.append(store[stripped])
            i += 1
            continue
        if _is_block_start(stripped):
            flush()
            break
        buffer.append(stripped)
        i += 1
    flush()
    return "\n\n".join(out), i


def _parse_masthead(lines: list[str], store: dict[str, str]) -> tuple[str, str]:
    """Turn the ``**Key:** value`` block under the title into author/date lines."""
    fields = {}
    for line in lines:
        match = re.match(r"\*\*([^*]+):\*\*\s*(.*)", line.strip())
        if match:
            fields[match.group(1).strip().lower()] = match.group(2).strip().rstrip("  ")
    author_lines, date_lines = [], []
    if fields.get("author"):
        author_lines.append(inline(fields["author"], store))
    if fields.get("repository"):
        repo = fields["repository"]
        link = re.match(r"\[([^\]]+)\]\(([^)]+)\)", repo)
        if link:
            author_lines.append(
                rf"\normalsize\href{{{link.group(2)}}}{{\texttt{{{_escape_code(link.group(1))}}}}}")
    for key in ("units", "data"):
        if fields.get(key):
            date_lines.append(f"{key.title()}: {inline(fields[key], store)}")
    author = r" \\ ".join(author_lines)
    date = r" \textperiodcentered{} ".join(date_lines) if date_lines else ""
    return author, rf"\small {date}" if date else ""


def convert(text: str) -> tuple[str, str, str, str]:
    """Return (title, author, date, latex body) for the markdown source."""
    protected, store = _protect(text)
    lines = protected.split("\n")
    title, author, date = "", "", ""
    body: list[str] = []
    seen_heading = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("### "):
            body.append(rf"\subsection{{{inline(_strip_number(stripped[4:]), store)}}}")
            i += 1
            continue
        if stripped.startswith("## "):
            name = _strip_number(stripped[3:])
            if name == "Abstract":
                block, offset = _paragraphs(lines[i + 1:], store)
                body.append("\\begin{abstract}\n" + block + "\n\\end{abstract}")
                i += offset + 1
            else:
                body.append(rf"\section{{{inline(name, store)}}}")
                i += 1
            seen_heading = True
            continue
        if stripped.startswith("# "):
            title = inline(stripped[2:], store)
            i += 1
            continue
        if stripped.startswith("@@PRE") and stripped.endswith("@@"):
            body.append(store[stripped])
            i += 1
            continue
        if stripped.startswith("|"):
            end = i
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            body.append(_table_block(lines[i:end], store))
            i = end
            continue
        image = re.match(r"!\[([^\]]*)\]\(([^)\s]+)\)", stripped)
        if image:
            body.append(_figure_block(image.group(1), image.group(2), store))
            i += 1
            continue
        if re.match(r"(-|\d+\.)\s", stripped):
            block, i = _list_block(lines, i, store)
            body.append(block)
            continue
        if stripped.startswith("**") and not seen_heading:
            end = i
            masthead = []
            while end < len(lines) and lines[end].strip():
                masthead.append(lines[end])
                end += 1
            author, date = _parse_masthead(masthead, store)
            i = end
            continue
        block, offset = _paragraphs(lines[i:], store)
        body.append(block)
        i += offset
    return title, author, date, "\n\n".join(part for part in body if part)


def render_tex(source: str) -> str:
    title, author, date, body = convert(source)
    return (PREAMBLE
            + rf"\title{{{title}}}" + "\n"
            + rf"\author{{{author}}}" + "\n"
            + rf"\date{{{date}}}" + "\n"
            + "\\begin{document}\n\\maketitle\n\n"
            + body
            + "\n\n\\end{document}\n")


def _engine() -> list[str] | None:
    override = os.environ.get("TEX_ENGINE")
    if override:
        return [override]
    for name in ("tectonic", "xelatex", "lualatex", "pdflatex"):
        found = shutil.which(name)
        if found:
            return [found]
    # Tectonic's default single-binary location on Windows.
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local) / "tectonic" / "tectonic.exe"
        if candidate.exists():
            return [str(candidate)]
    return None


def compile_tex(engine: list[str]) -> None:
    exe = engine[0]
    if Path(exe).stem == "tectonic":
        command = [exe, "--outdir", str(DOCS), str(TEX_OUT)]
        subprocess.run(command, cwd=DOCS, check=True)
        return
    command = [exe, "-interaction=nonstopmode", "-halt-on-error",
               f"-output-directory={DOCS}", TEX_OUT.name]
    for _ in range(2):
        subprocess.run(command, cwd=DOCS, check=True)
    for suffix in (".aux", ".out", ".log", ".toc"):
        (DOCS / (TEX_OUT.stem + suffix)).unlink(missing_ok=True)


def main() -> int:
    TEX_OUT.write_text(render_tex(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {TEX_OUT.relative_to(DOCS.parent)}")
    engine = _engine()
    if engine is None:
        print("no TeX engine found: install Tectonic (https://tectonic-typesetting.github.io) "
              "or a TeX Live/MiKTeX distribution to produce the PDF")
        return 1
    compile_tex(engine)
    print(f"wrote {PDF_OUT.relative_to(DOCS.parent)} ({PDF_OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
