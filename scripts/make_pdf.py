#!/usr/bin/env python3
"""
Render the paper to PDF. No pandoc on this machine, so this is a focused
Markdown -> LaTeX converter for the subset the draft actually uses.

Deliberately narrow. It handles headings, bold/italic/code, fenced code blocks,
pipe tables, bullet and numbered lists, horizontal rules, and the specific
Unicode the draft contains. It does NOT try to be a general Markdown engine —
a general one would be mostly untested code, and the failure mode of a silent
mis-render in a paper is worse than a crash.

Anything it cannot map raises rather than passing through, because a stray `%`
reaching LaTeX comments out the rest of a line and that is exactly the kind of
silent corruption this project keeps finding in its own analyses.

Usage:
  python scripts/make_pdf.py notes/paper_draft.md -o paper.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# Unicode the draft uses -> LaTeX. Anything outside ASCII and not listed here is
# an error, so a new symbol cannot silently vanish from the rendered paper.
UNI = {
    "—": "---", "–": "--", "…": r"\ldots{}", "×": r"$\times$", "≈": r"$\approx$",
    "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$", "µ": r"\textmu{}",
    "§": r"\S{}", "±": r"$\pm$", "·": r"$\cdot$", "−": "-", "÷": r"$\div$", "’": "'", "‘": "'",
    "“": "``", "”": "''", "≠": r"$\neq$", "λ": r"$\lambda$", "σ": r"$\sigma$",
    "\u00a0": "~", "\u2011": "-", "\u2212": "-",
}
SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(t: str) -> str:
    out = []
    for ch in t:
        if ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ch in UNI:
            out.append(UNI[ch])
        elif ch == "\x00" or ord(ch) < 128:
            out.append(ch)
        else:
            raise ValueError(f"unmapped character {ch!r} (U+{ord(ch):04X}) — add it to UNI")
    return "".join(out)


def inline(t: str) -> str:
    """Inline markup.

    Code spans are pulled out to placeholders FIRST, then emphasis is applied,
    then the code is restored. Splitting on backticks and emphasising each
    fragment separately -- the obvious implementation -- breaks every construct
    where bold wraps a code span, which in this draft is eight of them
    (`**`F` is not the collectives**`, the figure captions, a table cell). Those
    rendered with literal asterisks in the first PDF.
    """
    holds: list[str] = []

    def stash(m: re.Match) -> str:
        holds.append(m.group(1))
        return f"\x00{len(holds) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", t)
    for i, c in enumerate(holds):
        t = t.replace(f"\x00{i}\x00", r"\texttt{" + esc(c) + "}")
    return t


def convert(md: str) -> str:
    lines = md.split("\n")
    body: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            body.append("\\begin{lstlisting}\n" + "\n".join(block) + "\n\\end{lstlisting}")
            continue

        if re.match(r"^---+\s*$", ln):
            body.append(r"\medskip\hrule\medskip"); i += 1; continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl, txt = len(m.group(1)), inline(m.group(2))
            cmd = {1: "title", 2: "section", 3: "subsection", 4: "subsubsection"}[lvl]
            body.append(f"\\{cmd}{{{txt}}}" if cmd != "title"
                        else f"\\begin{{center}}\\LARGE\\bfseries {txt}\\end{{center}}")
            i += 1; continue

        # pipe table: a header row followed by a |---| separator
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, data = rows[0], rows[2:]
            ncol = max(len(r) for r in rows)
            spec = "l" + "r" * (ncol - 1)
            # \resizebox so a wide table shrinks to the column instead of
            # overflowing it. Without this, columns past the text width are
            # silently CLIPPED -- the flag-on column of the mechanism table
            # vanished from the first rendered PDF and only a digit count
            # against the source caught it.
            t = ["\\begin{center}\\resizebox{\\columnwidth}{!}{%",
                 "\\small\\begin{tabular}{" + spec + "}\\hline"]
            t.append(" & ".join(f"\\textbf{{{inline(c)}}}" for c in head) + r" \\ \hline")
            for r in data:
                r = r + [""] * (ncol - len(r))
                t.append(" & ".join(inline(c) for c in r) + r" \\")
            t.append("\\hline\\end{tabular}}\\end{center}")
            body.append("\n".join(t)); continue

        if re.match(r"^\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                raw = [re.sub(r"^\s*[-*]\s+", "", lines[i])]; i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    raw.append(lines[i].strip()); i += 1
                items.append(inline(" ".join(raw)))
            body.append("\\begin{itemize}\\itemsep2pt\n" +
                        "\n".join(f"\\item {x}" for x in items) + "\n\\end{itemize}")
            continue

        prev_blank = i == 0 or not lines[i - 1].strip()
        m_ol = re.match(r"^\s*(\d{1,2})\.\s+", ln)
        if m_ol and prev_blank and int(m_ol.group(1)) <= 20:
            items = []
            while i < len(lines) and re.match(r"^\s*\d{1,2}\.\s+", lines[i]):
                raw = [re.sub(r"^\s*\d+\.\s+", "", lines[i])]; i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    raw.append(lines[i].strip()); i += 1
                # Join BEFORE inline(): bold and italics routinely span the
                # wrapped lines of one item, and per-line parsing leaves the
                # markers literal in the PDF.
                items.append(inline(" ".join(raw)))
            body.append("\\begin{enumerate}\\itemsep2pt\n" +
                        "\n".join(f"\\item {x}" for x in items) + "\n\\end{enumerate}")
            continue

        if not ln.strip():
            body.append(""); i += 1; continue

        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|\||```|---+\s*$|\s*[-*]\s)", lines[i]):
            para.append(lines[i]); i += 1
        body.append(inline(" ".join(x.strip() for x in para)))

    return "\n\n".join(body)


PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.75in]{geometry}
% cmap emits ToUnicode CMaps and lmodern supplies a font with real Unicode
% mappings for the fi/fl ligatures. Without both, "flatness" extracts as
% "latness" and "prefill" as "prell" -- a reviewer cannot text-search the
% submission, which a reviewer told us.
\usepackage{cmap}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage{textcomp,amsmath,amssymb,listings,parskip,graphicx}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,frame=single,
        columns=fullflexible,xleftmargin=2pt}
\setlength{\columnsep}{18pt}
\usepackage{sectsty}\allsectionsfont{\sffamily}
\pagestyle{plain}
\begin{document}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("paper.pdf"))
    args = ap.parse_args(argv)

    if not shutil.which("pdflatex"):
        print("[pdf] pdflatex not found", file=sys.stderr); return 1
    try:
        tex = PREAMBLE + convert(args.source.read_text()) + "\n\\end{document}\n"
    except ValueError as e:
        print(f"[pdf] {e}", file=sys.stderr); return 1

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "p.tex").write_text(tex)
        for _ in range(2):                      # twice, so any refs settle
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "p.tex"],
                               cwd=d, capture_output=True, text=True)
        pdf = d / "p.pdf"
        if not pdf.exists():
            tail = "\n".join(r.stdout.splitlines()[-25:])
            print(f"[pdf] pdflatex produced no output:\n{tail}", file=sys.stderr)
            return 1
        shutil.copy2(pdf, args.out)
    size = args.out.stat().st_size
    pages = subprocess.run(["pdfinfo", str(args.out)], capture_output=True, text=True).stdout
    npages = next((l.split()[-1] for l in pages.splitlines() if l.startswith("Pages")), "?")
    print(f"[pdf] {args.out}  {size / 1024:.0f} KB, {npages} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
