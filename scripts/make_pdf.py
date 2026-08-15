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
    "\u2020": r"$\dagger$", "\u2021": r"$\ddagger$", "\u0394": r"$\Delta$", "\u03c3": r"$\sigma$",
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


# A value and its unit must not be split across a line break. siunitx would do
# this and much else; a non-breaking space is the part that affects the reader.
UNIT_RE = re.compile(r"(?<=\d)[ ](?=(?:µs|ms|ns|s|MB|GB|KB|TFLOP/s|GB/s|req/s|"
                     r"tokens|FLOP/byte)\b)")


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
    # [tab:slug] -> a real cross-reference, so prose can name a table by number
    t = re.sub(r"\[(tab:[\w-]+)\]", r"Table~\\ref{\1}", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", t)
    for i, c in enumerate(holds):
        t = t.replace(f"\x00{i}\x00", r"\texttt{" + esc(c) + "}")
    t = UNIT_RE.sub("~", t)
    return t


# A figure reference in the draft looks like
#   **[Figure 1 — `figures/fig1_lens.png`]** *caption text...*
# Thirteen drafts rendered that as literal text because this converter had no
# image handling at all. The PNGs existed on disk the whole time.
FIGRE = re.compile(r"^\*\*\[Figure (\d+)[^\]]*?`?([\w./-]+\.png)`?\]\*\*\s*(.*)$")


def convert(md: str) -> str:
    lines = md.split("\n")
    body: list[str] = []
    pending_caption: tuple[str, str] | None = None
    i = 0
    while i < len(lines):
        ln = lines[i]

        # A table caption, captured here rather than after paragraph processing.
        # Read from `body` it has already been through inline(), which escapes
        # "{#tab:x}" into "\\{\\#tab:x\\}" -- so the label never matched and the
        # escaped text leaked into the caption instead.
        mcap = re.match(r"^Table:\s*(.*?)\s*(?:\{#(tab:[\w-]+)\})?\s*$", ln)
        if mcap and ln.startswith("Table:"):
            pending_caption = (mcap.group(1), mcap.group(2) or "")
            i += 1
            continue

        # A figure line: emit an actual float, not the literal markdown.
        mfig = FIGRE.match(ln)
        if mfig:
            num, path, cap = mfig.groups()
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```")):
                cap += " " + lines[i].strip(); i += 1
            cap = inline(cap.strip().strip("*"))
            body.append("\\begin{figure}[t]\\centering\n"
                        f"\\includegraphics[width=\\columnwidth]{{{path}}}\n"
                        f"\\caption{{{cap}}}\n\\end{{figure}}")
            continue

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

        m = re.match(r"^(#{1,5})\s+(.*)$", ln)
        if m:
            # LaTeX numbers sections itself, and the markdown heading carries its
            # own number, so "### 4.8 Model scale" rendered as "5.8 4.8 Model
            # scale". Strip the authored number and let LaTeX own it.
            #
            # But ONLY headings that were authored with a number may be numbered
            # by LaTeX. "## Abstract" has none, and numbering it consumed the 1,
            # pushing Introduction to 2 and Results to 5 while every in-text
            # cross-reference still said 4 -- roughly forty pointers off by one,
            # in the PDF only. The markdown was self-consistent, which is why a
            # linter that checked the source found nothing. An unnumbered heading
            # must render unnumbered, so the authored numbering IS the rendered
            # numbering.
            raw = re.sub(r"^\d+(\.\d+)*\.?\s+", "", m.group(2))
            numbered = raw != m.group(2)
            lvl, txt = len(m.group(1)), inline(raw)
            # Level 5 is a run-in \paragraph: a topic title for a block of
            # argument, unnumbered, so a section can be scanned without every
            # claim being promoted to a numbered subsection.
            cmd = {1: "title", 2: "section", 3: "subsection",
                   4: "subsubsection", 5: "paragraph"}[lvl]
            star = "" if numbered else "*"
            body.append(f"\\{cmd}{star}{{{txt}}}" if cmd != "title"
                        else f"\\begin{{center}}\\LARGE\\bfseries {txt}\\end{{center}}")
            i += 1; continue

        # pipe table: a header row followed by a |---| separator
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            # A caption line may sit immediately above the table:
            #   Table: what it shows {#tab:slug}
            # Without one a table renders bare, which is what a reviewer objected
            # to: no number, no caption, and nothing prose can refer to.
            caption = label = ""
            if pending_caption:
                caption, label = pending_caption
                pending_caption = None
            elif body and isinstance(body[-1], str):
                raw_cap = body[-1].strip()
                # Try the labelled form first. A single optional group lets the
                # non-greedy caption swallow "{#tab:x}" and silently drop every
                # label, which is what happened: 24 captions, 0 labels.
                mcap = re.match(r"^Table:\s*(.*?)\s*\{#(tab:[\w-]+)\}\s*$", raw_cap)
                if mcap:
                    caption, label = mcap.group(1), mcap.group(2)
                    body.pop()
                elif raw_cap.startswith("Table:"):
                    caption, label = raw_cap[6:].strip(), ""
                    body.pop()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, data = rows[0], rows[2:]
            ncol = max(len(r) for r in rows)
            spec = "l" + "r" * (ncol - 1)
            # Wide tables must stay READABLE, not merely fit. Shrinking every
            # table to one column width with \resizebox made a seven-column
            # table render at perhaps five point -- legible to a script checking
            # that the cells survived, and not to a reader. The rule now:
            #
            #   narrow  -> one column, \small, no scaling
            #   wide    -> a table* float spanning both columns at \footnotesize
            #
            # \resizebox is kept only as a last resort for a table too wide even
            # for the full text width, because the alternative is silent
            # clipping: the flag-on column of the mechanism table vanished from
            # the first rendered PDF and only a digit count against the source
            # caught it.
            widest = max(sum(len(c) for c in r) + 3 * ncol for r in rows)
            longest_cell = max(len(c) for r in rows for c in r)
            cap_tex = ""
            if caption:
                cap_tex = ("\\caption{" + inline(caption) + "}"
                           + ("\\label{" + label + "}" if label else ""))
            if longest_cell > 34:
                # Long prose cells must WRAP. tabularx does that, but only for
                # the columns declared X, and the wide column is NOT always the
                # last one: in the failure taxonomy it is the third of four, so
                # putting X last gave it negative width and clipped it silently
                # ("what it isprovenance", "used under another" -> "one confi").
                # Declare X for EVERY column that carries long text, so the slack
                # is distributed among the columns that actually need it.
                colmax = [max(len(r[c]) if c < len(r) else 0 for r in rows)
                          for c in range(ncol)]
                wide = [c for c, w in enumerate(colmax) if w > 20]
                if not wide:
                    wide = [ncol - 1]
                # Equal X columns split the line evenly, which is wrong when one
                # column carries three times the text of another: the taxonomy
                # table gave "class" as much width as "what it is". Weight each
                # X column by its own content length, normalised so the weights
                # sum to the number of X columns, which is what tabularx expects.
                tot = sum(colmax[c] for c in wide) or 1
                parts = []
                for c in range(ncol):
                    if c in wide:
                        w = len(wide) * colmax[c] / tot
                        parts.append(f">{{\\hsize={w:.2f}\\hsize}}X")
                    else:
                        parts.append("l")
                xspec = "".join(parts)
                # Wrapping is not enough when the whole table is wide: four prose
                # columns squeezed into one text column leave each a few
                # characters across. Give those the full page width as well.
                if widest > 85:
                    t = ["\\begin{table*}[tbp]\\centering\\small",
                         cap_tex,
                         "\\begin{tabularx}{\\textwidth}{" + xspec + "}\\hline"]
                    close = "\\hline\\end{tabularx}\\end{table*}"
                else:
                    t = ["\\begin{table}[tbp]\\centering\\small",
                         cap_tex,
                         "\\begin{tabularx}{\\columnwidth}{" + xspec + "}\\hline"]
                    close = "\\hline\\end{tabularx}\\end{table}"
            elif ncol >= 6 or widest > 48:
                # Short cells, too many for one column: span the page at a
                # readable size rather than scaling the whole table down.
                t = ["\\begin{table*}[tbp]\\centering\\footnotesize",
                     cap_tex,
                     "\\begin{tabular}{" + spec + "}\\hline"]
                close = "\\hline\\end{tabular}\\end{table*}"
            else:
                t = ["\\begin{table}[tbp]\\centering\\small",
                     cap_tex,
                     "\\begin{tabular}{" + spec + "}\\hline"]
                close = "\\hline\\end{tabular}\\end{table}"
            t.append(" & ".join(f"\\textbf{{{inline(c)}}}" for c in head) + r" \\ \hline")
            for r in data:
                r = r + [""] * (ncol - len(r))
                t.append(" & ".join(inline(c) for c in r) + r" \\")
            t.append(close)
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
                r"^(#{1,5}\s|\||```|---+\s*$|\s*[-*]\s)", lines[i]):
            para.append(lines[i]); i += 1
        body.append(inline(" ".join(x.strip() for x in para)))

    out = "\n\n".join(body)

    # Every authored heading must produce a heading command. A level-5 heading
    # sitting inside a paragraph block was absorbed as body text and disappeared
    # from the structure with no error -- 28 authored, 27 emitted -- and it still
    # appeared in the PDF as prose, so a check for its TEXT could not see it.
    # Counting the commands is what catches it.
    # Count headings in the source with fenced blocks removed: a "#####" line
    # inside a code fence is not a heading, and counting it made this check
    # report a drop that had not happened.
    md_nofence = re.sub(r"```.*?```", "", md, flags=re.S)
    authored = len(re.findall(r"^#{2,5}\s+\S", md_nofence, re.M))
    emitted = sum(out.count(f"\\{c}") for c in
                  ("section{", "section*{", "subsection{", "subsection*{",
                   "subsubsection{", "subsubsection*{", "paragraph{", "paragraph*{"))
    # \subsection also matches inside \subsubsection, so count precisely instead.
    emitted = len(re.findall(r"\\(?:sub)*section\*?\{|\\paragraph\*?\{", out))
    if emitted != authored:
        raise ValueError(
            f"{authored} headings authored but {emitted} emitted — one was "
            "absorbed into a paragraph or list block")
    return out


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
\usepackage{textcomp,amsmath,amssymb,listings,parskip,graphicx,tabularx}
\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,frame=single,
        columns=fullflexible,xleftmargin=2pt}
\setlength{\columnsep}{18pt}
\usepackage{sectsty}\allsectionsfont{\sffamily}
\graphicspath{{GRAPHICSROOT/}}
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
        root = args.source.resolve().parent.parent
        tex = (PREAMBLE.replace("GRAPHICSROOT", str(root))
               + convert(args.source.read_text()) + "\n\\end{document}\n")
    except ValueError as e:
        print(f"[pdf] {e}", file=sys.stderr); return 1

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "p.tex").write_text(tex)
        for _ in range(2):                      # twice, so any refs settle
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "p.tex"],
                               cwd=d, capture_output=True, text=True,
                               errors="replace")
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
