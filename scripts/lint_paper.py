#!/usr/bin/env python3
"""
Lint the paper — source AND rendered PDF — for the defect classes we have shipped.

Every check here exists because the defect reached a reader. Reading the draft
again is not a control: the figures were missing for fourteen drafts and the
doubled section numbers survived a reviewer reporting them, because I checked
with a grep that could not match and then reported "does not reproduce". §6 of
the paper argues that a pipeline can execute a number and cannot execute prose;
this is that argument applied to the manuscript itself.

Checks, each named after the failure that motivated it:

  FIG-MISSING    a referenced figure file does not exist
  FIG-UNRENDERED a figure is referenced but no image is embedded in the PDF
  FIG-LITERAL    figure markup survived into the PDF text as literal characters
  MD-LITERAL     markdown emphasis/code markers survived into the PDF text
  NUM-DOUBLED    a heading shows two section numbers ("5.8 4.8 Model scale")
  NUM-DRIFT      a heading's rendered number differs from its authored one, so
                 every in-text cross-reference points one section off
  LIGATURE       fi/fl ligatures are not extractable ("latness" for "flatness")
  CELL-DROPPED   a numeric table cell in the source is absent from the PDF
  DRAFT-VOICE    references to our own unpublished drafts, meaningless to a reader
  NOTEBOOK-VOICE first-person narration of the research process
  SELF-ADDRESSED editorial notes to self ("stated once", "note to self")
  ACRONYM        an acronym used before it is expanded
  XREF           a cross-reference pointing at a section or table that is gone
  TABLE-UNCAPTIONED  a table with no caption, so it cannot be numbered or cited

Exit non-zero on any finding, so `reproduce_all.sh` fails closed.

Usage:
  python scripts/lint_paper.py notes/paper_draft.md paper.pdf
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# Phrases that address the reader about drafts they have never seen, or narrate
# the authors' journey. Both were flagged by multiple reviewers before removal.
DRAFT_VOICE = [
    r"earlier draft", r"an earlier version", r"previous draft", r"we published earlier",
    r"in a previous version", r"this draft", r"the last draft",
]
NOTEBOOK_VOICE = [
    r"we set out to", r"six sessions in", r"it turned out that we",
    r"we then realised", r"we spent \w+ sessions", r"our first attempt",
    r"we had assumed", r"we were wrong about",
]
SELF_ADDRESSED = [
    r"stated once", r"note to self", r"TODO", r"FIXME", r"XXX",
    r"we should", r"needs rewriting", r"rewrite this",
]
# Colloquial or literary register. A reviewer objected that passages read as
# narrative rather than as technical prose, and named these as instances. Each
# has a plain replacement: "not a free lunch" -> "a point on a trade-off curve",
# "a hair of headroom" -> "a small reduction", "the tell" -> "the indication".
REGISTER = [
    r"free lunch", r"a hair of", r"\bthe tell\b", r"flourish", r"lopsided",
    r"least comfortable", r"whatever the batch", r"this kind of work",
    r"surviv\w* contact with", r"\bslack the server had anyway\b",
    r"\bnice\b", r"\bcool\b", r"\bhuge\b", r"\bawesome\b",
    # classes found in the full-paper scan
    r"groping", r"hides under", r"\bshipped\b", r"deserves its own",
    r"\bkilled\b", r"wrong way round", r"want re-measuring", r"rides? inside",
    r"tempting argument", r"honest caveat", r"worth saying", r"\bdies\b",
    r"is a warning", r"the effect lives", r"says why", r"prediction missed",
    r"\bpathology\b", r"hide in", r"no longer claim", r"no longer describe",
]
# A percentage belongs to a change in a quantity, not to the quantity itself.
# "46% of p50 latency" is not English; "reduces p50 latency by 46%" is.
UNITS_MISUSE = [
    # Only latency/throughput metrics: "8.8% of KV capacity" is a correct
    # statement about a fraction of a resource, whereas "46% of p50 latency"
    # means a reduction and is not English.
    r"\d+(?:\.\d+)?% of (?:p50|p95|p99|median|end-to-end|e2e|goodput|throughput|"
    r"tail latency)",
]
# Single-clause dramatic declaratives. Kept as an explicit list rather than a
# pattern, because a general rule for "short sentence" flags correct prose.
DRAMATIC = [
    r"\bIt is wrong\.", r"\bNeither holds\.", r"\bIt does not\.(?= )",
    r"\bIt falls by \d", r"\bThat is that\.", r"\bFull stop\.",
]
# Acronym -> the expansion that must appear at or before first bare use.
ACRONYMS = {
    "TP": "tensor-parallel", "RPA": "Ragged Paged Attention",
    "MAPE": "mean absolute percentage error", "MFU": "model",
    "TTFT": "time to first token", "ITL": "inter-token",
    "KV": None, "HBM": "high-bandwidth", "XLA": None, "MXU": None,
}


def pdftext(pdf: pathlib.Path, layout: bool = False) -> str:
    # Plain extraction reflows the two columns into reading order, which is what
    # the prose checks want. It also separates a section heading from its number,
    # so the numbering check must read -layout output instead, where the number
    # and the title stay on one line.
    cmd = ["pdftotext"] + (["-layout"] if layout else []) + [str(pdf), "-"]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return r.stdout


def norm(s: str) -> str:
    for a, b in (("–", "-"), ("—", "-"), ("−", "-"), ("§", "S"), ("…", "..."),
                 ("µ", "u"), ("×", "x"), ("≈", "~"), ("≤", "<="), ("’", "'"),
                 (" ", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("pdf", type=pathlib.Path, nargs="?")
    args = ap.parse_args(argv)

    md = args.source.read_text()
    root = args.source.resolve().parent.parent
    findings: list[tuple[str, str]] = []

    def flag(code: str, msg: str) -> None:
        findings.append((code, msg))

    # --- source-only checks -------------------------------------------------
    figs = re.findall(r"\[Figure\s+(\d+)[^\]]*?`?([\w./-]+\.png)`?\]", md)
    for num, path in figs:
        if not (root / path).exists():
            flag("FIG-MISSING", f"Figure {num} references {path}, which does not exist")

    for pat in DRAFT_VOICE:
        for m in re.finditer(pat, md, re.I):
            flag("DRAFT-VOICE", f"{pat!r} at char {m.start()}: "
                                f"{md[max(0, m.start() - 40):m.start() + 40]!r}")
    for pat in NOTEBOOK_VOICE:
        for m in re.finditer(pat, md, re.I):
            flag("NOTEBOOK-VOICE", f"{pat!r}: {md[max(0, m.start() - 30):m.start() + 50]!r}")
    for pat in SELF_ADDRESSED:
        for m in re.finditer(pat, md):
            flag("SELF-ADDRESSED", f"{pat!r}: {md[max(0, m.start() - 30):m.start() + 40]!r}")
    for pat in REGISTER:
        for m in re.finditer(pat, md, re.I):
            flag("REGISTER", f"{pat!r}: {md[max(0, m.start() - 35):m.start() + 45]!r}")
    for pat in UNITS_MISUSE:
        for m in re.finditer(pat, md, re.I):
            flag("UNITS-PHRASING",
                 f"a percentage is of a CHANGE, not of the quantity: "
                 f"{md[max(0, m.start() - 20):m.start() + 60]!r}")
    for pat in DRAMATIC:
        for m in re.finditer(pat, md):
            flag("DRAMATIC", f"{pat!r}: {md[max(0, m.start() - 45):m.start() + 35]!r}")

    # Acronym expanded at or before first bare use. Search a whitespace-collapsed
    # copy: expansions routinely wrap across lines ("mean absolute percentage\n
    # error"), and a literal search misses them, which reported a defect the
    # paper did not have.
    flatmd = re.sub(r"\s+", " ", md).lower()
    offs, run = [], 0                       # map flat index -> original index
    for ch in md:
        offs.append(run)
        run += 1
    for ac, expansion in ACRONYMS.items():
        if expansion is None:
            continue
        uses = [m.start() for m in re.finditer(rf"\b{ac}\b", md)]
        if not uses:
            continue
        first_flat = len(re.sub(r"\s+", " ", md[:uses[0]]))
        exp = flatmd.find(re.sub(r"\s+", " ", expansion).lower())
        if exp < 0 or exp > first_flat:
            flag("ACRONYM", f"{ac} used at char {uses[0]} before {expansion!r} appears")

    # Every table needs a caption, so it can be numbered and referred to. A bare
    # table was the state of all twenty-four before a reviewer said so.
    md_lines = md.split("\n")
    for i, l in enumerate(md_lines):
        if i + 1 < len(md_lines) and re.match(r"^\|[\s:|-]+\|\s*$", md_lines[i + 1]) \
                and l.startswith("|"):
            prev = md_lines[i - 1].strip() if i else ""
            if not prev.startswith("Table:"):
                flag("TABLE-UNCAPTIONED",
                     f"table at line {i + 1} has no 'Table:' caption line: {l[:60]!r}")
    # A cross-reference must point at a label that exists.
    labels = set(re.findall(r"\{#(tab:[\w-]+)\}", md))
    for m in re.finditer(r"\[(tab:[\w-]+)\]", md):
        if m.group(1) not in labels:
            flag("XREF", f"[{m.group(1)}] referenced but no table carries that label")

    # Cross-references must point at sections that exist.
    heads = set()
    for m in re.finditer(r"^#{2,4}\s+(\d+(?:\.\d+)*)", md, re.M):
        heads.add(m.group(1))
    for m in re.finditer(r"§(\d+(?:\.\d+)*)", md):
        if m.group(1) not in heads:
            flag("XREF", f"§{m.group(1)} referenced but no such section heading")

    # --- rendered-PDF checks ------------------------------------------------
    if args.pdf and args.pdf.exists():
        txt = pdftext(args.pdf)
        flat = norm(txt)

        if figs:
            n_img = 0
            r = subprocess.run(["pdfimages", "-list", str(args.pdf)],
                               capture_output=True, text=True, errors="replace")
            for ln in r.stdout.splitlines()[2:]:
                if len(ln.split()) > 2 and ln.split()[2] == "image":
                    n_img += 1
            if n_img < len(figs):
                flag("FIG-UNRENDERED",
                     f"{len(figs)} figures referenced, {n_img} images embedded in the PDF")

        for pat, code in ((r"\[Figure\s+\d+", "FIG-LITERAL"),
                          (r"\*\*", "MD-LITERAL"),
                          (r"(?<!\w)`\w", "MD-LITERAL")):
            hits = re.findall(pat, txt)
            if hits:
                flag(code, f"{len(hits)} occurrence(s) of {pat!r} survived into the PDF text")

        # Only a HEADING can show doubled numbers: two section-like numbers
        # followed by a capitalised word on one line. Matching bare numbers on
        # consecutive lines flags every table, which is how a looser version of
        # this check produced 52 false positives and hid the real ones.
        for m in re.finditer(r"^ *(\d+(?:\.\d+)+) +(\d+(?:\.\d+)+) +([A-Z][a-z]+)",
                             txt, re.M):
            flag("NUM-DOUBLED", f"heading shows two numbers: {m.group(0).strip()[:60]!r}")

        # A heading's authored number must be the number the PDF shows. LaTeX
        # numbered the unnumbered Abstract as 1, so Introduction rendered as 2
        # and Results as 5 while every in-text "S4.x" still said 4 -- ~40 dead
        # cross-references, none visible in the source. XREF above checks the
        # markdown against itself and cannot see this; only the render can.
        lay = norm(pdftext(args.pdf, layout=True))
        for m in re.finditer(r"^#{2,3}\s+(\d+(?:\.\d+)*)\.?\s+(\S+(?:\s+\S+){0,2})",
                             md, re.M):
            num, words = m.group(1), norm(m.group(2))
            first = words.split()[0].rstrip(":,")
            if len(first) < 4:
                continue
            # A heading's own words recur in body text and cross-references, and
            # a number happens to sit before some of those too ("...see S7.
            # Related work covers..."). Taking the FIRST occurrence flagged
            # Related work as misnumbered when the heading itself was correct.
            # So look at every occurrence and flag only if NONE carries the
            # authored number: one correct rendering is proof of no drift.
            seen: list[str] = []
            for pm in re.finditer(re.escape(first), lay):
                before = lay[max(0, pm.start() - 12):pm.start()]
                bm = re.search(r"(\d+(?:\.\d+)*)\s*$", before)
                if bm:
                    if bm.group(1) == num:
                        seen = []
                        break
                    seen.append(bm.group(1))
            if seen:
                flag("NUM-DRIFT",
                     f"heading {num!r} ({first}) never appears with its own "
                     f"number in the PDF; saw {sorted(set(seen))[:3]}")

        broken = re.findall(r"\b(?:latness|prell|rooine|dierence|conguration|"
                            r"signicant|nding|dened|rst|eciency)\b", txt)
        if broken:
            flag("LIGATURE", f"{len(broken)} unextractable ligature(s), e.g. {broken[:4]}")

        cells = {re.sub(r"[*`]", "", c).strip()
                 for ln in md.split("\n") if ln.startswith("|") and "---" not in ln
                 for c in ln.strip().strip("|").split("|")}
        cells = {c for c in cells if re.search(r"\d", c) and len(c) < 24}
        missing = [c for c in sorted(cells) if norm(c) not in flat]
        for c in missing:
            flag("CELL-DROPPED", f"table cell {c!r} not found in the PDF text")

    # --- report -------------------------------------------------------------
    if not findings:
        print(f"[lint] {args.source.name}: clean"
              + (f" ({len(figs)} figures rendered)" if figs else ""))
        return 0
    by: dict[str, list[str]] = {}
    for code, msg in findings:
        by.setdefault(code, []).append(msg)
    for code in sorted(by):
        print(f"[lint] {code}: {len(by[code])}")
        for m in by[code][:4]:
            print(f"         {m}")
        if len(by[code]) > 4:
            print(f"         ... and {len(by[code]) - 4} more")
    print(f"[lint] {len(findings)} finding(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
