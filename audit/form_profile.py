#!/usr/bin/env python
"""Compare the form of a document against the sibling of the same kind.

The form of a document on this track is a measured object: the H1, the contents block, the section
headings, the column names and column counts of its tables, and how many tables, bold labels, bullets and
code fences each section carries. A new document is written to the sibling's profile, then diffed against
it here before the gate.

Usage
-----
    python tools/form_profile.py <sibling.md> <new.md> [--json <out>]

Exit status is 0 when the two documents have the same form, 1 when they do not.

What fails
----------
* a heading count, order or title that differs from the sibling;
* a contents line that differs (count, numbering or title);
* a column name used by the new document that the sibling never uses (an invented column). Only the
  first two columns of a table are compared: a third column names the statistic that table measures
  (`drift (rel_mse)`, `contribution (1 - cos)`), and the statistic is the checkpoint's own business;
* a table whose column count the sibling never uses;
* a table count per section that differs by more than one.

What is deliberately not judged
-------------------------------
* the number of measured rows inside a table;
* the wording of prose;
* the values, which are this run's own measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOC = re.compile(r"^\d+\.\s+\[")
HEAD = re.compile(r"^(#{1,3})\s+(.*)$")
TABLE_SEP = re.compile(r"^\|\s*:?-{2,}")
BOLD = re.compile(r"^\*\*[^*]+\*\*")


def _cells(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _body(lines):
    """Drop fenced code: a `#` comment inside a fence is not a heading."""
    out, in_fence = [], False
    for ln in lines:
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else ln)
    return out


def profile(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8").splitlines()
    lines = _body(raw)

    h1 = [ln[2:].strip() for ln in lines if ln.startswith("# ")]
    heads = [(len(m.group(1)), m.group(2).strip()) for m in
             (HEAD.match(ln) for ln in lines) if m and len(m.group(1)) >= 2]
    toc = [ln[ln.index("[") + 1: ln.rindex("]")] for ln in lines if TOC.match(ln)]

    columns = []
    widths = {}
    for i, ln in enumerate(lines):
        if TABLE_SEP.match(ln) and i:
            header = lines[i - 1]
            if header.startswith("|"):
                cells = tuple(_cells(header))
                columns.append(cells[:2])
                widths[len(cells)] = widths.get(len(cells), 0) + 1

    sections = []
    current = {"title": "(preamble)", "tables": 0, "bold": 0, "bullets": 0, "fences": 0}
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            current["fences"] += 1
            continue
        if in_fence:
            continue
        if ln.startswith("## "):
            sections.append(current)
            current = {"title": ln[3:].strip(), "tables": 0, "bold": 0, "bullets": 0, "fences": 0}
        elif TABLE_SEP.match(ln) and i and lines[i - 1].startswith("|"):
            current["tables"] += 1
        elif BOLD.match(ln):
            current["bold"] += 1
        elif ln.startswith("- "):
            current["bullets"] += 1
    sections.append(current)

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(raw),
        "h1": h1,
        "headings": heads,
        "toc": toc,
        "columns": columns,
        "column_widths": widths,
        "sections": sections,
        "tables": sum(s["tables"] for s in sections),
        "bold": sum(s["bold"] for s in sections),
        "bullets": sum(s["bullets"] for s in sections),
        "fences": sum(s["fences"] for s in sections),
    }


def compare(sib: dict, new: dict):
    errors, notes = [], []

    if len(sib["h1"]) != 1 or len(new["h1"]) != 1:
        errors.append("H1: sibling %d line(s), new %d - exactly one is the form"
                      % (len(sib["h1"]), len(new["h1"])))

    if len(sib["toc"]) != len(new["toc"]):
        errors.append("contents: sibling %d line(s), new %d" % (len(sib["toc"]), len(new["toc"])))
    for i, (a, b) in enumerate(zip(sib["toc"], new["toc"]), 1):
        if a != b:
            errors.append("contents line %d: %r vs %r" % (i, a, b))

    sib_titles = [t for _, t in sib["headings"]]
    new_titles = [t for _, t in new["headings"]]
    if len(sib_titles) != len(new_titles):
        errors.append("headings: sibling %d, new %d" % (len(sib_titles), len(new_titles)))
        for extra in new_titles[len(sib_titles):]:
            errors.append("heading not in the sibling: %r" % extra)
    for i, (a, b) in enumerate(zip(sib_titles, new_titles), 1):
        if a != b:
            errors.append("heading %d: %r vs %r" % (i, a, b))

    sib_cols = set(sib["columns"])
    for col in new["columns"]:
        if col not in sib_cols:
            errors.append("column set the sibling does not use: %s" % " | ".join(col))

    for w, n in sorted(new["column_widths"].items()):
        if w not in sib["column_widths"]:
            errors.append("%d-column table(s) x%d: the sibling has no %d-column table" % (w, n, w))

    for s in new["sections"]:
        twin = next((x for x in sib["sections"] if x["title"] == s["title"]), None)
        if twin is None:
            continue
        if abs(s["tables"] - twin["tables"]) > 1:
            errors.append("section %r: %d table(s) against the sibling's %d"
                          % (s["title"], s["tables"], twin["tables"]))

    for key in ("fences", "bullets"):
        if sib[key] and abs(new[key] - sib[key]) > 2:
            notes.append("%s: sibling %d, new %d" % (key, sib[key], new[key]))

    return errors, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sibling")
    ap.add_argument("new")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sib, new = profile(Path(args.sibling)), profile(Path(args.new))
    errors, notes = compare(sib, new)

    print("sibling  %s" % args.sibling)
    print("         %d B, %d lines, %d headings, %d contents lines, columns %s"
          % (sib["bytes"], sib["lines"], len(sib["headings"]), len(sib["toc"]),
             sorted(sib["column_widths"].items())))
    print("new      %s" % args.new)
    print("         %d B, %d lines, %d headings, %d contents lines, columns %s"
          % (new["bytes"], new["lines"], len(new["headings"]), len(new["toc"]),
             sorted(new["column_widths"].items())))

    for note in notes:
        print("[NOTE] %s" % note)
    for err in errors:
        print("[FAIL] %s" % err)
    print("FORM: %s" % ("PASS" if not errors else "FAIL"))

    if args.json:
        Path(args.json).write_text(
            json.dumps({"sibling": sib, "new": new, "errors": errors, "notes": notes,
                        "verdict": "PASS" if not errors else "FAIL"},
                       indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
