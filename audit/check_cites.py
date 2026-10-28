# -*- coding: utf-8 -*-
"""Check every citation in a verdict report against the file it points at.

Two passes, because one was not enough. Pass 1 reads four-column table rows and compares the
number in the first cell with the value the citation resolves to. Pass 2 sweeps the whole
document for `records/x.json` -> `field` citations: a citation written in prose, or in a table
that is not four columns wide, used to escape the check entirely - and one such citation named a
record file that did not exist while the tool printed PASS. Anything that cannot be resolved is
a failure, not a warning: a citation that does not resolve is a claim with no record behind it.

    python check_cites.py <report.md> --root <dir holding the records>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROW = re.compile(r"^\|\s*(?P<num>[^|]+?)\s*\|\s*(?P<what>[^|]+?)\s*\|\s*(?P<rec>[^|]+?)\s*\|\s*(?P<cause>[^|]+?)\s*\|\s*$")
CITE = re.compile(r"`([^`]+?\.(?:json|log|txt))`\s*(?:→|->)\s*`([^`]+)`")
NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", re.I)


def resolve(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None, False
    return cur, True


def cell_number(cell):
    s = cell.replace(",", "").replace("−", "-")
    m = NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def check_citation(fname, field, root, claimed, label, ok_fail):
    """Resolve one citation. Appends (label, problem) to ok_fail when it fails."""
    path = os.path.join(root, fname)
    if not os.path.exists(path):
        ok_fail.append((label, "file missing: %s" % fname))
        return None
    if fname.endswith(".json"):
        obj = json.load(open(path, encoding="utf-8"))
        val, found = resolve(obj, field)
        if not found:
            ok_fail.append((label, "no such field: %s -> %s" % (fname, field)))
        return val
    text = open(path, encoding="utf-8", errors="replace").read()
    if claimed is None:
        if field.split(":")[0][:20] not in text:
            ok_fail.append((label, "label not in %s: %s" % (fname, field)))
        return None
    nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.I)]
    if not any(abs(v - claimed) <= max(1e-6, abs(claimed) * 1e-2) for v in nums):
        ok_fail.append((label, "%s has no value near %r" % (fname, claimed)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    text = open(args.report, encoding="utf-8").read()
    lines = text.splitlines()
    ok = fail = skip = 0
    checked = set()

    # ---- pass 1: four-column table rows, with the number compared ----
    for line in lines:
        m = ROW.match(line)
        if not m:
            continue
        num_cell = m.group("num")
        c = CITE.search(m.group("rec"))
        if not c:
            continue
        fname, field = c.group(1), c.group(2)
        claimed = cell_number(num_cell)
        problems = []
        val = check_citation(fname, field, args.root, claimed, num_cell.strip()[:22], problems)
        checked.add((fname, field))
        if problems:
            print("FAIL  %-22s %s" % (problems[0][0], problems[0][1]))
            fail += 1
            continue
        if val is None or claimed is None or not isinstance(val, (int, float)):
            print("SKIP  %-22s non-numeric row (%s)" % (num_cell.strip()[:22],
                                                        type(val).__name__))
            skip += 1
            continue
        if abs(val - claimed) <= max(0.00005, abs(val) * 5e-4) or round(val, 4) == claimed:
            ok += 1
        else:
            print("FAIL  %-22s %s -> %s = %r (claimed %r)"
                  % (num_cell.strip()[:22], fname, field, val, claimed))
            fail += 1

    # ---- pass 2: every citation anywhere in the document ----
    for c in CITE.finditer(text):
        fname, field = c.group(1), c.group(2)
        if (fname, field) in checked:
            continue
        checked.add((fname, field))
        problems = []
        check_citation(fname, field, args.root, None, "prose %s" % fname, problems)
        if problems:
            print("FAIL  prose citation  %s" % problems[0][1])
            fail += 1
        else:
            ok += 1

    print("\nresolved %d   failed %d   skipped %d   citations checked %d"
          % (ok, fail, skip, len(checked)))
    print("CITATION CHECK: %s" % ("PASS" if fail == 0 else "FAIL"))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
