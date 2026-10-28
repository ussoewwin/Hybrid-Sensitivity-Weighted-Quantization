# -*- coding: utf-8 -*-
"""Check every citation in a verdict report against the file it points at.

Two passes, because one was not enough. Pass 1 reads four-column table rows and compares the
number in the first cell with the value the citation resolves to. Pass 2 sweeps the whole
document for `records/x.json` -> `field` citations: a citation written in prose, or in a table
that is not four columns wide, used to escape the check entirely - and one such citation named a
record file that did not exist while the tool printed PASS. Anything that cannot be resolved is
a failure, not a warning: a citation that does not resolve is a claim with no record behind it.
A citation into a log or txt file is matched on the labelled line only, and hashes are compared
as hash strings - a number elsewhere in the file, or the digits of an unrelated sha, proves
nothing.

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


def check_citation(fname, field, root, claimed, label, ok_fail, cell_text=""):
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
    # log/txt citation: the dotted label must name real line tokens, a cited hash is matched
    # as a hash (never as a float), and a cited number must sit on the labelled line.
    text = open(path, encoding="utf-8", errors="replace").read()
    label_tokens = [x for x in field.split(":")[0].split(".") if len(x) >= 2]
    if not label_tokens:
        ok_fail.append((label, "empty label in %s: %s" % (fname, field)))
        return None
    lines = [ln for ln in text.splitlines() if all(tok in ln for tok in label_tokens)]
    if not lines:
        ok_fail.append((label, "no line carrying %r in %s"
                        % (".".join(label_tokens), fname)))
        return None
    if claimed is None:
        return None
    hexes = re.findall(r"[0-9a-f]{6,}", str(cell_text).lower())
    if hexes:
        blob = " ".join(lines).lower()
        for h in hexes:
            if h not in blob:
                ok_fail.append((label, "%s line %r lacks %s"
                                % (fname, ".".join(label_tokens), h)))
                break
        return None
    nums = []
    for ln in lines:
        for tok in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", ln, flags=re.I):
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    if not any(abs(v - claimed) <= max(1e-6, abs(claimed) * 1e-2) for v in nums):
        ok_fail.append((label, "%s line %r has no value near %r"
                        % (fname, ".".join(label_tokens), claimed)))
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
        val = check_citation(fname, field, args.root, claimed, num_cell.strip()[:22],
                             problems, num_cell)
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
    cited_files = {c.group(1) for c in CITE.finditer(text)}
    for m in re.finditer(r"`([^`]+?\.(?:json|log|txt))`", text):
        name = m.group(1)
        p2 = os.path.join(args.root, name)
        if not os.path.exists(p2):
            # inventory lines sometimes give the bare file name; resolve that too
            p2 = os.path.join(args.root, "records", os.path.basename(name))
        if not os.path.exists(p2):
            print("FAIL  record named but not on disk: %s" % name)
            fail += 1
        elif name not in cited_files:
            print("WARN  named as inventory, no field cited: %s" % name)
    for c in CITE.finditer(text):
        fname, field = c.group(1), c.group(2)
        if (fname, field) in checked:
            continue
        checked.add((fname, field))
        problems = []
        check_citation(fname, field, args.root, None, "prose %s" % fname, problems, c.group(0))
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
