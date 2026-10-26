# -*- coding: utf-8 -*-
"""Check every citation in a verdict report against the file it points at.

A report row cites `record.json` -> `dotted.path` or `record.log` -> `label`.
This opens the record, resolves the path, and compares the value with the number
in the row's first cell. Rows whose citation cannot be resolved are failures, not
warnings: a citation that does not resolve is a claim with no record behind it.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    lines = open(args.report, encoding="utf-8").read().splitlines()
    ok = fail = skip = 0
    for line in lines:
        m = ROW.match(line)
        if not m:
            continue
        num_cell = m.group("num")
        rec_cell = m.group("rec")
        c = CITE.search(rec_cell)
        if not c:
            continue
        fname, field = c.group(1), c.group(2)
        path = os.path.join(args.root, fname)
        if not os.path.exists(path):
            print("FAIL  %-22s file missing: %s" % (num_cell.strip()[:22], fname))
            fail += 1
            continue
        claimed = cell_number(num_cell)
        if fname.endswith(".json"):
            obj = json.load(open(path, encoding="utf-8"))
            val, found = resolve(obj, field)
            if not found:
                print("FAIL  %-22s no such field: %s -> %s" % (num_cell.strip()[:22], fname, field))
                fail += 1
                continue
            if claimed is None or not isinstance(val, (int, float)):
                print("SKIP  %-22s non-numeric row (%s)" % (num_cell.strip()[:22], type(val).__name__))
                skip += 1
                continue
            if abs(val - claimed) <= max(0.00005, abs(val) * 5e-4) or round(val, 4) == claimed:
                ok += 1
            else:
                print("FAIL  %-22s %s -> %s = %r (claimed %r)" % (num_cell.strip()[:22], fname, field, val, claimed))
                fail += 1
        else:
            text = open(path, encoding="utf-8", errors="replace").read()
            if claimed is None:
                if field.split(":")[0][:20] in text:
                    ok += 1
                else:
                    print("FAIL  %-22s label not in %s: %s" % (num_cell.strip()[:22], fname, field))
                    fail += 1
            else:
                nums = []
                for tok in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, flags=re.I):
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        pass
                hit = any(abs(v - claimed) <= max(1e-6, abs(claimed) * 1e-2) for v in nums)
                if hit:
                    ok += 1
                else:
                    print("FAIL  %-22s %s has no value near %r" % (num_cell.strip()[:22], fname, claimed))
                    fail += 1

    print("\nresolved %d   failed %d   skipped %d" % (ok, fail, skip))
    print("CITATION CHECK: %s" % ("PASS" if fail == 0 else "FAIL"))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
