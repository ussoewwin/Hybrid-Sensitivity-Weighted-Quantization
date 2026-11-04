# -*- coding: utf-8 -*-
"""Check every citation in a verdict report against the file it points at.

Two passes, because one was not enough. Pass 1 reads four-column table rows and compares the
number or identifier in the first cell with the value the citation resolves to. Pass 2 sweeps
the whole document for `records/x.json` -> `field` citations: a citation written in prose, or
in a table that is not four columns wide, used to escape the check entirely.

Supports SHA-256 string verification, small numeric tolerances, list/dict field matches,
and an --all flag to audit every verdict report in md/backdoor/verdicts/.

    python check_cites.py <report.md> --root <dir holding the records>
    python check_cites.py --all --root md/backdoor/verdicts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROW = re.compile(r"^\|\s*(?P<num>[^|]+?)\s*\|\s*(?P<what>[^|]+?)\s*\|\s*(?P<rec>[^|]+?)\s*\|\s*(?P<cause>[^|]+?)\s*\|\s*$")
CITE = re.compile(r"`([^`]+?\.(?:json|log|txt))`\s*(?:→|->)\s*`([^`]+)`")
NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", re.I)


def tol(value: float) -> float:
    """Tolerance for a claimed number: 5e-5 absolute, 5e-5 relative.

    Tightened from the earlier 5e-4 relative band (plan 4.2): a claim that needs
    a wider band than this is a claim the record does not actually support.
    """
    return max(5e-5, abs(value) * 5e-5)


def decimals_in(raw: str) -> int:
    """Decimals of the first number written in the cell (0 when it carries none)."""
    m = re.search(r"\d+\.(\d+)", raw or "")
    return len(m.group(1)) if m else 0


def numeric_match(claimed: float, actual: float, raw: str = "") -> bool:
    """A claim must sit inside the rounding interval of the precision it is written at.

    The band is max(5e-5 absolute, 5e-5 relative, half a unit of the last written
    digit) - so a value printed to three decimals may differ by at most 5e-4, and a
    claim printed to full precision must match to 5e-5.
    """
    half_unit = 0.5 * (10.0 ** -decimals_in(raw)) if raw else 0.0
    return abs(actual - claimed) <= max(tol(actual), half_unit)


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
    # strip markdown formatting like bold/backticks
    s = cell.replace("**", "").replace("`", "").replace(",", "").replace("−", "-").strip()
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
        if "/" in field and " -> " not in field:
            # derived value: the checker recomputes the ratio, the report never asserts it
            num, den = field.split("/", 1)
            a, ok_a = resolve(obj, num.strip())
            b, ok_b = resolve(obj, den.strip())
            if not ok_a or not ok_b:
                ok_fail.append((label, "no such field: %s -> %s" % (fname, field)))
                return None
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b == 0:
                ok_fail.append((label, "derived value not numeric: %s -> %s" % (fname, field)))
                return None
            return ("derived", a / b)
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
    hexes = re.findall(r"[0-9a-f]{6,}", str(cell_text).lower())
    if hexes:
        blob = " ".join(lines).lower()
        for h in hexes:
            if h not in blob:
                ok_fail.append((label, "%s line %r lacks %s"
                                % (fname, ".".join(label_tokens), h)))
                return None
        return ("matched_hex", hexes)
    nums = []
    for ln in lines:
        for tok in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", ln, flags=re.I):
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    if claimed is not None:
        matched = [v for v in nums if numeric_match(claimed, v)]
        if not matched:
            ok_fail.append((label, "%s line %r has no value near %r"
                            % (fname, ".".join(label_tokens), claimed)))
            return None
        return ("matched_num", matched[0])
    return ("matched_log_line", lines[0])


def verify_report(report_path, root):
    """Verify citations in a single report."""
    text = open(report_path, encoding="utf-8").read()
    lines = text.splitlines()
    ok = fail = skip = 0
    checked = set()

    # ---- pass 1: four-column table rows, with the number or string compared ----
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
        val = check_citation(fname, field, root, claimed, num_cell.strip()[:22],
                             problems, num_cell)
        checked.add((fname, field))
        if problems:
            print("FAIL  %-22s %s" % (problems[0][0], problems[0][1]))
            fail += 1
            continue

        # Matched tuple from log/txt check
        if isinstance(val, tuple) and val[0].startswith("matched_"):
            ok += 1
            continue

        # Derived ratio: compare the recomputed value, never a hand-written one
        if isinstance(val, tuple) and val[0] == "derived":
            ratio = val[1]
            if claimed is not None and numeric_match(claimed, ratio, num_cell):
                ok += 1
            else:
                print("FAIL  %-22s %s -> %s = %r (recomputed, claimed %r)"
                      % (num_cell.strip()[:22], fname, field, round(ratio, 6), claimed))
                fail += 1
            continue

        # Numeric value from json
        if isinstance(val, (int, float)) and claimed is not None:
            if numeric_match(claimed, val, num_cell):
                ok += 1
            else:
                print("FAIL  %-22s %s -> %s = %r (claimed %r)"
                      % (num_cell.strip()[:22], fname, field, val, claimed))
                fail += 1
            continue

        # String value (SHA hash, dtype name, etc.)
        if isinstance(val, str):
            # If cell claims a number and string represents a number:
            if claimed is not None:
                try:
                    fval = float(val.replace(",", "").strip())
                    if numeric_match(claimed, fval, num_cell):
                        ok += 1
                        continue
                except ValueError:
                    pass
            # Hex / SHA hash exact prefix/suffix match
            cell_hexes = re.findall(r"[0-9a-f]{6,}", num_cell.lower())
            vlow = val.lower()
            if cell_hexes and all(h == vlow or vlow.startswith(h) for h in cell_hexes):
                ok += 1
                continue
            # an abbreviated hash (`4d3e6c8c...c8c13e`) states its head and tail
            if cell_hexes and len(cell_hexes) >= 2 and ("..." in num_cell or "\u2026" in num_cell) \
                    and vlow.startswith(cell_hexes[0]) and vlow.endswith(cell_hexes[-1]):
                ok += 1
                continue
            # Textual match
            clean_cell = num_cell.replace("`", "").replace("**", "").strip().lower()
            clean_val = val.strip().lower()
            if clean_val in clean_cell or clean_cell in clean_val:
                ok += 1
                continue
            print("FAIL  %-22s %s -> %s string mismatch: %r"
                  % (num_cell.strip()[:22], fname, field, val))
            fail += 1
            continue

        # Dict value (e.g., dtypes dict or breakdown dict)
        if isinstance(val, dict):
            clean_cell = num_cell.replace("`", "").replace("**", "").strip()
            if any(k.lower() in clean_cell.lower() for k in val.keys()):
                ok += 1
                continue
            # Check if numbers in cell match values in dict
            cell_nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", clean_cell)]
            dict_vals = [float(v) for v in val.values() if isinstance(v, (int, float))]
            if cell_nums and any(cn in dict_vals for cn in cell_nums):
                ok += 1
                continue
            print("FAIL  %-22s dict has no key or value the cell states" % num_cell.strip()[:22])
            fail += 1
            continue

        # List value
        if isinstance(val, list):
            clean_cell = num_cell.replace("`", "").replace("**", "").strip()
            if claimed == 0 and len(val) == 0:
                ok += 1
                continue
            if claimed is not None and claimed == len(val):
                ok += 1
                continue
            flat_nums = []
            flat_texts = []
            for item in val:
                if isinstance(item, (int, float)):
                    flat_nums.append(float(item))
                elif isinstance(item, dict):
                    for kk, vv in item.items():
                        if isinstance(vv, (int, float)):
                            flat_nums.append(float(vv))
                        elif isinstance(vv, str):
                            flat_texts.append(vv.strip().lower())
                elif isinstance(item, str):
                    flat_texts.append(item.strip().lower())
            if claimed is not None and any(numeric_match(claimed, v, num_cell) for v in flat_nums):
                ok += 1
                continue
            # text row: every token the cell lists must occur in the record's own strings
            tokens = [x.strip().lower() for x in re.split(r"[/,]", clean_cell) if len(x.strip()) >= 3]
            if tokens and all(any(tok in txt for txt in flat_texts) for tok in tokens):
                ok += 1
                continue
            print("FAIL  %-22s list holds no element/value the cell states" % num_cell.strip()[:22])
            fail += 1
            continue

        print("FAIL  %-22s unverifiable value type (%s)" % (num_cell.strip()[:22], type(val).__name__))
        fail += 1

    # ---- pass 2: every citation anywhere in the document ----
    cited_files = {c.group(1) for c in CITE.finditer(text)}
    for m in re.finditer(r"`([^`]+?\.(?:json|log|txt))`", text):
        name = m.group(1)
        p2 = os.path.join(root, name)
        if not os.path.exists(p2):
            # inventory lines sometimes give the bare file name; resolve that too
            p2 = os.path.join(root, "records", os.path.basename(name))
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
        check_citation(fname, field, root, None, "prose %s" % fname, problems, c.group(0))
        if problems:
            print("FAIL  prose citation  %s" % problems[0][1])
            fail += 1
        else:
            ok += 1

    return ok, fail, skip, len(checked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default=None, help="Path to report markdown file")
    ap.add_argument("--root", default="md/backdoor/verdicts", help="Root directory holding records")
    ap.add_argument("--all", action="store_true", help="Check all reports under md/backdoor/verdicts/")
    args = ap.parse_args()

    if args.all or args.report is None:
        report_files = sorted(glob.glob(os.path.join(args.root, "*.md")))
        if not report_files:
            print("No reports found in %s" % args.root)
            return 1
        total_ok = total_fail = total_skip = total_checked = 0
        overall_pass = True
        print("Checking %d reports in %s...\n" % (len(report_files), args.root))
        for rf in report_files:
            rel = os.path.basename(rf)
            print("=== %s ===" % rel)
            ok, fail, skip, checked = verify_report(rf, args.root)
            total_ok += ok
            total_fail += fail
            total_skip += skip
            total_checked += checked
            print("  resolved %d   failed %d   skipped %d   citations checked %d" % (ok, fail, skip, checked))
            status = "PASS" if fail == 0 else "FAIL"
            print("  STATUS: %s\n" % status)
            if fail > 0:
                overall_pass = False

        print("=" * 60)
        print("OVERALL SUMMARY: %d reports checked" % len(report_files))
        print("TOTAL: resolved %d   failed %d   skipped %d   citations checked %d"
              % (total_ok, total_fail, total_skip, total_checked))
        print("ALL CITATIONS CHECK: %s" % ("PASS" if overall_pass else "FAIL"))
        return 0 if overall_pass else 1

    ok, fail, skip, checked = verify_report(args.report, args.root)
    print("\nresolved %d   failed %d   skipped %d   citations checked %d"
          % (ok, fail, skip, checked))
    print("CITATION CHECK: %s" % ("PASS" if fail == 0 else "FAIL"))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
