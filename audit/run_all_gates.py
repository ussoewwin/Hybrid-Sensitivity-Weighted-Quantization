# -*- coding: utf-8 -*-
"""Unified gate runner for backdoor audit validation.

Runs all 4 mandatory verification gates in sequence:
  1. procedure_check.py (checks the procedure document)
  2. check_cites.py --all (comprehensive citation verification across all 13 verdict reports)
  3. selftest.py (the self-tests on synthetic models, packs, and container edges)
  4. markdown_lint (verifies no unreplaced placeholders, TODOs, or corrupted headers in md/backdoor/)

Usage:
  python audit/run_all_gates.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import time

import glob
import os
import re
import subprocess
import sys


def run_subcommand(title, cmd):
    print(f"\n{'='*70}\n[GATE] {title}\n{'='*70}")
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    passed = (res.returncode == 0)
    print(f"--> {title}: {'PASS' if passed else 'FAIL'}")
    return passed


def lint_markdown_documents():
    print(f"\n{'='*70}\n[GATE] Markdown Linting (Placeholders & Structure)\n{'='*70}")
    docs = glob.glob("md/backdoor/*.md") + glob.glob("md/backdoor/verdicts/*.md")
    placeholders = [
        re.compile(r"\bTODO\b", re.I),
        re.compile(r"\bTBD\b", re.I),
        re.compile(r"\bFIXME\b", re.I),
        re.compile(r"\bXXX\b"),
        re.compile(r"\bREPLACE_ME\b", re.I),
        re.compile(r"<\.\.\.>", re.I),
        re.compile(r"\[insert\b", re.I),
    ]

    # Exclude files that legitimately mention TODO in self-explanatory code or rules
    excluded_lint_files = {"Backdoor_Audit_Improvement_Plan.md"}

    issues = []
    checked_count = 0
    for doc in sorted(docs):
        fname = os.path.basename(doc)
        if fname in excluded_lint_files:
            continue
        checked_count += 1
        lines = open(doc, encoding="utf-8").readlines()
        headers = set()
        for idx, line in enumerate(lines, 1):
            for pat in placeholders:
                if pat.search(line):
                    # verify it is not inside an explanation of placeholders
                    if "placeholder" not in line.lower():
                        issues.append(f"{doc}:{idx} placeholder match: {line.strip()}")
            if line.startswith("#"):
                h = line.strip()
                # Check for duplicate major section headings (level 1 or 2)
                if line.startswith("# ") or line.startswith("## "):
                    if h in headers and "identity" not in h.lower() and "findings" not in h.lower() and "---" not in h:
                        issues.append(f"{doc}:{idx} duplicate header: {h}")
                    headers.add(h)

    if issues:
        for iss in issues:
            print("  FAIL:", iss)
        print(f"\nMarkdown Lint: FAIL ({len(issues)} issues found across {checked_count} docs)")
        return False
    print(f"Checked {checked_count} documents. No unreplaced placeholders or duplicate headers found.")
    print("--> Markdown Linting: PASS")
    return True


def verify_sums():
    """P2-3: every record listed in SHA256SUMS must hash to the listed digest."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(repo, "md", "backdoor", "verdicts", "records")
    manifest = os.path.join(d, "SHA256SUMS")
    if not os.path.exists(manifest):
        print("  FAIL: records/SHA256SUMS missing")
        return False
    lines = io.open(manifest, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    listed = {}
    for line in lines:
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        name = name.strip()
        if name.startswith("gate_run_"):
            # the gate run log changes on every run and is not evidence a report cites
            continue
        listed[name] = digest
    bad = missing = 0
    for name, digest in sorted(listed.items()):
        path = os.path.join(d, name)
        if not os.path.exists(path):
            missing += 1
            print("  FAIL: listed record missing:", name)
            continue
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != digest:
            bad += 1
            print("  FAIL: digest mismatch:", name)
    present = [f for f in os.listdir(d)
               if f != "SHA256SUMS" and not f.startswith("gate_run_")
               and os.path.isfile(os.path.join(d, f))]
    unlisted = sorted(f for f in present if f not in listed)
    if unlisted:
        print("  FAIL: record not in SHA256SUMS:", ", ".join(unlisted[:6]))
    ok = (bad == 0 and missing == 0 and not unlisted)
    print("  %d records verified, %d mismatched, %d missing, %d unlisted"
          % (len(listed), bad, missing, len(unlisted)))
    print("--> SHA256SUMS Verification: %s" % ("PASS" if ok else "FAIL"))
    return ok


def regen_sums():
    """W18: regenerate records/SHA256SUMS from the records that are on disk.

    The manifest is the list of the records the reports cite; adding a record without
    registering it is the failure this closes. gate_run_* logs are excluded because they
    change on every run.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(repo, "md", "backdoor", "verdicts", "records")
    manifest = os.path.join(d, "SHA256SUMS")
    names = sorted(f for f in os.listdir(d)
                   if f != "SHA256SUMS" and not f.startswith("gate_run_")
                   and os.path.isfile(os.path.join(d, f)))
    rows = []
    for name in names:
        digest = hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest()
        rows.append("%s  %s" % (digest, name))
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("--> SHA256SUMS regenerated: %d records" % len(rows))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="regenerate records/SHA256SUMS from the records on disk before verifying")
    args = ap.parse_args()
    if args.regen:
        regen_sums()
    py_exec = sys.executable
    gates = [
        ("Procedure Check", [py_exec, "audit/procedure_check.py"]),
        ("Citation Check (all reports)", [py_exec, "audit/check_cites.py", "--all", "--root", "md/backdoor/verdicts"]),
        ("Selftest", [py_exec, "audit/selftest.py"]),
    ]

    results = []
    log_rows = []
    for title, cmd in gates:
        t0 = time.time()
        ok = run_subcommand(title, cmd)
        elapsed = round(time.time() - t0, 2)
        results.append((title, ok))
        script = cmd[1] if len(cmd) > 1 else ""
        sha = ""
        if script and os.path.exists(script):
            sha = hashlib.sha256(open(script, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
        log_rows.append({"gate": title, "status": "PASS" if ok else "FAIL",
                         "seconds": elapsed, "script": script, "script_sha256": sha})

    sums_ok = verify_sums()
    results.append(("SHA256SUMS Verification", sums_ok))

    lint_ok = lint_markdown_documents()
    results.append(("Markdown Linting", lint_ok))

    print(f"\n{'='*70}\nALL GATES SUMMARY\n{'='*70}")
    all_passed = True
    for title, ok in results:
        status_str = "PASS" if ok else "FAIL"
        print(f"  {title:35s}: {status_str}")
        if not ok:
            all_passed = False

    print(f"\nOVERALL VERDICT: {'ALL GATES PASS' if all_passed else 'GATE FAILURE'}")

    # The run is recorded where the reports can cite it, not only on the console.
    rec_dir = os.path.join("md", "backdoor", "verdicts", "records")
    if os.path.isdir(rec_dir):
        today = datetime.date.today().isoformat()
        rec = {
            "date": today,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "overall": "PASS" if all_passed else "FAIL",
            "gates": log_rows,
            "markdown_lint": "PASS" if lint_ok else "FAIL",
        }
        # W20: a same-day re-run keeps the first run of the day and writes the next one
        # as a numbered sibling, so no run log is overwritten.
        out = os.path.join(rec_dir, "gate_run_%s.json" % today)
        if os.path.exists(out):
            n = 1
            while os.path.exists(os.path.join(rec_dir, "gate_run_%s_%d.json" % (today, n))):
                n += 1
            out = os.path.join(rec_dir, "gate_run_%s_%d.json" % (today, n))
        with open(out, "w", encoding="utf-8", newline="\n") as fp:
            json.dump(rec, fp, indent=2)
        print("run log: %s" % out)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
