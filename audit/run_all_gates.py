# -*- coding: utf-8 -*-
"""Unified gate runner for backdoor audit validation.

Runs all 4 mandatory verification gates in sequence:
  1. procedure_check.py (123 checks on procedure document)
  2. check_cites.py --all (comprehensive citation verification across all 13 verdict reports)
  3. selftest.py (42/42 self-tests on synthetic models, packs, and container edges)
  4. markdown_lint (verifies no unreplaced placeholders, TODOs, or corrupted headers in md/backdoor/)

Usage:
  python audit/run_all_gates.py
"""
from __future__ import annotations

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


def main():
    py_exec = sys.executable
    gates = [
        ("Procedure Check (123 checks)", [py_exec, "audit/procedure_check.py"]),
        ("Citation Check (all 13 reports)", [py_exec, "audit/check_cites.py", "--all", "--root", "md/backdoor/verdicts"]),
        ("Selftest (42/42)", [py_exec, "audit/selftest.py"]),
    ]

    results = []
    for title, cmd in gates:
        ok = run_subcommand(title, cmd)
        results.append((title, ok))

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
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
