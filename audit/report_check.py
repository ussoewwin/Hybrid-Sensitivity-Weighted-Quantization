# -*- coding: utf-8 -*-
"""Report gate: every verdict report must carry the sections the procedure demands.

The procedure requires each report to keep a plain verdict sentence, an identity section, and a
LIMITS section (candidate set and how it was generated, the positions probed, the reference-class
boundary) plus an explicit statement of what the verdict does not establish. That requirement was
prose only; this checker makes it a gate, so a report that gains numbers while losing its limits
cannot pass.

Local absolute paths are reported as WARN, never as FAIL: they are a pre-existing condition in
older text the audit must not silently rewrite.

    python report_check.py [--root <dir of reports>] [--cites]
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

VERDICT_RE = re.compile(r"\*\*Verdict:\s*no backdoor", re.I)
IDENTITY_RE = re.compile(r"^#+\s*\d+\.\s*Identity", re.M)
LIMITS_RE = re.compile(r"^#+\s*(\d+\.?\s*)?LIMITS", re.M)
PATH_RE = re.compile(r"[A-Za-z]:(?!//)[\\/][^\s\"`')]+")
GUARANTEE_RE = re.compile(r"proves that no backdoor exists|is definitely safe|guaranteed safe",
                          re.I)

REQUIRED = [
    (re.compile(r"^#\s*Verdict\s+-", re.M), "title line '# Verdict - <file>'"),
    (VERDICT_RE, "the plain verdict sentence"),
    (re.compile(r"strength [ABC]", re.I), "the weight-level closure strength (A/B/C)"),
    (re.compile(r"coverage \d[\d,]*/\d[\d,]*( audited)? tensors", re.I), "the audited-tensor coverage fraction"),
    (IDENTITY_RE, "an identity section"),
    (LIMITS_RE, "a LIMITS section"),
    (re.compile(r"[Cc]andidate set"), "candidate set size and generation"),
    (re.compile(r"[Pp]ositions?\b"), "positions probed"),
    (re.compile(r"[Rr]eference-class boundary"), "reference-class boundary"),
    (re.compile(r"records/[A-Za-z0-9_.\-]+\.(json|log|txt)"), "citations to record files"),
    (re.compile(r"does not establish|not covered|unreachable|out of reach"),
     "an explicit statement of what the verdict does not establish"),
]


def check_file(path):
    text = io.open(path, encoding="utf-8", errors="replace").read()
    problems = []
    warnings = []
    for pat, what in REQUIRED:
        if not pat.search(text):
            problems.append("missing " + what)
    if GUARANTEE_RE.search(text):
        problems.append("contains an unbounded safety guarantee")
    n = len(PATH_RE.findall(text))
    if n:
        warnings.append("local absolute path: %d occurrence(s)" % n)
    return problems, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.normpath(os.path.join(HERE, "..", "md", "verdicts")))
    ap.add_argument("--cites", action="store_true", help="also run check_cites.py per report")
    args = ap.parse_args()
    root = os.path.normpath(args.root)
    files = sorted(f for f in os.listdir(root) if f.endswith(".md"))
    fails = 0
    for f in files:
        p = os.path.join(root, f)
        problems, warnings = check_file(p)
        cites_note = ""
        if args.cites and not problems:
            r = subprocess.run([sys.executable, os.path.join(HERE, "check_cites.py"), p,
                               "--root", root], capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
            out = (r.stdout or "").strip().splitlines()
            line = next((x for x in reversed(out) if "resolved" in x), "")
            cites_note = " | " + line.strip()
            if "failed 0" not in line:
                problems.append("citations not clean:" + line.strip()[:60])
        for w in warnings:
            print("WARN %s: %s" % (f, w))
        if problems:
            fails += 1
            print("FAIL %s" % f)
            for x in problems:
                print("      - " + x)
        else:
            n = len(io.open(p, encoding="utf-8", errors="replace").read().splitlines())
            print("PASS %s (%d lines)%s" % (f, n, cites_note))
    print("report_check: %d/%d reports pass" % (len(files) - fails, len(files)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
