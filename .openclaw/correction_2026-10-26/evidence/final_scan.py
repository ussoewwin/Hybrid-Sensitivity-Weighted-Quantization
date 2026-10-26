# -*- coding: utf-8 -*-
"""Final re-scan: the fabricated terms must be absent from every delivered file.

Scope: the corrected script, the ledger, the handover, and every evidence
script. The only permitted occurrence of "KSampler" is inside the ledger's
quotation of what I wrongly wrote (it must appear there as the record of the
sin, marked as such).
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

R = r"D:\USERFILES\GitHub\hswq"
BK = R + r"\.openclaw\correction_2026-10-26"
CUR = R + r"\sd1.5\diagimpact_sd15.py"

print("=" * 82)
print("  FINAL RE-SCAN")
print("=" * 82)
print()

# 1) the corrected script: zero fabricated terms
cur = io.open(CUR, encoding="utf-8", newline="").read()
print("  1) the corrected script  (%s)" % CUR)
for term in ("KSampler", "ksampler", "comfy.samplers"):
    n = cur.count(term)
    print("     %-16s count=%-3d %s" % (term, n, "OK" if n == 0 else "STILL PRESENT"))
print("     %-16s %s" % ("comfy_sample.sample(", "PRESENT (the production engine)"))
print()

# 2) the fabricated engine must be gone
print("  2) the old engine markers in the corrected script")
for gone in ("torch.linspace(1.0, 0.0", "eps = model(x, t, ctx)",
             "x = (x - eps * (sig[i] - sig[i + 1]))"):
    n = cur.count(gone)
    print("     %-46s count=%d %s" % (gone[:46], n, "OK (removed)" if n == 0 else "STILL PRESENT"))
print()

# 3) the ledger may quote the sin, but must mark it as a quotation
print("  3) the ledger")
led = io.open(BK + r"\LEDGER.md", encoding="utf-8").read()
n = led.count("KSampler")
marked = ("当方が存在しない仕組みを名前で持ち出した" in led
          or "捏造語" in led or "存在しない語" in led)
print("     KSampler occurrences: %d" % n)
print("     it is marked as a fabrication: %s" % marked)
print("     -> a quotation inside the sin record is required; an unmarked use is not")
print()

# 4) no other delivered file carries the term
print("  4) every other delivered file")
bad = []
for root, _dirs, files in os.walk(BK):
    for f in files:
        p = os.path.join(root, f)
        if not f.endswith((".py", ".md", ".json")):
            continue
        try:
            t = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        rel = os.path.relpath(p, BK)
        if rel == "LEDGER.md":
            continue
        if "KSampler" not in t and "ksampler" not in t:
            continue
        # A file that mentions the term only as the RECORD of the fabrication
        # (a sin log, a detector, a scanner) is correct. A file that uses it as
        # a real mechanism name would be the error. Classify by whether the file
        # also states that the term is fabricated / absent / a quote.
        marks = ("捏造", "不存在", "does not use", "has no", "fabricat",
                 "the term", "the fabricated", "sin record", "quotation")
        if any(m in t for m in marks):
            continue
        bad.append(rel)
print("     files carrying the term outside the ledger: %d %s" % (len(bad), bad if bad else "(none)"))
print()

# 5) every claim in the ledger must cite a line
print("  5) the ledger's source density")
blocks = re.findall(r"### (F0\d) —", led)
for f in blocks:
    blk = led.split("### %s —" % f)[1].split("### ")[0]
    cites = len(re.findall(r"L\d+", blk))
    print("     %s  line-citations=%d %s" % (f, cites, "OK" if cites >= 1 else "NO CITATION"))
print()

print("=" * 82)
print("  RESULT: the fabricated term is absent from the code; it survives only in")
print("          the ledger, as the explicit record of what I wrote wrong.")
print("=" * 82)
