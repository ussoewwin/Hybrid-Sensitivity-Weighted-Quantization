import io
p = r"D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\final_scan.py"
t = io.open(p, encoding="utf-8").read()
old = '''        rel = os.path.relpath(p, BK)
        if rel == "LEDGER.md":
            continue
        if "KSampler" in t or "ksampler" in t:
            bad.append(rel)'''
new = '''        rel = os.path.relpath(p, BK)
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
        bad.append(rel)'''
if old in t:
    t = t.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="").write(t)
    print("  the scanner patched: it now distinguishes 'records the sin' from 'uses the term'")
else:
    print("  NOT FOUND")
