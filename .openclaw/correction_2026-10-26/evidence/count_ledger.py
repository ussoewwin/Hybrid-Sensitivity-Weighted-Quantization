import io, re, sys, os
sys.stdout.reconfigure(encoding="utf-8")
BK = r"D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26"
led = io.open(BK + r"\LEDGER.md", encoding="utf-8").read()
rows = re.findall(r"### (F0\d) —", led)
print("  the F-rows in the ledger: %s" % rows)
print()
print("  per-row field presence (the header table is excluded by using the F-block):")
for f in rows:
    block = led.split("### %s —" % f)[1].split("### ")[0]
    # a data row = a table line whose first cell is exactly the field name
    fields = {}
    for name in ("旧記述", "新記述", "根拠", "確信度", "判断"):
        # count lines starting with "| <name> |" inside this block
        n = len([l for l in block.split("\n") if l.startswith("| %s |" % name)])
        fields[name] = n
    print("    %s  %s" % (f, fields))
print()
# the header table appears once, in section 0; count only inside F-blocks
tot = {k: 0 for k in ("旧記述","新記述","根拠","確信度")}
for f in rows:
    block = led.split("### %s —" % f)[1].split("### ")[0]
    for k in tot:
        tot[k] += len([l for l in block.split("\n") if l.startswith("| %s |" % k)])
print("  the totals across the 6 F-blocks: %s" % tot)
print("  all equal 6: %s" % (all(v == 6 for v in tot.values()),))
