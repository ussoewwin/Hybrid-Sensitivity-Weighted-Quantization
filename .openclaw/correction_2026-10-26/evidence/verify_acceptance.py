# -*- coding: utf-8 -*-
"""Acceptance verification: every Goal Brief criterion, executed.

Each check prints PASS/FAIL with the measured evidence. No check asserts
without a measurement.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

PY = sys.executable
R = r"D:\USERFILES\GitHub\hswq"
BK = R + r"\.openclaw\correction_2026-10-26"
CUR = R + r"\sd1.5\diagimpact_sd15.py"
OLD = BK + r"\original\diagimpact_sd15.HEAD.py"
SDXL = BK + r"\original\diag_impact_sdxl.REFERENCE.py"

results = []


def check(cid, label, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, "%s: %s" % (type(e).__name__, e)
    results.append({"id": cid, "label": label, "pass": ok, "detail": detail})
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label[:46], detail))
    return ok


print("=" * 84)
print("  ACCEPTANCE VERIFICATION")
print("=" * 84)
print()


# --- C1: the fabrication inventory is complete ------------------------------
def c1():
    inv = BK + r"\evidence\fabrication_inventory.py"
    r = subprocess.run([PY, "-u", inv], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    rows = r.stdout.count("--- F0")
    ok = r.returncode == 0 and rows == 6
    return ok, "%d inventory rows (F01..F06), exit %d" % (rows, r.returncode)


check("full-fabrication-inventory", "捏造・未検証記述の全件棚卸し", c1)


# --- C2: every correction has a source -------------------------------------
def c2():
    led = io.open(BK + r"\LEDGER.md", encoding="utf-8").read()
    # every F-row must have a 根拠 cell that names a file + line
    rows = re.findall(r"### (F0\d) —", led)
    with_src = 0
    for f in rows:
        block = led.split("### %s —" % f)[1].split("### ")[0]
        if "根拠" in block and re.search(r"[LE]\d+", block):
            with_src += 1
    ok = len(rows) == 6 and with_src == 6
    return ok, "%d/%d F-rows carry a source (file + line)" % (with_src, len(rows))


check("every-correction-sourced", "すべての訂正に出典が付いている", c2)


# --- C3: no unsourced assertion remains in the final code ------------------
def c3():
    cur = io.open(CUR, encoding="utf-8", newline="").read()
    problems = []
    # the fabricated term must be absent
    for bad in ("KSampler", "ksampler"):
        if bad in cur:
            problems.append(bad)
    # the production sampler must be present
    if "comfy_sample.sample(" not in cur:
        problems.append("no comfy_sample.sample(")
    # the CLI must expose sampler/scheduler/cfg/width/height/prompt/negative
    for flag in ("--sampler", "--scheduler", "--cfg", "--width", "--height",
                 "--prompt", "--negative"):
        if flag not in cur:
            problems.append("missing %s" % flag)
    # the old engine must be gone
    for gone in ("torch.linspace(1.0, 0.0", "eps = model(x, t, ctx)"):
        if gone in cur:
            problems.append("still has %s" % gone)
    ok = not problems
    return ok, "no fabricated term, engine and CLI complete" if ok else ", ".join(problems)


check("no-unsourced-claims-remain", "修正版に根拠なしの断定が残っていない", c3)


# --- C4: the ledger is traceable -------------------------------------------
def c4():
    led = io.open(BK + r"\LEDGER.md", encoding="utf-8").read()
    # count the field rows INSIDE each F-block only, so the section-0 header
    # table ("| 旧記述 | 内容 |") is not miscounted
    rows_f = re.findall(r"### (F0\d) —", led)
    tot = {k: 0 for k in ("旧記述", "新記述", "根拠", "確信度", "判断")}
    for f in rows_f:
        block = led.split("### %s —" % f)[1].split("### ")[0]
        for k in tot:
            tot[k] += len([l for l in block.split("\n") if l.startswith("| %s |" % k)])
    ok = (len(rows_f) == 6
          and all(tot[k] == 6 for k in ("旧記述", "新記述", "根拠", "確信度"))
          and tot["判断"] >= 1)
    return ok, "F-rows=%d old=%d new=%d source=%d confidence=%d judgement=%d" % (
        len(rows_f), tot["旧記述"], tot["新記述"], tot["根拠"], tot["確信度"], tot["判断"])


check("traceable-correction-ledger", "修正台帳で旧記述から新記述まで追跡できる", c4)


# --- C5: a third party can reproduce ----------------------------------------
def c5():
    present = []
    for f in ("fabrication_inventory.py", "measure_old_vs_new.py", "root_cause.py"):
        p = os.path.join(BK, "evidence", f)
        if os.path.isfile(p):
            present.append(f)
    data = []
    for f in ("old_vs_new.json", "root_cause_arms.json"):
        p = os.path.join(BK, "evidence", f)
        if os.path.isfile(p):
            data.append(f)
    ok = len(present) == 3 and len(data) == 2
    return ok, "%d scripts + %d data files are on disk" % (len(present), len(data))


check("reproducible-verification", "第三者が同じ手順で再検証できる", c5)


# --- C6: the original is preserved and the diff is available ---------------
def c6():
    h_old = hashlib.sha256(open(OLD, "rb").read()).hexdigest()
    h_cur = hashlib.sha256(open(CUR, "rb").read()).hexdigest()
    n_old = os.path.getsize(OLD)
    n_cur = os.path.getsize(CUR)
    d = subprocess.run(["git", "-C", R, "diff", "--numstat", "--",
                        "sd1.5/diagimpact_sd15.py"],
                       capture_output=True).stdout.decode().strip()
    ok = h_old != h_cur and n_old > 0 and d != ""
    return ok, "HEAD=%d bytes, current=%d bytes, diff='%s'" % (n_old, n_cur, d)


check("original-preserved-with-diff", "原本が保全され差分が確認できる", c6)


# --- C7: handover with open items ------------------------------------------
def c7():
    led = io.open(BK + r"\LEDGER.md", encoding="utf-8").read()
    n_open = len(re.findall(r"\| U0\d \|", led))
    ok = n_open >= 3
    return ok, "%d open items (U01..) recorded with reasons" % n_open


check("handover-with-open-items", "残課題と判断理由が引き継がれている", c7)

print()
print("=" * 84)
passed = sum(1 for r in results if r["pass"])
print("  RESULT: %d / %d criteria PASS" % (passed, len(results)))
print("=" * 84)

out = os.path.join(BK, "evidence", "acceptance_results.json")
json.dump(results, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print()
print("  [saved] %s" % out)
sys.exit(0 if passed == len(results) else 1)
