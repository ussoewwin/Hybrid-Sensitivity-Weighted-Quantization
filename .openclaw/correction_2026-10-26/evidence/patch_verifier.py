import io, re
p = r"D:\USERFILES\GitHub\hswq\.openclaw\correction_2026-10-26\evidence\verify_acceptance.py"
t = io.open(p, encoding="utf-8").read()
old = '''    has_old = led.count("| 旧記述 |")
    has_new = led.count("| 新記述 |")
    has_src = led.count("| 根拠 |")
    has_conf = led.count("| 確信度 |")
    ok = has_old == 6 and has_new == 6 and has_src == 6 and has_conf == 6
    return ok, "old=%d new=%d source=%d confidence=%d (each must be 6)" % (
        has_old, has_new, has_src, has_conf)'''
new = '''    # count the field rows INSIDE each F-block only, so the section-0 header
    # table ("| 旧記述 | 内容 |") is not miscounted
    rows_f = re.findall(r"### (F0\\d) —", led)
    tot = {k: 0 for k in ("旧記述", "新記述", "根拠", "確信度", "判断")}
    for f in rows_f:
        block = led.split("### %s —" % f)[1].split("### ")[0]
        for k in tot:
            tot[k] += len([l for l in block.split("\\n") if l.startswith("| %s |" % k)])
    ok = (len(rows_f) == 6
          and all(tot[k] == 6 for k in ("旧記述", "新記述", "根拠", "確信度"))
          and tot["判断"] >= 1)
    return ok, "F-rows=%d old=%d new=%d source=%d confidence=%d judgement=%d" % (
        len(rows_f), tot["旧記述"], tot["新記述"], tot["根拠"], tot["確信度"], tot["判断"])'''
if old in t:
    t = t.replace(old, new)
    io.open(p, "w", encoding="utf-8", newline="").write(t)
    print("  the verifier patched")
else:
    print("  NOT FOUND")
