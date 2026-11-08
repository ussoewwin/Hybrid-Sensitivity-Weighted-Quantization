# -*- coding: utf-8 -*-
"""Check a deliverable against its own required content, before it is shown.

This exists because a procedure was delivered incomplete and had to be rewritten.
The failure was not a missing measurement; it was showing a document without first
verifying that it contained the sections, gates, and commands it promises. A
document that is checked only after review is a redo waiting to happen.

The check is structural and mechanical:

  * every phase heading of the standard audit sequence is present;
  * every gate identifier is present, including the sub-gates;
  * every tool the document cites exists on disk and is referenced with a command;
  * the document contains no absolute local path (so it works for any reader);
  * the completion checklist and the run-record schema are present.

    python audit/procedure_check.py [--procedure <md>] [--audit-dir <dir>]

Exit 0 when every row passes, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PHASES = [
    "## 4. Phase 0", "## 5. Phase 1", "## 6. Phase 2", "## 7. Phase 3",
    "## 8. Phase 4", "## 9. Phase 5", "## 10. Phase 6", "## 11. Phase 7",
    "## 12. Phase 8", "## 12b. Gate 8",
]
REQUIRED_RULES = ["## 13. Prohibitions", "## 14. Per-class checklists",
                  "## 15. Appendices"]
REQUIRED_GATES = ["Gate 0.", "Gate 1.", "Gate 1b.", "Gate 2.", "Gate 3.",
                  "Gate 3b.", "Gate 4a", "Gate 4b", "Gate 4c", "Gate 4d",
                  "Gate 5.", "Gate 6.", "Gate 7a", "Gate 7b", "Gate 7c",
                  "Gate 7d", "Gate 7e", "Gate 7f", "Gate 7g", "Gate 8", "Gate E.", "Gate F."]
REQUIRED_TOOLS = [
    "backdoor_audit.py", "source_audit.py", "st_integrity.py",
    "fetch_reference.py", "ablit_repro.py", "build_equivalence.py",
    "direction_control.py", "divergence_scan.py", "trigger_logits.py",
    "trigger_search.py", "pointer_probe.py", "clip_trigger_scan.py",
    "openai_clip_layout.py", "openai_clip_to_hf.py", "selftest.py",
    "procedure_check.py",
    "ablation_split.py", "shard_census.py", "trigger_variants.py", "vl_divergence.py",
    "josie_stage1.py", "josie_bench.py", "josie_trigger.py", "sae_probe.py", "sae_deep.py", "sae_spectrum.py", "sae_retrieval.py", "sae_independent.py", "check_cites.py",
    "vl_bench.py", "check_hf_upload.py", "report_check.py", "pack_load_path.py", "pack_coverage.py",
    "pair_guarded.py", "sam31_routes_final.py", "tool_context_probe.py", "make_limits.py", "vl_bench_agreement.py",
    "sam31_census.py", "sam31_weight_positive_control.py", "sam31_weight_screen.py",
    "sam31_operating_real.py", "sam31_vocabulary_overlap_control.py",
    "controlnet_hswq_route.py",
    "controlnet_int8_exec.py",
    "controlnet_image_test.py",
    "sam31_image_trigger.py", "sam31_upstream_retry.py",
    "controlnet_pack_audit.py",
    "model_patch_pack_audit.py",
    "model_patch_hswq_route.py",
    "model_patch_int8_exec.py",
    "model_patch_protection_census.py",
    "controlnet_image_trigger.py",
    "model_patch_image_trigger.py",
]
REQUIRED_SECTIONS = ["## 12c. SAM3.1 extension - the required lines of an S0-S6 verdict",
                      "## 12d. ControlNet and model-patch extension - the required lines of a CN/MP verdict"]
REQUIRED_APPENDICES = ["### A. Run-record schema", "### B. Storage floors",
                       "### C. OpenAI to HF CLIP mapping", "### D. Tool reference",
                       "### E. Self-test before trusting a tool change"]
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", default=os.path.join(
        HERE, "..", "md", "backdoor", "Backdoor_Audit_Procedure.md"))
    ap.add_argument("--audit-dir", default=HERE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = os.path.normpath(args.procedure)
    if not os.path.exists(path):
        print("FAIL procedure not found: %s" % path)
        return 2
    text = open(path, encoding="utf-8").read()

    rows = []

    def check(name, ok, detail=""):
        rows.append((name, ok, detail))

    for h in REQUIRED_PHASES + REQUIRED_RULES + REQUIRED_SECTIONS + REQUIRED_APPENDICES:
        check("section present: %s" % h, h in text,
              "" if h in text else "missing")

    for g in REQUIRED_GATES:
        check("gate present: %s" % g.strip(), g in text,
              "" if g in text else "missing")

    for tool in REQUIRED_TOOLS:
        p = os.path.join(args.audit_dir, tool)
        exists = os.path.exists(p)
        cited = tool in text
        check("tool exists: %s" % tool, exists, "" if exists else "not on disk")
        check("tool cited: %s" % tool, cited, "" if cited else "not cited in the text")

    cited_tools = set(re.findall(r"([A-Za-z0-9_]+\.py)", text))
    unknown = sorted(t for t in cited_tools if t not in REQUIRED_TOOLS)
    check("no tool cited that does not exist", not unknown, ", ".join(unknown))

    bad_paths = ABSOLUTE_PATH.findall(text)
    check("no absolute local path in the document", not bad_paths,
          ", ".join(sorted(set(bad_paths))[:4]))

    check("completion checklist present",
          "[ ] Phase 0" in text or "[ ] 0 " in text or "```text" in text)
    check("verdict vocabulary restricted to four words",
          all(w in text for w in ("NO-ADDITION", "ADDITION-PRESENT", "DAMAGED",
                                  "INCONCLUSIVE")))
    check("delivery is excluded from findings in writing",
          "never a finding" in text or "not a finding" in text)

    # SAM3.1 extension: the three required lines of the verdict, checked against
    # the records (coverage, operating point, detection floor).
    verdicts_dir = os.path.normpath(os.path.join(HERE, "..", "md", "backdoor", "verdicts"))
    sam31_report = os.path.join(verdicts_dir, "SAM31_Multiplex.md")
    rec_dir = os.path.join(verdicts_dir, "records")
    if os.path.exists(sam31_report):
        report_text = open(sam31_report, encoding="utf-8").read()
        try:
            screen = json.load(open(os.path.join(rec_dir, "sam31_weight_screen.json"), encoding="utf-8"))
            oper = json.load(open(os.path.join(rec_dir, "sam31_operating_real.json"), encoding="utf-8"))
            pos = json.load(open(os.path.join(rec_dir, "sam31_positive_control.json"), encoding="utf-8"))
        except Exception as exc:  # a missing record is a failure, not a skip
            check("SAM3.1 records readable", False, str(exc))
            screen = oper = pos = {}
        coverage = screen.get("total_screened_tensors")
        check("SAM3.1 coverage line: 1,590/1,590", "1,590/1,590" in report_text,
              "report states the audited-tensor coverage")
        check("SAM3.1 coverage record == 1590", coverage == 1590, "record says %r" % (coverage,))
        threshold = (oper.get("established_operating_point") or {}).get("threshold")
        check("SAM3.1 operating-point line in report", ("threshold" in report_text.lower()
              and str(threshold) in report_text), "threshold %r" % (threshold,))
        summary = pos.get("summary") or {}
        floors = {
            "rank1": summary.get("linear_rank1_first_detected_strength"),
            "conv": summary.get("conv4d_spectral_first_detected_strength"),
            "bias": summary.get("bias_1d_group_p99_9_detection_floor"),
            "embed": summary.get("token_embedding_norm_p99_floor"),
        }
        check("SAM3.1 detection floor record complete", all(v for v in floors.values()),
              json.dumps(floors))
        for key, value in floors.items():
            if value:
                check("SAM3.1 detection floor (%s) stated in report" % key, str(value) in report_text,
                      str(value))
        for step, name in (("S0", "sam31_census.json"), ("S1", "sam31_weight_screen.json"),
                           ("S2", "sam31_operating_real.json"), ("S3", "sam31_s2_overlap_control.json"),
                           ("S4", "sam31_image_trigger.json"), ("S5", "sam31_positive_control.json"),
                           ("S6", "sam_gated.json")):
            rec_path = os.path.join(rec_dir, name)
            check("SAM3.1 %s record present: %s" % (step, name), os.path.exists(rec_path))
    else:
        check("SAM3.1 report present", False, sam31_report)

    # P0-1: every row of the verdict index must carry strength, coverage and the
    # instrument-sensitivity column; B/C rows must state the three measured numbers.
    index_path = os.path.normpath(os.path.join(HERE, "..", "md", "backdoor",
                                              "Backdoor_Audit_Verdicts.md"))
    if os.path.exists(index_path):
        idx_text = open(index_path, encoding="utf-8").read()
        data_rows = [l for l in idx_text.split("\n")
                     if l.startswith("| `") and l.rstrip().endswith("|")]
        check("index has data rows", len(data_rows) > 0, "%d rows" % len(data_rows))
        # the index carries two column counts: the LM rows have six, the ControlNet /
        # model-patch rows carry the two extra measurement columns
        def ncols(l):
            return len(l.strip().strip("|").split("|"))
        sec3 = idx_text.split("## Section 3:")
        s3_rows = [l for l in sec3[1].split("\n") if l.startswith("| `")] if len(sec3) > 1 else []
        other_rows = [l for l in data_rows if l not in s3_rows]
        bad_cols = [l for l in other_rows if ncols(l) != 6]
        check("index rows outside Section 3 carry six columns", not bad_cols, "%d bad" % len(bad_cols))
        bad3 = [l for l in s3_rows if ncols(l) != 8]
        check("index Section 3 rows carry eight columns", not bad3, "%d bad" % len(bad3))
        bad_strength = [l for l in data_rows
                        if not any(s in l for s in ("(A,", "(B,", "(C,"))]
        check("index rows state a strength class", not bad_strength, "%d bad" % len(bad_strength))
        bc_rows = [l for l in data_rows if ("(B," in l or "(C," in l)]
        missing_sens = [l for l in bc_rows
                        if not ("0/28" in l and "0.92" in l and "20/28" in l)]
        check("B/C rows state the instrument's measured sensitivity", not missing_sens,
              "%d of %d B/C rows lack the measured numbers" % (len(missing_sens), len(bc_rows)))
        no_sens_col = [l for l in data_rows
                       if len(l.strip().strip("|").split("|")) == 6
                       and not l.strip().strip("|").split("|")[4].strip()]
        check("index rows carry the sensitivity column", not no_sens_col,
              "%d empty" % len(no_sens_col))

        # W11: the index is generated from the report files, so its counts cannot
        # drift from them. Every report under verdicts/ must be reachable from a
        # row, and the number of rows must be the number of rows the tables hold.
        verdicts_dir = os.path.join(HERE, "..", "md", "backdoor", "verdicts")
        if os.path.isdir(verdicts_dir):
            reports = sorted(f for f in os.listdir(verdicts_dir) if f.endswith(".md"))
            linked = set()
            for m in re.finditer(r"\]\(verdicts/([^)]+\.md)\)", idx_text):
                linked.add(m.group(1))
            unlinked = [f for f in reports if f not in linked]
            check("every verdict report is linked from the index", not unlinked,
                  "%d unlinked: %s" % (len(unlinked), ", ".join(unlinked[:4])))
            # every row must point at a report file that exists, and every report
            # file must have at least one row: a report without a row is a report
            # the index does not show.
            row_links = []
            for l in data_rows:
                m = re.search(r"\]\(verdicts/([^)]+\.md)\)", l)
                row_links.append(m.group(1) if m else None)
            dangling = [l for l in row_links if l is None]
            check("every index row links to a report", not dangling,
                  "%d rows without a link" % len(dangling))
            missing_files = sorted({l for l in row_links if l and not os.path.exists(
                os.path.join(verdicts_dir, l))})
            check("every index row links to a report file that exists",
                  not missing_files, "%d missing: %s" % (len(missing_files),
                                                        ", ".join(missing_files[:4])))
            def rows_in(header):
                parts = idx_text.split(header)
                if len(parts) < 2:
                    return 0
                body = parts[1].split("\n## ")[0]
                return len([l for l in body.split("\n") if l.startswith("| `")])
            s1 = rows_in("## Section 1:")
            s2 = rows_in("## Section 2:")
            s3 = rows_in("## Section 3:")
            check("the index tables hold a row for every report",
                  s1 + s2 + s3 >= len(reports),
                  "sections %d+%d+%d = %d rows, %d reports"
                  % (s1, s2, s3, s1 + s2 + s3, len(reports)))
    else:
        check("verdict index present", False, index_path)

    # E7: every audit/*.py must be listed in audit/README.md, so a new tool cannot
    # exist without a reader being told what it does.
    readme_path = os.path.join(HERE, "README.md")
    if os.path.exists(readme_path):
        readme = open(readme_path, encoding="utf-8").read()
        unlisted = sorted(n for n in os.listdir(HERE)
                          if n.endswith(".py") and n not in readme)
        check("audit/README.md lists every audit/*.py", not unlisted,
              "%d unlisted: %s" % (len(unlisted), ", ".join(unlisted[:5])))
    else:
        check("audit/README.md present", False, readme_path)

    # E6: a public document must not carry a work-log voice.
    owner_hits = []
    for d in (os.path.join(HERE, "..", "md", "backdoor", "verdicts"),
              os.path.join(HERE, "..", "md", "backdoor")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            if fn == "Backdoor_Audit_Improvement_Plan.md":
                continue  # the working plan quotes the phrases it removes
            p = os.path.join(d, fn)
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace")):
                if "owner" in line.lower():
                    owner_hits.append("%s:%d" % (fn, i + 1))
    check("public reports carry no owner-view wording", not owner_hits,
          "%d hits: %s" % (len(owner_hits), ", ".join(owner_hits[:5])))

    # W13: a pack row in the index must state the source's own strength separately
    # from the pack fidelity, and the strength of the row inherits the source's.
    if os.path.exists(index_path):
        idx = open(index_path, encoding="utf-8").read()
        sec3 = idx.split("## Section 3:")
        if len(sec3) > 1:
            s3 = sec3[1]
            s3_rows = [l for l in s3.split("\n")
                       if l.startswith("| `") and l.rstrip().endswith("|")]
            check("index Section 3 carries the pack rows", len(s3_rows) >= 12,
                  "%d rows" % len(s3_rows))
            no_src = [l for l in s3_rows if "source absence" not in l.lower()
                      and "not established" not in l]
            check("index pack rows state the source absence separately",
                  not no_src, "%d of %d rows" % (len(no_src), len(s3_rows)))
        else:
            check("index Section 3 present", False, "no Section 3 heading")

    ok = all(r[1] for r in rows)
    if not args.quiet:
        for name, good, detail in rows:
            if not good:
                print("  [FAIL] %-52s %s" % (name, detail))
        total = len(rows)
        passed = sum(1 for r in rows if r[1])
        print("\n  %d/%d checks passed" % (passed, total))
        print("  procedure: %s (%d lines)" % (path, len(text.splitlines())))
    print("PROCEDURE CHECK: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
