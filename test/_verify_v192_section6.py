# -*- coding: utf-8 -*-
"""Verify md/V1.92_to_V2.0_Changes.md section 6 code blocks match source line ranges."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
v192 = (ROOT / "quantize_zib_hswq_v1.92.py").read_text(encoding="utf-8").splitlines()
v20 = (ROOT / "quantize_zib_hswq_v2.0.py").read_text(encoding="utf-8").splitlines()
md = (ROOT / "md" / "V1.92_to_V2.0_Changes.md").read_text(encoding="utf-8")

# (label, file_key, start, end) — must match test/_gen_v192_v20_section6.py
CHECKS = [
    ("6-0 v192", "v192", 1, 11),
    ("6-0 v20", "v20", 1, 21),
    ("6-1", "v20", 350, 366),
    ("6-2", "v20", 369, 402),
    ("6-3", "v20", 405, 419),
    ("6-4", "v20", 422, 443),
    ("6-5", "v20", 446, 465),
    ("6-6", "v20", 468, 490),
    ("6-7", "v20", 493, 500),
    ("6-8", "v20", 503, 514),
    ("6-9", "v20", 517, 528),
    ("6-10", "v20", 531, 630),
    ("6-11 v192", "v192", 606, 615),
    ("6-11 v20", "v20", 913, 926),
    ("6-11b v192", "v192", 636, 661),
    ("6-11b v20", "v20", 947, 986),
    ("6-12 v192", "v192", 364, 420),
    ("6-12 v20", "v20", 658, 727),
    ("6-12b v192", "v192", 880, 881),
    ("6-12b v20", "v20", 1205, 1220),
    ("6-13", "v20", 1222, 1251),
    ("6-14 v20", "v20", 1297, 1319),
    ("6-14 v192", "v192", 984, 994),
    ("6-15", "v20", 1427, 1446),
    ("6-16", "v20", 1526, 1551),
    ("6-17", "v20", 1601, 1607),
    ("6-18 v192", "v192", 908, 963),
    ("6-19 v192", "v192", 776, 779),
    ("6-19 v20", "v20", 1101, 1104),
    ("6-20 v192", "v192", 1119, 1123),
    ("6-20 v20", "v20", 1465, 1474),
    ("6-21 v192", "v192", 1196, 1198),
    ("6-21 v20", "v20", 1572, 1573),
]

SRC = {"v192": v192, "v20": v20}


def slice_src(key: str, start: int, end: int) -> str:
    return "\n".join(SRC[key][start - 1 : end])


def main() -> int:
    errors = []
    for label, key, start, end in CHECKS:
        expected = slice_src(key, start, end)
        if expected not in md:
            errors.append(f"MISSING: {label} L{start}-{end} ({key})")
            continue
        # forbid ellipsis-only truncation in section 6 excerpts (except inside f-strings)
        if "\n...\n" in f"\n{expected}\n":
            errors.append(f"ELLIPSIS in source {label}")

    sec6 = md.split("## 6.")[1].split("## 7.")[0] if "## 6." in md else ""
    if "..." in sec6:
        bad = [
            ln
            for ln in sec6.splitlines()
            if "..." in ln
            and 'print(f"' not in ln
            and 'print("' not in ln
            and "no `...` omissions" not in ln
            and "−96" not in ln  # prose diff stats, not ellipsis
        ]
        if bad:
            errors.append(f"Suspicious ... lines in section 6: {bad[:5]}")

    headings = re.findall(r"^### (6-\S+)", sec6, re.M)
    expected_headings = [
        "6-0.",
        "6-1.",
        "6-2.",
        "6-3.",
        "6-4.",
        "6-5.",
        "6-6.",
        "6-7.",
        "6-8.",
        "6-9.",
        "6-10.",
        "6-11.",
        "6-11b.",
        "6-12.",
        "6-12b.",
        "6-13.",
        "6-14.",
        "6-15.",
        "6-16.",
        "6-17.",
        "6-18.",
        "6-19.",
        "6-20.",
        "6-21.",
    ]
    for eh in expected_headings:
        if not any(h.startswith(eh.rstrip(".")) for h in headings):
            errors.append(f"Missing heading {eh}")

    if errors:
        print("VERIFY FAILED:")
        for e in errors:
            print(" ", e)
        return 1
    print(f"VERIFY OK: {len(CHECKS)} blocks, {len(headings)} headings, md lines={len(md.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
