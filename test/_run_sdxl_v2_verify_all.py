"""Run all SDXL V2.0 script checks (static + py_compile + optional venv import)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "quantize_sdxl_hswq_v2.0.py"
VENV_PY = Path(r"D:\USERFILES\fp8e4m3\venv\Scripts\python.exe")


def run(cmd: list[str], label: str) -> int:
    print(f"\n=== {label} ===")
    print(" ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"FAIL: {label} (exit {r.returncode})")
    else:
        print(f"PASS: {label}")
    return r.returncode


def main() -> int:
    rc = 0
    rc |= run([sys.executable, str(ROOT / "test/_verify_sdxl_v2_static.py")], "static")
    rc |= run([sys.executable, "-m", "py_compile", str(TARGET)], "py_compile")
    if VENV_PY.is_file():
        rc |= run(
            [str(VENV_PY), str(ROOT / "test/_verify_sdxl_v2_import.py")],
            "venv import + remap smoke",
        )
    else:
        print(f"\nSKIP: venv import ({VENV_PY} not found)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
