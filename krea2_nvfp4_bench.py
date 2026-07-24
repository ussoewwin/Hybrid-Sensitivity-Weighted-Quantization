#!/usr/bin/env python3
"""Thin launcher: run `python krea2_nvfp4_bench.py ...` from the repo root.

Implementation lives in benchmark/krea2_nvfp4_bench.py.
"""
from pathlib import Path
import runpy

_TARGET = Path(__file__).resolve().parent / "benchmark" / "krea2_nvfp4_bench.py"
runpy.run_path(str(_TARGET), run_name="__main__")
