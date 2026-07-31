#!/usr/bin/env python3
"""
Thin launcher: run `python seedvr2_nvfp4_bench.py ...` from seedvr2_videoupscaler/.

Implementation lives in ../benchmark/seedvr2_nvfp4_bench.py.
"""
from pathlib import Path
import runpy

_TARGET = Path(__file__).resolve().parent.parent / "benchmark" / "seedvr2_nvfp4_bench.py"
runpy.run_path(str(_TARGET), run_name="__main__")
