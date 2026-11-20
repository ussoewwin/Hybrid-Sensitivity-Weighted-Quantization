#!/usr/bin/env python3
"""Entry point at the repository root for the Flux1 ConvRot W4A8 converter.

The implementation lives in ``Flux1/native_convert_w4a8_flux.py``. This file only
loads it, so the converter can be started from the repository root as well as
from the ``Flux1`` directory, with the same arguments either way:

    python native_convert_w4a8_flux.py --model <flux1.safetensors> --output <out.safetensors> \
        --clip_path <t5xxl.safetensors> --clip_l_path <clip_l.safetensors> --comfy_path <ComfyUI>

Nothing is duplicated here: there is one implementation, and this is a loader for it.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "Flux1", "native_convert_w4a8_flux.py")

if not os.path.isfile(TARGET):
    raise SystemExit("Error: Flux1/native_convert_w4a8_flux.py is missing next to this file")

sys.argv[0] = TARGET
runpy.run_path(TARGET, run_name="__main__")
