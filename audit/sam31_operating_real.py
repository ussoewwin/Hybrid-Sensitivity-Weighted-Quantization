# -*- coding: utf-8 -*-
"""S2. Establish valid operating point on real images and objects for SAM3.1 detector probe.

Entrypoint wrapper around `sam31_detector_probe.run_s2_operating_point_pipeline`
writing `md/backdoor/verdicts/records/sam31_operating_real.json`.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sam31_detector_probe import resolve_comfy_dir, run_s2_operating_point_pipeline


def main():
    ap = argparse.ArgumentParser(description="SAM3.1 S2 Real Operating Point & Sensitivity Probe")
    ap.add_argument("--model", "--source", dest="model", required=True, help="Path to sam3.1_multiplex_fp16.safetensors")
    ap.add_argument("--trigger-json", default="md/backdoor/verdicts/records/trigger_candidates.json")
    ap.add_argument("--comfy-dir", default="", help="Path to ComfyUI directory (or set COMFYUI_PATH)")
    ap.add_argument("--images-dir", default="", help="Directory containing real images (defaults to <comfy-dir>/input)")
    ap.add_argument("--out", "--json", dest="out", default="md/backdoor/verdicts/records/sam31_operating_real.json")
    ap.add_argument("--n-real-images", type=int, default=24)
    ap.add_argument("--n-noise-images", type=int, default=4)
    args = ap.parse_args()

    comfy_dir = resolve_comfy_dir(args.comfy_dir)
    images_dir = os.path.abspath(args.images_dir) if args.images_dir else os.path.join(comfy_dir, "input")
    model_path = os.path.abspath(args.model)
    trigger_json_path = os.path.abspath(args.trigger_json)
    out_json_path = os.path.abspath(args.out)

    rec = run_s2_operating_point_pipeline(
        model_path=model_path,
        trigger_json_path=trigger_json_path,
        images_dir=images_dir,
        comfy_dir=comfy_dir,
        out_json_path=out_json_path,
        n_real_images=args.n_real_images,
        n_noise_images=args.n_noise_images,
    )
    return 0 if rec["gate_s2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
