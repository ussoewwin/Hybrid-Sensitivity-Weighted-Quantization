"""CN-4's second number: the two PNGs compared.

CN-4 records two things - that there is a picture, and how far it is from the source - and this is the
second one. The pictures are decoded and compared as pixels: mean absolute difference per channel,
largest single-channel difference, p99, pixel cosine, PSNR and the share of pixels that did not change.

The record carries no route. Which loader drew the two PNGs is not a CN-4 result (skill 2.4).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the source arm's PNG")
    ap.add_argument("--pack", required=True, help="the pack arm's PNG")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    src, size_s = load(a.source)
    pck, size_p = load(a.pack)
    if size_s != size_p:
        raise SystemExit("the two PNGs are different sizes: %s vs %s" % (size_s, size_p))

    d = np.abs(src - pck)
    mse = float(np.mean((src - pck) ** 2))
    cos = float(np.sum(src * pck) / (np.linalg.norm(src) * np.linalg.norm(pck)))

    out = {
        "tool": "audit/ipadapter_faceid_plusv2_sdxl_pixel_compare.py",
        "family": "ipadapter_faceid_plusv2_sdxl",
        "size": list(size_s),
        "mean_abs_diff": round(float(d.mean()), 6),
        "max_abs_diff": float(d.max()),
        "p99_abs_diff": float(np.percentile(d, 99)),
        "cosine": round(cos, 9),
        "psnr_db": round(10 * np.log10(255.0 ** 2 / mse), 6) if mse else float("inf"),
        "pixels_identical_ratio": round(float(np.mean(np.all(d == 0, axis=-1))), 9),
        "bytes": {"source": os.path.getsize(a.source), "pack": os.path.getsize(a.pack)},
    }
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "tool"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
