# -*- coding: utf-8 -*-
"""The T5-XXL candidate packs: build one, measure it, remove it, repeat.

The pack is 12-16 GB and the disk has to hold the source, the shipped pack and
the candidate at the same time, so only one candidate is resident at a time.

The observable is the T5 encoder's hidden state, on a fixed item set, with the
baseline built through the same ops class (the stock path and the
`mixed_precision_ops` path are not numerically equivalent for this model).

Usage:
    python sweep_t5_candidates.py <source> <shipped_pack> <impact_enc> <impact_dec> <workdir> [n_items]
"""
import os
import sys
import json
import types
import statistics
import subprocess
import importlib

import torch

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-HSWQ-Loader-and-Tools")
CONV = r"D:\USERFILES\GitHub\hswq\clip_convert\convert_t5_impact_convrot_int8.py"
IMP = r"D:\USERFILES\GitHub\hswq\impact"

sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes"))
os.chdir(COMFY)
import folder_paths  # noqa: E402
import server as comfy_server  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.quant_ops import QUANT_ALGOS  # noqa: E402

if not hasattr(comfy_server.PromptServer, "instance"):
    comfy_server.PromptServer.instance = None

_spec = importlib.util.spec_from_file_location(
    "hswq_pkg_init", os.path.join(NODE, "__init__.py"), submodule_search_locations=[NODE])
_mod = importlib.util.module_from_spec(_spec)
sys.modules["hswq_pkg_init"] = _mod
_spec.loader.exec_module(_mod)

SRC = sys.argv[1]
SHIPPED = sys.argv[2]
ENC_IMP = sys.argv[3]
DEC_IMP = sys.argv[4]
WORK = sys.argv[5]
N = int(sys.argv[6]) if len(sys.argv) > 6 else 32

os.makedirs(WORK, exist_ok=True)
DT = torch.float16


def load_t5(path, device="cuda", dtype=DT):
    """The T5 text encoder, the way ComfyUI's `t5xxl` path loads it.

    The runtime is `comfy.text_encoders.sd3_clip`: `t5_xxl_detect` gives the
    dtype and the config, `sd3_clip(t5=True)` builds the model, and the
    checkpoint goes into the `t5xxl` member.
    """
    import comfy.text_encoders.sd3_clip as sd3
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        detect = sd3.t5_xxl_detect(sd, "")
        dtype_t5 = detect.pop("dtype_t5", dtype)
        clip = sd3.sd3_clip(clip_l=False, clip_g=False, t5=True,
                            dtype_t5=dtype_t5, **detect)(
            device=device, dtype=dtype)
    finally:
        sys.argv = saved
    clip.t5xxl.load_sd(sd)
    return clip.to(device)


def make_batch(n_tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(100, 30000, (1, n_tokens), generator=g)
    mask = torch.ones_like(ids)
    return ids.to(device), mask.to(device)


def hidden(clip, ids, mask):
    with torch.no_grad():
        out = clip.t5xxl.transformer(ids, mask)
    h = out[0] if isinstance(out, (tuple, list)) else out
    return h.detach().float().reshape(-1).cpu()


def main():
    ops = comfy_ops.mixed_precision_ops(
        {"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, DT,
        full_precision_mm=False, disabled=[])
    print("[1] the FP16 baseline ...", flush=True)
    c16 = load_t5(SRC)
    ids, mask = make_batch(256, 42, "cuda")
    ref = hidden(c16, ids, mask)
    print("    the reference: %s finite=%s" % (
        tuple(ref.shape), bool(torch.isfinite(ref).all())), flush=True)
    del c16
    torch.cuda.empty_cache()

    def measure(path):
        c8 = load_t5(path)
        ck = comfy_ops.quant_ops.ck
        orig = ck.int8_linear
        cnt = [0]

        def counted(*a, **kw):
            cnt[0] += 1
            return orig(*a, **kw)

        ck.int8_linear = counted
        try:
            h = hidden(c8, ids, mask)
        finally:
            ck.int8_linear = orig
        cos = torch.nn.functional.cosine_similarity(
            ref.reshape(1, -1), h.reshape(1, -1)).item()
        del c8
        torch.cuda.empty_cache()
        return cos, cnt[0], bool(torch.isfinite(h).all())

    rows = []
    for path, tag in ((SHIPPED, "the shipped pack"),
                      (os.path.join(WORK, "t5_e40_d80.safetensors"), "e40_d80")):
        if not os.path.isfile(path):
            continue
        cos, cnt, fin = measure(path)
        mb = os.path.getsize(path) / 1024 ** 2
        print("    [%s] cos=%.6f int8_linear=%d finite=%s size=%.1f MB" % (
            tag, cos, cnt, fin, mb), flush=True)
        rows.append({"tag": tag, "cos": cos, "int8": cnt, "file_mb": mb})

    for tag in ():
        pl = os.path.join(IMP, "t5_protect_%s.json" % tag)
        outp = os.path.join(WORK, "t5_%s.safetensors" % tag)
        print("[3] the candidate %s ..." % tag, flush=True)
        cmd = [sys.executable, CONV, "--input", SRC, "--output", outp,
               "--impact", ENC_IMP, "--comfy_path", COMFY,
               "--drift-threshold", "0.25", "--quantize-decoder",
               "--decoder-impact", DEC_IMP, "--decoder-max-drift", "0.25",
               "--protect-list", pl]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print("    the converter failed: %s" % r.stderr[-400:], flush=True)
            continue
        cos, cnt, fin = measure(outp)
        mb = os.path.getsize(outp) / 1024 ** 2
        rows.append({"tag": tag, "cos": cos, "int8": cnt, "file_mb": mb})
        print("    cos=%.6f int8_linear=%d finite=%s size=%.1f MB" % (cos, cnt, fin, mb), flush=True)
        os.remove(outp)

    print()
    print("%-18s %12s %12s %12s" % ("candidate", "cos", "int8", "MB"))
    for r in rows:
        print("%-18s %12.6f %12d %12.1f" % (r["tag"], r["cos"], r["int8"], r["file_mb"]))
    print("JSON " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
