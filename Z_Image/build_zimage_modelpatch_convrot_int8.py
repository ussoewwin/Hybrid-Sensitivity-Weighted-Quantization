"""Build the Z-Image-Turbo-Fun-Controlnet model-patch ConvRot INT8 pack.

This is a **model patch**, not a ControlNet: ComfyUI loads it with
``ModelPatchLoader`` and applies it with the ``ZImageFunControlnet`` node. The
pack is written in the **post-conversion layout** (``z_image_convert`` already
applied: the fused ``attention.qkv``), because the HSWQ model-patch loader runs
``z_image_convert`` on load and expects the quantized keys to survive it
unchanged.

The protection set is given explicitly with ``--protect-list`` (checkpoint
names, i.e. the post-conversion runtime names). The per-layer drift is a
screen, not the selection criterion.

Usage:
    python Z_Image/build_zimage_modelpatch_convrot_int8.py \
        --input <source>.safetensors --output <pack>.safetensors \
        [--protect-list <json>] [--groupsize 256]
"""
import argparse
import json
import math
import os
import sys

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(COMFY)
import torch  # noqa: E402
import comfy.utils  # noqa: E402
import comfy_extras.nodes_model_patch as nmp  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def had(sz):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < sz:
        h = torch.kron(h, h4)
    return h / math.sqrt(sz)


_H = {}


def group_size_for(n, preferred):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and (gs & (gs - 1)) == 0 and (gs.bit_length() - 1) % 2 == 0:
            return gs
        gs //= 4
    return None


def rot(w, gs):
    if gs not in _H:
        _H[gs] = had(gs)
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), _H[gs].T).reshape(w.shape)


def qrow(w):
    a = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = a / 127.0
    return (w / s).round().clamp(-127, 127).to(torch.int8), s.float()


def keep(t, key):
    """A tensor that is not quantized: floating point goes to BF16."""
    if t.is_floating_point() and not key.endswith("weight_scale"):
        return t.to(torch.bfloat16)
    return t


def load_protect_list(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip()}
    if isinstance(data, dict):
        return set(data.get("protect", []))
    return set(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--protect-list", default=None)
    ap.add_argument("--groupsize", type=int, default=256)
    a = ap.parse_args()

    sd = comfy.utils.load_torch_file(a.input, safe_load=True)
    print("the source tensors: %d" % len(sd))
    sd = nmp.z_image_convert(sd)
    print("after z_image_convert: %d" % len(sd))

    prot = load_protect_list(a.protect_list)
    if prot:
        print("the explicit protect list: %d keys" % len(prot))

    out = {}
    meta = {}
    nlin = nprot = 0
    for k, t in sd.items():
        if not (k.endswith(".weight") and t.ndim == 2):
            out[k] = keep(t, k)
            continue
        mk = k[:-7]
        inf = int(t.shape[1])
        gs = group_size_for(inf, a.groupsize)
        if mk in prot or gs is None or int(t.shape[0]) % 4 != 0:
            out[k] = keep(t, k)
            nprot += 1
            continue
        q, s = qrow(rot(t.float(), gs))
        out[k] = q
        out[mk + ".weight_scale"] = s  # FP32 on purpose
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
        out[mk + ".comfy_quant"] = torch.tensor(
            list(json.dumps(conf, separators=(",", ":")).encode()), dtype=torch.uint8)
        meta[mk] = conf
        nlin += 1
    save_file(out, a.output,
              metadata={"_quantization_metadata": json.dumps(
                  {"format_version": "1.0", "layers": meta}, separators=(",", ":"))})
    print("  ConvRot INT8: %d | protected/kept 2D: %d" % (nlin, nprot))
    print("  the size: %.1f MB" % (os.path.getsize(a.output) / 1024 ** 2))


if __name__ == "__main__":
    main()
