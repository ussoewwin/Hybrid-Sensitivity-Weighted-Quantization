"""Build the CLIP-SAE-ViT-L-14 ConvRot INT8 pack, impact-driven.

The pack carries both towers of the checkpoint - the text tower and the vision
tower - under the checkpoint's own names, which is what ComfyUI's loader reads.

The FP16-kept set is given explicitly with `--protect-list`, measured on the
**aggregate** output through the real pack path. The per-layer drift is a
screen, not the selection criterion: it measures the WEIGHT error only (the
reconstruction is placed in the module and the plain forward is used), while
the pack's real path also quantizes the ACTIVATION for that layer, and that is
where most of the error comes from.

The list holds the checkpoint's names, with the tower's prefix:
  text_model.encoder.layers.N.*        the text tower
  vision_model.encoder.layers.N.*      the vision tower
  text_model.embeddings.token_embedding, text_projection, ...

The names come from the impact measurement, which uses the runtime's names
(`transformer.text_model.encoder.layers.N.*` for the text tower and
`encoder.layers.N.*` for the vision tower), so they are mapped here the same
way the old threshold-based protection was.

Usage:
    python sdxl/build_sae_vitl14_convrot_int8.py \
        --impact-text impact/impact_sae_vitl14_text_fixed.json \
        --impact-vision impact/impact_sae_vitl14_vision_fixed.json \
        --protect-list impact/sae_protect_<n>.json --out <pack>
"""
import argparse
import json
import math
import os
import sys

COMFY = r"D:\USERFILES\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch  # noqa: E402
import comfy.utils  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

D = r"D:\USERFILES\ComfyUI\ComfyUI\models\clip"
SRC = os.path.join(D, "CLIP-SAE-ViT-L-14-FP32.safetensors")
IMP = r"D:\USERFILES\GitHub\hswq\impact"
GROUP = 256


def text_to_checkpoint(name):
    """The checkpoint's name for a measured text-tower layer."""
    if name.startswith("transformer.text_model."):
        return "text_model." + name[len("transformer.text_model."):]
    return name


def vision_to_checkpoint(name):
    """The checkpoint's name for a measured vision-tower layer."""
    if name.startswith("encoder."):
        return "vision_model." + name
    return name


def load_protect_list(path):
    """The FP16-kept layers, in the checkpoint's names."""
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


def had(sz):
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < sz:
        h = torch.kron(h, h4)
    return h / math.sqrt(sz)


_H = {}


def rot(w, gs=GROUP):
    if gs not in _H:
        _H[gs] = had(gs)
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), _H[gs].T).reshape(w.shape)


def qrow(w):
    a = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    s = a / 127.0
    return (w / s).round().clamp(-127, 127).to(torch.int8), s.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impact-text", required=True)
    ap.add_argument("--impact-vision", required=True)
    ap.add_argument("--protect-list", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--drift-threshold", type=float, default=1e9,
                    help="only used when --protect-list is not given")
    args = ap.parse_args()

    impt = {}
    for k, v in json.load(open(args.impact_text, encoding="utf-8"))["impacts"].items():
        impt[text_to_checkpoint(k)] = v
    impv = {}
    for k, v in json.load(open(args.impact_vision, encoding="utf-8"))["impacts"].items():
        impv[vision_to_checkpoint(k)] = v

    prot = set()
    explicit = load_protect_list(args.protect_list)
    if explicit:
        prot = set(explicit)
        print(f"  the explicit protect list: {len(prot)} layers")
    else:
        prot = {k for k, v in impt.items() if v > args.drift_threshold}
        prot |= {k for k, v in impv.items() if v > args.drift_threshold}
        print(f"  the drift threshold {args.drift_threshold} -> protected {len(prot)}")

    sd = comfy.utils.load_torch_file(args.src, safe_load=True)
    out = {}
    meta = {}
    nlin = 0
    nprot = 0

    def keep(t, k):
        """A tensor that is not quantized.

        The floating-point ones go to FP16. The checkpoint is FP32, but
        ComfyUI's CLIP runs in FP16, so an FP32 kept tensor is cast to FP16 at
        load time anyway - storing FP16 loses nothing at runtime and halves
        the bytes. The INT8 scales stay FP32: they carry the dequantization
        precision. (The bigG and CLIP-L packs already ship FP16 kept tensors
        and pass the acceptance.)
        """
        if t.dtype in (torch.float32, torch.float16, torch.bfloat16) \
                and not k.endswith("weight_scale"):
            return t.to(torch.float16)
        return t

    for k, t in sd.items():
        if not (k.endswith(".weight") and t.ndim == 2):
            out[k] = keep(t, k)
            continue
        mk = k[:-7]
        inf = int(t.shape[1])
        if mk in prot or inf % GROUP != 0 or t.shape[0] % 4 != 0:
            out[k] = keep(t, k)
            nprot += 1
            continue
        q, s = qrow(rot(t.float(), GROUP))
        out[k] = q
        out[mk + ".weight_scale"] = s  # FP32 on purpose
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
        out[mk + ".comfy_quant"] = torch.tensor(
            list(json.dumps(conf, separators=(",", ":")).encode()), dtype=torch.uint8)
        meta[mk] = conf
        nlin += 1
    save_file(out, args.out,
              metadata={"_quantization_metadata": json.dumps(
                  {"format_version": "1.0", "layers": meta}, separators=(",", ":"))})
    from collections import Counter
    side = Counter("text" if k.startswith("text_model") else
                   ("vision" if k.startswith("vision_model") else "other") for k in meta)
    print(f"  ConvRot INT8: {nlin} | protected: {nprot} | the split: {dict(side)}")
    print(f"  the size: {os.path.getsize(args.out)/1024**2:.1f} MB")


if __name__ == "__main__":
    main()
