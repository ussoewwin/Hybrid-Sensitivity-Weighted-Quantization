# -*- coding: utf-8 -*-
"""SAM 3.1 Multiplex ConvRot INT8 converter driven by measured impact.

The protection set comes from an impact measurement of THIS checkpoint
(sibling script sdxl/diag_impact_sam31.py), not from static weight statistics.
For SAM the measured quantity is the drift of the model's own output (mask
logits from the tracker SAM-decoder path and the detector's boxes/scores/masks
from the text path) when a single layer is replaced by its ConvRot INT8
reconstruction.

Quantization scope - three rules, all confirmed by a measured quality gate
against the FP16 baseline on the real ComfyUI runtime (tracker mask IoU):

1. **Only `nn.Linear` layers are quantized.** ComfyUI's INT8 load path
   (`MixedPrecisionOps` + the HSWQ loader patch) attaches an INT8
   `QuantizedTensor` to Linear layers; Conv2d layers stay FP16. Writing a
   quantized Conv2d into the checkpoint therefore produced a *mixed* model -
   Conv2d entries read as unrotated INT8 while the runtime kept FP16 - which
   corrupted the mask (measured IoU 0.00, i.e. an inverted mask).

2. **The group size is fixed at 256.** ComfyUI's ConvRot kernel is 2D-only and
   its fused path only activates when `convrot_groupsize == 256`
   (`comfy/ops.py`, `comfy_kitchen/backends/cuda`). A Linear whose input width
   is not a multiple of 256 cannot be rotated by that kernel and would be read
   back as unrotated INT8; such layers stay in their original precision.

3. **Only modules the runtime actually has are considered.** The checkpoint
   stores more parameter names than the ComfyUI runtime exposes; quantizing a
   name the runtime does not have yields a mixed model. The converter asks
   ComfyUI for the runtime module list and only converts a layer that is a real
   `nn.Linear` whose weight is present in the runtime state dict.

4. **The `detector.backbone.language_backbone.*` subtree is kept, not dropped.**
   It looks like dead weight in the mask path (zeroing it leaves both the
   tracker mask and the detector boxes/scores/masks bit-identical), but ComfyUI
   uses it as the **CLIP text encoder**: `comfy/supported_models.py` sets
   `text_encoder_key_prefix = ["detector.backbone.language_backbone."]` and
   stashes the non-resizer keys into `_clip_stash` to build the CLIP. Dropping
   the subtree makes `load_checkpoint_guess_config` return `clip=None`, and
   every `CLIPTextEncode` then fails with "clip input is invalid: None".
   (An earlier revision of this converter dropped it and broke the CLIP; do not
   reintroduce that.)

With rules 1-3 applied, the measured quality gate on the real loader path is:
tracker mask flip 0.0022 / IoU 0.9978 / logit relMSE 5.1e-04 at the shipped
0.02 threshold, and tracker mask flip 0.0016 / IoU 0.9983 / relMSE 4.3e-04 at
0.05. The two are equivalent in quality while 0.02 protects more layers (225
vs 174), so 0.02 is the shipped default. The remaining detector-path difference
is a bfloat16-vs-float16 dtype effect, not a quantization effect (measured by
comparing FP16-weights-bf16-activations against INT8-weights-bf16-activations:
tracker flip 0.0012, detector IoU 0.951).

Usage:
  python clip_convert/convert_sam31_convrot_int8.py \
      --input sam3.1_multiplex_fp16.safetensors \
      --output sam3.1_multiplex_convrot_int8_t002.safetensors \
      --impact impact/impact_sam3.1_multiplex.json \
      --comfy_path <ComfyUI checkout> \
      --drift-threshold 0.02
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 256
DEFAULT_DRIFT_THRESHOLD = 0.02

# The checkpoint's `detector.backbone.language_backbone.*` subtree is NOT dead
# weight: ComfyUI treats it as the CLIP text encoder
# (comfy/supported_models.py: text_encoder_key_prefix =
# ["detector.backbone.language_backbone."], and the non-resizer keys are
# stashed into _clip_stash to build the CLIP). Removing it makes
# load_checkpoint_guess_config return clip=None, and any CLIPTextEncode node
# then fails with "clip input is invalid: None". It is therefore kept.
DEAD_PREFIXES = ()


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size ** 0.5)


def rotate_weight(w, h, gs):
    """Rotate a 2D weight by the Hadamard matrix, group-wise along the input
    axis: W_rot = W @ H^T. This is the convention ComfyUI's ConvRot kernel
    inverts at runtime."""
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def quant_rowwise(w):
    amax = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    s = amax / 127.0
    q = (w / s.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float32)


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def runtime_usable_modules(comfy_path: str, model_path: str):
    """Return (usable, all_modules).

    `usable` is the set of `nn.Linear` module names the ComfyUI runtime
    actually has and whose weight is present in the runtime state dict.
    `all_modules` is every module name the runtime has (used to tell a module
    the runtime folds away from one it merely keeps in another precision)."""
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)
    cwd = os.getcwd()
    try:
        os.chdir(comfy_path)
        import comfy.utils
        import comfy.sd
        sd = comfy.utils.load_torch_file(model_path)
        saved = sys.argv
        sys.argv = [saved[0]]
        try:
            m = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
        finally:
            sys.argv = saved
        net = m.model
        inner = getattr(net, "diffusion_model", net)
        msd = inner.state_dict()
        types = {name: type(mod).__name__ for name, mod in inner.named_modules()}
        usable = {name for name, t in types.items()
                  if t == "Linear" and (name + ".weight") in msd}
        return usable, set(types.keys())
    finally:
        os.chdir(cwd)


def load_protect_list(path):
    """The FP16-kept layers, from a JSON {"protect": [...]} or a text file
    with one runtime module name per line."""
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


def convert(input_path, output_path, impact_json, comfy_path,
            drift_threshold=DEFAULT_DRIFT_THRESHOLD, protect_list=None):
    print(f"Loading (streamed): {input_path}")
    src = safe_open(input_path, framework="pt", device="cpu")
    keys = sorted(src.keys())
    print(f"  raw keys: {len(keys)}")

    impacts = json.load(open(impact_json, encoding="utf-8"))["impacts"]
    usable, all_modules = runtime_usable_modules(comfy_path, input_path)
    print(f"  runtime modules: {len(all_modules)} (usable Linear: {len(usable)})")

    protect = {n for n, drift in impacts.items() if drift > drift_threshold}
    explicit = load_protect_list(protect_list)
    if explicit:
        # An explicit list is authoritative: the per-layer drift is a marginal
        # effect (it measures the weight error only, while the pack's real
        # path also quantizes the activation), so it does not rank the layers
        # the way the aggregate output does. The list comes from the
        # aggregate contribution measured through the real pack path
        # (sdxl/select_sam31_protect.py).
        protect = set(explicit)
    print(f"  measured candidates: {len(impacts)}; drift > {drift_threshold} -> protected {len(protect)}"
          + (" (an explicit list)" if explicit else ""))

    tensors = {k: src.get_tensor(k) for k in keys}

    out = {}
    meta_layers = {}
    n_lin = n_prot = n_kept = 0
    n_no_runtime = n_nonlinear = n_dead = 0
    h = build_hadamard(GROUP_SIZE)
    for key in sorted(tensors):
        t = tensors[key]
        if ".attn.freqs_cis" in key:
            continue
        if key.startswith(DEAD_PREFIXES):
            n_dead += 1
            continue
        if not (key.endswith(".weight") and t.ndim in (2, 4)
                and t.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            out[key] = t
            n_kept += 1
            continue
        mk = key[: -len(".weight")]
        # rule 3: the runtime must actually have this module
        if mk not in usable:
            out[key] = t
            if mk in all_modules:
                n_nonlinear += 1
            else:
                n_no_runtime += 1
            continue
        # protection from the measured impact
        if mk in protect:
            out[key] = t
            n_prot += 1
            continue
        # rules 1-2: Linear only, input width a multiple of 256 (see docstring)
        inf = int(t.shape[1])
        of = int(t.shape[0])
        if t.ndim != 2 or inf % GROUP_SIZE != 0 or of % 4 != 0 or inf % 4 != 0:
            out[key] = t
            n_prot += 1
            continue
        q, scale = quant_rowwise(rotate_weight(t.float(), h, GROUP_SIZE))
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
        out[key] = q
        out[f"{mk}.weight_scale"] = scale
        out[f"{mk}.comfy_quant"] = _encode(conf)
        meta_layers[mk] = conf

    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    print(f"\nSaving: {output_path}")
    print(f"  ConvRot INT8 Linear: {n_lin}")
    print(f"  protected (drift or shape): {n_prot}")
    print(f"  dropped (dead, runtime never loads): {n_dead}")
    print(f"  kept (no runtime module / not Linear): {n_no_runtime} / {n_nonlinear}")
    print(f"  kept as-is (1D, buffers): {n_kept}")
    save_file(out, output_path, metadata=metadata)
    print(f"  Input {os.path.getsize(input_path)/(1024**3):.2f} GB -> Output {os.path.getsize(output_path)/(1024**3):.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SAM 3.1 ConvRot INT8 converter with measured impact protection")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--impact", required=True)
    ap.add_argument("--comfy_path", required=True)
    ap.add_argument("--drift-threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    ap.add_argument("--protect-list", default=None,
                    help="the FP16-kept layers, as a JSON {\"protect\": [...]} or a "
                         "text file with one runtime module name per line; an "
                         "explicit list is authoritative over the drift threshold")
    a = ap.parse_args()
    convert(a.input, a.output, a.impact, a.comfy_path, drift_threshold=a.drift_threshold,
            protect_list=a.protect_list)
