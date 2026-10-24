# -*- coding: utf-8 -*-
"""SD1.5 UNet ConvRot INT8 pack builder driven by the measured aggregate
contribution.

The protection set comes from `sd1.5/measure_sd15_unet_contrib.py` (the
contribution measured on the UNet's OWN output, float64, 8 seeds). The top-K
layers by `1 - cos` are kept in FP16; every other ConvRot-eligible Linear/Conv2d
goes to INT8.

**CLIP is not quantized** (the caller's instruction): only the UNet subtree
(`model.diffusion_model.`) is touched. The checkpoint's CLIP passes through
byte-identical, and **the VAE is dropped** - the pack ships UNet + CLIP only.

Bias correction (optional, `--bias_correction`, default OFF - same default as
V1.0): folds the systematic INT8 output shift into the layer bias,

    E[(W_q - W) x]  ~  (W_q - W) @ E[x]

using per-input-channel activation means collected **in this same run** (one
command, exactly as `sd1.5/quantize_sd1.5_hswq_v1.0.py` does it). Enabling it
requires `--comfy` because the calibration needs a forward pass.

Rules:
  1. Only 2D Linear and 4D Conv2d weights under `model.diffusion_model.` are
     quantized. Conv2d rotates along the input-channel axis.
  2. Group size 256, falling back to 64 / 16 / 4 where the input width demands.
  3. Kept tensors are stored FP16 (the runtime runs FP16); the INT8
     `weight_scale` stays FP32 by design, with its rank matching the weight
     (`(O,1)` for 2D, `(O,1,1,1)` for 4D).

Usage:
  python sd1.5/build_sd15_unet_convrot_int8.py \
      --input <checkpoint>.safetensors --output <pack>.safetensors \
      --contrib impact/contrib_sd15_unet.json --top-k 80 \
      [--comfy <ComfyUI root> --bias_correction]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file, save_file

GROUP_SIZE = 256
UNET_PREFIX = "model.diffusion_model."


def build_hadamard(size, dtype=torch.float32):
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError("Hadamard size must be a power of 4, got %s" % size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / (size ** 0.5)


def group_size_for(n, preferred=GROUP_SIZE):
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_linear(w, h, gs):
    of, inf = w.shape
    g = inf // gs
    return torch.matmul(w.view(of, g, gs), h.T.to(w.dtype)).reshape(w.shape)


def rotate_conv(w, h, gs):
    oc, ic, kh, kw = w.shape
    flat = w.permute(0, 2, 3, 1).contiguous().view(-1, ic)
    of, inf = flat.shape
    g = inf // gs
    rot = torch.matmul(flat.view(of, g, gs), h.T.to(w.dtype)).reshape(flat.shape)
    return rot.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()


def quant_rowwise(w):
    """Per-OUTPUT-CHANNEL scale, stored in the weight's own rank.

    `amax(dim=-1)` is right for a 2D Linear ([O, I] -> (O,1)), but wrong for a
    4D Conv: the scale must reduce over every non-output axis and be stored
    (O,1,1,1). A flattened `(O,)` does not broadcast the same way and produced
    NaN through the Conv2d path.
    """
    rd = tuple(range(1, w.dim()))
    amax = w.abs().amax(dim=rd).float().clamp(min=1e-30)
    s = amax / 127.0
    if w.dim() == 2:
        sv = s.view(-1, 1)
    elif w.dim() == 4:
        sv = s.view(-1, 1, 1, 1)
    else:
        raise ValueError("unsupported weight rank %d" % w.dim())
    q = (w / sv.to(w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, sv.to(torch.float32)


def bias_delta(weight_fp, weight_dq, act_mean):
    """The systematic output shift of an INT8 layer, to cancel into the bias.

        E[(W_q - W) x]  ~  (W_q - W) @ E[x]

    Linear (O, I)       -> delta[o] = sum_i err[o,i] * mu[i]
    Conv2d (O, I, K, K) -> delta[o] = sum_{i,kh,kw} err[o,i,kh,kw] * mu[i]

    Same formula as V1.0's `compute_int8_bias_delta`
    (`sd1.5/quantize_sd1.5_hswq_v1.0.py`).
    """
    if act_mean is None:
        return None
    err = (weight_dq.float() - weight_fp.float())
    mu = act_mean.float()
    if err.ndim == 2:
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None


def _encode(config):
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
                        dtype=torch.uint8)


# --------------------------------------------------------------------------
# the calibration pass - IDENTICAL to V1.0 (sd1.5/quantize_sd1.5_hswq_v1.0.py)
#
# V1.0 does NOT use `from_single_file`. It builds the stock SD1.5 pipeline with
# `from_pretrained`, then remaps this checkpoint's ComfyUI keys onto diffusers
# keys and loads the UNet weights into that pipeline. `from_single_file` fails on
# this checkpoint (`CLIPTextModel has no attribute text_model`), which is why
# the key-mapping route is the one that works.
#
# The two helpers are imported from V1.0 itself rather than re-implemented, so
# the calibration cannot drift from it:
#   detect_unet_config_from_keys(state_dict)
#   unet_to_diffusers_mapping(unet_config, state_dict)
# --------------------------------------------------------------------------
def _import_v10_helpers():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    v10 = os.path.join(here, "quantize_sd1.5_hswq_v1.0.py")
    if not os.path.exists(v10):
        sys.exit("error: %s not found (needed for the calibration key mapping)" % v10)
    spec = importlib.util.spec_from_file_location("hswq_sd15_v10", v10)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hswq_sd15_v10"] = mod
    spec.loader.exec_module(mod)
    return mod.detect_unet_config_from_keys, mod.unet_to_diffusers_mapping


def collect_act_means(ckpt_path, calib_file, n_samples=32, n_steps=25, seed=42,
                      base_repo="stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Per-input-channel activation means, via the SAME calibration V1.0 runs.

    * pipeline: `StableDiffusionPipeline.from_pretrained(base_repo)`; the UNet is
      then overwritten with this checkpoint's weights (V1.0's route)
    * prompts: one per line in `calib_file`, cycled to `n_samples` (V1.0)
    * 25 steps, seed 42, `output_type="latent"` (V1.0)
    * hooks on every Conv2d/Linear; `x.mean(dim=(0,2,3))` for 4D (V1.0)
    """
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel  # noqa: F401
    from safetensors.torch import load_file
    import torch as _t

    detect_unet_config_from_keys, unet_to_diffusers_mapping = _import_v10_helpers()

    dev = "cuda" if _t.cuda.is_available() else "cpu"
    dt = _t.float16 if dev == "cuda" else _t.float32

    print("    the state dict: %s" % ckpt_path, flush=True)
    state_dict = load_file(ckpt_path)
    unet_config = detect_unet_config_from_keys(state_dict)
    print("    the detected UNet config: %s" % unet_config, flush=True)

    print("    the pipeline: %s" % base_repo, flush=True)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_repo, torch_dtype=dt, use_safetensors=True, variant="fp16").to(dev)
    except Exception as e:
        print("    [warn] from_pretrained failed (%s); building a bare pipeline" % str(e)[:80], flush=True)
        unet = UNet2DConditionModel(
            sample_size=64, in_channels=4, out_channels=4, layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                              "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=768, attention_head_dim=8,
            use_linear_projection=False)
        pipe = StableDiffusionPipeline(
            vae=None, text_encoder=None, tokenizer=None, unet=unet,
            scheduler=None, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False).to(dev)

    try:
        unet_dev = str(next(pipe.unet.parameters()).device)
    except StopIteration:
        unet_dev = "unknown"
    if not unet_dev.startswith("cuda") and dev == "cuda":
        raise RuntimeError("the UNet landed on %r, not CUDA; refusing to calibrate "
                           "(it would hang at 0/25 on CPU fp16)" % unet_dev)

    # the VAE is never used with output_type="latent" - keep it off the GPU
    _vae = getattr(pipe, "vae", None)
    if isinstance(_vae, _t.nn.Module):
        _vae.to("cpu")

    print("    remapping the UNet weights ...", flush=True)
    mapping = unet_to_diffusers_mapping(unet_config, state_dict)
    new_sd = {dk: state_dict[ck] for ck, dk in mapping.items() if ck in state_dict}
    m, u = pipe.unet.load_state_dict(new_sd, strict=False)
    print("    the UNet loaded: %d tensors (missing %d / unexpected %d)" % (
        len(new_sd), len(m), len(u)), flush=True)

    unet = pipe.unet
    acc, cnt, handles = {}, {}, []

    def hook_fn(module, inputs, output, name=None):
        x = inputs[0].detach().float()
        if x.dim() == 4:
            ch = x.mean(dim=(0, 2, 3))
        elif x.dim() == 3:
            ch = x.mean(dim=(0, 1))
        elif x.dim() == 2:
            ch = x.mean(dim=0)
        else:
            return
        if name not in acc:
            acc[name] = _t.zeros_like(ch)
            cnt[name] = 0
        acc[name] += ch
        cnt[name] += 1

    for name, module in unet.named_modules():
        if isinstance(module, (_t.nn.Conv2d, _t.nn.Linear)):
            handles.append(module.register_forward_hook(
                lambda mm_, ii, oo, n=name: hook_fn(mm_, ii, oo, n)))

    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(prompts) < n_samples:
        prompts = (prompts * (n_samples // len(prompts) + 1))[:n_samples]
    else:
        prompts = prompts[:n_samples]

    print("    the calibration: %d samples x %d steps (seed %d)" % (
        len(prompts), n_steps, seed), flush=True)
    generator = _t.Generator(device=dev).manual_seed(seed)
    for i, prompt in enumerate(prompts):
        with _t.no_grad():
            pipe(prompt=prompt, num_inference_steps=n_steps,
                 output_type="latent", generator=generator)
        if (i + 1) % 8 == 0:
            print("      [%d/%d]" % (i + 1, len(prompts)), flush=True)

    for h in handles:
        h.remove()
    del pipe, unet
    if _t.cuda.is_available():
        _t.cuda.empty_cache()

    # The hooks record DIFFUSERS names; the pack uses COMFY names. Translate back
    # so the builder can look each layer up by its checkpoint-relative name.
    inv = {dk: ck for ck, dk in mapping.items()}
    out_means = {}
    for dname, mu in acc.items():
        mean = (mu / cnt[dname]).cpu()
        ck = inv.get(dname) or inv.get(dname + ".weight")
        if ck is None:
            ck = dname
        # strip ".weight" and the unet prefix -> the pack's `short` name
        key = ck[:-len(".weight")] if ck.endswith(".weight") else ck
        key = key.replace("model.diffusion_model.", "")
        out_means[key] = mean
    print("    the activation means: %d layers" % len(out_means), flush=True)
    return out_means


def convert(input_path, output_path, contrib_json, top_k,
            bias_correction=False, calib_file="", n_calib=32, n_steps=25):
    src = load_file(input_path)

    contrib = json.load(open(contrib_json, encoding="utf-8"))["impacts"]
    ranked = sorted(((n, v) for n, v in contrib.items() if v is not None),
                    key=lambda kv: -kv[1])
    protect = {n for n, _ in ranked[:top_k]}
    print("the measured candidates: %d | the adopted protection: %d" % (len(ranked), len(protect)))

    act_means = {}
    if bias_correction:
        if not calib_file:
            sys.exit("error: --bias_correction needs --calib_file <prompts.txt>")
        act_means = collect_act_means(input_path, calib_file,
                                      n_samples=n_calib, n_steps=n_steps)

    h = build_hadamard(GROUP_SIZE)
    out, meta_layers = {}, {}
    n_lin = n_prot = n_kept = n_clip = n_vae_dropped = n_bias = 0
    _bias_miss_no_mean = _bias_miss_none = 0
    for name in sorted(src):
        t = src[name]
        if name.startswith("first_stage_model."):
            # the VAE is dropped entirely (this delivery ships UNet + CLIP only)
            n_vae_dropped += 1
            continue
        if not name.startswith(UNET_PREFIX):
            # CLIP passes through untouched (it is not quantized)
            if name.startswith("cond_stage_model."):
                n_clip += 1
            out[name] = t
            n_kept += 1
            continue
        # Linear AND Conv2d both go INT8. comfy_kitchen has no int8_conv, but the
        # HSWQ loader's `_patch_ops_decode_and_conv()` installs the Conv2d path.
        # Running an INT8 Conv2d pack WITHOUT that arm is what produced NaN.
        if not (name.endswith(".weight") and t.ndim in (2, 4)):
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            n_kept += 1
            continue
        mk = name[:-len(".weight")]
        short = name[len(UNET_PREFIX):-len(".weight")]
        if short in protect:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        inf = int(t.shape[1])
        gs = group_size_for(inf)
        if gs is None or int(t.shape[0]) % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        hh = h if gs == GROUP_SIZE else build_hadamard(gs)
        w = t.float()
        rot = rotate_linear(w, hh, gs) if t.ndim == 2 else rotate_conv(w, hh, gs)
        q, scale = quant_rowwise(rot)
        n_lin += 1
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
        out[name] = q
        out["%s.weight_scale" % mk] = scale
        out["%s.comfy_quant" % mk] = _encode(conf)
        meta_layers[mk] = conf

        # bias correction: fold (W_q - W) @ E[x] into the layer bias, in the
        # ORIGINAL frame (the pack stores the rotated weight, so the dequantised
        # weight has to be rotated back before forming the error)
        if act_means:
            mu = act_means.get(short)
            if mu is None:
                _bias_miss_no_mean += 1
            if mu is not None:
                if t.ndim == 2:
                    w_dq = torch.matmul((q.float() * scale).view(t.shape[0], -1, gs), hh).reshape(t.shape)
                else:
                    qf = q.float() * scale
                    oc, ic, kh, kw = t.shape
                    flat = qf.permute(0, 2, 3, 1).contiguous().view(-1, ic)
                    back = torch.matmul(flat.view(-1, ic // gs, gs), hh).reshape(flat.shape)
                    w_dq = back.view(oc, kh, kw, ic).permute(0, 3, 1, 2).contiguous()
                d = bias_delta(t.float(), w_dq, mu)
                if d is None:
                    _bias_miss_none += 1
                if d is not None:
                    bkey = "%s.bias" % mk
                    base_bias = src.get(bkey)
                    if base_bias is None:
                        base_bias = torch.zeros_like(d)
                    out[bkey] = (base_bias.float() - d).to(torch.float16)
                    n_bias += 1

    metadata = {"_quantization_metadata": json.dumps(
        {"format_version": "1.0", "layers": meta_layers}, separators=(",", ":"))}
    save_file(out, output_path, metadata=metadata)
    print("  the ConvRot INT8: %d" % n_lin)
    print("  the protected (FP16 kept 2D/4D): %d" % n_prot)
    print("  the untouched (CLIP/1D): %d (of which CLIP %d)" % (n_kept, n_clip))
    print("  the VAE tensors DROPPED: %d" % n_vae_dropped)
    if act_means:
        print("  the bias-corrected layers: %d (skipped: %d had no act_mean, %d had a shape mismatch)"
              % (n_bias, _bias_miss_no_mean, _bias_miss_none))
        _expected = sum(1 for _k in meta_layers)
        if n_bias == 0:
            raise RuntimeError("bias correction was requested but wrote 0 layers - refusing to "
                               "save a pack labelled as bias-corrected")
    print("  the output: %.1f MB" % (os.path.getsize(output_path) / 1024 ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--calib_file", default="",
                    help="calibration prompts, one per line (required with "
                         "--bias_correction; the repo ships "
                         "sample/calibration_prompts_128.txt)")
    ap.add_argument("--bias_correction", action="store_true",
                    help="cancel the systematic INT8 output bias into the layer "
                         "bias (default OFF, same default as V1.0)")
    ap.add_argument("--num_calib_samples", type=int, default=32,
                    help="calibration samples (V1.0 default: 32)")
    ap.add_argument("--num_inference_steps", type=int, default=25,
                    help="denoising steps per calibration sample (V1.0 default: 25)")
    a = ap.parse_args()
    convert(a.input, a.output, a.contrib, a.top_k,
            bias_correction=a.bias_correction, calib_file=a.calib_file,
            n_calib=a.num_calib_samples, n_steps=a.num_inference_steps)
