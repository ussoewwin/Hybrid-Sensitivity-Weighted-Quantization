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
command, exactly as `archives/quantize_sd1.5_hswq_v1.0.py` does it). Enabling it
requires `--comfy` because the calibration needs a forward pass.

Rules:
  1. Only 2D Linear and 4D Conv2d weights under `model.diffusion_model.` are
     quantized. Conv2d rotates along the input-channel axis.
  2. Group size 256, falling back to 64 / 16 / 4 where the input width demands.
  3. Kept tensors are stored FP16 (the runtime runs FP16); the INT8
     `weight_scale` stays FP32 by design, with its rank matching the weight
     (`(O,1)` for 2D, `(O,1,1,1)` for 4D).

Usage:
  python sd1.5/quantize_sd1.5_hswq_v1.1.py \
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
    (`archives/quantize_sd1.5_hswq_v1.0.py`).
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
# the calibration pass - IDENTICAL to V1.0 (archives/quantize_sd1.5_hswq_v1.0.py)
#
# V1.0 does NOT use `from_single_file`. It builds the stock SD1.5 pipeline with
# `from_pretrained`, then remaps this checkpoint's ComfyUI keys onto diffusers
# keys and loads the UNet weights into that pipeline. `from_single_file` fails on
# this checkpoint (`CLIPTextModel has no attribute text_model`), which is why
# the key-mapping route is the one that works.
#
# The two helpers below are carried IN THIS FILE (see the inlined block), so the
# calibration cannot drift and this script never reaches into an archive:
#   detect_unet_config_from_keys(state_dict)
#   unet_to_diffusers_mapping(unet_config, state_dict)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# the key-mapping helpers, INLINED
#
# These were previously imported from the retired V1.0 script. V1.0 now lives
# under `archives/`, and this script must not reach into an archive (nor restore
# it) - so the two functions are carried here, unchanged, and this file is
# self-contained.
# --------------------------------------------------------------------------
def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
    filtered_keys = [k for k in state_dict_keys if k.startswith(key_prefix)]
    unet_config = {}
    if f"{key_prefix}input_blocks.0.0.weight" in state_dict_keys:
        model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
        num_res_blocks = []
        channel_mult = []
        transformer_depth = []
        transformer_depth_output = []
        input_block_count = count_blocks(state_dict_keys, f"{key_prefix}input_blocks" + '.{}.')
        last_res_blocks = 0
        last_channel_mult = 0
        for count in range(input_block_count):
            prefix = f"{key_prefix}input_blocks.{count}."
            prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
            block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
            if len(block_keys) == 0: break
            block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))
            if f"{prefix}0.op.weight" in block_keys:
                num_res_blocks.append(last_res_blocks)
                channel_mult.append(last_channel_mult)
                last_res_blocks = 0
                last_channel_mult = 0
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                transformer_depth_output.append(out)
            else:
                res_block_prefix = f"{prefix}0.in_layers.0.weight"
                if res_block_prefix in block_keys:
                    last_res_blocks += 1
                    last_channel_mult = state_dict[f"{prefix}0.out_layers.3.weight"].shape[0] // model_channels
                    out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                    transformer_depth.append(out)
                res_block_prefix = f"{prefix_output}0.in_layers.0.weight"
                if res_block_prefix in block_keys_output:
                    out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                    transformer_depth_output.append(out)
        num_res_blocks.append(last_res_blocks)
        channel_mult.append(last_channel_mult)
        if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
            transformer_depth_middle = count_blocks(state_dict_keys, f"{key_prefix}middle_block.1.transformer_blocks." + '{}')
        elif f"{key_prefix}middle_block.0.in_layers.0.weight" in state_dict_keys:
            transformer_depth_middle = -1
        else:
            transformer_depth_middle = -2
        unet_config["num_res_blocks"] = num_res_blocks
        unet_config["channel_mult"] = channel_mult
        unet_config["transformer_depth"] = transformer_depth
        unet_config["transformer_depth_output"] = transformer_depth_output
        unet_config["transformer_depth_middle"] = transformer_depth_middle
    return unet_config


def unet_to_diffusers_mapping(unet_config, state_dict=None, key_prefix="model.diffusion_model."):
    if "num_res_blocks" not in unet_config: return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    num_blocks = len(channel_mult)
    if state_dict is not None:
        import re
        state_dict_keys = list(state_dict.keys())
        filtered_keys = [k.replace(key_prefix, "") for k in state_dict_keys if k.startswith(key_prefix)]
        transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'input_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in transformer_counts: transformer_counts[block_idx] = 0
                transformer_counts[block_idx] = max(transformer_counts[block_idx], trans_idx + 1)
        output_transformer_counts = {}
        for key in filtered_keys:
            match = re.match(r'output_blocks\.(\d+)\.1\.transformer_blocks\.(\d+)', key)
            if match:
                block_idx = int(match.group(1))
                trans_idx = int(match.group(2))
                if block_idx not in output_transformer_counts: output_transformer_counts[block_idx] = 0
                output_transformer_counts[block_idx] = max(output_transformer_counts[block_idx], trans_idx + 1)
        middle_transformer_count = 0
        for key in filtered_keys:
            match = re.match(r'middle_block\.1\.transformer_blocks\.(\d+)', key)
            if match:
                trans_idx = int(match.group(1))
                middle_transformer_count = max(middle_transformer_count, trans_idx + 1)
        transformers_mid = middle_transformer_count if middle_transformer_count > 0 else unet_config.get("transformer_depth_middle", None)
    else:
        transformer_depth = unet_config["transformer_depth"][:]
        transformer_depth_output = unet_config["transformer_depth_output"][:]
        transformers_mid = unet_config.get("transformer_depth_middle", None)
        transformer_counts = None
        output_transformer_counts = None
    UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias", "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias", "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias", "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias", "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias", "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias"}
    UNET_MAP_ATTENTIONS = {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"}
    TRANSFORMER_BLOCKS = {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias", "attn1.to_q.weight", "attn1.to_q.bias", "attn1.to_k.weight", "attn1.to_k.bias", "attn1.to_v.weight", "attn1.to_v.bias", "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias"}
    UNET_MAP_BASIC = {("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"), ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"), ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"), ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"), ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"), ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"), ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias")}
    # Map only tensors present in this checkpoint's state_dict (auto from weights).
    # No invented Diffusers names, no fixed KEEP list, no inject.
    if state_dict is None:
        raise RuntimeError(
            "unet_to_diffusers_mapping requires state_dict; refuse maps without "
            "Comfy presence checks"
        )
    _sd_keys = set(state_dict.keys())
    _comfy_bare = {
        (k[len(key_prefix):] if k.startswith(key_prefix) else k)
        for k in _sd_keys
    }

    def _comfy_present(comfy_bare: str) -> bool:
        return comfy_bare in _comfy_bare or f"{key_prefix}{comfy_bare}" in _sd_keys

    def _map_put(diff_key: str, comfy_bare: str) -> None:
        if not _comfy_present(comfy_bare):
            return
        diffusers_unet_map[diff_key] = comfy_bare

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                _map_put(
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "input_blocks.{}.0.{}".format(n, b),
                )
            if transformer_counts is not None: num_transformers = transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth.pop(0) if transformer_depth else 0
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "input_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            n += 1
        # Last DownBlock has no downsampler in SD1.5  -  register only if op exists.
        if _comfy_present("input_blocks.{}.0.op.weight".format(n)):
            for k in ["weight", "bias"]:
                _map_put(
                    "down_blocks.{}.downsamplers.0.conv.{}".format(x, k),
                    "input_blocks.{}.0.op.{}".format(n, k),
                )
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        _map_put("mid_block.attentions.{}.{}".format(i, b), "middle_block.1.{}".format(b))
    if transformers_mid:
        for t in range(transformers_mid):
            for b in TRANSFORMER_BLOCKS:
                _map_put(
                    "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b),
                    "middle_block.1.transformer_blocks.{}.{}".format(t, b),
                )
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            _map_put(
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b]),
                "middle_block.{}.{}".format(n, b),
            )
    num_res_blocks_rev = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks_rev[x] + 1) * x
        l = num_res_blocks_rev[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                _map_put(
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b]),
                    "output_blocks.{}.0.{}".format(n, b),
                )
            c += 1
            if output_transformer_counts is not None: num_transformers = output_transformer_counts.get(n, 0)
            else: num_transformers = transformer_depth_output.pop() if transformer_depth_output else 0
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    _map_put(
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b),
                        "output_blocks.{}.1.{}".format(n, b),
                    )
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        _map_put(
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b),
                            "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b),
                        )
            # Upsample: only if this checkpoint has that Comfy conv (presence).
            # Missing tensor → no Diffusers entry.
            if i == l - 1:
                for k in ["weight", "bias"]:
                    _map_put(
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k),
                        "output_blocks.{}.{}.conv.{}".format(n, c, k),
                    )
            n += 1
    for k, v in UNET_MAP_BASIC:
        _map_put(v, k)
    for _dk, _ck in diffusers_unet_map.items():
        if not _comfy_present(_ck):
            raise RuntimeError(
                f"Map integrity FATAL: mapped Comfy key {_ck!r} absent in checkpoint"
            )
    comfyui_to_diffusers_map = {v: k for k, v in diffusers_unet_map.items()}
    comfyui_to_diffusers_map = {f"{key_prefix}{k}": v for k, v in comfyui_to_diffusers_map.items()}

    return comfyui_to_diffusers_map


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
