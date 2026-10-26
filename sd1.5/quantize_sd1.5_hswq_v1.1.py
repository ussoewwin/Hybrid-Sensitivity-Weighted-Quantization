# -*- coding: utf-8 -*-
"""SD1.5 UNet ConvRot INT8 pack builder driven by the measured trajectory impact
or aggregate contribution.

The protection set comes from `sd1.5/diagimpact_sd15.py` (Step 1 trajectory
sensitivity) or `sd1.5/measure_sd15_unet_contrib.py`. The top-K highest-impact
layers are kept in FP16; every other ConvRot-eligible Linear/Conv2d goes to INT8.

**CLIP is not quantized** (the caller's instruction): only the UNet subtree
(`model.diffusion_model.`) is touched. The checkpoint's CLIP passes through
byte-identical, and **the VAE is dropped** - the pack ships UNet + CLIP only.

Bias correction (optional, `--bias_correction`, default OFF - same default as
V1.0): folds the systematic INT8 output shift into the layer bias,

    E[(W_q - W) x]  ~  (W_q - W) @ E[x]

using per-input-channel activation means collected **in this same run** via
ComfyUI's native `dpmpp_2m / karras` sampling trajectory on the checkpoint's
own UNet and CLIP (a Diffusers fallback is NOT provided: the calibration must
run on the production sampler).

Rules:
  1. Only 2D Linear and 4D Conv2d weights under `model.diffusion_model.` are
     quantized. Conv2d rotates along the input-channel axis.
  2. Group size 256 (or `--groupsize`), falling back to 64 / 16 / 4 where the
     input width demands.
  3. Kept tensors are stored FP16 (the runtime runs FP16); the INT8
     `weight_scale` stays FP32 by design, with its rank matching the weight
     (`(O,1)` for 2D, `(O,1,1,1)` for 4D).

Usage:
  python sd1.5/quantize_sd1.5_hswq_v1.1.py \
      --input <checkpoint>.safetensors --output <pack>.safetensors \
      --contrib impact/impact_sd15_traj_<model>.json --top-k 80 \
      [--comfy <ComfyUI root> --bias_correction --calib_file sample/calibration_prompts_128.txt]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

# Strip ComfyUI's auto-appended backend:cudaMallocAsync before torch initializes
# CUDA so calibration runs on PyTorch's standard caching allocator.
_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if _alloc_conf and "backend:" in _alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        p for p in (s.strip() for s in _alloc_conf.split(","))
        if p and not p.startswith("backend:")
    )

import torch
from safetensors.torch import load_file, save_file

GROUP_SIZE = 256
UNET_PREFIX = "model.diffusion_model."
_DEFAULT_PROMPT = "masterpiece, best quality, 1girl, solo, standing, simple background"


def _norm_unet_key(k: str) -> str:
    if k.startswith("model.diffusion_model."):
        k = k[len("model.diffusion_model."):]
    elif k.startswith("diffusion_model."):
        k = k[len("diffusion_model."):]
    if k.endswith(".weight"):
        k = k[:-len(".weight")]
    return k


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
    q = (w.float() / sv).round().clamp(-127, 127).to(torch.int8)
    return q, sv.to(torch.float32)


def bias_delta(weight_fp, weight_dq, act_mean):
    """The systematic output shift of an INT8 layer, to cancel into the bias.

        E[(W_q - W) x]  ~  (W_q - W) @ E[x]

    Linear (O, I)       -> delta[o] = sum_i err[o,i] * mu[i]
    Conv2d (O, I, K, K) -> delta[o] = sum_{i,kh,kw} err[o,i,kh,kw] * mu[i]
    """
    if act_mean is None:
        return None
    err = (weight_dq.float() - weight_fp.float())
    mu = act_mean.float().to(err.device)
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
# ComfyUI-native calibration pass (the only path; no Diffusers fallback)
# --------------------------------------------------------------------------

def _install_torchaudio_stub() -> None:
    import importlib.machinery

    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_torchaudio_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    ta = _stub_mod("torchaudio", is_package=True)
    functional = _stub_mod("torchaudio.functional")
    functional.resample = lambda waveform, orig_freq, new_freq, *a, **k: waveform
    transforms = _stub_mod("torchaudio.transforms")

    class _MelSpectrogram:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _MelScale:
        def __init__(self, *args, **kwargs):
            pass

    transforms.MelSpectrogram = _MelSpectrogram
    transforms.MelScale = _MelScale
    ta.functional = functional
    ta.transforms = transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms


def _install_comfy_aimdo_stub() -> None:
    import importlib.machinery

    for key in list(sys.modules):
        if key == "comfy_aimdo" or key.startswith("comfy_aimdo."):
            del sys.modules[key]

    def _stub_mod(name: str, *, is_package: bool = False):
        mod = types.ModuleType(name)
        mod.__file__ = "<hswq_comfy_aimdo_stub>"
        if is_package:
            mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        mod.__spec__ = spec
        return mod

    root = _stub_mod("comfy_aimdo", is_package=True)
    control = _stub_mod("comfy_aimdo.control")
    control.init = lambda *a, **k: False
    control.init_device = lambda *a, **k: False
    control.analyze = lambda *a, **k: None
    control.set_log_debug = lambda *a, **k: None
    control.set_log_critical = lambda *a, **k: None
    control.set_log_error = lambda *a, **k: None
    control.set_log_warning = lambda *a, **k: None
    control.set_log_info = lambda *a, **k: None
    control.get_total_vram = lambda *a, **k: 0

    model_vbar = _stub_mod("comfy_aimdo.model_vbar")

    class _ModelVBAR:
        def __init__(self, *a, **k):
            raise RuntimeError("comfy_aimdo stub: VBAR disabled for calibration")

    model_vbar.ModelVBAR = _ModelVBAR
    model_vbar.vbars_analyze = lambda *a, **k: 0
    model_vbar.vbar_fault = lambda *a, **k: None
    model_vbar.vbar_unpin = lambda *a, **k: None
    model_vbar.vbar_signature_compare = lambda *a, **k: False

    torch_mod = _stub_mod("comfy_aimdo.torch")
    torch_mod.aimdo_to_tensor = lambda *a, **k: None
    torch_mod.hostbuf_to_tensor = lambda *a, **k: None

    hostbuf = _stub_mod("comfy_aimdo.hostbuf")
    hostbuf.hostbuf_allocate = lambda *a, **k: None
    hostbuf.hostbuf_Optional = object

    root.control = control
    root.model_vbar = model_vbar
    root.torch = torch_mod
    root.hostbuf = hostbuf
    sys.modules["comfy_aimdo"] = root
    sys.modules["comfy_aimdo.control"] = control
    sys.modules["comfy_aimdo.model_vbar"] = model_vbar
    sys.modules["comfy_aimdo.torch"] = torch_mod
    sys.modules["comfy_aimdo.hostbuf"] = hostbuf


def _resolve_comfy_root(comfy_path: str) -> str:
    candidates = []
    if comfy_path:
        candidates.append(comfy_path)
    env_p = os.environ.get("COMFY_PATH", "")
    if env_p:
        candidates.append(env_p)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "ComfyUI-master"))
    candidates.append(r"D:\USERFILES\ComfyUI\ComfyUI")
    for c in candidates:
        root = os.path.abspath(c)
        if os.path.isdir(os.path.join(root, "comfy")):
            return root
    return ""


def collect_act_means_comfy(
    ckpt_path: str,
    calib_file: str,
    comfy_root: str,
    n_samples: int = 32,
    n_steps: int = 25,
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    cfg: float = 7.0,
    sampler: str = "dpmpp_2m",
    scheduler: str = "karras",
) -> dict:
    """Collect per-input-channel activation means using ComfyUI's native UNet + sampler."""
    sys.path = [comfy_root] + [p for p in sys.path if os.path.abspath(p or ".") != comfy_root]
    _install_torchaudio_stub()
    _install_comfy_aimdo_stub()
    try:
        import psutil  # noqa: F401
    except ImportError:
        _ps = types.ModuleType("psutil")
        _ps.virtual_memory = lambda: types.SimpleNamespace(total=64 * 1024 ** 3, available=32 * 1024 ** 3)
        sys.modules["psutil"] = _ps

    import comfy.options
    comfy.options.args_parsing = False
    import comfy.model_management as mm
    import comfy.sample as comfy_sample
    import comfy.sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0]] if sys.argv else []
    try:
        patcher, clip, _vae = comfy.sd.load_checkpoint_guess_config(
            os.path.abspath(ckpt_path),
            output_vae=False,
            output_clip=True,
            embedding_directory=None,
        )[:3]
    finally:
        sys.argv = saved_argv

    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        prompts = [_DEFAULT_PROMPT]
    if len(prompts) < n_samples:
        prompts = (prompts * (n_samples // len(prompts) + 1))[:n_samples]
    else:
        prompts = prompts[:n_samples]

    positives = [clip.encode_from_tokens_scheduled(clip.tokenize(p)) for p in prompts]
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    if dev == "cuda":
        torch.cuda.empty_cache()

    net = patcher.model
    net.to(dev)
    net.eval()
    unet = net.diffusion_model

    acc, cnt, handles = {}, {}, []

    def hook_fn(module, inputs, output, name=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        if not torch.is_tensor(x) or not torch.is_floating_point(x):
            return
        xd = x.detach().float()
        if xd.dim() == 4:
            ch = xd.mean(dim=(0, 2, 3)).cpu()
        elif xd.dim() == 3:
            ch = xd.mean(dim=(0, 1)).cpu()
        elif xd.dim() == 2:
            ch = xd.mean(dim=0).cpu()
        else:
            return
        if name not in acc:
            acc[name] = ch
            cnt[name] = 1
        else:
            acc[name] += ch
            cnt[name] += 1

    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(
                module.register_forward_hook(
                    lambda mm_, ii, oo, n=name: hook_fn(mm_, ii, oo, n)
                )
            )

    latent = torch.zeros(
        [1, 4, height // 8, width // 8],
        device=mm.intermediate_device(),
        dtype=mm.intermediate_dtype(),
    )
    latent = comfy_sample.fix_empty_latent_channels(patcher, latent, 8, None)
    if latent.device.type != dev:
        latent = latent.to(dev)

    print(
        f"    the calibration (ComfyUI {sampler}/{scheduler} cfg={cfg}): "
        f"{len(prompts)} samples x {n_steps} steps (seed {seed})",
        flush=True,
    )
    for i, pos in enumerate(positives):
        noise = comfy_sample.prepare_noise(latent, seed + i, None)
        with torch.no_grad():
            comfy_sample.sample(
                patcher, noise, n_steps, cfg, sampler, scheduler,
                pos, negative, latent, denoise=1.0,
                disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=False, noise_mask=None,
                callback=None, disable_pbar=True, seed=seed + i,
            )
        if (i + 1) % 8 == 0 or (i + 1) == len(positives):
            print("      [%d/%d]" % (i + 1, len(positives)), flush=True)

    for h in handles:
        h.remove()

    out_means = {
        name: (acc[name] / cnt[name])
        for name in acc
        if cnt.get(name, 0) > 0
    }
    del unet, net, patcher, positives, negative
    if dev == "cuda":
        torch.cuda.empty_cache()

    print("    the activation means: %d layers" % len(out_means), flush=True)
    return out_means


# --------------------------------------------------------------------------
# the key-mapping helpers, INLINED (used only for Diffusers fallback)
# --------------------------------------------------------------------------
def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        return last_transformer_depth
    return 0


def detect_unet_config_from_keys(state_dict, key_prefix="model.diffusion_model."):
    state_dict_keys = list(state_dict.keys())
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
                      comfy_path="", width=512, height=512):
    """Collect per-input-channel activation means.

    ComfyUI-native trajectory sampling (the checkpoint's own CLIP + dpmpp_2m
    karras CFG=7.0 on ComfyUI's native UNet module names). Requires a ComfyUI
    root; there is no Diffusers fallback.
    """
    comfy_root = _resolve_comfy_root(comfy_path)
    if not comfy_root:
        raise SystemExit(
            "error: no ComfyUI root found for bias correction. "
            "pass --comfy <ComfyUI root> or set COMFY_PATH. "
            "(the Diffusers fallback was removed: the calibration must use the "
            "checkpoint's own CLIP and ComfyUI's native dpmpp_2m/karras sampler.)"
        )
    return collect_act_means_comfy(
        ckpt_path,
        calib_file,
        comfy_root,
        n_samples=n_samples,
        n_steps=n_steps,
        seed=seed,
        width=width,
        height=height,
    )


def convert(input_path, output_path, contrib_json, top_k,
            bias_correction=False, calib_file="", n_calib=32, n_steps=25,
            comfy_path="", preferred_groupsize=GROUP_SIZE, protect_list="",
            calib_seed=42, width=512, height=512):
    src = load_file(input_path)

    with open(contrib_json, encoding="utf-8") as f:
        raw_impacts = json.load(f)["impacts"]
    contrib = {}
    for k, v in raw_impacts.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        contrib[_norm_unet_key(k)] = float(v)

    ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
    protect = {n for n, _ in ranked[:int(top_k)]}
    if protect_list:
        with open(protect_list, encoding="utf-8") as f:
            extra = json.load(f)
        for k in extra:
            protect.add(_norm_unet_key(k))
    print("the measured candidates: %d | the adopted protection: %d" % (len(ranked), len(protect)))

    act_means = {}
    if bias_correction:
        if not calib_file:
            sys.exit("error: --bias_correction needs --calib_file <prompts.txt>")
        act_means = collect_act_means(
            input_path, calib_file,
            n_samples=n_calib, n_steps=n_steps,
            seed=calib_seed, comfy_path=comfy_path,
            width=width, height=height,
        )

    h = build_hadamard(preferred_groupsize)
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
        gs = group_size_for(inf, preferred=preferred_groupsize)
        if gs is None or int(t.shape[0]) % 4 != 0:
            out[name] = t.to(torch.float16)
            n_prot += 1
            continue
        hh = h if gs == preferred_groupsize else build_hadamard(gs)
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
        if n_bias == 0:
            raise RuntimeError("bias correction was requested but wrote 0 layers - refusing to "
                               "save a pack labelled as bias-corrected")
    print("  the output: %.1f MB" % (os.path.getsize(output_path) / 1024 ** 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--top-k", type=int, default=80)
    ap.add_argument("--groupsize", type=int, default=GROUP_SIZE,
                    help="preferred ConvRot Hadamard group size (power of 4, default 256)")
    ap.add_argument("--protect_list", default="",
                    help="optional JSON list of extra UNet layer names to keep FP16")
    ap.add_argument("--comfy", "--comfy_path", dest="comfy",
                    default=os.environ.get("COMFY_PATH", "ComfyUI-master"),
                    help="ComfyUI root directory (used for ComfyUI-native --bias_correction calibration)")
    ap.add_argument("--calib_file", default="",
                    help="calibration prompts, one per line (required with "
                         "--bias_correction; the repo ships "
                         "sample/calibration_prompts_128.txt)")
    ap.add_argument("--bias_correction", action="store_true",
                    help="cancel the systematic INT8 output bias into the layer "
                         "bias (default OFF, same default as V1.0)")
    ap.add_argument("--num_calib_samples", type=int, default=32,
                    help="calibration samples (default: 32)")
    ap.add_argument("--num_inference_steps", type=int, default=25,
                    help="denoising steps per calibration sample (default: 25)")
    ap.add_argument("--calib_seed", type=int, default=42,
                    help="starting seed for calibration samples (default: 42)")
    ap.add_argument("--width", type=int, default=512,
                    help="calibration latent width in pixels (default: 512)")
    ap.add_argument("--height", type=int, default=512,
                    help="calibration latent height in pixels (default: 512)")
    a = ap.parse_args()
    convert(
        a.input, a.output, a.contrib, a.top_k,
        bias_correction=a.bias_correction, calib_file=a.calib_file,
        n_calib=a.num_calib_samples, n_steps=a.num_inference_steps,
        comfy_path=a.comfy, preferred_groupsize=a.groupsize,
        protect_list=a.protect_list, calib_seed=a.calib_seed,
        width=a.width, height=a.height,
    )
