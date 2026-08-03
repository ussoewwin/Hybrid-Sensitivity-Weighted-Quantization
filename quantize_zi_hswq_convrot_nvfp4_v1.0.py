"""Z-Image / ZIT NVFP4 + HSWQ ConvRot INT8 protect (quantize entry).

Pack / ConvRot / save path is inlined from the hardcode convert (copy, not import).

**HSWQ complete contract (this file):**
Protect key selection — per-model autonomous from ALL four directions (no hand-cut):
  1) Analyze THIS (weight distribution / character table / autonomous tunables)
  2) DualMonitor calib (sensitivity + channel Imp — every measured tensor)
  3) Histogram V4 pack-MSE @ absmax (weighted histogram calibration)
  4) SVD leverage inside V4 + gray RELEASE of hard-veto fences
Then one joint ranking / priority combinator → top N keys (default 60).
Optimal settings (alpha, fences, severity, protect set) are derived per model
from that joint judgment — never fence-only, Imp-drop, or SVD-only demote.

**Completeness rules (binding):**
- DualMonitor sens: unknown layers get median-of-measured (no 0.0 demotion,
  no drop from ranking).
- DualMonitor Imp: missing → channel-neutral ones; every layer runs V4/SVD.
- ConvRot bias: --bias_correction + ConvRot ON is rejected (pre/post rotation
  mixing is a blasphemy shortcut).
- --protect-keys-hardcode is reference-only bypass; it is NOT the HSWQ path.

Protect path:
  ConvRot rotate (W @ H^T) → row-wise INT8 + weight_scale + int8_tensorwise stamp
  in _quantization_metadata AND per-layer ``.comfy_quant`` (uint8 JSON; ComfyUI load).

Remaining Linear 2D: NVFP4 (+ FULL ConvRot by default) + same ``.comfy_quant``.
Kitchen Turbo blacklist: bfloat16 (unchanged).

Calib recipe (HSWQ How-to): 32 samples / 25 steps DualMonitor (sens + Imp).
Optional ``--protect-keys-json`` skips auto ranking. Optional Card1 bias.

Post-convert bench (default ON): after save, subprocess
  benchmark/zi_convrot_nvfp4_bench.py. Pass --no-bench to skip.

Example:
  D:\\USERFILES\\fp8e4m3\\venv\\Scripts\\python.exe \\
    quantize_zi_hswq_convrot_nvfp4_v1.0.py \\
    --model ... --output ... --device cuda \\
    --clip_path ... --comfy_path ... --calib_file ...
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("Error: comfy_kitchen not found (install in the active venv).")
    sys.exit(1)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# ConvRot / INT8 helpers (inlined — do NOT import native_convert_int8)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), same as comfy_kitchen ConvRot."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    h_matrix = h4
    current_size = 4
    while current_size < size:
        h_matrix = torch.kron(h_matrix, h4)
        current_size *= 4
    h_matrix = h_matrix / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_matrix
    return h_matrix


def convrot_group_size_for_features(n: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group size <= preferred that divides n (or None)."""
    if n < 4:
        return None
    gs = preferred
    while gs >= 4:
        if n % gs == 0 and math.log(gs, 4) % 1 == 0:
            return gs
        gs //= 4
    return None


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
    """Offline Linear: W_rot = W @ H^T (group-wise). Matches comfy_kitchen._rotate_weight."""
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}"
        )
    group_count = in_features // group_size
    weight_grouped = weight.view(out_features, group_count, group_size)
    return torch.matmul(
        weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)
    ).reshape(weight.shape)


def quantize_int8_tensorwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorwise INT8: scalar weight_scale (plain ComfyUI int8_tensorwise)."""
    amax = max(w.abs().max().item(), 1e-6)
    scale = torch.tensor(amax / 127.0, dtype=torch.float32)
    q = (w / scale.item()).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_int8_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel INT8 for Linear: weight_scale [out, 1]."""
    abs_max = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-30)
    scale = abs_max / 127.0
    q = (w / scale.to(dtype=w.dtype)).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(dtype=torch.float32)


def _encode_comfy_quant(config: dict) -> torch.Tensor:
    """ComfyUI layer marker: uint8 JSON (same as convert_old_quants / int8_sdxl)."""
    return torch.tensor(
        list(json.dumps(config, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )


# ---------------------------------------------------------------------------
# DualMonitor (Card 1) — signed per-in-channel E[x] for bias fold
# ---------------------------------------------------------------------------
class DualMonitor:
    """Per-layer act moments for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None

    def update(self, input_tensor, output_tensor, module=None):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp = input_tensor.detach().float()
            is_conv2d = isinstance(module, torch.nn.Conv2d)
            if is_conv2d and inp.dim() == 4:
                reduce_dims = (0, 2, 3)
            elif inp.dim() >= 2:
                reduce_dims = tuple(range(inp.dim() - 1))
            else:
                current_imp = torch.ones(1, device=inp.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp.device, dtype=torch.float32)
                reduce_dims = None
            if reduce_dims is not None:
                current_imp = inp.abs().mean(dim=reduce_dims)
                current_act = inp.mean(dim=reduce_dims)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
            elif current_imp.shape == self.channel_importance.shape:
                c = self.count
                self.channel_importance = (
                    self.channel_importance * c + current_imp
                ) / (c + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * c + current_act
                ) / (c + 1)
            self.count += 1

    def get_sensitivity(self) -> float:
        """Output variance (HSWQ DualMonitor pillar)."""
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        return variance if math.isfinite(variance) else 0.0


_dual_monitors: dict[str, DualMonitor] = {}


def _hook_fn(module, input, output, name):
    if name not in _dual_monitors:
        _dual_monitors[name] = DualMonitor()
    _dual_monitors[name].update(input[0], output, module)


def compute_nvfp4_bias_delta(weight_fp, weight_dq, act_mean):
    """Card 1: delta ≈ (W_q - W) contracted with per-in-channel E[x]."""
    if act_mean is None:
        return None
    err = weight_dq.float() - weight_fp.float()
    mu = act_mean.float().to(device=err.device)
    if err.ndim == 2:
        if mu.numel() != err.shape[1]:
            return None
        return err @ mu
    if err.ndim == 4:
        if mu.numel() != err.shape[1]:
            return None
        return (err * mu.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    return None


def _dequant_nvfp4(qdata: torch.Tensor, params) -> torch.Tensor:
    full = TensorCoreNVFP4Layout.dequantize(qdata, params)
    orig = tuple(params.orig_shape)
    if tuple(full.shape) != orig:
        return full[tuple(slice(0, s) for s in orig)]
    return full


def run_card1_calib(
    model_path: str,
    clip_path: str,
    comfy_path: str,
    calib_file: str,
    num_samples: int = 32,
    num_inference_steps: int = 25,
    tokenizer_path: str | None = None,
    seed: int = 42,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, float],
    dict[str, torch.Tensor],
]:
    """Unquantized NextDiT calib with DualMonitor on Linear.

    Returns ``(act_mean, sens, channel_importance)`` keyed by module name.
    Module name keys match ``_meta_base_key`` (e.g. ``layers.0.attention.to_q``).
    TE path mirrors ``benchmark/zi_convrot_nvfp4_bench.py`` (Qwen3_4B + Qwen2Tokenizer).
    """
    global _dual_monitors
    _dual_monitors = {}

    from benchmark.zi_convrot_nvfp4_bench import (
        encode_prompt,
        load_zit_model,
        resolve_path,
        resolve_tokenizer_offline,
        run_inference,
        setup_comfy,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Card1] Loading unquantized ZI for DualMonitor from {model_path}")
    setup_comfy(comfy_path)

    # qk_norm Lumina calls ck.rms_rope; older kitchen wheels lack it
    # (same as benchmark/zi_convrot_nvfp4_bench.py after setup_comfy).
    from kitchen_rms_rope_fallback import ensure_kitchen_rms_rope

    ensure_kitchen_rms_rope()

    from comfy.text_encoders import llama as llama_module
    from transformers import Qwen2Tokenizer
    import comfy.ops

    tok_resolved = resolve_tokenizer_offline(tokenizer_path, comfy_path)
    if tok_resolved:
        print(f"[Card1] Tokenizer (disk): {tok_resolved}")
        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                tok_resolved, local_files_only=True
            )
        except Exception as e:
            print(f"[Card1] local_files_only failed ({e}); retrying...")
            tokenizer = Qwen2Tokenizer.from_pretrained(tok_resolved)
    else:
        mid = tokenizer_path if tokenizer_path else "Qwen/Qwen2.5-7B-Instruct"
        print(f"[Card1] Tokenizer repo id: {mid}")
        tokenizer = Qwen2Tokenizer.from_pretrained(mid, local_files_only=True)

    resolved_clip = resolve_path(clip_path, is_file=True)
    text_encoder = llama_module.Qwen3_4B(
        config_dict={},
        device=device,
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    ).to(device)
    print(f"[Card1] Loading CLIP weights from: {resolved_clip}")
    text_encoder.load_state_dict(load_file(resolved_clip), strict=False)
    text_encoder.eval()

    model, _n_cq, _is_za = load_zit_model(
        model_path, device, comfy_path, is_nvfp4=False
    )
    model.eval()

    hooks = []
    n_lin = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    lambda m, i, o, n=name: _hook_fn(m, i, o, n)
                )
            )
            n_lin += 1
    print(f"[Card1] DualMonitor hooks on {n_lin} Linear modules")

    with open(calib_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    prompts = prompts[: max(1, int(num_samples))]
    print(
        f"[Card1] Calibrating {len(prompts)} prompts, "
        f"{num_inference_steps} steps each..."
    )

    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Card1 calib")):
            cond, mask = encode_prompt(prompt, text_encoder, tokenizer, device)
            # ZI Turbo: same signature as benchmark/zi_convrot_nvfp4_bench.run_inference
            # (cond-only; no CFG guidance / uncond kwargs).
            run_inference(
                model,
                cond,
                mask,
                int(num_inference_steps),
                int(seed) + i,
                device,
            )
            del cond, mask
            if device == "cuda":
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    act_mean_dict: dict[str, torch.Tensor] = {}
    sens_dict: dict[str, float] = {}
    importance_dict: dict[str, torch.Tensor] = {}
    for name, mon in _dual_monitors.items():
        meta_n = _normalize_module_meta(name)
        if mon.channel_act_mean is not None:
            act_t = mon.channel_act_mean.detach().cpu().float()
            act_mean_dict[name] = act_t
            # Also index by ST-facing meta so bias / Imp land without drop.
            if meta_n not in act_mean_dict:
                act_mean_dict[meta_n] = act_t
        s = float(mon.get_sensitivity())
        if s > 0.0 and math.isfinite(s):
            sens_dict[name] = s
            prev_s = sens_dict.get(meta_n)
            sens_dict[meta_n] = s if prev_s is None else max(float(prev_s), s)
        if mon.channel_importance is not None:
            imp_t = mon.channel_importance.detach().cpu().float()
            importance_dict[name] = imp_t
            if meta_n not in importance_dict:
                importance_dict[meta_n] = imp_t
    print(
        f"[Card1] Collected act_mean={len(act_mean_dict)} "
        f"sens={len(sens_dict)} importance={len(importance_dict)}"
    )

    del model, text_encoder, tokenizer
    _dual_monitors = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return act_mean_dict, sens_dict, importance_dict


_DEFAULT_GROUPSIZE = 256

# Kitchen model_type → (BLACKLIST, FP8_LAYERS) — Z-Image only
# (same strings as convert_to_nvfp4_node.py)
_Z_IMAGE_PROFILES: dict[str, tuple[list[str], list[str]]] = {
    "Z-Image-Turbo": (
        [
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
            "final_layer",
        ],
        [],
    ),
    "Z-Image-Base": (
        [
            "attention",
            "adaLN_modulation",
            "norm",
            "final_layer",
            "cap_embedder",
            "x_embedder",
            "noise_refiner",
            "context_refiner",
            "t_embedder",
        ],
        [],
    ),
}

_DEFAULT_MODEL_TYPE = "Z-Image-Turbo"

# Owner-allowed hardcode: moodyRealMix_zitV7 protect N=60 (2026-08-02).
# Source: auto60 + drop 3 kurt-only adaLN + add 3 NVFP4-outside abs top.
# Keys JSON: test/_moodyRealMix_zitV7_protect60_swap3_keys.json
# N=60 fixed. No N raise.
_INT8_PROTECT_SOURCE = (
    "moodyRealMix_zitV7_nvfp4_int8protect60_swap3_kurtAdaLN_to_nvfp4Abs"
)
_INT8_PROTECT_KEYSET: frozenset[str] = frozenset(
    (
        "model.diffusion_model.layers.6.feed_forward.w2.weight",
        "model.diffusion_model.layers.4.feed_forward.w2.weight",
        "model.diffusion_model.layers.11.feed_forward.w2.weight",
        "model.diffusion_model.layers.7.feed_forward.w2.weight",
        "model.diffusion_model.layers.13.feed_forward.w2.weight",
        "model.diffusion_model.layers.12.feed_forward.w2.weight",
        "model.diffusion_model.layers.9.feed_forward.w2.weight",
        "model.diffusion_model.layers.10.feed_forward.w2.weight",
        "model.diffusion_model.layers.5.feed_forward.w2.weight",
        "model.diffusion_model.layers.14.feed_forward.w2.weight",
        "model.diffusion_model.layers.15.feed_forward.w2.weight",
        "model.diffusion_model.layers.18.feed_forward.w2.weight",
        "model.diffusion_model.layers.1.feed_forward.w2.weight",
        "model.diffusion_model.layers.28.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.8.feed_forward.w2.weight",
        "model.diffusion_model.layers.19.feed_forward.w2.weight",
        "model.diffusion_model.layers.3.feed_forward.w2.weight",
        "model.diffusion_model.layers.2.feed_forward.w2.weight",
        "model.diffusion_model.layers.24.feed_forward.w1.weight",
        "model.diffusion_model.layers.29.attention.qkv.weight",
        "model.diffusion_model.layers.16.feed_forward.w1.weight",
        "model.diffusion_model.layers.20.feed_forward.w2.weight",
        "model.diffusion_model.layers.16.feed_forward.w2.weight",
        "model.diffusion_model.layers.0.feed_forward.w2.weight",
        "model.diffusion_model.layers.17.feed_forward.w1.weight",
        "model.diffusion_model.layers.13.attention.out.weight",
        "model.diffusion_model.layers.19.feed_forward.w3.weight",
        "model.diffusion_model.layers.29.adaLN_modulation.0.weight",
        "model.diffusion_model.layers.23.feed_forward.w1.weight",
        "model.diffusion_model.layers.23.feed_forward.w2.weight",
        "model.diffusion_model.layers.25.feed_forward.w2.weight",
        "model.diffusion_model.layers.26.feed_forward.w3.weight",
        "model.diffusion_model.layers.19.feed_forward.w1.weight",
        "model.diffusion_model.layers.28.feed_forward.w3.weight",
        "model.diffusion_model.layers.17.feed_forward.w2.weight",
        "model.diffusion_model.layers.22.feed_forward.w1.weight",
        "model.diffusion_model.layers.22.feed_forward.w2.weight",
        "model.diffusion_model.layers.21.feed_forward.w1.weight",
        "model.diffusion_model.layers.18.feed_forward.w1.weight",
        "model.diffusion_model.layers.28.attention.qkv.weight",
        "model.diffusion_model.layers.11.attention.out.weight",
        "model.diffusion_model.layers.10.attention.qkv.weight",
        "model.diffusion_model.layers.13.feed_forward.w3.weight",
        "model.diffusion_model.layers.27.attention.qkv.weight",
        "model.diffusion_model.layers.12.attention.out.weight",
        "model.diffusion_model.layers.9.attention.qkv.weight",
        "model.diffusion_model.layers.16.attention.qkv.weight",
        "model.diffusion_model.layers.14.attention.out.weight",
        "model.diffusion_model.layers.28.feed_forward.w2.weight",
        "model.diffusion_model.layers.9.attention.out.weight",
        "model.diffusion_model.layers.3.feed_forward.w3.weight",
        "model.diffusion_model.layers.25.feed_forward.w1.weight",
        "model.diffusion_model.layers.24.feed_forward.w3.weight",
        "model.diffusion_model.layers.11.attention.qkv.weight",
        "model.diffusion_model.layers.26.feed_forward.w1.weight",
        "model.diffusion_model.layers.8.attention.qkv.weight",
        "model.diffusion_model.layers.24.feed_forward.w2.weight",
        "model.diffusion_model.layers.25.feed_forward.w3.weight",
        "model.diffusion_model.layers.19.attention.qkv.weight",
        "model.diffusion_model.layers.21.feed_forward.w2.weight",
    )
)


def _resolve_int8_protect_keyset(
    int8_protect_keys: frozenset[str] | list[str] | None,
    int8_protect_source: str | None,
) -> tuple[frozenset[str], str]:
    """Require an explicit protect keyset (HSWQ / JSON / hardcode flag).

    Silent fallback to baked ``_INT8_PROTECT_KEYSET`` is forbidden on the
    auto path — pass keys from ``select_int8_protect_keys_hswq`` or CLI.
    """
    if int8_protect_keys is None:
        raise ValueError(
            "int8_protect_keys is required (HSWQ select, --protect-keys-json, "
            "or --protect-keys-hardcode). Silent hardcode default is forbidden."
        )
    keyset = frozenset(int8_protect_keys)
    if not keyset:
        raise ValueError("int8_protect_keys is empty")
    source = int8_protect_source or "injected_keyset"
    return keyset, source


def _st_weight_to_meta(st_key: str) -> str:
    """ST ``...weight`` → DualMonitor module meta (no ``.weight`` suffix)."""
    base = st_key[: -len(".weight")] if st_key.endswith(".weight") else st_key
    return _meta_base_key(base)


def _normalize_module_meta(name: str) -> str:
    """Strip diffusion prefixes so Card1 module names match ST meta."""
    n = str(name or "")
    if n.endswith(".weight"):
        n = n[: -len(".weight")]
    if "model.diffusion_model." in n:
        return n.split("model.diffusion_model.")[-1]
    if "diffusion_model." in n:
        return n.split("diffusion_model.")[-1]
    return n


def _fuse_attention_qkv_aliases(
    values_by_meta: dict[str, float],
) -> dict[str, float]:
    """Bridge fused ``.attention.qkv`` ↔ split ``to_q/to_k/to_v`` DualMonitor keys.

    Comfy NextDiT loads often expose fused ``qkv`` Linear while some ST / hook
    layouts expose split projections. Missing this bridge zeroes DualMonitor
    on attention → analyze/V4 dominate alone (pillar incompleteness).
    """
    out = dict(values_by_meta)
    attn_prefixes: set[str] = set()
    for meta in list(out.keys()):
        if meta.endswith(".attention.to_q"):
            attn_prefixes.add(meta[: -len(".to_q")])
        elif meta.endswith(".attention.to_k"):
            attn_prefixes.add(meta[: -len(".to_k")])
        elif meta.endswith(".attention.to_v"):
            attn_prefixes.add(meta[: -len(".to_v")])
        elif meta.endswith(".attention.qkv"):
            attn_prefixes.add(meta[: -len(".qkv")])
    for attn in attn_prefixes:
        qkv = f"{attn}.qkv"
        split_vals = [
            float(out[k])
            for k in (f"{attn}.to_q", f"{attn}.to_k", f"{attn}.to_v")
            if k in out and float(out[k]) > 0.0
        ]
        if qkv not in out and split_vals:
            out[qkv] = max(split_vals)
        elif qkv in out and split_vals:
            out[qkv] = max(float(out[qkv]), max(split_vals))
        if qkv in out:
            qv = float(out[qkv])
            for suf in (".to_q", ".to_k", ".to_v"):
                sk = f"{attn}{suf}"
                if sk not in out and qv > 0.0:
                    out[sk] = qv
    return out


def _build_sens_by_meta(
    sens_dict: dict[str, float],
    meta_to_st: dict[str, str],
) -> dict[str, float]:
    """Map DualMonitor sens onto ST meta — use ALL measured sens values.

    Exact meta + qkv↔to_* fuse first; then suffix / leaf recovery so DualMonitor
    keys with different prefixes still land on the Linear-2D pool (no drop).
    """
    raw: dict[str, float] = {}
    for name, s in sens_dict.items():
        try:
            sf = float(s)
        except (TypeError, ValueError):
            continue
        if sf <= 0.0 or not math.isfinite(sf):
            continue
        meta = _normalize_module_meta(name)
        prev = raw.get(meta)
        raw[meta] = sf if prev is None else max(prev, sf)
    fused = _fuse_attention_qkv_aliases(raw)
    out: dict[str, float] = {}
    for m in meta_to_st:
        if m in fused:
            out[m] = float(fused[m])
            continue
        best: float | None = None
        leaf = m.split(".")[-1]
        for fk, fv in fused.items():
            if fk == m:
                best = float(fv)
                break
            if fk.endswith("." + m) or m.endswith("." + fk):
                best = float(fv) if best is None else max(best, float(fv))
                continue
            if fk.split(".")[-1] == leaf and (
                fk.endswith(".attention." + leaf) or m.endswith(".attention." + leaf)
            ):
                # Same attention leaf under alternate block prefix — keep max.
                if ".attention." in fk and ".attention." in m:
                    best = float(fv) if best is None else max(best, float(fv))
        if best is not None and best > 0.0:
            out[m] = best
    return out


def _lookup_importance_tensor(
    key: str,
    importance_dict: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Resolve DualMonitor Imp for one meta — exact, then suffix / attention leaf.

    Same recovery spirit as ``_build_sens_by_meta``: never drop a measured Imp
    tensor because the hook prefix differs from the ST meta string.
    """
    key_n = _normalize_module_meta(key)
    if key_n in importance_dict and importance_dict[key_n] is not None:
        return importance_dict[key_n]
    if key in importance_dict and importance_dict[key] is not None:
        return importance_dict[key]
    leaf = key_n.split(".")[-1]
    recovered: list[torch.Tensor] = []
    for raw_k, tensor in importance_dict.items():
        if tensor is None:
            continue
        rk = _normalize_module_meta(raw_k)
        if rk == key_n:
            return tensor
        if rk.endswith("." + key_n) or key_n.endswith("." + rk):
            recovered.append(tensor)
            continue
        if rk.split(".")[-1] != leaf:
            continue
        if ".attention." not in rk or ".attention." not in key_n:
            continue
        if not (
            rk.endswith(".attention." + leaf) and key_n.endswith(".attention." + leaf)
        ):
            continue
        rk_attn = rk[: -(len(leaf) + 1)]
        kn_attn = key_n[: -(len(leaf) + 1)]
        if rk_attn == kn_attn or rk_attn.endswith(kn_attn) or kn_attn.endswith(rk_attn):
            recovered.append(tensor)
    if not recovered:
        return None
    if len(recovered) == 1:
        return recovered[0]
    # Multiple DualMonitor Imp storages for one meta — use ALL (max after align).
    n0 = max(int(t.numel()) for t in recovered)
    aligned = [_align_importance_1d(t, n0) for t in recovered]
    return torch.stack(aligned, dim=0).max(dim=0).values


def _align_importance_1d(
    imp: torch.Tensor,
    in_features: int | None,
) -> torch.Tensor:
    """Use all DualMonitor Imp values; pad/truncate like V4 histogram path."""
    v = imp.detach().float().reshape(-1)
    if in_features is None or int(in_features) <= 0:
        return v
    n = int(in_features)
    if int(v.numel()) == n:
        return v
    # Concat(q|k|v) → three equal chunks: use elementwise max (all three).
    if int(v.numel()) == 3 * n and n > 0:
        chunks = v.view(3, n)
        return chunks.max(dim=0).values
    if int(v.numel()) > n:
        return v[:n]
    pad = torch.ones(n - int(v.numel()), dtype=v.dtype, device=v.device)
    return torch.cat([v, pad], dim=0)


def _max_align_parts(
    parts: list[torch.Tensor],
    in_features: int | None,
) -> torch.Tensor:
    """Elementwise-max across DualMonitor Imp parts after length align."""
    aligned = [_align_importance_1d(p, in_features) for p in parts]
    if in_features is None or int(in_features) <= 0:
        n0 = max(int(a.numel()) for a in aligned)
        aligned = [_align_importance_1d(a, n0) for a in aligned]
    stacked = torch.stack(aligned, dim=0)
    return stacked.max(dim=0).values


def _importance_for_meta(
    meta: str,
    importance_dict: dict[str, torch.Tensor],
    *,
    in_features: int | None = None,
) -> torch.Tensor | None:
    """Resolve DualMonitor channel-importance — use ALL related Imp tensors.

    qkv ↔ to_q/to_k/to_v aliases are fused by elementwise max after length
    align (pad/truncate / 3×concat split). Never drop available DualMonitor
    Imp because lengths differ; never SVD-only demote by returning None when
    any related Imp exists.
    """
    meta_n = _normalize_module_meta(meta)
    parts: list[torch.Tensor] = []
    direct = _lookup_importance_tensor(meta_n, importance_dict)
    if direct is not None:
        parts.append(direct)
    if meta_n.endswith(".attention.qkv"):
        attn = meta_n[: -len(".qkv")]
        for suf in (".to_q", ".to_k", ".to_v"):
            t = _lookup_importance_tensor(f"{attn}{suf}", importance_dict)
            if t is not None:
                parts.append(t)
    elif meta_n.endswith((".attention.to_q", ".attention.to_k", ".attention.to_v")):
        # Split ST name + fused DualMonitor Imp (or sibling splits).
        if meta_n.endswith(".to_q"):
            attn = meta_n[: -len(".to_q")]
        elif meta_n.endswith(".to_k"):
            attn = meta_n[: -len(".to_k")]
        else:
            attn = meta_n[: -len(".to_v")]
        for suf in (".qkv", ".to_q", ".to_k", ".to_v"):
            t = _lookup_importance_tensor(f"{attn}{suf}", importance_dict)
            if t is not None:
                parts.append(t)
    if not parts:
        return None
    # Deduplicate identical storage while keeping every distinct DualMonitor vector.
    uniq: list[torch.Tensor] = []
    seen: set[int] = set()
    for p in parts:
        pid = int(p.data_ptr()) if p.numel() > 0 else id(p)
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(p)
    if len(uniq) == 1:
        return _align_importance_1d(uniq[0], in_features)
    return _max_align_parts(uniq, in_features)


def select_int8_protect_keys_hswq(
    model_path: str,
    sens_dict: dict[str, float],
    importance_dict: dict[str, torch.Tensor],
    *,
    protect_n: int = 60,
    model_type: str = _DEFAULT_MODEL_TYPE,
    device: str = "cuda",
) -> tuple[frozenset[str], str]:
    """HSWQ four-pillar INT8 protect ranking → top ``protect_n`` ST keys.

    Per-model autonomous judgment from ALL directions (no hand-cut):
      1) Analyze THIS (``analyze_unet(run_v4=True)`` + character + tunables)
      2) DualMonitor sensitivity + channel Imp — every calib tensor used
      3) Histogram V4 pack-MSE @ absmax (weighted histogram calibration)
      4) SVD leverage + analyze×V4×DualMonitor gray RELEASE of fences
      → joint ranking / priority combinator → truncate by key count N

    Optimal settings are derived from this model’s own Analyze + DualMonitor
    + V4/SVD evidence — never hardcode protect, never Imp-drop, never
    raise-away on shape mismatch (align Imp to in_features instead).

    ``fp16_budget_mb`` is analyze-side threshold input only
    (``derive_nvfp4_autonomous_tunables``). This path does **not** select
    FP16 keep layers or fill a MiB budget — keys are INT8 protect targets.

    Ranking namespace is DualMonitor / ST meta (qkv ↔ to_* aliases resolved).
    Returned keys are full safetensors weight keys for convert.
    """
    if int(protect_n) <= 0:
        raise ValueError(f"protect_n must be > 0, got {protect_n!r}")
    if model_type not in _Z_IMAGE_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_Z_IMAGE_PROFILES)}"
        )
    if not sens_dict:
        raise ValueError(
            "DualMonitor sens_dict is empty — Card1 calib required for HSWQ"
        )

    from analyze.analyze_zi_convrot_nvfp4_distribution import (
        NVFP4_FP16_BUDGET_MB_HARD,
        _robust_iqr,
        _safe_percentile,
        analyze_unet,
        apply_fp16_infinite_priority_branches,
        apply_fp16_infinite_ranking_branches,
        build_nvfp4_analyze_character_table,
        derive_nvfp4_autonomous_tunables,
        derive_priority_combinator,
        measure_v4_nvfp4_mse_at_absmax,
        nvfp4_fp16_budget_analyze_severity,
        nvfp4_fp16_budget_priority,
    )
    from histogram.weighted_histogram_mse_v4_nvfp4 import (
        HSWQWeightedHistogramOptimizerV4,
    )

    blacklist, _fp8 = _Z_IMAGE_PROFILES[model_type]
    # run_v4=True: weight tensors enter enrich so V4 contract is recorded.
    # complete=True still requires DualMonitor Imp — gray release runs below.
    print("\n[HSWQ] Analyze THIS (weight distribution + V4 contract)...")
    profile = analyze_unet(model_path, run_v4=True)
    st_layers = profile.get("layers") or {}
    if not isinstance(st_layers, dict) or not st_layers:
        raise RuntimeError(f"[HSWQ] analyze_unet returned no layers for {model_path}")
    v4_stub = ((profile.get("optimal_settings_nvfp4") or {}).get("v4") or {})
    if not bool(v4_stub.get("svd_enabled", True)):
        raise RuntimeError(
            "[HSWQ] analyze V4 contract has svd_enabled=False — SVD cut forbidden"
        )
    print(
        f"[HSWQ] analyze V4 stub: v4_ran={v4_stub.get('v4_ran')} "
        f"complete={v4_stub.get('complete')} "
        f"reason={v4_stub.get('reason')!r} "
        f"(DualMonitor Imp gray release follows)"
    )

    print(f"[HSWQ] Loading weights for V4 pack-MSE from {model_path}")
    sd = load_file(model_path)

    # meta → ST weight key (prefer model.diffusion_model. prefix present in file)
    meta_to_st: dict[str, str] = {}
    meta_layers: dict[str, dict] = {}
    for st_key, entry in st_layers.items():
        if not isinstance(st_key, str) or not st_key.endswith(".weight"):
            continue
        if _is_non_diffusion_key(st_key):
            continue
        if any(marker in st_key for marker in blacklist):
            continue
        w = sd.get(st_key)
        if w is None or int(getattr(w, "ndim", 0)) != 2:
            continue
        meta = _st_weight_to_meta(st_key)
        # Prefer full prefixed ST key if several aliases exist
        prev = meta_to_st.get(meta)
        if prev is None or (
            st_key.startswith("model.diffusion_model.")
            and not prev.startswith("model.diffusion_model.")
        ):
            meta_to_st[meta] = st_key
        if isinstance(entry, dict):
            meta_layers[meta] = dict(entry)

    if not meta_to_st:
        raise RuntimeError(
            "[HSWQ] No Linear-2D UNet weight candidates after blacklist filter"
        )

    sens_by_meta = _build_sens_by_meta(sens_dict, meta_to_st)
    n_pool = len(meta_to_st)
    n_matched = len(sens_by_meta)
    match_ratio = float(n_matched) / float(n_pool) if n_pool else 0.0
    # DualMonitor must contribute to THIS pool. Expand aliases before failing.
    if n_matched < 1:
        raise RuntimeError(
            "[HSWQ] DualMonitor sens matched zero ST Linear-2D keys after "
            f"qkv/suffix recovery (pool={n_pool}). Calib names do not land "
            "on UNet Linear-2D — cannot form joint four-pillar judgment."
        )
    # Unknown sens → median of measured pool (no 0.0 demotion, no drop).
    # Every Linear-2D stays in ranking with a real axis value from measured data.
    _sens_measured = sorted(float(v) for v in sens_by_meta.values() if float(v) > 0)
    sens_unknown = (
        _sens_measured[len(_sens_measured) // 2] if _sens_measured else 0.0
    )
    for _m in meta_to_st:
        if _m not in sens_by_meta:
            sens_by_meta[_m] = float(sens_unknown)
    n_sens_unknown = n_pool - n_matched
    print(
        f"[HSWQ] DualMonitor sens matched={n_matched}/{n_pool} "
        f"Linear-2D (ratio={match_ratio:.3f}) unknown={n_sens_unknown} "
        f"→ unknown_sens={sens_unknown:.6g} (no drop)"
    )

    norm_profile = {"layers": meta_layers, "source": model_path}
    tunables = derive_nvfp4_autonomous_tunables(
        norm_profile,
        dualmonitor_sensitivities=sens_by_meta,
        fp16_budget_mb=float(NVFP4_FP16_BUDGET_MB_HARD),
    )
    if str(tunables.get("quant_format")) != "nvfp4":
        raise RuntimeError(
            "[HSWQ] derive_nvfp4_autonomous_tunables must set quant_format=nvfp4; "
            f"got {tunables.get('quant_format')!r}"
        )
    alpha = float(tunables.get("alpha_auto", 0.0) or 0.0)
    if alpha <= 0.0:
        raise ValueError(
            "[HSWQ] alpha_auto must be > 0 after DualMonitor resolve "
            f"(got {alpha!r})"
        )
    beta = 1.0 - alpha
    print(
        f"[HSWQ] alpha_auto={alpha:.6g} beta={beta:.6g} "
        f"| DualMonitor sens matched={n_matched}/{n_pool} "
        f"Linear-2D (ratio={match_ratio:.3f})"
    )

    ek = max(abs(float(tunables.get("extreme_kurtosis", 1e-6))), 1e-6)
    eo = max(float(tunables.get("extreme_outlier", 1e-6)), 1e-6)
    hm = max(float(tunables.get("huge_magnitude", 1e-6)), 1e-6)
    hard_veto: set[str] = set()
    for meta, entry in meta_layers.items():
        k = float(entry.get("kurtosis", 0) or 0)
        o = float(entry.get("outlier_ratio", 0) or 0)
        m = float(entry.get("abs_max", 0) or 0)
        if k >= ek or o >= eo or m >= hm:
            hard_veto.add(meta)
    print(
        f"[HSWQ] Hard VETO (fence) n={len(hard_veto)} "
        f"(ek={ek:.4g} eo={eo:.4g} hm={hm:.4g})"
    )

    # Analyze × DualMonitor × V4 × SVD: gray RELEASE of fence hard-veto.
    # Use ALL DualMonitor Imp (aligned to in_features) — never skip Imp rows.
    st_weights: dict[str, torch.Tensor] = {
        st_key: sd[st_key] for st_key in meta_to_st.values()
    }
    st_importance: dict[str, torch.Tensor] = {}
    n_imp_dual = 0
    n_imp_ones = 0
    for meta, st_key in meta_to_st.items():
        w = sd[st_key]
        in_f = int(w.shape[1]) if int(getattr(w, "ndim", 0)) == 2 else None
        imp = _importance_for_meta(meta, importance_dict, in_features=in_f)
        if imp is None:
            # No DualMonitor Imp for this meta — channel-neutral ones so V4/SVD
            # still runs with Analyze + sens; never drop the layer from pillars.
            if in_f is None or int(in_f) <= 0:
                continue
            imp_f = torch.ones(int(in_f), dtype=torch.float32)
            n_imp_ones += 1
        else:
            imp_f = imp.detach().float()
            n_imp_dual += 1
        st_importance[st_key] = imp_f
        base = st_key[: -len(".weight")] if st_key.endswith(".weight") else st_key
        st_importance[base] = imp_f
        st_importance[meta] = imp_f
    if not st_importance:
        raise RuntimeError(
            "[HSWQ] No Linear-2D in_features for DualMonitor/V4 Imp map — "
            "empty st_importance after pool build"
        )
    v4_device_release = (
        device if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(
        f"[HSWQ] Analyze×V4×DualMonitor×SVD gray release "
        f"(Imp DualMonitor={n_imp_dual}/{len(meta_to_st)} "
        f"ones_fallback={n_imp_ones} "
        f"keys={len(st_importance)} device={v4_device_release})..."
    )
    v4_release = measure_v4_nvfp4_mse_at_absmax(
        st_weights,
        device=v4_device_release,
        tunables=tunables,
        importance_by_layer=st_importance,
    )
    if not bool(v4_release.get("complete")):
        raise RuntimeError(
            "[HSWQ] Analyze×V4 gray release incomplete: "
            f"reason={v4_release.get('reason')!r} "
            f"safe_sample={v4_release.get('safe_sample_count')} "
            f"skipped_no_imp={v4_release.get('skipped_no_importance')}. "
            "Four-pillar joint judgment requires complete V4 gray path."
        )
    released_meta: set[str] = set()
    for detail in v4_release.get("gray_detail") or []:
        if not isinstance(detail, dict):
            continue
        if detail.get("decision") != "RELEASE":
            continue
        name = detail.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name.endswith(".weight"):
            released_meta.add(_st_weight_to_meta(name))
        else:
            released_meta.add(_normalize_module_meta(name))
    n_fence = len(hard_veto)
    hard_veto -= released_meta
    print(
        f"[HSWQ] V4 gray RELEASE: fence={n_fence} → hard_veto={len(hard_veto)} "
        f"(released={int(v4_release.get('gray_released', 0))} "
        f"kept={int(v4_release.get('gray_kept', 0))} "
        f"safe_p75_mse={float(v4_release.get('safe_p75_mse', float('nan'))):.6g} "
        f"thr={float(v4_release.get('mse_release_threshold', float('nan'))):.6g} "
        f"alpha={float(v4_release.get('alpha', alpha)):.6g})"
    )
    # Seed MSE axis from THIS SVD V4 safe sample when present (same as analyze).
    safe_mses = [
        float(d["estimated_mse"])
        for d in (v4_release.get("safe_detail") or [])
        if isinstance(d, dict) and "estimated_mse" in d
    ]
    if len(safe_mses) >= 4:
        tunables["recommended_safe_p75_mse"] = float(
            v4_release.get("safe_p75_mse", _safe_percentile(safe_mses, 75.0))
        )
        tunables["recommended_mse_release_threshold"] = float(
            v4_release.get("mse_release_threshold", 0.0)
        )
        tunables["v4_svd_enabled"] = True
        tunables["v4_alpha"] = float(v4_release.get("alpha", alpha))
        tunables["v4_beta"] = float(v4_release.get("beta", beta))

    char_table = build_nvfp4_analyze_character_table(
        {"layers": meta_layers, "source": model_path},
        tunables,
        hard_veto_names=hard_veto,
    )
    pool = set(meta_to_st.keys()) | set(hard_veto) | set(char_table.keys())
    pool = {n for n in pool if n in meta_to_st}
    for meta in sens_by_meta:
        if meta in meta_to_st:
            pool.add(meta)

    v4_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    optimizer = HSWQWeightedHistogramOptimizerV4(
        bins=8192,
        num_candidates=1000,
        refinement_iterations=10,
        device=v4_device,
        alpha=alpha,
        beta=beta,
    )
    print(
        f"[HSWQ] V4 pack-MSE @ absmax for {len(pool)} Linear-2D layers "
        f"(device={v4_device})..."
    )
    measured: list[tuple[str, float, float, float, int]] = []
    for meta in sorted(pool):
        st_key = meta_to_st[meta]
        w = sd[st_key]
        dm_sens = float(sens_by_meta.get(meta, 0.0))
        row = char_table.get(meta, {})
        prof = meta_layers.get(meta, {})
        is_hv = meta in hard_veto
        k = float(row.get("kurtosis", prof.get("kurtosis", 0)) or 0)
        o = float(row.get("outlier_ratio", prof.get("outlier_ratio", 0)) or 0)
        m = float(row.get("abs_max", prof.get("abs_max", 0)) or 0)
        mad = float(row.get("mad_outlier_pct", prof.get("mad_outlier_pct", 0)) or 0)
        ps = float(row.get("profile_score", prof.get("profile_score", 0)) or 0)
        severity = nvfp4_fp16_budget_analyze_severity(
            kurtosis=k,
            outlier_ratio=o,
            abs_max=m,
            tunables=tunables,
            is_hard_veto=is_hv,
            layer_name=meta,
            mad_outlier_pct=mad,
            profile_score=ps,
        )
        in_f = int(w.shape[1]) if int(getattr(w, "ndim", 0)) == 2 else -1
        # Reuse gray-release DualMonitor Imp (same pillar tensor) — do not
        # re-resolve / re-ones and drift from Analyze×V4×Imp gray path.
        if meta in st_importance:
            imp = st_importance[meta].detach().float()
        else:
            imp = _importance_for_meta(
                meta, importance_dict, in_features=in_f if in_f > 0 else None
            )
            if imp is None:
                if in_f <= 0:
                    raise RuntimeError(
                        f"[HSWQ] Linear weight missing in_features for {meta!r} ({st_key})"
                    )
                imp = torch.ones(in_f, dtype=torch.float32)
            else:
                imp = imp.detach().float()
        if in_f > 0 and int(imp.numel()) != in_f:
            # Align already applied in _importance_for_meta; re-align as safety.
            imp = _align_importance_1d(imp, in_f)
        if v4_device == "cuda":
            imp = imp.to(device="cuda")
        w_meas = w.detach().float()
        if v4_device == "cuda":
            w_meas = w_meas.to(device="cuda")
        try:
            v4_out = optimizer.compute_pack_mse_absmax_with_svd(
                w_meas,
                channel_importance=imp,
                importance=imp,
                use_svd_leverage=True,
                layer_name=meta,
                linear_pack="nvfp4",
            )
            v4_mse = float(v4_out.get("estimated_mse", v4_out.get("mse", 0.0)))
        except Exception as e:
            raise RuntimeError(
                f"[HSWQ] V4 pack-MSE failed for {meta!r} ({st_key}): {e}. "
                "Joint four-pillar path cannot skip this layer."
            ) from e
        if v4_device == "cuda":
            del w_meas
            del imp
            torch.cuda.empty_cache()
        # 5th tuple slot is required by analyze ranking APIs; INT8 protect
        # truncates by key count N only (not weight-byte / MiB fill).
        measured.append((meta, dm_sens, v4_mse, severity, 0))

    del sd, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not measured:
        raise RuntimeError("[HSWQ] measured pool is empty after V4 scoring")

    veto_mask_pre = [name in hard_veto for name, *_ in measured]
    measured, branch_repairs, branch_profile = apply_fp16_infinite_ranking_branches(
        measured, veto_mask_pre,
    )
    print(
        f"[HSWQ] Infinite ranking branches: repairs={len(branch_repairs)} "
        f"cv(s/v/m)={branch_profile.get('cv_sens', float('nan')):.4g}/"
        f"{branch_profile.get('cv_sev', float('nan')):.4g}/"
        f"{branch_profile.get('cv_mse', float('nan')):.4g}"
    )

    sens_all = [float(row[1]) for row in measured]
    sev_all = [float(row[3]) for row in measured]
    mse_all = [float(row[2]) for row in measured]
    veto_mask = [row[0] in hard_veto for row in measured]
    sens_meas = [v for v in sens_all if v > 0]
    sev_meas = list(sev_all)
    mse_meas = [v for v in mse_all if v > 0]
    s_p50 = _safe_percentile(sens_meas, 50.0) if len(sens_meas) >= 2 else 0.0
    s_iqr = _robust_iqr(sens_meas) if len(sens_meas) >= 4 else 0.0
    v_p50 = _safe_percentile(sev_meas, 50.0) if len(sev_meas) >= 2 else 0.0
    v_iqr = _robust_iqr(sev_meas) if len(sev_meas) >= 4 else 0.0
    m_p50 = _safe_percentile(mse_meas, 50.0) if len(mse_meas) >= 2 else 0.0
    m_iqr = _robust_iqr(mse_meas) if len(mse_meas) >= 4 else 0.0
    combinator = derive_priority_combinator(
        s_iqr, v_iqr, m_iqr, s_p50, v_p50, m_p50,
        sens_vals=sens_all,
        sev_vals=sev_all,
        mse_vals=mse_all,
        is_hard_veto=veto_mask,
    )
    print(
        f"[HSWQ] Priority combinator form={combinator['form']} "
        f"w(sens/sev/mse)={combinator['w_sens']:.3f}/"
        f"{combinator['w_sev']:.3f}/{combinator['w_mse']:.3f}"
    )

    candidates: list[tuple[float, float, float, float, int, str]] = []
    for name, dm_sens, v4_mse, severity, extra in measured:
        priority = nvfp4_fp16_budget_priority(
            dm_sens, v4_mse, severity, combinator=combinator,
        )
        candidates.append(
            (priority, v4_mse, severity, dm_sens, int(extra), name)
        )
    candidates, prio_repairs = apply_fp16_infinite_priority_branches(
        candidates, branch_profile,
    )
    if prio_repairs:
        print(f"[HSWQ] Infinite priority repairs={len(prio_repairs)}")
    candidates.sort(
        key=lambda t: (t[0], t[1], t[2], t[3], t[5]),
        reverse=True,
    )

    n_take = min(int(protect_n), len(candidates))
    selected_meta = [row[5] for row in candidates[:n_take]]
    selected_st = [meta_to_st[m] for m in selected_meta]
    source = f"hswq_four_pillars_n{protect_n}"
    print(
        f"[HSWQ] Selected INT8 protect n={len(selected_st)}/{protect_n} "
        f"(pool={len(candidates)} source={source})"
    )
    for i, (meta, st_key) in enumerate(zip(selected_meta, selected_st)):
        pr, mse, sev, sens, _ex, _n = candidates[i]
        print(
            f"  [{i + 1:02d}] prio={pr:.6g} mse={mse:.6g} sev={sev:.4g} "
            f"sens={sens:.4g} | {st_key}"
        )
    return frozenset(selected_st), source


def _is_int8_protect_key(key: str, keyset: frozenset[str]) -> bool:
    """True if key is in analysis INT8 protect set (exact or prefix variants)."""
    if key in keyset:
        return True
    if key.startswith("diffusion_model."):
        alt = "model." + key
        if alt in keyset:
            return True
    if not key.startswith("model.diffusion_model."):
        alt = "model.diffusion_model." + key
        if alt in keyset:
            return True
    return False


_NON_DIFFUSION_MARKERS: tuple[str, ...] = (
    "conditioner.",
    "cond_stage_model.",
    "text_encoders.",
    "text_encoder.",
    "text_encoder_2.",
    "text_encoder_3.",
    "text_model.",
    "text_projection",
    "logit_scale",
    "clip_l.",
    "clip_g.",
    "t5xxl.",
    "first_stage_model.",
    "vae.",
)


def _is_non_diffusion_key(key: str) -> bool:
    return any(marker in key for marker in _NON_DIFFUSION_MARKERS)


def _find_z_image_key_prefix(state_dict) -> str:
    """Lumina2 / NextDiT / Z-Image signature (ComfyUI model_detection).

    Requires cap_embedder.1.weight and noise_refiner.0 attention
    (k_norm or fused qkv) under a known diffusion prefix.
    """
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        cap = f"{prefix}cap_embedder.1.weight"
        if cap not in state_dict:
            continue
        k_norm = f"{prefix}noise_refiner.0.attention.k_norm.weight"
        qkv = f"{prefix}noise_refiner.0.attention.qkv.weight"
        if k_norm in state_dict or qkv in state_dict:
            return prefix
    raise ValueError(
        "Not a Z-Image / ZIT (NextDiT / Lumina2) checkpoint: missing "
        "cap_embedder.1.weight + noise_refiner.0.attention.(k_norm|qkv).weight "
        "(under model.diffusion_model. / diffusion_model. / root)."
    )


def _meta_base_key(base_k_file: str) -> str:
    if "model.diffusion_model." in base_k_file:
        return base_k_file.split("model.diffusion_model.")[-1]
    if "diffusion_model." in base_k_file:
        return base_k_file.split("diffusion_model.")[-1]
    return base_k_file


def convert_to_nvfp4(
    input_path: str,
    output_path: str,
    device: str,
    model_type: str = _DEFAULT_MODEL_TYPE,
    enable_convrot: bool = True,
    group_size: int = _DEFAULT_GROUPSIZE,
    bias_correction: bool = False,
    calib_file: str | None = None,
    clip_path: str | None = None,
    comfy_path: str | None = None,
    tokenizer_path: str | None = None,
    num_calib_samples: int = 32,
    num_inference_steps: int = 25,
    int8_protect_keys: frozenset[str] | list[str] | None = None,
    int8_protect_source: str | None = None,
    precomputed_act_mean: dict[str, torch.Tensor] | None = None,
):
    if model_type not in _Z_IMAGE_PROFILES:
        raise ValueError(
            f"Unknown model_type={model_type!r}; "
            f"choose from {sorted(_Z_IMAGE_PROFILES)}"
        )
    blacklist, fp8_layers = _Z_IMAGE_PROFILES[model_type]
    protect_keyset, protect_source = _resolve_int8_protect_keyset(
        int8_protect_keys, int8_protect_source
    )

    rot_tag = "FULL ConvRot NVFP4" if enable_convrot else "plain NVFP4"
    print(
        f"Mode {model_type} | device={device} | {rot_tag} "
        f"+ ConvRot INT8 protect ({len(protect_keyset)} keys)"
    )
    print(
        f"  [INT8 protect] {len(protect_keyset)} keys from {protect_source} → "
        "ConvRot INT8 (rowwise)"
    )
    if enable_convrot:
        print(
            f"  [ConvRot] ON | preferred groupsize={int(group_size)} "
            f"(Linear 2D; skip rotate when in_features has no power-of-4 group)"
        )
    else:
        print("  [ConvRot] OFF | plain Kitchen NVFP4 packs only")

    act_mean_dict: dict[str, torch.Tensor] = {}
    bias_corr_pending: dict[str, torch.Tensor] = {}
    n_bias_corr = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_bad_shape = 0
    bias_corr_skipped_no_bias = 0
    if bias_correction and enable_convrot:
        # Do not hard-abort existing notebooks / scripts that request bias with
        # default ConvRot ON. The four-pillar protect path already ran; here we
        # degrade bias correction only (weights stay ConvRot NVFP4/INT8).
        print(
            "[Card 1] SKIP bias correction: ConvRot ON mixes pre-rotation acts "
            "with post-rotation weights (HSWQ blasphemy). "
            "Use --no-enable_convrot if bias correction is required."
        )
        bias_correction = False
    if bias_correction:
        if precomputed_act_mean is not None:
            act_mean_dict = precomputed_act_mean
            print(
                f"\n[Card 1] Using precomputed DualMonitor act_mean "
                f"({len(act_mean_dict)} Linear modules)"
            )
        else:
            if not calib_file:
                raise ValueError("--bias_correction requires --calib_file")
            if not clip_path:
                raise ValueError("--bias_correction requires --clip_path")
            if not comfy_path:
                raise ValueError("--bias_correction requires --comfy_path")
            if device != "cuda":
                raise ValueError("--bias_correction requires --device cuda")
            print("\n[Card 1] Running DualMonitor calibration (unquantized acts)...")
            act_mean_dict, _sens_unused, _imp_unused = run_card1_calib(
                model_path=input_path,
                clip_path=clip_path,
                comfy_path=comfy_path,
                calib_file=calib_file,
                tokenizer_path=tokenizer_path,
                num_samples=num_calib_samples,
                num_inference_steps=num_inference_steps,
            )
            print(
                f"[Card 1] act_mean for {len(act_mean_dict)} Linear modules "
                f"(keyed by module name = _meta_base_key)"
            )

    sd = load_file(input_path)
    prefix = _find_z_image_key_prefix(sd)
    print(f"Detected Z-Image key prefix: {prefix!r}")

    # Structural summary (helps audit Turbo vs Base choice)
    n_layers = sum(
        1
        for k in sd
        if k.startswith(f"{prefix}layers.") and k.endswith(".feed_forward.w1.weight")
    )
    has_noise = any(f"{prefix}noise_refiner." in k for k in sd)
    has_ctx = any(f"{prefix}context_refiner." in k for k in sd)
    print(
        f"Structure: layers(w1)={n_layers} "
        f"noise_refiner={has_noise} context_refiner={has_ctx}"
    )
    if model_type == "Z-Image-Base":
        print(
            "[!] Z-Image-Base Kitchen blacklist also matches layers.*.attention / "
            "adaLN_modulation / norm — NVFP4 candidates shrink to feed_forward "
            "2D weights mainly. ZIT / Turbo UNets usually want Z-Image-Turbo."
        )

    quant_map = {"format_version": "1.0", "layers": {}}
    new_sd: dict[str, torch.Tensor] = {}
    n_nvfp4 = 0
    n_convrot = 0
    n_plain_nvfp4 = 0
    n_bf16 = 0
    n_int8_protect = 0
    n_int8_convrot = 0
    n_int8_plain = 0

    print(f"Converting ({len(sd)} tensors)...")
    for k, v in tqdm(list(sd.items())):
        if any(name in k for name in blacklist):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        if _is_non_diffusion_key(k):
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1
            continue

        # Analysis ConvRot INT8 protect (before NVFP4) — injected keyset
        if _is_int8_protect_key(k, protect_keyset) and v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            w = v.float().cpu()
            used_gs = convrot_group_size_for_features(
                int(w.shape[1]), int(group_size)
            )
            if used_gs is not None:
                h_matrix = build_hadamard(
                    int(used_gs), device="cpu", dtype=torch.float32
                )
                w = rotate_weight(w, h_matrix, int(used_gs))
                q, scale = quantize_int8_rowwise(w)
                quant_config = {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": int(used_gs),
                }
                n_int8_convrot += 1
            else:
                q, scale = quantize_int8_tensorwise(w)
                quant_config = {"format": "int8_tensorwise"}
                n_int8_plain += 1
            new_sd[k] = q
            new_sd[f"{base_k_file}.weight_scale"] = scale
            # ComfyUI load peeks ``{prefix}comfy_quant`` next to the weight.
            new_sd[f"{base_k_file}.comfy_quant"] = _encode_comfy_quant(quant_config)
            quant_map["layers"][base_k_meta] = dict(quant_config)
            n_int8_protect += 1
            # Card1: protect path used to ``continue`` before NVFP4 BC — INT8 layers
            # never got (W_dq-W)@μ. Same math as NVFP4 branch / native_convert_int8_sdxl.
            if bias_correction and act_mean_dict is not None:
                act_mean = act_mean_dict.get(base_k_meta)
                if act_mean is None:
                    bias_corr_skipped_no_act += 1
                else:
                    weight_dq = q.float() * scale.float()
                    delta = compute_nvfp4_bias_delta(w, weight_dq, act_mean)
                    if delta is None:
                        bias_corr_skipped_bad_shape += 1
                    else:
                        bias_corr_pending[base_k_file] = (
                            (-delta).detach().float().cpu()
                        )
            continue

        if v.ndim == 2 and ".weight" in k:
            base_k_file = k.replace(".weight", "")
            base_k_meta = _meta_base_key(base_k_file)
            v_tensor = v.to(device=device, dtype=torch.bfloat16)

            if fp8_layers and any(name in k for name in fp8_layers):
                import comfy_kitchen as ck

                weight_scale = (
                    (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                )
                weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                new_sd[k] = weight_quantized.cpu()
                new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(
                    torch.bfloat16
                ).cpu()
                quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                if device == "cuda":
                    del v_tensor
                continue

            used_gs = None
            do_rotate = False
            w_for_q = v_tensor
            if enable_convrot:
                used_gs = convrot_group_size_for_features(
                    int(v_tensor.shape[1]), int(group_size)
                )
                if used_gs is not None:
                    h_matrix = build_hadamard(
                        int(used_gs), device="cpu", dtype=torch.float32
                    )
                    w_rot = rotate_weight(
                        v_tensor.float().cpu(), h_matrix, int(used_gs)
                    )
                    w_for_q = w_rot.to(device=device, dtype=torch.bfloat16)
                    do_rotate = True

            try:
                qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                if do_rotate and used_gs is not None:
                    quant_config = {
                        "format": "nvfp4",
                        "convrot": True,
                        "convrot_groupsize": int(used_gs),
                    }
                    n_convrot += 1
                else:
                    quant_config = {"format": "nvfp4"}
                    n_plain_nvfp4 += 1
                new_sd[f"{base_k_file}.comfy_quant"] = _encode_comfy_quant(
                    quant_config
                )
                quant_map["layers"][base_k_meta] = dict(quant_config)
                n_nvfp4 += 1

                # Card 1: accumulate bias delta while w_for_q (pre-quant) still lives.
                if bias_correction and act_mean_dict is not None:
                    act_mean = act_mean_dict.get(base_k_meta)
                    if act_mean is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        weight_dq = _dequant_nvfp4(qdata, params)
                        if weight_dq is None:
                            bias_corr_skipped_bad_shape += 1
                        else:
                            delta = compute_nvfp4_bias_delta(
                                w_for_q.float(), weight_dq, act_mean
                            )
                            if delta is None:
                                bias_corr_skipped_bad_shape += 1
                            else:
                                bias_corr_pending[base_k_file] = (
                                    (-delta).detach().float().cpu()
                                )
            except Exception:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                n_bf16 += 1

            if device == "cuda":
                if do_rotate:
                    del w_for_q
                del v_tensor
        else:
            new_sd[k] = v.to(dtype=torch.bfloat16)
            n_bf16 += 1

    if bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"NVFP4 Linear layers (Card 1)..."
        )
        for base_k_file, delta in bias_corr_pending.items():
            bias_key = f"{base_k_file}.bias"
            if bias_key not in new_sd:
                bias_corr_skipped_no_bias += 1
                continue
            bias = new_sd[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            new_sd[bias_key] = corrected.to(dtype=bias.dtype)
            n_bias_corr += 1
        print(
            f"  [Bias Correction] applied={n_bias_corr}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    by_tag = (
        "ComfyUI Kitchen NVFP4 Converter (Z-Image ConvRot + INT8 protect)"
        if enable_convrot
        else "ComfyUI Kitchen NVFP4 Converter (Z-Image INT8 protect)"
    )
    final_metadata["converted_by"] = by_tag
    final_metadata["converter_url"] = (
        "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
    )
    final_metadata["hswq_model"] = "z_image"
    final_metadata["hswq_kitchen_profile"] = model_type
    final_metadata["hswq_nvfp4_convrot"] = "1" if enable_convrot else "0"
    final_metadata["hswq_int8_protect"] = "1"
    final_metadata["hswq_int8_protect_n"] = str(n_int8_protect)
    final_metadata["hswq_int8_protect_convrot"] = str(n_int8_convrot)
    final_metadata["hswq_int8_protect_source"] = protect_source

    print(f"Saving | Type: {model_type} | Path: {output_path}")
    save_file(new_sd, output_path, metadata=final_metadata)
    total_bytes = os.path.getsize(output_path)
    print(f"Done. Size: {round(total_bytes / (1024**3), 2)} GiB")
    print(f"NVFP4+INT8 layers in metadata: {len(quant_map['layers'])}")
    print(
        f"  counted nvfp4 packs={n_nvfp4} | bf16 keep tensors={n_bf16} | "
        f"int8 protect={n_int8_protect} "
        f"(convrot={n_int8_convrot}, plain={n_int8_plain})"
    )
    print(f"FULL ConvRot enabled (NVFP4 path): {enable_convrot}")
    if enable_convrot:
        print(
            f"  ConvRot NVFP4 Linear: {n_convrot}, "
            f"plain NVFP4 (no group): {n_plain_nvfp4}"
        )

    del sd
    del new_sd
    del quant_map
    _release_vram("after native Z-Image NVFP4 convert save")


def _release_vram(label: str = "post-convert") -> None:
    print(f"[*] Releasing VRAM ({label})...")
    gc.collect()
    if not torch.cuda.is_available():
        print(f"[*] VRAM clear ({label}): CUDA not available")
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        alloc_mib = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv_mib = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[*] VRAM clear ({label}): "
            f"allocated={alloc_mib:.1f} MiB reserved={reserv_mib:.1f} MiB"
        )
    except Exception:
        print(f"[*] VRAM clear ({label}): done")


# Exact --prompt / --steps from zi_convrot_nvfp4_bench.py Example (fixed; not CLI).
_FIXED_ZI_CONVROT_BENCH_PROMPT = (
    "A beautiful cyberpunk city at night, high detail."
)
_FIXED_ZI_CONVROT_BENCH_STEPS = 25
# Seed fixed inside this chain (not a parent CLI). Same default as
# benchmark/zi_convrot_nvfp4_bench.py --seed (must be passed explicitly).
_FIXED_ZI_CONVROT_BENCH_SEED = 42


def run_post_convert_zi_convrot_nvfp4_bench(
    *,
    script_dir: str,
    fp16_path: str,
    nvfp4_path: str,
    clip_path: str,
    comfy_path: str,
    vae_path: str | None = None,
    token: str | None = None,
) -> int:
    """After save: subprocess benchmark/zi_convrot_nvfp4_bench.py.

    Owner body argv order + seed fixed inside:
      --fp16 --nvfp4 --clip_path --comfy_path
      [--vae] [--token] --prompt --steps 25 --seed <fixed>
    """
    bench_script = os.path.join(
        script_dir, "benchmark", "zi_convrot_nvfp4_bench.py"
    )
    if not os.path.isfile(bench_script):
        print(f"[FATAL] Post-convert bench script not found: {bench_script}")
        return 1
    if not os.path.isfile(fp16_path):
        print(
            f"[FATAL] Post-convert bench: FP16 (--model) missing: {fp16_path}"
        )
        return 1
    if not os.path.isfile(nvfp4_path):
        print(
            f"[FATAL] Post-convert bench: NVFP4 (--output) missing: {nvfp4_path}"
        )
        return 1
    if not clip_path or not os.path.isfile(clip_path):
        print(
            f"[FATAL] Post-convert bench: --clip_path missing: {clip_path}"
        )
        return 1
    if not comfy_path or not os.path.isdir(comfy_path):
        print(
            f"[FATAL] Post-convert bench: --comfy_path missing: {comfy_path}"
        )
        return 1
    if vae_path and not os.path.isfile(vae_path):
        print(f"[FATAL] Post-convert bench: --vae missing: {vae_path}")
        return 1

    _release_vram("pre-zi_convrot_nvfp4_bench subprocess")

    # Owner body order (bench body untouched).
    cmd = [
        sys.executable,
        bench_script,
        "--fp16",
        fp16_path,
        "--nvfp4",
        nvfp4_path,
        "--clip_path",
        clip_path,
        "--comfy_path",
        comfy_path,
    ]
    if vae_path:
        cmd.extend(["--vae", vae_path])
    if token:
        cmd.extend(["--token", token])
    cmd.extend(
        [
            "--prompt",
            _FIXED_ZI_CONVROT_BENCH_PROMPT,
            "--steps",
            str(_FIXED_ZI_CONVROT_BENCH_STEPS),
            "--seed",
            str(_FIXED_ZI_CONVROT_BENCH_SEED),
        ]
    )

    print("=" * 60)
    print("[*] Post-convert ZI ConvRot NVFP4 bench (owner body shape)")
    print(f"    script: {bench_script}")
    print(f"    --fp16: {fp16_path}")
    print(f"    --nvfp4: {nvfp4_path}")
    print(f"    --clip_path: {clip_path}")
    print(f"    --comfy_path: {comfy_path}")
    if vae_path:
        print(f"    --vae: {vae_path}")
    if token:
        print("    --token: (provided)")
    print(f"    --prompt: {_FIXED_ZI_CONVROT_BENCH_PROMPT}")
    print(f"    --steps: {_FIXED_ZI_CONVROT_BENCH_STEPS}")
    print(f"    --seed: {_FIXED_ZI_CONVROT_BENCH_SEED} (fixed inside)")
    print("=" * 60)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Full run log: capture stdout + stderr in memory; write one complete
    # log/<script>_<ts>.txt at process end (normal or error).
    # ------------------------------------------------------------------
    import io
    import traceback

    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(
        _log_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.txt",
    )
    _log_buf = io.StringIO()

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, _log_buf)
    sys.stderr = _Tee(_orig_stderr, _log_buf)

    _log_flushed = False

    def _flush_full_log() -> None:
        nonlocal _log_flushed
        if _log_flushed:
            return
        _log_flushed = True
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        with open(_log_path, "w", encoding="utf-8") as fh:
            fh.write(_log_buf.getvalue())
        print(f"[log] Full run log written: {_log_path}")

    _orig_exit = sys.exit

    def _exit_with_log(code=0):
        _flush_full_log()
        _orig_exit(code)

    sys.exit = _exit_with_log

    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / ZIT NVFP4 + HSWQ ConvRot INT8 protect. "
            "Default: DualMonitor calib + four-pillar ranking → top N "
            "(default 60) as ConvRot INT8; rest NVFP4. FULL ConvRot ON. "
            "Kitchen profile default Z-Image-Turbo. "
            "Post-convert zi_convrot_nvfp4_bench default ON."
        )
    )
    parser.add_argument(
        "--model",
        "--input",
        dest="model",
        type=str,
        required=True,
        help="Path to Z-Image / ZIT BF16/FP16 .safetensors",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output .safetensors"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=_DEFAULT_MODEL_TYPE,
        choices=sorted(_Z_IMAGE_PROFILES.keys()),
        help=(
            "Kitchen Z-Image profile (default: Z-Image-Turbo; "
            "use Z-Image-Base only for Kitchen Base blacklist)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Quantize device",
    )
    parser.add_argument(
        "--no-convrot",
        dest="enable_convrot",
        action="store_false",
        help="Disable ConvRot; pack plain Kitchen NVFP4 only.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=_DEFAULT_GROUPSIZE,
        help=f"Preferred ConvRot Hadamard group size (default {_DEFAULT_GROUPSIZE}).",
    )
    parser.add_argument(
        "--protect-n",
        type=int,
        default=60,
        help="HSWQ INT8 protect key count N (default 60; truncate after ranking)",
    )
    parser.add_argument(
        "--protect-keys-json",
        type=str,
        default=None,
        help=(
            "Optional JSON list/object of ST weight keys; skips HSWQ auto ranking"
        ),
    )
    parser.add_argument(
        "--protect-keys-hardcode",
        action="store_true",
        help=(
            "Use baked moodyRealMix_zitV7 swap3 N=60 keyset "
            "(skips HSWQ auto ranking)"
        ),
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help=(
            "Enable Card 1 bias += -(W_q-W)@mu_x after packs. "
            "Reuses DualMonitor act_mean from HSWQ calib when available."
        ),
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=None,
        help=(
            "JSONL prompts for DualMonitor (required for HSWQ auto ranking "
            "and for --bias_correction)"
        ),
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default=None,
        help="Qwen3-4B text encoder path (HSWQ Card1 / post-convert bench)",
    )
    parser.add_argument(
        "--comfy_path",
        type=str,
        default=None,
        help="ComfyUI root path (HSWQ Card1 / post-convert bench)",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help=(
            "Optional VAE path for post-convert bench "
            "(forwarded as --vae when provided)"
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face token for post-convert bench "
            "(forwarded as --token when provided)"
        ),
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After save, run benchmark/zi_convrot_nvfp4_bench.py "
            "(fp16/nvfp4/clip/comfy/prompt/steps=25; "
            "optional --vae/--token when provided). "
            "Pass --no-bench to skip."
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional local Qwen2Tokenizer dir (offline Card 1)",
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=32,
        help="DualMonitor prompt count (default: 32; HSWQ How-to)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="DualMonitor sample_euler steps (default: 25; HSWQ How-to)",
    )
    parser.set_defaults(enable_convrot=True)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    n_modes = sum(
        1
        for flag in (
            bool(args.protect_keys_json),
            bool(args.protect_keys_hardcode),
        )
        if flag
    )
    if n_modes > 1:
        print(
            "Error: use at most one of --protect-keys-json / "
            "--protect-keys-hardcode (default is HSWQ auto)"
        )
        sys.exit(1)

    precomputed_act_mean: dict[str, torch.Tensor] | None = None
    keyset: frozenset[str]
    source: str

    # HSWQ complete path: hardcode / JSON keysets are reference-only.
    # Owner must explicitly use --protect-keys-hardcode or --protect-keys-json
    # to skip four-pillar auto; silent or default hardcode is blasphemy.
    # (Documented in this file's docstring + _INT8_PROTECT_SOURCE comment.)
    if args.protect_keys_hardcode:
        print(
            "[hardcode] NOTE: --protect-keys-hardcode bypasses HSWQ four-pillar "
            "judgment. Use only when the owner explicitly orders hardcode."
        )
        keyset = frozenset(_INT8_PROTECT_KEYSET)
        source = _INT8_PROTECT_SOURCE
        print(f"[hardcode] INT8 protect n={len(keyset)} source={source}")
        if len(keyset) != 60:
            print(f"Error: hardcode keyset must be 60, got {len(keyset)}")
            sys.exit(1)
    elif args.protect_keys_json:
        json_path = str(args.protect_keys_json)
        if not os.path.exists(json_path):
            print(f"Error: --protect-keys-json not found: {json_path}")
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            raw_keys = payload.get("keys", payload.get("protect_keys"))
            if raw_keys is None and all(isinstance(k, str) for k in payload.keys()):
                raw_keys = list(payload.keys())
        elif isinstance(payload, list):
            raw_keys = payload
        else:
            print(
                "Error: --protect-keys-json must be a list of keys or "
                "an object with keys/protect_keys"
            )
            sys.exit(1)
        if not isinstance(raw_keys, list) or not raw_keys:
            print("Error: --protect-keys-json produced an empty key list")
            sys.exit(1)
        keyset = frozenset(str(k) for k in raw_keys)
        source = f"protect_keys_json:{os.path.basename(json_path)}"
        print(f"[json] INT8 protect n={len(keyset)} source={source}")
    else:
        # Default: HSWQ four-pillar auto (ZI analyze + DualMonitor + V4 + priority)
        if not args.calib_file:
            print("Error: HSWQ auto ranking requires --calib_file")
            sys.exit(1)
        if not args.clip_path:
            print("Error: HSWQ auto ranking requires --clip_path")
            sys.exit(1)
        if not args.comfy_path:
            print("Error: HSWQ auto ranking requires --comfy_path")
            sys.exit(1)
        if str(args.device) != "cuda":
            print("Error: HSWQ auto ranking requires --device cuda")
            sys.exit(1)
        print(
            f"\n[HSWQ] DualMonitor calib "
            f"(samples={int(args.num_calib_samples)} "
            f"steps={int(args.num_inference_steps)})..."
        )
        act_mean_dict, sens_dict, importance_dict = run_card1_calib(
            model_path=args.model,
            clip_path=args.clip_path,
            comfy_path=args.comfy_path,
            calib_file=args.calib_file,
            tokenizer_path=args.tokenizer_path,
            num_samples=int(args.num_calib_samples),
            num_inference_steps=int(args.num_inference_steps),
        )
        precomputed_act_mean = act_mean_dict
        keyset, source = select_int8_protect_keys_hswq(
            args.model,
            sens_dict,
            importance_dict,
            protect_n=int(args.protect_n),
            model_type=str(args.model_type),
            device=str(args.device),
        )
        print(f"[HSWQ] INT8 protect n={len(keyset)} source={source}")

    convert_to_nvfp4(
        args.model,
        args.output,
        device=str(args.device),
        model_type=str(args.model_type),
        enable_convrot=bool(args.enable_convrot),
        group_size=int(args.group_size),
        bias_correction=bool(args.bias_correction),
        calib_file=args.calib_file,
        clip_path=args.clip_path,
        comfy_path=args.comfy_path,
        tokenizer_path=args.tokenizer_path,
        num_calib_samples=int(args.num_calib_samples),
        num_inference_steps=int(args.num_inference_steps),
        int8_protect_keys=keyset,
        int8_protect_source=source,
        precomputed_act_mean=precomputed_act_mean,
    )

    if args.bench:
        bench_rc = run_post_convert_zi_convrot_nvfp4_bench(
            script_dir=os.path.dirname(os.path.abspath(__file__)),
            fp16_path=args.model,
            nvfp4_path=args.output,
            clip_path=args.clip_path,
            comfy_path=args.comfy_path,
            vae_path=args.vae,
            token=args.token,
        )
        if bench_rc != 0:
            print(f"[FATAL] Post-convert bench exited with code {bench_rc}")
            sys.exit(bench_rc)
    else:
        print("[*] Post-convert bench skipped (--no-bench)")

    _flush_full_log()
