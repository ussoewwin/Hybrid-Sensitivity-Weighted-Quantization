"""
Z-Image / NextDiT INT8 quantization — HSWQ V1.0
================================================

ZI-format pipeline (load / calib / Static+Structural VETO / Z-Anime):
  same infrastructure as quantize_zib_hswq_v2.0.py (loaded via importlib).

INT8 FP16 protect (HSWQ — per-checkpoint auto analysis → auto-optimal):
  - Owner hard frame: FP16 overhead vs all-INT8 == 300 MiB exactly.
  - DualMonitor sensitivity × analyze severity × V4 estimated_mse rank
    with infinite THIS-model ranking / priority branches (no fixed formula,
    no keep_ratio % cut). Extreme fill under 300 MiB only truncates.
  - Pack amax stays absmax (tensorwise) or per-out-channel (Card 3).
  - Card 1 (--bias_correction): bias += -(W_q - W) @ mu_x
    mu_x = DualMonitor.channel_act_mean from ZITCalibrationPipeline.
  - Card 3 (--per_channel_int8): per-out-channel scale (O,1) / (O,1,1,1).
    Format tag stays int8_tensorwise for ComfyUI kitchen dequant.

CLI style matches ZIB v2.0 (--input/--output/--calib_file/--clip_path/...).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys

import torch
from safetensors.torch import save_file
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "ComfyUI-master"))
histogram_dir = os.path.join(current_dir, "histogram")
if histogram_dir not in sys.path:
    sys.path.insert(0, histogram_dir)


def _load_zib_v20():
    """Load quantize_zib_hswq_v2.0.py (ZI-format engine)."""
    path = os.path.join(current_dir, "quantize_zib_hswq_v2.0.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ZI engine not found: {path}")
    spec = importlib.util.spec_from_file_location("quantize_zib_hswq_v2_0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_zib_hswq_v2_0"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_hswq_int8_budget():
    """Load INT8 300 MiB budget + infinite-branch helpers (shared HSWQ path)."""
    path = os.path.join(current_dir, "quantize_sdxl_hswq_v3.0.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HSWQ INT8 budget engine not found: {path}")
    mod_name = "quantize_hswq_int8_budget_v3_0"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class DualMonitorInt8:
    """ZI DualMonitor + signed channel_act_mean for Card 1 bias correction."""

    def __init__(self):
        self.output_sum = 0.0
        self.output_sq_sum = 0.0
        self.count = 0
        self.channel_importance = None
        self.channel_act_mean = None

    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            out_detached = output_tensor.detach().float()
            out_clamped = torch.clamp(out_detached, -65504.0, 65504.0)
            mean_val = out_clamped.mean().item()
            sq_mean_val = (out_clamped ** 2).mean().item()
            if math.isfinite(mean_val) and math.isfinite(sq_mean_val):
                self.output_sum += mean_val
                self.output_sq_sum += sq_mean_val
            inp = input_tensor.detach().float()
            if inp.dim() == 4:
                current_imp = inp.abs().mean(dim=(0, 2, 3))
                current_act = inp.mean(dim=(0, 2, 3))
            elif inp.dim() == 3:
                current_imp = inp.abs().mean(dim=(0, 1))
                current_act = inp.mean(dim=(0, 1))
            elif inp.dim() == 2:
                current_imp = inp.abs().mean(dim=0)
                current_act = inp.mean(dim=0)
            else:
                current_imp = torch.ones(1, device=inp.device, dtype=torch.float32)
                current_act = torch.zeros(1, device=inp.device, dtype=torch.float32)
            if self.channel_importance is None:
                self.channel_importance = current_imp
                self.channel_act_mean = current_act
            else:
                c = self.count
                self.channel_importance = (
                    self.channel_importance * c + current_imp
                ) / (c + 1)
                self.channel_act_mean = (
                    self.channel_act_mean * c + current_act
                ) / (c + 1)
            self.count += 1

    def get_sensitivity(self):
        if self.count == 0:
            return 0.0
        mean = self.output_sum / self.count
        variance = (self.output_sq_sum / self.count) - mean ** 2
        return variance if math.isfinite(variance) else 0.0


def pack_tensorwise(weight: torch.Tensor):
    """Symmetric per-tensor INT8: scale = amax / 127."""
    w = weight.float()
    amax = max(float(w.abs().max().item()), 1e-6)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32)


def pack_channelwise(weight: torch.Tensor):
    """Card 3: per-out-channel INT8. Scale shape (O,1) or (O,1,1,1)."""
    w = weight.float()
    reduce_dims = tuple(range(1, w.dim()))
    amax = torch.clamp(w.abs().amax(dim=reduce_dims).reshape(-1), min=1e-6)
    scale = amax / 127.0
    if w.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
        amax_view = amax.view(-1, 1, 1, 1)
    elif w.dim() == 2:
        scale_view = scale.view(-1, 1)
        amax_view = amax.view(-1, 1)
    else:
        raise ValueError(f"unsupported weight ndim={w.dim()} for --per_channel_int8")
    clamped = torch.clamp(w, -amax_view, amax_view)
    q = (clamped / scale_view).round().clamp(-127, 127).to(torch.int8)
    return q, scale_view.to(dtype=torch.float32)


def compute_int8_bias_delta(weight_fp, weight_dq, act_mean):
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


def _emit_int8_meta(out_dict, prefixed_module, scale):
    out_dict[f"{prefixed_module}.weight_scale"] = scale
    out_dict[f"{prefixed_module}.comfy_quant"] = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
        dtype=torch.uint8,
    )


def _v4_score_all_fp16_candidates(
    *,
    hswq_int8,
    model,
    dual_monitors,
    target_modules,
    hard_veto_layers,
    alpha,
    beta,
    device,
    mse_cache=None,
):
    """V4 estimated_mse @ absmax for ALL target Linear/Conv — no keep_ratio cut.

    Truncation is only the 300 MiB budget pass over THIS-model priority order
    (auto analysis → infinite branches → extreme fill).
    """
    return hswq_int8._build_v4_calib_fp16_candidates(
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        mse_cache=dict(mse_cache or {}),
        alpha=alpha,
        beta=beta,
        device=device,
    )


def main():
    hswq_int8 = _load_hswq_int8_budget()
    budget_hard = float(hswq_int8.FP16_BUDGET_MB_HARD)

    parser = argparse.ArgumentParser(
        description=(
            "Z-Image / NextDiT INT8 HSWQ V1.0 — 300 MiB FP16 frame + "
            "per-checkpoint auto analysis → infinite-branch fill + "
            "Card 1 bias correction + Card 3 per-channel (ZI format via zib v2.0)."
        )
    )
    # --- ZI CLI (same as quantize_zib_hswq_v2.0.py) ---
    parser.add_argument("--input", type=str, required=True, help="Path to input safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to output safetensors")
    parser.add_argument("--calib_file", type=str, required=True, help="Calibration prompts text")
    parser.add_argument("--clip_path", type=str, required=True, help="Text encoder safetensors")
    parser.add_argument("--num_calib_samples", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--fp16_budget_mb",
        type=float,
        default=budget_hard,
        help=(
            f"Owner hard ceiling for FP16 overhead vs all-INT8 "
            f"(must be exactly {budget_hard:g} MiB). Auto analysis fills "
            f"this frame; never redefine or exceed it."
        ),
    )
    parser.add_argument("--comfy_path", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    # --- INT8 cards ---
    parser.add_argument(
        "--per_channel_int8",
        action="store_true",
        help="Card 3: per-out-channel amax/scale. Default tensorwise absmax.",
    )
    parser.add_argument(
        "--bias_correction",
        action="store_true",
        help="Card 1: DualMonitor act_mean bias fold after INT8 pack.",
    )
    parser.add_argument(
        "--bias_correction_top_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of INT8 layers (by DualMonitor sensitivity, high first) "
            "that receive Card 1. Default None = autonomous from THIS "
            "checkpoint DualMonitor / analyze character."
        ),
    )
    args = parser.parse_args()
    args.fp16_budget_mb = hswq_int8._require_fp16_budget_mb_hard(
        float(args.fp16_budget_mb)
    )
    _bc_top_override = args.bias_correction_top_ratio

    zib = _load_zib_v20()
    script_dir = current_dir

    raw_input_arg = args.input
    resolved_input, tried_inputs = zib.resolve_weights_path(raw_input_arg, script_dir)
    if not os.path.isfile(resolved_input):
        print("[FATAL] Input weights file not found.")
        for p in tried_inputs:
            print(f"    - {p}")
        sys.exit(1)
    args.input = resolved_input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("HSWQ Z-Image INT8 V1.0 (V4 + Card1 + Card3, ZI format)")
    print("=" * 60)

    # --- ComfyUI / tokenizer / TE (same block shape as ZIB v2.0) ---
    comfy_path = args.comfy_path
    if comfy_path is None:
        comfy_path = os.environ.get(
            "COMFYUI_PATH", os.path.join(os.getcwd(), "ComfyUI")
        )
    if os.path.exists(comfy_path) and comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    try:
        import comfy.ops
        from comfy.text_encoders import llama as llama_module
        from transformers import Qwen2Tokenizer
        from safetensors.torch import load_file as _load_file

        tokenizer_dir = zib.resolve_tokenizer_offline(
            args.tokenizer_path, args.comfy_path, args.clip_path
        )
        if tokenizer_dir:
            print(f"  Loading tokenizer from disk: {tokenizer_dir}")
            try:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    tokenizer_dir, local_files_only=True
                )
            except Exception:
                tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
        else:
            model_id = args.tokenizer_path or "Qwen/Qwen2.5-7B-Instruct"
            print(f"  Trying Repo ID (STRICT LOCAL): {model_id}")
            tokenizer = Qwen2Tokenizer.from_pretrained(
                model_id, local_files_only=True
            )

        print(f"[*] Loading Text Encoder: {args.clip_path}")
        te_sd = _load_file(args.clip_path)
        text_encoder = llama_module.Qwen3_4B(
            config_dict={},
            device=device,
            dtype=torch.float16,
            operations=comfy.ops.disable_weight_init,
        )
        text_encoder.load_state_dict(te_sd, strict=False)
        text_encoder.eval()
    except Exception as e:
        print(f"[FATAL] Failed to load tokenizer/text_encoder: {e}")
        sys.exit(1)

    # --- Profile (analyze_zib_distribution) ---
    analyze_script = os.path.join(script_dir, "analyze", "analyze_zib_distribution.py")
    if not os.path.exists(analyze_script):
        analyze_script = os.path.join(script_dir, "analyze_zib_distribution.py")
    input_abs = os.path.abspath(args.input)
    input_root = os.path.splitext(os.path.basename(args.input))[0]
    profile_path = args.profile
    is_auto = False
    if not profile_path:
        profile_path = os.path.join(script_dir, f"{input_root}_distribution_profile.json")
        is_auto = True
    should_run_analysis = is_auto or not os.path.exists(profile_path)
    if should_run_analysis:
        if os.path.exists(analyze_script):
            print("[*] Executing distribution analysis:")
            print(f"    Script: {analyze_script}")
            subprocess.run(
                [
                    sys.executable,
                    analyze_script,
                    "--input",
                    input_abs,
                    "--output",
                    profile_path,
                ],
                check=True,
            )
        else:
            print(f"[*] Warning: Analysis script NOT found: {analyze_script}")

    model_profile = {}
    if os.path.exists(profile_path):
        print(f"[*] Loading Analysis Data: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            model_profile = profile_data.get("layers", profile_data)

    is_zanime_profile_flag = bool(model_profile) and zib._is_zanime_profile(
        model_profile
    )
    if is_zanime_profile_flag:
        n_before = len(model_profile)
        model_profile = zib._convert_zanime_profile_to_nextdit(model_profile)
        print(
            f"  [Z-Anime profile bridge] entries: {n_before} -> {len(model_profile)}"
        )

    alpha, beta, get_layer_search_low, hard_veto_layers = zib.derive_hswq_strategy(
        model_profile,
        is_zanime=is_zanime_profile_flag,
        use_bf16_calibration=is_zanime_profile_flag,
    )
    (
        model,
        original_state_dict,
        stripped_state_dict,
        zit_config,
        detected_prefix,
        is_zanime,
        zanime_reverse_map,
        inference_dtype,
    ) = zib.load_zit_model(args.input, device, args.comfy_path)

    # --- Autonomous VETO (same as ZIB v2.0) ---
    if is_zanime:
        structural_veto = zib._compute_structural_veto(model, hard_veto_layers)
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(
                f"  [Z-Anime Structural VETO] +{len(structural_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        proj_veto = zib._compute_per_projection_qkv_veto(
            model, hard_veto_layers, zib._QKV_PROJ_VETO_THRESH_ZANIME
        )
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(
                f"  [Z-Anime Per-Projection VETO] +{len(proj_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
    else:
        print("  [V2.0 Autonomous VETO] Structural + per-projection qkv + key-pattern.")
        structural_veto = zib._compute_structural_veto(model, hard_veto_layers)
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(
                f"  [Structural VETO] +{len(structural_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        proj_veto = zib._compute_per_projection_qkv_veto(
            model, hard_veto_layers, zib._QKV_PROJ_VETO_THRESH_DEFAULT
        )
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(
                f"  [Per-Projection VETO] +{len(proj_veto)} "
                f"(total {len(hard_veto_layers)})"
            )
        keypattern_veto = zib._compute_nextdit_keypattern_veto(
            model, hard_veto_layers
        )
        if keypattern_veto:
            hard_veto_layers = hard_veto_layers.union(keypattern_veto)
            print(f"  [Key-Pattern VETO] hard_veto total: {len(hard_veto_layers)}")

    pipeline = zib.ZITCalibrationPipeline(
        model, text_encoder, tokenizer, device, dtype=inference_dtype
    )

    # Patch DualMonitor for Card 1 signed means (do not alter zib file on disk).
    zib.dual_monitors.clear()
    dual_monitors = zib.dual_monitors

    def hook_fn_int8(module, input, output, name):
        if name not in dual_monitors:
            dual_monitors[name] = DualMonitorInt8()
        dual_monitors[name].update(input[0], output)

    print("Preparing calibration (Dual Monitor hooks; Card 1 act means)...")
    handles, target_modules = [], []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn_int8(m, i, o, n)
            )
            handles.append(handle)
            target_modules.append(name)

    with open(args.calib_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if len(prompts) < args.num_calib_samples:
        prompts = (prompts * (args.num_calib_samples // max(len(prompts), 1) + 1))[
            : args.num_calib_samples
        ]
    else:
        prompts = prompts[: args.num_calib_samples]

    print(
        f"Running calibration ({args.num_calib_samples} samples, "
        f"{args.num_inference_steps} steps)..."
    )
    for i, prompt in enumerate(prompts):
        print(f"\nSample {i+1}/{args.num_calib_samples}: {prompt[:50]}...")
        with torch.no_grad():
            pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps)
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in handles:
        h.remove()

    _norm_profile = {}
    for _pk, _pv in model_profile.items():
        if isinstance(_pv, dict):
            _stripped = _pk
            for _pfx in zib.ZIT_PREFIXES:
                if _pfx and _stripped.startswith(_pfx):
                    _stripped = _stripped[len(_pfx):]
                    break
            if _stripped.endswith(".weight"):
                _stripped = _stripped[:-7]
            _norm_profile[_stripped] = _pv

    if not is_zanime:
        _supp = zib._autonomous_supplemental_veto(
            model, hard_veto_layers, _norm_profile
        )
        if _supp:
            hard_veto_layers = hard_veto_layers.union(_supp)
            print(
                f"  [Supplemental VETO] +{len(_supp)} "
                f"(total {len(hard_veto_layers)})"
            )

    # --- Per-checkpoint auto analysis → auto-optimal FP16 inside 300 MiB ---
    # DualMonitor refresh α/β (never keep pre-calib stale mix). No keep_ratio.
    if not _norm_profile:
        raise ValueError(
            "ZI INT8 FP16 budget requires THIS-checkpoint layer profile "
            "(auto analysis → derive_int8_autonomous_tunables). "
            "Run analyze / supply --profile before quantize."
        )
    veto_tunables = hswq_int8.resolve_veto_tunables(
        _norm_profile,
        dual_monitors=dual_monitors,
        fp16_budget_mb=float(args.fp16_budget_mb),
    )
    alpha = float(veto_tunables.alpha_auto)
    if alpha <= 0.0:
        raise ValueError(
            "INT8 Full-SVD×RMS alpha_auto must be > 0 after DualMonitor resolve "
            f"(alpha==0 is SVD cut / rebellion). got alpha_auto={alpha}"
        )
    beta = 1.0 - alpha
    print(
        f"  [Dynamic Alpha/Beta INT8 after DualMonitor] "
        f"alpha={alpha!r}, beta={beta!r} "
        f"(THIS analyze character → Full-SVD×RMS; Imp×Sens×V4 MSE fill "
        f"{float(args.fp16_budget_mb):g} MiB)"
    )
    if _bc_top_override is None:
        args.bias_correction_top_ratio = float(
            veto_tunables.bias_correction_top_ratio
        )
        print(
            f"  [Autonomous bias_correction_top_ratio after DualMonitor] "
            f"{args.bias_correction_top_ratio!r}"
        )
    else:
        args.bias_correction_top_ratio = float(_bc_top_override)

    mse_cache: dict = {}
    dynamic_keep_layers, mse_cache = _v4_score_all_fp16_candidates(
        hswq_int8=hswq_int8,
        model=model,
        dual_monitors=dual_monitors,
        target_modules=target_modules,
        hard_veto_layers=hard_veto_layers,
        alpha=alpha,
        beta=beta,
        device=device,
        mse_cache=mse_cache,
    )
    # FULL union — budget only truncates (Hard VETO may demote if over frame).
    keep_layers = dynamic_keep_layers.union(hard_veto_layers)
    keep_layers, hard_veto_layers, budget_stats = hswq_int8._apply_fp16_budget_cap(
        model=model,
        keep_layers=keep_layers,
        hard_veto_layers=hard_veto_layers,
        budget_mb=float(args.fp16_budget_mb),
        norm_profile=_norm_profile,
        veto_tunables=veto_tunables,
        dual_monitors=dual_monitors,
        mse_cache=mse_cache,
        alpha=alpha,
        beta=beta,
        device=device,
    )
    # get_layer_search_low unused for INT8 pack (absmax); kept for parity logging
    _ = get_layer_search_low
    _ = mse_cache

    act_mean_dict = {}
    sens_dict = {}
    for name, mon in dual_monitors.items():
        if getattr(mon, "channel_act_mean", None) is not None:
            act_mean_dict[name] = mon.channel_act_mean.detach().float().cpu()
        sens_dict[name] = float(mon.get_sensitivity())
    print(
        f"  [Card 1 DualMonitor] act_mean={len(act_mean_dict)} "
        f"sens={len(sens_dict)}"
    )

    keep_dtype = torch.bfloat16 if is_zanime else torch.float16
    mode = "per-channel (Card 3)" if args.per_channel_int8 else "tensorwise"
    print(f"\nConverting to INT8 ({mode}) | FP16/BF16 keep={len(keep_layers)}")

    # Card 1 scope among INT8 layers
    bc_allowed = None
    if args.bias_correction:
        int8_candidates = [
            n for n in target_modules if n not in keep_layers
        ]
        top_ratio = float(args.bias_correction_top_ratio)
        top_ratio = 0.0 if top_ratio < 0.0 else (1.0 if top_ratio > 1.0 else top_ratio)
        ranked = sorted(
            int8_candidates,
            key=lambda n: sens_dict.get(n, 0.0),
            reverse=True,
        )
        n_bc = int(len(ranked) * top_ratio + 1e-9)
        if top_ratio > 0.0 and n_bc < 1 and ranked:
            n_bc = 1
        if top_ratio >= 1.0:
            bc_allowed = None
            print(
                f"  [Bias Correction] scope=ALL {len(ranked)} INT8 layers "
                f"(top_ratio=1.0)."
            )
        else:
            bc_allowed = set(ranked[:n_bc])
            print(
                f"  [Bias Correction] top {n_bc}/{len(ranked)} by sensitivity "
                f"(top_ratio={top_ratio:.3f})."
            )

    bias_corr_pending: dict[str, torch.Tensor] = {}
    bias_corr_applied = 0
    bias_corr_skipped_no_bias = 0
    bias_corr_skipped_no_act = 0
    bias_corr_skipped_low_sens = 0
    bias_corr_skipped_bad_shape = 0
    converted_count = 0
    kept_count = 0
    output_state_dict = {}
    quant_meta_layers = {}

    for stripped_key, value in tqdm(stripped_state_dict.items(), desc="Converting"):
        module_name = (
            stripped_key[:-7] if stripped_key.endswith(".weight") else None
        )
        is_matmul_weight = (
            module_name is not None
            and value.ndim >= 2
            and value.dtype
            in (torch.float16, torch.float32, torch.bfloat16)
        )

        # Z-Anime keep qkv: split to Diffusers projections (ZI format)
        if (
            is_zanime
            and module_name
            and module_name in keep_layers
            and module_name.endswith(".attention.qkv")
        ):
            base = module_name[: -len(".qkv")]
            chunks = torch.chunk(value.to(keep_dtype), 3, dim=0)
            for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                output_state_dict[
                    f"{detected_prefix}{base}.{tag}.weight"
                ] = chunk.contiguous().clone()
            kept_count += 1
            continue

        if module_name and module_name in keep_layers and is_matmul_weight:
            output_state_dict[detected_prefix + stripped_key] = value.to(keep_dtype)
            kept_count += 1
            continue

        if is_matmul_weight and module_name and module_name not in keep_layers:
            # Z-Anime fused qkv → split INT8 to_q/to_k/to_v
            if is_zanime and module_name.endswith(".attention.qkv"):
                base = module_name[: -len(".qkv")]
                chunks = torch.chunk(value, 3, dim=0)
                for tag, chunk in zip(("to_q", "to_k", "to_v"), chunks):
                    if args.per_channel_int8 and chunk.ndim in (2, 4):
                        q, scale = pack_channelwise(chunk.contiguous())
                    else:
                        q, scale = pack_tensorwise(chunk.contiguous())
                    tgt_module = f"{base}.{tag}"
                    tgt_key = f"{detected_prefix}{tgt_module}.weight"
                    output_state_dict[tgt_key] = q
                    _emit_int8_meta(
                        output_state_dict,
                        f"{detected_prefix}{tgt_module}",
                        scale,
                    )
                    quant_meta_layers[f"{detected_prefix}{tgt_module}"] = {
                        "format": "int8_tensorwise"
                    }
                    converted_count += 1
                    if args.bias_correction:
                        proj_name = f"{base}.{tag}"
                        # Bias for Diffusers projections rarely present on qkv path;
                        # still attempt if act_mean exists under fused name for q only.
                        act = act_mean_dict.get(module_name)
                        if (
                            bc_allowed is not None
                            and module_name not in bc_allowed
                        ):
                            bias_corr_skipped_low_sens += 1
                        elif act is None:
                            bias_corr_skipped_no_act += 1
                        else:
                            w_dq = q.float() * (
                                scale
                                if scale.ndim > 0
                                else scale
                            )
                            # Per-chunk Linear share: scale act to chunk in-dim
                            if act.numel() == chunk.shape[1]:
                                delta = compute_int8_bias_delta(
                                    chunk, w_dq, act
                                )
                                if delta is None:
                                    bias_corr_skipped_bad_shape += 1
                                else:
                                    bias_corr_pending[
                                        f"{detected_prefix}{tgt_module}"
                                    ] = (-delta).detach().float().cpu()
                continue

            # Standard Linear/Conv INT8
            if args.per_channel_int8 and value.ndim in (2, 4):
                q, scale = pack_channelwise(value)
            elif args.per_channel_int8:
                output_state_dict[detected_prefix + stripped_key] = value.to(
                    keep_dtype
                )
                kept_count += 1
                continue
            else:
                q, scale = pack_tensorwise(value)
            weight_dq = q.float() * scale
            out_key = detected_prefix + stripped_key
            prefixed_module = detected_prefix + module_name
            output_state_dict[out_key] = q
            _emit_int8_meta(output_state_dict, prefixed_module, scale)
            quant_meta_layers[prefixed_module] = {"format": "int8_tensorwise"}
            converted_count += 1

            if args.bias_correction:
                if bc_allowed is not None and module_name not in bc_allowed:
                    bias_corr_skipped_low_sens += 1
                else:
                    act = act_mean_dict.get(module_name)
                    if act is None:
                        bias_corr_skipped_no_act += 1
                    else:
                        delta = compute_int8_bias_delta(value, weight_dq, act)
                        if delta is None:
                            bias_corr_skipped_bad_shape += 1
                        else:
                            bias_corr_pending[prefixed_module] = (
                                (-delta).detach().float().cpu()
                            )
            continue

        # Passthrough (norms, embeds, biases, etc.)
        if is_zanime:
            new_value = (
                value.to(torch.bfloat16)
                if value.dtype != torch.bfloat16
                else value
            )
        else:
            new_value = (
                value.to(torch.float16)
                if value.dtype == torch.bfloat16
                else value
            )
        output_state_dict[detected_prefix + stripped_key] = new_value

    if args.bias_correction and bias_corr_pending:
        print(
            f"\n[Bias Correction] Applying deltas to {len(bias_corr_pending)} "
            f"INT8 modules..."
        )
        for module_key, delta in bias_corr_pending.items():
            bias_key = f"{module_key}.bias"
            if bias_key not in output_state_dict:
                bias_corr_skipped_no_bias += 1
                continue
            bias = output_state_dict[bias_key]
            corrected = bias.float() + delta.to(
                device=bias.device, dtype=torch.float32
            )
            output_state_dict[bias_key] = corrected.to(dtype=bias.dtype)
            bias_corr_applied += 1
        print(
            f"  [Bias Correction] applied={bias_corr_applied}, "
            f"no_bias={bias_corr_skipped_no_bias}, "
            f"no_act={bias_corr_skipped_no_act}, "
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape}"
        )
    elif args.bias_correction:
        print(
            f"  [Bias Correction] No deltas pending "
            f"(no_act={bias_corr_skipped_no_act}, "
            f"low_sens={bias_corr_skipped_low_sens}, "
            f"bad_shape={bias_corr_skipped_bad_shape})"
        )

    if is_zanime:
        before_n = len(output_state_dict)
        output_state_dict = zib._denormalize_zanime_output(
            output_state_dict, zanime_reverse_map
        )
        print(
            f"  [Z-Anime] Diffusers key restoration: "
            f"{before_n} -> {len(output_state_dict)} keys."
        )

    metadata = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "quant": "int8_tensorwise",
                "engine": "quantize_zi_int8_hswq_v1.0",
                "fp16_budget_mb": float(args.fp16_budget_mb),
                "fp16_budget_used_mb": float(
                    budget_stats.get("used_mb", 0.0)
                    if isinstance(budget_stats, dict)
                    else 0.0
                ),
                "fp16_priority_form": (
                    budget_stats.get("priority_form")
                    if isinstance(budget_stats, dict)
                    else None
                ),
                "n_fp16_keep": int(len(keep_layers)),
                "per_channel_int8": bool(args.per_channel_int8),
                "bias_correction": bool(args.bias_correction),
                "bias_correction_top_ratio": float(
                    args.bias_correction_top_ratio
                    if args.bias_correction_top_ratio is not None
                    else 1.0
                ),
                "layers": quant_meta_layers,
            }
        )
    }

    print(f"Saving: {args.output}")
    used_mb = float(
        budget_stats.get("used_mb", 0.0) if isinstance(budget_stats, dict) else 0.0
    )
    print(
        f"  INT8 layers: {converted_count} | FP16/BF16 keep: {kept_count} | "
        f"FP16 budget {used_mb:.2f}/{float(args.fp16_budget_mb):g} MiB | "
        f"Card3={args.per_channel_int8} | Card1={args.bias_correction} "
        f"(applied={bias_corr_applied})"
    )
    save_file(output_state_dict, args.output, metadata=metadata)
    print("Saved.")


if __name__ == "__main__":
    main()
