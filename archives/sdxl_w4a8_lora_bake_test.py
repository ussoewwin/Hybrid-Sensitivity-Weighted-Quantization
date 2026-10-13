#!/usr/bin/env python3
"""SDXL ConvRot W4A8 LoRA bake verifier (int8-fast style: dequant -> bake -> requant).

Attaches a deterministic synthetic LoRA to the REAL ConvRot W4A8 pack through the
normal ComfyUI LoRA path (comfy.sd.load_lora_for_models + full device load) and
proves the bake actually requantizes the W4A8 tensors:

  [W] layout kept  : baked W4A8 Linear still holds AsymW4A8Int8Layout,
                     baked INT8 Linear still holds TensorWiseINT8Layout
                     (requant happened - not dequant-to-float, not discard)
  [C] weight moved : dequantized weight changed after attach (LoRA delta applied)
  [L] LoRA live    : W4A8 trajectory with LoRA differs from without
                     (delta survives requantization, is not rounded away)
  [F] bake fidelity: baked W4A8+LoRA dequant matches FP16+LoRA dequant
                     (cos >= 0.97), i.e. requant keeps the baked result close
                     to the float-space LoRA result
  [T] parity       : W4A8+LoRA latent trajectory vs FP16+LoRA trajectory
                     (informational final-cos, not a gate)

Covers all three pack layer classes explicitly:
  asym_w4a8_int8  (W4A8 Linear)           -> stock set_weight requant
  int8_tensorwise (shape-critical INT8)   -> stock set_weight requant
  float16 (Conv2d fallback / plain lin)   -> float diff (control)

Patch stack + ComfyUI bootstrap reused from benchmark/sdxl_w4a8_traj_compare.py
(which loads benchmark/sdxl convrot w4a8/comfy_quant_w4a8.py).

Usage:
    python benchmark/sdxl_w4a8_lora_bake_test.py \
        --fp16 <sdxl.safetensors> --w4a8 <sdxl_w4a8.safetensors> \
        --comfy_path <ComfyUI-master> [--steps 4] [--seeds 42] [--lora-rel 0.05]
"""
import argparse
import os
import sys

import torch

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sdxl_w4a8_traj_compare import (  # noqa: E402
    _clear_argv_for_comfy,
    _cos,
    _load_checkpoint,
    _restore_argv,
    apply_w4a8_patches,
    run_trajectory,
    setup_comfy,
)

_PREFIX = "diffusion_model."


def _module_of(model, module_name):
    return model.model.diffusion_model.get_submodule(module_name)


def _float_weight(mod):
    w = getattr(mod, "weight", None)
    if w is None:
        return None
    if hasattr(w, "dequantize"):
        return w.dequantize().float().cpu()
    return w.detach().float().cpu()


def _pick_layers(model, caps):
    """Deterministic layer picks per pack class, spread across block prefixes."""
    picks = {k: [] for k in caps}
    seen = {k: set() for k in picks}
    for name, mod in model.model.diffusion_model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        layout = getattr(w, "_layout_cls", None)
        cls = type(mod).__name__
        if layout == "AsymW4A8Int8Layout":
            bucket = "w4a8"
        elif layout == "TensorWiseINT8Layout":
            bucket = "int8"
        elif cls == "Linear":
            bucket = "fp16_lin"
        elif cls == "Conv2d":
            bucket = "fp16_conv"
        else:
            continue
        if len(picks[bucket]) >= caps[bucket]:
            continue
        prefix = name.split(".")[0]
        if prefix in seen[bucket]:
            continue
        picks[bucket].append(name)
        seen[bucket].add(prefix)
    return picks



def main() -> int:
    ap = argparse.ArgumentParser(
        description="SDXL ConvRot W4A8 LoRA bake verifier (requant path check)"
    )
    ap.add_argument("--fp16", required=True, help="FP16 baseline SDXL checkpoint")
    ap.add_argument("--w4a8", "--quant", dest="w4a8_path", required=True,
                    help="ConvRot W4A8 quantized SDXL checkpoint")
    ap.add_argument("--comfy_path", required=True, help="ComfyUI-master root")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--lora-rel", type=float, default=0.05,
                    help="synthetic LoRA delta norm relative to each weight norm")
    ap.add_argument("--prompt",
                    default="masterpiece, best quality, 1girl, solo, standing, simple background")
    ap.add_argument("--negative", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("No seeds given")
        return 1

    setup_comfy(args.comfy_path)
    apply_w4a8_patches()

    import comfy.sd
    import comfy.model_management as mm
    import comfy.sample as comfy_sample

    results = []

    def report(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        results.append(bool(ok))

    def traj(patcher, pos, neg, lat):
        out = {}
        for s in seeds:
            xs, x0s, final = run_trajectory(
                patcher, pos, neg, lat, seed=s, steps=args.steps,
                cfg=args.cfg, sampler_name="dpmpp_2m", scheduler="karras",
            )
            out[s] = final.detach().float().cpu()
        return out

    def make_latent(patcher):
        lat = {"samples": torch.zeros(
            [1, 4, args.height // 8, args.width // 8],
            device=mm.intermediate_device(), dtype=mm.intermediate_dtype(),
        )}
        lat["samples"] = comfy_sample.fix_empty_latent_channels(
            patcher, lat["samples"], lat.get("downscale_ratio_spacial", None), None
        )
        return lat

    saved_argv = _clear_argv_for_comfy()
    try:
        # ===== W4A8 pack =====
        print("[W4A8] loading checkpoint...")
        w4a8, clip, vae = _load_checkpoint(args.w4a8_path)

        picks = _pick_layers(w4a8, {"w4a8": 3, "int8": 2, "fp16_lin": 1, "fp16_conv": 1})
        print("  picked (module paths inside diffusion_model):")
        for b in ("w4a8", "int8", "fp16_lin", "fp16_conv"):
            for n in picks[b]:
                print(f"    [{b}] {n}")
        if len(picks["w4a8"]) < 3 or len(picks["int8"]) < 2:
            print("[FATAL] insufficient layers picked from W4A8 pack")
            return 1

        all_names = [(n, _module_of(w4a8, n)) for b in picks for n in picks[b]]
        pre = {n: _float_weight(m) for n, m in all_names}

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(args.negative))
        latent = make_latent(w4a8)

        print("[W4A8] base trajectories...")
        base_final = traj(w4a8, positive, negative, latent)

        # build synthetic LoRA from PRE-bake weights
        lora = {}
        for n, m in all_names:
            w = pre[n]
            g = torch.Generator(device="cpu").manual_seed(20261011 + (sum(map(ord, n)) % 65521))
            d = torch.randn(w.shape, generator=g)
            d = d * (args.lora_rel * w.norm().item() / max(d.norm().item(), 1e-9))
            lora[f"{_PREFIX}{n}.diff"] = d.to(torch.float16)
        print(f"[LoRA] attaching {len(lora)} diff tensors (strength 1.0)...")

        w4a8_lora, _ = comfy.sd.load_lora_for_models(w4a8, None, lora, 1.0, 0)
        mm.unload_all_models()
        mm.soft_empty_cache()
        mm.load_models_gpu([w4a8_lora], force_full_load=True)

        baked = {}
        for b in ("w4a8", "int8", "fp16_lin", "fp16_conv"):
            for n in picks[b]:
                mod = _module_of(w4a8_lora, n)
                w = getattr(mod, "weight", None)
                layout = getattr(w, "_layout_cls", None)
                baked[n] = _float_weight(mod)
                if b == "w4a8":
                    report(f"[W] W4A8 layout kept", layout == "AsymW4A8Int8Layout",
                           f"{n} layout={layout}")
                elif b == "int8":
                    report(f"[W] INT8 layout kept", layout == "TensorWiseINT8Layout",
                           f"{n} layout={layout}")
                maxd = float((baked[n] - pre[n]).abs().max().item())
                relmax = maxd / max(float(pre[n].abs().max().item()), 1e-9)
                report(f"[C] weight moved ({b})", maxd > 1e-6 and relmax > 0.005,
                       f"{n} max|dW|={maxd:.4f} rel={relmax * 100:.1f}%")

        print("[W4A8] lora trajectories...")
        lora_final = traj(w4a8_lora, positive, negative, latent)
        for s in seeds:
            c = _cos(base_final[s], lora_final[s])
            report(f"[L] LoRA live seed{s}", c < 0.999,
                   f"base-vs-lora final-cos={c:.5f}")

        del w4a8_lora, w4a8, clip, vae, base_final
        mm.unload_all_models()
        mm.soft_empty_cache()

        # ===== FP16 + same LoRA (targets) =====
        print("[FP16] loading checkpoint...")
        fp16, clip2, vae2 = _load_checkpoint(args.fp16)
        positive2 = clip2.encode_from_tokens_scheduled(clip2.tokenize(args.prompt))
        negative2 = clip2.encode_from_tokens_scheduled(clip2.tokenize(args.negative))
        latent2 = make_latent(fp16)

        fp16_lora, _ = comfy.sd.load_lora_for_models(fp16, None, lora, 1.0, 0)
        mm.unload_all_models()
        mm.soft_empty_cache()
        mm.load_models_gpu([fp16_lora], force_full_load=True)

        print("[FP16+lora] trajectories...")
        fp16_final = traj(fp16_lora, positive2, negative2, latent2)

        # [F] baked W4A8/INT8 float vs fp16+LoRA float
        for b in ("w4a8", "int8"):
            for n in picks[b]:
                tgt = _float_weight(_module_of(fp16_lora, n))
                c = _cos(baked[n], tgt)
                report(f"[F] bake fidelity ({b})", c >= (0.97 if b == "w4a8" else 0.99),
                       f"{n} cos(baked, fp16+lora)={c:.5f}")

        # [T] parity (informational)
        for s in seeds:
            c = _cos(lora_final[s], fp16_final[s])
            print(f"[INFO][T] parity seed{s}: w4a8+lora vs fp16+lora final-cos={c:.5f}")

        del fp16_lora, fp16, clip2, vae2, fp16_final
        mm.unload_all_models()
    finally:
        _restore_argv(saved_argv)

    n_pass = sum(results)
    print()
    print(f"RESULT: {n_pass}/{len(results)} PASS")
    return 0 if (results and n_pass == len(results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
