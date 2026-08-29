# -*- coding: utf-8 -*-
"""Flux1 per-layer trajectory impact: inject ONE layer's NVFP4 error, run N steps, measure x divergence.

Reverse hybrid NVFP4 method（ZI の Z_Image/diag_impact.py を Flux1 に移植）:
1. diag_impact.py         -> impact_<model>.json（層ごとの relative MSE、昇順 = NVFP4 化しても安全）
2. gen_reverse_nvfp4.py   -> hybrid nv{K} アーティファクト（低影響層を INT8 → NVFP4 に reverse 変換）
3. benchmark/flux1_nvfp4/flux1_convrot_nvfp4_bench.py -> SSIM チェック

Usage:
    python Flux1/diag_impact.py <base_model.safetensors> <all_int8_artifact.safetensors> <impact_out.json> \
        [--comfy-path <comfyui-root>] [--repo-root <repo-root>] [--steps 4] [--seed 42]
"""
import argparse
import importlib.util
import json
import os
import sys

import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Flux1 per-layer NVFP4 trajectory impact")
    ap.add_argument("base", help="baseline fp16/bf16 Flux1 safetensors")
    ap.add_argument("artifact", help="all-INT8 ConvRot safetensors (layer list source)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", default=None, help="ComfyUI root path")
    ap.add_argument("--repo-root", default=None, help="repo root containing benchmark/; default = parent of this script dir")
    ap.add_argument("--steps", type=int, default=4, help="trajectory denoising steps (default 4)")
    ap.add_argument("--seed", type=int, default=42, help="trajectory seed (default 42)")
    ap.add_argument("--latent-size", type=int, default=64, help="latent H/W (default 64; 16GB VRAM 推奨)")
    return ap.parse_args()


def nvfp4_quant_error(w):
    """TRUE NVFP4 quantization error via comfy_kitchen roundtrip
    (E2M1 x 16-element blocks + global scale)."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as _NVFP4
    w2 = w if w.is_contiguous() else w.contiguous()
    qdata, params = _NVFP4.quantize(w2)
    return _NVFP4.dequantize(qdata, params)


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def main():
    a = parse_args()
    device = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = os.path.abspath(a.repo_root) if a.repo_root else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.join(repo, "benchmark"))
    sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(
        "fluxbench", os.path.join(repo, "benchmark", "flux1_nvfp4", "flux_int8_bench.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    comfy_path = a.comfy_path
    if not comfy_path:
        comfy_path = os.path.join(repo, "ComfyUI-master")
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        if os.path.isdir(joined):
            comfy_path = os.path.abspath(joined)
        else:
            comfy_path = os.path.abspath(comfy_path)
    else:
        comfy_path = os.path.abspath(comfy_path)
    bench.setup_comfy(comfy_path)
    bench.apply_int8_patches()

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model = bench._load_diffusion_model(a.base)  # ModelPatcher
    dm = model.model.diffusion_model
    dm.eval()

    # flux の Linear (weight + in_features) モジュール一覧
    mods = {n: m for n, m in dm.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    from safetensors import safe_open
    artifact = os.path.abspath(a.artifact)
    if not os.path.isfile(artifact):
        raise FileNotFoundError(f"artifact not found: {a.artifact!r}")
    with safe_open(artifact, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
        layers = list(meta["layers"].keys())
    print(f"layers to measure: {len(layers)}", flush=True)

    steps = int(a.steps)
    seed = int(a.seed)
    ls = int(a.latent_size)

    txt = torch.randn(1, 512, 4096, device=device, dtype=torch.bfloat16)
    vec = torch.randn(1, 768, device=device, dtype=torch.bfloat16)
    t = torch.full((1,), 1.0, device=device)
    guidance = torch.full((1,), 3.5, device=device)
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def run():
        x = torch.randn(1, 16, ls, ls, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device).manual_seed(seed))
        with torch.no_grad():
            for step in range(steps):
                out = dm(x, t, txt, vec, guidance=guidance)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
        return x

    print("[*] pristine run", flush=True)
    x_ref = run()
    print("[*] pristine done", flush=True)

    impacts = {}
    done = 0
    for n in layers:
        nmod = n[len("model.diffusion_model."):] if n.startswith("model.diffusion_model.") else n
        if nmod not in mods:
            print(f"  SKIP (not a module): {n}", flush=True)
            continue
        m = mods[nmod]
        w0 = m.weight.data.clone()
        m.weight.data.copy_(nvfp4_quant_error(w0))
        try:
            x_t = run()
            imp = rel_mse(x_t, x_ref)
        except Exception as e:
            print(f"  ERR {n}: {e}", flush=True)
            imp = float("nan")
        m.weight.data.copy_(w0)
        impacts[n] = imp
        done += 1
        if done % 25 == 0 or done == len(layers):
            print(f"  [{done}/{len(layers)}]", flush=True)

    xr = x_ref.float().reshape(x_ref.shape[0], -1)
    json.dump({"x_ref_norm": float((xr * xr).sum().item()), "impacts": impacts},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
