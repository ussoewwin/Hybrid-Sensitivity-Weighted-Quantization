# -*- coding: utf-8 -*-
"""ZIT per-layer trajectory impact: inject ONE layer's NVFP4 error, run 4 steps, measure x divergence.

Reverse hybrid NVFP4 method (see md/How to quantize Z Image - Hybrid NVFP4.md):
1. diag_impact.py  -> impact_<model>.json (relative MSE per layer, ascending = safest first)
2. gen_reverse_nvfp4.py -> hybrid nv{K} artifact
3. benchmark/zi_convrot_nvfp4_bench_native.py -> all-5-seed SSIM check (>= 0.97 each)

Usage:
    python Z_Image/diag_impact.py <base_model.safetensors> <sci_1off_artifact.safetensors> <impact_out.json> \
        [--comfy-path <comfyui-root>] [--repo-root <repo-root>]
"""
import argparse
import importlib.util
import json
import os
import sys

import torch


def parse_args():
    ap = argparse.ArgumentParser(description="ZIT per-layer NVFP4 trajectory impact")
    ap.add_argument("base", help="baseline fp16/bf16 NextDiT safetensors")
    ap.add_argument("artifact", help="sci_1off complete ConvRot INT8 safetensors (layer list source)")
    ap.add_argument("out", help="output impact json path")
    ap.add_argument("--comfy-path", default="ComfyUI-master", help="ComfyUI root path")
    ap.add_argument("--repo-root", default=None, help="repo root containing benchmark/ (for the bench module); default = parent of this script dir")
    return ap.parse_args()


def nvfp4_quant_error(w, group=256):
    """per-group e4m3 quant-dequant reconstruction (i.e. NVFP4-quantized weights)."""
    wf = w.float()
    orig = wf.reshape(wf.shape[0], -1)
    k = orig.shape[1]
    n_groups = (k + group - 1) // group
    pad = n_groups * group - k
    if pad:
        orig = torch.nn.functional.pad(orig, (0, pad))
    g = orig.reshape(orig.shape[0], n_groups, group)
    amax = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0
    q = (g / scale).to(torch.float8_e4m3fn).float()
    dq = q * scale
    if pad:
        dq = dq.reshape(orig.shape[0], n_groups, group)[:, :, :k]
    return dq.reshape(w.shape).to(w.dtype)


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
        "bench", os.path.join(repo, "benchmark", "zi_convrot_nvfp4_bench.py")
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    comfy_path = a.comfy_path
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        if os.path.isdir(joined):
            comfy_path = os.path.abspath(joined)
        else:
            comfy_path = os.path.abspath(comfy_path)
    else:
        comfy_path = os.path.abspath(comfy_path)
    bench.setup_comfy(comfy_path)

    embeds = torch.randn(1, 256, 2560, device=device, dtype=torch.float16)

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model, _, _ = bench.load_zit_model(a.base, device, comfy_path, is_nvfp4=False)
    model.eval()
    mods = {n: m for n, m in model.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    from safetensors import safe_open
    artifact = bench.resolve_path(a.artifact, is_file=True)
    if not os.path.isfile(artifact):
        nested = os.path.join(repo, os.path.basename(a.artifact))
        if os.path.isfile(nested):
            artifact = os.path.abspath(nested)
            print(f"  Found: {artifact}", flush=True)
    if not os.path.isfile(artifact):
        raise FileNotFoundError(
            f"artifact not found: {a.artifact!r} (resolved {artifact!r})"
        )
    with safe_open(artifact, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
        layers = list(meta["layers"].keys())
    print(f"layers to measure: {len(layers)}", flush=True)

    def run4():
        x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16,
                        generator=torch.Generator(device).manual_seed(42))
        sigmas = torch.linspace(1.0, 0.0, 5, device=device)
        with torch.no_grad():
            for step in range(4):
                out = model(x, sigmas[step:step + 1], embeds, None, attention_mask=None)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.float16)
        return x

    print("[*] pristine run", flush=True)
    x_ref = run4()
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
            x_t = run4()
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
