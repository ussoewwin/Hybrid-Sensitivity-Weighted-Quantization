#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Krea2 INT8 artifact selector: FULL (bf16/fp16) vs native convrot INT8.

Inputs (two):
  1. FULL base bf16/fp16 SingleStreamDiT  (e.g. test.safetensors)
  2. NATIVE convrot INT8 artifact         (e.g. test2.safetensors)

Method (diag_impact trajectory ranking, applied to native INT8):
  1. run the pristine trajectory with the FULL model
  2. for every eligible Linear: dequantize the native INT8 weight from the
     native artifact, inverse-rotate it back into the base domain, inject it
     into the FULL model, re-run the trajectory, and record rel-MSE vs
     pristine  ->  that layer's native-INT8 conversion impact
  3. the --keep N highest-impact Linears are REPLACED by their original bf16
     weights in the output; every other layer keeps its native INT8 weights

Output: a copy of the native INT8 artifact with the --keep N Linears upgraded
to bf16 (their weight_scale / comfy_quant / metadata entries removed).

Usage (one line):
    python Krea2/gen_int8_impact.py test.safetensors test2.safetensors \
        --out test3.safetensors --keep N --comfy-path ComfyUI-master
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import math
import os
import sys
import types

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# ConvRot / Hadamard (same math as hswq_convrot_int8_krea2_v1.5)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_DEFAULT_GROUPSIZE = 256


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized regular Hadamard (power-of-4), symmetric + orthogonal."""
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


def rotate_weight(
    weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Group-wise right-multiply: W @ H^T. Because the regular Hadamard is
    symmetric and orthogonal (H^T H = I), this is ALSO the inverse rotation:
    W_rot @ H = W @ H^T @ H = W. Used both ways here on purpose."""
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


# ---------------------------------------------------------------------------
# ComfyUI bootstrap (same as Krea2/diag_impact.py @ 8/24 baseline)
# ---------------------------------------------------------------------------
def _clear_argv_for_comfy():
    saved = list(sys.argv)
    sys.argv = [saved[0]]
    return saved


def _restore_argv(saved):
    sys.argv = saved


def _install_torchaudio_stub():
    import importlib.machinery
    for key in list(sys.modules):
        if key == "torchaudio" or key.startswith("torchaudio."):
            del sys.modules[key]

    def _stub(name, pkg=False):
        m = types.ModuleType(name)
        m.__file__ = "<hswq_torchaudio_stub>"
        if pkg:
            m.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = []
        else:
            spec = importlib.machinery.ModuleSpec(name, loader=None)
        m.__spec__ = spec
        return m

    ta = _stub("torchaudio", True)
    func = _stub("torchaudio.functional")
    func.resample = lambda w, o, n, *a, **k: w
    tr = _stub("torchaudio.transforms")

    class _MS:
        def __init__(self, *a, **k):
            pass

        def __call__(self, x):
            return x

        def to(self, *args, **kwargs):
            return self

    class _ML:
        def __init__(self, *a, **k):
            pass

    tr.MelSpectrogram = _MS
    tr.MelScale = _ML
    ta.functional = func
    ta.transforms = tr
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = func
    sys.modules["torchaudio.transforms"] = tr


def _install_comfy_stubs():
    _install_torchaudio_stub()
    try:
        import comfy_aimdo  # noqa: F401
    except Exception:
        m = types.ModuleType("comfy_aimdo")
        m.__file__ = "<hswq_comfy_aimdo_stub>"
        m.__path__ = []
        sys.modules["comfy_aimdo"] = m
        sys.modules["comfy_aimdo.filter"] = types.ModuleType("comfy_aimdo.filter")
        sys.modules["comfy_aimdo.filter"].filter_modules = lambda *a, **k: None
    try:
        import psutil  # noqa: F401
    except Exception:
        class _VM:
            total = 64 * 1024 ** 3
            available = 32 * 1024 ** 3

        class _P:
            def memory_info(self):
                return types.SimpleNamespace(rss=0)

            def memory_full_info(self):
                return types.SimpleNamespace(uss=0)

            def cpu_percent(self, interval=None):
                return 0.0

            def num_threads(self):
                return 1

        ps = types.ModuleType("psutil")
        ps.virtual_memory = lambda: _VM()
        ps.Process = lambda: _P()
        sys.modules["psutil"] = ps


def _ensure_comfyui(comfy_path=None):
    """Locate the ComfyUI root. Repository-internal ComfyUI-master ONLY."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.normpath(os.path.join(here, ".."))
    candidates = []
    if comfy_path:
        candidates.append(os.path.abspath(comfy_path))
    candidates.append(os.path.join(repo, "ComfyUI-master"))
    env = os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(env)
    for root in candidates:
        if os.path.isfile(os.path.join(root, "comfy", "ldm", "krea2", "model.py")) \
                and os.path.isfile(os.path.join(root, "comfy", "ops.py")):
            return root
    raise FileNotFoundError(
        "ComfyUI root (needs comfy/ops.py + comfy/ldm/krea2/model.py) not found. "
        "Expected <repo>/ComfyUI-master. Pass --comfy-path or set COMFYUI_PATH."
    )


def _load_comfy_pkg(comfy_root):
    """Import `comfy` EXCLUSIVELY from comfy_root (repo ComfyUI-master)."""
    for key in list(sys.modules):
        if key == "comfy" or key.startswith("comfy."):
            del sys.modules[key]
    if comfy_root in sys.path:
        sys.path.remove(comfy_root)
    sys.path.insert(0, comfy_root)
    mod = importlib.import_module("comfy")
    if os.path.abspath(getattr(mod, "__path__", [None])[0] or "") != \
            os.path.abspath(os.path.join(comfy_root, "comfy")):
        raise ImportError(
            f"comfy resolved outside repo tree: {getattr(mod, '__path__', None)} "
            f"(expected {os.path.join(comfy_root, 'comfy')})"
        )
    return mod


# ---------------------------------------------------------------------------
# Krea2 detect + load (same as diag_impact.py)
# ---------------------------------------------------------------------------
def _find_krea2_key_prefix(keys):
    for prefix in ("model.diffusion_model.", "diffusion_model.", ""):
        if f"{prefix}txtfusion.projector.weight" in keys:
            return prefix
    raise ValueError("Not a Krea2 checkpoint: missing txtfusion.projector.weight")


def detect_krea2_dit_config(sd, prefix):
    head_dim = 128
    fw = sd[f"{prefix}first.weight"]
    features = int(fw.shape[0])
    channels = int(fw.shape[1] // 4)
    br = re.compile(r"^" + re.escape(prefix) + r"blocks\.(\d+)\.")
    layers = 0
    for k in sd:
        m = br.match(k)
        if m:
            layers = max(layers, int(m.group(1)) + 1)
    if layers <= 0:
        raise ValueError("Krea2 detect failed: no blocks.* keys")
    wq = sd[f"{prefix}blocks.0.attn.wq.weight"]
    wk = sd[f"{prefix}blocks.0.attn.wk.weight"]
    txtlayers = int(sd[f"{prefix}txtfusion.projector.weight"].shape[1])
    txtdim = int(sd[f"{prefix}txtfusion.layerwise_blocks.0.prenorm.scale"].shape[0])
    return {
        "image_model": "krea2",
        "features": features,
        "channels": channels,
        "patch": 2,
        "layers": layers,
        "heads": int(wq.shape[0] // head_dim),
        "kvheads": int(wk.shape[0] // head_dim),
        "txtlayers": txtlayers,
        "txtdim": txtdim,
    }


def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Krea2 INT8 artifact selector: compare FULL (bf16/fp16) vs native "
            "convrot INT8 via per-Layer trajectory divergence; restore the "
            "--keep N highest-impact Linears to original bf16."
        )
    )
    ap.add_argument("base", help="FULL baseline bf16/fp16 safetensors (test)")
    ap.add_argument("native", help="NATIVE convrot INT8 safetensors (test2)")
    ap.add_argument("--out", "-o", required=True,
                    help="output INT8 safetensors path (test3)")
    ap.add_argument("--keep", type=int, default=0,
                    help="restore the top-N highest-impact Linears to original "
                         "bf16 in the output (0 = keep native INT8 everywhere)")
    ap.add_argument("--steps", type=int, default=4,
                    help="trajectory denoising steps for impact measurement (default 4)")
    ap.add_argument("--lat", type=int, default=128,
                    help="latent H/W (default 128)")
    ap.add_argument("--seq", type=int, default=256,
                    help="random context token seq length (default 256)")
    ap.add_argument("--seed", type=int, default=42,
                    help="trajectory seed (default 42)")
    ap.add_argument("--comfy-path", default="ComfyUI-master",
                    help="ComfyUI root path (default: ComfyUI-master)")
    return ap.parse_args()


def main():
    a = parse_args()
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("gen_int8_impact requires CUDA.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    keep_n = max(0, int(a.keep))

    # ------------------------------------------------------------------
    # Load the NATIVE convrot INT8 artifact (test2) on CPU.
    # ------------------------------------------------------------------
    print(f"Loading native INT8 artifact: {a.native}")
    native_sd = load_file(a.native)
    with safe_open(a.native, framework="pt") as fh:
        meta_raw = (fh.metadata() or {}).get("_quantization_metadata", "{}")
    meta_layers = json.loads(meta_raw).get("layers", {})
    native_prefix = _find_krea2_key_prefix(native_sd)
    print(f"  native key prefix: {native_prefix!r}")
    print(f"  native quantized layers in metadata: {len(meta_layers)}")

    def native_weight_entry(stripped_name: str):
        """Return (q, scale, convrot_groupsize) for a stripped module name."""
        base_k = f"{native_prefix}{stripped_name}"
        q = native_sd.get(f"{base_k}.weight")
        if q is None:
            return None
        sc = native_sd.get(f"{base_k}.weight_scale")
        entry = meta_layers.get(stripped_name, {})
        gs = int(entry.get("convrot_groupsize", 0) or 0)
        return q, sc, gs

    comfy_root = _ensure_comfyui(a.comfy_path)
    print(f"[Krea2] ComfyUI root: {comfy_root}")

    # ------------------------------------------------------------------
    # Load the FULL bf16/fp16 BASE (test) on CUDA.
    # ------------------------------------------------------------------
    full_sd = load_file(a.base)
    full_prefix = _find_krea2_key_prefix(full_sd)

    saved = _clear_argv_for_comfy()
    try:
        _install_comfy_stubs()
        _load_comfy_pkg(comfy_root)
        try:
            import comfy.options
            comfy.options.enable_args_parsing(False)
        except ImportError:
            pass
        import comfy.ops
        from comfy.ldm.krea2.model import SingleStreamDiT

        cfg = detect_krea2_dit_config(full_sd, full_prefix)
        print(f"Detected Krea2 DiT config: {cfg}")
        kw = {k: v for k, v in cfg.items() if k != "image_model"}
        dit = SingleStreamDiT(
            **kw, device=device, dtype=torch.bfloat16,
            operations=comfy.ops.manual_cast,
        )
        stripped = {}
        for k, v in full_sd.items():
            if full_prefix and k.startswith(full_prefix):
                stripped[k[len(full_prefix):]] = v
            elif not full_prefix:
                stripped[k] = v
        missing, unexpected = dit.load_state_dict(stripped, strict=False)
        print(
            f"  [Krea2] load_state_dict missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
        dev = str(next(dit.parameters()).device)
        if not dev.startswith("cuda"):
            raise RuntimeError(f"Krea2 DiT landed on {dev!r}, not CUDA")
        dit.eval()
        del stripped
        gc.collect()

        txtlayers = int(cfg["txtlayers"])
        txtdim = int(cfg["txtdim"])
        channels = int(cfg["channels"])
        steps = int(a.steps)
        seed = int(a.seed)

        gen_ctx = torch.Generator(device=device).manual_seed(seed)
        context = torch.randn(
            1, int(a.seq), txtlayers * txtdim, device=device, dtype=torch.bfloat16,
            generator=gen_ctx,
        )
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)

        def run():
            x = torch.randn(1, channels, int(a.lat), int(a.lat), device=device,
                            dtype=torch.bfloat16,
                            generator=torch.Generator(device).manual_seed(seed))
            with torch.no_grad():
                for step in range(steps):
                    out = dit(x, sigmas[step:step + 1], context)
                    if isinstance(out, tuple):
                        out = out[0]
                    x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.bfloat16)
            return x

        # 1) pristine trajectory with the FULL model
        print("[*] pristine FULL run", flush=True)
        x_ref = run()
        print("[*] pristine done", flush=True)

        # 2) per-Layer native-INT8 impact:
        #    inject the native INT8 weight (dequantized, inverse-rotated back
        #    into the base domain) and measure rel-MSE vs pristine.
        eligible: dict[str, torch.nn.Module] = {}
        for n, m in dit.named_modules():
            if not hasattr(m, "weight") or not hasattr(m, "in_features"):
                continue
            if m.weight is None or m.weight.data.ndim != 2:
                continue
            if m.weight.data.dtype not in (torch.float16, torch.bfloat16):
                continue
            full_key = f"{full_prefix}{n}.weight" if full_prefix else f"{n}.weight"
            if native_sd.get(f"{native_prefix}{n}.weight") is None:
                continue
            if "bias" in n:
                continue
            eligible[n] = m
        print(
            f"eligible Linears present in the native artifact: {len(eligible)}",
            flush=True,
        )

        impacts: dict[str, float] = {}
        done = 0
        for n, m in eligible.items():
            entry = native_weight_entry(n)
            w0 = m.weight.data.clone()
            try:
                q, sc, gs = entry
                w_rot_dq = q.float() * sc.reshape(-1, 1)
                if gs and gs >= 4:
                    h = build_hadamard(int(gs), device=w_rot_dq.device)
                    w_dq = rotate_weight(w_rot_dq, h, int(gs))
                else:
                    w_dq = w_rot_dq
                m.weight.data.copy_(w_dq.to(w0.dtype))
                x_t = run()
                imp = rel_mse(x_t, x_ref)
            except Exception as e:
                print(f"  ERR {n}: {e}", flush=True)
                imp = float("nan")
            m.weight.data.copy_(w0)
            impacts[n] = imp
            done += 1
            if done % 25 == 0 or done == len(eligible):
                print(f"  [{done}/{len(eligible)}]", flush=True)

        finite = [v for v in impacts.values() if math.isfinite(v)]
        print(
            f"impact measured: {len(finite)}/{len(impacts)} layers  "
            f"max={max(finite) if finite else float('nan'):.3e}  "
            f"mean={sum(finite) / len(finite) if finite else float('nan'):.3e}",
            flush=True,
        )
    finally:
        _restore_argv(saved)

    # 3) ranking: highest native-INT8 impact -> restore original bf16
    ranked = sorted(
        ((k, v) for k, v in impacts.items() if math.isfinite(v)),
        key=lambda kv: kv[1], reverse=True,
    )
    keep_set = {k for k, _ in ranked[:keep_n]}
    if keep_n > 0:
        print(f"[keep] top {keep_n} highest-impact Linears restored to bf16:")
        for k, v in ranked[:keep_n]:
            print(f"  KEEP {k}  impact={v:.3e}")

    # 4) build the output artifact: native INT8 weights everywhere except the
    #    --keep N Linears, which are replaced by the FULL bf16 weights.
    print(f"Building output: {a.out}")
    new_sd = dict(native_sd)
    restored = 0
    for k in sorted(keep_set):
        native_base = f"{native_prefix}{k}"
        full_base = f"{full_prefix}{k}"
        full_w = full_sd.get(f"{full_base}.weight")
        if full_w is None:
            continue
        new_sd[f"{native_base}.weight"] = full_w
        new_sd.pop(f"{native_base}.weight_scale", None)
        new_sd.pop(f"{native_base}.comfy_quant", None)
        meta_layers.pop(k, None)
        restored += 1

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": meta_layers}
        )
    }
    print(
        f"Restored to bf16: {restored}/{keep_n} layers  "
        f"native INT8 layers remaining: {len(meta_layers)}"
    )
    save_file(new_sd, a.out, metadata=metadata)
    print("Done!")


if __name__ == "__main__":
    main()
