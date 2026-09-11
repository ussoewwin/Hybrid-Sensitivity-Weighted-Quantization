# -*- coding: utf-8 -*-
"""ZIT per-layer trajectory impact: inject ONE layer's NVFP4 error, run 4 steps, measure x divergence.

Reverse hybrid NVFP4 method (see md/How to quantize Z Image - Hybrid NVFP4.md):
1. diag_impact.py  -> impact_<model>.json (relative MSE per layer, ascending = safest first)
2. gen_reverse_nvfp4.py -> hybrid nv{K} artifact
3. benchmark/zi_convrot_nvfp4_bench_v2.py -> all-5-seed SSIM check (>= 0.97 each)

Usage:
    python Z_Image/diag_impact.py <base_model.safetensors> <sci_1off_artifact.safetensors> <impact_out.json> \
        [--comfy-path <comfyui-root>] [--repo-root <repo-root>]
"""
import argparse
import json
import os
import sys

import torch
from pathlib import Path
import re
import types
from safetensors.torch import load_file


# --- inlined helper definitions (no external bench import) ---

def _decode_comfy_quant_blob(blob) -> dict | None:
    """Decode uint8 comfy_quant tensor / bytes to a dict, or None."""
    if blob is None:
        return None
    try:
        if hasattr(blob, "detach"):
            raw = bytes(blob.detach().cpu().tolist())
        elif isinstance(blob, (bytes, bytearray)):
            raw = bytes(blob)
        else:
            raw = bytes(blob)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _fuse_zanime_attention(state_dict):
    new_dict = dict(state_dict)
    prefixes = set()
    for k in list(new_dict.keys()):
        m = re.match(r"^(.+?\.attention)\.to_q\.weight$", k)
        if m:
            prefixes.add(m.group(1))
    for prefix in prefixes:
        kq, kk, kv = f"{prefix}.to_q.weight", f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
        if kq in new_dict and kk in new_dict and kv in new_dict:
            qkv = torch.cat([new_dict[kq], new_dict[kk], new_dict[kv]], dim=0)
            new_dict[f"{prefix}.qkv.weight"] = qkv
            del new_dict[kq], new_dict[kk], new_dict[kv]
    rename_map = {
        ".attention.to_out.0.weight": ".attention.out.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
    }
    for k in list(new_dict.keys()):
        for src, dst in rename_map.items():
            if k.endswith(src):
                new_dict[k.replace(src, dst)] = new_dict.pop(k)
                break
    return new_dict


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


def count_armed_convrot_linears(model) -> tuple[int, int]:
    """Return (n_armed_convrot, n_linear_modules) on a loaded NextDiT."""
    n_lin = 0
    n_armed = 0
    for _name, mod in model.named_modules():
        if not hasattr(mod, "weight"):
            continue
        # MixedPrecision Linear / ops Linear typically expose in_features.
        if not hasattr(mod, "in_features"):
            continue
        n_lin += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_armed += 1
    return n_armed, n_lin


def count_convrot_comfy_quant_markers(state_dict: dict) -> tuple[int, int]:
    """Count .comfy_quant markers that are nvfp4 / nvfp4+convrot."""
    n_nv = 0
    n_cr = 0
    for k, v in state_dict.items():
        if not k.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant_blob(v)
        if not conf or str(conf.get("format", "")).lower() != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv


def count_convrot_in_quant_metadata(metadata: dict | None) -> tuple[int, int, str]:
    """Return (n_convrot, n_nvfp4, hswq_nvfp4_convrot flag string)."""
    meta = metadata or {}
    flag = str(meta.get("hswq_nvfp4_convrot", "") or "")
    raw = meta.get("_quantization_metadata")
    if not raw:
        return 0, 0, flag
    try:
        qmap = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return 0, 0, flag
    layers = (qmap or {}).get("layers") or {}
    n_nv = 0
    n_cr = 0
    for conf in layers.values():
        if not isinstance(conf, dict):
            continue
        fmt = str(conf.get("format", "")).lower()
        if fmt != "nvfp4":
            continue
        n_nv += 1
        if conf.get("convrot") is True or str(conf.get("convrot", "")).lower() in (
            "1",
            "true",
        ):
            n_cr += 1
    return n_cr, n_nv, flag


def detect_zit_config_from_keys(state_dict):
    state_dict_keys = list(state_dict.keys())
    zit_config = {}
    layer_indices = set()
    for key in state_dict_keys:
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))

    zit_config["num_layers"] = max(layer_indices) + 1 if layer_indices else 30
    if "x_embedder.weight" in state_dict:
        zit_config["hidden_size"] = state_dict["x_embedder.weight"].shape[0]
    elif "all_x_embedder.2-1.weight" in state_dict:
        zit_config["hidden_size"] = state_dict["all_x_embedder.2-1.weight"].shape[0]
    else:
        zit_config["hidden_size"] = 3072

    refiner_indices = set()
    for key in state_dict_keys:
        if key.startswith("context_refiner."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                refiner_indices.add(int(parts[1]))
    zit_config["num_context_refiner"] = max(refiner_indices) + 1 if refiner_indices else 2

    w1_key = "layers.0.feed_forward.w1.weight"
    if w1_key in state_dict:
        # NVFP4 packs K (in_features); out_features (shape[0]) stays logical.
        zit_config["intermediate_size"] = int(state_dict[w1_key].shape[0])
        print(f"  Detected Intermediate Size: {zit_config['intermediate_size']}")
    else:
        zit_config["intermediate_size"] = None

    zit_config["qk_norm"] = any(k.endswith(".attention.q_norm.weight") for k in state_dict_keys)
    if zit_config["qk_norm"]:
        print("  Detected qk_norm=True (q_norm/k_norm weights present)")

    return zit_config


def load_zit_model(path, device="cuda", comfy_path=None, is_nvfp4=False, require_convrot=False):
    if comfy_path:
        p = Path(comfy_path)
        if not p.is_absolute():
            cand = Path(__file__).resolve().parent.parent / comfy_path
            p = cand if cand.is_dir() else Path(comfy_path)
        resolved = str(p.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    from comfy.ldm.lumina.model import NextDiT
    import comfy.ops
    import comfy.utils

    args_path = resolve_path(path, is_file=True)
    print(f"Loading state_dict: {os.path.basename(args_path)}")

    if is_nvfp4:
        # Kitchen NVFP4: inject .comfy_quant from file metadata, then mixed_precision load.
        state_dict, metadata = comfy.utils.load_torch_file(args_path, return_metadata=True)
        n_cr_meta, n_nv_meta, flag = count_convrot_in_quant_metadata(metadata)
        print(
            f"  [CONVROT meta] hswq_nvfp4_convrot={flag!r} "
            f"nvfp4_layers={n_nv_meta} convrot_stamps={n_cr_meta}"
        )
        if require_convrot and n_cr_meta <= 0:
            print(
                "CRITICAL ERROR: This is the ConvRot NVFP4 bench. "
                "Checkpoint has zero convrot stamps in _quantization_metadata."
            )
            print(
                "  Re-convert with native_convert_nvfp4_zi.py (ConvRot ON, default) "
                "to produce e.g. *_nvfp4_convrot.safetensors"
            )
            print(f"  Path: {args_path}")
            sys.exit(1)
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, "", metadata=metadata or {}
        )
        n_cq = sum(1 for k in state_dict if k.endswith(".comfy_quant"))
        n_cr_cq, n_nv_cq = count_convrot_comfy_quant_markers(state_dict)
        print(f"  [NVFP4] convert_old_quants -> {n_cq} .comfy_quant markers")
        print(
            f"  [CONVROT markers] nvfp4={n_nv_cq} with_convrot={n_cr_cq}"
        )
        if require_convrot and n_cr_cq <= 0:
            print(
                "CRITICAL ERROR: After convert_old_quants, no .comfy_quant "
                "markers carry convrot:true. Refusing plain NVFP4."
            )
            sys.exit(1)
    else:
        state_dict = load_file(args_path)

    converted_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, "dtype") and v.dtype == torch.bfloat16:
            converted_dict[k] = v.to(torch.float16)
        else:
            converted_dict[k] = v

    prefixes_to_try = [
        "",
        "model.",
        "model.diffusion_model.",
        "diffusion_model.",
    ]
    best_prefix = ""
    for prefix in prefixes_to_try:
        if prefix == "":
            continue
        if any(k.startswith(prefix) for k in converted_dict.keys()):
            sample_key = f"{prefix}layers.0.attention_norm1.weight"
            if sample_key in converted_dict:
                best_prefix = prefix
                print(f"  [Prefix Detection] Detected prefix: '{prefix}'")
                break

    if best_prefix:
        print(f"  [Prefix Strip] Stripping prefix: '{best_prefix}'")
        stripped_dict = {}
        for k, v in converted_dict.items():
            if k.startswith(best_prefix):
                stripped_dict[k[len(best_prefix) :]] = v
            else:
                # Keep bare .comfy_quant markers from Kitchen metadata (no prefix).
                stripped_dict[k] = v
        converted_dict = stripped_dict

    is_zanime = any(k.startswith("all_x_embedder.2-1") for k in converted_dict.keys())
    if is_zanime:
        print("  [Model Detection] Z-Anime key naming detected. Normalizing...")
        converted_dict = normalize_zanime_keys(converted_dict)

    config = detect_zit_config_from_keys(converted_dict)
    print(
        f"  [Config Detection] hidden_size={config['hidden_size']}, "
        f"layers={config['num_layers']}"
    )

    kwargs = {}
    if config.get("intermediate_size"):
        ratio = config["intermediate_size"] / config["hidden_size"]
        kwargs["ffn_dim_multiplier"] = ratio
        print(
            f"  Calculated FFN Dim Multiplier: {ratio:.4f} "
            f"(Dim: {config['hidden_size']} -> {config['intermediate_size']})"
        )
    if config.get("qk_norm"):
        kwargs["qk_norm"] = True

    def _build_and_load(ops):
        model_local = NextDiT(
            patch_size=2,
            in_channels=16,
            dim=config["hidden_size"],
            n_layers=config["num_layers"],
            n_refiner_layers=config["num_context_refiner"],
            n_heads=config["hidden_size"] // 128,
            n_kv_heads=config["hidden_size"] // 128,
            multiple_of=256,
            norm_eps=1e-5,
            cap_feat_dim=2560,
            z_image_modulation=True,
            pad_tokens_multiple=64,
            device="cpu",
            dtype=torch.float16,
            operations=ops,
            **kwargs,
        )
        try:
            missing_local, unexpected_local = model_local.load_state_dict(
                converted_dict, strict=False, assign=True
            )
        except TypeError:
            print("  [Warning] assign=True unsupported; quantized dtypes may cast.")
            missing_local, unexpected_local = model_local.load_state_dict(
                converted_dict, strict=False
            )
        except RuntimeError as e:
            print(f"  CRITICAL ERROR: Model Size Mismatch. Error: {e}")
            print(f"  Config: {config}")
            sys.exit(1)
        return model_local, missing_local, unexpected_local

    if is_nvfp4:
        print("Using mixed_precision_ops for NVFP4 model load...")
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.float16)
        model, missing, unexpected = _build_and_load(ops)
    else:
        print("Using standard operations for FP16 model load...")
        ops = comfy.ops.disable_weight_init
        model, missing, unexpected = _build_and_load(ops)

    print(
        f"  [Keys] Matched: {len(converted_dict) - len(unexpected)}, "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    if len(missing) > len(list(model.parameters())) * 0.5:
        print(f"  Warning: Many keys still missing. First 5: {list(missing)[:5]}")

    if is_nvfp4:
        model = model.to(device)
        print(f"  Note: NVFP4 model loaded on {device} (mixed_precision / assign=True).")
        n_armed, n_lin = count_armed_convrot_linears(model)
        print(
            f"  [CONVROT armed] Linears with _hswq_nvfp4_convrot: "
            f"{n_armed} / {n_lin}"
        )
        if require_convrot and n_armed <= 0:
            print(
                "CRITICAL ERROR: ConvRot stamps present but zero Linears armed "
                "(_hswq_nvfp4_convrot). Load/parity path failed."
            )
            sys.exit(1)
    else:
        model = model.to(device).to(torch.float16)
        print(f"  Note: FP16 model loaded on {device}.")

    # assign=True shares Parameter storage with converted_dict; drop refs so
    # `del model` actually frees CUDA weights before the next phase.
    n_comfy_quant = sum(
        1 for k in converted_dict if k.endswith(".comfy_quant") or ".comfy_quant" in k
    )
    converted_dict.clear()
    del converted_dict

    model.eval()
    return model, n_comfy_quant, is_zanime


def normalize_zanime_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("all_"):
            new_key = re.sub(r"^all_(.*?)\.2-1", r"\1", new_key)
        normalized[new_key] = value
    return _fuse_zanime_attention(normalized)


def resolve_path(path, is_file=True):
    if not path:
        return None
    if os.path.exists(path):
        return path

    target = os.path.basename(path)
    print(f"  Note: {target} not found at {path}. Searching recursively...")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        root_abs = os.path.abspath(root)
        if "ComfyUI" in root_abs or "node_modules" in root_abs:
            continue
        if is_file and target in files:
            found = os.path.join(root, target)
            print(f"  Found: {found}")
            return found
    return path


def setup_comfy(comfy_path: str) -> None:
    comfy_root = Path(comfy_path).resolve()
    if not comfy_root.is_dir():
        raise FileNotFoundError(f"--comfy_path not found: {comfy_root}")
    sys.path = [str(comfy_root)] + [p for p in sys.path if Path(p).resolve() != comfy_root]
    bench_dir = Path(__file__).resolve().parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    _install_torchaudio_stub()


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

    comfy_path = a.comfy_path
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        if os.path.isdir(joined):
            comfy_path = os.path.abspath(joined)
        else:
            comfy_path = os.path.abspath(comfy_path)
    else:
        comfy_path = os.path.abspath(comfy_path)
    setup_comfy(comfy_path)

    embeds = torch.randn(1, 256, 2560, device=device, dtype=torch.float16)

    a.base = a.base.strip()
    a.artifact = a.artifact.strip()
    a.out = a.out.strip()

    model, _, _ = load_zit_model(a.base, device, comfy_path, is_nvfp4=False)
    model.eval()
    mods = {n: m for n, m in model.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    print(f"modules with weight/in_features: {len(mods)}", flush=True)

    from safetensors import safe_open
    artifact = resolve_path(a.artifact, is_file=True)
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
