# -*- coding: utf-8 -*-
"""Z Image hybrid ConvRot NVFP4: per-layer input_scale calibration (amax method).

Adds the missing activation scales to a hybrid NVFP4 artifact so the W4A4
TensorCore path (scaled_mm_nvfp4 / cuBLAS FP4) can use calibrated per-tensor
act scales instead of placeholder ones / step-0 amax freeze.

Method (mirrors hswq_sdxl_convert_nvfp4_1.0.py input_scale calib):
  - load the BASE fp16/bf16 UNet via the bench loader (same as diag_impact.py)
  - attach forward hooks on the NVFP4 target Linears
  - run N calibration prompts through a fixed-seed 4-step Euler trajectory
    (same trajectory as diag_impact.py: randn x0 seed 42, 4 steps)
  - per layer: Hadamard-rotate the input activations (group size from the
    checkpoint metadata, same as inference rotate_last_dim), then take the
    running absmax over all calib runs
  - write  <layer>.input_scale = max(amax, 1e-12) / (F8_E4M3_MAX * F4_E2M1_MAX)
    as an F32 scalar tensor into a copy of the hybrid artifact

The rotation MUST happen before amax ("rotate first, then amax"): the hybrid
weights are stored already rotated (W @ H^T), so the runtime quantizes rotated
activations. An unrotated amax is in the wrong domain and mis-scales the grid.

Usage (from the clone directory, same as diag_impact.py):
    python Z_Image/calib_input_scale_nvfp4.py \
        "<path-to-unet>/<base>.safetensors" \
        "<path-to-unet>/<model>_hswq_hybrid_nv110_convrot_nvfp4.safetensors" \
        "<path-to-unet>/<model>_hswq_hybrid_nv110_convrot_nvfp4_calib.safetensors" \
        [--comfy-path <path-to-ComfyUI>] [--repo-root <repo-root>] \
        [--prompts <prompts.txt>] [--samples 32] [--device cuda]

After it finishes, bench the calibrated artifact (all-5-seed decoded SSIM)
before using the TC path; the parity loader ignores input_scale, so the
calibrated file stays 100% compatible with the stock parity workflow.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from pathlib import Path
import re
import types


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
    ap = argparse.ArgumentParser(description="Z Image hybrid NVFP4 input_scale calibration")
    ap.add_argument("base", help="baseline fp16/bf16 NextDiT safetensors (same as diag_impact.py input 1)")
    ap.add_argument("hybrid", help="hybrid ConvRot NVFP4 artifact (gen_reverse_nvfp4.py output)")
    ap.add_argument("out", help="output safetensors path (copy of hybrid + input_scale keys)")
    ap.add_argument("--comfy-path", default="ComfyUI-master", help="ComfyUI root path")
    ap.add_argument("--repo-root", default=None, help="repo root containing benchmark/ (default: parent of this dir)")
    ap.add_argument("--prompts", default=None, help="UTF-8 text file, one prompt per line (default: synthetic set)")
    ap.add_argument("--samples", type=int, default=32, help="number of calibration trajectories")
    ap.add_argument("--steps", type=int, default=4, help="number of Euler sampling steps per trajectory (default: 4)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Hadamard rotation (same math as native_convert_int8.build_hadamard /
# rotate_last_dim used at inference; reimplemented locally so this script has
# no repo-internal import dependency beyond the bench loader).
# ---------------------------------------------------------------------------

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sylvester Hadamard matrix of order n (power of 2), normalized: H @ H.T = I."""
    key = int(n)
    h = _HADAMARD_CACHE.get(key)
    if h is not None and h.device == device:
        return h
    h1 = torch.ones(1, 1, device=device, dtype=dtype)
    hm = h1
    while hm.shape[0] < n:
        hm = torch.cat(
            [torch.cat([hm, hm], dim=1), torch.cat([hm, -hm], dim=1)], dim=0
        )
    hm = hm / (n ** 0.5)
    if hm.shape[0] != n:
        raise ValueError(f"Hadamard order {n} is not a power of two")
    _HADAMARD_CACHE[key] = hm
    return hm


def _group_size_for(in_features: int, preferred: int = 256) -> int | None:
    """Largest power-of-4 group <= preferred that divides in_features.

    Mirrors native_convert_int8.convrot_group_size_for_features exactly
    (the rule the converter used when rotating the weights offline), so the
    amax domain matches the weight rotation domain bit-for-bit.
    """
    if in_features < 4:
        return None
    gs = preferred
    while gs >= 4:
        if in_features % gs == 0 and (gs & (gs - 1)) == 0 and _is_pow4(gs):
            return gs
        gs //= 4
    return None


def _is_pow4(n: int) -> bool:
    while n > 1:
        if n % 4:
            return False
        n //= 4
    return n == 1


def _rotate_last_dim(x: torch.Tensor, gs: int) -> torch.Tensor:
    """Hadamard-rotate the last dim in groups of gs. x: (..., in_features) fp32."""
    *lead, last = x.shape
    if last % gs != 0:
        raise ValueError(f"last dim {last} not divisible by group_size={gs}")
    y = x.reshape(*lead, last // gs, gs)
    h = _hadamard(gs, x.device, torch.float32)
    return torch.einsum("...gh,hk->...gk", y, h).reshape(*lead, last)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _default_prompts() -> list[str]:
    """Diverse synthetic prompts (Z-Image caption style) covering content variety."""
    base = [
        "a beautiful cyberpunk city at night, neon lights, high detail",
        "a portrait of a woman with freckles, studio lighting, 85mm",
        "a snowy mountain range at sunrise, crisp air, wide shot",
        "a bowl of ramen on a wooden table, steam, shallow depth of field",
        "an old library with tall shelves and warm lamps",
        "a red sports car on a coastal road at golden hour",
        "a cat sleeping on a windowsill, soft afternoon light",
        "an abstract painting with bold blue and orange strokes",
        "a busy street market with colorful produce stalls",
        "a lone tree in a wheat field under dramatic clouds",
        "a glass of iced coffee with condensation, close-up",
        "a modern minimal living room with plants and wood furniture",
        "a rocket launch at dawn photographed from a distance",
        "a close-up of a mechanical watch movement, macro photography",
        "a foggy forest path with moss covered stones",
        "a chef plating a fine dining dish in a dark kitchen",
    ]
    return base


def main() -> int:
    a = parse_args()
    device = a.device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    repo = os.path.abspath(a.repo_root) if a.repo_root else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    comfy_path = a.comfy_path
    if not os.path.isabs(comfy_path):
        joined = os.path.join(repo, comfy_path)
        comfy_path = os.path.abspath(joined) if os.path.isdir(joined) else os.path.abspath(comfy_path)
    setup_comfy(comfy_path)

    a.base, a.hybrid, a.out = a.base.strip(), a.hybrid.strip(), a.out.strip()

    # 1) Which layers are NVFP4 (from hybrid metadata) + their convrot groupsize
    with safe_open(a.hybrid, framework="pt", device="cpu") as f:
        meta = json.loads(f.metadata()["_quantization_metadata"])
    targets = {
        name[len("model.diffusion_model."):] if name.startswith("model.diffusion_model.") else name: info
        for name, info in meta["layers"].items()
        if info.get("format") == "nvfp4"
    }
    print(f"NVFP4 target layers: {len(targets)}")

    # 2) Load BASE fp16 model (same loader as diag_impact.py; weights NOT quantized)
    model, _, _ = load_zit_model(a.base, device, comfy_path, is_nvfp4=False)
    model.eval()

    mods = {n: m for n, m in model.named_modules()
            if hasattr(m, "weight") and hasattr(m, "in_features")}
    hooks, tracked = [], {}
    for n, info in targets.items():
        if n not in mods:
            print(f"  WARN: target not found as module: {n}")
            continue
        m = mods[n]
        gs = None
        if info.get("convrot"):
            gs = _group_size_for(int(m.in_features), int(info.get("convrot_groupsize", 256)))
            if gs is None:
                print(f"  WARN: no valid Hadamard group for {n} (in={m.in_features}); skipping rotation (unrotated amax)")
        tracked[n] = {"gs": gs, "amax": 0.0}

    def _make_hook(name: str):
        def hook(m, inp, out):
            if not inp or inp[0] is None:
                return
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x):
                return
            x_f = x.detach().reshape(-1, int(m.in_features)).float()
            st = tracked[name]
            if st["gs"]:
                x_f = _rotate_last_dim(x_f, int(st["gs"]))
            amax = float(x_f.abs().amax().clamp_min(1e-12).item())
            if amax > st["amax"]:
                st["amax"] = amax
        return hook

    for n in tracked:
        hooks.append(mods[n].register_forward_hook(_make_hook(n)))
    print(f"hooks attached: {len(hooks)}")

    # 3) Calibration trajectories (fixed-seed 4-step Euler, same as diag_impact.py)
    prompts = _default_prompts()
    if a.prompts:
        with open(a.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < a.samples:
        prompts = (prompts * (a.samples // len(prompts) + 1))[:a.samples]
    else:
        prompts = prompts[:a.samples]
    steps = max(1, int(a.steps))
    print(f"calibrating: {len(prompts)} trajectories x {steps} steps, seed 42")

    # Text embeddings: the bench trajectory (diag_impact.py) uses random embeds
    # with seed 42. Keep the same contract so amax matches the measured regime,
    # but vary the per-sample seed with sample index for coverage.
    def run_trajectory(sample_idx: int):
        g = torch.Generator(device).manual_seed(42 + sample_idx)
        embeds = torch.randn(1, 256, 2560, device=device, dtype=torch.float16, generator=g)
        x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16,
                        generator=torch.Generator(device).manual_seed(42))
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
        with torch.no_grad():
            for step in range(steps):
                out = model(x, sigmas[step:step + 1], embeds, None, attention_mask=None)
                if isinstance(out, tuple):
                    out = out[0]
                x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.float16)
        return x

    for i in range(len(prompts)):
        run_trajectory(i)
        if (i + 1) % 8 == 0 or i + 1 == len(prompts):
            print(f"  [{i + 1}/{len(prompts)}] amax coverage: "
                  f"{sum(1 for v in tracked.values() if v['amax'] > 0)}/{len(tracked)}")
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()

    missing = [n for n, v in tracked.items() if v["amax"] <= 0]
    if missing:
        print(f"WARN: {len(missing)} layers saw no activation: {missing[:5]}...")
        for n in missing:
            del tracked[n]

    # 4) Write input_scale keys into a copy of the hybrid artifact
    from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
    denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
    print(f"input_scale formula: amax / {denom:.0f}")

    sd = load_file(a.hybrid)
    prefix = ""
    for p in ("model.diffusion_model.", "diffusion_model.", ""):
        for k in sd.keys():
            if k.startswith(p) and k.endswith(".weight"):
                prefix = p
                break
        if prefix:
            break

    written = 0
    for n, v in tracked.items():
        n_clean = n[len("model.diffusion_model."):] if n.startswith("model.diffusion_model.") else n
        n_clean = n_clean[len("diffusion_model."):] if n_clean.startswith("diffusion_model.") else n_clean
        full = f"{prefix}{n_clean}"
        sd[f"{full}.input_scale"] = torch.tensor(
            max(v["amax"], 1e-12) / denom, dtype=torch.float32
        )
        written += 1
    save_file(sd, a.out, metadata={"_quantization_metadata": json.dumps(meta)})
    print(f"input_scale written: {written} layers")
    print(f"saved: {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB decimal)")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
