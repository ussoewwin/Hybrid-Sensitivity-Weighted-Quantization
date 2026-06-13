"""Simulate V2.0 autonomous VETO coverage on moody V7 (no GPU calibration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_qpath = ROOT / "quantize_zib_hswq_v1.93.py"
_spec = importlib.util.spec_from_file_location("quantize_v193", _qpath)
_qmod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_qmod)

ZIT_PREFIXES = _qmod.ZIT_PREFIXES
_QKV_PROJ_VETO_THRESH_DEFAULT = _qmod._QKV_PROJ_VETO_THRESH_DEFAULT
_W2_OUTLIER_LIVE_THRESH = _qmod._W2_OUTLIER_LIVE_THRESH
_autonomous_supplemental_veto = _qmod._autonomous_supplemental_veto
_collect_mse_release_candidates = _qmod._collect_mse_release_candidates
_compute_nextdit_keypattern_veto = _qmod._compute_nextdit_keypattern_veto
_compute_per_projection_qkv_veto = _qmod._compute_per_projection_qkv_veto
_compute_structural_veto = _qmod._compute_structural_veto
_layer_weight_stats = _qmod._layer_weight_stats
_weight_profile_drift = _qmod._weight_profile_drift
derive_hswq_strategy = _qmod.derive_hswq_strategy

V7_CKPT = ROOT / "moodyRealMix_zitV7.safetensors"
V7_PROF = ROOT / "moodyRealMix_zitV7_distribution_profile.json"
KEEP_RATIO = 0.05


def short(k: str) -> str:
    return k.replace(".weight", "").split("model.diffusion_model.")[-1]


def keypattern_v7(layer_names: list[str]) -> set[str]:
    _boundary = (".cap_embedder.1", ".final_layer.linear", ".x_embedder")
    _suffixes = (
        ".attention.qkv",
        ".feed_forward.w2",
        ".adaLN_modulation",
        ".attention.out",
    )
    out = set()
    for name in layer_names:
        if name.startswith("t_embedder."):
            out.add(name)
            continue
        if name.endswith(_boundary):
            out.add(name)
            continue
        if any(name.endswith(s) for s in _suffixes):
            out.add(name)
    return out


def _make_linear(weight: torch.Tensor) -> torch.nn.Linear:
    """Real nn.Linear so v1.93 veto helpers pass isinstance checks."""
    out_f, in_f = weight.shape
    lin = torch.nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        lin.weight.copy_(weight)
    lin.eval()
    return lin


def load_linear_layers(path: Path) -> dict[str, torch.Tensor]:
    layers = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".weight") and "scale" not in k and "comfy_quant" not in k:
                if ".attention." in k or ".feed_forward." in k or k.endswith(
                    (
                        "x_embedder.weight",
                        "cap_embedder.1.weight",
                        "final_layer.linear.weight",
                        "t_embedder.1.weight",
                        "t_embedder.mlp.0.weight",
                        "t_embedder.mlp.2.weight",
                    )
                ):
                    nk = k
                    for pfx in ZIT_PREFIXES:
                        if pfx and nk.startswith(pfx):
                            nk = nk[len(pfx) :]
                            break
                    layers[nk.replace(".weight", "")] = f.get_tensor(k)
    return {k: v for k, v in layers.items() if v.ndim == 2}


def build_fake_model(linear_layers: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Minimal module tree for veto helpers."""
    root = torch.nn.Module()

    def ensure(path_parts):
        cur = root
        for i, part in enumerate(path_parts):
            if not hasattr(cur, part):
                if i == len(path_parts) - 1:
                    setattr(
                        cur,
                        part,
                        _make_linear(linear_layers[".".join(path_parts)]),
                    )
                else:
                    sub = torch.nn.Module()
                    setattr(cur, part, sub)
                    cur = sub
            else:
                cur = getattr(cur, part)
        return cur

    for name in linear_layers:
        ensure(name.split("."))
    return root


def norm_profile(raw: dict) -> dict:
    out = {}
    for pk, pv in raw.items():
        if not isinstance(pv, dict):
            continue
        sk = pk
        for pfx in ZIT_PREFIXES:
            if pfx and sk.startswith(pfx):
                sk = sk[len(pfx) :]
                break
        if sk.endswith(".weight"):
            sk = sk[:-7]
        out[sk] = pv
    return out


def main():
    prof_raw = json.loads(V7_PROF.read_text(encoding="utf-8"))
    layers_w = load_linear_layers(V7_CKPT)
    model = build_fake_model(layers_w)
    layer_names = sorted(layers_w.keys())

    alpha, beta, get_search_low, hard_veto = derive_hswq_strategy(
        prof_raw.get("layers", prof_raw),
        is_zanime=False,
        use_bf16_calibration=False,
    )
    np = norm_profile(prof_raw.get("layers", prof_raw))

    structural = _compute_structural_veto(model, hard_veto)
    hard2 = hard_veto | structural
    qkv_proj = _compute_per_projection_qkv_veto(model, hard2, _QKV_PROJ_VETO_THRESH_DEFAULT)
    hard3 = hard2 | qkv_proj
    kp_veto = _compute_nextdit_keypattern_veto(model, hard3)
    hard4 = hard3 | kp_veto
    supp = _autonomous_supplemental_veto(model, hard4, np)
    hard5 = hard4 | supp
    mse_cands = _collect_mse_release_candidates(hard5, structural, np, model)
    mse_cands -= kp_veto

    # dynamic keep (pre-MSE release)
    sens = []
    for name in layer_names:
        if name in hard5:
            continue
        p = np.get(name, {})
        k = p.get("kurtosis", 0)
        o = p.get("outlier_ratio", 0)
        m = p.get("abs_max", 0)
        score = k + o * 2.0 + m * 0.5
        w = layers_w[name]
        score += _weight_profile_drift(w, p) * 50.0
        sens.append((name, score))
    sens.sort(key=lambda x: -x[1])
    n_dyn = int(len(sens) * KEEP_RATIO)
    dynamic = {x[0] for x in sens[:n_dyn]}
    keep_pre_mse = hard5 | dynamic

    kp = keypattern_v7(layer_names)

    print("=== V7 V2.0 simulation (pre full MSE trial) ===")
    print("hard_veto(profile):", len(hard_veto))
    print("+ structural:", len(structural), "total", len(hard2))
    print("+ qkv per-proj:", len(qkv_proj), "total", len(hard3))
    print("+ key-pattern:", len(kp_veto), "total", len(hard4))
    print("+ supplemental:", len(supp), "total", len(hard5))
    print("MSE release candidates:", len(mse_cands))
    print("dynamic keep r0.05:", len(dynamic))
    print("keep_layers (pre-MSE):", len(keep_pre_mse))
    print("old keypattern VETO:", len(kp))

    missing_after_kp = sorted(kp - hard4)
    missing = sorted(kp - hard5)
    extra = sorted(hard5 - kp)
    print("\nkeypattern but NOT in V2 after key-pattern step:", len(missing_after_kp))
    print("keypattern but NOT in V2 final hard_veto (risk FP8):", len(missing))
    for x in missing[:25]:
        p = np.get(x, {})
        w = layers_w[x]
        lk, lo, lm = _layer_weight_stats(w)
        print(
            f"  {x}: prof k={p.get('kurtosis',0):.1f} o={p.get('outlier_ratio',0):.1f} m={p.get('abs_max',0)} "
            f"| live k={lk:.1f} o={lo:.1f} m={lm:.2f}"
        )
    if len(missing) > 25:
        print(f"  ... +{len(missing)-25} more")

    print("\nV2 hard_veto but NOT keypattern:", len(extra))
    for x in extra[:15]:
        print(f"  {x}")

    # qkv coverage
    qkv_names = [n for n in layer_names if n.endswith(".attention.qkv")]
    kp_qkv = {n for n in qkv_names if n in kp}
    v2_qkv = {n for n in qkv_names if n in hard5}
    print(f"\nqkv layers total {len(qkv_names)} | keypattern {len(kp_qkv)} | V2 hard {len(v2_qkv)}")
    qkv_miss = sorted(kp_qkv - v2_qkv)
    print("qkv in keypattern but NOT V2 veto:", len(qkv_miss))
    for n in qkv_miss[:10]:
        w = layers_w[n]
        chunks = torch.chunk(w.float(), 3, dim=0)
        amax = [c.abs().max().item() for c in chunks]
        print(f"  {n}: q/k/v amax={amax}")

    w2_names = [n for n in layer_names if n.endswith(".feed_forward.w2")]
    kp_w2 = {n for n in w2_names if n in kp}
    v2_w2 = {n for n in w2_names if n in hard5}
    print(f"\nw2 layers total {len(w2_names)} | keypattern {len(kp_w2)} | V2 hard {len(v2_w2)}")


if __name__ == "__main__":
    main()
