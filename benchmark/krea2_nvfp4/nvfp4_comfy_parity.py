"""
Krea2 NVFP4 bench — keep HSWQ load + HSWQ TC forward (VRAM-correct).

Package-local under benchmark/krea2_nvfp4/ only. Never import benchmark/nvfp4/.

After apply_comfy_quant_nvfp4_patches():
  1) Keep HSWQ ``load_nvfp4_linear_module`` (do NOT replace with stock Comfy
     ``_orig_load`` — that path was the VRAM crime: stock + full_precision /
     float×NVFP4 dequant → packed + FP16 dual residency ~27 GB Task Manager).
  2) Linear.forward → KEEP HSWQ TC wrap (act rotate → NVFP4 quant → scaled_mm).
  3) Kitchen addmm/linear/mm float×NVFP4 gap filled by nvfp4_addmm_patch.

ConvRot: TC forward already does online x@H when ``_hswq_nvfp4_convrot``.
"""
from __future__ import annotations

_APPLIED = False


def _load_chain_has_hswq_full_load(load_fn) -> bool:
    """True if any wrapper in the load chain is HSWQ NVFP4 full load."""
    seen = set()
    cur = load_fn
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "_hswq_nvfp4_full_load", False):
            return True
        # INT8 / other wrappers close over previous load as ``original_load``
        nxt = None
        if getattr(cur, "__closure__", None) is not None:
            for n, cell in zip(cur.__code__.co_freevars, cur.__closure__):
                if n in ("_orig_load", "original_load", "patched_load"):
                    try:
                        nxt = cell.cell_contents
                    except ValueError:
                        nxt = None
                    break
        cur = nxt
    return False


def apply_nvfp4_comfy_parity() -> bool:
    """Runtime only. Imports stay inside krea2_nvfp4 (never benchmark/nvfp4/)."""
    global _APPLIED
    from .nvfp4_addmm_patch import register_nvfp4_addmm_handler

    register_nvfp4_addmm_handler()

    if _APPLIED:
        return True

    import comfy.ops as ops

    # --- Load: MUST keep HSWQ nvfp4_load (never stock _orig_load) ---
    if not _load_chain_has_hswq_full_load(ops._load_quantized_module):
        raise RuntimeError(
            "[BENCH] nvfp4 parity: HSWQ load_nvfp4_linear_module missing from "
            "ops._load_quantized_module chain; apply_comfy_quant_nvfp4_patches() "
            "must run first (stock Comfy load destroys VRAM)"
        )

    # --- Forward: KEEP TC wrap (do not unwrap to stock dequant) ---
    mp0 = ops.mixed_precision_ops()
    lin_fwd = mp0.Linear.forward
    if not getattr(lin_fwd, "_hswq_nvfp4_full_forward", False):
        raise RuntimeError(
            "[BENCH] nvfp4 parity: HSWQ TC Linear.forward missing; "
            "apply_comfy_quant_nvfp4_patches() must run first "
            "(stock F.linear dequant destroys VRAM)"
        )

    _APPLIED = True
    print(
        "[BENCH] nvfp4 HSWQ load + HSWQ TC forward: "
        "load=load_nvfp4_linear_module (_hswq_nvfp4 arm); "
        "Linear.forward=TC (act rot → quant → scaled_mm); "
        "addmm/linear/mm float×NVFP4 registered; "
        "stock Comfy load NOT used",
        flush=True,
    )
    return True


def audit_nvfp4_loaded_modules(model) -> dict:
    """Count NVFP4 / HSWQ / full_precision_mm flags after load (VRAM path audit)."""
    root = getattr(model, "model", model)
    n_qf = 0
    n_hswq = 0
    n_fp_mm = 0
    n_convrot = 0
    for mod in root.modules():
        if getattr(mod, "quant_format", None) != "nvfp4":
            continue
        n_qf += 1
        if getattr(mod, "_hswq_nvfp4", False):
            n_hswq += 1
        if getattr(mod, "_full_precision_mm", False):
            n_fp_mm += 1
        if getattr(mod, "_hswq_nvfp4_convrot", False):
            n_convrot += 1
    return {
        "nvfp4_layers": n_qf,
        "hswq_armed": n_hswq,
        "full_precision_mm": n_fp_mm,
        "convrot_armed": n_convrot,
    }


def require_nvfp4_vram_safe_load(model) -> None:
    """Abort if loaded NVFP4 layers are on the dual-residency (dequant) path."""
    from .nvfp4_tc_gate import nvfp4_tc_enabled

    stats = audit_nvfp4_loaded_modules(model)
    print(
        f"  [BENCH] NVFP4 load audit: layers={stats['nvfp4_layers']} "
        f"hswq={stats['hswq_armed']} convrot={stats['convrot_armed']} "
        f"full_precision_mm={stats['full_precision_mm']}",
        flush=True,
    )
    if stats["nvfp4_layers"] == 0:
        raise RuntimeError(
            "[BENCH] NVFP4 load audit: zero quant_format=nvfp4 layers "
            "(wrong ckpt or stock load path)"
        )
    if stats["hswq_armed"] < stats["nvfp4_layers"]:
        raise RuntimeError(
            f"[BENCH] NVFP4 load audit: only {stats['hswq_armed']}/"
            f"{stats['nvfp4_layers']} layers have _hswq_nvfp4 "
            "(stock load — VRAM dual residency)"
        )
    if nvfp4_tc_enabled() and stats["full_precision_mm"] > 0:
        raise RuntimeError(
            f"[BENCH] NVFP4 load audit: {stats['full_precision_mm']} layers still "
            "have _full_precision_mm=True on a TC GPU (stock dequant — VRAM crime)"
        )
