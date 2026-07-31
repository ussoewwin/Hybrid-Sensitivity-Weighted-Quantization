**Tag:** `v1.1`  
**Commit:** `00956f6`  
**Date:** 2026-07-28

This release adds **HSWQ INT8 native loading** for **SeedVR2 Video Upscaler** DiT weights (`int8_tensorwise` + `comfy_quant` / `weight_scale`) via construction-time `comfy.ops.mixed_precision_ops`, so INT8 packs stay quantized through `load_state_dict` instead of expanding to full FP16.

---

## 1. Summary

| Topic | Status |
|-------|--------|
| **Problem** | Post-load Linear replace (GGUF-style) never hits `comfy.ops` `_load_quantized_module` → full INT8→FP16 expand → VRAM savings lost |
| **Fix** | Inject `mixed_precision_ops` when building NaDiT → load with QuantizedTensor path |
| **Scope** | **DiT only** (3B / 7B). VAE remains FP16 |
| **Gate** | `comfy_quant` JSON `format == int8_tensorwise` (not filename heuristics) |
| **Tree** | `seedvr2_videoupscaler/` in this repo (+ `benchmark/seedvr2_int8_bench.py`) |

---

## 2. Why post-load replace is wrong for HSWQ

GGUF-style loaders swap Linears **after** weights are on the module. HSWQ / `comfy_quant` needs ComfyUI’s **state_dict load hooks** (`_load_from_state_dict` → `_load_quantized_module`). Building with plain `torch.nn.Linear` and swapping later skips that path.

---

## 3. Load path (construction-time ops)

1. **Detect** HSWQ INT8 via `checkpoint_is_hswq_int8` (`*.comfy_quant` → `int8_tensorwise`).
2. **Build** DiT under meta with `create_object(..., operations=get_hswq_mixed_precision_ops(fp16))` (YAML unchanged; ops injected only for this path).
3. **Prep before load:** move `comfy_quant` tensors to CPU; patch `factory_kwargs["device"]` to the real device so QuantizedTensor is not stuck on meta.
4. **`load_state_dict`** — markers become QuantizedTensor; other layers stay normal Parameters.
5. Independent of the GGUF branch.

---

## 4. Code surface in this tag

| Path | Role |
|------|------|
| `seedvr2_videoupscaler/src/optimization/int8_native_ops.py` | Detect / `mixed_precision_ops` / comfy_quant→CPU / factory device patch |
| `seedvr2_videoupscaler/src/common/config.py` | `create_object(..., **extra_kwargs)` for `operations=` |
| `seedvr2_videoupscaler/src/core/model_loader.py` | Detect → inject ops → prep → load |
| `seedvr2_videoupscaler/src/utils/model_registry.py` | Register `seedvr2_7b_int8_convrot` / sharp; `resolve_dit_config_folder` uses registry `size` |
| `seedvr2_videoupscaler/src/models/dit_3b/**`, `dit_7b/**` | Propagate `operations` / `ops.Linear` through MLP, embedding, patch, attention, window blocks |
| `benchmark/seedvr2_int8_bench.py` | FP16 vs native INT8 DiT comparison (same loader path) |

---

## 5. Usage notes

- Requires a ComfyUI build that provides `comfy.ops.mixed_precision_ops`.
- Place DiT packs under `models/SEEDVR2/` (or your node’s model dir); select the INT8 safetensors in the node UI or pass paths to the bench script.
- Example weights: `seedvr2_7b_int8_convrot.safetensors`, `seedvr2_7b_sharp_int8_convrot.safetensors`.
- VAE example remains FP16 (e.g. `ema_vae_fp16.safetensors`).

Bench (from a SeedVR2 custom_nodes install or this tree):

```bash
python seedvr2_int8_bench.py \
  --fp16 seedvr2_ema_7b_fp16.safetensors \
  --int8 seedvr2_7b_int8_convrot.safetensors \
  --vae ema_vae_fp16.safetensors
```

---

## 6. Scope of this GitHub Release

Tags commit **`00956f6`** (`feat-seedvr2-hswq-int8-native-ops-in-repo`) on this repository.

**Not included:** uploading binary model weights as release assets (distribute packs separately / Hugging Face if applicable).

---

## 7. Links

- **Repository:** https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization  
- **This release:** https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/releases/tag/v1.1  
- **Commit:** https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/commit/00956f6fe394fb83e5ce44331b76fbffacd71c74  
