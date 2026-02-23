# SDXL HSWQ SageAttention2 Integration Technical Guide

## Overview

SDXL HSWQ V1.3/V1.6 integrates SageAttention2 to speed up calibration. Enable it with the `--sa2` option (about 10% faster).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  HSWQ Calibration Pipeline                                  │
├─────────────────────────────────────────────────────────────┤
│  1. Detect --sa2 flag                                        │
│  2. Import & enable SageAttention2                           │
│  3. Run calibration loop (SA2 accelerates Attention)         │
│  4. Remove hooks & disable SA2 (restore original SDPA)       │
│  5. Compute quantization parameters (normal path)             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Global state

```python
_sage_attn_available = False  # SA2 import success flag
_original_sdpa = None         # Hold original SDPA function
```

### 2. Import helper

```python
def try_import_sage_attention():
    global _sage_attn_available
    try:
        from sageattention import sageattn
        _sage_attn_available = True
        return True
    except ImportError:
        return False
```

### 3. Monkey-patch enable

`F.scaled_dot_product_attention` is replaced by a wrapper that:

- Saves the original in `_original_sdpa`.
- If `attn_mask is not None` or `is_causal` is true: calls original SDPA with the same arguments (SA2 does not support these).
- Otherwise: calls `sageattn(query, key, value, is_causal=False)`. On any exception, falls back to original SDPA with the same arguments.

```python
def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if attn_mask is not None or is_causal:
        return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    try:
        return sageattn(query, key, value, is_causal=False)
    except Exception:
        return _original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
```

### 4. Restore original SDPA

```python
def disable_sage_attention():
    global _original_sdpa
    if _original_sdpa is not None:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _original_sdpa = None
```

## Fallback behavior

| Condition | Behavior |
|-----------|----------|
| `attn_mask` provided | Fall back to original SDPA |
| `is_causal=True` | Fall back to original SDPA |
| SA2 runtime error | Fall back to original SDPA |
| SA2 not installed | Warn and use standard SDPA |

## Usage

```bash
# V1.3 (standard precision, Fast histogram)
python quantize_sdxl_hswq_v1.3.py \
    --input model.safetensors \
    --output model_fp8.safetensors \
    --calib_file prompts.txt \
    --sa2

# V1.6 (high precision)
python quantize_sdxl_hswq_v1.6.py \
    --input model.safetensors \
    --output model_fp8.safetensors \
    --calib_file prompts.txt \
    --sa2
```

## Technical notes

### Why SA2 is safe during calibration

1. **Goal is statistics**: Calibration only collects weight amax and sensitivity statistics. Small numerical differences from SA2 do not affect the result.
2. **Inference output is discarded**: Generated images are not used; only statistics from intermediate activations are used.
3. **Quantization parameters use normal path**: SA2 is used only during calibration; amax computation runs at full precision.

### SA2 vs FlashAttention2

| Item | SageAttention2 | FlashAttention2 |
|------|----------------|------------------|
| Speed | ~10% faster | Baseline |
| Precision | INT8 quantized (small error) | Full precision |
| Windows | ✅ (woct0rdho fork) | ❌ (community build needed) |
| Ease of use | Simple (pip install) | Hard (build required) |

## Version support

| Script | SA2 (`--sa2`) | Quantization |
|--------|----------------|--------------|
| V1.1 (archives) | ❌ | Standard (bins=4096) |
| V1.2 (archives) | ✅ | Standard (bins=4096, 200 candidates, 3 refinements) |
| **V1.3** | ✅ | Standard + Fast histogram (bins=4096, 200 candidates, 3 refinements) |
| V1.5 (archives) | ❌ | High precision (bins=8192, 1000 candidates, 10 refinements) |
| **V1.6** | ✅ | High precision (bins=8192, 1000 candidates, 10 refinements) |

## Dependencies

```bash
# SageAttention2 install (Windows)
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
pip install triton-windows

# Linux
pip install sageattention triton
```
