# HSWQ `DualMonitor` 2D Input Support Fix — Full Explanation

## 1. Overview
The `DualMonitor` class, a core component of the HSWQ (Hybrid Sensitivity-Weighted Quantization) algorithm, had a bug where 2D input tensors of shape `(B, C)` were not handled correctly. This fix ensures that input importance is computed properly for layers such as `adaLN_modulation` and embedding layers, which are common in architectures like Z-Image Turbo (ZIT).

## 2. Background and Problem

### Previous Behavior (Before)
The original `DualMonitor` implementation handled input tensors as follows:

*   **4D `(B, C, H, W)`**: For Conv2d layers. Mean over axes `(0, 2, 3)` to obtain per-channel importance `(C,)`.
*   **3D `(B, T, C)`**: For Transformer blocks. Mean over axes `(0, 1)` to obtain per-channel importance `(C,)`.
*   **Other (including 2D)**: Fallback — returned a **scalar value of `1.0`** (uniform weight) for all channels.

### Impact on Z-Image Turbo (NextDiT)
In the ZIT (Lumina-Next) architecture, many Linear layers are used for conditioning and modulation in addition to standard Transformer blocks. These layers take **2D tensors** of shape `(Batch, Channels)`.

*   **adaLN_modulation**: Present in each Transformer block; generates modulation parameters from timestep and caption embeddings. ZIT has **34 such layers**.
*   **t_embedder / cap_embedder / x_embedder**: Input embedding layers; they also process 2D inputs.

With the old implementation, input importance for these layers was not computed and was effectively treated as a uniform `1.0`. This lost the information about which channels matter, and degraded the quality of the importance-weighted optimization (Weighted Histogram MSE) during quantization.

## 3. Fix (After)

Support for 2D input tensors was added to the `DualMonitor.update()` method.

```python
# Logic after fix (excerpt)
inp_detached = input_tensor.detach()
if inp_detached.dim() == 4:     # Conv2d: (B, C, H, W)
    current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
elif inp_detached.dim() == 3:   # Transformer: (B, T, C)
    current_imp = inp_detached.abs().mean(dim=(0, 1))
elif inp_detached.dim() == 2:   # Linear/embedding: (B, C) — added
    current_imp = inp_detached.abs().mean(dim=0)          # -> (C,)
else:
    # 1D or below: fallback
    current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
```

### Technical Details
*   **Operation**: Mean is taken only over the batch dimension `dim=0`.
*   **Result**: An importance vector of shape `(C,)` is obtained, matching the number of input channels `C`.

## 4. Effect and Scope

### Effect on Z-Image Turbo (ZIT)
The following layer types now receive correct importance vectors:

| Layer type | Input shape | Count (ZIT) | Before (Importance) | After (Importance) |
| :--- | :--- | :--- | :--- | :--- |
| **adaLN_modulation** | `(B, 256)` | **34** | ✗ 1.0 (Scalar) | **✓ (256,) Vector** |
| t_embedder.mlp | `(B, 256/1024)` | 2 | ✗ 1.0 (Scalar) | **✓ Vector** |
| cap_embedder | `(B, 2560)` | 1 | ✗ 1.0 (Scalar) | **✓ (2560,) Vector** |
| x_embedder | *Reshaped 2D* | 1 | ✗ 1.0 (Scalar) | **✓ Vector** |
| Transformer Linear | `(B, T, C)` | 168 | ✓ Vector | ✓ Vector (unchanged) |

The **34 `adaLN_modulation` layers** in particular control generation quality; improving their quantization accuracy can improve prompt adherence and image quality.

### Effect on Other Models
*   **SDXL**: The SDXL UNet is mainly Conv2d (4D) and Transformer (3D), but any embedding or projector layers with 2D inputs benefit from this fix as well. There are no adverse effects.

## 5. Files Where the Fix Is Applied

The fix is applied (or recommended) in the following scripts:

1.  **`quantize_zit_hswq_v1.py`** (ZIT V1.1 / HSWQ V1)
    *   Status: **Applied**
2.  `quantize_sdxl_hswq_v1.1.py` (SDXL / HSWQ V1)
    *   Status: Applied (recommended)
3.  `quantize_sdxl_hswq_v2.1_scaled.py` (SDXL / HSWQ V2)
    *   Status: Applied (recommended)

## 6. Conclusion
With this fix, HSWQ treats “input strength (activation magnitude)” as importance consistently across all tensor shapes, making quantization more robust and accurate. The change is especially important for next-generation architectures (NextDiT-style) with complex conditioning and modulation.
