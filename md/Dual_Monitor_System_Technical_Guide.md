# Dual Monitor System — Technical Guide

**Sources:** `quantize_sdxl_hswq_v1.3.py`, `quantize_flux_hswq_v1.6.py`, `quantize_zit_hswq_v1.5.py`  
During calibration, the HSWQ pipeline collects two metrics per layer: **Sensitivity** (output variance) and **Importance** (per-channel input activation). This document explains the formulas, implementation, and how they are used for layer selection and weighted histogram MSE.

---

## 1. Overview

| Metric | What it measures | How it is used |
|--------|------------------|----------------|
| **Sensitivity** | Output variance \(\mathrm{Var}(Y)\) of each layer | Sort layers by sensitivity; **top 10–25% kept in FP16** (most critical). |
| **Importance** | Per-channel mean absolute value of **input** \(X\) | Used as **weights** \(I_c\) in the weighted histogram when computing optimal amax. |

Both are collected **simultaneously** in a single calibration forward pass via a single hook per layer (`DualMonitor.update(input_tensor, output_tensor)`).

---

## 2. Sensitivity (Output Variance)

### 2.1 Formula

For each layer, over all calibration forward passes (samples × steps), we accumulate:

- \(\mathit{output\_sum}\) = sum of batch mean of output \(Y\)
- \(\mathit{output\_sq\_sum}\) = sum of batch mean of \(Y^2\)
- \(\mathit{count}\) = number of updates (calls to `update()`)

Then:

$$\mathrm{Var}(Y) = \frac{\mathit{output\_sq\_sum}}{\mathit{count}} - \left( \frac{\mathit{output\_sum}}{\mathit{count}} \right)^2 = E[Y^2] - (E[Y])^2.$$

Layers with **higher variance** are treated as more “sensitive”: corrupting them (e.g. by quantization) tends to hurt image quality more. So we **keep the top N% by sensitivity in FP16** and quantize the rest.

### 2.2 Code (SDXL V1.3)

**Accumulation (per forward):**

```253:264:quantize_sdxl_hswq_v1.3.py
    def update(self, input_tensor, output_tensor):
        with torch.no_grad():
            # 1. Sensitivity Update (Output Variance)
            # output_tensor: (Batch, Channels, H, W) or (Batch, Tokens, Channels)
            
            out_detached = output_tensor.detach().float()  # cast to FP32
            # mean and mean of squares
            batch_mean = out_detached.mean().item()
            batch_sq_mean = (out_detached ** 2).mean().item()
            
            self.output_sum += batch_mean
            self.output_sq_sum += batch_sq_mean
```

**Variance (after calibration):**

```286:291:quantize_sdxl_hswq_v1.3.py
    def get_sensitivity(self):
        # variance = E[X^2] - (E[X])^2
        if self.count == 0: return 0.0
        mean = self.output_sum / self.count
        sq_mean = self.output_sq_sum / self.count
        variance = sq_mean - mean ** 2
        return variance
```

Flux 1.6 and ZIT 1.5 use the same formula (accumulate `output_sum`, `output_sq_sum`, `count`; then `get_sensitivity()` returns `sq_mean - mean**2`).

### 2.3 Usage: Layer Selection

After calibration, layers are sorted by sensitivity **descending**; the top `keep_ratio` (e.g. 10% or 25%) are kept in FP16, the rest are quantized.

**SDXL V1.3:**

```366:378:quantize_sdxl_hswq_v1.3.py
    print("\nRunning layer sensitivity analysis...")
    layer_sensitivities = []
    for name in target_modules:
        if name in dual_monitors:
            sensitivity = dual_monitors[name].get_sensitivity()
            layer_sensitivities.append((name, sensitivity))
    
    # Sort by sensitivity (descending)
    layer_sensitivities.sort(key=lambda x: x[1], reverse=True)
    
    # Top N% to keep in FP16
    num_keep = int(len(layer_sensitivities) * args.keep_ratio)
    keep_layers = set([x[0] for x in layer_sensitivities[:num_keep]])
```

Flux 1.6 and ZIT 1.5 follow the same pattern: build `layer_sensitivities`, sort by sensitivity descending, take top `keep_ratio` into `keep_layers`.

---

## 3. Importance (Input Mean Absolute Value)

### 3.1 Formula

For each layer, at each forward we compute a **per-channel** importance from the **input** tensor \(X\):

$$I_c = \frac{1}{|\Omega|} \sum_{\Omega} |X_{c}|,$$

where \(\Omega\) is the set of dimensions over which we average (batch and spatial/token dimensions), so the result is a vector of shape \((C,)\) for \(C\) input channels. This is the **mean absolute value** of the input per channel.

It is then updated as a **running average** over calibration steps:

$$\mathit{channel\_importance}^{(t+1)} = \frac{\mathit{channel\_importance}^{(t)} \cdot \mathit{count} + \mathit{current\_imp}}{\mathit{count} + 1}.$$

This vector is later passed to the weighted histogram as the importance weights when computing the optimal clipping threshold (amax) for that layer’s weights.

### 3.2 Input Shape Handling (2D / 3D / 4D)

The implementation supports 4D (Conv2d), 3D (Transformer), and **2D** (Linear/embedding) inputs so that importance is always a per-channel vector \((C,)\):

| Shape | Typical use | Reduction | Result shape |
|-------|-----------------|-----------|--------------|
| 4D `(B, C, H, W)` | Conv2d | `mean(dim=(0,2,3))` | \((C,)\) |
| 3D `(B, T, C)` | Transformer | `mean(dim=(0,1))` | \((C,)\) |
| 2D `(B, C)` | Linear, embedding, adaLN | `mean(dim=0)` | \((C,)\) |
| Other | Fallback | — | `(1,)` (uniform) |

**SDXL V1.3 / ZIT 1.5 (explicit branches):**

```266:277:quantize_sdxl_hswq_v1.3.py
            # 2. Importance Update (Input Activation)
            # V1.1: 2D input support
            inp_detached = input_tensor.detach()
            if inp_detached.dim() == 4: # Conv2d: (B, C, H, W)
                current_imp = inp_detached.abs().mean(dim=(0, 2, 3))  # -> (C,)
            elif inp_detached.dim() == 3: # Transformer: (B, T, C)
                current_imp = inp_detached.abs().mean(dim=(0, 1))     # -> (C,)
            elif inp_detached.dim() == 2:  # Linear/embedding: (B, C) e.g. time_embedding
                current_imp = inp_detached.abs().mean(dim=0)          # -> (C,)
            else:
                # 1D or less: fallback (uniform weight); should not occur in practice
                current_imp = torch.ones(1, device=inp_detached.device, dtype=inp_detached.dtype)
```

**Running average:**

```279:284:quantize_sdxl_hswq_v1.3.py
            if self.channel_importance is None:
                self.channel_importance = current_imp
            else:
                self.channel_importance = (self.channel_importance * self.count + current_imp) / (self.count + 1)
            
            self.count += 1
```

Flux 1.6 uses the same 4D/3D/2D logic and running average (compact one-liners). ZIT 1.5 matches SDXL (including 2D support for adaLN_modulation, t_embedder, etc.); see [HSWQ DualMonitor Fix Report](HSWQ_DualMonitor_Fix_Report.md).

### 3.3 Usage: Weighted Histogram

When computing the optimal amax for a layer that is **not** in `keep_layers`, the quantizer calls the HSWQ optimizer with that layer’s weight and, if available, `dual_monitors[name].channel_importance` as the importance vector. That vector is used to build the weighted histogram \(H(i)\) and then to minimize weighted MSE.

**SDXL V1.3:**

```405:416:quantize_sdxl_hswq_v1.3.py
            # Get importance
            importance = None
            if name in dual_monitors:
                importance = dual_monitors[name].channel_importance
            
            # HSWQ: full weighted MSE via module; scaled=False for compatibility
            optimal_amax = hswq_optimizer.compute_optimal_amax(
                module.weight.data, 
                importance,
                scaled=False  # compatibility mode
            )
            weight_amax_dict[name + ".weight"] = optimal_amax
```

**Flux 1.6:** Same idea; for fused modules, importance is taken from the representative module (e.g. `first_fused_module_name`) if available. **ZIT 1.5:** Same as SDXL (importance from `dual_monitors[name].channel_importance` when present).

---

## 4. Hook Registration and Scope

All three scripts register a **forward hook** on every `Conv2d` and `Linear` module. Each hook calls `DualMonitor.update(input[0], output)` so that one `DualMonitor` instance per layer accumulates both sensitivity and importance over the whole calibration.

| Script | Target model | Module types |
|--------|----------------|--------------|
| SDXL V1.3 | `pipeline.unet` | `Conv2d`, `Linear` |
| Flux 1.6 | `pipeline.transformer` | `Conv2d`, `Linear` |
| ZIT 1.5 | `model` (NextDiT) | `Conv2d`, `Linear` |

**SDXL V1.3 (registration):**

```326:334:quantize_sdxl_hswq_v1.3.py
    print("Preparing calibration (registering Dual Monitor hooks)...")
    handles = []
    target_modules = []
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handle = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            handles.append(handle)
            target_modules.append(name)
```

**Hook (SDXL):**

```296:305:quantize_sdxl_hswq_v1.3.py
def hook_fn(module, input, output, name):
    if name not in dual_monitors:
        dual_monitors[name] = DualMonitor()
    
    # input is tuple (tensor, ...)
    inp = input[0]
    # output is tensor
    out = output
    
    dual_monitors[name].update(inp, out)
```

Flux 1.6 and ZIT 1.5 use the same pattern (one `DualMonitor` per layer name, `update(input[0], output)`).

---

## 5. Summary

| Step | What happens |
|------|----------------|
| 1. Calibration | For each sample, run inference with hooks; each Conv2d/Linear calls `DualMonitor.update(inp, out)`. |
| 2. Sensitivity | Per layer: accumulate mean and mean-of-squares of output; then \(\mathrm{Var}(Y) = E[Y^2] - (E[Y])^2\). |
| 3. Importance | Per layer: per-channel mean absolute value of input, running-averaged over updates → `channel_importance` \((C,)\). |
| 4. Layer selection | Sort layers by sensitivity (desc); top `keep_ratio` → FP16; rest → quantized. |
| 5. Quantization | For each quantized layer, run weighted histogram MSE with weight and `channel_importance` (if present) to get optimal amax. |

Together, **Sensitivity** drives which layers stay in FP16, and **Importance** drives how the weighted MSE optimization distributes error across channels when choosing the clipping threshold for the rest.

**See also:** [Weighted Histogram MSE — Technical Guide](Weighted_Histogram_MSE_Technical_Guide.md) (how importance is used inside the histogram), [HSWQ DualMonitor Fix Report](HSWQ_DualMonitor_Fix_Report.md) (2D input support for ZIT and others).
