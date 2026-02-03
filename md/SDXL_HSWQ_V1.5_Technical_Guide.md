# SDXL HSWQ V1.5 Technical Guide

**High Precision Tuned Edition — ZIT V1.5 Methodology Applied**

---

## Overview

HSWQ V1.5 applies the high-precision tuning methodology proven in Z-Image Turbo (ZIT) V1.5 to SDXL. It keeps the core V1.1 algorithm while significantly improving the accuracy of MSE optimization.

---

## Parameter Comparison

| Parameter | V1.1 | V1.5 | Factor | Effect |
|-----------|------|------|--------|--------|
| **bins** | 4096 | 8192 | 2x | Higher histogram resolution |
| **candidates** | 200 | 1000 | 5x | Denser amax search |
| **iterations** | 3 | 10 | 3.3x | Better convergence |

---

## Parameter Details

### 1. bins: 8192 (histogram bin count)

#### Issue in V1.1
```
bins=4096:
├── Weight range: [-448, +448] (FP8 E4M3 max)
├── Bin width: 896 / 4096 ≈ 0.219
└── Problem: weight distribution captured only in 0.219 units
```

#### Improvement in V1.5
```
bins=8192:
├── Weight range: [-448, +448]
├── Bin width: 896 / 8192 ≈ 0.109
└── Improvement: 2× resolution for accurate weight distribution
```

#### Mathematical meaning

The histogram is a discrete approximation of the weight probability density:

```
P(w) ≈ histogram[bin(w)] / total_count

bin(w) = floor((w - min_val) / bin_width)
```

More bins reduce the integration error in the MSE approximation:

```
MSE = ∫ (w - Q(w))² × P(w) dw
    ≈ Σ (bin_center - Q(bin_center))² × histogram[bin] / total
```

**2× bins → integration error roughly halved**

---

### 2. candidates: 1000 (amax candidate count)

#### Issue in V1.1
```
candidates=200, search_range=(0.5, 1.0):
├── Search range: max_val × [0.5, 1.0]
├── Step: (1.0 - 0.5) × max_val / 200 = 0.0025 × max_val
└── Example: max_val=100 → step=0.25 → coarse search
```

#### Improvement in V1.5
```
candidates=1000, search_range=(0.5, 1.0):
├── Search range: max_val × [0.5, 1.0]
├── Step: (1.0 - 0.5) × max_val / 1000 = 0.0005 × max_val
└── Example: max_val=100 → step=0.05 → fine search
```

#### Grid search visualization

```
V1.1 (200 candidates):
|----|----|----|----|----|----|----|----|
0.50 0.56 0.62 0.69 0.75 0.81 0.87 0.94 1.00

V1.5 (1000 candidates):
|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|
0.50                                      1.00
      ↑ 5× denser grid captures optimal amax
```

**5× candidates → better discovery of optimal amax**

#### Importance of amax search in scaled=False mode

> [!IMPORTANT]
> HSWQ uses `scaled=False` (Unscaled mode), but amax search remains critical.

**Scaled vs Unscaled mode:**

```
Scaled Mode (scaled=True):
  quantized = clamp(w, -amax, amax) / amax * 448
  dequantized = quantized / 448 * amax
  → amax affects both clipping and scaling

Unscaled Mode (scaled=False):
  quantized = clamp(w, -amax, amax) → FP8 cast
  → amax affects only the clipping threshold
```

**Why amax matters in Unscaled mode:**

```
Example weight distribution:
  Most values: in [-50, +50]
  Outliers: -200, +180, etc.

amax=200 (keep outliers):
  ├── Outliers preserved
  └── Problem: FP8's limited precision spread over [-200, +200]
       → lower representation accuracy in main range [-50, +50]

amax=80 (moderate clip):
  ├── Outliers (-200, +180) clipped to ±80
  └── Benefit: FP8 precision concentrated in [-80, +80]
       → better accuracy in main range
       → clip error on outliers < gain from precision

amax=30 (excessive clip):
  ├── Main values [-50, +50] heavily clipped
  └── Problem: large information loss
```

**MSE structure (Unscaled mode):**

```
MSE(amax) = clipping error + quantization error

         │
   MSE   │    ╲
         │     ╲       ／
         │      ╲     ／
         │       ╲   ／
         │        ╲_／ ← optimal amax (precisely found in V1.5)
         │
         └────────────────── amax
              small → large
         
Small amax: clipping error↑, quantization error↓
Large amax: clipping error↓, quantization error↑
```

**V1.1 vs V1.5 accuracy (Unscaled mode):**

```
V1.1 (200 candidates):
  max_val=100 → search step = 0.25
  → If optimal amax is 77.3, only 77.25 or 77.50 available
  → MSE: 0.25 unit residual error

V1.5 (1000 candidates):
  max_val=100 → search step = 0.05
  → If optimal amax is 77.3, 77.30 can be selected
  → MSE: optimized to 0.05 unit
```

> [!NOTE]
> **Conclusion**: Even with `scaled=False`, amax search is important. amax decides where to clip and controls the trade-off of where to concentrate FP8's limited precision. 1000 candidates identify this trade-off more accurately.

---

### 3. iterations: 10 (refinement iterations)

#### Refinement algorithm

Each iteration narrows the search range by 1/4:

```python
for iteration in range(refinement_iterations + 1):
    # Find best amax in current range
    best_amax = search_in_range(low, high, candidates)
    
    # Narrow range for next iteration
    range_width = (high - low) / 4
    low = max(max_val * 0.1, best_amax - range_width)
    high = min(max_val * 1.2, best_amax + range_width)
```

#### V1.1 vs V1.5 convergence

```
Initial range: [50, 100] (max_val=100, range=50)

V1.1 (3 refinements):
  Iter 0: [50, 100] → best=75 → range width=50
  Iter 1: [62.5, 87.5] → best=78 → range width=12.5
  Iter 2: [74.9, 81.1] → best=77 → range width=3.125
  Iter 3: [76.2, 77.8] → final=77.2
  Final precision: ±0.78 (1/4 of range width)

V1.5 (10 refinements):
  Iter 0: [50, 100] → range width=50
  Iter 1: range width=12.5
  Iter 2: range width=3.125
  Iter 3: range width=0.781
  Iter 4: range width=0.195
  Iter 5: range width=0.049
  Iter 6: range width=0.012
  Iter 7: range width=0.003
  Iter 8: range width=0.0008
  Iter 9: range width=0.0002
  Iter 10: range width=0.00005
  Final precision: ±0.00001

Improvement: V1.5 identifies optimal amax with ~78,000× the precision of V1.1
```

**10 iterations → convergence precision improves exponentially**

---

## Increased Compute Cost

| Item | V1.1 | V1.5 | Increase |
|------|------|------|----------|
| Histogram build | O(n × 4096) | O(n × 8192) | 2x |
| Candidate evaluation | O(200 × 4096) | O(1000 × 8192) | 10x |
| Total iterations | 4 | 11 | 2.75x |
| **Estimated runtime** | 1x | **~27x** | - |

> [!WARNING]
> V1.5 runtime is about 27× longer. Large models may take several hours.

---

## Theoretical Basis for Quality Gain

### MSE minimization accuracy

Quantization MSE is defined as:

```
MSE(amax) = E[(W - Q(W, amax))²]

where:
  W = original weight
  Q(W, amax) = decode(cast_FP8(clamp(W, -amax, amax)))
```

This function is non-convex and can have local minima.

**V1.5 improvements:**
1. **Denser grid** (1000 candidates): lower risk of missing the optimum
2. **Higher-resolution histogram** (8192 bins): lower integration error in MSE
3. **Deeper refinement** (10 iterations): better convergence to the true minimum

---

## Recommended Use

| Scenario | Version | Reason |
|----------|---------|--------|
| Development / testing | V1.1 | Fast feedback |
| Production quality critical | **V1.5** | Highest accuracy |
| Limited compute | V1.1 | Lower memory, faster |
| Public / distribution | **V1.5** | Best effort |

---

## Code Diff

```diff
- # Initialize HSWQ optimizer (bins=4096, 200 candidates, 3 refinements)
+ # Initialize HSWQ V1.5 high-precision optimizer (bins=8192, 1000 candidates, 10 refinements)
+ # ZIT V1.5 precision methodology applied
+ print("※ V1.5 High Precision Mode: bins=8192, candidates=1000, iterations=10")
  hswq_optimizer = HSWQWeightedHistogramOptimizer(
-     bins=4096,
-     num_candidates=200,
-     refinement_iterations=3,
+     bins=8192,               # High-res histogram (2× V1.1)
+     num_candidates=1000,     # Dense grid (5× V1.1)
+     refinement_iterations=10, # Deep search (3×+ V1.1)
      device=device
  )
```

---

## Conclusion

HSWQ V1.5 trades higher compute cost for:

1. **2× histogram resolution** → accurate weight distribution
2. **5× search density** → better discovery of optimal amax
3. **~78,000× convergence precision** → closer to the theoretical optimum

The same methodology achieved **SSIM 0.888** in ZIT V1.5; similar quality gains are expected for SDXL.
