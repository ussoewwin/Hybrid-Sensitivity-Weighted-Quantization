# Flux1 ConvRot W4A8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ Reverse Hybrid ConvRot W4A8** vs **Native ConvRot W4A8** on the FLUX.1 architecture.

**Source:** `benchmark result/flux1w4a8.txt`  
**Evaluation Script:** `benchmark/flux1_convrot_w4a8_traj_compare.py` (25-seed deterministic trajectory analysis)

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `F1D_unlockV3.1off r170` | Reverse hybrid pack: the K = 170 lowest-impact layers moved to ConvRot W4A8, the remaining converted layers stay ConvRot INT8 |
| `native convrot w4a8` | Native converter output: every eligible layer converted to ConvRot W4A8 |

---

## 1. Summary Comparison (HSWQ Reverse Hybrid W4A8 vs Native ConvRot W4A8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F1D_unlockV3** | r170 | **0.95383** | 0.93006 | **+0.02377** | **1.1161** | 1.7803 | **0/25 (0%)** | 1/25 (4%) | **HSWQ** |

**Winner** = better on both mean cosine and mean MSE.

---

## 2. Detailed Results per Model

### 2.1. F1D_unlockV3.1off (r170)

#### Metric Overview
| Metric / Property | HSWQ Reverse Hybrid W4A8 | Native ConvRot W4A8 | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95383** | 0.93006 | **+0.02377** |
| **Min Final Cosine** (↑ better) | **0.80155** | 0.78136 | **+0.02019** |
| **Max Final Cosine** (↑ better) | **0.99135** | 0.98410 | **+0.00725** |
| **Mean Final Latent MSE** (↓ better) | **1.1161** | 1.7803 | **-0.6642 (37% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 1/25 (4%) | **HSWQ eliminates bifurcations** |
| **Trajectory Verdict** | 11/25 same-image, 14 drifted | 4/25 same-image, 20 drifted, 1 bifurcated | **HSWQ preserves trajectory structure** |
| **Wall time per seed** | 2.97 s | 2.94 s | **+37.0% against the FP16 reference (4.71 s per seed)** |

#### Side-by-Side per Seed (F1D_unlockV3.1off)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97675** | 0.96796 | +0.00879 | **0.8685** | 1.2290 | -0.3605 | drifted | drifted | **HSWQ** |
| **137** | **0.80155** | 0.78136 | +0.02019 | **3.5640** | 4.5410 | -0.9770 | drifted | drifted | **HSWQ** |
| **849** | **0.97373** | 0.97142 | +0.00231 | **1.7080** | 1.5840 | +0.1240 | drifted | drifted | — |
| **2024** | **0.91622** | 0.84532 | +0.07090 | **1.1610** | 2.2850 | -1.1240 | drifted | drifted | **HSWQ** |
| **7391** | **0.97366** | 0.93990 | +0.03376 | **0.7998** | 1.8650 | -1.0652 | drifted | drifted | **HSWQ** |
| **18429** | **0.95980** | 0.88293 | +0.07687 | **0.6067** | 2.0000 | -1.3933 | drifted | drifted | **HSWQ** |
| **53082** | **0.98388** | 0.97878 | +0.00510 | **0.7104** | 0.9295 | -0.2191 | same-image | drifted | **HSWQ** |
| **149206** | **0.96337** | 0.87512 | +0.08825 | **0.8784** | 3.1280 | -2.2496 | drifted | drifted | **HSWQ** |
| **382715** | **0.99135** | 0.97897 | +0.01238 | **0.4348** | 1.0390 | -0.6042 | same-image | drifted | **HSWQ** |
| **826401** | **0.97848** | 0.91392 | +0.06456 | **0.7706** | 3.0610 | -2.2904 | drifted | drifted | **HSWQ** |
| **1938502** | **0.98579** | 0.98004 | +0.00575 | **0.5885** | 0.8111 | -0.2226 | same-image | same-image | **HSWQ** |
| **4710928** | **0.95488** | 0.95779 | -0.00291 | **1.6560** | 1.5870 | +0.0690 | drifted | drifted | Native |
| **8391642** | **0.83609** | 0.83340 | +0.00269 | **2.8820** | 3.0300 | -0.1480 | drifted | drifted | **HSWQ** |
| **15820493** | **0.98748** | 0.95480 | +0.03268 | **0.6754** | 2.3740 | -1.6986 | same-image | drifted | **HSWQ** |
| **36192847** | **0.98957** | 0.97730 | +0.01227 | **0.4190** | 0.9186 | -0.4996 | same-image | drifted | **HSWQ** |
| **71058294** | **0.98959** | 0.98151 | +0.00808 | **0.5244** | 0.9268 | -0.4024 | same-image | same-image | **HSWQ** |
| **128491703** | **0.98998** | 0.97815 | +0.01183 | **0.4379** | 0.9700 | -0.5321 | same-image | drifted | **HSWQ** |
| **285039184** | **0.98464** | 0.98410 | +0.00054 | **1.0110** | 1.0230 | -0.0120 | same-image | same-image | **HSWQ** |
| **491730285** | **0.95715** | 0.94839 | +0.00876 | **1.1020** | 1.4070 | -0.3050 | drifted | drifted | **HSWQ** |
| **762019483** | **0.82570** | 0.79032 | +0.03538 | **2.5270** | 2.9520 | -0.4250 | drifted | bifurcated | **HSWQ** |
| **938174026** | **0.97490** | 0.95993 | +0.01497 | **0.8528** | 1.3760 | -0.5232 | drifted | drifted | **HSWQ** |
| **1409285713** | **0.90547** | 0.83401 | +0.07146 | **1.3290** | 2.6240 | -1.2950 | drifted | drifted | **HSWQ** |
| **2683910547** | **0.98053** | 0.97728 | +0.00325 | **0.7968** | 0.9325 | -0.1357 | same-image | drifted | **HSWQ** |
| **3851729406** | **0.98272** | 0.97818 | +0.00454 | **0.7750** | 0.9878 | -0.2128 | same-image | drifted | **HSWQ** |
| **4195820371** | **0.98248** | 0.98049 | +0.00199 | **0.8233** | 0.9269 | -0.1036 | same-image | same-image | **HSWQ** |

---

## 3. Key Findings and Trajectory Analysis

1. **Mean cosine:** the reverse hybrid pack reaches 0.95383 against 0.93006 for the native W4A8 pack (+0.02377).
2. **Latent MSE:** 1.1161 against 1.7803, about 37% lower error.
3. **Bifurcations:** 0/25 for the reverse hybrid pack against 1/25 for the native pack; same-image seeds 11/25 against 4/25.
4. **Worst case:** the minimum per-seed cosine is 0.80155 (reverse hybrid) against 0.78136 (native).
5. **Speed:** 2.97 s per seed against 4.71 s for the FP16 reference, +37.0%.

---

## 4. Metric Definitions

- **Final Cosine (final-cos):** cosine similarity between the final denoised latent of the FP16 reference and the quantized model. Closer to 1.0 means identical composition, lighting and semantic fidelity.
- **Final MSE (final-mse):** mean squared error of the final latent tensor against the FP16 reference.
- **Max Step Drop (max-drop):** maximum single-step cosine drop between consecutive sampling steps, quantifying sudden trajectory instability.
- **Bifurcated:** max-step-drop > 0.05 on any single step; a sudden trajectory jump into a different picture attractor basin, not gradual degradation.
- **Verdict:**
  - `same-image`: per-seed final cosine ≥ 0.98, producing virtually indistinguishable generation.
  - `drifted (different image)`: gradual, continuous deviation across steps while maintaining coherent compositional structure.
  - `bifurcated @step N`: discontinuous trajectory jump at step N.
- **Setup tags:** `r170` = K = 170 lowest-impact layers moved to ConvRot W4A8 by `Flux1/gen_reverse_w4a8.py`; the rest stay ConvRot INT8. Taken verbatim from each run tag in `flux1w4a8.txt`.
- **Protocol:** 25 fixed seeds from the standard seed set. The resolution, step count, cfg, sampler and scheduler of this run are not written in the score log; the script defaults are 12 steps, 1024×1024, cfg 1.0, euler / simple. See [How to quantize Flux1 - Hybrid ConvRot W4A8](../md/How%20to%20quantize%20Flux1%20-%20Hybrid%20ConvRot%20W4A8.md).
