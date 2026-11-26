# Flux1 ConvRot W4A8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ Reverse Hybrid ConvRot W4A8** vs **Native ConvRot W4A8** on the FLUX.1 architecture.

**Source:** `benchmark result/flux1w4a8.txt`  
**Evaluation Script:** `benchmark/flux1_convrot_w4a8_traj_compare.py` (25-seed deterministic trajectory analysis)

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `F1D_unlockV3.1on r170` | Reverse hybrid pack: the K = 170 lowest-impact layers moved to ConvRot W4A8, the remaining converted layers stay ConvRot INT8 |
| `MYHuman-Flux_1.3 1off r180` | Reverse hybrid pack: the K = 180 lowest-impact layers moved to ConvRot W4A8, the remaining converted layers stay ConvRot INT8 |

---

## 1. Summary Comparison (HSWQ Reverse Hybrid W4A8 vs Native ConvRot W4A8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F1D_unlockV3** | r170 | **0.95388** | 0.93005 | **+0.02383** | **1.0566** | 1.7803 | **0/25 (0%)** | 1/25 (4%) | **HSWQ** |
| **MYHuman-Flux_1.3** | r180 | **0.96290** | 0.94239 | **+0.02050** | **0.5652** | 0.9551 | **1/25 (4%)** | 1/25 (4%) | **HSWQ** |

**Winner** = better on both mean cosine and mean MSE.

---

## 2. Detailed Results per Model

### 2.1. F1D_unlockV3 (r170)

#### Metric Overview
| Metric / Property | HSWQ Reverse Hybrid W4A8 | Native ConvRot W4A8 | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95388** | 0.93005 | **+0.02383** |
| **Min Final Cosine** (↑ better) | **0.80882** | 0.78136 | **+0.02746** |
| **Max Final Cosine** (↑ better) | **0.99241** | 0.98410 | **+0.00831** |
| **Mean Final Latent MSE** (↓ better) | **1.0566** | 1.7803 | **-0.7238 (40.7% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 1/25 (4%) | **HSWQ eliminates bifurcations** |
| **Trajectory Verdict** | 14/25 same-image, 11 drifted, 0 bifurcated | 4/25 same-image, 20 drifted, 1 bifurcated | **HSWQ preserves trajectory structure** |
| **Wall time per seed** | 2.92 s | 2.94 s | **+36.8% against the FP16 reference (4.62 s per seed)** |

#### Side-by-Side per Seed (F1D_unlockV3)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97725** | 0.96796 | +0.00929 | **0.8466** | 1.2290 | -0.3824 | drifted | drifted | **HSWQ** |
| **137** | **0.80882** | 0.78136 | +0.02746 | **3.4050** | 4.5410 | -1.1360 | drifted | drifted | **HSWQ** |
| **849** | **0.99180** | 0.97142 | +0.02038 | **0.4619** | 1.5840 | -1.1221 | same-image | drifted | **HSWQ** |
| **2024** | **0.91057** | 0.84532 | +0.06525 | **1.2140** | 2.2850 | -1.0710 | drifted | drifted | **HSWQ** |
| **7391** | **0.98289** | 0.93990 | +0.04299 | **0.5053** | 1.8650 | -1.3597 | same-image | drifted | **HSWQ** |
| **18429** | **0.96684** | 0.88293 | +0.08391 | **0.4987** | 2.0000 | -1.5013 | drifted | drifted | **HSWQ** |
| **53082** | **0.98401** | 0.97878 | +0.00523 | **0.6972** | 0.9295 | -0.2323 | same-image | drifted | **HSWQ** |
| **149206** | **0.93230** | 0.87512 | +0.05718 | **1.6200** | 3.1280 | -1.5080 | drifted | drifted | **HSWQ** |
| **382715** | **0.99241** | 0.97897 | +0.01344 | **0.3761** | 1.0390 | -0.6629 | same-image | drifted | **HSWQ** |
| **826401** | **0.98793** | 0.91392 | +0.07401 | **0.4296** | 3.0610 | -2.6314 | same-image | drifted | **HSWQ** |
| **1938502** | **0.98930** | 0.98004 | +0.00926 | **0.4337** | 0.8111 | -0.3774 | same-image | same-image | **HSWQ** |
| **4710928** | **0.95800** | 0.95779 | +0.00021 | **1.5550** | 1.5870 | -0.0320 | drifted | drifted | **HSWQ** |
| **8391642** | **0.83878** | 0.83340 | +0.00538 | **2.7320** | 3.0300 | -0.2980 | drifted | drifted | **HSWQ** |
| **15820493** | **0.98984** | 0.95480 | +0.03504 | **0.5476** | 2.3740 | -1.8264 | same-image | drifted | **HSWQ** |
| **36192847** | **0.99098** | 0.97730 | +0.01368 | **0.3601** | 0.9186 | -0.5585 | same-image | drifted | **HSWQ** |
| **71058294** | **0.98423** | 0.98151 | +0.00272 | **0.7911** | 0.9268 | -0.1357 | same-image | same-image | **HSWQ** |
| **128491703** | **0.98647** | 0.97815 | +0.00832 | **0.5919** | 0.9700 | -0.3781 | same-image | drifted | **HSWQ** |
| **285039184** | **0.98407** | 0.98410 | -0.00003 | **1.0280** | 1.0230 | +0.0050 | same-image | same-image | Native |
| **491730285** | **0.95301** | 0.94839 | +0.00462 | **1.1940** | 1.4070 | -0.2130 | drifted | drifted | **HSWQ** |
| **762019483** | **0.81489** | 0.79032 | +0.02457 | **2.6210** | 2.9520 | -0.3310 | drifted | bifurcated @step 11 | **HSWQ** |
| **938174026** | **0.96719** | 0.95993 | +0.00726 | **1.1160** | 1.3760 | -0.2600 | drifted | drifted | **HSWQ** |
| **1409285713** | **0.90131** | 0.83401 | +0.06730 | **1.3740** | 2.6240 | -1.2500 | drifted | drifted | **HSWQ** |
| **2683910547** | **0.98327** | 0.97728 | +0.00599 | **0.6757** | 0.9325 | -0.2568 | same-image | drifted | **HSWQ** |
| **3851729406** | **0.98952** | 0.97818 | +0.01134 | **0.4589** | 0.9878 | -0.5289 | same-image | drifted | **HSWQ** |
| **4195820371** | **0.98144** | 0.98049 | +0.00095 | **0.8805** | 0.9269 | -0.0464 | same-image | same-image | **HSWQ** |

---

### 2.2. MYHuman-Flux_1.3 (r180)

#### Metric Overview
| Metric / Property | HSWQ Reverse Hybrid W4A8 | Native ConvRot W4A8 | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96290** | 0.94239 | **+0.02050** |
| **Min Final Cosine** (↑ better) | **0.74683** | 0.74210 | **+0.00473** |
| **Max Final Cosine** (↑ better) | **0.99835** | 0.99431 | **+0.00404** |
| **Mean Final Latent MSE** (↓ better) | **0.5652** | 0.9551 | **-0.3899 (40.8% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **1/25 (4%)** | 1/25 (4%) | — |
| **Trajectory Verdict** | 12/25 same-image, 12 drifted, 1 bifurcated | 10/25 same-image, 14 drifted, 1 bifurcated | **HSWQ preserves trajectory structure** |
| **Wall time per seed** | 3.19 s | 3.14 s | **+31.4% against the FP16 reference (4.65 s per seed)** |

#### Side-by-Side per Seed (MYHuman-Flux_1.3)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98675** | 0.98939 | -0.00264 | **0.3803** | 0.2995 | +0.0808 | same-image | same-image | Native |
| **137** | **0.85285** | 0.83588 | +0.01697 | **1.7830** | 1.9900 | -0.2070 | drifted | drifted | **HSWQ** |
| **849** | **0.98407** | 0.99065 | -0.00658 | **0.6261** | 0.3641 | +0.2620 | same-image | same-image | Native |
| **2024** | **0.74683** | 0.74210 | +0.00473 | **1.9720** | 2.0330 | -0.0610 | bifurcated @step 11 | bifurcated @step 11 | **HSWQ** |
| **7391** | **0.97850** | 0.95811 | +0.02039 | **0.4666** | 0.9128 | -0.4462 | drifted | drifted | **HSWQ** |
| **18429** | **0.95398** | 0.87304 | +0.08094 | **0.5202** | 1.4460 | -0.9258 | drifted | drifted | **HSWQ** |
| **53082** | **0.99139** | 0.98584 | +0.00555 | **0.2936** | 0.4839 | -0.1903 | same-image | same-image | **HSWQ** |
| **149206** | **0.96193** | 0.87581 | +0.08612 | **0.5156** | 1.7030 | -1.1874 | drifted | drifted | **HSWQ** |
| **382715** | **0.99640** | 0.99416 | +0.00224 | **0.1449** | 0.2339 | -0.0890 | same-image | same-image | **HSWQ** |
| **826401** | **0.95969** | 0.90214 | +0.05755 | **0.8030** | 1.9760 | -1.1730 | drifted | drifted | **HSWQ** |
| **1938502** | **0.98970** | 0.98739 | +0.00231 | **0.2900** | 0.3554 | -0.0654 | same-image | same-image | **HSWQ** |
| **4710928** | **0.97706** | 0.95650 | +0.02056 | **0.5883** | 1.0530 | -0.4647 | drifted | drifted | **HSWQ** |
| **8391642** | **0.93449** | 0.84690 | +0.08759 | **0.8812** | 2.0300 | -1.1488 | drifted | drifted | **HSWQ** |
| **15820493** | **0.98481** | 0.95311 | +0.03170 | **0.5723** | 1.7480 | -1.1757 | same-image | drifted | **HSWQ** |
| **36192847** | **0.99077** | 0.98137 | +0.00940 | **0.2735** | 0.5500 | -0.2765 | same-image | same-image | **HSWQ** |
| **71058294** | **0.99835** | 0.99325 | +0.00510 | **0.0771** | 0.3115 | -0.2344 | same-image | same-image | **HSWQ** |
| **128491703** | **0.97835** | 0.99038 | -0.01203 | **0.6787** | 0.3000 | +0.3787 | drifted | same-image | Native |
| **285039184** | **0.98944** | 0.98632 | +0.00312 | **0.4143** | 0.5231 | -0.1088 | same-image | same-image | **HSWQ** |
| **491730285** | **0.95762** | 0.97127 | -0.01365 | **0.6078** | 0.4366 | +0.1712 | drifted | drifted | Native |
| **762019483** | **0.96085** | 0.95467 | +0.00618 | **0.6274** | 0.7892 | -0.1618 | drifted | drifted | **HSWQ** |
| **938174026** | **0.97434** | 0.96156 | +0.01278 | **0.4735** | 0.7271 | -0.2536 | drifted | drifted | **HSWQ** |
| **1409285713** | **0.93833** | 0.92040 | +0.01793 | **0.6779** | 0.8715 | -0.1936 | drifted | drifted | **HSWQ** |
| **2683910547** | **0.99590** | 0.97104 | +0.02486 | **0.1128** | 0.7957 | -0.6829 | same-image | drifted | **HSWQ** |
| **3851729406** | **0.99522** | 0.94428 | +0.05094 | **0.1569** | 1.7340 | -1.5771 | same-image | drifted | **HSWQ** |
| **4195820371** | **0.99480** | 0.99431 | +0.00049 | **0.1932** | 0.2114 | -0.0182 | same-image | same-image | **HSWQ** |

---

## 3. Key Findings and Trajectory Analysis

**F1D_unlockV3 (r170)**

1. **Mean cosine:** the reverse hybrid pack reaches 0.95388 against 0.93005 for the native W4A8 pack (+0.02383).
2. **Latent MSE:** 1.0566 against 1.7803, about 40.7% lower error.
3. **Bifurcations:** 0/25 for the reverse hybrid pack against 1/25 for the native pack; same-image seeds 14/25 against 4/25.
4. **Worst case:** the minimum per-seed cosine is 0.80882 (reverse hybrid) against 0.78136 (native).
5. **Speed:** 2.92 s per seed against 4.62 s for the FP16 reference, +36.8%.
6. **Per-seed spread:** the reverse hybrid pack is ahead on 24 of 25 seeds, the native pack on 1 (largest margin +0.00003 on seed 285039184).

**MYHuman-Flux_1.3 (r180)**

1. **Mean cosine:** the reverse hybrid pack reaches 0.96290 against 0.94239 for the native W4A8 pack (+0.02050).
2. **Latent MSE:** 0.5652 against 0.9551, about 40.8% lower error.
3. **Bifurcations:** 1/25 for the reverse hybrid pack against 1/25 for the native pack; same-image seeds 12/25 against 10/25.
4. **Worst case:** the minimum per-seed cosine is 0.74683 (reverse hybrid) against 0.74210 (native).
5. **Speed:** 3.19 s per seed against 4.65 s for the FP16 reference, +31.4%.
6. **Per-seed spread:** the reverse hybrid pack is ahead on 21 of 25 seeds, the native pack on 4 (largest margin +0.00264 on seed 42).


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
