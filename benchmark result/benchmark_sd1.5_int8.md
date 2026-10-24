# SD1.5 ConvRot INT8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ ConvRot INT8** vs **Native ConvRot INT8** on the SD1.5 architecture family.

**Source:** `benchmark result/score_sd1.5_int8.txt`  
**Evaluation Script:** `benchmark/sd1.5_int8_traj_compare.py` (25-seed deterministic trajectory analysis)

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `1on` | Bias correction **ON** |
| `k<N>` | Protected-layer count for that pack (chosen per checkpoint by sweeping the impact ranking) |

---

## 1. Summary Comparison (HSWQ ConvRot INT8 vs Native ConvRot INT8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | HSWQ Speedup (↑) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.5xxmix9realistic_v40 0.5BRA7** | 1on k80 | **0.97542** | 0.96265 | **+0.01277** | **1.0432** | 1.4793 | **0/25 (0%)** | 0/25 (0%) | **+23.2%** | **HSWQ** |
| **Family Average** | — | **0.97542** | 0.96265 | **+0.01277** | **1.0432** | 1.4793 | **0/25 (0.0%)** | 0/25 (0.0%) | **+23.2%** | **HSWQ (1/1 model)** |

**Winner** = better on both mean cosine and mean MSE.

---

## 2. Detailed Results per Model

### 2.1. 0.5xxmix9realistic_v40 0.5BRA7 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97542** | 0.96265 | **+0.01277** |
| **Min Final Cosine** (↑ better) | **0.80943** | 0.77574 | **+0.03369** |
| **Max Final Cosine** (↑ better) | 0.99892 | **0.99913** | −0.00021 |
| **Mean Final Latent MSE** (↓ better) | **1.0432** | 1.4793 | **−0.4361 (29% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+23.2%** (1.70s → 1.31s) | +20.8% (1.71s → 1.35s) | — |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 13/25 same-image, 12/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.5xxmix9realistic_v40 0.5BRA7)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99572** | 0.98524 | **+0.01048** | **0.2087** | 0.7107 | **−0.5020** | same-image | same-image | **HSWQ** |
| **137** | **0.99411** | 0.97665 | **+0.01746** | **0.2118** | 0.8394 | **−0.6276** | same-image | drifted (different image) | **HSWQ** |
| **849** | **0.99865** | 0.96341 | **+0.03524** | **0.0458** | 1.2190 | **−1.1732** | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.98348** | 0.98124 | **+0.00224** | **0.8463** | 0.9582 | **−0.1119** | same-image | same-image | **HSWQ** |
| **7391** | **0.98741** | 0.97558 | **+0.01183** | **0.5527** | 1.0650 | **−0.5123** | same-image | drifted (different image) | **HSWQ** |
| **18429** | 0.96387 | **0.96543** | −0.00156 | 1.9930 | **1.8950** | +0.0980 | drifted (different image) | drifted (different image) | Native |
| **53082** | **0.80943** | 0.79762 | **+0.01181** | **6.0900** | 6.5170 | **−0.4270** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | 0.98499 | **0.99913** | −0.01414 | 0.7891 | **0.0453** | +0.7438 | same-image | same-image | Native |
| **382715** | **0.96260** | 0.96209 | **+0.00051** | **1.5220** | 1.5410 | **−0.0190** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.98585** | 0.97448 | **+0.01137** | **0.3901** | 0.7089 | **−0.3188** | same-image | drifted (different image) | **HSWQ** |
| **1938502** | 0.97897 | **0.98035** | −0.00138 | 1.0030 | **0.9388** | +0.0642 | drifted (different image) | same-image | Native |
| **4710928** | **0.95062** | 0.77574 | **+0.17488** | **1.4770** | 6.4890 | **−5.0120** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | 0.97884 | **0.98937** | −0.01053 | 1.0620 | **0.5352** | +0.5268 | drifted (different image) | same-image | Native |
| **15820493** | **0.98530** | 0.97984 | **+0.00546** | **0.8610** | 1.1730 | **−0.3120** | same-image | drifted (different image) | **HSWQ** |
| **36192847** | **0.97011** | 0.95999 | **+0.01012** | **1.4430** | 1.9080 | **−0.4650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **71058294** | 0.95238 | **0.95664** | −0.00426 | 2.4030 | **2.1850** | +0.2180 | drifted (different image) | drifted (different image) | Native |
| **128491703** | **0.99093** | 0.98421 | **+0.00672** | **0.4215** | 0.7354 | **−0.3139** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99352** | 0.98210 | **+0.01142** | **0.3035** | 0.8376 | **−0.5341** | same-image | same-image | **HSWQ** |
| **491730285** | 0.99153 | **0.99665** | −0.00512 | 0.4032 | **0.1584** | +0.2448 | same-image | same-image | Native |
| **762019483** | **0.94085** | 0.91077 | **+0.03008** | **3.5410** | 5.2940 | **−1.7530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.99892** | 0.99748 | **+0.00144** | **0.0501** | 0.1193 | **−0.0692** | same-image | same-image | **HSWQ** |
| **1409285713** | **0.99875** | 0.99244 | **+0.00631** | **0.0517** | 0.3116 | **−0.2599** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99177** | 0.98950 | **+0.00227** | **0.2598** | 0.3309 | **−0.0711** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99851** | 0.99689 | **+0.00162** | **0.0754** | 0.1567 | **−0.0813** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99840** | 0.99338 | **+0.00502** | **0.0752** | 0.3105 | **−0.2353** | same-image | same-image | **HSWQ** |
| **Mean** | **0.97542** | 0.96265 | **+0.01277** | **1.0432** | 1.4793 | **−0.4361** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Bifurcation behaviour:** across the SD1.5 checkpoint (25 seed evaluations per arm), neither Native ConvRot INT8 nor HSWQ ConvRot INT8 produces a bifurcated seed (0/25 each). HSWQ raises the same-image count to 16/25 vs 13/25 for Native.

2. **Meaningful cosine gain:** the mean final cosine is 0.97542 for HSWQ vs 0.96265 for Native (+0.01277). HSWQ is ahead on mean cosine, min cosine and mean MSE.

3. **Latent MSE reduction:** mean final latent MSE is 1.0432 (HSWQ) vs 1.4793 (Native), about 29% lower error.

4. **Worst-case robustness:** the minimum cosine is 0.80943 (HSWQ) vs 0.77574 (Native); the best case is essentially tied (0.99892 vs 0.99913), so the gain is on the median and the worst case rather than the best case.

5. **Wall-clock:** HSWQ averages a +23.2% speedup over FP16 per seed (1.70s → 1.31s); Native averages +20.8% (1.71s → 1.35s).

6. **The gain is concentrated, not uniform.** Of 25 seeds, 19 favour HSWQ and 6 favour Native (no seed ties on both metrics). The largest single gain is seed **4710928** (Native 0.77574 → HSWQ 0.95062, **+0.17488**) - an unstable trajectory rescued rather than a small nudge on every seed. The largest loss is seed **149206** (Native 0.99913 → HSWQ 0.98499, **−0.01414**).

7. **Trajectory structure moves in HSWQ's favour.** Five seeds that drifted under Native (​`137`, `849`, `7391`, `826401`, `15820493`) hold as same-image under HSWQ; two move the other way (`1938502`, `8391642`).

---

## 4. Metric Definitions

- **Final Cosine (final-cos):** cosine similarity between the final denoised latent of the FP16 reference and the quantized model. Closer to 1.0 means identical composition, lighting and semantic fidelity.
- **Final MSE (final-mse):** mean squared error of the final latent tensor against the FP16 baseline.
- **Max Step Drop (max-drop):** maximum single-step cosine drop between consecutive sampling steps, quantifying sudden trajectory instability.
- **Bifurcated:** max-step-drop > 0.05 on any single step; a sudden trajectory jump into a different picture attractor basin, not gradual degradation.
- **Verdict:**
  - `same-image`: per-seed final cosine ≥ 0.98, producing virtually indistinguishable generation.
  - `drifted (different image)`: gradual, continuous deviation across steps while maintaining coherent compositional structure.
  - `bifurcated @step N`: discontinuous trajectory jump at step N.
- **Setup tags:** `1on` = bias correction ON; `k<N>` = protected-layer count for that pack. Taken verbatim from the HSWQ run tag in `score_sd1.5_int8.txt`.
- **Speedup:** `(FP16_avg_wall − INT8_avg_wall) / FP16_avg_wall` per seed; positive = INT8 is faster.
- **Protocol:** 25 fixed random seeds, 25 steps, 512×512, cfg 7.0, dpmpp_2m / karras. See [How to quantize SD1.5.md](../md/How%20to%20quantize%20SD1.5.md).
