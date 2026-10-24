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
| **0.5xxmix9realistic_v40 0.5BRA7** | 1on k80 | **0.97542** | 0.96265 | **+0.01277** | **1.0432** | 1.4793 | **16/25 (0%)** | 13/25 (0%) | **+23.2%** | **HSWQ** |
| **0.5xxmix9realistic_v40+0.5chilledReGenericV3** | 1on k80 | **0.95362** | 0.92644 | **+0.02719** | **1.3338** | 2.1379 | **10/25 (0%)** | 7/25 (0%) | **+20.1%** | **HSWQ** |
| **Family Average** | — | **0.96452** | 0.94454 | **+0.01998** | **1.1885** | 1.8086 | **26/50 (0.0%)** | 20/50 (0.0%) | **+21.6%** | **HSWQ (2/2 models)** |

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

### 2.2. 0.5xxmix9realistic_v40+0.5chilledReGenericV3 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95362** | 0.92644 | **+0.02719** |
| **Min Final Cosine** (↑ better) | **0.78997** | 0.74251 | **+0.04746** |
| **Max Final Cosine** (↑ better) | **0.99853** | 0.99383 | +0.00470 |
| **Mean Final Latent MSE** (↓ better) | **1.3338** | 2.1379 | **−0.8042 (38% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+20.1%** (1.72s → 1.37s) | +10.1% (1.80s → 1.62s) | — |
| **Trajectory Verdict** | 10/25 same-image, 15/25 drifted | 7/25 same-image, 18/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.5xxmix9realistic_v40+0.5chilledReGenericV3)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.90956** | 0.89356 | **+0.01600** | **2.4970** | 2.9430 | **−0.4460** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | 0.98045 | **0.99056** | −0.01011 | 0.5394 | **0.2706** | +0.2688 | same-image | same-image | Native |
| **849** | **0.89497** | 0.74251 | **+0.15246** | **2.6230** | 6.4140 | **−3.7910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.99708** | 0.98948 | **+0.00760** | **0.0813** | 0.3182 | **−0.2369** | same-image | same-image | **HSWQ** |
| **7391** | **0.99309** | 0.98123 | **+0.01186** | **0.2589** | 0.7013 | **−0.4424** | same-image | same-image | **HSWQ** |
| **18429** | **0.94390** | 0.93683 | **+0.00707** | **2.4650** | 2.7660 | **−0.3010** | drifted (different image) | drifted (different image) | **HSWQ** |
| **53082** | 0.93132 | **0.95245** | −0.02113 | 1.6400 | **1.1360** | +0.5040 | drifted (different image) | drifted (different image) | Native |
| **149206** | 0.78997 | **0.79120** | −0.00123 | 5.3850 | **5.3410** | +0.0440 | drifted (different image) | drifted (different image) | Native |
| **382715** | **0.97103** | 0.96239 | **+0.00864** | **1.0710** | 1.3710 | **−0.3000** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.94119** | 0.93905 | **+0.00214** | **1.4410** | 1.4930 | **−0.0520** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.86134** | 0.80738 | **+0.05396** | **3.6130** | 5.0780 | **−1.4650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.96718** | 0.93655 | **+0.03063** | **0.5262** | 1.0100 | **−0.4838** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.99831** | 0.99383 | **+0.00448** | **0.0465** | 0.1738 | **−0.1273** | same-image | same-image | **HSWQ** |
| **15820493** | **0.95612** | 0.88442 | **+0.07170** | **1.1920** | 3.2650 | **−2.0730** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.99011** | 0.98395 | **+0.00616** | **0.2462** | 0.3985 | **−0.1523** | same-image | same-image | **HSWQ** |
| **71058294** | **0.93428** | 0.88660 | **+0.04768** | **2.2900** | 4.4140 | **−2.1240** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.96297** | 0.92220 | **+0.04077** | **1.6590** | 3.4600 | **−1.8010** | drifted (different image) | drifted (different image) | **HSWQ** |
| **285039184** | **0.99853** | 0.98814 | **+0.01039** | **0.0470** | 0.3775 | **−0.3305** | same-image | same-image | **HSWQ** |
| **491730285** | **0.96189** | 0.95067 | **+0.01122** | **1.3370** | 1.9060 | **−0.5690** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | 0.91666 | **0.97300** | −0.05634 | 2.6900 | **0.8904** | +1.7996 | drifted (different image) | drifted (different image) | Native |
| **938174026** | **0.98770** | 0.96354 | **+0.02416** | **0.4410** | 1.3050 | **−0.8640** | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | **0.97974** | 0.83468 | **+0.14506** | **0.4779** | 3.8190 | **−3.3411** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | **0.99517** | 0.99229 | **+0.00288** | **0.1115** | 0.1793 | **−0.0678** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99793** | 0.97355 | **+0.02438** | **0.0989** | 1.2600 | **−1.1611** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.98011** | 0.89082 | **+0.08929** | **0.5667** | 3.1580 | **−2.5913** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.95362** | 0.92644 | **+0.02719** | **1.3338** | 2.1379 | **−0.8042** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Bifurcation behaviour:** across the 2 checkpoints (25 seed evaluations per arm), neither Native ConvRot INT8 nor HSWQ ConvRot INT8 produces a bifurcated seed (0/50 each). HSWQ raises the same-image count to 26/50 vs 20/50 for Native.

2. **Meaningful cosine gain:** the family mean final cosine is 0.96452 for HSWQ vs 0.94454 for Native (+0.01998). HSWQ is ahead on mean cosine in 2/2 models.

3. **Latent MSE reduction:** family mean final latent MSE is 1.1885 (HSWQ) vs 1.8086 (Native), about 34% lower error.

4. **Worst-case robustness:** the worst per-model minimum cosine is 0.78997 (HSWQ) vs 0.74251 (Native).

5. **Wall-clock:** HSWQ averages a +21.6% speedup over FP16 per seed across the family.

6. **The second checkpoint is harder for both arms.** `0.5xxmix9realistic_v40+0.5chilledReGenericV3` sits below `0.5xxmix9realistic_v40 0.5BRA7` on every aggregate for both HSWQ and Native; the HSWQ/Native gap is nonetheless wider there (+0.02718 vs +0.01277), so the protection buys more on the harder model, not less.

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
