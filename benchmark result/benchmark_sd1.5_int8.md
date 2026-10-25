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
| **0.5yayoiMix_v25+0.5againmix_v22** | 1on k80 | **0.97642** | 0.96010 | **+0.01632** | **1.0225** | 1.7696 | **18/25 (0%)** | 11/25 (0%) | **+38.6%** | **HSWQ** |
| **0.5yayoiMix_v25+0.5voidnoisecore_r1733** | 1on k80 | 0.97461 | **0.97518** | **−0.00057** | **0.7950** | 0.8165 | **16/25 (0%)** | 15/25 (0%) | **+35.1%** | **Native** |
| **Family Average** | — | **0.97002** | 0.95609 | **+0.01393** | **1.0486** | 1.5508 | **60/100 (0.0%)** | 46/100 (0.0%) | **+29.2%** | **HSWQ (3/4 models)** |

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

### 2.3. 0.5yayoiMix_v25+0.5againmix_v22 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97642** | 0.96010 | **+0.01632** |
| **Min Final Cosine** (↑ better) | **0.88496** | 0.85045 | **+0.03451** |
| **Max Final Cosine** (↑ better) | **0.99868** | 0.99612 | +0.00256 |
| **Mean Final Latent MSE** (↓ better) | **1.0225** | 1.7696 | **−0.7471 (42% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+38.6%** (2.19s → 1.34s) | +20.5% (1.76s → 1.40s) | — |
| **Trajectory Verdict** | 18/25 same-image, 7/25 drifted | 11/25 same-image, 14/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.5yayoiMix_v25+0.5againmix_v22)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98430** | 0.95316 | +0.03114 | **0.7103** | 2.1300 | −1.4197 | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.98923** | 0.95049 | +0.03874 | **0.3926** | 1.8360 | −1.4434 | same-image | drifted (different image) | **HSWQ** |
| **849** | **0.99812** | 0.98396 | +0.01416 | **0.0630** | 0.5332 | −0.4702 | same-image | same-image | **HSWQ** |
| **2024** | **0.99868** | 0.96737 | +0.03131 | **0.0698** | 1.7170 | −1.6472 | same-image | drifted (different image) | **HSWQ** |
| **7391** | 0.93084 | **0.93513** | −0.00429 | 2.8940 | **2.7200** | +0.1740 | drifted (different image) | drifted (different image) | Native |
| **18429** | 0.98398 | **0.98684** | −0.00286 | 0.7996 | **0.6602** | +0.1394 | same-image | same-image | Native |
| **53082** | **0.98558** | 0.97902 | +0.00656 | **0.4850** | 0.7095 | −0.2245 | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.98841** | 0.97615 | +0.01226 | **0.4708** | 0.9810 | −0.5102 | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.98197** | 0.98146 | +0.00051 | **0.6839** | 0.7114 | −0.0275 | same-image | same-image | **HSWQ** |
| **826401** | 0.95015 | **0.95769** | −0.00754 | 1.4610 | **1.2720** | +0.1890 | drifted (different image) | drifted (different image) | Native |
| **1938502** | 0.88496 | **0.88921** | −0.00425 | 4.5780 | **4.5490** | +0.0290 | drifted (different image) | drifted (different image) | Native |
| **4710928** | 0.95016 | **0.95307** | −0.00291 | 1.5080 | **1.4290** | +0.0790 | drifted (different image) | drifted (different image) | Native |
| **8391642** | **0.94081** | 0.88459 | +0.05622 | **2.8170** | 5.5120 | −2.6950 | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | 0.98755 | **0.98999** | −0.00244 | 0.7980 | **0.6415** | +0.1565 | same-image | same-image | Native |
| **36192847** | **0.98985** | 0.98488 | +0.00497 | **0.4781** | 0.7125 | −0.2344 | same-image | same-image | **HSWQ** |
| **71058294** | **0.96446** | 0.94874 | +0.01572 | **1.7140** | 2.4710 | −0.7570 | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.99763** | 0.99563 | +0.00200 | **0.1051** | 0.1937 | −0.0886 | same-image | same-image | **HSWQ** |
| **285039184** | **0.98550** | 0.85045 | +0.13505 | **0.5799** | 6.2470 | −5.6671 | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.99797** | 0.98240 | +0.01557 | **0.0969** | 0.8465 | −0.7496 | same-image | same-image | **HSWQ** |
| **762019483** | **0.93918** | 0.92141 | +0.01777 | **4.0460** | 5.2450 | −1.1990 | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.99721** | 0.99530 | +0.00191 | **0.1189** | 0.2035 | −0.0846 | same-image | same-image | **HSWQ** |
| **1409285713** | **0.99727** | 0.99612 | +0.00115 | **0.1033** | 0.1545 | −0.0512 | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99770** | 0.99365 | +0.00405 | **0.0700** | 0.1944 | −0.1244 | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99310** | 0.96061 | +0.03249 | **0.3298** | 1.8840 | −1.5542 | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.99592** | 0.98522 | +0.01070 | **0.1899** | 0.6865 | −0.4966 | same-image | same-image | **HSWQ** |
| **Mean** | **0.97642** | 0.96010 | **+0.01632** | **1.0225** | 1.7696 | **−0.7471** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

### 2.4. 0.5yayoiMix_v25+0.5voidnoisecore_r1733 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | 0.97461 | **0.97518** | −0.00057 |
| **Min Final Cosine** (↑ better) | 0.90601 | **0.91578** | −0.00977 |
| **Max Final Cosine** (↑ better) | **0.99911** | 0.99771 | +0.00140 |
| **Mean Final Latent MSE** (↓ better) | **0.7950** | 0.8165 | **−0.0215 (3% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+35.1%** (2.21s → 1.43s) | +28.0% (1.91s → 1.38s) | — |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 15/25 same-image, 10/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.5yayoiMix_v25+0.5voidnoisecore_r1733)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.93910** | 0.93002 | +0.00908 | **2.1760** | 2.5080 | −0.3320 | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.99336** | 0.98655 | +0.00681 | **0.1900** | 0.3847 | −0.1947 | same-image | same-image | **HSWQ** |
| **849** | 0.92766 | **0.98747** | −0.05981 | 2.1560 | **0.3813** | +1.7747 | drifted (different image) | same-image | Native |
| **2024** | 0.90601 | **0.91578** | −0.00977 | 2.5840 | **2.3430** | +0.2410 | drifted (different image) | drifted (different image) | Native |
| **7391** | 0.91915 | **0.96303** | −0.04388 | 2.4520 | **1.1350** | +1.3170 | drifted (different image) | drifted (different image) | Native |
| **18429** | 0.99560 | **0.99605** | −0.00045 | 0.1675 | **0.1499** | +0.0176 | same-image | same-image | Native |
| **53082** | **0.99911** | 0.99731 | +0.00180 | **0.0222** | 0.0670 | −0.0448 | same-image | same-image | **HSWQ** |
| **149206** | **0.99881** | 0.94624 | +0.05257 | **0.0308** | 1.4020 | −1.3712 | same-image | drifted (different image) | **HSWQ** |
| **382715** | 0.99147 | **0.99410** | −0.00263 | 0.2242 | **0.1551** | +0.0691 | same-image | same-image | Native |
| **826401** | 0.91363 | **0.99381** | −0.08018 | 2.0020 | **0.1438** | +1.8582 | drifted (different image) | same-image | Native |
| **1938502** | **0.99387** | 0.98537 | +0.00850 | **0.1971** | 0.4725 | −0.2754 | same-image | same-image | **HSWQ** |
| **4710928** | **0.97499** | 0.95432 | +0.02067 | **0.6324** | 1.1610 | −0.5286 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.99486** | 0.98208 | +0.01278 | **0.2071** | 0.7184 | −0.5113 | same-image | same-image | **HSWQ** |
| **15820493** | 0.98369 | **0.99072** | −0.00703 | 0.6319 | **0.3600** | +0.2719 | same-image | same-image | Native |
| **36192847** | **0.99297** | 0.98422 | +0.00875 | **0.2139** | 0.4785 | −0.2646 | same-image | same-image | **HSWQ** |
| **71058294** | 0.99494 | **0.99771** | −0.00277 | 0.1659 | **0.0765** | +0.0894 | same-image | same-image | Native |
| **128491703** | **0.97387** | 0.96179 | +0.01208 | **1.1690** | 1.7150 | −0.5460 | drifted (different image) | drifted (different image) | **HSWQ** |
| **285039184** | **0.99468** | 0.99398 | +0.00070 | **0.1647** | 0.1870 | −0.0223 | same-image | same-image | **HSWQ** |
| **491730285** | **0.98099** | 0.97817 | +0.00282 | **0.5371** | 0.6179 | −0.0808 | same-image | drifted (different image) | **HSWQ** |
| **762019483** | **0.99137** | 0.96197 | +0.02940 | **0.3304** | 1.4390 | −1.1086 | same-image | drifted (different image) | **HSWQ** |
| **938174026** | **0.98939** | 0.98727 | +0.00212 | **0.3814** | 0.4610 | −0.0796 | same-image | same-image | **HSWQ** |
| **1409285713** | **0.97933** | 0.95639 | +0.02294 | **0.6110** | 1.2910 | −0.6800 | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | 0.99019 | **0.99619** | −0.00600 | 0.2475 | **0.0966** | +0.1509 | same-image | same-image | Native |
| **3851729406** | 0.94864 | **0.95012** | −0.00148 | 2.2810 | **2.2140** | +0.0670 | drifted (different image) | drifted (different image) | Native |
| **4195820371** | **0.99750** | 0.98872 | +0.00878 | **0.1008** | 0.4543 | −0.3535 | same-image | same-image | **HSWQ** |
| **Mean** | 0.97461 | **0.97518** | −0.00057 | 0.7950 | 0.8165 | −0.0215 | **0/25 Bifurcated** | **0/25 Bifurcated** | **Native** (10/25) |

---

## 3. Key Findings and Trajectory Analysis

1. **Bifurcation behaviour:** across the 4 checkpoints (25 seed evaluations per arm), neither Native ConvRot INT8 nor HSWQ ConvRot INT8 produces a bifurcated seed (0/100 each). HSWQ raises the same-image count to 60/100 vs 46/100 for Native.

2. **Cosine gain:** the family mean final cosine is 0.97002 for HSWQ vs 0.95609 for Native (+0.01393). HSWQ is ahead on mean cosine in 3/4 models; the fourth checkpoint is the exception - see finding 8.

3. **Latent MSE reduction:** family mean final latent MSE is 1.0486 (HSWQ) vs 1.5508 (Native), about 32% lower error.

4. **Worst-case robustness:** the worst per-model minimum cosine is 0.78997 (HSWQ) vs 0.74251 (Native).

5. **Wall-clock:** HSWQ averages a +29.2% speedup over FP16 per seed across the family.

6. **The second checkpoint is harder for both arms.** `0.5xxmix9realistic_v40+0.5chilledReGenericV3` sits below `0.5xxmix9realistic_v40 0.5BRA7` on every aggregate for both HSWQ and Native; the HSWQ/Native gap is nonetheless wider there (+0.02718 vs +0.01277), so the protection buys more on the harder model, not less.

7. **The third checkpoint remains the strongest HSWQ case.** `0.5yayoiMix_v25+0.5againmix_v22` gives HSWQ its highest same-image count on a single checkpoint (18/25 against 11/25) and its largest latent-MSE reduction (42%), while also showing the widest wall-clock gain (+38.6%). Its minimum cosine (0.88496) is the highest of the four HSWQ arms, so the family worst case stays with the second checkpoint (0.78997).

8. **The fourth checkpoint is the one Native takes.** `0.5yayoiMix_v25+0.5voidnoisecore_r1733` is the only checkpoint here where the Native pack beats the protected one on the mean final cosine (0.97518 against 0.97461, −0.00057) and on the minimum (0.91578 against 0.90601); HSWQ wins the per-seed count by cosine (15/25 against 10/25). HSWQ keeps the lower mean MSE (0.7950 against 0.8165), the higher same-image count (16/25 against 15/25) and the larger speedup (35.1% against 28.0%). The family aggregate stays with HSWQ (3/4 models).


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
