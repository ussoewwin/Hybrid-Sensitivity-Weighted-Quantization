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
| **0.5yayoiMix_v25+0.5xxmix9realistic_v40** | 1on k80 | **0.96321** | 0.94839 | **+0.01483** | **1.1880** | 1.8368 | **16/25 (0%)** | 12/25 (0%) | **+31.3%** | **HSWQ** |
| **0.7_0.6hassaku+0.4dreamshaper+0.3anylora** | 1on k80 | **0.98061** | 0.95456 | **+0.02605** | **0.6839** | 1.7171 | **16/25 (0%)** | 9/25 (0%) | **+15.1%** | **HSWQ** |
| **0.7_0.7hassaku+0.3dreamshaper_8_0.3anylora** | 1on k82 | **0.96012** | 0.94130 | **+0.01882** | **1.2980** | 1.9021 | **14/25 (0%)** | 11/25 (0%) | **+22.6%** | **HSWQ** |
| **0.7chilledReGenericV3+0.3uberRealisticMerge_v13** | 1on k80 | **0.97332** | 0.95932 | **+0.01400** | **0.7196** | 1.0755 | **15/25 (0%)** | 13/25 (0%) | **+25.1%** | **HSWQ** |
| **0.7xxmix9realistic_v40+0.3BRA7** | 1on k80 | **0.97332** | 0.96174 | **+0.01158** | **0.9861** | 1.5077 | **18/25 (0%)** | 16/25 (0%) | **+24.5%** | **HSWQ** |
| **0.9chilledReGenericV3+0.1BRA7** | 1on k80 | **0.97352** | 0.96501 | **+0.00851** | **0.7374** | 0.9495 | **16/25 (0%)** | 14/25 (0%) | **+24.1%** | **HSWQ** |
| **0.9chilledReGenericV3+0.1majicmixRealistic_v7** | 1on k80 | **0.98043** | 0.96837 | **+0.01206** | **0.5168** | 0.7227 | **19/25 (0%)** | 14/25 (0%) | **+8.7%** | **HSWQ** |
| **Family Average** | — | **0.97133** | 0.95664 | **+0.01469** | **0.9386** | 1.4468 | **174/275 (0.0%)** | 135/275 (0.0%) | **+24.4%** | **HSWQ (10/11 models)** |

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

### 2.5. 0.5yayoiMix_v25+0.5xxmix9realistic_v40 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96321** | 0.94839 | **+0.01483** |
| **Min Final Cosine** (↑ better) | 0.74845 | 0.74461 | **+0.00384** |
| **Max Final Cosine** (↑ better) | **0.99934** | 0.99778 | +0.00156 |
| **Mean Final Latent MSE** (↓ better) | **1.1880** | 1.8368 | **−0.6488 (35% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+31.3%** (2.07s → 1.42s) | +27.0% (1.93s → 1.41s) | — |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 12/25 same-image, 13/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.5yayoiMix_v25+0.5xxmix9realistic_v40)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98329** | 0.95011 | +0.03318 | **0.5413** | 1.6050 | −1.0637 | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.97027** | 0.94307 | +0.02720 | **0.8476** | 1.6160 | −0.7684 | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.99356** | 0.89231 | +0.10125 | **0.2180** | 3.6670 | −3.4490 | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.99757** | 0.99158 | +0.00599 | **0.1167** | 0.4040 | −0.2873 | same-image | same-image | **HSWQ** |
| **7391** | 0.77261 | **0.78286** | −0.01025 | 8.4670 | **8.0260** | +0.4410 | drifted (different image) | drifted (different image) | Native |
| **18429** | **0.99690** | 0.96370 | +0.03320 | **0.1325** | 1.5430 | −1.4105 | same-image | drifted (different image) | **HSWQ** |
| **53082** | 0.89712 | **0.98891** | −0.09179 | 2.7960 | **0.3008** | +2.4952 | drifted (different image) | same-image | Native |
| **149206** | **0.99360** | 0.98090 | +0.01270 | **0.2280** | 0.6808 | −0.4528 | same-image | same-image | **HSWQ** |
| **382715** | **0.99108** | 0.97087 | +0.02021 | **0.3263** | 1.0740 | −0.7477 | same-image | drifted (different image) | **HSWQ** |
| **826401** | 0.96722 | **0.97236** | −0.00514 | 0.8957 | **0.7543** | +0.1414 | drifted (different image) | drifted (different image) | Native |
| **1938502** | **0.99716** | 0.99539 | +0.00177 | **0.0952** | 0.1540 | −0.0588 | same-image | same-image | **HSWQ** |
| **4710928** | **0.74845** | 0.74461 | +0.00384 | **5.5290** | 5.5880 | −0.0590 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.97035** | 0.96305 | +0.00730 | **1.2110** | 1.5060 | −0.2950 | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.98184** | 0.96981 | +0.01203 | **0.9230** | 1.5300 | −0.6070 | same-image | drifted (different image) | **HSWQ** |
| **36192847** | **0.98578** | 0.98213 | +0.00365 | **0.5046** | 0.6283 | −0.1237 | same-image | same-image | **HSWQ** |
| **71058294** | 0.98907 | **0.99148** | −0.00241 | 0.5057 | **0.3949** | +0.1108 | same-image | same-image | Native |
| **128491703** | 0.98327 | **0.99037** | −0.00710 | 0.7750 | **0.4453** | +0.3297 | same-image | same-image | Native |
| **285039184** | **0.99916** | 0.98881 | +0.01035 | **0.0350** | 0.4642 | −0.4293 | same-image | same-image | **HSWQ** |
| **491730285** | 0.97883 | **0.99391** | −0.01508 | 0.9372 | **0.2691** | +0.6681 | drifted (different image) | same-image | Native |
| **762019483** | **0.95276** | 0.79800 | +0.15476 | **2.4080** | 10.0800 | −7.6720 | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.99352** | 0.93627 | +0.05725 | **0.2799** | 2.7690 | −2.4891 | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | **0.94288** | 0.94043 | +0.00245 | **1.6830** | 1.7610 | −0.0780 | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | **0.99828** | 0.98353 | +0.01475 | **0.0454** | 0.4320 | −0.3866 | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99934** | 0.99778 | +0.00156 | **0.0325** | 0.1085 | −0.0760 | same-image | same-image | **HSWQ** |
| **4195820371** | 0.99641 | **0.99745** | −0.00104 | 0.1663 | **0.1184** | +0.0479 | same-image | same-image | Native |
| **Mean** | **0.96321** | 0.94839 | **+0.01483** | **1.1880** | 1.8368 | **−0.6488** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (18/25)** |

### 2.6. 0.7_0.6hassaku+0.4dreamshaper+0.3anylora (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98061** | 0.95456 | **+0.02605** |
| **Min Final Cosine** (↑ better) | **0.91963** | 0.81359 | **+0.10604** |
| **Max Final Cosine** (↑ better) | **0.99944** | 0.99873 | **+0.00071** |
| **Mean Final Latent MSE** (↓ better) | **0.6839** | 1.7171 | **−1.0332 (60.2% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | +15.1% (1.83s → 1.55s) | **+37.5%** (2.24s → 1.40s) | — |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 9/25 same-image, 16/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.7_0.6hassaku+0.4dreamshaper+0.3anylora)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98156** | 0.94565 | **+0.03591** | **0.5949** | 1.7770 | **−1.1821** | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.99555** | 0.99346 | **+0.00209** | **0.1342** | 0.1977 | **−0.0635** | same-image | same-image | **HSWQ** |
| **849** | **0.94149** | 0.93221 | **+0.00928** | **1.4310** | 1.6920 | **−0.2610** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.99194** | 0.93896 | **+0.05298** | **0.2505** | 1.8540 | **−1.6035** | same-image | drifted (different image) | **HSWQ** |
| **7391** | **0.99482** | 0.96736 | **+0.02746** | **0.2081** | 1.3000 | **−1.0919** | same-image | drifted (different image) | **HSWQ** |
| **18429** | **0.99637** | 0.98770 | **+0.00867** | **0.1603** | 0.5395 | **−0.3792** | same-image | same-image | **HSWQ** |
| **53082** | **0.95963** | 0.93761 | **+0.02202** | **1.1450** | 1.7510 | **−0.6060** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.99944** | 0.99742 | **+0.00202** | **0.0168** | 0.0769 | **−0.0601** | same-image | same-image | **HSWQ** |
| **382715** | **0.99388** | 0.98297 | **+0.01091** | **0.2503** | 0.6985 | **−0.4482** | same-image | same-image | **HSWQ** |
| **826401** | **0.97208** | 0.96454 | **+0.00754** | **0.9694** | 1.2120 | **−0.2426** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.99662** | 0.81359 | **+0.18303** | **0.1130** | 6.4240 | **−6.3110** | same-image | drifted (different image) | **HSWQ** |
| **4710928** | **0.99692** | 0.99241 | **+0.00451** | **0.0701** | 0.1724 | **−0.1023** | same-image | same-image | **HSWQ** |
| **8391642** | **0.97844** | 0.97682 | **+0.00162** | **0.8835** | 0.9495 | **−0.0660** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.95896** | 0.84014 | **+0.11882** | **1.7950** | 7.0680 | **−5.2730** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | 0.91963 | **0.93404** | −0.01441 | 2.4310 | **2.0000** | +0.4310 | drifted (different image) | drifted (different image) | Native |
| **71058294** | **0.97551** | 0.95118 | **+0.02433** | **1.0860** | 2.1630 | **−1.0770** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | 0.99850 | **0.99873** | −0.00023 | 0.0666 | **0.0563** | +0.0103 | same-image | same-image | Native |
| **285039184** | **0.98131** | 0.96877 | **+0.01254** | **0.6168** | 1.0360 | **−0.4192** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.99531** | 0.99458 | **+0.00073** | **0.2084** | 0.2407 | **−0.0323** | same-image | same-image | **HSWQ** |
| **762019483** | **0.95769** | 0.95402 | **+0.00367** | **1.8980** | 2.0590 | **−0.1610** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.95593** | 0.93018 | **+0.02575** | **1.9860** | 3.1510 | **−1.1650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1409285713** | **0.99806** | 0.98108 | **+0.01698** | **0.0518** | 0.4953 | **−0.4435** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98625** | 0.96722 | **+0.01903** | **0.3442** | 0.8334 | **−0.4892** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99875** | 0.99417 | **+0.00458** | **0.0649** | 0.3066 | **−0.2417** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99071** | 0.91914 | **+0.07157** | **0.3203** | 2.8740 | **−2.5537** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.98061** | 0.95456 | **+0.02605** | **0.6839** | 1.7171 | **−1.0332** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (23/25)** |

### 2.7. 0.7_0.7hassaku+0.3dreamshaper_8_0.3anylora (1on k82)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96012** | 0.94130 | **+0.01882** |
| **Min Final Cosine** (↑ better) | **0.69419** | 0.63851 | **+0.05568** |
| **Max Final Cosine** (↑ better) | **0.99911** | 0.99675 | **+0.00236** |
| **Mean Final Latent MSE** (↓ better) | **1.2980** | 1.9021 | **−0.6041 (31.8% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | +22.6% (1.71s → 1.32s) | **+28.0%** (1.91s → 1.37s) | — |
| **Trajectory Verdict** | 14/25 same-image, 11/25 drifted | 11/25 same-image, 14/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.7_0.7hassaku+0.3dreamshaper_8_0.3anylora)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.88097** | 0.63851 | **+0.24246** | **3.7570** | 11.8300 | **−8.0730** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | 0.97646 | **0.99527** | −0.01881 | 0.6959 | **0.1400** | +0.5559 | drifted (different image) | same-image | Native |
| **849** | **0.98844** | 0.97850 | **+0.00994** | **0.2744** | 0.5092 | **−0.2348** | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.69419** | 0.67768 | **+0.01651** | **8.4940** | 8.8680 | **−0.3740** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.99678** | 0.98999 | **+0.00679** | **0.1304** | 0.4019 | **−0.2715** | same-image | same-image | **HSWQ** |
| **18429** | **0.99795** | 0.99675 | **+0.00120** | **0.0889** | 0.1407 | **−0.0518** | same-image | same-image | **HSWQ** |
| **53082** | **0.95305** | 0.93033 | **+0.02272** | **1.3390** | 1.9770 | **−0.6380** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.99781** | 0.99441 | **+0.00340** | **0.0665** | 0.1682 | **−0.1017** | same-image | same-image | **HSWQ** |
| **382715** | **0.99891** | 0.98321 | **+0.01570** | **0.0456** | 0.6966 | **−0.6510** | same-image | same-image | **HSWQ** |
| **826401** | **0.92804** | 0.90144 | **+0.02660** | **2.4950** | 3.5490 | **−1.0540** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.98974** | 0.98541 | **+0.00433** | **0.3479** | 0.4919 | **−0.1440** | same-image | same-image | **HSWQ** |
| **4710928** | **0.89904** | 0.89137 | **+0.00767** | **2.3110** | 2.4560 | **−0.1450** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | 0.87997 | **0.96854** | −0.08857 | 4.9300 | **1.2680** | +3.6620 | drifted (different image) | drifted (different image) | Native |
| **15820493** | 0.93560 | **0.95075** | −0.01515 | 2.8600 | **2.1910** | +0.6690 | drifted (different image) | drifted (different image) | Native |
| **36192847** | **0.98783** | 0.98596 | **+0.00187** | **0.3637** | 0.4190 | **−0.0553** | same-image | same-image | **HSWQ** |
| **71058294** | **0.97540** | 0.96529 | **+0.01011** | **1.0920** | 1.5380 | **−0.4460** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.99911** | 0.98461 | **+0.01450** | **0.0390** | 0.6713 | **−0.6323** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99559** | 0.93319 | **+0.06240** | **0.1428** | 2.1700 | **−2.0272** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.99075** | 0.98630 | **+0.00445** | **0.4111** | 0.6069 | **−0.1958** | same-image | same-image | **HSWQ** |
| **762019483** | **0.97555** | 0.97423 | **+0.00132** | **1.0190** | 1.0620 | **−0.0430** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.97830** | 0.94346 | **+0.03484** | **0.9967** | 2.5530 | **−1.5563** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1409285713** | **0.99760** | 0.99047 | **+0.00713** | **0.0608** | 0.2456 | **−0.1848** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99688** | 0.93369 | **+0.06319** | **0.0826** | 1.7930 | **−1.7104** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | 0.99558 | **0.99570** | −0.00012 | 0.2280 | **0.2215** | +0.0065 | same-image | same-image | Native |
| **4195820371** | **0.99348** | 0.95754 | **+0.03594** | **0.2372** | 1.5840 | **−1.3468** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.96012** | 0.94130 | **+0.01882** | **1.2980** | 1.9021 | **−0.6041** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

### 2.8. 0.7chilledReGenericV3+0.3uberRealisticMerge_v13 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97332** | 0.95932 | **+0.01400** |
| **Min Final Cosine** (↑ better) | **0.87372** | 0.79971 | **+0.07401** |
| **Max Final Cosine** (↑ better) | **0.99942** | 0.99865 | **+0.00077** |
| **Mean Final Latent MSE** (↓ better) | **0.7196** | 1.0755 | **−0.3559 (33.1% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+25.1%** (1.77s → 1.33s) | +21.9% (1.82s → 1.42s) | — |
| **Trajectory Verdict** | 15/25 same-image, 10/25 drifted | 13/25 same-image, 12/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.7chilledReGenericV3+0.3uberRealisticMerge_v13)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.97589 | **0.99409** | −0.01820 | 0.7944 | **0.1953** | +0.5991 | drifted (different image) | same-image | Native |
| **137** | 0.93609 | **0.93924** | −0.00315 | 2.1130 | **2.0160** | +0.0970 | drifted (different image) | drifted (different image) | Native |
| **849** | **0.99683** | 0.99538 | **+0.00145** | **0.0761** | 0.1102 | **−0.0341** | same-image | same-image | **HSWQ** |
| **2024** | **0.92155** | 0.79971 | **+0.12184** | **1.5150** | 3.9080 | **−2.3930** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.99393** | 0.99171 | **+0.00222** | **0.1447** | 0.1970 | **−0.0523** | same-image | same-image | **HSWQ** |
| **18429** | 0.96838 | **0.96961** | −0.00123 | 1.2050 | **1.1580** | +0.0470 | drifted (different image) | drifted (different image) | Native |
| **53082** | **0.99942** | 0.99821 | **+0.00121** | **0.0146** | 0.0442 | **−0.0296** | same-image | same-image | **HSWQ** |
| **149206** | **0.96974** | 0.83337 | **+0.13637** | **0.8527** | 4.5340 | **−3.6813** | drifted (different image) | drifted (different image) | **HSWQ** |
| **382715** | **0.99765** | 0.99657 | **+0.00108** | **0.0536** | 0.0782 | **−0.0246** | same-image | same-image | **HSWQ** |
| **826401** | 0.93803 | **0.98380** | −0.04577 | 1.5700 | **0.4100** | +1.1600 | drifted (different image) | same-image | Native |
| **1938502** | 0.97057 | **0.97416** | −0.00359 | 0.7853 | **0.6816** | +0.1037 | drifted (different image) | drifted (different image) | Native |
| **4710928** | 0.98685 | **0.99358** | −0.00673 | 0.2964 | **0.1415** | +0.1549 | same-image | same-image | Native |
| **8391642** | **0.99221** | 0.99020 | **+0.00201** | **0.2063** | 0.2602 | **−0.0539** | same-image | same-image | **HSWQ** |
| **15820493** | **0.98627** | 0.97908 | **+0.00719** | **0.5290** | 0.8160 | **−0.2870** | same-image | drifted (different image) | **HSWQ** |
| **36192847** | **0.87372** | 0.86217 | **+0.01155** | **3.3500** | 3.8170 | **−0.4670** | drifted (different image) | drifted (different image) | **HSWQ** |
| **71058294** | **0.99930** | 0.99743 | **+0.00187** | **0.0228** | 0.0844 | **−0.0616** | same-image | same-image | **HSWQ** |
| **128491703** | **0.99024** | 0.98340 | **+0.00684** | **0.3147** | 0.5379 | **−0.2232** | same-image | same-image | **HSWQ** |
| **285039184** | 0.99389 | **0.99638** | −0.00249 | 0.1575 | **0.0948** | +0.0627 | same-image | same-image | Native |
| **491730285** | **0.99709** | 0.96816 | **+0.02893** | **0.0807** | 0.8698 | **−0.7891** | same-image | drifted (different image) | **HSWQ** |
| **762019483** | 0.90702 | **0.96964** | −0.06262 | 2.3790 | **0.8351** | +1.5439 | drifted (different image) | drifted (different image) | Native |
| **938174026** | **0.99729** | 0.92762 | **+0.06967** | **0.0723** | 1.9630 | **−1.8907** | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | 0.94808 | **0.95031** | −0.00223 | 1.2290 | **1.1810** | +0.0480 | drifted (different image) | drifted (different image) | Native |
| **2683910547** | 0.99505 | **0.99865** | −0.00360 | 0.1290 | **0.0352** | +0.0938 | same-image | same-image | Native |
| **3851729406** | **0.99940** | 0.99379 | **+0.00561** | **0.0181** | 0.1873 | **−0.1692** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99842** | 0.89677 | **+0.10165** | **0.0415** | 2.7310 | **−2.6895** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.97332** | 0.95932 | **+0.01400** | **0.7196** | 1.0755 | **−0.3559** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (15/25)** |

### 2.9. 0.7xxmix9realistic_v40+0.3BRA7 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97332** | 0.96174 | **+0.01158** |
| **Min Final Cosine** (↑ better) | **0.82978** | 0.78285 | **+0.04693** |
| **Max Final Cosine** (↑ better) | **0.99924** | 0.99657 | **+0.00267** |
| **Mean Final Latent MSE** (↓ better) | **0.9861** | 1.5077 | **−0.5216 (34.6% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+24.5%** (1.72s → 1.30s) | +20.5% (1.72s → 1.37s) | — |
| **Trajectory Verdict** | 18/25 same-image, 7/25 drifted | 16/25 same-image, 9/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.7xxmix9realistic_v40+0.3BRA7)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.82978** | 0.82061 | **+0.00917** | **5.9770** | 6.2450 | **−0.2680** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.99763** | 0.98900 | **+0.00863** | **0.0723** | 0.3279 | **−0.2556** | same-image | same-image | **HSWQ** |
| **849** | 0.98098 | **0.99521** | −0.01423 | 0.6915 | **0.1785** | +0.5130 | same-image | same-image | Native |
| **2024** | **0.99364** | 0.99107 | **+0.00257** | **0.3140** | 0.4402 | **−0.1262** | same-image | same-image | **HSWQ** |
| **7391** | **0.96834** | 0.95307 | **+0.01527** | **1.1940** | 1.7760 | **−0.5820** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.99486** | 0.99051 | **+0.00435** | **0.2267** | 0.4170 | **−0.1903** | same-image | same-image | **HSWQ** |
| **53082** | 0.98368 | **0.99189** | −0.00821 | 0.4981 | **0.2467** | +0.2514 | same-image | same-image | Native |
| **149206** | **0.99714** | 0.96796 | **+0.02918** | **0.1458** | 1.6250 | **−1.4792** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.99201** | 0.98820 | **+0.00381** | **0.3029** | 0.4448 | **−0.1419** | same-image | same-image | **HSWQ** |
| **826401** | 0.97182 | **0.98893** | −0.01711 | 0.8355 | **0.3265** | +0.5090 | drifted (different image) | same-image | Native |
| **1938502** | **0.99420** | 0.99180 | **+0.00240** | **0.2541** | 0.3639 | **−0.1098** | same-image | same-image | **HSWQ** |
| **4710928** | **0.83926** | 0.78285 | **+0.05641** | **4.2280** | 5.6020 | **−1.3740** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.91225** | 0.87960 | **+0.03265** | **3.7460** | 5.1390 | **−1.3930** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | 0.96071 | **0.97991** | −0.01920 | 2.1770 | **1.1000** | +1.0770 | drifted (different image) | drifted (different image) | Native |
| **36192847** | **0.99891** | 0.95509 | **+0.04382** | **0.0464** | 1.8720 | **−1.8256** | same-image | drifted (different image) | **HSWQ** |
| **71058294** | **0.98396** | 0.97952 | **+0.00444** | **0.7653** | 0.9764 | **−0.2111** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | **0.99924** | 0.98498 | **+0.01426** | **0.0359** | 0.7045 | **−0.6686** | same-image | same-image | **HSWQ** |
| **285039184** | 0.98457 | **0.98904** | −0.00447 | 0.7149 | **0.5083** | +0.2066 | same-image | same-image | Native |
| **491730285** | **0.99563** | 0.99030 | **+0.00533** | **0.1973** | 0.4363 | **−0.2390** | same-image | same-image | **HSWQ** |
| **762019483** | **0.97822** | 0.86974 | **+0.10848** | **1.2570** | 7.5330 | **−6.2760** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.99530** | 0.99438 | **+0.00092** | **0.2178** | 0.2588 | **−0.0410** | same-image | same-image | **HSWQ** |
| **1409285713** | **0.99487** | 0.98588 | **+0.00899** | **0.2042** | 0.5584 | **−0.3542** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99350** | 0.99200 | **+0.00150** | **0.1876** | 0.2279 | **−0.0403** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99707** | 0.99657 | **+0.00050** | **0.1437** | 0.1682 | **−0.0245** | same-image | same-image | **HSWQ** |
| **4195820371** | 0.99532 | **0.99542** | −0.00010 | 0.2204 | **0.2162** | +0.0042 | same-image | same-image | Native |
| **Mean** | **0.97332** | 0.96174 | **+0.01158** | **0.9861** | 1.5077 | **−0.5216** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

### 2.10. 0.9chilledReGenericV3+0.1BRA7 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97352** | 0.96501 | **+0.00851** |
| **Min Final Cosine** (↑ better) | **0.70374** | 0.69914 | **+0.00460** |
| **Max Final Cosine** (↑ better) | **0.99954** | 0.99844 | **+0.00110** |
| **Mean Final Latent MSE** (↓ better) | **0.7374** | 0.9495 | **−0.2121 (22.3% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+24.1%** (1.77s → 1.34s) | +16.2% (1.74s → 1.46s) | — |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 14/25 same-image, 11/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.9chilledReGenericV3+0.1BRA7)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.96060 | **0.97191** | −0.01131 | 1.4410 | **1.0050** | +0.4360 | drifted (different image) | drifted (different image) | Native |
| **137** | 0.93406 | **0.97069** | −0.03663 | 1.9780 | **0.8730** | +1.1050 | drifted (different image) | drifted (different image) | Native |
| **849** | **0.99935** | 0.99783 | **+0.00152** | **0.0168** | 0.0546 | **−0.0378** | same-image | same-image | **HSWQ** |
| **2024** | **0.95179** | 0.91598 | **+0.03581** | **1.0080** | 1.7360 | **−0.7280** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | 0.99537 | **0.99662** | −0.00125 | 0.1061 | **0.0788** | +0.0273 | same-image | same-image | Native |
| **18429** | **0.99843** | 0.99100 | **+0.00743** | **0.0656** | 0.3760 | **−0.3104** | same-image | same-image | **HSWQ** |
| **53082** | **0.99831** | 0.99804 | **+0.00027** | **0.0409** | 0.0473 | **−0.0064** | same-image | same-image | **HSWQ** |
| **149206** | **0.99251** | 0.99046 | **+0.00205** | **0.1869** | 0.2357 | **−0.0488** | same-image | same-image | **HSWQ** |
| **382715** | **0.99948** | 0.99747 | **+0.00201** | **0.0117** | 0.0573 | **−0.0456** | same-image | same-image | **HSWQ** |
| **826401** | **0.97873** | 0.93259 | **+0.04614** | **0.5463** | 1.7090 | **−1.1627** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.97223** | 0.93861 | **+0.03362** | **0.7180** | 1.5750 | **−0.8570** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.99643** | 0.97184 | **+0.02459** | **0.0680** | 0.5339 | **−0.4659** | same-image | drifted (different image) | **HSWQ** |
| **8391642** | **0.99906** | 0.99781 | **+0.00125** | **0.0268** | 0.0633 | **−0.0365** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99724** | 0.99350 | **+0.00374** | **0.1074** | 0.2543 | **−0.1469** | same-image | same-image | **HSWQ** |
| **36192847** | **0.99798** | 0.99429 | **+0.00369** | **0.0459** | 0.1291 | **−0.0832** | same-image | same-image | **HSWQ** |
| **71058294** | **0.99518** | 0.99455 | **+0.00063** | **0.1617** | 0.1820 | **−0.0203** | same-image | same-image | **HSWQ** |
| **128491703** | **0.99710** | 0.97976 | **+0.01734** | **0.1277** | 0.8841 | **−0.7564** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | 0.99566 | **0.99762** | −0.00196 | 0.1109 | **0.0609** | +0.0500 | same-image | same-image | Native |
| **491730285** | **0.99954** | 0.99844 | **+0.00110** | **0.0137** | 0.0455 | **−0.0318** | same-image | same-image | **HSWQ** |
| **762019483** | **0.70374** | 0.69914 | **+0.00460** | 8.5500 | **8.4020** | +0.1480 | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | **0.99918** | 0.98198 | **+0.01720** | **0.0233** | 0.5064 | **−0.4831** | same-image | same-image | **HSWQ** |
| **1409285713** | 0.96508 | **0.97188** | −0.00680 | 0.7538 | **0.6020** | +0.1518 | drifted (different image) | drifted (different image) | Native |
| **2683910547** | **0.99700** | 0.99398 | **+0.00302** | **0.0753** | 0.1478 | **−0.0725** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.97745** | 0.93272 | **+0.04473** | **0.7200** | 2.1370 | **−1.4170** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.93647** | 0.91643 | **+0.02004** | **1.5320** | 2.0250 | **−0.4930** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.97352** | 0.96501 | **+0.00851** | **0.7374** | 0.9495 | **−0.2121** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (20/25)** |

### 2.11. 0.9chilledReGenericV3+0.1majicmixRealistic_v7 (1on k80)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98043** | 0.96837 | **+0.01206** |
| **Min Final Cosine** (↑ better) | 0.86281 | **0.87371** | −0.01090 |
| **Max Final Cosine** (↑ better) | **0.99979** | 0.99785 | **+0.00194** |
| **Mean Final Latent MSE** (↓ better) | **0.5168** | 0.7227 | **−0.2059 (28.5% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | **0/25 (0%)** | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | +8.7% (1.82s → 1.66s) | **+25.2%** (1.82s → 1.36s) | — |
| **Trajectory Verdict** | 19/25 same-image, 6/25 drifted | 14/25 same-image, 11/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (0.9chilledReGenericV3+0.1majicmixRealistic_v7)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.90192 | **0.96272** | −0.06080 | 3.0810 | **1.1510** | +1.9300 | drifted (different image) | drifted (different image) | Native |
| **137** | **0.94944** | 0.93206 | **+0.01738** | **1.4470** | 1.9370 | **−0.4900** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.99810** | 0.99636 | **+0.00174** | **0.0466** | 0.0890 | **−0.0424** | same-image | same-image | **HSWQ** |
| **2024** | **0.99390** | 0.89781 | **+0.09609** | **0.1196** | 2.0100 | **−1.8904** | same-image | drifted (different image) | **HSWQ** |
| **7391** | **0.99833** | 0.99754 | **+0.00079** | **0.0367** | 0.0539 | **−0.0172** | same-image | same-image | **HSWQ** |
| **18429** | **0.99329** | 0.99190 | **+0.00139** | **0.2798** | 0.3379 | **−0.0581** | same-image | same-image | **HSWQ** |
| **53082** | **0.95892** | 0.90103 | **+0.05789** | **0.8835** | 2.0850 | **−1.2015** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.99907** | 0.98050 | **+0.01857** | **0.0220** | 0.4568 | **−0.4348** | same-image | same-image | **HSWQ** |
| **382715** | **0.99942** | 0.97709 | **+0.02233** | **0.0125** | 0.4933 | **−0.4808** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.99636** | 0.96610 | **+0.03026** | **0.0981** | 0.8976 | **−0.7995** | same-image | drifted (different image) | **HSWQ** |
| **1938502** | **0.99269** | 0.95713 | **+0.03556** | **0.1737** | 1.0330 | **−0.8593** | same-image | drifted (different image) | **HSWQ** |
| **4710928** | **0.99227** | 0.87371 | **+0.11856** | **0.1403** | 2.3200 | **−2.1797** | same-image | drifted (different image) | **HSWQ** |
| **8391642** | **0.99979** | 0.99643 | **+0.00336** | **0.0056** | 0.0949 | **−0.0893** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99146** | 0.97751 | **+0.01395** | **0.3115** | 0.8241 | **−0.5126** | same-image | drifted (different image) | **HSWQ** |
| **36192847** | **0.99798** | 0.99392 | **+0.00406** | **0.0438** | 0.1311 | **−0.0873** | same-image | same-image | **HSWQ** |
| **71058294** | **0.99609** | 0.99603 | **+0.00006** | **0.1238** | 0.1249 | **−0.0011** | same-image | same-image | **HSWQ** |
| **128491703** | **0.99542** | 0.99435 | **+0.00107** | **0.1978** | 0.2414 | **−0.0436** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99636** | 0.99626 | **+0.00010** | **0.0901** | 0.0932 | **−0.0031** | same-image | same-image | **HSWQ** |
| **491730285** | **0.97313** | 0.96613 | **+0.00700** | **0.7628** | 0.9616 | **−0.1988** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | 0.86281 | **0.98757** | −0.12476 | 3.5020 | **0.2844** | +3.2176 | drifted (different image) | same-image | Native |
| **938174026** | **0.99885** | 0.99785 | **+0.00100** | **0.0303** | 0.0574 | **−0.0271** | same-image | same-image | **HSWQ** |
| **1409285713** | **0.94200** | 0.89946 | **+0.04254** | **1.1180** | 1.9260 | **−0.8080** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | **0.99641** | 0.99451 | **+0.00190** | **0.0860** | 0.1329 | **−0.0469** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99647** | 0.99503 | **+0.00144** | **0.1115** | 0.1581 | **−0.0466** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99024** | 0.98023 | **+0.01001** | **0.2370** | 0.4710 | **−0.2340** | same-image | same-image | **HSWQ** |
| **Mean** | **0.98043** | 0.96837 | **+0.01206** | **0.5168** | 0.7227 | **−0.2059** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (23/25)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Bifurcation behaviour:** across the 11 checkpoints (25 seed evaluations per arm, 275 seeds total), neither Native ConvRot INT8 nor HSWQ ConvRot INT8 produces a single bifurcated seed (0/275 each). HSWQ significantly raises the same-image count to 174/275 vs 135/275 for Native.

2. **Cosine gain:** the family mean final cosine is 0.97133 for HSWQ vs 0.95664 for Native (+0.01469 gain). HSWQ is ahead on mean cosine in 10/11 models; the fourth checkpoint is the exception - see finding 8.

3. **Latent MSE reduction:** family mean final latent MSE is 0.9386 (HSWQ) vs 1.4468 (Native), about 35.1% lower error.

4. **Worst-case robustness:** the worst per-model minimum cosine is 0.69419 (HSWQ) vs 0.63851 (Native) observed on the seventh checkpoint; HSWQ prevents the near-bifurcation collapse seen in Native.

5. **Wall-clock:** HSWQ averages a +24.4% speedup over FP16 per seed across the family.

6. **The second checkpoint is harder for both arms.** `0.5xxmix9realistic_v40+0.5chilledReGenericV3` sits below `0.5xxmix9realistic_v40 0.5BRA7` on every aggregate for both HSWQ and Native; the HSWQ/Native gap is nonetheless wider there (+0.02718 vs +0.01277), so the protection buys more on the harder model, not less.

7. **The third checkpoint remains the strongest HSWQ case.** `0.5yayoiMix_v25+0.5againmix_v22` gives HSWQ its highest same-image count on a single checkpoint (18/25 against 11/25) and its largest latent-MSE reduction (42%), while also showing the widest wall-clock gain (+38.6%). The family worst case stays with the second checkpoint (0.78997).

8. **The fourth checkpoint is the one Native takes.** `0.5yayoiMix_v25+0.5voidnoisecore_r1733` is the only checkpoint here where the Native pack beats the protected one on the mean final cosine (0.97518 against 0.97461, −0.00057) and on the minimum (0.91578 against 0.90601); HSWQ wins the per-seed count by cosine (15/25 against 10/25). HSWQ keeps the lower mean MSE (0.7950 against 0.8165), the higher same-image count (16/25 against 15/25) and the larger speedup (35.1% against 28.0%). The family aggregate stays firmly with HSWQ (10/11 models).

9. **The fifth checkpoint returns the widest same-image margin of the set.** `0.5yayoiMix_v25+0.5xxmix9realistic_v40` gives HSWQ 16/25 against 12/25 for Native, a gap of 4, and its per-seed cosine count follows (18/25 against 7/25). Its minimum cosine (0.74845) is the second lowest of the five HSWQ arms, so the family worst case stays with the second checkpoint (0.74845).

10. **The sixth checkpoint (`0.7_0.6hassaku+0.4dreamshaper+0.3anylora`) establishes the highest fidelity of the suite.** Reaches a mean final cosine of 0.98061 (vs 0.95456 Native, +0.02605 gain) with 16/25 same-image seeds (vs 9/25 Native) and a 60.2% latent MSE reduction (0.6839 vs 1.7171). HSWQ wins 23 of 25 seeds on this model.

11. **The seventh checkpoint (`0.7_0.7hassaku+0.3dreamshaper_8_0.3anylora`) demonstrates high-protection stability on complex multi-model merges.** Evaluated under `1on k82`, HSWQ achieves a mean final cosine of 0.96012 vs 0.94130 (+0.01882) and 14/25 same-image seeds vs 11/25 for Native. Under Native quantization, seed 42 experiences severe degradation down to cosine 0.63851 with a step drop of 0.0469 (borderline bifurcation), whereas HSWQ maintains 0.88097, keeping the generation coherently anchored. HSWQ wins 21 of 25 seeds.

12. **The eighth checkpoint (`0.7chilledReGenericV3+0.3uberRealisticMerge_v13`) shows robust MSE reduction.** Under `1on k80`, HSWQ reaches a mean final cosine of 0.97332 vs 0.95932 (+0.01400 gain) and cuts latent MSE from 1.0755 down to 0.7196 (33.1% error reduction). In severe native drift cases like seed 2024 (native 0.79971, MSE 3.9080) and seed 149206 (native 0.83337, MSE 4.5340), HSWQ preserves coherent trajectories at 0.92155 and 0.96974.

13. **The ninth checkpoint (`0.7xxmix9realistic_v40+0.3BRA7`) demonstrates strong seed consistency.** Under `1on k80`, HSWQ achieves 0.97332 mean cosine vs 0.96174 Native (+0.01158 gain) and 18/25 same-image seeds vs 16/25 Native, reducing MSE from 1.5077 to 0.9861 (34.6% error reduction). HSWQ prevents extreme native tail degradation on seed 762019483 (native 0.86974, MSE 7.5330 -> HSWQ 0.97822, MSE 1.2570).

14. **The tenth checkpoint (`0.9chilledReGenericV3+0.1BRA7`) highlights precision anchoring.** Evaluated under `1on k80`, HSWQ delivers 0.97352 mean cosine vs 0.96501 Native (+0.00851 gain) and cuts MSE from 0.9495 to 0.7374 (22.3% error reduction). Seed 826401 drifts under native (0.93259, MSE 1.7090) while HSWQ maintains 0.97873 (MSE 0.5463). HSWQ wins 20 of 25 seeds.

15. **The eleventh checkpoint (`0.9chilledReGenericV3+0.1majicmixRealistic_v7`) delivers the highest same-image rate.** Reaches 0.98043 mean cosine vs 0.96837 Native (+0.01206 gain) and an impressive 19/25 same-image seeds (76%) vs 14/25 Native, with MSE reduced to 0.5168 vs 0.7227 (28.5% error reduction). Extreme native deviations on seed 2024 (native 0.89781, MSE 2.0100) and seed 4710928 (native 0.87371, MSE 2.3200) are completely eliminated by HSWQ (0.99390 and 0.99227). HSWQ wins 23 of 25 seeds.


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
