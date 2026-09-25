# SDXL ConvRot INT8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ ConvRot INT8** vs **Native ConvRot INT8** on the SDXL architecture family.

**Source:** `benchmark result/score_sdxl_int8.txt`  
**Evaluation Script:** `benchmark/sdxl_int8_traj_compare.py` (25-seed deterministic trajectory analysis)

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `reNNN` | Impact-analysis **re-estimation round** used for layer selection |
| `1on` | Bias correction **ON** |

---

## 1. Summary Comparison (HSWQ ConvRot INT8 vs Native ConvRot INT8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | HSWQ Speedup (↑) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **waiIllustriousSDXL_v170** | 1on re597 | **0.96230** | 0.93874 | **+0.02356** | **1.5957** | 2.6284 | **0/25 (0%)** | 0/25 (0%) | **+16.3%** | **HSWQ** |
| **prefectIllustriousXL_v8** | 1on re610 | **0.97100** | 0.96072 | **+0.01028** | **1.2351** | 1.6720 | **0/25 (0%)** | 0/25 (0%) | **+14.4%** | **HSWQ** |
| **novaAnimeXL_ilV190** | 1on re599 | **0.95882** | 0.94045 | **+0.01837** | **2.5678** | 3.6743 | **0/25 (0%)** | 0/25 (0%) | **+14.4%** | **HSWQ** |
| **waiREALCN_v150** | 1on re630 | **0.97979** | 0.95700 | **+0.02279** | **0.5952** | 1.3423 | **0/25 (0%)** | 0/25 (0%) | **+17.2%** | **HSWQ** |
| **oneObsession_v24** | 1on re572 | **0.94184** | 0.92192 | **+0.01992** | **3.3086** | 4.5380 | **0/25 (0%)** | 0/25 (0%) | **+15.2%** | **HSWQ** |
| **bluePencilXL_v031** | 1on re570 | **0.94755** | 0.90431 | **+0.04324** | **2.9047** | 5.1436 | **0/25 (0%)** | 0/25 (0%) | **+18.2%** | **HSWQ** |
| **JANKUTrainedChenkinNoobai_v777** | 1on re550 | **0.96362** | 0.94900 | **+0.01462** | **1.7861** | 2.7012 | **0/25 (0%)** | 0/25 (0%) | **+16.3%** | **HSWQ** |
| **epicrealismXL_pureFix** | 1on re570 | **0.98419** | 0.97888 | **+0.00531** | **0.6658** | 0.8696 | **0/25 (0%)** | 0/25 (0%) | **+20.6%** | **HSWQ** |
| **koronemixIllustrious_v70** | 1on re550 | **0.97224** | 0.94886 | **+0.02338** | **1.3348** | 2.3778 | **0/25 (0%)** | 0/25 (0%) | **+16.3%** | **HSWQ** |
| **koronemixVpred_v20** | 1on re550 | **0.95351** | 0.94315 | **+0.01036** | **3.6009** | 4.2807 | **1/25 (4%)** | 3/25 (12%) | **+16.9%** | **HSWQ** |
| **novaAsianXL_illustriousV70** | 1on re550 | **0.98783** | 0.97515 | **+0.01268** | **0.4571** | 0.9429 | **0/25 (0%)** | 0/25 (0%) | **+15.3%** | **HSWQ** |
| **realvisxlV30_v30TurboBakedvae** | 1on re650 | **0.96184** | 0.94210 | **+0.01974** | **2.2952** | 3.5083 | **0/25 (0%)** | 0/25 (0%) | **+6.5%** | **HSWQ** |
| **realvisxlV50_v40Bakedvae** | 1on re550 | **0.98860** | 0.98464 | **+0.00396** | **0.5307** | 0.7341 | **0/25 (0%)** | 0/25 (0%) | **+10.9%** | **HSWQ** |
| **realvisxlV50_v50Bakedvae** | 1on re550 | **0.98747** | 0.97734 | **+0.01013** | **0.5568** | 1.0037 | **0/25 (0%)** | 0/25 (0%) | **+10.8%** | **HSWQ** |
| **unholyDesireMixSinister_v90** | 1on re590 | **0.97121** | 0.95367 | **+0.01754** | **1.4192** | 2.2342 | **0/25 (0%)** | 0/25 (0%) | **+13.2%** | **HSWQ** |
| **uwazumimixILL_v50** | 1on re720 | **0.98034** | 0.97903 | **+0.00131** | **0.3771** | 0.4101 | **0/25 (0%)** | 0/25 (0%) | **+8.3%** | **HSWQ** |
| **waiANIPONYXL_v90** | 1on re650 | **0.98075** | 0.94910 | **+0.03165** | **0.7713** | 2.0174 | **0/25 (0%)** | 0/25 (0%) | **+11.3%** | **HSWQ** |
| **waiANIPONYXL_v140** | 1on re650 | **0.95741** | 0.93961 | **+0.01780** | **1.9102** | 2.7537 | **0/25 (0%)** | 0/25 (0%) | **+13.6%** | **HSWQ** |
| **waiREALISM_v10** | 1on re590 | **0.98091** | 0.97399 | **+0.00692** | **0.5663** | 0.7193 | **0/25 (0%)** | 0/25 (0%) | **+18.0%** | **HSWQ** |
| **Family Average** | — | **0.97006** | 0.95356 | **+0.01650** | **1.4989** | 2.2922 | **1/475 (0.2%)** | 3/475 (0.6%) | **+14.4%** | **HSWQ (19/19 models)** |

---

## 2. Detailed Results per Model

### 2.1. waiIllustriousSDXL_v170 (1on re597)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96230** | 0.93874 | **+0.02356** |
| **Min Final Cosine** (↑ better) | **0.83745** | 0.82858 | **+0.00887** |
| **Max Final Cosine** (↑ better) | **0.99675** | 0.98945 | **+0.00730** |
| **Mean Final Latent MSE** (↓ better) | **1.5957** | 2.6284 | **−1.0328 (39% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+16.3%** (9.45s → 7.91s) | +0.7% (9.13s → 9.07s) | — |
| **Trajectory Verdict** | 14/25 same-image, 11/25 drifted | 5/25 same-image, 20/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (waiIllustriousSDXL_v170)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98055** | 0.93136 | **+0.04919** | **1.1900** | 4.2190 | **−3.0290** | same-image | drifted (different image) | **HSWQ** |
| **137** | 0.94894 | **0.95036** | −0.00142 | 1.6890 | **1.6440** | +0.0450 | drifted (different image) | drifted (different image) | Native |
| **849** | **0.99132** | 0.96307 | **+0.02825** | **0.3762** | 1.6160 | **−1.2398** | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.99299** | 0.98573 | **+0.00726** | **0.3194** | 0.6513 | **−0.3319** | same-image | same-image | **HSWQ** |
| **7391** | **0.99132** | 0.89959 | **+0.09173** | **0.3182** | 3.6870 | **−3.3688** | same-image | drifted (different image) | **HSWQ** |
| **18429** | **0.89240** | 0.87191 | **+0.02049** | **2.8550** | 3.3800 | **−0.5250** | drifted (different image) | drifted (different image) | **HSWQ** |
| **53082** | **0.93095** | 0.85114 | **+0.07981** | **3.5200** | 7.6710 | **−4.1510** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | 0.83745 | **0.94595** | −0.10850 | 7.1700 | **2.3040** | +4.8660 | drifted (different image) | drifted (different image) | Native |
| **382715** | **0.98442** | 0.94187 | **+0.04255** | **0.6997** | 2.6690 | **−1.9693** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.93685** | 0.93371 | **+0.00314** | **2.6450** | 2.7840 | **−0.1390** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.99416** | 0.98945 | **+0.00471** | **0.1627** | 0.3092 | **−0.1465** | same-image | same-image | **HSWQ** |
| **4710928** | **0.84857** | 0.82858 | **+0.01999** | **6.1350** | 6.8990 | **−0.7640** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | 0.95564 | **0.96381** | −0.00817 | 1.7020 | **1.3900** | +0.3120 | drifted (different image) | drifted (different image) | Native |
| **15820493** | 0.97487 | **0.98224** | −0.00737 | 1.4230 | **1.0020** | +0.4210 | drifted (different image) | same-image | Native |
| **36192847** | 0.98042 | **0.98734** | −0.00692 | 1.0770 | **0.7060** | +0.3710 | same-image | same-image | Native |
| **71058294** | **0.99422** | 0.95820 | **+0.03602** | **0.2671** | 1.9400 | **−1.6729** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | **0.99675** | 0.97564 | **+0.02111** | **0.1713** | 1.2830 | **−1.1117** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | **0.97463** | 0.96144 | **+0.01319** | **1.3540** | 2.0520 | **−0.6980** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.98419** | 0.91100 | **+0.07319** | **0.5830** | 3.2830 | **−2.7000** | same-image | drifted (different image) | **HSWQ** |
| **762019483** | **0.98940** | 0.97894 | **+0.01046** | **0.5994** | 1.2010 | **−0.6016** | same-image | drifted (different image) | **HSWQ** |
| **938174026** | **0.98030** | 0.96978 | **+0.01052** | **1.1640** | 1.8050 | **−0.6410** | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | **0.98539** | 0.98355 | **+0.00184** | **0.5698** | 0.6437 | **−0.0739** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98864** | 0.87255 | **+0.11609** | **0.4154** | 4.6760 | **−4.2606** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.95818** | 0.85866 | **+0.09952** | **1.9150** | 6.6550 | **−4.7400** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | 0.96493 | **0.97257** | −0.00764 | 1.5710 | **1.2410** | +0.3300 | drifted (different image) | drifted (different image) | Native |
| **Mean** | **0.96230** | **0.93874** | **+0.02356** | **1.5957** | **2.6284** | **−1.0328** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

### 2.2. prefectIllustriousXL_v8 (1on re610)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97100** | 0.96072 | **+0.01028** |
| **Min Final Cosine** (↑ better) | **0.85930** | 0.82258 | **+0.03672** |
| **Max Final Cosine** (↑ better) | **0.99286** | 0.99352 | **−0.00066** |
| **Mean Final Latent MSE** (↓ better) | **1.2351** | 1.6720 | **−0.4369 (26% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+14.4%** (9.18s → 7.86s) | +3.7% (9.14s → 8.81s) | — |
| **Trajectory Verdict** | 13/25 same-image, 12/25 drifted | 10/25 same-image, 15/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (prefectIllustriousXL_v8)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97255** | 0.95632 | **+0.01623** | **1.5700** | 2.5210 | **−0.9510** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.98276** | 0.95960 | **+0.02316** | **0.5101** | 1.2040 | **−0.6939** | same-image | drifted (different image) | **HSWQ** |
| **849** | 0.95512 | **0.98089** | −0.02577 | 1.7770 | **0.7607** | +1.0163 | drifted (different image) | same-image | Native |
| **2024** | **0.90075** | 0.90004 | **+0.00071** | **4.2590** | 4.3340 | **−0.0750** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.98349** | 0.98201 | **+0.00148** | **0.7466** | 0.8205 | **−0.0739** | same-image | same-image | **HSWQ** |
| **18429** | **0.95714** | 0.91612 | **+0.04102** | **1.5020** | 2.9510 | **−1.4490** | drifted (different image) | drifted (different image) | **HSWQ** |
| **53082** | **0.98729** | 0.94765 | **+0.03964** | **0.6511** | 2.7190 | **−2.0679** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.98969** | 0.97187 | **+0.01782** | **0.4616** | 1.2630 | **−0.8014** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.98838** | 0.98606 | **+0.00232** | **0.4583** | 0.5571 | **−0.0988** | same-image | same-image | **HSWQ** |
| **826401** | **0.98685** | 0.97400 | **+0.01285** | **0.5131** | 1.0190 | **−0.5059** | same-image | drifted (different image) | **HSWQ** |
| **1938502** | 0.97782 | **0.99352** | −0.01570 | 0.7820 | **0.2293** | +0.5527 | drifted (different image) | same-image | Native |
| **4710928** | 0.97451 | **0.99161** | −0.01710 | 1.0760 | **0.3565** | +0.7195 | drifted (different image) | same-image | Native |
| **8391642** | **0.97580** | 0.96357 | **+0.01223** | **0.8506** | 1.2970 | **−0.4464** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | 0.98759 | **0.98977** | −0.00218 | 0.6711 | **0.5605** | +0.1106 | same-image | same-image | Native |
| **36192847** | **0.98504** | 0.94271 | **+0.04233** | **0.7274** | 2.8040 | **−2.0766** | same-image | drifted (different image) | **HSWQ** |
| **71058294** | 0.97290 | **0.98452** | −0.01162 | 1.2300 | **0.7153** | +0.5147 | drifted (different image) | same-image | Native |
| **128491703** | **0.98737** | 0.96169 | **+0.02568** | **0.6385** | 1.9480 | **−1.3095** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | **0.85930** | 0.82258 | **+0.03672** | **5.6010** | 6.8450 | **−1.2440** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.97157** | 0.96629 | **+0.00528** | **1.1370** | 1.3650 | **−0.2280** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | 0.94049 | **0.95420** | −0.01371 | 3.0330 | **2.3390** | +0.6940 | drifted (different image) | drifted (different image) | Native |
| **938174026** | 0.97905 | **0.98043** | −0.00138 | 1.0760 | **1.0100** | +0.0660 | drifted (different image) | same-image | Native |
| **1409285713** | **0.99249** | 0.98394 | **+0.00855** | **0.2929** | 0.6323 | **−0.3394** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98550** | 0.94498 | **+0.04052** | **0.5378** | 2.0240 | **−1.4862** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99286** | 0.99030 | **+0.00256** | **0.3177** | 0.4388 | **−0.1211** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.98872** | 0.97343 | **+0.01529** | **0.4572** | 1.0860 | **−0.6288** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.97100** | **0.96072** | **+0.01028** | **1.2351** | **1.6720** | **−0.4369** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (18/25)** |

---

### 2.3. novaAnimeXL_ilV190 (1on re599)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95882** | 0.94045 | **+0.01837** |
| **Min Final Cosine** (↑ better) | **0.84916** | 0.80572 | **+0.04344** |
| **Max Final Cosine** (↑ better) | **0.99507** | 0.99324 | **+0.00183** |
| **Mean Final Latent MSE** (↓ better) | **2.5678** | 3.6743 | **−1.1065 (30% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+14.4%** (9.16s → 7.85s) | +1.1% (9.32s → 9.22s) | — |
| **Trajectory Verdict** | 10/25 same-image, 15/25 drifted | 6/25 same-image, 19/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (novaAnimeXL_ilV190)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97427** | 0.94668 | **+0.02759** | **1.8680** | 3.8330 | **−1.9650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.99507** | 0.98063 | **+0.01444** | **0.2540** | 0.9891 | **−0.7351** | same-image | same-image | **HSWQ** |
| **849** | **0.90647** | 0.89799 | **+0.00848** | **5.1740** | 5.6640 | **−0.4900** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.98142** | 0.96539 | **+0.01603** | **1.2900** | 2.3820 | **−1.0920** | same-image | drifted (different image) | **HSWQ** |
| **7391** | **0.94806** | 0.92398 | **+0.02408** | **2.9470** | 4.2730 | **−1.3260** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.99391** | 0.97179 | **+0.02212** | **0.3069** | 1.4230 | **−1.1161** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.92728** | 0.91967 | **+0.00761** | **4.8720** | 5.4130 | **−0.5410** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.95938** | 0.92304 | **+0.03634** | **2.7930** | 5.2710 | **−2.4780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **382715** | **0.96301** | 0.87078 | **+0.09223** | **2.1880** | 7.4660 | **−5.2780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.96959** | 0.94607 | **+0.02352** | **1.6590** | 2.9370 | **−1.2780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.97206** | 0.94031 | **+0.03175** | **1.2100** | 2.6900 | **−1.4800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.97611** | 0.95731 | **+0.01880** | **1.6150** | 2.9190 | **−1.3040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.86722** | 0.86720 | **+0.00002** | 7.3270 | **7.3060** | +0.0210 | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.84916** | 0.80572 | **+0.04344** | **12.5200** | 16.2000 | **−3.6800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | 0.98630 | **0.99324** | −0.00694 | 1.1010 | **0.5429** | +0.5581 | same-image | same-image | Native |
| **71058294** | **0.96862** | 0.93181 | **+0.03681** | **1.8310** | 3.9650 | **−2.1340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.99387** | 0.98740 | **+0.00647** | **0.5048** | 1.0350 | **−0.5302** | same-image | same-image | **HSWQ** |
| **285039184** | 0.96540 | **0.97868** | −0.01328 | 1.7840 | **1.0960** | +0.6880 | drifted (different image) | drifted (different image) | Native |
| **491730285** | **0.90717** | 0.87559 | **+0.03158** | **4.6660** | 6.1770 | **−1.5110** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | **0.98472** | 0.98023 | **+0.00449** | **1.0570** | 1.3630 | **−0.3060** | same-image | same-image | **HSWQ** |
| **938174026** | **0.98891** | 0.98143 | **+0.00748** | **0.7902** | 1.3240 | **−0.5338** | same-image | same-image | **HSWQ** |
| **1409285713** | 0.93615 | **0.98377** | −0.04762 | 3.9820 | **0.9994** | +2.9826 | drifted (different image) | same-image | Native |
| **2683910547** | **0.98623** | 0.94248 | **+0.04375** | **0.7517** | 3.1390 | **−2.3873** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.98539** | 0.96370 | **+0.02169** | **0.8869** | 2.1840 | **−1.2971** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.98480** | 0.97645 | **+0.00835** | **0.8171** | 1.2670 | **−0.4499** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.95882** | **0.94045** | **+0.01837** | **2.5678** | **3.6743** | **−1.1065** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (22/25)** |

---

### 2.4. waiREALCN_v150 (1on re630)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97979** | 0.95700 | **+0.02279** |
| **Min Final Cosine** (↑ better) | **0.83068** | 0.69028 | **+0.14040** |
| **Max Final Cosine** (↑ better) | **0.99732** | 0.99364 | **+0.00368** |
| **Mean Final Latent MSE** (↓ better) | **0.5952** | 1.3423 | **−0.7470 (56% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+17.2%** (9.16s → 7.59s) | +6.7% (9.52s → 8.88s) | — |
| **Trajectory Verdict** | 21/25 same-image, 4/25 drifted | 16/25 same-image, 9/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (waiREALCN_v150)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99269** | 0.97259 | **+0.02010** | **0.2372** | 0.8982 | **−0.6610** | same-image | drifted (different image) | **HSWQ** |
| **137** | 0.83068 | **0.88804** | −0.05736 | 4.9210 | **3.1600** | +1.7610 | drifted (different image) | drifted (different image) | Native |
| **849** | **0.99621** | 0.99101 | **+0.00520** | **0.1180** | 0.2788 | **−0.1608** | same-image | same-image | **HSWQ** |
| **2024** | **0.99638** | 0.98782 | **+0.00856** | **0.1168** | 0.4032 | **−0.2864** | same-image | same-image | **HSWQ** |
| **7391** | 0.98359 | **0.98903** | −0.00544 | 0.4452 | **0.2985** | +0.1467 | same-image | same-image | Native |
| **18429** | **0.99469** | 0.99097 | **+0.00372** | **0.1631** | 0.2784 | **−0.1153** | same-image | same-image | **HSWQ** |
| **53082** | **0.99619** | 0.83808 | **+0.15811** | **0.1346** | 5.5560 | **−5.4214** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.99732** | 0.99363 | **+0.00369** | **0.0739** | 0.1750 | **−0.1011** | same-image | same-image | **HSWQ** |
| **382715** | **0.99492** | 0.98705 | **+0.00787** | **0.1407** | 0.3578 | **−0.2171** | same-image | same-image | **HSWQ** |
| **826401** | **0.99656** | 0.99042 | **+0.00614** | **0.1036** | 0.2864 | **−0.1828** | same-image | same-image | **HSWQ** |
| **1938502** | **0.87890** | 0.69028 | **+0.18862** | **3.3900** | 9.0700 | **−5.6800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.99595** | 0.99234 | **+0.00361** | **0.1445** | 0.2752 | **−0.1307** | same-image | same-image | **HSWQ** |
| **8391642** | **0.98995** | 0.98956 | **+0.00039** | **0.2793** | 0.2906 | **−0.0113** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99410** | 0.90897 | **+0.08513** | **0.1682** | 2.5710 | **−2.4028** | same-image | drifted (different image) | **HSWQ** |
| **36192847** | **0.99398** | 0.99364 | **+0.00034** | **0.2030** | 0.2185 | **−0.0155** | same-image | same-image | **HSWQ** |
| **71058294** | **0.98939** | 0.97712 | **+0.01227** | **0.2892** | 0.6200 | **−0.3308** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | **0.99123** | 0.99077 | **+0.00046** | **0.2487** | 0.2673 | **−0.0186** | same-image | same-image | **HSWQ** |
| **285039184** | 0.98316 | **0.98818** | −0.00502 | 0.4375 | **0.3063** | +0.1312 | same-image | same-image | Native |
| **491730285** | 0.96573 | **0.97064** | −0.00491 | 0.8790 | **0.7530** | +0.1260 | drifted (different image) | drifted (different image) | Native |
| **762019483** | **0.99475** | 0.99106 | **+0.00369** | **0.2052** | 0.3526 | **−0.1474** | same-image | same-image | **HSWQ** |
| **938174026** | **0.96457** | 0.86510 | **+0.09947** | **1.3270** | 5.1310 | **−3.8040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1409285713** | **0.99410** | 0.99181 | **+0.00229** | **0.1787** | 0.2492 | **−0.0705** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99647** | 0.97148 | **+0.02499** | **0.1130** | 0.9140 | **−0.8010** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99110** | 0.99056 | **+0.00054** | **0.2909** | 0.3094 | **−0.0185** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99221** | 0.98474 | **+0.00747** | **0.2728** | 0.5359 | **−0.2631** | same-image | same-image | **HSWQ** |
| **Mean** | **0.97979** | **0.95700** | **+0.02279** | **0.5952** | **1.3423** | **−0.7470** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.5. oneObsession_v24 (1on re572)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.94184** | 0.92192 | **+0.01992** |
| **Min Final Cosine** (↑ better) | **0.83022** | 0.86570 | **−0.03548** |
| **Max Final Cosine** (↑ better) | **0.98710** | 0.97748 | **+0.00962** |
| **Mean Final Latent MSE** (↓ better) | **3.3086** | 4.5380 | **−1.2294 (27% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+15.2%** (9.35s → 7.93s) | +2.2% (9.24s → 9.04s) | — |
| **Trajectory Verdict** | 3/25 same-image, 22/25 drifted | 25/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (oneObsession_v24)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.96778 | **0.96962** | −0.00184 | 1.8290 | **1.7100** | +0.1190 | drifted (different image) | drifted (different image) | Native |
| **137** | **0.97913** | 0.91220 | **+0.06693** | **1.3840** | 5.8530 | **−4.4690** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.95993** | 0.90381 | **+0.05612** | **2.1670** | 5.0930 | **−2.9260** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | 0.84554 | **0.87216** | −0.02662 | 8.0880 | **6.8250** | +1.2630 | drifted (different image) | drifted (different image) | Native |
| **7391** | **0.94333** | 0.86727 | **+0.07606** | **3.0740** | 7.1060 | **−4.0320** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.97878** | 0.96402 | **+0.01476** | **1.4490** | 2.4260 | **−0.9770** | drifted (different image) | drifted (different image) | **HSWQ** |
| **53082** | **0.89617** | 0.88997 | **+0.00620** | **5.1460** | 5.4800 | **−0.3340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.98710** | 0.95753 | **+0.02957** | **0.6585** | 2.1820 | **−1.5235** | same-image | drifted (different image) | **HSWQ** |
| **382715** | 0.93831 | **0.94826** | −0.00995 | 2.9750 | **2.5530** | +0.4220 | drifted (different image) | drifted (different image) | Native |
| **826401** | **0.91461** | 0.90627 | **+0.00834** | **4.4180** | 4.8350 | **−0.4170** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | 0.91262 | **0.93017** | −0.01755 | 4.9260 | **3.8110** | +1.1150 | drifted (different image) | drifted (different image) | Native |
| **4710928** | 0.94806 | **0.95528** | −0.00722 | 3.0240 | **2.6590** | +0.3650 | drifted (different image) | drifted (different image) | Native |
| **8391642** | **0.93928** | 0.92350 | **+0.01578** | **3.5900** | 4.5280 | **−0.9380** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.97746** | 0.94568 | **+0.03178** | **1.3750** | 3.3610 | **−1.9860** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.97796** | 0.96384 | **+0.01412** | **1.4820** | 2.4370 | **−0.9550** | drifted (different image) | drifted (different image) | **HSWQ** |
| **71058294** | 0.89666 | **0.92343** | −0.02677 | 5.3630 | **3.9120** | +1.4510 | drifted (different image) | drifted (different image) | Native |
| **128491703** | **0.98708** | 0.97748 | **+0.00960** | **0.7415** | 1.2940 | **−0.5525** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | **0.96663** | 0.88820 | **+0.07843** | **2.3490** | 7.6450 | **−5.2960** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.96693** | 0.90936 | **+0.05757** | **1.8750** | 5.1000 | **−3.2250** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | **0.92256** | 0.91168 | **+0.01088** | **5.5030** | 6.3160 | **−0.8130** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | 0.83022 | **0.86626** | −0.03604 | 9.6890 | **7.3270** | +2.3620 | drifted (different image) | drifted (different image) | Native |
| **1409285713** | **0.98194** | 0.94153 | **+0.04041** | **1.0350** | 3.3260 | **−2.2910** | same-image | drifted (different image) | **HSWQ** |
| **2683910547** | **0.97880** | 0.96376 | **+0.01504** | **1.2880** | 2.1970 | **−0.9090** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3851729406** | **0.94387** | 0.89104 | **+0.05283** | **3.8150** | 7.4710 | **−3.6560** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.90534** | 0.86570 | **+0.03964** | **5.4710** | 8.0040 | **−2.5330** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.94184** | **0.92192** | **+0.01992** | **3.3086** | **4.5380** | **−1.2294** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (18/25)** |

---

### 2.6. bluePencilXL_v031 (1on re570)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.94755** | 0.90431 | **+0.04324** |
| **Min Final Cosine** (↑ better) | **0.89652** | 0.72054 | **+0.17598** |
| **Max Final Cosine** (↑ better) | **0.98776** | 0.96655 | **+0.02121** |
| **Mean Final Latent MSE** (↓ better) | **2.9047** | 5.1436 | **−2.2389 (44% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+18.2%** (9.56s → 7.82s) | +3.5% (9.20s → 8.88s) | — |
| **Trajectory Verdict** | 2/25 same-image, 23/25 drifted | 25/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (bluePencilXL_v031)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98706** | 0.94578 | **+0.04128** | **0.7301** | 3.0940 | **−2.3639** | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.93937** | 0.87846 | **+0.06091** | **3.6080** | 7.1570 | **−3.5490** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.91252** | 0.87251 | **+0.04001** | **4.1790** | 6.2080 | **−2.0290** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | 0.93680 | **0.94210** | −0.00530 | 4.1300 | **3.7800** | +0.3500 | drifted (different image) | drifted (different image) | Native |
| **7391** | **0.94554** | 0.88887 | **+0.05667** | **2.4480** | 4.9580 | **−2.5100** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98776** | 0.87931 | **+0.10845** | **0.6978** | 7.0070 | **−6.3092** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.93042** | 0.79392 | **+0.13650** | **3.9560** | 11.6800 | **−7.7240** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.96533** | 0.94440 | **+0.02093** | **1.6440** | 2.6460 | **−1.0020** | drifted (different image) | drifted (different image) | **HSWQ** |
| **382715** | **0.96536** | 0.95822 | **+0.00714** | **1.8200** | 2.2020 | **−0.3820** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.97964** | 0.92788 | **+0.05176** | **1.2930** | 4.5440 | **−3.2510** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.95662** | 0.95366 | **+0.00296** | **2.6500** | 2.8230 | **−0.1730** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.89863** | 0.89853 | **+0.00010** | 7.9010 | **7.8980** | +0.0030 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.93532** | 0.76883 | **+0.16649** | **3.3290** | 11.5100 | **−8.1810** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.95144** | 0.72054 | **+0.23090** | **2.2620** | 12.8700 | **−10.6080** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.96857** | 0.90039 | **+0.06818** | **1.5450** | 4.9230 | **−3.3780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **71058294** | **0.95542** | 0.94522 | **+0.01020** | **2.1090** | 2.5920 | **−0.4830** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.97607** | 0.96052 | **+0.01555** | **1.3480** | 2.2310 | **−0.8830** | drifted (different image) | drifted (different image) | **HSWQ** |
| **285039184** | **0.92742** | 0.92505 | **+0.00237** | **3.5570** | 3.6700 | **−0.1130** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | 0.93306 | **0.94051** | −0.00745 | 2.7890 | **2.5030** | +0.2860 | drifted (different image) | drifted (different image) | Native |
| **762019483** | **0.89652** | 0.88719 | **+0.00933** | **6.1810** | 6.7360 | **−0.5550** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | 0.94841 | **0.95464** | −0.00623 | 3.2560 | **2.8630** | +0.3930 | drifted (different image) | drifted (different image) | Native |
| **1409285713** | **0.95898** | 0.93026 | **+0.02872** | **2.1660** | 3.6870 | **−1.5210** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | 0.94897 | **0.96655** | −0.01758 | 3.0890 | **2.0350** | +1.0540 | drifted (different image) | drifted (different image) | Native |
| **3851729406** | **0.96632** | 0.94630 | **+0.02002** | **1.8670** | 2.9860 | **−1.1190** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.91721** | 0.87799 | **+0.03922** | **4.0630** | 5.9870 | **−1.9240** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.94755** | **0.90431** | **+0.04324** | **2.9047** | **5.1436** | **−2.2389** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.7. JANKUTrainedChenkinNoobai_v777 (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96362** | 0.94900 | **+0.01462** |
| **Min Final Cosine** (↑ better) | **0.84802** | 0.84412 | **+0.00390** |
| **Max Final Cosine** (↑ better) | **0.99396** | 0.99367 | **+0.00029** |
| **Mean Final Latent MSE** (↓ better) | **1.7861** | 2.7012 | **−0.9151 (34% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+16.3%** (9.48s → 7.93s) | +2.8% (9.41s → 9.14s) | — |
| **Trajectory Verdict** | 12/25 same-image, 13/25 drifted | 4/25 same-image, 21/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (JANKUTrainedChenkinNoobai_v777)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98711** | 0.84412 | **+0.14299** | **0.7557** | 9.1360 | **−8.3803** | same-image | drifted (different image) | **HSWQ** |
| **137** | 0.89793 | **0.97457** | −0.07664 | 4.4620 | **1.0840** | +3.3780 | drifted (different image) | drifted (different image) | Native |
| **849** | **0.95581** | 0.94920 | **+0.00661** | **2.2140** | 2.5260 | **−0.3120** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.97417** | 0.96734 | **+0.00683** | **1.8060** | 2.3060 | **−0.5000** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.97735** | 0.97335 | **+0.00400** | **1.0010** | 1.1820 | **−0.1810** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98150** | 0.96123 | **+0.02027** | **1.2320** | 2.5400 | **−1.3080** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.98707** | 0.92732 | **+0.05975** | **0.6345** | 3.6470 | **−3.0125** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.98842** | 0.97523 | **+0.01319** | **0.6033** | 1.2920 | **−0.6887** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.98653** | 0.96622 | **+0.02031** | **0.7988** | 1.9910 | **−1.1922** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.97249** | 0.96694 | **+0.00555** | **1.5200** | 1.8370 | **−0.3170** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.99245** | 0.98704 | **+0.00541** | **0.4976** | 0.9112 | **−0.4136** | same-image | same-image | **HSWQ** |
| **4710928** | **0.94664** | 0.92966 | **+0.01698** | **2.5910** | 3.3990 | **−0.8080** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | 0.84802 | **0.85418** | −0.00616 | 7.0670 | **6.8330** | +0.2340 | drifted (different image) | drifted (different image) | Native |
| **15820493** | **0.92782** | 0.88991 | **+0.03791** | **3.3850** | 5.1490 | **−1.7640** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.99396** | 0.97500 | **+0.01896** | **0.3501** | 1.4430 | **−1.0929** | same-image | drifted (different image) | **HSWQ** |
| **71058294** | **0.98868** | 0.98062 | **+0.00806** | **0.4578** | 0.7969 | **−0.3391** | same-image | same-image | **HSWQ** |
| **128491703** | 0.85636 | **0.91424** | −0.05788 | 5.7810 | **3.4320** | +2.3490 | drifted (different image) | drifted (different image) | Native |
| **285039184** | **0.99018** | 0.98454 | **+0.00564** | **0.4819** | 0.7575 | **−0.2756** | same-image | same-image | **HSWQ** |
| **491730285** | **0.97877** | 0.95056 | **+0.02821** | **1.0280** | 2.3810 | **−1.3530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | **0.98178** | 0.97446 | **+0.00732** | **1.3370** | 1.8450 | **−0.5080** | same-image | drifted (different image) | **HSWQ** |
| **938174026** | 0.99343 | **0.99367** | −0.00024 | 0.5238 | **0.4999** | +0.0239 | same-image | same-image | Native |
| **1409285713** | **0.97447** | 0.91309 | **+0.06138** | **1.8900** | 6.2870 | **−4.3970** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | 0.95411 | **0.96855** | −0.01444 | 2.1410 | **1.4740** | +0.6670 | drifted (different image) | drifted (different image) | Native |
| **3851729406** | **0.97237** | 0.93102 | **+0.04135** | **1.2870** | 3.4990 | **−2.2120** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.98298** | 0.97289 | **+0.01009** | **0.8064** | 1.2810 | **−0.4746** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.96362** | **0.94900** | **+0.01462** | **1.7861** | **2.7012** | **−0.9151** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (20/25)** |

---

### 2.8. epicrealismXL_pureFix (1on re570)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98419** | 0.97888 | **+0.00531** |
| **Min Final Cosine** (↑ better) | **0.90871** | 0.89986 | **+0.00885** |
| **Max Final Cosine** (↑ better) | **0.99829** | 0.99714 | **+0.00115** |
| **Mean Final Latent MSE** (↓ better) | **0.6658** | 0.8696 | **−0.2038 (23% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+20.6%** (9.79s → 7.78s) | −1.6% (9.19s → 9.33s) | — |
| **Trajectory Verdict** | 19/25 same-image, 6/25 drifted | 16/25 same-image, 9/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (epicrealismXL_pureFix)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.95692 | **0.95826** | −0.00134 | 2.0760 | **2.0090** | +0.0670 | drifted (different image) | drifted (different image) | Native |
| **137** | 0.98956 | **0.99487** | −0.00531 | 0.3754 | **0.1831** | +0.1923 | same-image | same-image | Native |
| **849** | **0.99687** | 0.98234 | **+0.01453** | **0.1286** | 0.7234 | **−0.5948** | same-image | same-image | **HSWQ** |
| **2024** | 0.98904 | **0.99067** | −0.00163 | 0.4960 | **0.4189** | +0.0771 | same-image | same-image | Native |
| **7391** | 0.95668 | **0.98329** | −0.02661 | 1.6040 | **0.6122** | +0.9918 | drifted (different image) | same-image | Native |
| **18429** | **0.99466** | 0.99288 | **+0.00178** | **0.2239** | 0.2987 | **−0.0748** | same-image | same-image | **HSWQ** |
| **53082** | 0.99333 | **0.99372** | −0.00039 | 0.3625 | **0.3391** | +0.0234 | same-image | same-image | Native |
| **149206** | **0.99456** | 0.98581 | **+0.00875** | **0.2025** | 0.5277 | **−0.3252** | same-image | same-image | **HSWQ** |
| **382715** | **0.99330** | 0.98821 | **+0.00509** | **0.2831** | 0.4962 | **−0.2131** | same-image | same-image | **HSWQ** |
| **826401** | **0.99329** | 0.99213 | **+0.00116** | **0.2396** | 0.2819 | **−0.0423** | same-image | same-image | **HSWQ** |
| **1938502** | **0.97629** | 0.96154 | **+0.01475** | **0.9857** | 1.5940 | **−0.6083** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.99605** | 0.99448 | **+0.00157** | **0.1606** | 0.2253 | **−0.0647** | same-image | same-image | **HSWQ** |
| **8391642** | **0.98767** | 0.97538 | **+0.01229** | **0.4901** | 0.9780 | **−0.4879** | same-image | drifted (different image) | **HSWQ** |
| **15820493** | **0.90871** | 0.89986 | **+0.00885** | **3.3080** | 3.6020 | **−0.2940** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.99478** | 0.97815 | **+0.01663** | **0.2561** | 1.0480 | **−0.7919** | same-image | drifted (different image) | **HSWQ** |
| **71058294** | **0.99223** | 0.97472 | **+0.01751** | **0.2838** | 0.9242 | **−0.6404** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | **0.99722** | 0.99152 | **+0.00570** | **0.1089** | 0.3316 | **−0.2227** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99549** | 0.99439 | **+0.00110** | **0.1694** | 0.2110 | **−0.0416** | same-image | same-image | **HSWQ** |
| **491730285** | **0.99667** | 0.98956 | **+0.00711** | **0.1437** | 0.4483 | **−0.3046** | same-image | same-image | **HSWQ** |
| **762019483** | **0.97691** | 0.96540 | **+0.01151** | **1.1360** | 1.6910 | **−0.5550** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | 0.95263 | **0.96984** | −0.01721 | 2.4470 | **1.5560** | +0.8910 | drifted (different image) | drifted (different image) | Native |
| **1409285713** | **0.99829** | 0.99714 | **+0.00115** | **0.0817** | 0.1355 | **−0.0538** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98468** | 0.93465 | **+0.05003** | **0.5372** | 2.2760 | **−1.7388** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99673** | 0.99239 | **+0.00434** | **0.1636** | 0.3809 | **−0.2173** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99213** | 0.99075 | **+0.00138** | **0.3808** | 0.4479 | **−0.0671** | same-image | same-image | **HSWQ** |
| **Mean** | **0.98419** | **0.97888** | **+0.00531** | **0.6658** | **0.8696** | **−0.2038** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

### 2.9. koronemixIllustrious_v70 (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97224** | 0.94886 | **+0.02338** |
| **Min Final Cosine** (↑ better) | **0.92210** | 0.86266 | **+0.05944** |
| **Max Final Cosine** (↑ better) | **0.99105** | 0.99300 | **−0.00195** |
| **Mean Final Latent MSE** (↓ better) | **1.3348** | 2.3778 | **−1.0430 (44% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+16.3%** (9.19s → 7.70s) | +1.8% (9.22s → 9.05s) | — |
| **Trajectory Verdict** | 13/25 same-image, 12/25 drifted | 6/25 same-image, 19/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (koronemixIllustrious_v70)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.96817** | 0.94494 | **+0.02323** | **2.0200** | 3.5310 | **−1.5110** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.98350** | 0.92078 | **+0.06272** | **0.5949** | 2.9330 | **−2.3381** | same-image | drifted (different image) | **HSWQ** |
| **849** | **0.98412** | 0.93154 | **+0.05258** | **0.6248** | 2.6780 | **−2.0532** | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.93938** | 0.86266 | **+0.07672** | **2.8590** | 6.4910 | **−3.6320** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.96638** | 0.95736 | **+0.00902** | **1.7110** | 2.1470 | **−0.4360** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98202** | 0.95168 | **+0.03034** | **0.8281** | 2.2930 | **−1.4649** | same-image | drifted (different image) | **HSWQ** |
| **53082** | 0.98203 | **0.98605** | −0.00402 | 1.0360 | **0.8097** | +0.2263 | same-image | same-image | Native |
| **149206** | **0.98316** | 0.97493 | **+0.00823** | **0.8470** | 1.2480 | **−0.4010** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.98384** | 0.93193 | **+0.05191** | **0.7427** | 3.1680 | **−2.4253** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.99105** | 0.97784 | **+0.01321** | **0.3752** | 0.9736 | **−0.5984** | same-image | drifted (different image) | **HSWQ** |
| **1938502** | **0.95885** | 0.88127 | **+0.07758** | **1.3220** | 3.7540 | **−2.4320** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.95653** | 0.90694 | **+0.04959** | **2.0990** | 4.4810 | **−2.3820** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.92210** | 0.91598 | **+0.00612** | **3.2490** | 3.5780 | **−0.3290** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.97424** | 0.94852 | **+0.02572** | **1.5820** | 3.1640 | **−1.5820** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.99068** | 0.98354 | **+0.00714** | **0.5313** | 0.9591 | **−0.4278** | same-image | same-image | **HSWQ** |
| **71058294** | **0.98717** | 0.91617 | **+0.07100** | **0.6398** | 4.1930 | **−3.5532** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | 0.98760 | **0.99024** | −0.00264 | 0.7354 | **0.5830** | +0.1524 | same-image | same-image | Native |
| **285039184** | **0.99100** | 0.95044 | **+0.04056** | **0.3889** | 2.1590 | **−1.7701** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.97546** | 0.97447 | **+0.00099** | **1.0850** | 1.1170 | **−0.0320** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | 0.98699 | **0.99300** | −0.00601 | 0.7532 | **0.4060** | +0.3472 | same-image | same-image | Native |
| **938174026** | 0.93393 | **0.94395** | −0.01002 | 3.6340 | **3.0950** | +0.5390 | drifted (different image) | drifted (different image) | Native |
| **1409285713** | **0.99037** | 0.98038 | **+0.00999** | **0.4563** | 0.9415 | **−0.4852** | same-image | same-image | **HSWQ** |
| **2683910547** | 0.94198 | **0.94306** | −0.00108 | 2.6070 | **2.5650** | +0.0420 | drifted (different image) | drifted (different image) | Native |
| **3851729406** | 0.97784 | **0.99006** | −0.01222 | 1.1950 | **0.5393** | +0.6557 | drifted (different image) | same-image | Native |
| **4195820371** | **0.96750** | 0.96370 | **+0.00380** | **1.4540** | 1.6390 | **−0.1850** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.97224** | **0.94886** | **+0.02338** | **1.3348** | **2.3778** | **−1.0430** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

### 2.10. koronemixVpred_v20 (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95351** | 0.94315 | **+0.01036** |
| **Min Final Cosine** (↑ better) | **0.83494** | 0.83299 | **+0.00195** |
| **Max Final Cosine** (↑ better) | **0.99567** | 0.99243 | **+0.00324** |
| **Mean Final Latent MSE** (↓ better) | **3.6009** | 4.2807 | **−0.6798 (16% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **1/25 (4%)** | 3/25 (12%) | **3 Native vs 1 HSWQ bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+16.9%** (9.17s → 7.62s) | +9.5% (9.74s → 8.81s) | — |
| **Trajectory Verdict** | 12/25 same-image, 12/25 drifted, 1/25 bifurcated | 6/25 same-image, 16/25 drifted, 3/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (koronemixVpred_v20)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97833** | 0.94924 | **+0.02909** | **1.5600** | 3.6730 | **−2.1130** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.99008** | 0.97687 | **+0.01321** | **0.8419** | 1.9930 | **−1.1511** | same-image | drifted (different image) | **HSWQ** |
| **849** | 0.93040 | **0.97097** | −0.04057 | 4.0290 | **1.6770** | +2.3520 | drifted (different image) | drifted (different image) | Native |
| **2024** | **0.99531** | 0.98955 | **+0.00576** | **0.5373** | 1.2230 | **−0.6857** | same-image | same-image | **HSWQ** |
| **7391** | 0.94102 | **0.96285** | −0.02183 | 5.9130 | **3.7050** | +2.2080 | drifted (different image) | drifted (different image) | Native |
| **18429** | **0.98630** | 0.93145 | **+0.05485** | **1.1440** | 5.7870 | **−4.6430** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.99567** | 0.97648 | **+0.01919** | **0.2796** | 1.5300 | **−1.2504** | same-image | drifted (different image) | **HSWQ** |
| **149206** | 0.92795 | **0.95034** | −0.02239 | 7.7360 | **5.3680** | +2.3680 | drifted (different image) | drifted (different image) | Native |
| **382715** | **0.98944** | 0.97884 | **+0.01060** | **0.7005** | 1.4170 | **−0.7165** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.93550** | 0.83436 | **+0.10114** | **3.3360** | 8.6990 | **−5.3630** | drifted (different image) | bifurcated @step 22 | **HSWQ** |
| **1938502** | **0.99520** | 0.98921 | **+0.00599** | **0.3294** | 0.7537 | **−0.4243** | same-image | same-image | **HSWQ** |
| **4710928** | 0.85851 | **0.93058** | −0.07207 | 10.4300 | **5.1990** | +5.2310 | drifted (different image) | drifted (different image) | Native |
| **8391642** | 0.97621 | **0.98459** | −0.00838 | 1.6250 | **1.0630** | +0.5620 | drifted (different image) | same-image | Native |
| **15820493** | **0.88523** | 0.88104 | **+0.00419** | **9.2400** | 9.6070 | **−0.3670** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | 0.98973 | **0.99243** | −0.00270 | 1.3370 | **1.0170** | +0.3200 | same-image | same-image | Native |
| **71058294** | **0.86461** | 0.84228 | **+0.02233** | **8.4380** | 10.0200 | **−1.5820** | drifted (different image) | bifurcated @step 22 | **HSWQ** |
| **128491703** | **0.97087** | 0.95633 | **+0.01454** | **3.8050** | 5.8090 | **−2.0040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **285039184** | **0.90831** | 0.88070 | **+0.02761** | **6.7230** | 9.2760 | **−2.5530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.94727** | 0.83299 | **+0.11428** | **3.5500** | 11.4100 | **−7.8600** | drifted (different image) | bifurcated @step 22 | **HSWQ** |
| **762019483** | 0.83494 | **0.93767** | −0.10273 | 13.7500 | **5.2570** | +8.4930 | bifurcated @step 22 | drifted (different image) | Native |
| **938174026** | **0.98111** | 0.93970 | **+0.04141** | **1.4500** | 4.6870 | **−3.2370** | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | **0.98721** | 0.95388 | **+0.03333** | **0.8186** | 2.9480 | **−2.1294** | same-image | drifted (different image) | **HSWQ** |
| **2683910547** | **0.98409** | 0.96162 | **+0.02247** | **1.0340** | 2.5360 | **−1.5020** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99379** | 0.99046 | **+0.00333** | **0.4219** | 0.6530 | **−0.2311** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99072** | 0.98436 | **+0.00636** | **0.9944** | 1.7100 | **−0.7156** | same-image | same-image | **HSWQ** |
| **Mean** | **0.95351** | **0.94315** | **+0.01036** | **3.6009** | **4.2807** | **−0.6798** | **1/25 Bifurcated** | **3/25 Bifurcated** | **HSWQ (18/25)** |

---

### 2.11. novaAsianXL_illustriousV70 (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98783** | 0.97515 | **+0.01268** |
| **Min Final Cosine** (↑ better) | **0.95235** | 0.90902 | **+0.04333** |
| **Max Final Cosine** (↑ better) | **0.99637** | 0.99369 | **+0.00268** |
| **Mean Final Latent MSE** (↓ better) | **0.4571** | 0.9429 | **−0.4858 (52% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+15.3%** (9.16s → 7.77s) | +5.4% (9.67s → 9.14s) | — |
| **Trajectory Verdict** | 21/25 same-image, 4/25 drifted | 15/25 same-image, 10/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (novaAsianXL_illustriousV70)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99610** | 0.99208 | **+0.00402** | **0.1820** | 0.3711 | **−0.1891** | same-image | same-image | **HSWQ** |
| **137** | **0.99550** | 0.98729 | **+0.00821** | **0.1554** | 0.4414 | **−0.2860** | same-image | same-image | **HSWQ** |
| **849** | **0.99229** | 0.99010 | **+0.00219** | **0.2817** | 0.3628 | **−0.0811** | same-image | same-image | **HSWQ** |
| **2024** | **0.99440** | 0.99085 | **+0.00355** | **0.2006** | 0.3289 | **−0.1283** | same-image | same-image | **HSWQ** |
| **7391** | **0.98763** | 0.95587 | **+0.03176** | **0.3784** | 1.3350 | **−0.9566** | same-image | drifted (different image) | **HSWQ** |
| **18429** | **0.99246** | 0.97833 | **+0.01413** | **0.2514** | 0.7240 | **−0.4726** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.97585** | 0.96455 | **+0.01130** | **0.9816** | 1.4420 | **−0.4604** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.98019** | 0.96071 | **+0.01948** | **0.7551** | 1.4980 | **−0.7429** | same-image | drifted (different image) | **HSWQ** |
| **382715** | 0.99280 | **0.99355** | −0.00075 | 0.2774 | **0.2494** | +0.0280 | same-image | same-image | Native |
| **826401** | **0.99441** | 0.98907 | **+0.00534** | **0.2113** | 0.4136 | **−0.2023** | same-image | same-image | **HSWQ** |
| **1938502** | 0.95244 | **0.95263** | −0.00019 | 1.5510 | **1.5480** | +0.0030 | drifted (different image) | drifted (different image) | Native |
| **4710928** | **0.99587** | 0.98636 | **+0.00951** | **0.1682** | 0.5572 | **−0.3890** | same-image | same-image | **HSWQ** |
| **8391642** | **0.98376** | 0.98372 | **+0.00004** | **0.6461** | 0.6466 | **−0.0005** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99569** | 0.99369 | **+0.00200** | **0.1507** | 0.2208 | **−0.0701** | same-image | same-image | **HSWQ** |
| **36192847** | 0.95235 | **0.95377** | −0.00142 | 1.7600 | **1.7110** | +0.0490 | drifted (different image) | drifted (different image) | Native |
| **71058294** | **0.99493** | 0.99029 | **+0.00464** | **0.1683** | 0.3224 | **−0.1541** | same-image | same-image | **HSWQ** |
| **128491703** | **0.99091** | 0.98433 | **+0.00658** | **0.3662** | 0.6337 | **−0.2675** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99554** | 0.94950 | **+0.04604** | **0.1590** | 1.8190 | **−1.6600** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.98368** | 0.97966 | **+0.00402** | **0.8356** | 1.0400 | **−0.2044** | same-image | drifted (different image) | **HSWQ** |
| **762019483** | **0.99432** | 0.98988 | **+0.00444** | **0.2428** | 0.4337 | **−0.1909** | same-image | same-image | **HSWQ** |
| **938174026** | **0.99510** | 0.99061 | **+0.00449** | **0.1998** | 0.3833 | **−0.1835** | same-image | same-image | **HSWQ** |
| **1409285713** | **0.99577** | 0.98946 | **+0.00631** | **0.1653** | 0.4140 | **−0.2487** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.97126** | 0.93298 | **+0.03828** | **1.0350** | 2.4330 | **−1.3980** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99637** | 0.90902 | **+0.08735** | **0.1551** | 3.8770 | **−3.7219** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.99614** | 0.99050 | **+0.00564** | **0.1491** | 0.3664 | **−0.2173** | same-image | same-image | **HSWQ** |
| **Mean** | **0.98783** | **0.97515** | **+0.01268** | **0.4571** | **0.9429** | **−0.4858** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (22/25)** |

---

### 2.12. realvisxlV30_v30TurboBakedvae (1on re650)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96184** | 0.94210 | **+0.01974** |
| **Min Final Cosine** (↑ better) | **0.84908** | 0.80735 | **+0.04173** |
| **Max Final Cosine** (↑ better) | **0.98386** | 0.97927 | **+0.00459** |
| **Mean Final Latent MSE** (↓ better) | **2.2952** | 3.5083 | **−1.2131 (35% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+6.5%** (9.18s → 8.58s) | +8.5% (9.61s → 8.79s) | — |
| **Trajectory Verdict** | 5/25 same-image, 20/25 drifted | 25/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (realvisxlV30_v30TurboBakedvae)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97351** | 0.96479 | **+0.00872** | **1.5510** | 2.0870 | **−0.5360** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.94617** | 0.93716 | **+0.00901** | **3.1910** | 3.7500 | **−0.5590** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.98040** | 0.97043 | **+0.00997** | **1.2890** | 1.9530 | **−0.6640** | same-image | drifted (different image) | **HSWQ** |
| **2024** | 0.93725 | **0.95969** | −0.02244 | 3.9430 | **2.5290** | +1.4140 | drifted (different image) | drifted (different image) | Native |
| **7391** | **0.97074** | 0.93665 | **+0.03409** | **1.7750** | 3.8660 | **−2.0910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98324** | 0.97110 | **+0.01214** | **1.0600** | 1.8200 | **−0.7600** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.84908** | 0.80735 | **+0.04173** | **8.1500** | 10.6900 | **−2.5400** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.98202** | 0.94112 | **+0.04090** | **0.9779** | 3.2210 | **−2.2431** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.93042** | 0.89331 | **+0.03711** | **4.4410** | 6.8760 | **−2.4350** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.96272** | 0.93340 | **+0.02932** | **2.1630** | 3.8780 | **−1.7150** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.98386** | 0.97753 | **+0.00633** | **1.0350** | 1.4440 | **−0.4090** | same-image | drifted (different image) | **HSWQ** |
| **4710928** | **0.96975** | 0.93491 | **+0.03484** | **1.7490** | 3.7620 | **−2.0130** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.97850** | 0.97513 | **+0.00337** | **1.2760** | 1.4800 | **−0.2040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.97127** | 0.96286 | **+0.00841** | **1.6490** | 2.1300 | **−0.4810** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | 0.95818 | **0.96864** | −0.01046 | 2.5540 | **1.9120** | +0.6420 | drifted (different image) | drifted (different image) | Native |
| **71058294** | **0.95743** | 0.83032 | **+0.12711** | **2.7650** | 10.8700 | **−8.1050** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | 0.96568 | **0.97927** | −0.01359 | 2.0350 | **1.2250** | +0.8100 | drifted (different image) | drifted (different image) | Native |
| **285039184** | **0.97565** | 0.94423 | **+0.03142** | **1.4410** | 3.2830 | **−1.8420** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.95563** | 0.93609 | **+0.01954** | **3.3390** | 4.8060 | **−1.4670** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | **0.97895** | 0.96986 | **+0.00909** | **1.2730** | 1.8240 | **−0.5510** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | 0.94554 | **0.95483** | −0.00929 | 2.9910 | **2.4930** | +0.4980 | drifted (different image) | drifted (different image) | Native |
| **1409285713** | **0.97909** | 0.95513 | **+0.02396** | **1.3310** | 2.8660 | **−1.5350** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2683910547** | **0.98117** | 0.95268 | **+0.02849** | **1.0570** | 2.6690 | **−1.6120** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.97821** | 0.95520 | **+0.02301** | **1.1980** | 2.4410 | **−1.2430** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.95150** | 0.94094 | **+0.01056** | **3.1460** | 3.8320 | **−0.6860** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96184** | **0.94210** | **+0.01974** | **2.2952** | **3.5083** | **−1.2131** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.13. realvisxlV50_v40Bakedvae (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98860** | 0.98464 | **+0.00396** |
| **Min Final Cosine** (↑ better) | **0.95295** | 0.93820 | **+0.01475** |
| **Max Final Cosine** (↑ better) | **0.99679** | 0.99444 | **+0.00235** |
| **Mean Final Latent MSE** (↓ better) | **0.5307** | 0.7341 | **−0.2034 (28% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+10.9%** (9.22s → 8.22s) | +3.9% (9.19s → 8.83s) | — |
| **Trajectory Verdict** | 22/25 same-image, 3/25 drifted | 20/25 same-image, 5/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (realvisxlV50_v40Bakedvae)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.98147 | **0.98551** | −0.00404 | 0.8978 | **0.7077** | +0.1901 | same-image | same-image | Native |
| **137** | **0.99021** | 0.93820 | **+0.05201** | **0.4659** | 2.9800 | **−2.5141** | same-image | drifted (different image) | **HSWQ** |
| **849** | **0.99345** | 0.98921 | **+0.00424** | **0.3304** | 0.5463 | **−0.2159** | same-image | same-image | **HSWQ** |
| **2024** | **0.98851** | 0.97570 | **+0.01281** | **0.6089** | 1.2800 | **−0.6711** | same-image | drifted (different image) | **HSWQ** |
| **7391** | **0.99077** | 0.98916 | **+0.00161** | **0.4215** | 0.4966 | **−0.0751** | same-image | same-image | **HSWQ** |
| **18429** | 0.98888 | **0.99340** | −0.00452 | 0.5423 | **0.3240** | +0.2183 | same-image | same-image | Native |
| **53082** | 0.98761 | **0.99220** | −0.00459 | 0.5237 | **0.3319** | +0.1918 | same-image | same-image | Native |
| **149206** | 0.95295 | **0.98236** | −0.02941 | 1.9410 | **0.7316** | +1.2094 | drifted (different image) | same-image | Native |
| **382715** | **0.97377** | 0.97129 | **+0.00248** | **1.2110** | 1.3280 | **−0.1170** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.99676** | 0.99444 | **+0.00232** | **0.1485** | 0.2553 | **−0.1068** | same-image | same-image | **HSWQ** |
| **1938502** | **0.98849** | 0.97733 | **+0.01116** | **0.5458** | 1.0770 | **−0.5312** | same-image | drifted (different image) | **HSWQ** |
| **4710928** | **0.98223** | 0.98194 | **+0.00029** | **0.7986** | 0.8169 | **−0.0183** | same-image | same-image | **HSWQ** |
| **8391642** | **0.99615** | 0.99341 | **+0.00274** | **0.1849** | 0.3167 | **−0.1318** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99638** | 0.99400 | **+0.00238** | **0.1795** | 0.2992 | **−0.1197** | same-image | same-image | **HSWQ** |
| **36192847** | **0.99550** | 0.99023 | **+0.00527** | **0.2100** | 0.4582 | **−0.2482** | same-image | same-image | **HSWQ** |
| **71058294** | **0.99679** | 0.98902 | **+0.00777** | **0.1459** | 0.5006 | **−0.3547** | same-image | same-image | **HSWQ** |
| **128491703** | 0.99036 | **0.99273** | −0.00237 | 0.4446 | **0.3371** | +0.1075 | same-image | same-image | Native |
| **285039184** | **0.99637** | 0.99110 | **+0.00527** | **0.1661** | 0.4082 | **−0.2421** | same-image | same-image | **HSWQ** |
| **491730285** | **0.99525** | 0.99069 | **+0.00456** | **0.2372** | 0.4661 | **−0.2289** | same-image | same-image | **HSWQ** |
| **762019483** | **0.98974** | 0.98090 | **+0.00884** | **0.5045** | 0.9374 | **−0.4329** | same-image | same-image | **HSWQ** |
| **938174026** | **0.96837** | 0.96373 | **+0.00464** | **1.5760** | 1.8180 | **−0.2420** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1409285713** | **0.99228** | 0.98750 | **+0.00478** | **0.3604** | 0.5873 | **−0.2269** | same-image | same-image | **HSWQ** |
| **2683910547** | 0.99301 | **0.99408** | −0.00107 | 0.3404 | **0.2891** | +0.0513 | same-image | same-image | Native |
| **3851729406** | **0.99618** | 0.99360 | **+0.00258** | **0.1797** | 0.3029 | **−0.1232** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.99362** | 0.98419 | **+0.00943** | **0.3032** | 0.7555 | **−0.4523** | same-image | same-image | **HSWQ** |
| **Mean** | **0.98860** | **0.98464** | **+0.00396** | **0.5307** | **0.7341** | **−0.2034** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

### 2.14. realvisxlV50_v50Bakedvae (1on re550)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98747** | 0.97734 | **+0.01013** |
| **Min Final Cosine** (↑ better) | **0.96747** | 0.91625 | **+0.05122** |
| **Max Final Cosine** (↑ better) | **0.99848** | 0.99588 | **+0.00260** |
| **Mean Final Latent MSE** (↓ better) | **0.5568** | 1.0037 | **−0.4470 (45% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+10.8%** (9.22s → 8.23s) | +6.9% (9.75s → 9.07s) | — |
| **Trajectory Verdict** | 19/25 same-image, 6/25 drifted | 13/25 same-image, 12/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (realvisxlV50_v50Bakedvae)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97578** | 0.97260 | **+0.00318** | **1.0690** | 1.2340 | **−0.1650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.99393** | 0.98395 | **+0.00998** | **0.2630** | 0.6989 | **−0.4359** | same-image | same-image | **HSWQ** |
| **849** | 0.98593 | **0.98935** | −0.00342 | 0.6733 | **0.5173** | +0.1560 | same-image | same-image | Native |
| **2024** | 0.98984 | **0.99219** | −0.00235 | 0.5134 | **0.3978** | +0.1156 | same-image | same-image | Native |
| **7391** | **0.98871** | 0.97765 | **+0.01106** | **0.4881** | 0.9745 | **−0.4864** | same-image | drifted (different image) | **HSWQ** |
| **18429** | **0.99549** | 0.98335 | **+0.01214** | **0.2060** | 0.7696 | **−0.5636** | same-image | same-image | **HSWQ** |
| **53082** | **0.97744** | 0.97390 | **+0.00354** | **0.8424** | 0.9781 | **−0.1357** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | **0.99379** | 0.95715 | **+0.03664** | **0.2312** | 1.6130 | **−1.3818** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.99065** | 0.98700 | **+0.00365** | **0.4081** | 0.5799 | **−0.1718** | same-image | same-image | **HSWQ** |
| **826401** | **0.99689** | 0.99154 | **+0.00535** | **0.1400** | 0.3848 | **−0.2448** | same-image | same-image | **HSWQ** |
| **1938502** | 0.97465 | **0.98233** | −0.00768 | 1.1900 | **0.8349** | +0.3551 | drifted (different image) | same-image | Native |
| **4710928** | **0.99848** | 0.99484 | **+0.00364** | **0.0696** | 0.2402 | **−0.1706** | same-image | same-image | **HSWQ** |
| **8391642** | **0.99467** | 0.99215 | **+0.00252** | **0.2510** | 0.3727 | **−0.1217** | same-image | same-image | **HSWQ** |
| **15820493** | **0.99827** | 0.99588 | **+0.00239** | **0.0783** | 0.1917 | **−0.1134** | same-image | same-image | **HSWQ** |
| **36192847** | **0.99586** | 0.99193 | **+0.00393** | **0.1740** | 0.3450 | **−0.1710** | same-image | same-image | **HSWQ** |
| **71058294** | **0.99641** | 0.99531 | **+0.00110** | **0.1533** | 0.2049 | **−0.0516** | same-image | same-image | **HSWQ** |
| **128491703** | **0.98291** | 0.97944 | **+0.00347** | **0.7371** | 0.8958 | **−0.1587** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | **0.97281** | 0.97086 | **+0.00195** | **1.2140** | 1.3180 | **−0.1040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | 0.96747 | **0.97186** | −0.00439 | 1.6240 | **1.4180** | +0.2060 | drifted (different image) | drifted (different image) | Native |
| **762019483** | **0.99054** | 0.91625 | **+0.07429** | **0.4510** | 4.0520 | **−3.6010** | same-image | drifted (different image) | **HSWQ** |
| **938174026** | **0.98581** | 0.95766 | **+0.02815** | **0.5801** | 1.7540 | **−1.1739** | same-image | drifted (different image) | **HSWQ** |
| **1409285713** | **0.99425** | 0.98629 | **+0.00796** | **0.2544** | 0.6131 | **−0.3587** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98409** | 0.97439 | **+0.00970** | **0.7253** | 1.1740 | **−0.4487** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.98383** | 0.97001 | **+0.01382** | **0.6867** | 1.2910 | **−0.6043** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.97816** | 0.94569 | **+0.03247** | **0.8961** | 2.2400 | **−1.3439** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.98747** | **0.97734** | **+0.01013** | **0.5568** | **1.0037** | **−0.4470** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.15. unholyDesireMixSinister_v90 (1on re590)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97121** | 0.95367 | **+0.01754** |
| **Min Final Cosine** (↑ better) | **0.90538** | 0.74470 | **+0.16068** |
| **Max Final Cosine** (↑ better) | **0.99466** | 0.99483 | **−0.00017** |
| **Mean Final Latent MSE** (↓ better) | **1.4192** | 2.2342 | **−0.8150 (36% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+13.2%** (9.16s → 7.96s) | +1.5% (9.39s → 9.24s) | — |
| **Trajectory Verdict** | 13/25 same-image, 12/25 drifted | 10/25 same-image, 15/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (unholyDesireMixSinister_v90)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.98524 | **0.98857** | −0.00333 | 0.9686 | **0.7480** | +0.2206 | same-image | same-image | Native |
| **137** | 0.98154 | **0.98462** | −0.00308 | 0.6944 | **0.5917** | +0.1027 | same-image | same-image | Native |
| **849** | 0.97962 | **0.98157** | −0.00195 | 0.9547 | **0.8610** | +0.0937 | drifted (different image) | same-image | Native |
| **2024** | **0.97425** | 0.94296 | **+0.03129** | **1.2770** | 2.7680 | **−1.4910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7391** | **0.91007** | 0.74470 | **+0.16537** | **3.7770** | 9.9080 | **−6.1310** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98684** | 0.92446 | **+0.06238** | **0.4782** | 2.7620 | **−2.2838** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.97951** | 0.93995 | **+0.03956** | **1.2420** | 3.6470 | **−2.4050** | drifted (different image) | drifted (different image) | **HSWQ** |
| **149206** | 0.91075 | **0.91689** | −0.00614 | 4.3830 | **4.0730** | +0.3100 | drifted (different image) | drifted (different image) | Native |
| **382715** | **0.98239** | 0.88912 | **+0.09327** | **0.9246** | 5.9140 | **−4.9894** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.99123** | 0.98669 | **+0.00454** | **0.4606** | 0.7029 | **−0.2423** | same-image | same-image | **HSWQ** |
| **1938502** | **0.96335** | 0.94883 | **+0.01452** | **1.3860** | 1.9510 | **−0.5650** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.99466** | 0.99307 | **+0.00159** | **0.3544** | 0.4581 | **−0.1037** | same-image | same-image | **HSWQ** |
| **8391642** | **0.94191** | 0.92282 | **+0.01909** | **2.6720** | 3.5320 | **−0.8600** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.97591** | 0.95999 | **+0.01592** | **1.5720** | 2.6090 | **−1.0370** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.99108** | 0.99038 | **+0.00070** | **0.5239** | 0.5656 | **−0.0417** | same-image | same-image | **HSWQ** |
| **71058294** | 0.95671 | **0.97151** | −0.01480 | 2.2540 | **1.4830** | +0.7710 | drifted (different image) | drifted (different image) | Native |
| **128491703** | **0.98952** | 0.98181 | **+0.00771** | **0.6194** | 1.1060 | **−0.4866** | same-image | same-image | **HSWQ** |
| **285039184** | **0.96746** | 0.96221 | **+0.00525** | **1.6120** | 1.8780 | **−0.2660** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | 0.90538 | **0.92614** | −0.02076 | 4.4560 | **3.5720** | +0.8840 | drifted (different image) | drifted (different image) | Native |
| **762019483** | **0.99066** | 0.96301 | **+0.02765** | **0.5985** | 2.3450 | **−1.7465** | same-image | drifted (different image) | **HSWQ** |
| **938174026** | **0.98945** | 0.98929 | **+0.00016** | **0.7171** | 0.7284 | **−0.0113** | same-image | same-image | **HSWQ** |
| **1409285713** | 0.98925 | **0.99483** | −0.00558 | 0.6745 | **0.3230** | +0.3515 | same-image | same-image | Native |
| **2683910547** | 0.96508 | **0.97502** | −0.00994 | 1.6430 | **1.1930** | +0.4500 | drifted (different image) | drifted (different image) | Native |
| **3851729406** | **0.98417** | 0.97181 | **+0.01236** | **0.9372** | 1.6770 | **−0.7398** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.99433** | 0.99145 | **+0.00288** | **0.3011** | 0.4584 | **−0.1573** | same-image | same-image | **HSWQ** |
| **Mean** | **0.97121** | **0.95367** | **+0.01754** | **1.4192** | **2.2342** | **−0.8150** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (17/25)** |

---

### 2.16. uwazumimixILL_v50 (1on re720)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98034** | 0.97903 | **+0.00131** |
| **Min Final Cosine** (↑ better) | **0.87515** | 0.86236 | **+0.01279** |
| **Max Final Cosine** (↑ better) | **0.99630** | 0.99542 | **+0.00088** |
| **Mean Final Latent MSE** (↓ better) | **0.3771** | 0.4101 | **−0.0330 (8% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+8.3%** (9.15s → 8.39s) | −0.9% (9.18s → 9.26s) | — |
| **Trajectory Verdict** | 17/25 same-image, 8/25 drifted | 17/25 same-image, 8/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (uwazumimixILL_v50)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99564** | 0.99240 | **+0.00324** | **0.1004** | 0.1736 | **−0.0732** | same-image | same-image | **HSWQ** |
| **137** | 0.96397 | **0.98762** | −0.02365 | 0.5844 | **0.2011** | +0.3833 | drifted (different image) | same-image | Native |
| **849** | **0.99630** | 0.99542 | **+0.00088** | **0.1060** | 0.1321 | **−0.0261** | same-image | same-image | **HSWQ** |
| **2024** | 0.97459 | **0.97995** | −0.00536 | 0.4097 | **0.3229** | +0.0868 | drifted (different image) | drifted (different image) | Native |
| **7391** | **0.99350** | 0.98862 | **+0.00488** | **0.1579** | 0.2808 | **−0.1229** | same-image | same-image | **HSWQ** |
| **18429** | **0.98843** | 0.96498 | **+0.02345** | **0.1795** | 0.5407 | **−0.3612** | same-image | drifted (different image) | **HSWQ** |
| **53082** | **0.99410** | 0.97719 | **+0.01691** | **0.1636** | 0.6439 | **−0.4803** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.98493** | 0.98467 | **+0.00026** | **0.3620** | 0.3710 | **−0.0090** | same-image | same-image | **HSWQ** |
| **382715** | **0.97446** | 0.96072 | **+0.01374** | **0.4317** | 0.6609 | **−0.2292** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | 0.98605 | **0.98987** | −0.00382 | 0.3964 | **0.2932** | +0.1032 | same-image | same-image | Native |
| **1938502** | **0.98995** | 0.98351 | **+0.00644** | **0.2119** | 0.3503 | **−0.1384** | same-image | same-image | **HSWQ** |
| **4710928** | 0.96100 | **0.96876** | −0.00776 | 0.6683 | **0.5318** | +0.1365 | drifted (different image) | drifted (different image) | Native |
| **8391642** | **0.99394** | 0.99182 | **+0.00212** | **0.1341** | 0.1807 | **−0.0466** | same-image | same-image | **HSWQ** |
| **15820493** | 0.94847 | **0.97445** | −0.02598 | 0.8842 | **0.4397** | +0.4445 | drifted (different image) | drifted (different image) | Native |
| **36192847** | 0.98692 | **0.99298** | −0.00606 | 0.2201 | **0.1179** | +0.1022 | same-image | same-image | Native |
| **71058294** | **0.97193** | 0.96268 | **+0.00925** | **0.5652** | 0.7583 | **−0.1931** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.99596** | 0.99515 | **+0.00081** | **0.0779** | 0.0931 | **−0.0152** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99046** | 0.98128 | **+0.00918** | **0.1530** | 0.3020 | **−0.1490** | same-image | same-image | **HSWQ** |
| **491730285** | **0.87515** | 0.86236 | **+0.01279** | **2.3960** | 2.6950 | **−0.2990** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | **0.99413** | 0.99282 | **+0.00131** | **0.1086** | 0.1321 | **−0.0235** | same-image | same-image | **HSWQ** |
| **938174026** | **0.99186** | 0.98210 | **+0.00976** | **0.1527** | 0.3333 | **−0.1806** | same-image | same-image | **HSWQ** |
| **1409285713** | **0.99407** | 0.99286 | **+0.00121** | **0.1026** | 0.1232 | **−0.0206** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99262** | 0.98769 | **+0.00493** | **0.1414** | 0.2349 | **−0.0935** | same-image | same-image | **HSWQ** |
| **3851729406** | **0.99509** | 0.99310 | **+0.00199** | **0.1172** | 0.1648 | **−0.0476** | same-image | same-image | **HSWQ** |
| **4195820371** | 0.97507 | **0.99277** | −0.01770 | 0.6024 | **0.1743** | +0.4281 | drifted (different image) | same-image | Native |
| **Mean** | **0.98034** | **0.97903** | **+0.00131** | **0.3771** | **0.4101** | **−0.0330** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (18/25)** |

---

### 2.17. waiANIPONYXL_v90 (1on re650)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98075** | 0.94910 | **+0.03165** |
| **Min Final Cosine** (↑ better) | **0.93939** | 0.83726 | **+0.10213** |
| **Max Final Cosine** (↑ better) | **0.99732** | 0.99242 | **+0.00490** |
| **Mean Final Latent MSE** (↓ better) | **0.7713** | 2.0174 | **−1.2461 (62% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+11.3%** (9.14s → 8.11s) | +4.8% (9.14s → 8.70s) | — |
| **Trajectory Verdict** | 14/25 same-image, 11/25 drifted | 8/25 same-image, 17/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (waiANIPONYXL_v90)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99732** | 0.99082 | **+0.00650** | **0.1037** | 0.3600 | **−0.2563** | same-image | same-image | **HSWQ** |
| **137** | **0.98433** | 0.95663 | **+0.02770** | **0.5540** | 1.5640 | **−1.0100** | same-image | drifted (different image) | **HSWQ** |
| **849** | **0.97623** | 0.96168 | **+0.01455** | **0.8589** | 1.4010 | **−0.5421** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.99567** | 0.97886 | **+0.01681** | **0.1548** | 0.7625 | **−0.6077** | same-image | drifted (different image) | **HSWQ** |
| **7391** | **0.96551** | 0.94276 | **+0.02275** | **1.3370** | 2.2030 | **−0.8660** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.99386** | 0.98433 | **+0.00953** | **0.2065** | 0.5382 | **−0.3317** | same-image | same-image | **HSWQ** |
| **53082** | **0.98210** | 0.97757 | **+0.00453** | **0.7702** | 0.9666 | **−0.1964** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.99097** | 0.87516 | **+0.11581** | **0.3150** | 4.3320 | **−4.0170** | same-image | drifted (different image) | **HSWQ** |
| **382715** | **0.97174** | 0.95678 | **+0.01496** | **0.8704** | 1.3270 | **−0.4566** | drifted (different image) | drifted (different image) | **HSWQ** |
| **826401** | **0.97720** | 0.85579 | **+0.12141** | **0.9195** | 5.9430 | **−5.0235** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.99520** | 0.98215 | **+0.01305** | **0.1637** | 0.6184 | **−0.4547** | same-image | same-image | **HSWQ** |
| **4710928** | **0.97210** | 0.93281 | **+0.03929** | **1.2360** | 2.9970 | **−1.7610** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8391642** | **0.99108** | 0.96629 | **+0.02479** | **0.3891** | 1.4680 | **−1.0789** | same-image | drifted (different image) | **HSWQ** |
| **15820493** | **0.97144** | 0.83726 | **+0.13418** | **1.0790** | 6.3360 | **−5.2570** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | 0.97988 | **0.98319** | −0.00331 | 0.6792 | **0.5720** | +0.1072 | drifted (different image) | same-image | Native |
| **71058294** | **0.97089** | 0.94087 | **+0.03002** | **1.1210** | 2.3010 | **−1.1800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **128491703** | **0.99204** | 0.98145 | **+0.01059** | **0.3477** | 0.8140 | **−0.4663** | same-image | same-image | **HSWQ** |
| **285039184** | **0.99162** | 0.97012 | **+0.02150** | **0.2807** | 1.0090 | **−0.7283** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | **0.98492** | 0.98121 | **+0.00371** | **0.7756** | 0.9716 | **−0.1960** | same-image | same-image | **HSWQ** |
| **762019483** | **0.95617** | 0.87370 | **+0.08247** | **1.5840** | 4.7160 | **−3.1320** | drifted (different image) | drifted (different image) | **HSWQ** |
| **938174026** | 0.98086 | **0.99242** | −0.01156 | 0.8625 | **0.3430** | +0.5195 | same-image | same-image | Native |
| **1409285713** | **0.99475** | 0.98039 | **+0.01436** | **0.1942** | 0.7332 | **−0.5390** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.99408** | 0.97856 | **+0.01552** | **0.2417** | 0.8802 | **−0.6385** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.93939** | 0.90623 | **+0.03316** | **2.8900** | 4.6430 | **−1.7530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4195820371** | **0.96942** | 0.94034 | **+0.02908** | **1.3480** | 2.6360 | **−1.2880** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.98075** | **0.94910** | **+0.03165** | **0.7713** | **2.0174** | **−1.2461** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (23/25)** |

---

### 2.18. waiANIPONYXL_v140 (1on re650)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.95741** | 0.93961 | **+0.01780** |
| **Min Final Cosine** (↑ better) | **0.78631** | 0.70634 | **+0.07997** |
| **Max Final Cosine** (↑ better) | **0.99484** | 0.99220 | **+0.00264** |
| **Mean Final Latent MSE** (↓ better) | **1.9102** | 2.7537 | **−0.8435 (31% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+13.6%** (9.42s → 8.14s) | +1.4% (9.31s → 9.18s) | — |
| **Trajectory Verdict** | 11/25 same-image, 14/25 drifted | 5/25 same-image, 20/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (waiANIPONYXL_v140)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97734** | 0.97562 | **+0.00172** | **0.9412** | 1.0210 | **−0.0798** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.95168** | 0.94285 | **+0.00883** | **1.9060** | 2.2810 | **−0.3750** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849** | **0.97135** | 0.95025 | **+0.02110** | **1.1720** | 2.0320 | **−0.8600** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.99484** | 0.99220 | **+0.00264** | **0.2166** | 0.3288 | **−0.1122** | same-image | same-image | **HSWQ** |
| **7391** | **0.97680** | 0.95558 | **+0.02122** | **0.9849** | 1.8850 | **−0.9001** | drifted (different image) | drifted (different image) | **HSWQ** |
| **18429** | **0.98365** | 0.98245 | **+0.00120** | **0.6520** | 0.6998 | **−0.0478** | same-image | same-image | **HSWQ** |
| **53082** | **0.98257** | 0.95604 | **+0.02653** | **0.7344** | 1.8760 | **−1.1416** | same-image | drifted (different image) | **HSWQ** |
| **149206** | **0.87495** | 0.85493 | **+0.02002** | **4.5950** | 5.4290 | **−0.8340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **382715** | **0.98895** | 0.96431 | **+0.02464** | **0.3879** | 1.2660 | **−0.8781** | same-image | drifted (different image) | **HSWQ** |
| **826401** | **0.98405** | 0.98345 | **+0.00060** | **0.7366** | 0.7818 | **−0.0452** | same-image | same-image | **HSWQ** |
| **1938502** | **0.97658** | 0.97482 | **+0.00176** | **0.8935** | 0.9696 | **−0.0761** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4710928** | **0.98119** | 0.92641 | **+0.05478** | **0.9712** | 3.7310 | **−2.7598** | same-image | drifted (different image) | **HSWQ** |
| **8391642** | **0.96832** | 0.91160 | **+0.05672** | **1.4230** | 3.8840 | **−2.4610** | drifted (different image) | drifted (different image) | **HSWQ** |
| **15820493** | **0.78631** | 0.70634 | **+0.07997** | **11.1700** | 15.8600 | **−4.6900** | drifted (different image) | drifted (different image) | **HSWQ** |
| **36192847** | **0.95715** | 0.94822 | **+0.00893** | **1.7100** | 2.0860 | **−0.3760** | drifted (different image) | drifted (different image) | **HSWQ** |
| **71058294** | **0.98679** | 0.93622 | **+0.05057** | **0.5381** | 2.6210 | **−2.0829** | same-image | drifted (different image) | **HSWQ** |
| **128491703** | **0.96383** | 0.96021 | **+0.00362** | **1.7350** | 1.8760 | **−0.1410** | drifted (different image) | drifted (different image) | **HSWQ** |
| **285039184** | **0.95661** | 0.94620 | **+0.01041** | **1.8180** | 2.2520 | **−0.4340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **491730285** | **0.96946** | 0.96124 | **+0.00822** | **1.6800** | 2.1490 | **−0.4690** | drifted (different image) | drifted (different image) | **HSWQ** |
| **762019483** | 0.90568 | **0.92444** | −0.01876 | 3.8630 | **3.1130** | +0.7500 | drifted (different image) | drifted (different image) | Native |
| **938174026** | 0.98322 | **0.98736** | −0.00414 | 0.9037 | **0.6830** | +0.2207 | same-image | same-image | Native |
| **1409285713** | **0.98952** | 0.98557 | **+0.00395** | **0.4528** | 0.6280 | **−0.1752** | same-image | same-image | **HSWQ** |
| **2683910547** | **0.98343** | 0.97035 | **+0.01308** | **0.7925** | 1.4290 | **−0.6365** | same-image | drifted (different image) | **HSWQ** |
| **3851729406** | **0.98356** | 0.96652 | **+0.01704** | **0.9131** | 1.8620 | **−0.9489** | same-image | drifted (different image) | **HSWQ** |
| **4195820371** | **0.85733** | 0.82715 | **+0.03018** | **6.5650** | 8.0990 | **−1.5340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.95741** | **0.93961** | **+0.01780** | **1.9102** | **2.7537** | **−0.8435** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (23/25)** |

---

### 2.19. waiREALISM_v10 (1on re590)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98091** | 0.97399 | **+0.00692** |
| **Min Final Cosine** (↑ better) | **0.90798** | 0.90241 | **+0.00557** |
| **Max Final Cosine** (↑ better) | **0.99712** | 0.99572 | **+0.00140** |
| **Mean Final Latent MSE** (↓ better) | **0.5663** | 0.7193 | **−0.1530 (21% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Speedup (avg wall/seed)** (↑ better) | **+18.0%** (9.65s → 7.91s) | +0.7% (9.82s → 9.75s) | — |
| **Trajectory Verdict** | 19/25 same-image, 6/25 drifted | 17/25 same-image, 8/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (waiREALISM_v10)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.96436 | **0.99096** | −0.02660 | 1.1900 | **0.3011** | +0.8889 | drifted (different image) | same-image | Native |
| **137** | **0.98939** | 0.98224 | **+0.00715** | **0.2843** | 0.4751 | **−0.1908** | same-image | same-image | **HSWQ** |
| **849** | **0.99198** | 0.99161 | **+0.00037** | **0.2318** | 0.2413 | **−0.0095** | same-image | same-image | **HSWQ** |
| **2024** | **0.99542** | 0.98914 | **+0.00628** | **0.1033** | 0.2442 | **−0.1409** | same-image | same-image | **HSWQ** |
| **7391** | **0.98119** | 0.94644 | **+0.03475** | **0.4231** | 1.2100 | **−0.7869** | same-image | drifted (different image) | **HSWQ** |
| **18429** | **0.99251** | 0.97760 | **+0.01491** | **0.1831** | 0.5449 | **−0.3618** | same-image | drifted (different image) | **HSWQ** |
| **53082** | 0.97256 | **0.99133** | −0.01877 | 0.9969 | **0.3146** | +0.6823 | drifted (different image) | same-image | Native |
| **149206** | 0.91668 | **0.95181** | −0.03513 | 2.3060 | **1.3090** | +0.9970 | drifted (different image) | drifted (different image) | Native |
| **382715** | **0.99649** | 0.99506 | **+0.00143** | **0.0893** | 0.1253 | **−0.0360** | same-image | same-image | **HSWQ** |
| **826401** | **0.90798** | 0.90241 | **+0.00557** | **2.4780** | 2.6360 | **−0.1580** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1938502** | **0.99185** | 0.99085 | **+0.00100** | **0.2075** | 0.2323 | **−0.0248** | same-image | same-image | **HSWQ** |
| **4710928** | **0.99276** | 0.98256 | **+0.01020** | **0.2062** | 0.4981 | **−0.2919** | same-image | same-image | **HSWQ** |
| **8391642** | **0.99637** | 0.99289 | **+0.00348** | **0.0982** | 0.1924 | **−0.0942** | same-image | same-image | **HSWQ** |
| **15820493** | **0.98261** | 0.98166 | **+0.00095** | **0.6900** | 0.7255 | **−0.0355** | same-image | same-image | **HSWQ** |
| **36192847** | **0.99466** | 0.96271 | **+0.03195** | **0.1987** | 1.3920 | **−1.1933** | same-image | drifted (different image) | **HSWQ** |
| **71058294** | 0.94889 | **0.98896** | −0.04007 | 1.5420 | **0.3351** | +1.2069 | drifted (different image) | same-image | Native |
| **128491703** | **0.99678** | 0.96197 | **+0.03481** | **0.0844** | 0.9817 | **−0.8973** | same-image | drifted (different image) | **HSWQ** |
| **285039184** | **0.98593** | 0.92507 | **+0.06086** | **0.2754** | 1.4890 | **−1.2136** | same-image | drifted (different image) | **HSWQ** |
| **491730285** | 0.98013 | **0.99067** | −0.01054 | 0.8658 | **0.4028** | +0.4630 | same-image | same-image | Native |
| **762019483** | **0.99526** | 0.98799 | **+0.00727** | **0.1777** | 0.4487 | **−0.2710** | same-image | same-image | **HSWQ** |
| **938174026** | **0.99437** | 0.98762 | **+0.00675** | **0.1764** | 0.3870 | **−0.2106** | same-image | same-image | **HSWQ** |
| **1409285713** | 0.98858 | **0.98921** | −0.00063 | 0.3286 | **0.3095** | +0.0191 | same-image | same-image | Native |
| **2683910547** | **0.97906** | 0.90359 | **+0.07547** | **0.5915** | 2.7030 | **−2.1115** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3851729406** | **0.99712** | 0.99572 | **+0.00140** | **0.1028** | 0.1527 | **−0.0499** | same-image | same-image | **HSWQ** |
| **4195820371** | **0.98973** | 0.98955 | **+0.00018** | **0.3271** | 0.3316 | **−0.0045** | same-image | same-image | **HSWQ** |
| **Mean** | **0.98091** | **0.97399** | **+0.00692** | **0.5663** | **0.7193** | **−0.1530** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (19/25)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Bifurcation behaviour:** across all 19 SDXL models (475 seed evaluations per arm), Native ConvRot INT8 shows 3 bifurcated seeds while HSWQ ConvRot INT8 shows 1. HSWQ keeps the same-image count at 260/475 vs 174/475 for Native.

2. **Meaningful cosine gain:** the family mean cosine is 0.97006 for HSWQ vs 0.95356 for Native (+0.01650). HSWQ is ahead on mean cosine in 19/19 models.

3. **Latent MSE reduction:** mean final latent MSE is 1.4989 (HSWQ) vs 2.2922 (Native), about 35% lower error on average.

4. **Worst-case robustness:** the worst per-model minimum cosine is 0.78631 (HSWQ) vs 0.69028 (Native); best per-model maximum cosine is 0.99848 (HSWQ) vs 0.99714 (Native).

5. **Wall-clock:** HSWQ averages a +14.4% speedup over FP16 per seed across the family.

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
- **Setup tags:** `reNNN` = impact-analysis re-estimation round used for layer selection; `1on` = bias correction ON. Taken verbatim from each HSWQ run tag in `score_sdxl_int8.txt`.
- **Speedup:** `(FP16_avg_wall − INT8_avg_wall) / FP16_avg_wall` per seed; positive = INT8 is faster.
- **Protocol:** 25 fixed random seeds, 25 steps, 1024×1024, cfg 7.0, dpmpp_2m / karras. See [How to quantize SDXL.md](../md/How%20to%20quantize%20SDXL.md).