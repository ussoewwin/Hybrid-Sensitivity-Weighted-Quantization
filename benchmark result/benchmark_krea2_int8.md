# Krea2 ConvRot INT8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ ConvRot INT8** vs **Native ConvRot INT8** on the Krea2 architecture family.

**Source:** `benchmark result/score_krea2_int8.txt`  
**Evaluation Script:** `benchmark/krea2_int8_traj_compare.py` (25-seed deterministic trajectory analysis)

---

## 1. Summary Comparison (HSWQ ConvRot INT8 vs Native ConvRot INT8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **moodyKrea2Mix_v7seBF16TYJRVol2** | — | **0.97343** | 0.96413 | **+0.00930** | **0.0628** | 0.0848 | **0/25 (0%)** | 1/25 (4%) | **HSWQ** |
| **gonzalomoKrea2_v40** | hswq not needed | — | 0.98507 | — | — | 0.0364 | — | 0/25 (0%) | Native only |
| **moodyCutieMixKrea2_v40BF16** | — | **0.96453** | 0.95139 | **+0.01314** | **0.0855** | 0.1172 | **0/25 (0%)** | 2/25 (8%) | **HSWQ** |
| **sickOllie_krea2** | s+4 b+4 | **0.97226** | 0.96882 | **+0.00344** | **0.0737** | 0.0829 | **0/25 (0%)** | 0/25 (0%) | **HSWQ** |
| **Family Average** | — | **0.97007** | 0.96145 | **+0.00863** | **0.0740** | 0.0949 | **0/75 (0.0%)** | 3/75 (4.0%) | **HSWQ (3/3 models)** |

---

## 2. Detailed Results per Model

### 2.1. moodyKrea2Mix_v7seBF16TYJRVol2

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97343** | 0.96413 | **+0.00930** |
| **Min Final Cosine** (↑ better) | **0.90685** | 0.87333 | **+0.03352** |
| **Max Final Cosine** (↑ better) | **0.99608** | 0.99105 | **+0.00503** |
| **Mean Final Latent MSE** (↓ better) | **0.0628** | 0.0848 | **−0.0220 (26% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 1/25 (4%) | **1/25 Native bifurcations eliminated** |
| **Trajectory Verdict** | 14/25 same-image, 11/25 drifted | 9/25 same-image, 15/25 drifted, 1/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyKrea2Mix_v7seBF16TYJRVol2)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98759** | 0.97099 | **+0.01660** | **0.0296** | 0.0688 | **−0.0392** | same-image | drifted (different image) | **HSWQ** |
| **1337** | **0.96481** | 0.95965 | **+0.00516** | **0.0843** | 0.0965 | **−0.0122** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.90685** | 0.87333 | **+0.03352** | **0.2144** | 0.2889 | **−0.0745** | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **8888** | 0.98259 | **0.98483** | −0.00224 | 0.0415 | **0.0360** | +0.0055 | same-image | same-image | Native |
| **12345** | **0.96205** | 0.91851 | **+0.04354** | **0.0910** | 0.1955 | **−0.1045** | drifted (different image) | drifted (different image) | **HSWQ** |
| **45678** | **0.96279** | 0.96217 | **+0.00062** | **0.0871** | 0.0883 | **−0.0012** | drifted (different image) | drifted (different image) | **HSWQ** |
| **98765** | **0.98003** | 0.96927 | **+0.01076** | **0.0460** | 0.0708 | **−0.0248** | same-image | drifted (different image) | **HSWQ** |
| **102030** | **0.98458** | 0.97461 | **+0.00997** | **0.0376** | 0.0624 | **−0.0248** | same-image | drifted (different image) | **HSWQ** |
| **246810** | **0.99119** | 0.98905 | **+0.00214** | **0.0214** | 0.0266 | **−0.0052** | same-image | same-image | **HSWQ** |
| **314159** | **0.98160** | 0.98156 | **+0.00004** | **0.0447** | 0.0450 | **−0.0002** | same-image | same-image | **HSWQ** |
| **555555** | **0.95277** | 0.94863 | **+0.00414** | **0.1108** | 0.1203 | **−0.0095** | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.97905** | 0.97458 | **+0.00447** | **0.0490** | 0.0594 | **−0.0104** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849201** | **0.96899** | 0.94606 | **+0.02293** | **0.0730** | 0.1265 | **−0.0535** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1048576** | **0.98412** | 0.97973 | **+0.00439** | **0.0383** | 0.0488 | **−0.0105** | same-image | drifted (different image) | **HSWQ** |
| **2847193** | **0.97023** | 0.95968 | **+0.01055** | **0.0728** | 0.0985 | **−0.0256** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5928104** | 0.98525 | **0.98547** | −0.00022 | 0.0355 | **0.0350** | +0.0006 | same-image | same-image | Native |
| **7182818** | **0.99608** | 0.99098 | **+0.00510** | **0.0095** | 0.0219 | **−0.0124** | same-image | same-image | **HSWQ** |
| **9999999** | **0.98612** | 0.98100 | **+0.00512** | **0.0313** | 0.0426 | **−0.0113** | same-image | same-image | **HSWQ** |
| **14285714** | **0.99096** | 0.97558 | **+0.01538** | **0.0203** | 0.0548 | **−0.0345** | same-image | drifted (different image) | **HSWQ** |
| **27182818** | 0.98454 | **0.98921** | −0.00467 | 0.0356 | **0.0249** | +0.0108 | same-image | same-image | Native |
| **48151623** | 0.98021 | **0.98362** | −0.00341 | 0.0481 | **0.0398** | +0.0083 | same-image | same-image | Native |
| **67391024** | **0.97416** | 0.90208 | **+0.07208** | **0.0624** | 0.2383 | **−0.1759** | drifted (different image) | drifted (different image) | **HSWQ** |
| **83910294** | 0.93227 | **0.95476** | −0.02249 | 0.1584 | **0.1045** | +0.0539 | drifted (different image) | drifted (different image) | Native |
| **105829104** | **0.99180** | 0.99105 | **+0.00075** | **0.0203** | 0.0221 | **−0.0019** | same-image | same-image | **HSWQ** |
| **204858291** | 0.95509 | **0.95686** | −0.00177 | 0.1076 | **0.1032** | +0.0044 | drifted (different image) | drifted (different image) | Native |
| **Mean** | **0.97343** | **0.96413** | **+0.00930** | **0.0628** | **0.0848** | **−0.0220** | **0/25 Bifurcated** | **1/25 Bifurcated** | **HSWQ (19/25)** |

---

### 2.2. gonzalomoKrea2_v40 (hswq not needed)

#### Metric Overview
| Metric / Property | Native ConvRot INT8 (Full Model) |
| :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.98507** |
| **Min Final Cosine** (↑ better) | **0.93737** |
| **Max Final Cosine** (↑ better) | **0.99785** |
| **Mean Final Latent MSE** (↓ better) | **0.0364** |
| **Bifurcated Seeds Rate** (↓ better) | 0/25 (0%) |
| **Trajectory Verdict** | 20/25 same-image, 5/25 drifted |

#### Per Seed (gonzalomoKrea2_v40)
| Seed | Native Cosine | Native MSE | Max Drop | Native Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **42** | 0.99180 | 0.0194 | 0.0029 | same-image |
| **1337** | 0.95553 | 0.1014 | 0.0147 | drifted (different image) |
| **2024** | 0.99424 | 0.0135 | 0.0020 | same-image |
| **8888** | 0.99208 | 0.0201 | 0.0026 | same-image |
| **12345** | 0.95126 | 0.1224 | 0.0173 | drifted (different image) |
| **45678** | 0.99732 | 0.0069 | 0.0008 | same-image |
| **98765** | 0.99007 | 0.0242 | 0.0032 | same-image |
| **102030** | 0.99468 | 0.0145 | 0.0019 | same-image |
| **246810** | 0.98353 | 0.0413 | 0.0057 | same-image |
| **314159** | 0.99421 | 0.0150 | 0.0020 | same-image |
| **555555** | 0.98969 | 0.0248 | 0.0033 | same-image |
| **777777** | 0.99011 | 0.0253 | 0.0034 | same-image |
| **849201** | 0.93737 | 0.1514 | 0.0217 | drifted (different image) |
| **1048576** | 0.99785 | 0.0056 | 0.0007 | same-image |
| **2847193** | 0.99602 | 0.0107 | 0.0014 | same-image |
| **5928104** | 0.99226 | 0.0215 | 0.0028 | same-image |
| **7182818** | 0.98263 | 0.0390 | 0.0061 | same-image |
| **9999999** | 0.97723 | 0.0555 | 0.0081 | drifted (different image) |
| **14285714** | 0.98607 | 0.0370 | 0.0046 | same-image |
| **27182818** | 0.99299 | 0.0179 | 0.0022 | same-image |
| **48151623** | 0.99710 | 0.0073 | 0.0010 | same-image |
| **67391024** | 0.97813 | 0.0506 | 0.0069 | drifted (different image) |
| **83910294** | 0.98762 | 0.0307 | 0.0038 | same-image |
| **105829104** | 0.99581 | 0.0101 | 0.0013 | same-image |
| **204858291** | 0.98118 | 0.0432 | 0.0061 | same-image |
| **Mean** | **0.98507** | **0.0364** | — | **0/25 Bifurcated** |

---

### 2.3. moodyCutieMixKrea2_v40BF16

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96453** | 0.95139 | **+0.01314** |
| **Min Final Cosine** (↑ better) | **0.88941** | 0.87967 | **+0.00974** |
| **Max Final Cosine** (↑ better) | **0.99424** | 0.99071 | **+0.00353** |
| **Mean Final Latent MSE** (↓ better) | **0.0855** | 0.1172 | **−0.0317 (27% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 2/25 (8%) | **2/25 Native bifurcations eliminated** |
| **Trajectory Verdict** | 8/25 same-image, 17/25 drifted | 7/25 same-image, 16/25 drifted, 2/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyCutieMixKrea2_v40BF16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.96488** | 0.94885 | **+0.01603** | **0.0838** | 0.1215 | **−0.0377** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1337** | **0.95580** | 0.95349 | **+0.00231** | **0.1086** | 0.1143 | **−0.0057** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.96684** | 0.96364 | **+0.00320** | **0.0842** | 0.0917 | **−0.0075** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8888** | 0.97654 | **0.99071** | −0.01417 | 0.0547 | **0.0216** | +0.0331 | drifted (different image) | same-image | Native |
| **12345** | **0.95958** | 0.95378 | **+0.00580** | **0.0902** | 0.1042 | **−0.0140** | drifted (different image) | drifted (different image) | **HSWQ** |
| **45678** | **0.94574** | 0.88234 | **+0.06340** | **0.1308** | 0.2824 | **−0.1516** | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **98765** | 0.98228 | **0.98781** | −0.00553 | 0.0415 | **0.0286** | +0.0129 | same-image | same-image | Native |
| **102030** | **0.94868** | 0.91648 | **+0.03220** | **0.1303** | 0.2123 | **−0.0820** | drifted (different image) | drifted (different image) | **HSWQ** |
| **246810** | **0.94953** | 0.94444 | **+0.00509** | **0.1203** | 0.1320 | **−0.0117** | drifted (different image) | drifted (different image) | **HSWQ** |
| **314159** | **0.99326** | 0.98451 | **+0.00875** | **0.0166** | 0.0381 | **−0.0215** | same-image | same-image | **HSWQ** |
| **555555** | **0.97307** | 0.94570 | **+0.02737** | **0.0628** | 0.1261 | **−0.0633** | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.94104** | 0.93328 | **+0.00776** | **0.1369** | 0.1548 | **−0.0179** | drifted (different image) | drifted (different image) | **HSWQ** |
| **849201** | 0.98352 | **0.98907** | −0.00555 | 0.0392 | **0.0261** | +0.0132 | same-image | same-image | Native |
| **1048576** | **0.97224** | 0.95901 | **+0.01323** | **0.0661** | 0.0974 | **−0.0313** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2847193** | **0.96101** | 0.93877 | **+0.02224** | **0.0972** | 0.1517 | **−0.0546** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5928104** | **0.88941** | 0.87967 | **+0.00974** | **0.2734** | 0.2970 | **−0.0236** | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **7182818** | **0.99424** | 0.98825 | **+0.00599** | **0.0143** | 0.0292 | **−0.0149** | same-image | same-image | **HSWQ** |
| **9999999** | **0.98209** | 0.95534 | **+0.02675** | **0.0423** | 0.1064 | **−0.0641** | same-image | drifted (different image) | **HSWQ** |
| **14285714** | 0.98499 | **0.98533** | −0.00034 | 0.0353 | **0.0346** | +0.0007 | same-image | same-image | Native |
| **27182818** | **0.94003** | 0.90243 | **+0.03760** | **0.1367** | 0.2232 | **−0.0865** | drifted (different image) | drifted (different image) | **HSWQ** |
| **48151623** | **0.93634** | 0.90701 | **+0.02933** | **0.1553** | 0.2269 | **−0.0716** | drifted (different image) | drifted (different image) | **HSWQ** |
| **67391024** | **0.98924** | 0.98459 | **+0.00465** | **0.0261** | 0.0374 | **−0.0113** | same-image | same-image | **HSWQ** |
| **83910294** | **0.97830** | 0.97317 | **+0.00513** | **0.0500** | 0.0618 | **−0.0118** | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | **0.96112** | 0.94846 | **+0.01266** | **0.1008** | 0.1343 | **−0.0335** | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.98342** | 0.96867 | **+0.01475** | **0.0399** | 0.0760 | **−0.0360** | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.96453** | **0.95139** | **+0.01314** | **0.0855** | **0.1172** | **−0.0317** | **0/25 Bifurcated** | **2/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.4. sickOllie_krea2 (s+4 b+4)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97226** | 0.96882 | **+0.00344** |
| **Min Final Cosine** (↑ better) | **0.91630** | 0.88425 | **+0.03205** |
| **Max Final Cosine** (↑ better) | **0.99528** | 0.99725 | **−0.00197** |
| **Mean Final Latent MSE** (↓ better) | **0.0737** | 0.0829 | **−0.0092 (11% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Trajectory Verdict** | 13/25 same-image, 12/25 drifted | 12/25 same-image, 13/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (sickOllie_krea2)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.99011** | 0.98965 | **+0.00046** | **0.0274** | 0.0287 | **−0.0013** | same-image | same-image | **HSWQ** |
| **1337** | 0.97900 | **0.98365** | −0.00465 | 0.0575 | **0.0448** | +0.0127 | drifted (different image) | same-image | Native |
| **2024** | 0.98364 | **0.99274** | −0.00910 | 0.0439 | **0.0195** | +0.0244 | same-image | same-image | Native |
| **8888** | 0.98910 | **0.98919** | −0.00009 | 0.0294 | **0.0290** | +0.0003 | same-image | same-image | Native |
| **12345** | **0.98427** | 0.97998 | **+0.00429** | **0.0416** | 0.0530 | **−0.0114** | same-image | drifted (different image) | **HSWQ** |
| **45678** | **0.98982** | 0.98774 | **+0.00208** | **0.0273** | 0.0328 | **−0.0056** | same-image | same-image | **HSWQ** |
| **98765** | 0.94784 | **0.97769** | −0.02985 | 0.1352 | **0.0581** | +0.0771 | drifted (different image) | drifted (different image) | Native |
| **102030** | **0.92469** | 0.92134 | **+0.00335** | **0.2009** | 0.2106 | **−0.0097** | drifted (different image) | drifted (different image) | **HSWQ** |
| **246810** | **0.98043** | 0.98011 | **+0.00032** | **0.0538** | 0.0547 | **−0.0010** | same-image | same-image | **HSWQ** |
| **314159** | **0.97996** | 0.97403 | **+0.00593** | **0.0569** | 0.0739 | **−0.0170** | drifted (different image) | drifted (different image) | **HSWQ** |
| **555555** | **0.94596** | 0.92658 | **+0.01938** | **0.1381** | 0.1878 | **−0.0497** | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.98881** | 0.97775 | **+0.01106** | **0.0301** | 0.0601 | **−0.0299** | same-image | drifted (different image) | **HSWQ** |
| **849201** | **0.97818** | 0.95904 | **+0.01914** | **0.0563** | 0.1056 | **−0.0493** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1048576** | **0.98689** | 0.96255 | **+0.02434** | **0.0343** | 0.0984 | **−0.0641** | same-image | drifted (different image) | **HSWQ** |
| **2847193** | 0.98603 | **0.98918** | −0.00315 | 0.0391 | **0.0303** | +0.0088 | same-image | same-image | Native |
| **5928104** | **0.99228** | 0.99190 | **+0.00038** | **0.0210** | 0.0221 | **−0.0010** | same-image | same-image | **HSWQ** |
| **7182818** | 0.98317 | **0.99152** | −0.00835 | 0.0433 | **0.0218** | +0.0215 | same-image | same-image | Native |
| **9999999** | 0.92188 | **0.92832** | −0.00644 | 0.2034 | **0.1865** | +0.0169 | drifted (different image) | drifted (different image) | Native |
| **14285714** | 0.97627 | **0.99159** | −0.01532 | 0.0652 | **0.0231** | +0.0420 | drifted (different image) | same-image | Native |
| **27182818** | **0.91630** | 0.88425 | **+0.03205** | **0.2281** | 0.3174 | **−0.0893** | drifted (different image) | drifted (different image) | **HSWQ** |
| **48151623** | 0.99106 | **0.99171** | −0.00065 | 0.0253 | **0.0235** | +0.0019 | same-image | same-image | Native |
| **67391024** | 0.99528 | **0.99725** | −0.00197 | 0.0122 | **0.0071** | +0.0051 | same-image | same-image | Native |
| **83910294** | **0.95043** | 0.91328 | **+0.03715** | **0.1269** | 0.2229 | **−0.0960** | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | **0.97406** | 0.97232 | **+0.00174** | **0.0688** | 0.0734 | **−0.0046** | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.97098** | 0.96707 | **+0.00391** | **0.0764** | 0.0862 | **−0.0099** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.97226** | **0.96882** | **+0.00344** | **0.0737** | **0.0829** | **−0.0092** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (15/25)** |

---

## 3. Decoded-image results (not on the trajectory gate)

These checkpoints were measured with the decoded-image bench (one fixed prompt and seed, single decoded image, MSE / SSIM against the BF16 reference). Trajectory data has not been produced for them.  
`+N` = number of BF16 protect layers; `1off` = bias correction OFF.

| Model | Setup | HSWQ MSE (↓ better) | HSWQ SSIM (↑ better) | Native MSE | Native SSIM | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| unstableDissolution_Bf16 | +20 1off | **6.5205** | **0.9737** | 10.9235 | 0.9355 | **HSWQ** |
| fasciumKREA2_3MERGE | +10 1off | **6.8111** | **0.9607** | 11.4368 | 0.9170 | **HSWQ** |

---

## 4. Key Findings and Trajectory Analysis

1. **Bifurcations:** across the 3 checkpoints measured on both arms (75 seed evaluations per arm), Native ConvRot INT8 shows 3 bifurcated seeds and HSWQ ConvRot INT8 shows 0. Same-image count is 35/75 (HSWQ) vs 28/75 (Native).

2. **Mean cosine:** family mean is 0.97007 (HSWQ) vs 0.96145 (Native) (+0.00863); HSWQ leads on mean cosine in 3/3 checkpoints.

3. **Latent MSE:** mean final latent MSE is 0.0740 (HSWQ) vs 0.0949 (Native), about 22% lower error on average.

4. **Native-only checkpoint:** `gonzalomoKrea2_v40` reaches mean cosine 0.98507 with 0/25 bifurcated under **native ConvRot INT8 alone**; per the Krea2 recommendation, HSWQ is not needed for this checkpoint (its row carries no HSWQ arm).

5. **Not on the trajectory gate:** the two decoded-image checkpoints (section 3) are reported with the decoded-image bench only; they are not part of the trajectory gate.

---

## 5. Metric Definitions

- **Final Cosine (final-cos):** cosine similarity between the final denoised latent of the FP16 reference and the quantized model. Closer to 1.0 means identical composition, lighting and semantic fidelity.
- **Final MSE (final-mse):** mean squared error of the final latent tensor against the FP16 baseline.
- **Max Step Drop (max-drop):** maximum single-step cosine drop between consecutive sampling steps, quantifying sudden trajectory instability.
- **Bifurcated:** max-step-drop > 0.05 on a single step; a sudden jump into a different picture attractor basin, not gradual degradation.
- **Verdict:**
  - `same-image`: per-seed final cosine ≥ 0.98.
  - `drifted (different image)`: gradual, continuous deviation across steps while keeping a coherent, related image.
  - `bifurcated @step N`: discontinuous trajectory jump at step N.
- **Protocol:** 25 fixed random seeds, 25 steps, dpmpp_2m / karras. See [How to quantize Krea2.md](../md/How%20to%20quantize%20Krea2.md).