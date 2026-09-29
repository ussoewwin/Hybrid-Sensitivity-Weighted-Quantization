# Krea2 ConvRot INT8 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ ConvRot INT8** vs **Native ConvRot INT8** on the Krea2 architecture family.

**Source:** `benchmark result/score_krea2_int8.txt`  
**Evaluation Script:** `benchmark/krea2_int8_traj_compare.py` (25-seed deterministic trajectory analysis)

**Column labels from the score log:**

| Label | Meaning |
|-------|---------|
| `s+N` | `--keep_sensitive N`: top N sensitive layers reverted to BF16 by the 4-axis composite ranking (DualMonitor E[x^2] × HistCosine V5 × NVFP4 measured error × SVD Leverage) |
| `b+N` | `--blacklist_keep N`: additional N layers reverted to BF16 from the same ranking pool |
| `hswq不要` | Native ConvRot INT8 only — no HSWQ pack benchmarked for this checkpoint |

---

## 1. Summary Comparison (HSWQ ConvRot INT8 vs Native ConvRot INT8)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **moodyKrea2Mix_v7seBF16TYJRVol2** | s+5 | **0.97127** | 0.96393 | **+0.00734** | **0.0681** | 0.0856 | **0/25 (0%)** | 1/25 (4%) | **HSWQ** |
| **moodyCutieMixKrea2_v40BF16** | — | **0.96453** | 0.95139 | **+0.01314** | **0.0855** | 0.1172 | **0/25 (0%)** | 2/25 (8%) | **HSWQ** |
| **sickOllie_krea2** | s+4 b+4 | **0.97226** | 0.96882 | **+0.00344** | **0.0737** | 0.0829 | **0/25 (0%)** | 0/25 (0%) | **HSWQ** |
| **moodyKrea2MixUncensoredV8SEGoes_v8se_bf16** | — | **0.97671** | 0.95265 | **+0.02406** | **0.0551** | 0.1119 | **0/25 (0%)** | 2/25 (8%) | **HSWQ** |
| **DasiwaKrea2TurboRaw_cutedisasterV2Turbo_full_bf16** | s+7 | **0.97419** | 0.95267 | **+0.02152** | **0.0602** | 0.1107 | **0/25 (0%)** | 0/25 (0%) | **HSWQ** |
| **Family Average** | — | **0.97179** | 0.95789 | **+0.01390** | **0.0685** | 0.1016 | **0/125 (0.0%)** | 5/125 (4.0%) | **HSWQ (5/5 models)** |

**Winner** = better on both mean cosine and mean MSE.

---

## 2. Detailed Results per Model

### 2.1. moodyKrea2Mix_v7seBF16TYJRVol2 (s+5)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97127** | 0.96393 | **+0.00734** |
| **Min Final Cosine** (↑ better) | **0.93938** | 0.87902 | **+0.06036** |
| **Max Final Cosine** (↑ better) | **0.99268** | 0.99339 | **−0.00071** |
| **Mean Final Latent MSE** (↓ better) | **0.0681** | 0.0856 | **−0.0175 (20% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 1/25 (4%) | **HSWQ eliminates bifurcations** |
| **Trajectory Verdict** | 12/25 same-image, 13/25 drifted | 6/25 same-image, 18/25 drifted, 1/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyKrea2Mix_v7seBF16TYJRVol2)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98754** | 0.97412 | +0.01342 | **0.0297** | 0.0616 | −0.0318 | same-image | drifted (different image) | **HSWQ** |
| **1337** | **0.93938** | 0.93332 | +0.00606 | **0.1451** | 0.1595 | −0.0144 | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.94179** | 0.93727 | +0.00452 | **0.1327** | 0.1429 | −0.0102 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8888** | **0.98247** | 0.95843 | +0.02404 | **0.0419** | 0.0987 | −0.0568 | same-image | drifted (different image) | **HSWQ** |
| **12345** | 0.95980 | **0.96532** | −0.00552 | 0.0965 | **0.0834** | +0.0130 | drifted (different image) | drifted (different image) | **Native** |
| **45678** | **0.96849** | 0.96540 | +0.00309 | **0.0735** | 0.0810 | −0.0075 | drifted (different image) | drifted (different image) | **HSWQ** |
| **98765** | **0.98550** | 0.97919 | +0.00631 | **0.0334** | 0.0480 | −0.0146 | same-image | drifted (different image) | **HSWQ** |
| **102030** | 0.98489 | **0.98690** | −0.00201 | 0.0369 | **0.0320** | +0.0049 | same-image | same-image | **Native** |
| **246810** | **0.99039** | 0.98597 | +0.00442 | **0.0233** | 0.0340 | −0.0107 | same-image | same-image | **HSWQ** |
| **314159** | 0.96365 | **0.97125** | −0.00760 | 0.0883 | **0.0700** | +0.0184 | drifted (different image) | drifted (different image) | **Native** |
| **555555** | 0.94976 | **0.95627** | −0.00651 | 0.1178 | **0.1024** | +0.0154 | drifted (different image) | drifted (different image) | **Native** |
| **777777** | **0.94035** | 0.93927 | +0.00108 | **0.1403** | 0.1423 | −0.0020 | drifted (different image) | drifted (different image) | **HSWQ** |
| **849201** | 0.95911 | **0.96225** | −0.00314 | 0.0966 | **0.0889** | +0.0077 | drifted (different image) | drifted (different image) | **Native** |
| **1048576** | **0.98591** | 0.97925 | +0.00666 | **0.0340** | 0.0500 | −0.0161 | same-image | drifted (different image) | **HSWQ** |
| **2847193** | **0.95994** | 0.95130 | +0.00864 | **0.0979** | 0.1195 | −0.0216 | drifted (different image) | drifted (different image) | **HSWQ** |
| **5928104** | 0.98272 | **0.98358** | −0.00086 | 0.0416 | **0.0395** | +0.0021 | same-image | same-image | **Native** |
| **7182818** | 0.99219 | **0.99339** | −0.00120 | 0.0190 | **0.0160** | +0.0029 | same-image | same-image | **Native** |
| **9999999** | **0.99268** | 0.98502 | +0.00766 | **0.0165** | 0.0336 | −0.0172 | same-image | same-image | **HSWQ** |
| **14285714** | **0.98267** | 0.97050 | +0.01217 | **0.0389** | 0.0664 | −0.0275 | same-image | drifted (different image) | **HSWQ** |
| **27182818** | **0.98306** | 0.97835 | +0.00471 | **0.0392** | 0.0501 | −0.0109 | same-image | drifted (different image) | **HSWQ** |
| **48151623** | **0.97795** | 0.97378 | +0.00417 | **0.0535** | 0.0636 | −0.0101 | drifted (different image) | drifted (different image) | **HSWQ** |
| **67391024** | **0.97545** | 0.87902 | +0.09643 | **0.0591** | 0.2940 | −0.2349 | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **83910294** | **0.95625** | 0.94919 | +0.00706 | **0.1020** | 0.1178 | −0.0158 | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | 0.98770 | **0.99016** | −0.00246 | 0.0305 | **0.0243** | +0.0061 | same-image | same-image | **Native** |
| **204858291** | **0.95209** | 0.94973 | +0.00236 | **0.1148** | 0.1202 | −0.0054 | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.97127** | 0.96393 | **+0.00734** | **0.0681** | 0.0856 | **−0.0175** | **0/25 Bifurcated** | **1/25 Bifurcated** | **HSWQ (17/25)** |

---

### 2.2. moodyCutieMixKrea2_v40BF16

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96453** | 0.95139 | **+0.01314** |
| **Min Final Cosine** (↑ better) | **0.88941** | 0.87967 | **+0.00974** |
| **Max Final Cosine** (↑ better) | **0.99424** | 0.99071 | **+0.00353** |
| **Mean Final Latent MSE** (↓ better) | **0.0855** | 0.1172 | **−0.0317 (27% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 2/25 (8%) | **HSWQ eliminates bifurcations** |
| **Trajectory Verdict** | 8/25 same-image, 17/25 drifted | 7/25 same-image, 16/25 drifted, 2/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyCutieMixKrea2_v40BF16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.96488** | 0.94885 | +0.01603 | **0.0838** | 0.1215 | −0.0377 | drifted (different image) | drifted (different image) | **HSWQ** |
| **1337** | **0.95580** | 0.95349 | +0.00231 | **0.1086** | 0.1143 | −0.0057 | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.96684** | 0.96364 | +0.00320 | **0.0842** | 0.0917 | −0.0075 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8888** | 0.97654 | **0.99071** | −0.01417 | 0.0547 | **0.0216** | +0.0331 | drifted (different image) | same-image | **Native** |
| **12345** | **0.95958** | 0.95378 | +0.00580 | **0.0902** | 0.1042 | −0.0140 | drifted (different image) | drifted (different image) | **HSWQ** |
| **45678** | **0.94574** | 0.88234 | +0.06340 | **0.1308** | 0.2824 | −0.1516 | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **98765** | 0.98228 | **0.98781** | −0.00553 | 0.0415 | **0.0286** | +0.0129 | same-image | same-image | **Native** |
| **102030** | **0.94868** | 0.91648 | +0.03220 | **0.1303** | 0.2123 | −0.0820 | drifted (different image) | drifted (different image) | **HSWQ** |
| **246810** | **0.94953** | 0.94444 | +0.00509 | **0.1203** | 0.1320 | −0.0117 | drifted (different image) | drifted (different image) | **HSWQ** |
| **314159** | **0.99326** | 0.98451 | +0.00875 | **0.0166** | 0.0381 | −0.0215 | same-image | same-image | **HSWQ** |
| **555555** | **0.97307** | 0.94570 | +0.02737 | **0.0628** | 0.1261 | −0.0633 | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.94104** | 0.93328 | +0.00776 | **0.1369** | 0.1548 | −0.0179 | drifted (different image) | drifted (different image) | **HSWQ** |
| **849201** | 0.98352 | **0.98907** | −0.00555 | 0.0392 | **0.0261** | +0.0132 | same-image | same-image | **Native** |
| **1048576** | **0.97224** | 0.95901 | +0.01323 | **0.0661** | 0.0974 | −0.0313 | drifted (different image) | drifted (different image) | **HSWQ** |
| **2847193** | **0.96101** | 0.93877 | +0.02224 | **0.0972** | 0.1517 | −0.0546 | drifted (different image) | drifted (different image) | **HSWQ** |
| **5928104** | **0.88941** | 0.87967 | +0.00974 | **0.2734** | 0.2970 | −0.0236 | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **7182818** | **0.99424** | 0.98825 | +0.00599 | **0.0143** | 0.0292 | −0.0149 | same-image | same-image | **HSWQ** |
| **9999999** | **0.98209** | 0.95534 | +0.02675 | **0.0423** | 0.1064 | −0.0641 | same-image | drifted (different image) | **HSWQ** |
| **14285714** | 0.98499 | **0.98533** | −0.00034 | 0.0353 | **0.0346** | +0.0007 | same-image | same-image | **Native** |
| **27182818** | **0.94003** | 0.90243 | +0.03760 | **0.1367** | 0.2232 | −0.0865 | drifted (different image) | drifted (different image) | **HSWQ** |
| **48151623** | **0.93634** | 0.90701 | +0.02933 | **0.1553** | 0.2269 | −0.0716 | drifted (different image) | drifted (different image) | **HSWQ** |
| **67391024** | **0.98924** | 0.98459 | +0.00465 | **0.0261** | 0.0374 | −0.0113 | same-image | same-image | **HSWQ** |
| **83910294** | **0.97830** | 0.97317 | +0.00513 | **0.0500** | 0.0618 | −0.0118 | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | **0.96112** | 0.94846 | +0.01266 | **0.1008** | 0.1343 | −0.0335 | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.98342** | 0.96867 | +0.01475 | **0.0399** | 0.0760 | −0.0360 | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.96453** | 0.95139 | **+0.01314** | **0.0855** | 0.1172 | **−0.0317** | **0/25 Bifurcated** | **2/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.3. sickOllie_krea2 (s+4 b+4)

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
| **42** | **0.99011** | 0.98965 | +0.00046 | **0.0274** | 0.0287 | −0.0013 | same-image | same-image | **HSWQ** |
| **1337** | 0.97900 | **0.98365** | −0.00465 | 0.0575 | **0.0448** | +0.0127 | drifted (different image) | same-image | **Native** |
| **2024** | 0.98364 | **0.99274** | −0.00910 | 0.0439 | **0.0195** | +0.0244 | same-image | same-image | **Native** |
| **8888** | 0.98910 | **0.98919** | −0.00009 | 0.0294 | **0.0290** | +0.0003 | same-image | same-image | **Native** |
| **12345** | **0.98427** | 0.97998 | +0.00429 | **0.0416** | 0.0530 | −0.0114 | same-image | drifted (different image) | **HSWQ** |
| **45678** | **0.98982** | 0.98774 | +0.00208 | **0.0273** | 0.0328 | −0.0056 | same-image | same-image | **HSWQ** |
| **98765** | 0.94784 | **0.97769** | −0.02985 | 0.1352 | **0.0581** | +0.0771 | drifted (different image) | drifted (different image) | **Native** |
| **102030** | **0.92469** | 0.92134 | +0.00335 | **0.2009** | 0.2106 | −0.0097 | drifted (different image) | drifted (different image) | **HSWQ** |
| **246810** | **0.98043** | 0.98011 | +0.00032 | **0.0538** | 0.0547 | −0.0010 | same-image | same-image | **HSWQ** |
| **314159** | **0.97996** | 0.97403 | +0.00593 | **0.0569** | 0.0739 | −0.0170 | drifted (different image) | drifted (different image) | **HSWQ** |
| **555555** | **0.94596** | 0.92658 | +0.01938 | **0.1381** | 0.1878 | −0.0497 | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.98881** | 0.97775 | +0.01106 | **0.0301** | 0.0601 | −0.0299 | same-image | drifted (different image) | **HSWQ** |
| **849201** | **0.97818** | 0.95904 | +0.01914 | **0.0563** | 0.1056 | −0.0493 | drifted (different image) | drifted (different image) | **HSWQ** |
| **1048576** | **0.98689** | 0.96255 | +0.02434 | **0.0343** | 0.0984 | −0.0641 | same-image | drifted (different image) | **HSWQ** |
| **2847193** | 0.98603 | **0.98918** | −0.00315 | 0.0391 | **0.0303** | +0.0088 | same-image | same-image | **Native** |
| **5928104** | **0.99228** | 0.99190 | +0.00038 | **0.0210** | 0.0221 | −0.0010 | same-image | same-image | **HSWQ** |
| **7182818** | 0.98317 | **0.99152** | −0.00835 | 0.0433 | **0.0218** | +0.0215 | same-image | same-image | **Native** |
| **9999999** | 0.92188 | **0.92832** | −0.00644 | 0.2034 | **0.1865** | +0.0169 | drifted (different image) | drifted (different image) | **Native** |
| **14285714** | 0.97627 | **0.99159** | −0.01532 | 0.0652 | **0.0231** | +0.0420 | drifted (different image) | same-image | **Native** |
| **27182818** | **0.91630** | 0.88425 | +0.03205 | **0.2281** | 0.3174 | −0.0893 | drifted (different image) | drifted (different image) | **HSWQ** |
| **48151623** | 0.99106 | **0.99171** | −0.00065 | 0.0253 | **0.0235** | +0.0019 | same-image | same-image | **Native** |
| **67391024** | 0.99528 | **0.99725** | −0.00197 | 0.0122 | **0.0071** | +0.0051 | same-image | same-image | **Native** |
| **83910294** | **0.95043** | 0.91328 | +0.03715 | **0.1269** | 0.2229 | −0.0960 | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | **0.97406** | 0.97232 | +0.00174 | **0.0688** | 0.0734 | −0.0046 | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.97098** | 0.96707 | +0.00391 | **0.0764** | 0.0862 | −0.0099 | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.97226** | 0.96882 | **+0.00344** | **0.0737** | 0.0829 | **−0.0092** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (15/25)** |

---

### 2.4. moodyKrea2MixUncensoredV8SEGoes_v8se_bf16

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97671** | 0.95265 | **+0.02406** |
| **Min Final Cosine** (↑ better) | **0.89544** | 0.80774 | **+0.08770** |
| **Max Final Cosine** (↑ better) | **0.99424** | 0.99230 | **+0.00194** |
| **Mean Final Latent MSE** (↓ better) | **0.0551** | 0.1119 | **−0.0568 (51% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 2/25 (8%) | **HSWQ eliminates bifurcations** |
| **Trajectory Verdict** | 16/25 same-image, 9/25 drifted | 7/25 same-image, 16/25 drifted, 2/25 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyKrea2MixUncensoredV8SEGoes_v8se_bf16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.94518** | 0.89347 | +0.05171 | **0.1277** | 0.2513 | −0.1236 | drifted (different image) | drifted (different image) | **HSWQ** |
| **1337** | **0.99260** | 0.96672 | +0.02588 | **0.0178** | 0.0800 | −0.0622 | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.96639** | 0.95802 | +0.00837 | **0.0831** | 0.1041 | −0.0210 | drifted (different image) | drifted (different image) | **HSWQ** |
| **8888** | **0.96461** | 0.95269 | +0.01192 | **0.0834** | 0.1118 | −0.0284 | drifted (different image) | drifted (different image) | **HSWQ** |
| **12345** | **0.98197** | 0.96419 | +0.01778 | **0.0438** | 0.0872 | −0.0434 | same-image | drifted (different image) | **HSWQ** |
| **45678** | **0.99336** | 0.99230 | +0.00106 | **0.0144** | 0.0168 | −0.0023 | same-image | same-image | **HSWQ** |
| **98765** | **0.98496** | 0.95724 | +0.02772 | **0.0335** | 0.0954 | −0.0619 | same-image | drifted (different image) | **HSWQ** |
| **102030** | **0.99070** | 0.97254 | +0.01816 | **0.0236** | 0.0695 | −0.0459 | same-image | drifted (different image) | **HSWQ** |
| **246810** | **0.99297** | 0.98440 | +0.00857 | **0.0172** | 0.0383 | −0.0211 | same-image | same-image | **HSWQ** |
| **314159** | **0.99398** | 0.86291 | +0.13107 | **0.0151** | 0.3468 | −0.3317 | same-image | bifurcated @step 11 | **HSWQ** |
| **555555** | **0.98408** | 0.91514 | +0.06894 | **0.0323** | 0.1726 | −0.1403 | same-image | drifted (different image) | **HSWQ** |
| **777777** | 0.98677 | **0.98813** | −0.00136 | 0.0282 | **0.0253** | +0.0029 | same-image | same-image | **Native** |
| **849201** | **0.95146** | 0.80774 | +0.14372 | **0.1137** | 0.4560 | −0.3423 | drifted (different image) | bifurcated @step 11 | **HSWQ** |
| **1048576** | 0.98860 | **0.98868** | −0.00008 | 0.0279 | **0.0279** | +0.0000 | same-image | same-image | **Native** |
| **2847193** | **0.98795** | 0.93817 | +0.04978 | **0.0298** | 0.1536 | −0.1238 | same-image | drifted (different image) | **HSWQ** |
| **5928104** | 0.89544 | **0.93467** | −0.03923 | 0.2636 | **0.1630** | +0.1006 | drifted (different image) | drifted (different image) | **Native** |
| **7182818** | 0.94790 | **0.96747** | −0.01957 | 0.1275 | **0.0801** | +0.0474 | drifted (different image) | drifted (different image) | **Native** |
| **9999999** | **0.96302** | 0.95425 | +0.00877 | **0.0738** | 0.0911 | −0.0173 | drifted (different image) | drifted (different image) | **HSWQ** |
| **14285714** | **0.98874** | 0.98098 | +0.00776 | **0.0239** | 0.0404 | −0.0165 | same-image | same-image | **HSWQ** |
| **27182818** | **0.99424** | 0.98754 | +0.00670 | **0.0119** | 0.0257 | −0.0138 | same-image | same-image | **HSWQ** |
| **48151623** | **0.97412** | 0.97238 | +0.00174 | **0.0627** | 0.0671 | −0.0043 | drifted (different image) | drifted (different image) | **HSWQ** |
| **67391024** | **0.99330** | 0.95448 | +0.03882 | **0.0166** | 0.1127 | −0.0961 | same-image | drifted (different image) | **HSWQ** |
| **83910294** | **0.98905** | 0.97731 | +0.01174 | **0.0218** | 0.0451 | −0.0233 | same-image | drifted (different image) | **HSWQ** |
| **105829104** | **0.97211** | 0.95864 | +0.01347 | **0.0689** | 0.1023 | −0.0334 | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.99419** | 0.98613 | +0.00806 | **0.0139** | 0.0332 | −0.0193 | same-image | same-image | **HSWQ** |
| **Mean** | **0.97671** | 0.95265 | **+0.02406** | **0.0551** | 0.1119 | **−0.0568** | **0/25 Bifurcated** | **2/25 Bifurcated** | **HSWQ (21/25)** |

---

### 2.5. DasiwaKrea2TurboRaw_cutedisasterV2Turbo_full_bf16 (s+7)

#### Metric Overview
| Metric / Property | HSWQ ConvRot INT8 | Native ConvRot INT8 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97419** | 0.95267 | **+0.02152** |
| **Min Final Cosine** (↑ better) | **0.90386** | 0.89645 | **+0.00741** |
| **Max Final Cosine** (↑ better) | **0.99496** | 0.99296 | **+0.00200** |
| **Mean Final Latent MSE** (↓ better) | **0.0602** | 0.1107 | **−0.0505 (46% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/25 (0%)** | 0/25 (0%) | **Zero bifurcations** |
| **Trajectory Verdict** | 12/25 same-image, 13/25 drifted | 3/25 same-image, 22/25 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (DasiwaKrea2TurboRaw_cutedisasterV2Turbo_full_bf16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97770** | 0.97751 | +0.00019 | **0.0534** | 0.0537 | −0.0002 | drifted (different image) | drifted (different image) | **HSWQ** |
| **1337** | **0.98844** | 0.96263 | +0.02581 | **0.0267** | 0.0856 | −0.0589 | same-image | drifted (different image) | **HSWQ** |
| **2024** | **0.98101** | 0.96529 | +0.01572 | **0.0435** | 0.0795 | −0.0360 | same-image | drifted (different image) | **HSWQ** |
| **8888** | **0.98924** | 0.97905 | +0.01019 | **0.0246** | 0.0478 | −0.0232 | same-image | drifted (different image) | **HSWQ** |
| **12345** | **0.97069** | 0.89645 | +0.07424 | **0.0694** | 0.2448 | −0.1754 | drifted (different image) | drifted (different image) | **HSWQ** |
| **45678** | **0.99037** | 0.98684 | +0.00353 | **0.0222** | 0.0303 | −0.0081 | same-image | same-image | **HSWQ** |
| **98765** | **0.97854** | 0.96608 | +0.01246 | **0.0488** | 0.0769 | −0.0281 | drifted (different image) | drifted (different image) | **HSWQ** |
| **102030** | **0.96639** | 0.92917 | +0.03722 | **0.0888** | 0.1871 | −0.0983 | drifted (different image) | drifted (different image) | **HSWQ** |
| **246810** | **0.98162** | 0.95378 | +0.02784 | **0.0413** | 0.1039 | −0.0626 | same-image | drifted (different image) | **HSWQ** |
| **314159** | **0.97791** | 0.91370 | +0.06421 | **0.0535** | 0.2089 | −0.1554 | drifted (different image) | drifted (different image) | **HSWQ** |
| **555555** | **0.96899** | 0.90800 | +0.06099 | **0.0692** | 0.2046 | −0.1354 | drifted (different image) | drifted (different image) | **HSWQ** |
| **777777** | **0.98738** | 0.97226 | +0.01512 | **0.0296** | 0.0649 | −0.0353 | same-image | drifted (different image) | **HSWQ** |
| **849201** | **0.98380** | 0.94152 | +0.04228 | **0.0389** | 0.1403 | −0.1014 | same-image | drifted (different image) | **HSWQ** |
| **1048576** | **0.99496** | 0.99189 | +0.00307 | **0.0118** | 0.0189 | −0.0071 | same-image | same-image | **HSWQ** |
| **2847193** | **0.95833** | 0.93793 | +0.02040 | **0.0966** | 0.1433 | −0.0467 | drifted (different image) | drifted (different image) | **HSWQ** |
| **5928104** | **0.98663** | 0.94178 | +0.04485 | **0.0318** | 0.1390 | −0.1072 | same-image | drifted (different image) | **HSWQ** |
| **7182818** | 0.99039 | **0.99296** | −0.00257 | 0.0222 | **0.0162** | +0.0059 | same-image | same-image | **Native** |
| **9999999** | 0.90386 | **0.90615** | −0.00229 | 0.2213 | **0.2157** | +0.0056 | drifted (different image) | drifted (different image) | **Native** |
| **14285714** | **0.96502** | 0.94589 | +0.01913 | **0.0832** | 0.1286 | −0.0454 | drifted (different image) | drifted (different image) | **HSWQ** |
| **27182818** | **0.96912** | 0.95311 | +0.01601 | **0.0733** | 0.1107 | −0.0374 | drifted (different image) | drifted (different image) | **HSWQ** |
| **48151623** | **0.95791** | 0.95433 | +0.00358 | **0.1016** | 0.1103 | −0.0087 | drifted (different image) | drifted (different image) | **HSWQ** |
| **67391024** | **0.98870** | 0.96937 | +0.01933 | **0.0246** | 0.0666 | −0.0420 | same-image | drifted (different image) | **HSWQ** |
| **83910294** | **0.96293** | 0.94333 | +0.01960 | **0.0825** | 0.1265 | −0.0440 | drifted (different image) | drifted (different image) | **HSWQ** |
| **105829104** | **0.95367** | 0.94977 | +0.00390 | **0.1031** | 0.1121 | −0.0090 | drifted (different image) | drifted (different image) | **HSWQ** |
| **204858291** | **0.98106** | 0.97792 | +0.00314 | **0.0438** | 0.0510 | −0.0072 | same-image | drifted (different image) | **HSWQ** |
| **Mean** | **0.97419** | 0.95267 | **+0.02152** | **0.0602** | 0.1107 | **−0.0505** | **0/25 Bifurcated** | **0/25 Bifurcated** | **HSWQ (23/25)** |

---

### 2.6. gonzalomoKrea2_v40 — Native ConvRot INT8 only (`hswq不要`)

#### Metric Overview
| Metric / Property | Native ConvRot INT8 (Full Model) |
| :--- | :--- |
| **Mean Final Cosine** (↑ better) | 0.98507 |
| **Min Final Cosine** (↑ better) | 0.93737 |
| **Max Final Cosine** (↑ better) | 0.99785 |
| **Mean Final Latent MSE** (↓ better) | 0.0364 |
| **Bifurcated Seeds Rate** (↓ better) | 0/25 (0%) |
| **Trajectory Verdict** | 20/25 same-image, 5/25 drifted |

Mean final cosine 0.98507 ≥ 0.98 with zero bifurcations clears the native fidelity gate defined in [How to quantize Krea2.md](../md/How%20to%20quantize%20Krea2.md), so no HSWQ pack was benchmarked for this checkpoint (`hswq不要`).

---

## 3. Key Findings and Trajectory Analysis

1. **Zero HSWQ trajectory bifurcations across the family:**
   Across the 5 compared checkpoints (125 seed evaluations per arm), Native ConvRot INT8 produces 5 bifurcated seeds (4.0%) — sudden single-step trajectory jumps into a completely different picture attractor basin — while HSWQ ConvRot INT8 shows **0/125 (0%)**. The same-image seed count rises from 35/125 (Native) to 61/125 (HSWQ).
2. **Consistent cosine gain across all checkpoints:**
   The family mean final cosine is 0.97179 for HSWQ vs 0.95789 for Native (+0.01390); HSWQ is ahead on mean cosine in 5/5 models.
3. **Latent MSE reduction:**
   Family mean final latent MSE is 0.0685 (HSWQ) vs 0.1016 (Native), about 33% lower error on average.
4. **Worst-case robustness:**
   The worst per-model minimum cosine is 0.88941 (HSWQ) vs 0.80774 (Native); the deepest Native collapse (0.80774 on moodyKrea2MixUncensoredV8SEGoes_v8se_bf16) is lifted well clear of the collapse zone under HSWQ.
5. **Native-only fidelity gate (`hswq不要`):**
   gonzalomoKrea2_v40 reaches mean final cosine 0.98507 with 20/25 same-image and 0 bifurcations under Native ConvRot INT8 alone, clearing the mean ≥ 0.98 gate in [How to quantize Krea2.md](../md/How%20to%20quantize%20Krea2.md) — no HSWQ protection pack is needed for this checkpoint.

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
- **Setup tags:** `s+N` = `--keep_sensitive N` (top N sensitive layers reverted to BF16 by the 4-axis composite ranking); `b+N` = `--blacklist_keep N` (additional N layers reverted from the same ranking pool). Taken verbatim from each HSWQ run tag in `score_krea2_int8.txt`; `—` = no tag recorded in the log. `hswq不要` = native-only checkpoint (no HSWQ pack benchmarked).
- **Protocol:** 25 fixed random seeds, 25 steps, 1024×1024, cfg 1.0, euler / simple. See [How to quantize Krea2.md](../md/How%20to%20quantize%20Krea2.md).
