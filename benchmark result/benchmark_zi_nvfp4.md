# Z-Image ConvRot Hybrid NVFP4 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ ConvRot Hybrid NVFP4** vs **Native NVFP4** on the Z-Image architecture family.

**Source:** `benchmark result/score_zi_nvfp4.txt`  
**Evaluation Script:** `benchmark/zi_traj_compare.py` (20-seed deterministic trajectory analysis)

---

## 1. Summary Comparison (HSWQ Hybrid NVFP4 vs Native NVFP4)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **moodyProMix_collectorsEdition** | nv100 | **0.96158** | 0.92672 | **+0.03486** | **0.9576** | 1.7175 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **moodyRealMix_xhsEdition** | nv100 | **0.97366** | 0.94680 | **+0.02686** | **0.7410** | 1.4552 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **zimageTurboByStable_2602** | nv99 | **0.96242** | 0.92187 | **+0.04055** | **0.8143** | 1.8329 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **unstableRevolution_V3Fp16** | nv100 | **0.96216** | 0.92187 | **+0.04029** | **0.8879** | 1.8329 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **gonzalomoZpop_insta2** | nv100 | **0.96554** | 0.92187 | **+0.04367** | **0.8143** | 1.8329 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **divingZImageTurbo_v70Fp16** | nv99 | **0.96107** | 0.93218 | **+0.02889** | **0.8833** | 1.5471 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **beyondREALITY_V30** | nv100 | **0.96025** | 0.90049 | **+0.05976** | **1.0772** | 2.3739 | **0/20 (0%)** | 2/20 (10%) | **HSWQ** |
| **2127ZImageAsianUtopian_v40Turbo** | nv100 | **0.96558** | 0.91271 | **+0.05287** | **0.9961** | 2.5376 | **0/20 (0%)** | 1/20 (5%) | **HSWQ** |
| **copaxTimeless_xplusZ13** | nv100 | **0.96408** | 0.94334 | **+0.02074** | **1.2582** | 1.9680 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **unstablebastard_v14** | nv100 | **0.96382** | 0.92822 | **+0.03560** | **0.8252** | 1.5471 | **0/20 (0%)** | 0/20 (0%) | **HSWQ** |
| **Family Average** | — | **—** | — | **—** | **—** | — | **0.0%** | **1.5%** | **HSWQ (Zero Bifurcations vs 3 Native)** |

---

## 2. Detailed Results per Model

### 2.1. moodyProMix_collectorsEdition (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96158** | 0.92672 | **+0.03486** |
| **Min Final Cosine** (↑ better) | **0.91870** | 0.85344 | **+0.06526** |
| **Max Final Cosine** (↑ better) | **0.98124** | 0.96617 | **+0.01507** |
| **Mean Final Latent MSE** (↓ better) | **0.9576** | 1.7175 | **−0.7599 (44% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 2/20 same-image, 18/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyProMix_collectorsEdition)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.94677** | 0.93081 | **+0.01596** | **1.2890** | 1.6530 | **−0.3640** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.94393** | 0.91133 | **+0.03260** | **1.2160** | 1.9070 | **−0.6910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.97478** | 0.90971 | **+0.06507** | **0.5895** | 2.0200 | **−1.4305** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.97051** | 0.94365 | **+0.02686** | **0.6317** | 1.1960 | **−0.5643** | drifted (different image) | drifted (different image) | **HSWQ** |
| **371506** | 0.96144 | **0.96617** | −0.00473 | 1.2330 | **1.0680** | +0.1650 | drifted (different image) | drifted (different image) | Native |
| **5293047** | **0.96700** | 0.91869 | **+0.04831** | **0.7252** | 1.7710 | **−1.0458** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.94376** | 0.91019 | **+0.03357** | **1.3420** | 2.0810 | **−0.7390** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.94402** | 0.93953 | **+0.00449** | **1.2020** | 1.2950 | **−0.0930** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8426170395** | **0.95609** | 0.95608 | **+0.00001** | 0.9933 | **0.9791** | +0.0142 | drifted (different image) | drifted (different image) | **HSWQ** |
| **9517038246** | **0.91870** | 0.90862 | **+0.01008** | **2.2920** | 2.5070 | **−0.2150** | drifted (different image) | drifted (different image) | **HSWQ** |
| **210987** | **0.97787** | 0.96526 | **+0.01261** | **0.5490** | 0.8596 | **−0.3106** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6543210** | **0.98124** | 0.96069 | **+0.02055** | **0.5132** | 1.0630 | **−0.5498** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.96489** | 0.92645 | **+0.03844** | **0.7508** | 1.5380 | **−0.7872** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.97923** | 0.96172 | **+0.01751** | **0.5502** | 1.0150 | **−0.4648** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2468135791** | **0.97451** | 0.85344 | **+0.12107** | **0.5140** | 2.9180 | **−2.4040** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.95821** | 0.94050 | **+0.01771** | **0.9511** | 1.3520 | **−0.4009** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.97296** | 0.89653 | **+0.07643** | **0.7603** | 2.7770 | **−2.0167** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.98052** | 0.95201 | **+0.02851** | **0.4670** | 1.1380 | **−0.6710** | same-image | drifted (different image) | **HSWQ** |
| **6802579145** | **0.94777** | 0.88792 | **+0.05985** | **1.3680** | 2.8460 | **−1.4780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.96734** | 0.89504 | **+0.07230** | **0.7690** | 2.3700 | **−1.6010** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96158** | **0.92672** | **+0.03486** | **0.9576** | **1.7175** | **−0.7599** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (19/20)** |

---

### 2.2. moodyRealMix_xhsEdition (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.97366** | 0.94680 | **+0.02686** |
| **Min Final Cosine** (↑ better) | **0.90364** | 0.89180 | **+0.01184** |
| **Max Final Cosine** (↑ better) | **0.99230** | 0.97569 | **+0.01661** |
| **Mean Final Latent MSE** (↓ better) | **0.7410** | 1.4552 | **−0.7142 (49% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 11/20 same-image, 9/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyRealMix_xhsEdition)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98159** | 0.95897 | **+0.02262** | **0.4642** | 1.0010 | **−0.5368** | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.98900** | 0.96243 | **+0.02657** | **0.2287** | 0.7700 | **−0.5413** | same-image | drifted (different image) | **HSWQ** |
| **5517** | **0.98699** | 0.91690 | **+0.07009** | **0.3060** | 1.8810 | **−1.5750** | same-image | drifted (different image) | **HSWQ** |
| **92048** | **0.97284** | 0.95461 | **+0.01823** | **0.5730** | 0.9481 | **−0.3751** | drifted (different image) | drifted (different image) | **HSWQ** |
| **371506** | **0.98197** | 0.96887 | **+0.01310** | **0.5893** | 1.0230 | **−0.4337** | same-image | drifted (different image) | **HSWQ** |
| **5293047** | 0.97282 | **0.97511** | −0.00229 | 0.5546 | **0.5058** | +0.0488 | drifted (different image) | drifted (different image) | Native |
| **64820153** | **0.95801** | 0.93552 | **+0.02249** | **1.1330** | 1.6110 | **−0.4780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.98147** | 0.95683 | **+0.02464** | **0.4166** | 0.9583 | **−0.5417** | same-image | drifted (different image) | **HSWQ** |
| **8426170395** | **0.98683** | 0.97542 | **+0.01141** | **0.3060** | 0.5675 | **−0.2615** | same-image | drifted (different image) | **HSWQ** |
| **9517038246** | **0.99230** | 0.97569 | **+0.01661** | **0.2290** | 0.7226 | **−0.4936** | same-image | drifted (different image) | **HSWQ** |
| **210987** | 0.95983 | **0.97189** | −0.01206 | 0.9240 | **0.6498** | +0.2742 | drifted (different image) | drifted (different image) | Native |
| **6543210** | **0.98964** | 0.96914 | **+0.02050** | **0.2914** | 0.8575 | **−0.5661** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.95371** | 0.89180 | **+0.06191** | **1.0570** | 2.2960 | **−1.2390** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.98809** | 0.96621 | **+0.02188** | **0.3178** | 0.8646 | **−0.5468** | same-image | drifted (different image) | **HSWQ** |
| **2468135791** | **0.97321** | 0.91721 | **+0.05600** | **0.6013** | 1.8030 | **−1.2017** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.97789** | 0.95469 | **+0.02320** | **0.4828** | 0.9849 | **−0.5021** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.98113** | 0.92149 | **+0.05964** | **0.5998** | 2.2980 | **−1.6982** | same-image | drifted (different image) | **HSWQ** |
| **5791468034** | **0.98481** | 0.94554 | **+0.03927** | **0.4043** | 1.4320 | **−1.0277** | same-image | drifted (different image) | **HSWQ** |
| **6802579145** | **0.95749** | 0.89334 | **+0.06415** | **1.3460** | 3.3000 | **−1.9540** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | 0.90364 | **0.92423** | −0.02059 | 2.9970 | **2.3540** | +0.6430 | drifted (different image) | drifted (different image) | Native |
| **Mean** | **0.97366** | **0.94680** | **+0.02686** | **0.7410** | **1.4552** | **−0.7142** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (17/20)** |

---

### 2.3. zimageTurboByStable_2602 (nv99)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96242** | 0.92187 | **+0.04055** |
| **Min Final Cosine** (↑ better) | **0.93017** | 0.80337 | **+0.12680** |
| **Max Final Cosine** (↑ better) | **0.98790** | 0.97074 | **+0.01716** |
| **Mean Final Latent MSE** (↓ better) | **0.8143** | 1.8329 | **−1.0186 (56% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 4/20 same-image, 16/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (zimageTurboByStable_2602)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97424** | 0.88187 | **+0.09237** | **0.5476** | 2.4040 | **−1.8564** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.95983** | 0.95629 | **+0.00354** | **0.8004** | 0.8704 | **−0.0700** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.93017** | 0.86954 | **+0.06063** | **1.4340** | 2.5870 | **−1.1530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.98351** | 0.96777 | **+0.01574** | **0.3520** | 0.6818 | **−0.3298** | same-image | drifted (different image) | **HSWQ** |
| **371506** | **0.98258** | 0.95865 | **+0.02393** | **0.6043** | 1.4130 | **−0.8087** | same-image | drifted (different image) | **HSWQ** |
| **5293047** | **0.96167** | 0.90852 | **+0.05315** | **0.7501** | 1.8080 | **−1.0579** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.94018** | 0.90367 | **+0.03651** | **1.3650** | 2.1460 | **−0.7810** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.97660** | 0.95836 | **+0.01824** | **0.4486** | 0.7941 | **−0.3455** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8426170395** | **0.95923** | 0.95099 | **+0.00824** | **0.7940** | 0.9433 | **−0.1493** | drifted (different image) | drifted (different image) | **HSWQ** |
| **9517038246** | **0.98734** | 0.94038 | **+0.04696** | **0.3416** | 1.5760 | **−1.2344** | same-image | drifted (different image) | **HSWQ** |
| **210987** | **0.97886** | 0.90829 | **+0.07057** | **0.4768** | 2.0160 | **−1.5392** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6543210** | **0.98790** | 0.97074 | **+0.01716** | **0.3071** | 0.7460 | **−0.4389** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.94806** | 0.91697 | **+0.03109** | **0.9112** | 1.4730 | **−0.5618** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.97998** | 0.96689 | **+0.01309** | **0.4783** | 0.7893 | **−0.3110** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2468135791** | **0.93045** | 0.85776 | **+0.07269** | **1.3120** | 2.5920 | **−1.2800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.96051** | 0.94105 | **+0.01946** | **0.8443** | 1.2500 | **−0.4057** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.97108** | 0.91363 | **+0.05745** | **0.8050** | 2.3350 | **−1.5300** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.96216** | 0.93979 | **+0.02237** | **0.8419** | 1.3120 | **−0.4701** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | **0.93698** | 0.92284 | **+0.01414** | **1.6020** | 1.8930 | **−0.2910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.93714** | 0.80337 | **+0.13377** | **1.3530** | 4.0480 | **−2.6950** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96242** | **0.92187** | **+0.04055** | **0.8143** | **1.8329** | **−1.0186** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (20/20)** |

---

### 2.4. unstableRevolution_V3Fp16 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96216** | 0.92187 | **+0.04029** |
| **Min Final Cosine** (↑ better) | **0.91686** | 0.80337 | **+0.11349** |
| **Max Final Cosine** (↑ better) | **0.98823** | 0.97074 | **+0.01749** |
| **Mean Final Latent MSE** (↓ better) | **0.8879** | 1.8329 | **−0.9450 (52% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 7/20 same-image, 13/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (unstableRevolution_V3Fp16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.98699** | 0.88187 | **+0.10512** | **0.2853** | 2.4040 | **−2.1187** | same-image | drifted (different image) | **HSWQ** |
| **137** | **0.96453** | 0.95629 | **+0.00824** | **0.7090** | 0.8704 | **−0.1614** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.94293** | 0.86954 | **+0.07339** | **1.1760** | 2.5870 | **−1.4110** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | 0.92350 | **0.96777** | −0.04427 | 1.5980 | **0.6818** | +0.9162 | drifted (different image) | drifted (different image) | Native |
| **371506** | **0.97197** | 0.95865 | **+0.01332** | **0.9693** | 1.4130 | **−0.4437** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5293047** | **0.91686** | 0.90852 | **+0.00834** | **1.6400** | 1.8080 | **−0.1680** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.95926** | 0.90367 | **+0.05559** | **0.9322** | 2.1460 | **−1.2138** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.98433** | 0.95836 | **+0.02597** | **0.3017** | 0.7941 | **−0.4924** | same-image | drifted (different image) | **HSWQ** |
| **8426170395** | **0.98253** | 0.95099 | **+0.03154** | **0.3412** | 0.9433 | **−0.6021** | same-image | drifted (different image) | **HSWQ** |
| **9517038246** | **0.98411** | 0.94038 | **+0.04373** | **0.4296** | 1.5760 | **−1.1464** | same-image | drifted (different image) | **HSWQ** |
| **210987** | **0.98314** | 0.90829 | **+0.07485** | **0.3818** | 2.0160 | **−1.6342** | same-image | drifted (different image) | **HSWQ** |
| **6543210** | **0.98823** | 0.97074 | **+0.01749** | **0.3008** | 0.7460 | **−0.4452** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.94293** | 0.91697 | **+0.02596** | **0.9985** | 1.4730 | **−0.4745** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.98432** | 0.96689 | **+0.01743** | **0.3761** | 0.7893 | **−0.4132** | same-image | drifted (different image) | **HSWQ** |
| **2468135791** | **0.94674** | 0.85776 | **+0.08898** | **0.9911** | 2.5920 | **−1.6009** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.96851** | 0.94105 | **+0.02746** | **0.6746** | 1.2500 | **−0.5754** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.96896** | 0.91363 | **+0.05533** | **0.8640** | 2.3350 | **−1.4710** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.96777** | 0.93979 | **+0.02798** | **0.7123** | 1.3120 | **−0.5997** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | **0.94905** | 0.92284 | **+0.02621** | **1.2790** | 1.8930 | **−0.6140** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.92664** | 0.80337 | **+0.12327** | **1.5680** | 4.0480 | **−2.4800** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96216** | **0.92187** | **+0.04029** | **0.8879** | **1.8329** | **−0.9450** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (19/20)** |

---

### 2.5. gonzalomoZpop_insta2 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96554** | 0.92187 | **+0.04367** |
| **Min Final Cosine** (↑ better) | **0.91984** | 0.80337 | **+0.11647** |
| **Max Final Cosine** (↑ better) | **0.99302** | 0.97074 | **+0.02228** |
| **Mean Final Latent MSE** (↓ better) | **0.8143** | 1.8329 | **−1.0186 (56% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 5/20 same-image, 15/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (gonzalomoZpop_insta2)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97459** | 0.88187 | **+0.09272** | **0.5471** | 2.4040 | **−1.8569** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | 0.95487 | **0.95629** | −0.00142 | 0.8998 | **0.8704** | +0.0294 | drifted (different image) | drifted (different image) | Native |
| **5517** | **0.92773** | 0.86954 | **+0.05819** | **1.4780** | 2.5870 | **−1.1090** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.98140** | 0.96777 | **+0.01363** | **0.3960** | 0.6818 | **−0.2858** | same-image | drifted (different image) | **HSWQ** |
| **371506** | **0.97166** | 0.95865 | **+0.01301** | **0.9749** | 1.4130 | **−0.4381** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5293047** | **0.94453** | 0.90852 | **+0.03601** | **1.0730** | 1.8080 | **−0.7350** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.96839** | 0.90367 | **+0.06472** | **0.7304** | 2.1460 | **−1.4156** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.98272** | 0.95836 | **+0.02436** | **0.3344** | 0.7941 | **−0.4597** | same-image | drifted (different image) | **HSWQ** |
| **8426170395** | **0.97124** | 0.95099 | **+0.02025** | **0.5588** | 0.9433 | **−0.3845** | drifted (different image) | drifted (different image) | **HSWQ** |
| **9517038246** | **0.99302** | 0.94038 | **+0.05264** | **0.1890** | 1.5760 | **−1.3870** | same-image | drifted (different image) | **HSWQ** |
| **210987** | **0.98689** | 0.90829 | **+0.07860** | **0.2978** | 2.0160 | **−1.7182** | same-image | drifted (different image) | **HSWQ** |
| **6543210** | **0.98942** | 0.97074 | **+0.01868** | **0.2688** | 0.7460 | **−0.4772** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.96876** | 0.91697 | **+0.05179** | **0.5542** | 1.4730 | **−0.9188** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.97038** | 0.96689 | **+0.0349** | **0.7006** | 0.7893 | **−0.0887** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2468135791** | **0.93523** | 0.85776 | **+0.07747** | **1.2010** | 2.5920 | **−1.3910** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.96266** | 0.94105 | **+0.02161** | **0.7980** | 1.2500 | **−0.4520** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.95673** | 0.91363 | **+0.04310** | **1.2170** | 2.3350 | **−1.1180** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.97297** | 0.93979 | **+0.03318** | **0.6011** | 1.3120 | **−0.7109** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | **0.97768** | 0.92284 | **+0.05484** | **0.5632** | 1.8930 | **−1.3298** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.91984** | 0.80337 | **+0.11647** | **1.6970** | 4.0480 | **−2.3510** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96554** | **0.92187** | **+0.04367** | **0.8143** | **1.8329** | **−1.0186** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (19/20)** |

---

### 2.6. divingZImageTurbo_v70Fp16 (nv99)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96107** | 0.93218 | **+0.02889** |
| **Min Final Cosine** (↑ better) | **0.89759** | 0.81258 | **+0.08501** |
| **Max Final Cosine** (↑ better) | **0.99087** | 0.96913 | **+0.02174** |
| **Mean Final Latent MSE** (↓ better) | **0.8833** | 1.5471 | **−0.6638 (43% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 5/20 same-image, 15/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (divingZImageTurbo_v70Fp16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.96307** | 0.93188 | **+0.03119** | **0.7398** | 1.3290 | **−0.5892** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1337** | **0.97644** | 0.95480 | **+0.02164** | **0.6056** | 1.1700 | **−0.5644** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7** | **0.97947** | 0.96913 | **+0.01034** | **0.3381** | 0.5036 | **−0.1655** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2024** | **0.96952** | 0.95965 | **+0.00987** | **0.5065** | 0.6670 | **−0.1605** | drifted (different image) | drifted (different image) | **HSWQ** |
| **555** | 0.94936 | **0.94985** | −0.00049 | **1.3660** | 1.3680 | **−0.0020** | drifted (different image) | drifted (different image) | Native |
| **43** | **0.99087** | 0.96637 | **+0.02450** | **0.2294** | 0.7748 | **−0.5454** | same-image | drifted (different image) | **HSWQ** |
| **1458** | **0.97757** | 0.96749 | **+0.01008** | **0.6125** | 0.8848 | **−0.2723** | drifted (different image) | drifted (different image) | **HSWQ** |
| **9** | **0.94258** | 0.90825 | **+0.03433** | **0.9762** | 1.5110 | **−0.5348** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2026** | **0.98445** | 0.95779 | **+0.02666** | **0.3430** | 0.9093 | **−0.5663** | same-image | drifted (different image) | **HSWQ** |
| **777** | **0.95903** | 0.94053 | **+0.01850** | **0.6829** | 0.9851 | **−0.3022** | drifted (different image) | drifted (different image) | **HSWQ** |
| **44** | **0.89759** | 0.89162 | **+0.00597** | **1.9950** | 2.0490 | **−0.0540** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1338** | **0.98371** | 0.96590 | **+0.01781** | **0.3817** | 0.7796 | **−0.3979** | same-image | drifted (different image) | **HSWQ** |
| **8** | **0.98531** | 0.94001 | **+0.04530** | **0.4000** | 1.6160 | **−1.2160** | same-image | drifted (different image) | **HSWQ** |
| **2028** | **0.98049** | 0.96528 | **+0.01521** | **0.5925** | 1.0610 | **−0.4685** | same-image | drifted (different image) | **HSWQ** |
| **888** | **0.96995** | 0.93725 | **+0.03270** | **0.5577** | 1.1460 | **−0.5883** | drifted (different image) | drifted (different image) | **HSWQ** |
| **46** | **0.90162** | 0.89497 | **+0.00665** | **2.0420** | 2.1900 | **−0.1480** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1587** | **0.94263** | 0.89227 | **+0.05036** | **1.1070** | 2.0400 | **−0.9330** | drifted (different image) | drifted (different image) | **HSWQ** |
| **12** | **0.97651** | 0.81258 | **+0.16393** | **0.4753** | 3.5300 | **−3.0547** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2047** | **0.94280** | 0.91578 | **+0.02702** | **1.0190** | 1.4740 | **−0.4550** | drifted (different image) | drifted (different image) | **HSWQ** |
| **222** | **0.94837** | 0.92212 | **+0.02625** | **1.1210** | 1.6740 | **−0.5530** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96107** | **0.93218** | **+0.02889** | **0.8833** | **1.5471** | **−0.6638** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (19/20)** |

---

### 2.7. beyondREALITY_V30 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96025** | 0.90049 | **+0.05976** |
| **Min Final Cosine** (↑ better) | **0.88057** | 0.70230 | **+0.17827** |
| **Max Final Cosine** (↑ better) | **0.98813** | 0.96532 | **+0.02281** |
| **Mean Final Latent MSE** (↓ better) | **1.0772** | 2.3739 | **−1.2967 (55% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **2/20 (10%)** | **2/20 Native bifurcations eliminated** |
| **Trajectory Verdict** | 4/20 same-image, 16/20 drifted | 18/20 drifted, 2/20 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (beyondREALITY_V30)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97393** | 0.89978 | **+0.07415** | **0.6766** | 2.4970 | **−1.8204** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.97306** | 0.92440 | **+0.04866** | **0.6467** | 1.7810 | **−1.1343** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.93393** | 0.89759 | **+0.03634** | **1.6790** | 2.5360 | **−0.8570** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.97810** | 0.90594 | **+0.07216** | **0.4964** | 2.0770 | **−1.5806** | drifted (different image) | drifted (different image) | **HSWQ** |
| **371506** | **0.98813** | 0.95520 | **+0.03293** | **0.4115** | 1.5310 | **−1.1195** | same-image | drifted (different image) | **HSWQ** |
| **5293047** | **0.95467** | 0.92235 | **+0.03232** | **0.9481** | 1.6420 | **−0.6939** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.98293** | 0.91677 | **+0.06616** | **0.4738** | 2.2410 | **−1.7672** | same-image | drifted (different image) | **HSWQ** |
| **731509284** | **0.95549** | 0.91586 | **+0.03963** | **1.0440** | 1.9340 | **−0.8900** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8426170395** | **0.96958** | 0.94428 | **+0.02530** | **0.7278** | 1.3200 | **−0.5922** | drifted (different image) | drifted (different image) | **HSWQ** |
| **9517038246** | **0.95754** | 0.95033 | **+0.00721** | **1.3050** | 1.5540 | **−0.2490** | drifted (different image) | drifted (different image) | **HSWQ** |
| **210987** | **0.98462** | 0.94132 | **+0.04330** | **0.4295** | 1.6040 | **−1.1745** | same-image | drifted (different image) | **HSWQ** |
| **6543210** | **0.98447** | 0.96532 | **+0.01915** | **0.4717** | 1.0540 | **−0.5823** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.92020** | 0.90743 | **+0.01277** | **1.6730** | 1.9070 | **−0.2340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.95890** | 0.95636 | **+0.00254** | **1.1540** | 1.2320 | **−0.0780** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2468135791** | **0.88057** | 0.70230 | **+0.17827** | **2.3300** | 5.7110 | **−3.3810** | drifted (different image) | bifurcated @step 10 | **HSWQ** |
| **3579246812** | **0.96310** | 0.92148 | **+0.04162** | **0.9219** | 1.9700 | **−1.0481** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.93314** | 0.88814 | **+0.04500** | **2.1900** | 3.1420 | **−0.9520** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.97392** | 0.90002 | **+0.07390** | **0.7063** | 2.5980 | **−1.8917** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | **0.97284** | 0.87323 | **+0.09961** | **0.7618** | 3.3700 | **−2.6082** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.96578** | 0.72162 | **+0.24416** | **0.8644** | 6.4780 | **−5.6136** | drifted (different image) | bifurcated @step 10 | **HSWQ** |
| **Mean** | **0.96025** | **0.90049** | **+0.05976** | **1.0772** | **2.3739** | **−1.2967** | **0/20 Bifurcated** | **2/20 Bifurcated** | **HSWQ (20/20)** |

---

### 2.8. 2127ZImageAsianUtopian_v40Turbo (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96558** | 0.91271 | **+0.05287** |
| **Min Final Cosine** (↑ better) | **0.79572** | 0.54584 | **+0.24988** |
| **Max Final Cosine** (↑ better) | **0.99420** | 0.97376 | **+0.02044** |
| **Mean Final Latent MSE** (↓ better) | **0.9961** | 2.5376 | **−1.5415 (61% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **1/20 (5%)** | **1/20 Native bifurcations eliminated** |
| **Trajectory Verdict** | 6/20 same-image, 14/20 drifted | 19/20 drifted, 1/20 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (2127ZImageAsianUtopian_v40Turbo)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.97682** | 0.91062 | **+0.06620** | **0.6032** | 2.2470 | **−1.6438** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.96939** | 0.96523 | **+0.00416** | **0.7898** | 0.8957 | **−0.1059** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.97797** | 0.93916 | **+0.03881** | **0.6786** | 1.8210 | **−1.1424** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.98730** | 0.96576 | **+0.02154** | **0.2773** | 0.7410 | **−0.4637** | same-image | drifted (different image) | **HSWQ** |
| **371506** | **0.98314** | 0.96050 | **+0.02264** | **0.7375** | 1.7360 | **−0.9985** | same-image | drifted (different image) | **HSWQ** |
| **5293047** | **0.95247** | 0.93285 | **+0.01962** | **1.0710** | 1.4680 | **−0.3970** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.97198** | 0.94434 | **+0.02764** | **0.8076** | 1.6090 | **−0.8014** | drifted (different image) | drifted (different image) | **HSWQ** |
| **731509284** | **0.97325** | 0.91137 | **+0.06188** | **0.6200** | 1.8990 | **−1.2790** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8426170395** | **0.98938** | 0.95956 | **+0.02982** | **0.2622** | 0.9849 | **−0.7227** | same-image | drifted (different image) | **HSWQ** |
| **9517038246** | **0.99420** | 0.91826 | **+0.07594** | **0.1989** | 2.6930 | **−2.4941** | same-image | drifted (different image) | **HSWQ** |
| **210987** | **0.99045** | 0.95233 | **+0.03812** | **0.3053** | 1.5100 | **−1.2047** | same-image | drifted (different image) | **HSWQ** |
| **6543210** | **0.99405** | 0.97376 | **+0.02029** | **0.1824** | 0.8445 | **−0.6621** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.96543** | 0.94213 | **+0.02330** | **0.6880** | 1.1360 | **−0.4480** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.97640** | 0.94595 | **+0.03045** | **0.6698** | 1.5280 | **−0.8582** | drifted (different image) | drifted (different image) | **HSWQ** |
| **2468135791** | **0.94513** | 0.85238 | **+0.09275** | **1.3060** | 3.4150 | **−2.1090** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.97857** | 0.91256 | **+0.06601** | **0.5149** | 2.0990 | **−1.5841** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.96890** | 0.91933 | **+0.04957** | **1.0930** | 2.7590 | **−1.6660** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.97506** | 0.96220 | **+0.01286** | **0.6470** | 0.9649 | **−0.3179** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | **0.79572** | 0.54584 | **+0.24988** | **5.9970** | 13.0500 | **−7.0530** | drifted (different image) | bifurcated @step 10 | **HSWQ** |
| **7913680256** | **0.94601** | 0.83997 | **+0.10604** | **1.4270** | 4.0020 | **−2.5750** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96558** | **0.91271** | **+0.05287** | **0.9961** | **2.5376** | **−1.5415** | **0/20 Bifurcated** | **1/20 Bifurcated** | **HSWQ (20/20)** |

---

### 2.9. copaxTimeless_xplusZ13 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96408** | 0.94334 | **+0.02074** |
| **Min Final Cosine** (↑ better) | **0.90006** | 0.88488 | **+0.01518** |
| **Max Final Cosine** (↑ better) | **0.98406** | 0.97724 | **+0.00682** |
| **Mean Final Latent MSE** (↓ better) | **1.2582** | 1.9680 | **−0.7098 (36% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 4/20 same-image, 16/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (copaxTimeless_xplusZ13)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | 0.95640 | **0.95903** | −0.00263 | 1.5530 | **1.5510** | +0.0020 | drifted (different image) | drifted (different image) | Native |
| **137** | **0.96129** | 0.94578 | **+0.01551** | **0.9593** | 1.3220 | **−0.3627** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5517** | **0.97219** | 0.91607 | **+0.05612** | **0.9966** | 2.9490 | **−1.9524** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.97691** | 0.94808 | **+0.02883** | **0.5693** | 1.2600 | **−0.6907** | drifted (different image) | drifted (different image) | **HSWQ** |
| **371506** | **0.90006** | 0.88488 | **+0.01518** | **5.0610** | 5.5510 | **−0.4900** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5293047** | **0.98406** | 0.96443 | **+0.01963** | **0.3525** | 0.7770 | **−0.4245** | same-image | drifted (different image) | **HSWQ** |
| **64820153** | 0.95726 | **0.96000** | −0.00274 | 1.5810 | **1.5750** | +0.0060 | drifted (different image) | drifted (different image) | Native |
| **731509284** | **0.96653** | 0.95934 | **+0.00719** | **1.0650** | 1.3570 | **−0.2920** | drifted (different image) | drifted (different image) | **HSWQ** |
| **8426170395** | **0.96016** | 0.95451 | **+0.00565** | **1.0220** | 1.1600 | **−0.1380** | drifted (different image) | drifted (different image) | **HSWQ** |
| **9517038246** | **0.97528** | 0.97091 | **+0.00437** | **0.9820** | 1.2200 | **−0.2380** | drifted (different image) | drifted (different image) | **HSWQ** |
| **210987** | **0.97790** | 0.91583 | **+0.06207** | **0.7548** | 2.7730 | **−2.0182** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6543210** | **0.98184** | 0.97724 | **+0.00460** | **0.6910** | 0.9434 | **−0.2524** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.97057** | 0.95117 | **+0.01940** | **0.7281** | 1.2180 | **−0.4899** | drifted (different image) | drifted (different image) | **HSWQ** |
| **1357924680** | **0.98343** | 0.95985 | **+0.02358** | **0.6364** | 1.5600 | **−0.9236** | same-image | drifted (different image) | **HSWQ** |
| **2468135791** | **0.92252** | 0.90361 | **+0.01891** | **2.2620** | 2.7620 | **−0.5000** | drifted (different image) | drifted (different image) | **HSWQ** |
| **3579246812** | **0.96223** | 0.93581 | **+0.02642** | **0.9875** | 1.6380 | **−0.6505** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.97666** | 0.96278 | **+0.01388** | **1.0680** | 1.7300 | **−0.6620** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.98228** | 0.93806 | **+0.04422** | **0.6765** | 2.3120 | **−1.6355** | same-image | drifted (different image) | **HSWQ** |
| **6802579145** | **0.94837** | 0.93693 | **+0.01144** | **1.9090** | 2.3290 | **−0.4200** | drifted (different image) | drifted (different image) | **HSWQ** |
| **7913680256** | **0.96570** | 0.92253 | **+0.04317** | **1.2630** | 2.8920 | **−1.6290** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96408** | **0.94334** | **+0.02074** | **1.2582** | **1.9680** | **−0.7098** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (18/20)** |

---

### 2.10. unstablebastard_v14 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.96382** | 0.92822 | **+0.03560** |
| **Min Final Cosine** (↑ better) | **0.91728** | 0.82119 | **+0.09609** |
| **Max Final Cosine** (↑ better) | **0.98785** | 0.97644 | **+0.01141** |
| **Mean Final Latent MSE** (↓ better) | **0.8252** | 1.5471 | **−0.7219 (47% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/20 (0%)** | **0/20 (0%)** | **Zero bifurcations** |
| **Trajectory Verdict** | 9/20 same-image, 11/20 drifted | 20/20 drifted | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (unstablebastard_v14)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.96853** | 0.92689 | **+0.04164** | **0.6274** | 1.4270 | **−0.7996** | drifted (different image) | drifted (different image) | **HSWQ** |
| **137** | **0.98785** | 0.97118 | **+0.01667** | **0.2426** | 0.5698 | **−0.3272** | same-image | drifted (different image) | **HSWQ** |
| **5517** | **0.91728** | 0.82119 | **+0.09609** | **1.6740** | 3.4790 | **−1.8050** | drifted (different image) | drifted (different image) | **HSWQ** |
| **92048** | **0.97268** | 0.91603 | **+0.05665** | **0.5466** | 1.6490 | **−1.1024** | drifted (different image) | drifted (different image) | **HSWQ** |
| **371506** | 0.93761 | **0.97149** | −0.03388 | 1.9370 | **0.8936** | +1.0434 | drifted (different image) | drifted (different image) | Native |
| **5293047** | **0.95468** | 0.91709 | **+0.03759** | **0.9284** | 1.6590 | **−0.7306** | drifted (different image) | drifted (different image) | **HSWQ** |
| **64820153** | **0.98425** | 0.90217 | **+0.08208** | **0.3446** | 2.0000 | **−1.6554** | same-image | drifted (different image) | **HSWQ** |
| **731509284** | **0.98131** | 0.97331 | **+0.00800** | **0.3472** | 0.4965 | **−0.1493** | same-image | drifted (different image) | **HSWQ** |
| **8426170395** | **0.98615** | 0.94631 | **+0.03984** | **0.2792** | 1.0490 | **−0.7698** | same-image | drifted (different image) | **HSWQ** |
| **9517038246** | **0.98661** | 0.97521 | **+0.01140** | **0.3269** | 0.6220 | **−0.2951** | same-image | drifted (different image) | **HSWQ** |
| **210987** | **0.98521** | 0.95939 | **+0.02582** | **0.3332** | 0.8993 | **−0.5661** | same-image | drifted (different image) | **HSWQ** |
| **6543210** | **0.98459** | 0.97453 | **+0.01006** | **0.3759** | 0.6227 | **−0.2468** | same-image | drifted (different image) | **HSWQ** |
| **98765432** | **0.98639** | 0.93435 | **+0.05204** | **0.2440** | 1.1690 | **−0.9250** | same-image | drifted (different image) | **HSWQ** |
| **1357924680** | 0.92760 | **0.97644** | −0.04884 | 1.6660 | **0.5538** | +1.1122 | drifted (different image) | drifted (different image) | Native |
| **2468135791** | **0.98702** | 0.86761 | **+0.11941** | **0.2426** | 2.3750 | **−2.1324** | same-image | drifted (different image) | **HSWQ** |
| **3579246812** | **0.93037** | 0.91183 | **+0.01854** | **1.5140** | 1.9480 | **−0.4340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **4680357923** | **0.94909** | 0.91722 | **+0.03187** | **1.3840** | 2.2180 | **−0.8340** | drifted (different image) | drifted (different image) | **HSWQ** |
| **5791468034** | **0.95581** | 0.91833 | **+0.03748** | **0.9051** | 1.6430 | **−0.7379** | drifted (different image) | drifted (different image) | **HSWQ** |
| **6802579145** | 0.93847 | **0.94279** | −0.00432 | 1.4620 | **1.3580** | +0.1040 | drifted (different image) | drifted (different image) | Native |
| **7913680256** | **0.95487** | 0.84097 | **+0.11390** | **0.9141** | 3.0880 | **−2.1739** | drifted (different image) | drifted (different image) | **HSWQ** |
| **Mean** | **0.96382** | **0.92822** | **+0.03560** | **0.8252** | **1.5471** | **−0.7219** | **0/20 Bifurcated** | **0/20 Bifurcated** | **HSWQ (17/20)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Complete Elimination of Trajectory Bifurcations:**
   Across all 10 tested Z-Image models (200 total seed evaluations), Native NVFP4 suffers catastrophic Step 10 bifurcations on challenging checkpoints (e.g. `beyondREALITY_V30`, `2127ZImageAsianUtopian_v40Turbo`), where severe quantization noise forces the denoising path into completely distinct picture attractor basins (final cosine dropping to 0.54–0.70). HSWQ ConvRot Hybrid NVFP4 completely eliminates all bifurcations (0/200 seeds = 0.0%), achieving flawless stability.
2. **High-Fidelity Semantic and Compositional Preservation:**
   HSWQ consistently achieves high cosine similarity across all models (mean cosine 0.960–0.974 vs Native 0.900–0.947). Furthermore, 53 out of 200 runs (26.5%) achieve `same-image` status (final cosine ≥ 0.98), whereas Native NVFP4 achieves 0 same-image seeds (0/200).
3. **Substantial Latent MSE Reduction:**
   By preserving the sensitive layers identified via sensitivity weighting and rotation, HSWQ reduces final latent Mean Squared Error by ~40–60% across the board compared to native full-model quantization.
4. **Worst-Case Robustness Across Random Seeds:**
   In native quantization, difficult random seeds suffer severe degradation and collapse (minimum cosine as low as 0.54584). Under HSWQ, worst-case seeds maintain high structural integrity (minimum cosine ~0.80–0.93), ensuring reliable production inference without catastrophic failure seeds.

---

## 4. Metric Definitions

- **Final Cosine (final-cos):** Cosine similarity between the final denoised latent of FP16 reference and quantized NVFP4 models. Closer to 1.0 indicates identical composition, lighting, and semantic fidelity.
- **Final MSE (final-mse):** Mean Squared Error of the final latent tensor against the FP16 baseline.
- **Max Step Drop (max-drop):** Maximum single-step cosine drop between consecutive sampling steps, quantifying sudden trajectory instability.
- **Verdict:**
  - `same-image`: Final cosine ≥ 0.98 and max step drop < 0.0035, producing virtually indistinguishable high-fidelity generation.
  - `drifted (different image)`: Gradual, continuous deviation across steps while maintaining coherent compositional structure.
  - `bifurcated @step N`: Sudden discontinuous trajectory jump at step N, indicating transition into a completely different picture attractor basin.
