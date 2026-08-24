# Krea2 Hybrid NVFP4 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ Hybrid NVFP4** vs **Native NVFP4** on the Krea2 architecture family.

**Source:** enchmark result/score_krea2_nvfp4.txt  
**Evaluation Script:** enchmark/krea2_traj_compare.py (10-seed deterministic trajectory analysis)

---

## 1. Summary Comparison (HSWQ Hybrid NVFP4 vs Native NVFP4)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **moodyKrea2Mix_v70BF16** | nv103 | **0.92592** | 0.89619 | **+0.02973** | **0.1974** | 0.2787 | **1/10 (10%)** | 3/10 (30%) | **HSWQ** |
| **moodyCutieMixKrea2_v30** | nv108 | **0.92238** | 0.85489 | **+0.06749** | **0.2061** | 0.3927 | **0/10 (0%)** | 7/10 (70%) | **HSWQ** |
| **Family Average** | — | **—** | — | **—** | **—** | — | **5.0%** | **50.0%** | **HSWQ (10.0x less bifurcation)** |

---

## 2. Detailed Results per Model

### 2.1. moodyKrea2Mix_v70BF16 (nv103)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.92592** | 0.89619 | **+0.02973** |
| **Min Final Cosine** (↑ better) | **0.88216** | 0.78284 | **+0.09932** |
| **Max Final Cosine** (↑ better) | **0.96298** | 0.93534 | **+0.02764** |
| **Mean Final Latent MSE** (↓ better) | **0.1974** | 0.2787 | **−0.0812 (29% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **1/10 (10%)** | **3/10 (30%)** | **3.0x less trajectory divergence** |
| **Trajectory Verdict** | 9/10 drifted, 1/10 bifurcated | 7/10 drifted, 3/10 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyKrea2Mix_v70BF16)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.92440** | 0.92187 | **+0.00253** | **0.2056** | 0.2168 | **−0.0112** | drifted | drifted | **HSWQ** |
| **1337** | **0.94750** | 0.85590 | **+0.09160** | **0.1422** | 0.3937 | **−0.2515** | drifted | bifurcated | **HSWQ** |
| **7** | 0.91403 | **0.92211** | −0.00808 | 0.2384 | **0.2169** | +0.0215 | drifted | drifted | Native |
| **2024** | **0.88216** | 0.87861 | **+0.00355** | **0.3261** | 0.3319 | **−0.0058** | bifurcated | bifurcated | **HSWQ** |
| **555** | 0.91497 | **0.91696** | −0.00199 | 0.2183 | **0.2134** | +0.0049 | drifted | drifted | Native |
| **123456789** | **0.90219** | 0.78284 | **+0.11935** | **0.2547** | 0.5744 | **−0.3197** | drifted | bifurcated | **HSWQ** |
| **505430789** | **0.94957** | 0.92321 | **+0.02636** | **0.1283** | 0.1989 | **−0.0706** | drifted | drifted | **HSWQ** |
| **789654321** | **0.94840** | 0.93534 | **+0.01306** | **0.1427** | 0.1820 | **−0.0393** | drifted | drifted | **HSWQ** |
| **430** | **0.96298** | 0.92710 | **+0.03588** | **0.0968** | 0.1951 | **−0.0983** | drifted | drifted | **HSWQ** |
| **44285** | **0.91304** | 0.89795 | **+0.01509** | **0.2214** | 0.2637 | **−0.0423** | drifted | drifted | **HSWQ** |
| **Mean** | **0.92592** | **0.89619** | **+0.02973** | **0.1974** | **0.2787** | **−0.0812** | **1/10 Bifurcated** | **3/10 Bifurcated** | **HSWQ (8/10)** |

---

### 2.2. moodyCutieMixKrea2_v30 (nv108)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.92238** | 0.85489 | **+0.06749** |
| **Min Final Cosine** (↑ better) | **0.88249** | 0.72642 | **+0.15607** |
| **Max Final Cosine** (↑ better) | **0.95085** | 0.93485 | **+0.01600** |
| **Mean Final Latent MSE** (↓ better) | **0.2061** | 0.3927 | **−0.1866 (48% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **0/10 (0%)** | **7/10 (70%)** | **7.0x less trajectory divergence** |
| **Trajectory Verdict** | 10/10 drifted, 0/10 bifurcated | 3/10 drifted, 7/10 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyCutieMixKrea2_v30)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.92809** | 0.81156 | **+0.11653** | **0.1916** | 0.5217 | **−0.3301** | drifted | bifurcated | **HSWQ** |
| **1337** | **0.90469** | 0.90252 | **+0.00217** | **0.2650** | 0.2736 | **−0.0086** | drifted | drifted | **HSWQ** |
| **7** | **0.95082** | 0.92614 | **+0.02468** | **0.1357** | 0.2093 | **−0.0736** | drifted | drifted | **HSWQ** |
| **2024** | **0.92340** | 0.87019 | **+0.05321** | **0.2129** | 0.3646 | **−0.1517** | drifted | bifurcated | **HSWQ** |
| **555** | **0.89569** | 0.84002 | **+0.05567** | **0.2854** | 0.4314 | **−0.1460** | drifted | bifurcated | **HSWQ** |
| **123456789** | **0.93659** | 0.72642 | **+0.21017** | **0.1703** | 0.7326 | **−0.5623** | drifted | bifurcated | **HSWQ** |
| **505430789** | 0.92071 | **0.93485** | −0.01414 | 0.2169 | **0.1811** | +0.0358 | drifted | drifted | Native |
| **789654321** | **0.93043** | 0.86515 | **+0.06528** | **0.1666** | 0.3287 | **−0.1621** | drifted | bifurcated | **HSWQ** |
| **430** | **0.95085** | 0.82793 | **+0.12292** | **0.1379** | 0.4917 | **−0.3538** | drifted | bifurcated | **HSWQ** |
| **44285** | **0.88249** | 0.84406 | **+0.03843** | **0.2790** | 0.3925 | **−0.1135** | drifted | bifurcated | **HSWQ** |
| **Mean** | **0.92238** | **0.85489** | **+0.06749** | **0.2061** | **0.3927** | **−0.1866** | **0/10 Bifurcated** | **7/10 Bifurcated** | **HSWQ (9/10)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Massive Reduction in Catastrophic Trajectory Bifurcations:**
   Across both tested Krea2 models, Native NVFP4 suffers catastrophic Step 11 bifurcations on a high percentage of runs, where trajectory drift forces the model into completely distinct attractor basins. HSWQ Hybrid NVFP4 drops the overall bifurcation rate significantly, achieving massive stability improvement.
2. **Consistent Superiority Across Realistic and Stylized Checkpoints:**
   Both realistic/semi-realistic and anime/stylized checkpoints exhibit consistent cosine improvements and substantial MSE reductions.
3. **Worst-Case Robustness:**
   In native quantization, catastrophic seeds drop very low. Under HSWQ, worst-case seeds are preserved, maintaining coherent output quality across random seeds.

---

## 4. Metric Definitions

- **Final Cosine (inal-cos):** Cosine similarity between the final denoised latent of FP16 and NVFP4 models. Closer to 1.0 indicates identical composition and semantics.
- **Final MSE (inal-mse):** Mean Squared Error of the final latent tensor.
- **Max Step Drop (max-drop):** Maximum single-step cosine drop between consecutive sampling steps, measuring sudden trajectory instability.
- **Verdict:**
  - drifted (different image): Gradual, continuous deviation across steps while staying in the same semantic path.
  - ifurcated @step N: Sudden discontinuous trajectory jump at step N, indicating transition into a completely different picture attractor basin.
