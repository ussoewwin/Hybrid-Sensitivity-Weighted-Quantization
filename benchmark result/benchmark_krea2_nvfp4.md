# Krea2 Hybrid NVFP4 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ Hybrid NVFP4** vs **Native NVFP4** on the Krea2 architecture family.

**Source:** `benchmark result/score_krea2_nvfp4.txt`  
**Evaluation Script:** `benchmark/krea2_traj_compare.py` (10-seed deterministic trajectory analysis)

---

## 1. Summary Comparison (HSWQ Hybrid NVFP4 vs Native NVFP4)

### Cross-Model Overview

| Model | Setup | HSWQ Mean Cosine (↑) | Native Mean Cosine (↑) | Δ Cosine | HSWQ Mean MSE (↓) | Native Mean MSE (↓) | HSWQ Bifurcated (↓) | Native Bifurcated (↓) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **moodyKrea2Mix_v70BF16** | nv100 | **0.92539** | 0.85158 | **+0.07381** | **0.2016** | 0.4493 | **1/10 (10%)** | 6/10 (60%) | **HSWQ** |
| **moodyCutieMixKrea2_v40** | nv100 | **0.91640** | 0.85489 | **+0.06151** | **0.2310** | 0.4127 | **2/10 (20%)** | 7/10 (70%) | **HSWQ** |
| **Family Average** | — | **0.92090** | 0.85324 | **+0.06766** | **0.2163** | 0.4310 | **15.0%** | **65.0%** | **HSWQ (4.3x less bifurcation)** |

---

## 2. Detailed Results per Model

### 2.1. moodyKrea2Mix_v70BF16 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 (nv100) | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.92539** | 0.85158 | **+0.07381** |
| **Min Final Cosine** (↑ better) | **0.85636** | 0.65046 | **+0.20590** |
| **Max Final Cosine** (↑ better) | **0.96673** | 0.94404 | **+0.02269** |
| **Mean Final Latent MSE** (↓ better) | **0.2016** | 0.4493 | **-0.2477 (55% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **1/10 (10%)** | **6/10 (60%)** | **6x less trajectory divergence** |
| **Trajectory Verdict** | 9/10 drifted, 1/10 bifurcated | 4/10 drifted, 6/10 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyKrea2Mix_v70)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.93916** | 0.72759 | **+0.21157** | **0.1649** | 0.7588 | **-0.5939** | drifted | bifurcated | **HSWQ** |
| **1337** | **0.94059** | 0.90078 | **+0.03981** | **0.1608** | 0.2752 | **-0.1144** | drifted | drifted | **HSWQ** |
| **7** | 0.93514 | **0.94404** | -0.00890 | 0.1798 | **0.1567** | +0.0231 | drifted | drifted | Native |
| **2024** | **0.90219** | 0.87697 | **+0.02522** | **0.2697** | 0.3408 | **-0.0711** | drifted | bifurcated | **HSWQ** |
| **555** | **0.91541** | 0.87665 | **+0.03876** | **0.2173** | 0.3213 | **-0.1040** | drifted | bifurcated | **HSWQ** |
| **123456789** | **0.85636** | 0.82164 | **+0.03472** | **0.3735** | 0.4815 | **-0.1080** | bifurcated | bifurcated | **HSWQ** |
| **505430789** | **0.94238** | 0.91140 | **+0.03098** | **0.1469** | 0.2316 | **-0.0847** | drifted | drifted | **HSWQ** |
| **789654321** | **0.94777** | 0.92871 | **+0.01906** | **0.1441** | 0.1978 | **-0.0537** | drifted | drifted | **HSWQ** |
| **430** | **0.96673** | 0.87755 | **+0.08918** | **0.0866** | 0.3258 | **-0.2392** | drifted | bifurcated | **HSWQ** |
| **44285** | **0.90817** | 0.65046 | **+0.25771** | **0.2355** | 0.9042 | **-0.6687** | drifted | bifurcated | **HSWQ** |
| **Mean** | **0.92539** | **0.85158** | **+0.07381** | **0.2016** | **0.4493** | **-0.2477** | **1/10 Bifurcated** | **6/10 Bifurcated** | **HSWQ (9/10)** |

---

### 2.2. moodyCutieMixKrea2_v40 (nv100)

#### Metric Overview
| Metric / Property | HSWQ Hybrid NVFP4 | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Final Cosine** (↑ better) | **0.91640** | 0.85489 | **+0.06151** |
| **Min Final Cosine** (↑ better) | **0.85790** | 0.72642 | **+0.13148** |
| **Max Final Cosine** (↑ better) | **0.95765** | 0.93485 | **+0.02280** |
| **Mean Final Latent MSE** (↓ better) | **0.2310** | 0.4127 | **-0.1817 (44% error reduction)** |
| **Bifurcated Seeds Rate** (↓ better) | **2/10 (20%)** | **7/10 (70%)** | **3.5x less trajectory divergence** |
| **Trajectory Verdict** | 8/10 drifted, 2/10 bifurcated | 3/10 drifted, 7/10 bifurcated | **HSWQ preserves trajectory structure** |

#### Side-by-Side per Seed (moodyCutieMixKrea2_v40)
| Seed | HSWQ Cosine | Native Cosine | Δ Cosine (↑ better) | HSWQ MSE | Native MSE | Δ MSE (↓ better) | HSWQ Verdict | Native Verdict | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | **0.95765** | 0.81156 | **+0.14609** | **0.1159** | 0.5217 | **-0.4058** | drifted | bifurcated | **HSWQ** |
| **1337** | **0.92784** | 0.90252 | **+0.02532** | **0.2033** | 0.2736 | **-0.0703** | drifted | drifted | **HSWQ** |
| **7** | **0.94068** | 0.92614 | **+0.01454** | **0.1667** | 0.2093 | **-0.0426** | drifted | drifted | **HSWQ** |
| **2024** | **0.91390** | 0.87019 | **+0.04371** | **0.2457** | 0.3646 | **-0.1189** | drifted | bifurcated | **HSWQ** |
| **555** | **0.85790** | 0.84002 | **+0.01788** | **0.3931** | 0.4314 | **-0.0383** | bifurcated | bifurcated | **HSWQ** |
| **123456789** | **0.88832** | 0.72642 | **+0.16190** | **0.3078** | 0.7326 | **-0.4248** | drifted | bifurcated | **HSWQ** |
| **505430789** | 0.86966 | **0.93485** | -0.06519 | 0.3640 | **0.1811** | +0.1829 | bifurcated | drifted | Native |
| **789654321** | **0.92974** | 0.86515 | **+0.06459** | **0.1694** | 0.3287 | **-0.1593** | drifted | bifurcated | **HSWQ** |
| **430** | **0.93182** | 0.82793 | **+0.10389** | **0.1951** | 0.4917 | **-0.2966** | drifted | bifurcated | **HSWQ** |
| **44285** | **0.94644** | 0.84406 | **+0.10238** | **0.1292** | 0.3925 | **-0.2633** | drifted | bifurcated | **HSWQ** |
| **Mean** | **0.91640** | **0.85489** | **+0.06151** | **0.2310** | **0.4127** | **-0.1817** | **2/10 Bifurcated** | **7/10 Bifurcated** | **HSWQ (9/10)** |

---

## 3. Key Findings and Trajectory Analysis

1. **Massive Reduction in Catastrophic Trajectory Bifurcations:**
   Across both tested Krea2 models, Native NVFP4 suffers catastrophic Step 11 bifurcations on **65% (13/20)** of all runs, where trajectory drift forces the model into completely distinct attractor basins. HSWQ Hybrid NVFP4 drops the overall bifurcation rate to **15% (3/20)**, achieving a **4.3x stability improvement**.
2. **Consistent Superiority Across Realistic and Stylized Checkpoints:**
   Both realistic/semi-realistic (`moodyKrea2Mix_v70`) and anime/stylized (`moodyCutieMixKrea2_v40`) checkpoints exhibit consistent cosine improvements (+0.06 to +0.07) and substantial MSE reductions (44%–55%).
3. **Worst-Case Robustness:**
   In native quantization, catastrophic seeds drop as low as 0.650 (`v70`) and 0.726 (`v40`). Under HSWQ, worst-case seeds never drop below **0.856**, preserving coherent output quality across random seeds.

---

## 4. Metric Definitions

- **Final Cosine (`final-cos`):** Cosine similarity between the final denoised latent of FP16 and NVFP4 models. Closer to 1.0 indicates identical composition and semantics.
- **Final MSE (`final-mse`):** Mean Squared Error of the final latent tensor.
- **Max Step Drop (`max-drop`):** Maximum single-step cosine drop between consecutive sampling steps, measuring sudden trajectory instability.
- **Verdict:**
  - `drifted (different image)`: Gradual, continuous deviation across steps while staying in the same semantic path.
  - `bifurcated @step N`: Sudden discontinuous trajectory jump at step N, indicating transition into a completely different picture attractor basin.
