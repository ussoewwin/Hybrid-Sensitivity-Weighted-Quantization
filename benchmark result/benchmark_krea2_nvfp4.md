# Krea2 Hybrid NVFP4 Benchmark Test Results

Deterministic per-step latent trajectory divergence benchmark comparing **FP16 reference** vs **HSWQ Hybrid NVFP4 (nv100)** vs **Native NVFP4** on the Krea2 architecture (`moodyKrea2Mix_v70BF16.safetensors`).

**Source:** `benchmark result/score_krea2_nvfp4.txt`  
**Evaluation Script:** `benchmark/krea2_traj_compare.py` (10-seed deterministic trajectory analysis)

---

## 1. Summary Comparison (HSWQ Hybrid NVFP4 vs Native NVFP4)

| Metric / Property | HSWQ Hybrid NVFP4 (nv100) | Native NVFP4 (Full Model) | Advantage |
| :--- | :--- | :--- | :--- |
| **Target Model** | `moodyKrea2Mix_v70BF16` (nv100) | `moodyKrea2Mix_v70BF16` | — |
| **Mean Final Cosine Similarity** (↑ better) | **0.92539** | 0.85158 | **+0.07381** (Significantly higher fidelity) |
| **Min Final Cosine Similarity** (↑ better) | **0.85636** | 0.65046 | **+0.20590** (Robust worst-case stability) |
| **Max Final Cosine Similarity** (↑ better) | **0.96673** | 0.94404 | **+0.02269** |
| **Mean Final Latent MSE** (↓ better) | **0.2016** | 0.4493 | **-0.2477** (55% lower error) |
| **Bifurcated Seeds Rate** (↓ better) | **1/10 (10%)** | **6/10 (60%)** | **6x less catastrophic trajectory divergence** |
| **Trajectory Verdict** | 9/10 drifted, 1/10 bifurcated | 4/10 drifted, 6/10 bifurcated | **HSWQ preserves trajectory structure** |

---

## 2. Multi-Seed Detailed Results (10 Seeds)

### HSWQ Hybrid NVFP4 (`moodyKrea2Mix_v70BF16 nv100`)

| Seed | Final Cosine (↑ better) | Final MSE (↓ better) | Max Step Drop | Trajectory Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **42** | 0.93916 | 0.1649 | 0.0256 | drifted (different image) |
| **1337** | 0.94059 | 0.1608 | 0.0252 | drifted (different image) |
| **7** | 0.93514 | 0.1798 | 0.0274 | drifted (different image) |
| **2024** | 0.90219 | 0.2697 | 0.0424 | drifted (different image) |
| **555** | 0.91541 | 0.2173 | 0.0347 | drifted (different image) |
| **123456789** | 0.85636 | 0.3735 | 0.0612 | bifurcated @step 11 |
| **505430789** | 0.94238 | 0.1469 | 0.0241 | drifted (different image) |
| **789654321** | 0.94777 | 0.1441 | 0.0222 | drifted (different image) |
| **430** | **0.96673** | **0.0866** | **0.0140** | drifted (different image) |
| **44285** | 0.90817 | 0.2355 | 0.0357 | drifted (different image) |
| **Mean** | **0.92539** | **0.2016** | **0.0313** | **90% stable drift / 10% bifurcated** |

---

### Native NVFP4 Baseline (`moodyKrea2Mix_v70BF16 Native NVFP4`)

| Seed | Final Cosine (↑ better) | Final MSE (↓ better) | Max Step Drop | Trajectory Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **42** | 0.72759 | 0.7588 | 0.1094 | bifurcated @step 11 |
| **1337** | 0.90078 | 0.2752 | 0.0421 | drifted (different image) |
| **7** | 0.94404 | 0.1567 | 0.0235 | drifted (different image) |
| **2024** | 0.87697 | 0.3408 | 0.0538 | bifurcated @step 11 |
| **555** | 0.87665 | 0.3213 | 0.0519 | bifurcated @step 11 |
| **123456789** | 0.82164 | 0.4815 | 0.0738 | bifurcated @step 11 |
| **505430789** | 0.91140 | 0.2316 | 0.0372 | drifted (different image) |
| **789654321** | 0.92871 | 0.1978 | 0.0300 | drifted (different image) |
| **430** | 0.87755 | 0.3258 | 0.0514 | bifurcated @step 11 |
| **44285** | 0.65046 | 0.9042 | 0.1263 | bifurcated @step 11 |
| **Mean** | **0.85158** | **0.4493** | **0.0650** | **40% drift / 60% bifurcated** |

---

## 3. Side-by-Side Comparison per Seed

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
| **Summary** | **0.92539** | **0.85158** | **+0.07381** | **0.2016** | **0.4493** | **-0.2477** | **1/10 Bifurcated** | **6/10 Bifurcated** | **HSWQ (9/10)** |

---

## 4. Key Findings and Trajectory Analysis

1. **Catastrophic Bifurcation Suppression:**
   - **Native NVFP4** suffers from severe trajectory bifurcation at Step 11 on **60% of test seeds (6/10)**, where accumulation of quant noise causes the denoising trajectory to jump into an entirely different attractor basin (cosine dropping as low as 0.650).
   - **HSWQ Hybrid NVFP4 (nv100)** shelters sensitive layers in INT8 using the reverse method, dropping the bifurcation rate down to **10% (1/10)**.
2. **Superior Mean Fidelity:**
   - HSWQ improves mean cosine similarity from **0.85158 to 0.92539 (+0.07381)** and reduces mean latent MSE by **55% (0.4493 → 0.2016)** across 10 deterministic seeds.
3. **Worst-Case Resilience:**
   - The worst-case seed under Native NVFP4 dropped to **0.65046** (seed 44285), whereas HSWQ's worst-case seed maintained **0.85636**, completely avoiding collapsed outputs.

---

## 5. Metric Definitions

- **Final Cosine (`final-cos`):** Cosine similarity between the final denoised latent of FP16 and NVFP4 models. Closer to 1.0 indicates identical composition and semantics.
- **Final MSE (`final-mse`):** Mean Squared Error of the final latent tensor.
- **Max Step Drop (`max-drop`):** Maximum single-step cosine drop between consecutive sampling steps, measuring sudden trajectory instability.
- **Verdict:**
  - `drifted (different image)`: Gradual, continuous deviation across steps while staying in the same semantic path.
  - `bifurcated @step N`: Sudden discontinuous trajectory jump at step N, indicating transition into a completely different picture attractor basin.
