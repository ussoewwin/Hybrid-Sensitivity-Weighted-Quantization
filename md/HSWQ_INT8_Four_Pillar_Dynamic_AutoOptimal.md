# HSWQ INT8 — Four-Pillar Dynamic Per-Model Auto-Optimal Logic

**Document version:** 1.0  
**Date:** 2026-07-18  
**Scope:** How **four mandatory surfaces** jointly produce **checkpoint-specific** INT8 settings and the FP16 protection set.  
**Primary code:**

| Role | Path |
|---|---|
| SDXL entry | `quantize_sdxl_hswq_v3.0.py` |
| Z Image entry | `quantize_zi_int8_hswq_v1.0.py` |
| Analyze / tunables / ranking math | `analyze/analyze_sdxl_distribution.py` |
| Histogram V4 + Full-SVD×RMS | `histogram/weighted_histogram_mse_v4_int8.py` |

**Companion:** [HSWQ INT8 Technical Overview](HSWQ_INT8_SDXL_Technical_Guide.md)

This document is **not** a keep-ratio recipe and **not** a fixed cross-model priority table. Every numeric knob below is derived from **this checkpoint’s** analyze JSON and **this run’s** DualMonitor calibration, then consumed by V4+SVD scoring under a **hard FP16 budget ceiling**.

---

## 0. The four pillars (non-negotiable)

| # | Pillar | Artifact | What it contributes |
|---|---|---|---|
| **A** | **Analyze JSON** | `{model}_distribution_profile.json` (`layers` + `summary`) | Weight-space character: kurtosis / outlier / abs_max / MAD / shape flags → Hard VETO fences, severity, score weights, **`alpha_auto`**, gray-zone / BC knobs |
| **B** | **DualMonitor** | Live hooks during calibration | **Sensitivity** `Var(y)` for ranking + pool; **Importance** `mean(|x|)` channel map that **multiplies** V4 hybrid; activation moments for bias correction |
| **C** | **Histogram V4 calibration** | `HSWQWeightedHistogramOptimizerV4` + `INT8Quantizer` | Weighted histogram at **absmax**; outputs **`estimated_mse`** used only to **rank FP16 keep** (pack amax stays absmax) |
| **D** | **SVD** | `torch.linalg.svd` inside `compute_hybrid_leverage_scores` | Full-SVD leverage map mixed with RMS magnitude: **`α·SVD + β·RMS`** (`α = alpha_auto > 0`). DualMonitor Imp **multiplies**; it **never replaces** SVD |

**Forbidden hand-waves (already treated as rebellion in code):**

- Skip DualMonitor and fall back to profile_score-only FP16 ranking.  
- `use_svd_leverage=False` on the INT8 measure path.  
- `alpha_auto == 0` / `alpha == 0` (“SVD computed then cut”).  
- Replace SVD hybrid with Importance alone.  
- Mag / Conv tax **outside** the FP16 budget (Linear+Conv share **one** queue).  
- Non-zero `keep_ratio` (INT8 contract: **r0**).  
- A unified fixed priority formula shared across all models.  
- Pool gate **`severity ≥ 1`** (drops continuous `0 < sev < 1` — thinking-stop).  
- Combinator **`form=uniform` → `priority=1.0`** for every layer (size-only fill — thinking-stop).

---

## 1. End-to-end causal chain

```text
[1] Analyze JSON (weight stats per Diffusers module name)
        │
        ├─► derive_int8_autonomous_tunables / resolve_veto_tunables
        │         Hard VETO fences, score_*_weight, mse_release_*,
        │         bias_correction_top_ratio, gray-zone, alpha_auto, …
        │
        └─► Hard VETO ∪ structural / key-pattern / MAD / supplemental VETO

[2] Calibration prompts → DualMonitor hooks
        │
        ├─► Sensitivity per layer  → pool + ranking axis
        ├─► Importance 1D         → × V4 hybrid (not a substitute)
        └─► re-resolve tunables with DualMonitor present

[3] V4 measure @ absmax (every target Linear/Conv; no keep_ratio pre-cut)
        │
        └─► hybrid = α·SVD_leverage + β·RMS_magnitude
                  × Imp (if present)
                  → weighted histogram → estimated_mse → mse_cache

[4] _apply_fp16_budget_cap (hard ceiling: SDXL 300 MiB / ZI 700 MiB)
        │
        ├─► pool = keep ∪ Hard VETO ∪ all analyze character layers ∪ (dm_sens>0)
        ├─► per layer: (dm_sens, estimated_mse, severity, extra_bytes)
        ├─► apply_fp16_infinite_ranking_branches  (THIS-model continuous knobs)
        ├─► derive_priority_combinator            (THIS IQR × VETO alignment;
        │         flat → equal_weight_geometric, never priority≡1)
        ├─► int8_fp16_budget_priority             (weighted geometric mix)
        ├─► apply_fp16_infinite_priority_branches
        └─► sort (-priority, +extra_bytes) → greedy fill until budget
                  winners = FP16 pack; demoted = INT8 (no Mag-forced FP16)
```

Pack policy remains orthogonal: **every packed INT8 layer uses absmax** (`search_low = 1.0`). V4 does **not** choose pack scale; it chooses **who stays FP16** under the ceiling.

---

## 2. Pillar A — Analyze JSON → per-model tunables

### 2.1 Load / generate

- Default path: `{input_basename}_distribution_profile.json` next to the quantize script root.  
- If missing (or `--profile` absent with auto path): run `analyze/analyze_sdxl_distribution.py --input … --output …` (**no skip**).  
- Remap Comfy / Safetensors keys → Diffusers module names via the load map before any veto / ranking.

### 2.2 Autonomous tunables (`derive_int8_autonomous_tunables`)

From **this** profile’s layer vectors (kurtosis / outlier / abs_max / MAD):

| Derived family | Mechanism (THIS profile only) |
|---|---|
| Hard VETO fences | `max(base Tukey-style fence, P99)` per axis |
| `score_k/o/m_weight` | IQR / scale, renormalized (flat axes shrink) |
| MSE release / gray-zone gates | THIS P75 (and related) on outlier / magnitude |
| `bias_correction_top_ratio` | Autonomous scope (refined again after DualMonitor) |
| **`alpha_auto`** | `_alpha_auto_from_this_character` — multi-axis mix; floor so non-degenerate profiles keep SVD in the mix |
| `fp16_budget_mb` | Must equal owner hard ceiling (**300** SDXL / **700** ZI); never redefined by analysis |

`alpha_auto` is the **SVD mix weight** for Full-SVD×RMS. Code refuses `alpha_auto ≤ 0` when a non-empty THIS profile is present (SVD-cut rebellion).

### 2.3 Severity for ranking

`int8_fp16_budget_analyze_severity(...)` maps layer stats + Hard-VETO flag → continuous / discrete severity used as one of the three ranking axes. Hard VETO layers remain labeled VETO even if they win budget seats.

---

## 3. Pillar B — DualMonitor (calibration, not keep_ratio)

### 3.1 What is measured

During calibration forward hooks:

| Signal | Definition | Consumer |
|---|---|---|
| Sensitivity | Output variance / DualMonitor `get_sensitivity()` | Pool membership (`sens > 0`); ranking axis `dm_sens` |
| Importance | Per-input-channel mean absolute activation | `_dualmonitor_channel_importance` → **multiplies** V4 hybrid |
| Act moments | For bias correction scope | BC top fraction by sensitivity |

### 3.2 Contracts

- `_apply_fp16_budget_cap` **raises** if `dual_monitors` is empty — no profile_score-only substitute.  
- `keep_ratio` stays **r0**. DualMonitor must **not** invent a percentage keep set; it only feeds **signals** into budget ranking.  
- After calib, tunables are **re-resolved** with DualMonitor present so α/β and related knobs match THIS run (not pre-calib stale α).

---

## 4. Pillar C — Histogram V4 @ absmax (FP16 damage score)

### 4.1 Pre-measure (`_build_v4_calib_fp16_candidates`)

- Iterates **all** target Linear/Conv with weights.  
- Builds `HSWQWeightedHistogramOptimizerV4(..., quantizer=INT8Quantizer, alpha=α, beta=β)`.  
- For each layer: `_measure_v4_mse_absmax_int8(..., use_svd_leverage=True)`.  
- Fills `mse_cache[name] = estimated_mse`.  
- **No keep_ratio pre-cut.** Truncation is **only** the later budget greedy pass.

### 4.2 Measure internals (`compute_optimal_amax_with_stats_int8_range`)

1. If `use_svd_leverage` and `ndim ≥ 2`: build hybrid via `compute_hybrid_leverage_scores`.  
2. If DualMonitor Imp present: `combined = hybrid * imp_expanded`.  
3. Else: `combined = hybrid`.  
4. `WeightedHistogram.build(weight, combined)` → MSE estimate at absmax / search_range `(1,1)`.  
5. Return `estimated_mse` (+ SVD mix stats for audit logs).

Higher `estimated_mse` ⇒ layer is **more expensive to INT8** ⇒ higher FP16 priority (subject to combinator weights).

---

## 5. Pillar D — SVD (alive in ranking, not cut)

### 5.1 Hybrid construction (`compute_hybrid_leverage_scores`)

```text
W → reshape 2D
RMS magnitude_2d = W² / ||W²||₂
U, S, Vh = svd(W)
leverage_2d = (U² @ S²) ⊙ (Vhᵀ² @ S²)   (σ²-weighted)
normalize leverage
hybrid_raw = α · leverage + β · magnitude     (α > 0 required)
normalize for histogram scale
```

Audit fields (stdout + optional `log/hswq_int8_svd_mix_full_trace_*.txt`):

- All singular values  
- `norms.svd_share_of_mix_l2`  
- `proof_svd_in_ranking.{alpha_gt_0, alpha_lev_l2_gt_0, svd_share_gt_0}`  
- Banner `[HSWQ SVD MIX FULL]`

### 5.2 What “using SVD” means here

| Correct | Incorrect (rebellion) |
|---|---|
| SVD leverage enters **hybrid** | SVD run then discarded (`α=0`) |
| Imp **multiplies** hybrid | Imp **replaces** hybrid |
| Measure always `use_svd_leverage=True` | Measure with `False` |
| α from **THIS** `alpha_auto` | Fixed 0.5/0.5 or silent zero |

**Skipped SVD paths** (logged `skipped`, ones map): `ndim < 2`, all-zero / non-finite weights, `torch.linalg.svd` LinAlgError. Those layers do not contribute leverage; they are not a license to disable SVD globally.

---

## 6. Joint ranking — dynamic per-model optimal FP16 set

### 6.1 Hard ceiling

| Family | Constant | Meaning |
|---|---|---|
| SDXL INT8 V3.0 | `FP16_BUDGET_MB_HARD = 300` | Max **extra** bytes vs all-INT8 |
| Z Image INT8 V1.0 | `FP16_BUDGET_MB_HARD = 700` | Same design, larger frame |

Any other `fp16_budget_mb` is refused. Auto-optimal logic **fills inside** the frame; it does not raise the frame.

### 6.2 Pool construction

```text
pool = keep_layers ∪ hard_veto_layers ∪ {all analyze character-table layers}
     ∪ {layers with DualMonitor sensitivity > 0}
```

Intersected with modules that have weights. **No `severity ≥ 1` gate** — continuous severity ranks inside the pool; a hard threshold that dropped `0 < sev < 1` was thinking-stop and is removed.

### 6.3 THIS-model continuous branches

Before priority:

1. **`apply_fp16_infinite_ranking_branches`** — repairs DualMonitor under-measure / sibling skew using knobs from **`derive_fp16_infinite_branch_profile`** (CVs, VETO alignment, dm_starvation, γ_sibling / γ_blend, …). Knobs are continuous functions of **this** measured pool — not a shared family floor recipe.  
2. **`derive_priority_combinator`** — weights `w_sens / w_sev / w_mse` from IQR/median dispersion, optionally gated by signed VETO-alignment (Cohen-style) so anti-aligned axes fade.  
3. **`int8_fp16_budget_priority`** —  
   `exp(w_s·log1p(sens/sref) + w_v·log1p(sev/vref) + w_m·log1p(mse/mref))`.  
   If dispersion collapses, combinator uses **`equal_weight_geometric`** (`w=1/3` each) so THIS layer’s measured sens/sev/mse still discriminate. **Forbidden:** `form=uniform` → constant `priority=1.0` (size-only fill / thinking-stop).  
4. **`apply_fp16_infinite_priority_branches`** — second continuous repair in priority space.

### 6.4 Fill rule (single queue)

```text
candidates.sort(key = (-priority, +extra_bytes))
used = 0
for each candidate:
  if used + extra_bytes ≤ budget_bytes: select as FP16; used += extra
  else: drop (INT8)
```

- Linear and Conv compete in **one** auto-priority queue.  
- **No Mag-outside tax.**  
- Budget winners only are packed FP16; demoted Hard VETO / Conv / Linear go INT8.  
- Over-ceiling selection raises `RuntimeError` (refuse to proceed).

### 6.5 Why this is “per-model optimal” (not a fixed recipe)

Different checkpoints produce different:

- analyze fences and `alpha_auto`  
- DualMonitor sensitivity / Importance maps  
- V4 `estimated_mse` landscapes  
- combinator weights and infinite-branch γ vectors  
- final FP16 winner sets under the **same** MiB ceiling  

Same code path; **different numbers every run**. That is the autonomous engine contract.

---

## 7. Operator visibility (prove all four ran)

| Checkpoint | What to look for |
|---|---|
| Analyze | `Loading Analysis Data: …_distribution_profile.json` / mandated analyze subprocess |
| DualMonitor | Calib progress; `[FP16 budget] … dm_sens=N`; refusal if DualMonitor missing |
| V4 | `[V4→FP16 protect] measuring V4 estimated_mse…` and `SVD×Imp=` / `SVD-only=` counts |
| SVD | `[HSWQ SVD SETTINGS LOCK]`; per-layer `[HSWQ SVD MIX FULL]`; `proof_svd_in_ranking.svd_share_gt_0` |
| Joint | `[Autonomous priority] form=… w(sens/sev/mse)=…`; `[FP16 budget] … ceiling=… used=…` |

---

## 8. Pathological SVD skip (logged, not a global cut)

Per-tensor ones fallback when `ndim < 2`, non-finite weights, or `torch.linalg.svd` LinAlgError — logged as `skipped`. That is **not** permission to disable SVD for the run.

---

## 9. One-sentence summary

**Analyze JSON sets THIS-model fences and `alpha_auto`; DualMonitor supplies sensitivity and Importance; Histogram V4 turns Full-SVD×RMS (×Imp) into `estimated_mse` at absmax; the budget pass mixes those three axes with THIS-model continuous branches and fills SDXL 300 MiB / ZI 700 MiB without Mag-outside or keep_ratio pre-cuts.**

That is the dynamic per-model auto-optimal logic. All four pillars are required; omitting any one returns the hand-wave / rebellion paths already banned in code.
