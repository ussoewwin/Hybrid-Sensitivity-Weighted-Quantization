# Z-Anime HSWQ Support: Complete Technical Explanation

## 1) Goal and Scope

This document explains all relevant Z-Anime support logic in this repository, with a focus on:

- What is different between Z-Anime and ZI/ZIB/ZIT paths.
- Which files were modified and why.
- How each important code path works.
- Why the benchmark-side MSE fix is done in `benchmark/zit_bench.py`.
- How we keep ZI/ZIB/ZIT behavior unchanged while adding Z-Anime-specific handling.

This is an implementation and maintenance guide, not a model theory paper.

---

## 2) Core Context

The project supports multiple model families under one workflow.  
Z-Anime behaves differently from ZI/ZIB/ZIT in key areas:

- **Checkpoint key format** (Diffusers-style attention keys vs NextDiT-style fused keys).
- **Native dtype tendencies** (Z-Anime commonly BF16-native).
- **Observed metric behavior** (raw latent MSE can look misleadingly large for Z-Anime).

Because of that, Z-Anime requires explicit branching (`is_zanime`) instead of implicit heuristic sharing with ZI/ZIB/ZIT.

---

## 3) File-Level Map

## Main quantization logic
- `quantize_zib_hswq_v1.92.py`

## Benchmark / fidelity measurement
- `benchmark/zit_bench.py`

## Related reference docs already in repo
- `md/ZIT_Benchmark_SSIM_Explanation.md`
- `test/benchmark_zit.md`

---

## 4) Z-Anime Handling in `quantize_zib_hswq_v1.92.py`

The quantizer contains Z-Anime-specific handling guarded by `is_zanime`.

## 4.1 `is_zanime` detection

The script identifies Z-Anime by key-pattern signals (for example prefixes like `all_x_embedder.2-1` and related key family structure).

Why this matters:
- Prevents accidental cross-impact on ZI/ZIB/ZIT.
- Keeps Z-Anime logic explicit and auditable.

## 4.2 Key normalization and denormalization

Z-Anime checkpoints can store attention weights in Diffusers-like separated tensors:

- `attention.to_q.weight`
- `attention.to_k.weight`
- `attention.to_v.weight`
- `attention.to_out.0.weight`

But NextDiT/Lumina-style execution expects fused structures like:

- `attention.qkv.weight`
- `attention.out.weight`

So the quantizer provides:

- normalization/fusion path for internal processing.
- denormalization/output mapping for final save compatibility.

This bridge is required so internal quantization logic and external model layout can both be correct.

## 4.3 Calibration dtype split

Z-Anime uses BF16-oriented paths during calibration/inference-sensitive stages:

- `torch.bfloat16` for Z-Anime branch.
- `torch.float16` remains for ZI/ZIB/ZIT branch.

Why:
- Avoids forcing all models into one dtype assumption.
- Reduces regressions caused by inappropriate dtype coercion.

## 4.4 Z-Anime upper clip policy

In strategy derivation, Z-Anime uses:

- `upper_clip = 0.90`

while non-Z-Anime paths keep:

- `upper_clip = 0.99`

This branch exists because Z-Anime showed quality sensitivity when using the broader cap.

## 4.5 Structural VETO and per-projection qkv VETO

Z-Anime branch includes stricter veto logic to avoid over-aggressive quantization on sensitive projections:

- shape/structure-sensitive screening.
- projection-level checks for qkv blocks.

This is why Z-Anime needs separate structural safeguards rather than full reuse of ZI defaults.

---

## 5) Benchmark Logic in `benchmark/zit_bench.py`

Recent work clarified that the MSE anomaly should be fixed in benchmark metric interpretation, without changing SSIM formula behavior.

## 5.1 Model loading returns `is_zanime`

`load_zit_model` now returns model + Z-Anime flag so downstream metric code can branch safely.

## 5.2 MSE split by model family

For Z-Anime:
- MSE is reported on normalized decoded 0-255 view (`calculate_normalized_mse`).

For ZI/ZIB/ZIT:
- MSE remains latent-space (`calculate_latent_mse`).

Rationale:
- Z-Anime raw latent MSE can be numerically large and misleading compared with visual fidelity.
- We keep SSIM unchanged while making MSE interpretation align with perceptual output for Z-Anime.

## 5.3 SSIM policy

`calculate_ssim_normalized` is not changed by this fix.  
The adjustment is in MSE path selection and output labeling, not SSIM formula.

## 5.4 Output formatting

Output now prints one MSE label per branch (no dual confusing display):

- Z-Anime: `MSE (0-255 view)`
- ZI/ZIB/ZIT: `MSE (latent)`

Formatting uses fixed-width labels for readable alignment.

---

## 6) Why This Does Not Break ZI/ZIB/ZIT

Non-regression safety is based on explicit branching:

- `if is_zanime: ... else: ...`

All Z-Anime-specific policies (dtype path, upper clip profile, metric path labeling) are branch-scoped.
ZI/ZIB/ZIT continue to use previous latent-MSE benchmark path and non-Z-Anime quantization defaults.

---

## 7) Practical Verification Checklist

When changing Z-Anime logic, validate in this order:

1. **Branch isolation**
   - Confirm all new behavior is behind `is_zanime`.
2. **Quantizer stability**
   - Ensure no unintended edits to generic ZI/ZIB/ZIT path.
3. **Benchmark metric consistency**
   - Z-Anime prints `MSE (0-255 view)`.
   - ZI/ZIB/ZIT print `MSE (latent)`.
4. **SSIM integrity**
   - No formula drift in `calculate_ssim_normalized`.
5. **Readable reporting**
   - MSE/SSIM lines remain aligned and single-valued per model.

---

## 8) Maintenance Rules (Recommended)

- Treat Z-Anime as an explicit compatibility branch, not an implicit side case.
- Prefer benchmark-side metric-space alignment before quantizer-wide structural edits when issue is metric interpretation.
- Preserve branch boundaries first, then optimize internals.
- Keep labels and metric definitions transparent in benchmark output.

---

## 9) Summary

Z-Anime support in this repo is intentionally branch-specific and grounded in practical differences:

- key layout compatibility,
- dtype behavior,
- clipping sensitivity,
- and metric interpretation needs.

The current design keeps Z-Anime fixes local while protecting ZI/ZIB/ZIT stability.  
For the recent MSE anomaly, the correct fix path is benchmark-side MSE space handling with unchanged SSIM formula.

