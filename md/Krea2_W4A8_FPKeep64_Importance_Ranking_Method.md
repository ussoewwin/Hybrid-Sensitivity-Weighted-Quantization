# Krea2 W4A8 FP-Keep Expansion to 64 BF16 Layers — Importance Ranking Method

Date: 2026-10-11. Model: `moodyKrea2MixUncensoredV8SEGoes_v8seBF16FP8_bf16.safetensors`
(24.48 GiB, 12.82 B params, `model.diffusion_model.*` bf16/f32).
Machine-readable records: `benchmark result/krea2_layer_relL2.json`,
`benchmark result/krea2_layer_abserr.json`.

## 1. Objective

The v8se W4A8 pack keeps 40 matmul layers in their base dtype, mirroring the
hswq ConvRot INT8 protection exactly (32 BF16 `txtfusion.layerwise_blocks.*` /
`refiner_blocks.*` attn/mlp + 8 F32 edge layers: first / last.linear / tmlp.0 /
tmlp.2 / tproj.1 / txtfusion.projector / txtmlp.1 / txtmlp.3).

Task: expand the **BF16 keep to 64 layers** by importance measured from this
file itself (no inheritance from any other pack's layer list). The 8 F32 layers
keep their F32 rule unchanged (base dtype, no cast, no quantization).

| class | before | after |
|---|---|---|
| BF16 keep (txtfusion 0-1, 0-2) | 32 | **64** |
| F32 keep (edge) | 8 | 8 |
| W4A8 converted | 224 | 192 |

## 2. Importance metric — definition and justification

For each W4A8 candidate layer W (the 224 2D layers with in_features % 256 == 0
that are not already protected):

```
W_dq = dequant( quant(W) )      # exact production path, see §3
relL2(W)      = ||W_dq - W||_F / ||W||_F          # relative requant error
err_norm(W)   = ||W_dq - W||_F = relL2 * ||W||_F  # absolute requant error
```

Two-stage ranking, because the first measurement produced a decisive negative
result (documented in §5):

1. **relL2 (relative)** — first instrument. Rationale: for a linear op,
   `||Δy|| <= ||ΔW|| * ||x||`, so relative weight perturbation proportionally
   bounds relative output perturbation. It is static (checkpoint-only, no
   activations, deterministic).
   Measured: relL2 spans **0.07264 – 0.07352 across all 224 layers — only
   1.22% spread**. ConvRot rotates every group toward a Gaussian-like
   spectrum, and g16 codebook quantization then commits near-identical
   *relative* error on every layer. The rank-32/33 gap is ~2e-6 (numerically
   indistinguishable). **relL2 alone has no discriminating power for this
   format** — reported as measured, not hidden.
2. **err_norm (absolute)** — the metric that actually separates layers.
   The trajectory bench's final-MSE divergence responds to absolute
   perturbation energy, which is err_norm = relL2 * ||W||_F: large-norm,
   high-parameter layers inject the most absolute error into the network.
   Measured spread: **max 37.9, min 6.2 → 6.12x** — real separation.

Selection rule (final): take the **top 32 of the 224 candidates by
err_norm**, add them to the existing 32 BF16 keeps → BF16 keep = 64.
Deterministic tie-break at the cut: larger `params`, then module name
ascending (applied; the cut line values are listed in §5).

## 3. Measurement procedure (exact converter path)

Runner: `Krea2/krea2_rank_fpkeep.py` (per-layer pack+dequant) and
`Krea2/krea2_rank_abserr.py` (norms from the base file). Both import routing
from `native_convert_w4a8_krea2.py` itself (`_is_fp_keep`, `_QUANT_GROUP_SIZE`)
so routing and math cannot diverge from production:

1. Enumerate `model.diffusion_model.*.weight`, ndim >= 2, float dtype.
2. Candidates = 2D, `in_features % 256 == 0`, not already protected (224).
3. CPU, fp32 input — identical to the converter's kitchen call:
   `quantize_w4a8_int8_weight(w, group_size=16, convrot_groupsize=256,
   symmetric=True, scale_dtype=fp8_e4m3fn, codebook=True, bits=4)`,
   then `dequantize_w4a8_int8_weight(...)` which returns the original
   (unrotated) basis — same decode the runtime applies.
4. `relL2` from Frobenius norms of (W_dq - W) and W.
5. `err_norm = relL2 * ||W||_F` with `||W||_F` read directly from the base
   checkpoint.
6. JSON dumps: per-layer `{module, N, K, params, base_dtype, routing, relL2,
   wnorm, err_norm}` for all 224 candidates; protected layers recorded with
   `protected_now=true, relL2=null` (never re-measured — they are never W4A8).

Why this matches production: same comfy_kitchen API the converter uses on CPU
tensors (eager backend; the CUDA fused kernel is the same storage contract on
device), same stamped parameters (`group_size=16`, `convrot_groupsize=256`,
symmetric codebook, fp8 group scales), same dequant the loader performs.

## 4. Measured outcome — selected 32 (err_norm descending)

All 32 selected layers are the **MLP (feed-forward) tensors of blocks 2-14**
(`down`: 13, `gate`: 10, `up`: 9; 100.7 M params each at gate/up/down;
down is 6144x16384, gate/up 16384x6144). Top of each block's FFN error mass.

| # | err_norm | relL2 | N x K | module |
|--:|--:|--:|--|--:|
| 1 | 37.9 | 0.07302 | 6144x16384 | blocks.9.mlp.down |
| 2 | 37.7 | 0.07302 | 6144x16384 | blocks.10.mlp.down |
| 3 | 37.3 | 0.07297 | 6144x16384 | blocks.11.mlp.down |
| 4 | 37.2 | 0.07299 | 6144x16384 | blocks.8.mlp.down |
| 5 | 37.0 | 0.07299 | 16384x6144 | blocks.9.mlp.gate |
| 6 | 37.0 | 0.07299 | 16384x6144 | blocks.11.mlp.gate |
| 7 | 36.8 | 0.07299 | 16384x6144 | blocks.8.mlp.gate |
| 8 | 36.4 | 0.07302 | 16384x6144 | blocks.11.mlp.up |
| 9 | 36.3 | 0.07302 | 6144x16384 | blocks.7.mlp.down |
| 10 | 36.1 | 0.07303 | 16384x6144 | blocks.9.mlp.up |
| 11 | 35.8 | 0.07302 | 16384x6144 | blocks.7.mlp.gate |
| 12 | 35.6 | 0.07303 | 6144x16384 | blocks.12.mlp.down |
| 13 | 35.6 | 0.07303 | 6144x16384 | blocks.6.mlp.down |
| 14 | 35.4 | 0.07307 | 16384x6144 | blocks.8.mlp.up |
| 15 | 35.2 | 0.07306 | 16384x6144 | blocks.12.mlp.gate |
| 16 | 35.1 | 0.07304 | 6144x16384 | blocks.5.mlp.down |
| 17 | 35.0 | 0.07308 | 16384x6144 | blocks.6.mlp.gate |
| 18 | 35.0 | 0.07309 | 16384x6144 | blocks.12.mlp.up |
| 19 | 34.6 | 0.07309 | 16384x6144 | blocks.5.mlp.gate |
| 20 | 34.6 | 0.07308 | 6144x16384 | blocks.13.mlp.down |
| 21 | 34.5 | 0.07307 | 6144x16384 | blocks.4.mlp.down |
| 22 | 34.4 | 0.07311 | 16384x6144 | blocks.10.mlp.up |
| 23 | 34.4 | 0.07311 | 16384x6144 | blocks.10.mlp.gate |
| 24 | 34.4 | 0.07311 | 16384x6144 | blocks.13.mlp.up |
| 25 | 34.3 | 0.07312 | 16384x6144 | blocks.7.mlp.up |
| 26 | 34.2 | 0.07312 | 16384x6144 | blocks.13.mlp.gate |
| 27 | 33.7 | 0.07312 | 6144x16384 | blocks.3.mlp.down |
| 28 | 33.7 | 0.07316 | 16384x6144 | blocks.4.mlp.gate |
| 29 | 33.3 | 0.07320 | 16384x6144 | blocks.6.mlp.up |
| 30 | 33.3 | 0.07315 | 6144x16384 | blocks.14.mlp.down |
| 31 | 33.0 | 0.07319 | 16384x6144 | blocks.14.mlp.up |
| 32 | 32.8 | 0.07319 | 6144x16384 | blocks.2.mlp.down |

Boundary: rank-32 err_norm 32.8; ranks 33-40 sit at 32.7-32.3 (smooth tail,
no cliff — stated honestly; the cut is at the rank-32 line by rule). The full
abserr ranking for all 224 candidates is in `krea2_layer_abserr.json`.

Structural reading (not a metric): the selected mass is the feed-forward of
the **middle blocks (2-14)**; attention tensors (wq/wk/wv/wo) of those blocks
fall below the cut because their `down`-side norms are smaller, and edge
blocks 0-1 / 15+ have lower ||W||_F on their FFN. This is the absolute-error
ranking behaving as intended: protect the tensors carrying the most requant
error energy.

## 5. Why relL2-only was rejected (negative result, kept as record)

Top-by-relL2 (0.07352-0.07320) are layers like `blocks.26/27.attn.wo`,
`blocks.4/16-21.mlp.up` — a set whose err_norm values are *not* at the top;
rank32 vs rank33 relL2 differ by 2e-6 (0.073201 vs 0.073199 in full
precision). Choosing by relL2 alone would be selecting inside measurement
noise. The absolute metric was computed from the base file directly
(torch.norm over bf16-as-float), not from the quantization run, so the
comparison is not circular.

## 6. Known limits

- err_norm is static and per-layer: no activation distribution, no
  cross-layer accumulation/cancellation. The 25-seed trajectory bench remains
  the ground truth for the final quality decision; this ranking only picks
  which 32 layers to lift to BF16 before that judgement.
- The existing 32 txtfusion BF16 keeps are not re-measured here (they never
  enter W4A8); the final protected count is 64 BF16 + 8 F32.
- Cut-line values are close (32.8 vs 32.7 at the boundary); the deterministic
  tie-break (params, then module name) fixes the set reproducibly. A bench
  comparison of this 64-keep pack against the 40-keep pack is the next
  validation step.

## 7. Outputs

- Selection table above → implemented as an explicit module list in
  `Krea2/native_convert_w4a8_krea2.py` (`_FP_KEEP_TOP64_ADDITIONS`, each entry
  annotated with its measured err_norm) merged with the existing
  `_FP_KEEP_EXACT` / `_FP_KEEP_PREFIXES` rules.
- Regenerated pack: `moodyKrea2MixUncensoredV8SEGoes_v8se_fpkeep64_convrot_w4a8.safetensors`.
- Verification protocol (same as the 40-keep pack): protected set keeps base
  dtype exactly (64 BF16 + 8 F32), quantized set == candidates minus the 32,
  Krea2 detection parity, W4A8 storage contract, offsets, NaN/Inf scan.
- Then: `benchmark/krea2_w4a8_traj_compare.py` FP16 vs fpkeep64 pack.
