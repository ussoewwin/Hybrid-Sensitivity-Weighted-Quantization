# SDXL INT8 Trajectory Bench — Why INT8 Was Not Faster Than FP16

**Date:** 2026-09-11
**Scope:** `benchmark/sdxl_int8_traj_compare.py`, `benchmark/int8/comfy_quant_int8.py`
**Artifacts:** `waiIllustriousSDXL_v170.safetensors` (FP16) vs
`waiIllustriousSDXL_v170hswq_r32_1off_convrot_int8.safetensors` (ConvRot INT8)
**Fix commit:** `82e611b` — *perf(int8-convrot): pooled bf16 activation rotate*

---

## 1. Symptom

The SDXL INT8 trajectory bench reproduced the FP16 trajectory correctly
(cosine ~0.96) but **INT8 was not faster than FP16**. Measured wall-clock ratio
over 25 steps x 6 seeds in a single process:

| | INT8 it/s | INT8 / FP16 | cos mean |
|---|---|---|---|
| before the fix | 2.43 | **1.069x** (slower) | 0.95472 |
| after the fix  | **2.90** | **0.894x** (faster) | 0.95867 |

A quantized checkpoint that is *slower* than its FP16 source means the
quantization win was being spent elsewhere. The job of this document is to
name exactly where, and to record the wrong turns taken on the way.

---

## 2. Measurement protocol

All numbers below come from the same harness so they are comparable:

* one Python process, both checkpoints resident, `_hard_free_vram()` between stages
* `--steps 25`, seeds `42,137,145,5000,10000,501285`, 1024x1024, cfg 7.0,
  `dpmpp_2m` / `karras`, `--attention` fixed per arm
* FP16 reference trajectories captured once, per-seed cosine against INT8
* profiler pass with `torch.profiler` (CUDA activity) after a warmup run

Cross-process comparisons were explicitly avoided: single-step or 6-step
cross-process timings were shown to be noise-dominated (see §5).

---

## 3. Root cause: the ConvRot activation rotate

ConvRot (rotation-based INT8) requires the **input activation** to be rotated by
the same Hadamard matrix that was applied to the weight offline
(`W_rot = W @ H^T`, therefore `x_rot = x @ H`). Removing the activation rotate
collapses the output (measured cos 0.333), so it cannot be skipped.

The rotate runs once per INT8 linear, i.e. hundreds of times per denoise step.
The previous implementation forced every activation through float32:

```python
def _rotate_activation_f32(x, h, group_size: int):
    orig_shape = x.shape
    features = orig_shape[-1]
    n_groups = features // group_size
    if x.dtype == torch.float32 and h.dtype == torch.float32 and h.device == x.device:
        x_grouped = x.reshape(-1, n_groups, group_size)
        return torch.matmul(x_grouped, h).reshape(orig_shape)
    x_f = x.reshape(-1, n_groups, group_size).float()   # fp16 -> fp32 promote
    h_f = h.to(dtype=torch.float32, device=x.device)
    return torch.matmul(x_f, h_f).reshape(orig_shape)   # fp32 GEMM, no out=
```

Two problems:

1. **fp32 promotion.** Input activations are fp16/bf16. Promoting them to fp32
   in a `[M, K] -> [M, K]` reshape+matmul, then handing fp32 to the int8
   quantizer, costs an extra cast pass and forces an FP32 SGEMM path.
2. **Allocation per call.** `torch.matmul(...)` with no `out=` allocates a new
   tensor on every invocation — hundreds of allocations per step.

Isolated cost over the **real** SDXL INT8 layer mix (M=4096, group size 256):

| rotate form | ms/step |
|---|---|
| pooled fp32 | 257.3 |
| **pooled bf16** | **52.5** |
| batched `bmm` fp32 | 540.3 |
| allocating fp32 matmul | 166.5 |

The rotate alone accounted for ~97 ms/step (fused rotate+quant 129.0 ms vs plain
quantize 32.0 ms over the same mix), which is the same order as the entire INT8
GEMM saving. That is the whole story of "why INT8 was not faster".

---

## 4. Op-level budget (1 step, FP16 vs INT8)

Delta from the profiler, FP16 -> INT8, one denoise step:

| bucket | FP16 ms | INT8 ms | delta |
|---|---|---|---|
| fp16 tensorop GEMM (removed) | 207.37 | 37.76 | **-169.61** |
| int8 linear (comfy) | 0.00 | 59.53 | +59.53 |
| cutlass int8 GEMM | 0.00 | 57.60 | +57.60 |
| `direct_copy` (dtype casts) | 2.11 | 25.43 | +23.32 |
| cutlass simt (residual fp32) | 0.00 | 10.50 | +10.50 |
| elementwise | 34.02 | 43.11 | +9.09 |
| conv im2col | 50.27 | 50.27 | 0.00 |
| sdpa flash | 38.97 | 38.56 | -0.41 |
| **total** | **353.6** | **345.1** | **-8.5** |

The INT8 GEMM saves 169.6 ms; rotate+quant, casts and a residual fp32 SGEMM
consume ~93 ms of it, leaving only 8.5 ms. The rotate was the single largest
recoverable item.

---

## 5. What was wrong on the way (and how it was corrected)

### 5.1 Misjudgement: "the kitchen fused ConvRot kernel must be faster"

`b4e7558` enabled the kitchen CUDA fused ConvRot kernels, based on a
**3 seeds x 6 steps** measurement showing `INT8/FP16 = 1.081x` with fused OFF
versus `0.976x` with fused ON.

That was wrong. Re-measured at full protocol (6 seeds x 25 steps, same process,
same seeds):

| arm | INT8 it/s | cos mean |
|---|---|---|
| fused OFF | 2.97 | ~0.958 |
| fused ON  | **2.89** (slowest) | **0.934** (1/6 same-image) |

The fused kernel was both the slowest and trajectory-drifting. It was reverted
in `30388e0`, restoring the quality-safe fused-OFF configuration.

**Lesson recorded:** short (6-step) cross-process timings are noise-dominated and
must not be used to accept a perf change. The trajectory difference was also
invisible at 6 steps and only surfaced at 25.

### 5.2 Misjudgement: the fused kernel was the lever, so the rotate was ignored

Because the first A/B pointed at the kernel switch, the actual dominant cost —
the fp32 activation rotate — was not instrumented until later. Once the staged
profiler (`rotquant_cost.py` / `rot_cost.py`) separated rotate from quantize, the
257 ms rotate became obvious.

### 5.3 Over-processing: forcing a float32 Hadamard

The patch also forced the Hadamard matrix to be rebuilt in float32 on every
access, on the theory that bf16 would introduce mid-cast noise:

```python
_orig_build = iu._build_hadamard
def _build_hadamard_f32_always(size, device="cpu", dtype=torch.float32):
    del dtype
    return _orig_build(size, device=device, dtype=torch.float32)
iu._build_hadamard = _build_hadamard_f32_always
```

This is unnecessary. For `n = 256` the Hadamard entries are `+-1/sqrt(256) = +-2^-4`,
which is **exactly representable in fp16 and bf16**. The measured difference in
resulting INT8 activation codes between the fp32-rotate and bf16-rotate paths is
**0.00011%** on the real layer shapes. The forcing block was removed.

### 5.4 Earlier dtype bugs in the same area (context)

Three commits preceded the speed fix while chasing a `RuntimeError: Expected
query, key, and value to have the same dtype`:

| commit | what |
|---|---|
| `baf4f87` | removed the online Conv2d activation rotate on a wrong premise — the rotate is mathematically required; reverted by `c443274` |
| `dffe5b0` | restored the input dtype **after** the Conv2d rotate (the float32 leak into attention was the actual crash) |
| `6cfbaa0` | moved the cast **before the permute-back** so the permute and conv no longer run in fp32 |

The final correctness/speed split is `c443274` + `6cfbaa0`, i.e. *keep the
required rotate, but never let fp32 escape the rotate*.

---

## 6. The fix (`82e611b`)

Replace the fp32 promote-and-allocate rotate with a **pooled bf16 matmul**:
one kernel, zero allocation, output buffer reused across calls.

```python
_ROT_POOL_HSWQ: dict = {}

def _rotate_activation_hswq(x, h, group_size: int):
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size
    x_grouped = x.reshape(-1, n_groups, group_size)
    if h.dtype != x_grouped.dtype or h.device != x_grouped.device:
        h = h.to(dtype=x_grouped.dtype, device=x_grouped.device)
    key = (tuple(x_grouped.shape), x_grouped.dtype, str(x_grouped.device))
    out = _ROT_POOL_HSWQ.get(key)
    if out is None:
        out = torch.empty_like(x_grouped)
        _ROT_POOL_HSWQ[key] = out
    torch.matmul(x_grouped, h, out=out)
    return out.reshape(orig_shape)

iu._rotate_activation = _rotate_activation_hswq
```

Design points, each tied to a measurement:

* **dtype = the activation's own dtype.** No promotion. Safe because the
  Hadamard entries are `+-2^-4` (exact in fp16/bf16); only the accumulator
  differs, and the resulting INT8 codes differ by 0.00011%.
* **Pooled output buffer** keyed by `(shape, dtype, device)`. One `torch.matmul`
  with `out=`, no per-call allocation. Buffer reuse is safe here because the
  caller consumes the rotated tensor before the next rotate of the same shape.
* **Rebind loops updated** to `_rotate_activation_hswq`, and the
  `_build_hadamard` forcing block removed (§5.3).
* **Fused kernel stays disabled** (fused OFF from `30388e0` is untouched).

No new flags, no new environment variables, no new log output.

---

## 7. Verification

**A. Full protocol A/B, one process, identical seeds (the deciding measurement)**

| arm | INT8 it/s | s/step | INT8/FP16 | cos mean | per-seed cos |
|---|---|---|---|---|---|
| P: fp32 rotate (before) | 2.43 | 0.412 | 1.069x | 0.95472 | 0.93941 0.96410 0.93818 0.98349 0.96263 0.94052 |
| Q: pooled bf16 rotate (after) | **2.90** | **0.345** | **0.894x** | **0.95867** | 0.96160 0.96534 0.93441 0.98384 0.96920 0.93760 |
| FP16 reference | 2.60 | — | — | — | — |

**B. Post-fix bench run (production command)**
`fusedOFF 25step: FP16 56.59s | INT8 51.22s (INT8/FP16 0.905x) | mean cos 0.95867`
— the 0.894x / 0.905x agreement across two runs confirms the win is real and not
a one-off.

**C. Isolated rotate cost.** 257.3 ms -> 52.5 ms over the real layer mix.

**D. Numerical equivalence.** INT8 activation codes differ from the fp32-rotate
path by **0.00011%** (measured over the exact layer shapes); the trajectory
cosine is *higher* after the change, not lower.

---

## 8. Reproduce

```
python benchmark/sdxl_int8_traj_compare.py ^
  --fp16  <ComfyUI>\models\unet\waiIllustriousSDXL_v170.safetensors ^
  --int8  <ComfyUI>\models\unet\waiIllustriousSDXL_v170hswq_r32_1off_convrot_int8.safetensors ^
  --comfy_path <ComfyUI-master> ^
  --steps 25 --seeds 42,137,145,5000,10000,501285
```

Expected: `INT8/FP16 ~0.89-0.91x`, `cos mean ~0.9587`.
The default `--attention sdpa` is correct for SDXL (see appendix).

---

## Appendix — SA2 on SDXL

SDXL already has the SA2 path (`--attention sage2` / `apply_sage2_attention`),
wired the same way as the Krea2 bench (`comfy.ldm.modules.attention`
`optimized_attention_masked` override). Measurement status:

| arm | INT8 it/s | cos mean | attention calls |
|---|---|---|---|
| stock SDPA | **2.99** | 0.95867 | 0 (not patched) |
| sage2, instrumented wrapper (sync per call) | 2.26 | 0.95716 | 21420 (all dim64, 0 fallback) |
| sage2, clean wrapper (no sync/stats) | 0.38 | 0.95716 | 21420 (all dim64, 0 fallback) |

SA2 is not used for SDXL. This is consistent with the existing record in
`md/Why_SA2_Removed_From_SDXL.md` ("SA2 lowers calibration scores ... no
meaningful speed improvement was observed for SDXL") and with the fact that SA2
is tuned for DiT attention geometry (Krea2 / Z-Image); SDXL is a UNet with
dim-head 64 across all 21420 calls.

**Unresolved:** the two sage2 arms disagree with each other by an order of
magnitude (66.3 s instrumented vs 391.0 s clean in the same process, same
seeds), while the trajectory result (cos 0.95716) is identical. The per-call
timing instrumentation is therefore not a trustworthy basis for SA2 numbers, and
the clean-arm figure is not yet explained. The direction (SA2 slower than SDPA
on SDXL) is consistent across both, but the magnitude is not settled and is
flagged rather than asserted.

---

## Related

* `md/Krea2_NVFP4_TC_Speed_Regression_and_Fix_2026-09-11.md` — the same class of
  bug on the NVFP4 TC path (fp32 promote in `rotate_last_dim_pooled`, 53% of the
  step, fixed by a TC-only pooled bf16 rotate).
* `md/Why_SA2_Removed_From_SDXL.md`
* `md/HSWQ_INT8_SDXL_Technical_Guide.md`
