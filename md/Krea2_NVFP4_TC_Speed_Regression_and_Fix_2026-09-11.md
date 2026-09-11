# Krea2 TC (NVFP4, W4A4) — Speed Regression: Root Cause and Fix

**Date:** 2026-09-11
**Scope:** `benchmark/krea2_traj_compare.py` + `benchmark/krea2_convrot_nvfp4/`
**Symptom:** `--tc` (native Tensor Core NVFP4) ran at **2.84 it/s** on RTX 5090 instead of the expected **~3.4 it/s**.
**Fix commit:** `dee9c93`.

---

## 1. Summary

The TC hot path spent **53% of every denoising step rotating the activation**
(the ConvRot Hadamard rotation that precedes NVFP4 activation quantization).

The rotation was implemented as a **3-kernel fp32 sequence per Linear**:

```
cast bf16 -> fp32   |   fp32 GEMM (256x256 per group)   |   copy fp32 -> bf16
```

Measured on the real Krea2 layer mix (256 tensors, M=4096 tokens, CC 12.0 GPU):

| Stage | Time | Share of TC step |
|---|---|---|
| **rotate (fp32, 3 kernels)** | **466.2 ms** | **53.1%** |
| activation NVFP4 quantize | 47.4 ms | 5.4% |
| cuBLAS FP4 TC GEMM | 363.7 ms | 41.5% |
| **TC total** | **877.2 ms** | 100% |
| rotate as a single pooled bf16 GEMM | **93.7 ms** | (10.7%) |

Replacing the TC-path rotation with the single-kernel pooled form removed the
bottleneck and returned the step time to the 3.4 it/s band.

Verification run (RTX 5090, `--tc`, 12 steps, 6 seeds):

| | before | after |
|---|---|---|
| NVFP4 it/s | 2.79 – 2.85 | **3.49 – 3.56** |
| FP16 it/s (unchanged reference) | 1.63 – 1.67 | 1.63 – 1.67 |
| TC GEMM hits | 13680 | 13680 |
| dequant fallbacks | 0 | 0 |

---

## 2. Root cause (measured, not inferred)

### 2.1 Where the time went

Per-stage timing over the actual Krea2 Linear mix (K/N read from the fp16
artifact header, M=4096):

```
  rotate(fp32, 3 kernels) :    466.2 ms  (53.1%)
  rotate(bf16, pooled)    :     93.7 ms  (10.7% of TC total)
  quant                   :     47.4 ms  ( 5.4%)
  gemm                    :    363.7 ms  (41.5%)
  TC TOTAL                :    877.2 ms
  FP16 TOTAL (baseline)   :   2103.7 ms
```

The rotation was the single largest line item, larger than the FP4 GEMM itself.

### 2.2 Why it was 3 kernels

`nvfp4_runtime.rotate_last_dim_pooled()` promoted every fp16/bf16 input to
float32 before the GEMM:

```python
compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
...
torch.matmul(x_grouped.to(compute_dtype), h_matrix, out=out32)   # fp32 GEMM
out.copy_(out32)                                                  # extra copy back
return out.reshape(orig_shape)
```

That is: one cast kernel, one fp32 GEMM, one copy kernel — plus a second
(bf16) output buffer next to the fp32 one.

The docstring of the same function states the intent ("reuses the matmul output
buffer"), so the pooled-buffer design had been lost for this path.

### 2.3 Why fp32 accumulation was there at all

The fp32 accumulate is a *quality guard for the parity / offline path*: the
hybrid ConvRot weights are stored as `W @ H^T` computed in float32, so the
matching activation rotation must also accumulate in float32 to keep W and x in
the same basis.

For the TC path this guard is unnecessary, because the rotated activation is
**immediately quantized to NVFP4** (`quantize_nvfp4_act_pooled`) before the
GEMM. The FP4 grid, not the fp32 accumulator, decides what the GEMM sees.

### 2.4 Numerical equivalence of the fast form (measured)

The replacement is a single pooled bf16 GEMM. Its outputs were compared against
the fp32 form **after the real NVFP4 quantizer**:

| Input distribution | NVFP4 code differences | Block-scale differences |
|---|---|---|
| gaussian (absmax 5.8) | **0 / 12,582,912** | 0 |
| gaussian x12 (absmax 65.5) | **0 / 12,582,912** | 0 |
| heavy-tail t(3) x5 (absmax 10,560) | **0 / 12,582,912** | 0 |
| 1% outliers (absmax 504) | **1 / 12,582,912** | 0 |

Max error against a float64 reference:

```
fp32 rotate : 4.745e-02
bf16 rotate : 4.745e-02   <- identical
```

The dominant error term against the exact reference is the bf16 *input*
representation, which is unchanged by this fix (the input is bf16 before and
after). Rotating it in fp32 or in bf16 does not change what the quantizer emits.

---

## 3. The fix

**Separation principle:** the TC path and the parity/offline path must not share
the rotate implementation, because their accuracy requirements differ.

Added a dedicated function and a dedicated buffer pool:

```python
# nvfp4_runtime.py
_ROT_OUT_POOL_TC: dict = {}          # separate buffer family; never mixed

def rotate_last_dim_pooled_tc(x, h_matrix, group_size):
    if x.dtype != torch.bfloat16:
        return rotate_last_dim_pooled(x, h_matrix, group_size)   # exact fallback
    ...
    torch.matmul(x_grouped, h_matrix, out=out)   # one kernel, pooled output
    return out.reshape(orig_shape)
```

Wired only into the TC branch of `forward_nvfp4()`; the
`_full_precision_mm` (parity / stock-dequant) branch still calls the fp32
`rotate_last_dim_pooled` unchanged.

| Element | Value |
|---|---|
| New function | `rotate_last_dim_pooled_tc` |
| New pool | `_ROT_OUT_POOL_TC` (cleared by `clear_nvfp4_runtime_pools`) |
| Call site | `nvfp4_forward.py`, TC branch only |
| Parity path | untouched (fp32 accumulate preserved) |
| Non-bf16 inputs | fall back to `rotate_last_dim_pooled` (exact current behaviour) |
| `py_compile` / AST | OK |

---

## 4. Verification

Command shape (unchanged from the standard bench invocation):

```bash
python benchmark/krea2_traj_compare.py \
  --fp16 <Krea2 fp16> --nvfp4 <hybrid NVFP4 artifact> \
  --clip_path <Qwen3_VL clip> --comfy_path ComfyUI-master \
  --steps 12 --seeds "42,137,145,5000,10000,501285" --tc
```

Observed after the fix:

```
[HSWQ NVFP4] NVFP4 Linears loaded: 190 (ConvRot: 190)
[HSWQ NVFP4] Calibrated input_scale active: 190 / 190 layers (from checkpoint)
[NVFP4] seed 42     100%|...| 12/12 [00:03<00:00,  3.49it/s]
[NVFP4] seed 137    100%|...| 12/12 [00:03<00:00,  3.56it/s]
[NVFP4] seed 145    100%|...| 12/12 [00:03<00:00,  3.56it/s]
[NVFP4] seed 5000   100%|...| 12/12 [00:03<00:00,  3.56it/s]
[NVFP4] seed 10000  100%|...| 12/12 [00:03<00:00,  3.56it/s]
[NVFP4] seed 501285 100%|...| 12/12 [00:03<00:00,  3.55it/s]
  [HSWQ NVFP4] Forward execution: TC GEMM hits=13680, dequant fallbacks=0
```

The registered `input_scale` is consumed (190/190), every Linear takes the
TC GEMM (13680 hits, 0 fallbacks), and the FP16 baseline is unchanged
(1.63–1.67 it/s), confirming the fix is confined to the TC path.

---

## 5. What was NOT the cause (ruled out by measurement)

| Candidate | Verdict | Evidence |
|---|---|---|
| Per-call `torch.equal` on the alpha cache | **not the cause** | 0.0005 ms/call; 12000 calls ≈ 6 ms |
| `cudaStreamSynchronize` in the hot loop | **not the cause** | no per-call sync in the TC hot path |
| DLpack wrapper calls (`_wrap_for_dlpack`) | **not the cause** | 3.84 us each; ~7.6 ms per step across 190 Linears |
| Missing `input_scale` wiring | **not the cause** | artifact supplies 190 input_scale keys; `qconfig["parameters"]` includes `input_scale` |
| TC gate disabled / dequant fallback | **not the cause** | `TC GEMM hits=13680, dequant fallbacks=0` in every run |
| CUDA Graph path absent | **not the cause** | that path is opt-in (`HSWQ_NVFP4_CUDAGRAPH=1`) and unused by this bench |
| bf16 rotation *accuracy* | **not the cause** | 0–1 code differences out of 12.58 M after the real quantizer (2.4) |
| SA2 attention | **not the cause** | `--attention` was `sdpa` (default) in the 2.84 and 3.56 runs alike |

---

## 6. Commit trail

| Commit | Change |
|---|---|
| `93330ce` | restore checkpoint `input_scale` priority (calibrated scale used directly) |
| `2c107b4` | alpha cache / logging / alternate `input_scale` key handling |
| `08cea91` | robust `bench_dir` resolution in `setup_comfy` |
| **`dee9c93`** | **perf: pooled bf16 act rotate on the TC path only (this document's fix)** |

Upstream reference for the fast pooled form: `benchmark/hswq_stack/nvfp4/nvfp4_runtime.py`
(Z Image stack) keeps a single-buffer pooled rotation; the Krea2 stack had
diverged by promoting the activation to fp32.

---

## 7. Reproduction of the numbers in this document

All figures above were produced locally on the same GPU family (CC 12.0,
FP4-TC capable) using:

- Krea2 Linear shapes read from the fp16 artifact header (`struct` + JSON header
  parse, no model load),
- `nvfp4_runtime.rotate_last_dim_pooled` / `rotate_last_dim_pooled_tc`,
  `quantize_nvfp4_act_pooled`, `scaled_mm_nvfp4_pooled` called directly,
- `torch.cuda.synchronize()` around each timed loop; warm-up iterations discarded.

The per-stage table in section 2.1 is a sum over the real per-layer `(K, N)`
pairs of the model at M = 4096 tokens.
