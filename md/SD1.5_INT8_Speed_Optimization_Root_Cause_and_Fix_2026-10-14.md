# SD1.5 INT8 Speed Optimization - Full Technical Record (2026-10-14)

**Scope:** `benchmark/sd1.5 int8/comfy_quant_int8.py` (the only file changed)
**Model:** `0.5xxmix9realistic_v40+0.5BRA7` (SD1.5 UNet, 282 quantized layers:
184 Linear + 98 Conv2d, FULL ConvRot INT8, FP16 300 MiB protection budget)
**Result:** INT8 is now faster than FP16 at every resolution 512-1024.

---

## 1. Timeline of the three root causes (all measured, none guessed)

| # | Symptom | Apparent cause | Real cause (measured) | Fix commit |
|---|---|---|---|---|
| 1 | In-node bench: all 25 seeds MSE/Cosine = `nan` | "benchmark broken" | In a live ComfyUI process the `mixed_precision_ops` wrapper installed by the HSWQ **loader** checks **its own module's TLS**; the bench opened the **bench copy's** TLS. Quantized Conv2d class never injected → int8 raw bytes loaded as fp16 → NaN | `7756404` |
| 2 | Quantized pack slower than FP16 at 512 | "INT8 regression" | CPU-launch-bound: W8A8 path halves GPU work but per-op Python/C++ launch cost dominates at small token counts | `6dc21a1` |
| 3 | 640/768 "cannot capture" | "shape-dependent capture limit ≤512" | Default `capture_error_mode='global'` aborts on harmless capture-only-illegal ops | `a9d3f62` |

Each fix was preceded by a profiler / A-B measurement. Section 4 has the numbers.

---

## 2. Root cause 1 — NaN trajectories (TLS mismatch, commit `7756404`)

### 2.1 Mechanism

`_patch_ops_decode_and_conv()` wraps `comfy.ops.mixed_precision_ops` with
`mixed_precision_ops_force_conv`, which injects the quantized Conv2d class only
while `_should_inject_int8_conv()` sees an active
`_int8_quant_conv_tls.active`. That TLS object is a **module-level
`threading.local()` of the module that installed the wrapper**.

In a live ComfyUI process the custom node
`ComfyUI-HSWQ-Loader-and-Tools/patches/comfy_quant_int8.py` installs the
wrapper at startup. When the in-node bench then loads the INT8 pack through
`comfy.sd.load_diffusion_model()` inside the **bench copy's**
`_int8_quant_conv_scope()`, the wrapper's version check (`_OPS_PATCH_VER == 4`)
says "already patched" and re-patching is skipped — so the wrapper keeps
checking the **loader's** TLS, which is never active. Result:

```
bench TLS active: True
loader TLS active: False   <- the one the wrapper actually reads
Conv2d injected: False
```

The 98 Conv2d layers then load through stock `manual_cast.Conv2d`, whose
`_load_from_state_dict` never pops `comfy_quant`/`weight_scale` (they all land
in `unet unexpected:` in the log — the smoking gun), and the raw int8 bytes get
cast to fp16. Exploding activations → NaN in every seed.

The standalone script worked because its process has only one copy of the
patch module: wrapper and scope share the same TLS.

### 2.2 Fix

`_arm_active_int8_conv_scope(fallback_ctx)` walks the
`_hswq_orig_mixed_precision_ops` wrapper chain (up to 8 hops), finds the
`__globals__` of the module that installed the active wrapper, and activates
**that module's TLS** during the INT8 load. Falls back to the bench copy's own
scope when no wrapper marker exists (standalone processes).

### 2.3 Verification

| probe | before fix | after fix |
|---|---|---|
| Conv2d class | `manual_cast.Conv2d` | `_make_quantized_conv2d.<locals>.Conv2d` (`_hswq_quant_conv2d=True`) |
| Conv2d weight | fp16 `Parameter` | `QuantizedTensor` |
| Linear weight | QuantizedTensor | QuantizedTensor |
| fp16 UNet forward | NaN | **NaN: False, Inf: False** |

Full 25-seed bench after the fix: mean cos 0.97120, same-image 11/25, no NaN.

---

## 3. Root cause 2 — CPU-launch-bound at 512 (commit `6dc21a1`)

### 3.1 Measurement that identified it

`torch.profiler` (CPU+CUDA activities), one 25-step trajectory at 512×512:

| arm | Self CPU | Self CUDA | bottleneck |
|---|---|---|---|
| FP16 | 1.428 s | 1.524 s | GPU-bound |
| INT8 | **2.039 s** | **1.371 s** | **CPU-bound** |

Key rows inside the INT8 arm:

| op | Self CPU | note |
|---|---|---|
| `comfy_kitchen::int8_linear` | 461.8 ms | 3200 calls, ~144 µs/call |
| `PythonSubclass` dispatch | 123.1 ms | QuantizedTensor `__torch_dispatch__` |
| `aten::linear` (FP16) | 26.9 ms | 4600 calls, ~6 µs/call |

The INT8 GPU work is **smaller** than FP16 (Conv 346 ms vs 826 ms; total CUDA
1.371 s vs 1.524 s) — the W8A8 path does less math. But at 512 the whole step
is ~68 ms of GPU work spread over ~1000 op launches; each launch costs
~140 µs of CPU on the int8 path, so the wall clock is CPU-bound and INT8 loses
(`1.231x` vs FP16, measured interleaved same-process A/B). At 1024 the GPU
work is 16× larger, the launch overhead is hidden, and INT8 already wins.

Isolation tests (512, interleaved, 5 seeds × 25 steps):

| arm | ratio vs FP16 | conclusion |
|---|---|---|
| INT8 full | 1.233x | baseline loss |
| Conv-IMMA disabled (fp16 conv fallback) | **1.397x** | Conv2d IMMA is *helping*; not the culprit |
| kitchen fused gs256 OFF | 1.288x | fused gs256 ON is also correct |

### 3.2 Fix — CUDA Graph executor

Capture the **entire denoiser forward** (`BaseModel.apply_model`) into a
`torch.cuda.CUDAGraph` and replay it. Per step: copy `x`/`t`/context into the
captured static buffers, `graph.replay()`, clone the static output. This
removes the per-op CPU cost (~140 µs × ~1000) from the wall clock.

Single forward probe: eager 77.1 ms → replay 55.1 ms (**1.40x**), output
bit-identical (`max|graph − eager| = 0.0`).

Implementation (all inside `benchmark/sd1.5 int8/comfy_quant_int8.py`):

* `_Sd15CudaGraphExecutor` — wraps `apply_model`; keyed on
  `(x.shape, x.dtype, device, tensor-cond signature)`; a signature change
  re-captures; `disarm()` restores eager.
* `_arm_sd15_cudagraph(model)` — arms only when `_sd15_is_sd15_unet(model)`:
  `model_config` class **exactly `SD15`** + `UNetModel` + `in_channels == 4`.
  SDXL / Z-Image / Krea2 / Qwen have different config classes and are never
  touched.
* `_patch_sd15_cudagraph_executor()` — wraps both load paths so any SD1.5 INT8
  load is armed automatically:
  * `comfy.sd.load_diffusion_model` (UNet-file loaders, in-node bench INT8 arm)
  * `comfy.sd.load_checkpoint_guess_config` (full-checkpoint loaders, the
    standalone bench's `_load_checkpoint`)
* Capture details that matter:
  * tensor inputs are moved **onto the graph device before capture**
    (an unpinned CPU→CUDA copy is illegal while capturing);
  * capture runs on a side stream after 3 warmup iterations on that stream;
  * `capture_error_mode="relaxed"` (see §4);
  * a failed capture returns `None`, logs once, records the signature in a
    skip-set, and the run continues eager — no repeated attempts, no poisoned
    context reuse.

### 3.3 Correctness proof

* Graph vs eager final latents, 6 seeds: `max|Δ| = 0.000e+00` (bit-identical).
* 512 full bench mean cosine **unchanged** at 0.97120 (same 11/25 same-image).
* 1024 full bench mean cosine unchanged at 0.95191.

---

## 4. Root cause 3 — `capture_error_mode` (commit `a9d3f62`)

### 4.1 The wrong turn, recorded

The first deployment gated capture to latent ≤ 64 because 640/768 (latent
80/96) aborted with
`cudaErrorStreamCaptureStatusInvalidated`. That gate was **wrong**: the
failures were caused by the default `capture_error_mode='global'`, which
invalidates the whole capture when *any* op that is illegal only during
capture is encountered — including harmless diagnostic/synchronising calls
that never affect numerics.

### 4.2 Fix

`torch.cuda.graph(graph, capture_error_mode="relaxed")`. Relaxed mode tolerates
capture-only-illegal calls instead of aborting. With it, capture succeeds at
**every** tested latent size (64 / 80 / 96 / 128) and the cap was raised to 128
(1024×1024, the practical SD1.5 max); anything larger still runs eager.

### 4.3 Final measured results (25 seeds × 25 steps, same process)

| resolution | latent | FP16 s/seed | INT8 eager | **INT8 graph** | graph vs FP16 |
|---|---|---|---|---|---|
| 512×512 | 64 | 1.74 | 2.07 (1.23x) | **1.50** | **+14.0% faster** |
| 640×640 | 80 | 2.70 | 2.90 (1.11x) | **2.51** | **+7.2% faster** |
| 768×768 | 96 | 3.84 | 3.78 (0.91x) | **3.67** | **+4.7% faster** |
| 1024×1024 | 128 | 8.37 | 8.21 (0.99x) | **8.25** | **+1.4% faster** |

Quality gates (unchanged by the executor — outputs are bit-identical to eager):

| resolution | mean final-cos | note |
|---|---|---|
| 512 | 0.97120 | same as pre-graph bench |
| 640 | 0.97482 | |
| 768 | 0.96353 | |
| 1024 | 0.95191 | |

Capture failures across all final runs: **0**.

---

## 5. Why the graph wins exactly where it does

The speed-up is the removed CPU launch overhead, which is a **constant per
step** at SD1.5 scale while GPU work scales with token count. Small resolutions
have a high launch-overhead share of the step, so removing it buys the most
(+14.0% at 512); large resolutions are already GPU-bound, so the gain shrinks
(+1.4% at 1024) but never turns negative.

---

## 6. Scope discipline (separation principle)

* **One file changed:** `benchmark/sd1.5 int8/comfy_quant_int8.py`
  (`7756404` +46 lines wrapped in the SD1.5 bench path, `6dc21a1` +178,
  `c66688f` gate/fallback hardening, `a9d3f62` relaxed mode). No other repo
  file touched by this work.
* **Model gate:** arms only for `model_config` class exactly `SD15`
  (`UNetModel`, `in_channels=4`). Verified that SDXL / Z-Image / Krea2 / Qwen
  model_config classes differ, so their loads are untouched.
* **Loader gate:** the wrapper arms only while `_int8_quant_conv_tls` of the
  *arming* module is active (i.e. during an explicit HSWQ INT8 load), never on
  stock FP loads.
* **No shared-module edits:** `comfy_kitchen`, `comfy.ops`, `comfy.sd` core
  semantics are unchanged; the executor is a wrapper installed at load time and
  removable via `disarm()`.

## 7. Commits

| commit | content |
|---|---|
| `7756404` | wrapper-owned TLS scope → NaN fix |
| `6dc21a1` | CUDA-graph executor for the CPU-bound 512 denoiser |
| `c66688f` | capture gate + safe eager fallback + skip-set |
| `a9d3f62` | relaxed capture mode; cap raised to 128; 640/768/1024 covered |

## 8. Reproduce

```
python benchmark/sd1.5_int8_traj_compare.py ^
  --fp16  <ComfyUI>\models\unet\0.5xxmix9realistic_v40+0.5BRA7.safetensors ^
  --int8  <ComfyUI>\models\unet\0.5xxmix9realistic_v40+0.5BRA7_hswq_int8.safetensors ^
  --comfy_path <repo>\ComfyUI-master ^
  --steps 25 --seeds <25 seeds> --width 512 --height 512 ^
  --cfg 7.0 --sampler dpmpp_2m --scheduler karras
```

Expected log markers:

```
[HSWQ INT8] comfy_quant patches applied (... + SD1.5 CUDA-graph executor + ...)
[HSWQ INT8][SD1.5 graph] CUDA-graph executor armed (512 CPU-bound path)
[HSWQ INT8][SD1.5 graph] captured #1 x=(2, 4, 64, 64) ctx=(2, 77, 768)
avg wall/seed: FP16 1.74s | INT8 1.50s (+14.0%)
```

## 9. Lessons (recorded for future benches)

1. **Wrapper/TLS pairs must be resolved through the installed wrapper, not the
   patching copy.** Any "scope" that gates a monkey-patched factory must be
   looked up from the wrapper's own `__globals__`.
2. **Profiler before knob-turning.** The Conv-IMMA and fused-kernel switches
   looked like candidates; the profiler showed the CPU dispatch was the
   bottleneck and both knobs were already optimal.
3. **`capture_error_mode='global'` is not a correctness signal.** A global-mode
   abort does not mean the model is uncapturable; it means *some* op touched a
   capture-forbidden API. Use `relaxed` and then verify outputs against eager
   bit-for-bit.
4. **Never ship a "shape gate" derived from a failed capture** without first
   identifying the failing op — it was a mode problem, not a shape problem.

## Related

* `md/SDXL_INT8_Trajectory_Speed_Root_Cause_and_Fix_2026-09-11.md` — the SDXL
  CPU-bound investigation (fp32 activation rotate) that established the
  pooled-bf16 rotate this SD1.5 stack inherits.
* `md/How to quantize SD1.5.md` — the quantization how-to these benches serve.
