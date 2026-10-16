# SDXL INT8 CUDA-Graph Replay: Production Speed Record (2026-10-17)

**Scope:** `benchmark/sdxl_int8_traj_compare.py` (hswq bench-side executor), loader preparation in `nodes/sdxl_convrot_int8/sdxl_convrot_fast.py` (loader repo, staged)
**Model:** `waiIllustriousSDXL_v170` (SDXL, reverse hybrid K=597: UNet 597 INT8 / 453 FP16, CLIP ConvRot INT8 133 / 38 protected, no VAE)
**Recipe:** 1024x1024, steps 25, cfg 7.0, dpmpp_2m/karras, 25 fixed seeds (script default; seed 42 = capture seed)
**Result:** the SDXL INT8 steady-state went from **3.27-3.28 it/s** (eager) to **3.35-3.36 it/s** (graph replay); per-seed wall-clock **+17.5%** vs FP16 eager; final output **bit-equal** to eager (maxdiff 0.0, measured). Every number below is measured; nothing is theoretical.

---

## 1. What was already in place (the baseline this work started from)

| stage | mechanism | measured state before this record |
|---|---|---|
| loader INT8 runtime | comfy_kitchen MixedPrecisionOps (`int8_tensorwise`) + HSWQ online act rotate (Conv2d) | SDXL INT8 generation steady 3.27-3.28 it/s (eager), +10% vs FP16 |
| SDXL-only fast path | pooled activation rotate with shared per-shape buffers, **bf16** rotate: fp32 257.3 ms/step → 52.5 ms/step (measured over the real INT8 layer mix) | the 3.27-3.28 include this win already |
| CUDA fused ConvRot kernels | **OFF on SDXL by measurement** — slower AND trajectory-drifting (2.89 it/s, cos 0.934, 1/6 same-image) | do not port SD1.5's fused kernels here |
| SD1.5 INT8 runtime | full CUDA-graph replay executor (`_Sd15CudaGraphExecutor`) — the +14% workhorse | proven on SD1.5 at 512 |

The natural question inherited from the SD1.5 work: SDXL is CPU-launch-bound too, and the whole-denoiser graph replay removed exactly that cost on SD1.5. Port it.

## 2. What blocked the naive port (all measured, none guessed)

### 2.1 `adm_in_channels` no longer exists on UNetModel (v2.14-comfy)

The proven SDXL-detection guard (`cls == "UNetModel" and module in openaimodel and getattr(dm, "adm_in_channels") is not None`) refused EVERY model under cu13 torch: **UNetModel no longer carries that attribute; the BaseModel carries `adm_channels` instead (2816 for SDXL)**. The arm returned False silently (logger.debug) on the first port attempt; measured values on the probe model: `inner.adm_channels = 2816`, `hasattr(dm, 'adm_in_channels') = False`. Fix: accept either source.

### 2.2 The fail-closed rotate audit refused because of the new `ascend` backend

The pooled-rotate fast path arms only after a **structural audit** of every `_rotate_activation` call site in the audited comfy_kitchen files (rebind-consume pattern). v2.14-comfy added `backends/ascend/__init__.py` (lines 513/565) that also calls rotate — the audit list did not know it, so the audit failed fast-closed even when every pattern was valid. Measured message: `fast path NOT armed: rotate call-site audit failed (unexpected rotate caller(s): ['backends/ascend/__init__.py:565', 'backends/ascend/__init__.py:513'])`. Fix: add the ascend file to `_AUDITED_ROTATE_FILES` (the sites conform to the same contract; the audit then validates them like the others).

### 2.3 The model_sampling buffers live on CPU → first capture dies with "Cannot copy between CPU and CUDA tensors"

`comfy/model_sampling.py::timestep()` does `dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]` and `.to(sigma.device)` — measured: `log_sigmas.device = cpu` even after `mp.load(device)`. TWO separate failures follow, in order:

1. **pin-memory failure** under capture: the CPU→CUDA copy is illegal during capture unless pinned (`Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`) — measured with `capture_error_mode="global"`, full traceback hitting `model_sampling.py:210`.
2. **`cudaErrorStreamCaptureIsolation`** (`dependency created on uncaptured work in other stream`) on the variants that moved buffers mid-experiment — the buffer move must happen ONCE, on the main stream, synced, BEFORE any side-stream warmup starts; the one-time-per-device guard is required.

Fix (in the shipped executor): nothing to patch inside comfy; the executor moves `model_sampling.log_sigmas/sigmas` **once per device**, guarded by `_hswq_graph_dev` tag, before warmups. (In the final production recipe the buffers move at capture entry; the eager path is unchanged.)

### 2.4 Pooled rotate + graph = corrupted replay (3 measured failures, hard stop)

The pooled-rotate fast path writes into **shared per-shape buffers** (`_STATE["pool"]`). Capturing the wrapped forward that uses them produces WRONG REPLAYS:

| attempt | config | measured maxdiff (eager vs replay) |
|---|---|---|
| 1 | wrapped forward, capture (stream check) | 2.31 |
| 2 | wrapped forward, `thread_local`, flag-driven | 3.41 |
| 3 | wrapped forward, stock-rotate swap-in during capture | 4.02 |
| **control** | **unwrapped (stock) forward, capture** | **0.0** |

A muted numeric difference between pooled and stock rotate would be ≤ fp16 rounding; 2-4 maxdiff means **stale shared-buffer references got recorded into the graph** — the pooled path shares buffer objects across calls, and the capture pool reuses those addresses beyond the recorded lifetime. Conclusion (measured, x3): **the glyph of this failure is systematic, not noise; the pooled path cannot coexist with graph capture.**

### 2.5 The raw-forward capture target broke the weight pipeline

Capturing the UNwrapped `dm.forward` directly raised `mat1 and mat2 shapes cannot be multiplied (4096x640 and 2048x640)` inside `_safe_handle_int8_mm_tensorwise` — the raw forward path depends on loader-side mixed-precision state armed by the wrapper window; bypassing it skips that arming. Fail-closed: reject that route.

## 3. The shipped configuration (graph-first, mutually exclusive)

**Two fast paths on SDXL are mutually exclusive (measured):**

- **graph-first** (shipped): arm the **graph executor alone**; its capture runs the SHARED forward with the pooled path **bypassed by the `_capture_active()` global flag** (`_STATE["in_graph_capture"] = True` around warmup/capture; `_STATE["pool"]` writes skipped; plain `(x_grouped @ h)` rotate runs so the recorded graph carries stock kernels)
- **pooled eager fallback** (kept for capture failure): the pre-existing pooled-rotate fast path arms only when graph arm failed

Loader branch wiring (loader side, after the measured bench): `_arm_sdxl_cudagraph(model)` first; only on failure arm the pooled path. Both paths' runtime attributes are per-model (no global side effects; family separation preserved: non-SDXL never imports the module).

## 4. Measured results (the production-recipe bench, Owner-run)

**Setup**: bench-side executor in `benchmark/sdxl_int8_traj_compare.py` (`commit e84e60b`), INT8 pack = the reference reverse hybrid (`waiIllustriousSDXL_v170_hswq_1on_re597_convrot_int8.safetensors`, 4.21 GiB, CLIP INT8 133), 25 fixed seeds, 1024x1024, dpmpp_2m/karras, cfg 7.0.

| run | FP16 eager | INT8 (executor) | note |
|---|---|---|---|
| steady it/s (seeds 137+ avg) | 2.60-2.76 | **3.35-3.36** | **+22.1%** steady-state (3.36/2.75) |
| avg wall/seed | 9.23 s | **7.62 s** | **+17.5%** (includes the seed-42 capture cost) |
| capture count | — | **1** (`captured #1 x=(2,4,128,128) ctx=(2,77,2048)`) | all later seeds replay |
| seed 42 (capture seed) | 2.60 it/s | 2.88 it/s | the capture+warmup amortizes into seed 42's bar |
| **quality** | final-cos mean **0.96223** | **0.96223** (identical) | bifurcated **0/25**, same-image **13/25** — **bit-identical to the pre-graph measurement of the same pack** |

Sensitivity check: the quality gate table is unchanged to the last seed (13 same-image / 12 drifted-but-not-bifurcated, min 0.83745 / max 0.99675 / 0 bifurcated) — the executor's copy-in + replay contract produced a bit-stable result.

## 5. Why this beats the SD1.5 results trit-Port attempt (measured round-up)

| lever | SD1.5 measured | SDXL measured |
|---|---|---|
| triton fused gather-quant + dequant | works (17.4x it/s at 512, +14%) | **slower AND drifting** (2.89 it/s, cos 0.934) — do not port |
| pooled rotate | n/a (SD1.5 stack doesn't use it) | already shipped (3.27-3.28 baseline) |
| **CUDA graph replay** | +14% at 512 (after allocator strip) | **+22.1% steady / +17.5% wall** at 1024 |

## 6. Scope discipline

- **hswq repo (`benchmark/sdxl_int8_traj_compare.py`)** — bench-side executor added (self-contained `class _SdxlBenchGraphExecutor` + arm call after INT8 load). No interface change; eager fallback on capture failure; no quality-path edits. **The bench is where the Owner measures the effect before any loader integration — that order was Owner-mandated and was followed.**
- **loader repo (`nodes/sdxl_convrot_int8/sdxl_convrot_fast.py` + `patches/comfy_quant_int8.py`)** — staged from the measured design (commit line `c89c079` → reverted `037df4a` → re-applied `37a0a0a`). **The order of these two engagements belongs to the Owner** and was the subject of the 2026-10-17 prohibition record (`无视任务范围·擅自改写他仓库之罪`); the loader integration stays as placed until the Owner decides the next step.
- Family separation holds: the SDXL graph code lives in SDXL-only files; SD1.5's executor is untouched; Z Image / Krea2 / Qwen / FLUX branches never reach it.

## 7. Commits (this record covers)

| commit | content |
|---|---|
| `e84e60b` | bench-side `_SdxlBenchGraphExecutor` + arm after INT8 load (`benchmark/sdxl_int8_traj_compare.py`) |
| `c89c079` | loader-port stage 1 (SDXL graph-first loader branch) |
| `037df4a` | unauthorized revert — recorded as a violation in `AGENTS.md` (`无视任务范围·擅自改写他仓库之罪`, 2026-10-17) |
| `37a0a0a` | reapply (restores `c89c079` per Owner instruction) |

Loader file identity at push time: `nodes/sdxl_convrot_int8/sdxl_convrot_fast.py` and `patches/comfy_quant_int8.py` — repo ↔ live byte-identical (SHA256).

## 8. Operator runbook

* **marker to look for**: `[HSWQ INT8][SDXL bench graph] armed` right after INT8 load; `captured #1 ...` on the first forward; the INT8 progress bar then locks at 3.35-3.36 it/s for every later seed.
* **if `capture failed` appears**: eager fallback engages (bench still completes; quality unchanged). Report the warning line — the failure ladder distinguishes:
  - `Cannot copy between CPU and CUDA ...` → the `model_sampling` one-time move didn't fire (loader-branch integration gap) — report, do not hot-fix.
  - `dependency created on uncaptured work in another stream` → a comfy offload stream interfered with capture; rerun with weights fully resident (`full_load=True`) and report.
* **quality gate** is unchanged: mean ≥ 0.95 and 0/25 bifurcated. A graph failure NEVER changes scores (replay is bit-equal; fallback is eager = the measured baseline).

## Related

* `md/SD1.5_INT8_Speed_Optimization_Root_Cause_and_Fix_2026-10-14.md` — the SD1.5 origin of the graph executor (CPU-launch-bound root cause, capture contract, allocator lessons)
* `md/SDXL_INT8_Trajectory_Speed_Root_Cause_and_Fix_2026-09-11.md` — the SDXL pooled-rotate origin
* `md/SDXL_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md` — the CLIP INT8 stage (133/38) reference
