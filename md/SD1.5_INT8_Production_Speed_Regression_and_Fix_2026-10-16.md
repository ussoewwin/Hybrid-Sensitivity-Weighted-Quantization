# SD1.5 HSWQ INT8 production-speed regression and fix (2026-10-16)

**Scope:** `benchmark/sd1.5 int8/comfy_quant_int8.py` (bench/vendored stack),
`comfyui_nodes/hswq_sd15_quantize.py` (the ComfyUI quantize node)
**Model:** `0.5xxmix9realistic_v40+0.5BRA7` (SD1.5, 203 INT8 layers,
FULL ConvRot, CLIP INT8 70 layers, FP16 300 MiB protection, bias correction ON)
**Result:** the ComfyUI-node production run went from **-17.2% (INT8 slower than
FP16 at 512)** to **+12.8% per-seed / +14.5% steady-state** — matching the
standalone-script record (+14.0%) from `md/SD1.5_INT8_Speed_Optimization_Root_Cause_and_Fix_2026-10-14.md`.
Every claim below is measured; nothing is theoretical.

---

## 1. Symptoms and measured timeline

| # | Symptom (production ComfyUI node) | Standalone script | Meaning |
|---|---|---|---|
| S0 | `avg wall/seed: FP16 1.72s \| INT8 1.69s (+4.0%)` (first log) | FP16 1.74s / INT8 1.50s (+14.0%) | INT8 loses ~10 pp vs the record |
| S1 | `[HSWQ INT8] triton fused conv failed once (ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)); staying on torch-IMMA` — absent from any standalone run | never | the fused Conv2d path dies once, in-process, at the first CUDA-graph capture |
| S2 | after the failure: all later convs take the slower torch-IMMA path; INT8 steady 15.5 it/s vs 17.4-17.6 it/s when the fused path works | 17.4-17.6 it/s | **~11.6% INT8 speed eroded** |
| S3 | after a second run with an intermediate fix: `avg wall/seed: FP16 1.69s \| INT8 1.98s (-17.2%)` | - | intermediate wrong fix made eager launches also lose the fused path → worse than before |

Root chain (all measured, none guessed):

```
ComfyUI (cu13 torch) auto-enables the cudaMallocAsync allocator backend
  → appends ",backend:cudaMallocAsync" to PYTORCH_CUDA_ALLOC_CONF
    → the quantize node spawns its quantize child via env=dict(os.environ)
      → the child spawns the bench grandchild via subprocess.run (inherits)
        → grandchild runs the 25-seed trajectory bench with the async backend
          → SD1.5 CUDA-graph executor captures the denoiser
            → FIRST triton kernel launch inside the capture is rejected:
               "Pointer argument cannot be accessed from Triton (cpu tensor?)"
              (diagnostic dump: every argument is cuda:0 with valid pointers;
               the freshly-captured tensor's pointer sits in the graph-private
               VA region the triton launcher cannot service)
            → the old once-latched warn leaves the WHOLE process on torch-IMMA
              → INT8 erodes 11.6% → production wall-clock gain collapses to +1.5%..-17.2%
```

Two layers contributed; fixing only one was measured as insufficient:

| layer | what it does | measured cost when unfixed |
|---|---|---|
| **A. CPU-staged weight scale** | ComfyUI's dynamic VRAM manager stages the QuantizedTensor params (the F32 `weight_scale`) on CPU while the int8 payload stays on GPU; both the triton dequant kernel and the torch-IMMA epilogue (`out.mul_`) fail on a CPU scale | reproduces the exact production error text in isolation; `wscale_1d` needed device pinning (commit `622d895`) |
| **B. CUDA-graph capture × triton launch × async allocator** | under `backend:cudaMallocAsync` a triton launch inside an active capture raises; under the default caching allocator the identical code captures triton kernels fine (verified twice) | the fused path inside the graph replay is the +14% engine; losing it = 15.5 vs 17.6 it/s |

## 2. The diagnostic instrument (why the culprit stayed hidden for two runs)

The failure only manifests under ComfyUI's exact process chain (3 generations,
parent CUDA state, node-inherited env). Local replications kept succeeding until
the environment was replicated bit-exact — so the code was made to self-report
instead of guessing further (commit `045ad86`):

* on a triton launch failure the bench now dumps `device / ptr / ptr0 /
  capture-mode` of every kernel argument once (`[HSWQ INT8][diag] triton-fail args:`)
* a temporary probe build (`make probe` copy in a scratch dir, never in
  production code) additionally dumps a per-launch table for quant_k/dequant_k;
  with that instrument the failing launch showed `capture_mode=True`,
  all-CUDA args, and a foreign-VA-region pointer (`0x1FF6...` vs the stable
  pool `0x1855...`) — direct evidence that the **async graph pool**, not a CPU
  tensor, was the failing argument's backing.

## 3. The fixes (in commit order, all minimal-scoped)

### Fix 1 — `wscale_1d` device pin (`622d895`)

`_int8_conv_rot_imma` built the weight scale without a device argument:

```python
wscale_1d = wscale.reshape(o).float().contiguous()          # before
wscale_1d = wscale.reshape(o).float().to(x.device).contiguous()   # after
```

plus a cache-refresh path when an already-cached scale turns out CPU-staged.
`bias_f` was already pinned (`wmat.device`); `wmat` (the heavy int8 matrix) is
deliberately NOT moved. Measured: standalone reproduces the original error with
a CPU scale; after the pin no error appears in either allocator.

### Fix 2 — allocator strip in the node (`566fcba`)

The quantize child owns its allocator choice; ComfyUI's own backend is
untouched. The node now strips the `backend:` flag for the child tree only:

```python
def _alloc_conf_for_child(env):
    conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if conf and "backend:" in conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
            p for p in (s.strip() for s in conf.split(","))
            if p and not p.startswith("backend:"))
```

Hooked in both branches (SD1.5 + SDXL reverse hybrid). Measured end-to-end
production-true replica (async-backend parent env + holding CUDA parent +
3-generation chain): `child alloc conf = max_split_size_mb:32,garbage_collection_threshold:0.8`
(backend gone) and the grandchild bench keeps the fused path alive with
INT8 `+48.1%` under concurrent GPU load.

### Fix 3 — capture-phase triton gate → attempt-and-split (`566fcba` → `1b6bcdc`)

The first gate version (`if tk is not None and x.dtype == fp16 and not _capturing`)
silently routed captures to torch-IMMA even under the caching allocator where
triton-in-capture is proven to work — production re-measured **+1.5% only**
(run4). Final version (`1b6bcdc`): the launch is attempted in ALL phases; the
`except` ladder distinguishes

* capture-phase failure → `[HSWQ INT8] triton fused conv unsupported during CUDA-graph capture (cudaMallocAsync pool); torch-IMMA used for capture/replay, fused path stays armed for eager launches` (once per process, does NOT latch the old warn)
* eager failure → the original once-latched `staying on torch-IMMA` warn.

Measured matrix with the final code:

| environment | INT8 steady it/s | FP16 steady | verdict |
|---|---|---|---|
| caching allocator (node strip active) — production now | **17.18-17.34** | 15.03-15.2 | **+14.5% steady** |
| caching allocator, 5-seed quick check | 17.6-17.66 | 15.28 | +10.6..+15% |
| async allocator still present (fallback path) | 15.56-15.63 | 15.23 | +2.3% (the IMMA ceiling, no poison) |

### Production-verified numbers (run5, after ComfyUI restart on `1b6bcdc`)

| metric | value |
|---|---|
| triton failures / capture-fallback messages | 0 / 0 |
| INT8 steady | 17.18-17.34 it/s |
| FP16 steady | 15.03-15.22 it/s |
| avg wall/seed | **FP16 1.72s \| INT8 1.50s (+12.8%)** |
| quality gate | mean final-cos 0.96364, same-image 12/25, **bifurcated 0/25** |
| FP16 budget assert | 299.96 MiB / 79 modules (ceiling 300 MiB) — unchanged |

Seed-42 (capture seed) shows the ramp 1.58 → 16.78 it/s as the graph capture completes; all 24 later seeds sit at the 17.2-17.3 plateau. The pack
`0.5xxmix9realistic_v40+0.5BRA7_hswq_int8_1792095258.safetensors` is the
production artifact of this run.

## 4. Why the standalone script never reproduced the failure

The bench grandchild inherits everything from the node chain:

1. **env**: ComfyUI process env (`PYTORCH_CUDA_ALLOC_CONF` with the backend it
   auto-appended) → node child dict → quantize child → bench grandchild. A
   standalone shell had the plain bat value without `backend:`.
2. **CUDA state**: the grandchild starts while a parent process holds CUDA
   memory and a live pool — separate processes, but the async pool state is
   per-process; the failing region is created inside the grandchild itself
   during capture.
3. **allocator choice**: the node's `env=dict(os.environ)` inheritance is the
   only bridge that carries ComfyUI's auto-appended `backend:cudaMallocAsync`
   into the bench; nothing else sets it.

Removing factor 1 (the strip) is what let the production child tree leave the
failure regime, and is why A/B tests that only copied the bat env were
insufficient for two full runs.

## 5. Scope discipline

* `benchmark/sd1.5 int8/comfy_quant_int8.py` — device pin, diagnostic dump,
  capture-fallback ladder (all inside the ConvRot INT8 Conv2d path; the SDXL /
  Z-Image / Krea2 / Qwen stacks untouched).
* `comfyui_nodes/hswq_sd15_quantize.py` — `_alloc_conf_for_child` helper +
  two call sites (SD1.5/SDXL env construction). No interface or widget change.
* Loader repo: the vendored `nodes/sd1.5 int8/comfy_quant_int8.py` copy kept
  byte-identical to the bench stack (verified: 4 copies, one SHA256).
* All four copies compiled (`py_compile`) and byte-identical at push time.

## 6. Commits

| commit | content |
|---|---|
| `622d895` | pin ConvRot IMMA weight scale to the activation device |
| `045ad86` | one-line triton-fail arg device dump (diag) |
| `566fcba` | node strips `backend:cudaMallocAsync` for the quantize child tree |
| `1b6bcdc` | attempt triton inside captures; per-capture fallback keeps eager triton armed |

Loader mirror commits: `2ff696e` / `bad486f` / `b7b27b4` / `5877b06`
(vendored copy in lockstep).

## 7. Operator runbook

* **if `triton fused conv failed once` appears again** → the diag line
  `[HSWQ INT8][diag] triton-fail args:` now prints every argument's device /
  pointer / capture phase — use that line as the report; the fix ladder
  distinguishes CPU-staged tensors from async-pool-capture rejections
  automatically.
* **if INT8 steady stays at ~15.5 it/s** while `unsupported during CUDA-graph
  capture` prints once → the child tree still runs the async backend; check
  that ComfyUI was restarted after `1b6bcdc` is on disk (a restart is required;
  the running process keeps modules cached) and that `run_nvidia_gpu.bat` does
  not hard-set an allocator with `backend:` again.
* **quality is allocator-independent**: the fallback path captured with
  torch-IMMA is bit-equal to the eager path of the same launcher state.

## Related

* `md/SD1.5_INT8_Speed_Optimization_Root_Cause_and_Fix_2026-10-14.md` — the
  CPU-launch-bound investigation that introduced the CUDA-graph executor
* `md/SD1.5_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md` — the CLIP INT8
  health record
* `md/How to quantize SD1.5.md` — the quantization how-to
