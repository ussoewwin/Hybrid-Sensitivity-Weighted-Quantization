# Diag → Reverse Hybrid Quantization (SDXL, v1.1) — Technical Guide

**Document version:** 2.0
**Date:** 2026-09-13
**Scope:** the SDXL ConvRot INT8 reverse hybrid, as implemented by

| Stage | Script |
|---|---|
| V3.1 predicate (protection) | `sdxl/build_protect_list_sdxl.py` → `sdxl/quantize_sdxl_hswq_v3.1.py` |
| Trajectory impact (diag) | `sdxl/diag_impact_sdxl.py` |
| Reverse hybrid converter | `sdxl/gen_reverse_int8_sdxl_v1.1.py` |
| Gate | `benchmark/sdxl_int8_traj_compare.py` |

Version 1.0 of this document covered SDXL and Krea2 together. This revision is **SDXL-only**: the Krea2
counterpart is documented separately, and the Z Image NVFP4 variant has its own how-to.

**Companion documents:** [Trajectory-Sensitivity Impact Ranking — Technical Guide](diag_impact_trajectory_sensitivity_technical_guide.md)
(the universal theory: interaction terms, nonlinear amplification, marginal effects),
[How to quantize SDXL](How%20to%20quantize%20SDXL.md) (the CLI contract).

This document describes the **machinery**: what is measured, with which kernel, on which trajectory,
with which ranking contract, and how the hybrid file is written. The mathematics of *why* a per-layer
scalar is only the first-order term of the sampling map — and why the reverse method measures in the
low-error regime — is in the companion theory guide.

---

## 1. What this pipeline is (one paragraph)

This is the **existing HSWQ SDXL selector extended with a reverse (trajectory-impact) step** — not a
replacement for it. The selector (calibration with **Dual Monitor hooks**, **weighted-histogram MSE
V4**, **full SVD**, the **300 MiB payload budget** and the key-pattern veto) still decides which layers
are protected and stay at FP16; that decision is the **V3.1 predicate**. The reverse step then measures
every layer the predicate left in the **pool** with a fixed-seed denoising trajectory, ranks them by
the drift that their own ConvRot-INT8 reconstruction causes, and converts the **K lowest-impact pool
layers to ConvRot INT8 in ascending impact order** — **every other layer, including the whole protected
set, stays at FP16** with the original dtype and size. The predicate supplies a **static** protection
from weight/activation statistics; the reverse step supplies the **dynamic** criterion (measured
trajectory propagation) for the remaining pool. The ranking is only trustworthy in the **low-error
regime**, where the interaction terms of the sampling map are negligible — which is why the conversion
is committed in ascending order and why the boundary of usability (the largest `K` that still passes)
is **measured, never extrapolated**.

## 2. Notation

| Term | Meaning |
|---|---|
| **candidate** | a ConvRot-eligible `Linear`/`Conv2d` of the base checkpoint (boundary layers excluded) — §4 |
| **pool** | the candidates that remain after the V3.1 predicate has protected its layers (with `--artifact v31`) — §6 |
| **protected set** | layers kept at FP16: the V3.1 predicate's set ∪ the pool layers above the K cutoff ∪ all non-matmul tensors |
| **`K`** | how many ranked pool entries the converter is asked to convert (a count, not a byte budget) |
| **converted** | layers actually written as ConvRot INT8 (≤ `K`: skips do not convert) |
| **impact(l)** | the trajectory divergence caused by quantizing layer `l` alone — §5 |
| **gate** | the deterministic 25-seed latent-trajectory comparison — §8 |

## 3. Why a trajectory criterion on top of the static predicate

Static per-layer measures (weight-histogram MSE, cosine similarity, SVD leverage) score a layer **in
isolation, in weight space**. The sampling map is not additive outside the infinitesimal limit: the
joint effect of quantizing many layers contains cross terms (which can be *negative* — error
cancellation) and second-order amplification, so a static scalar alone cannot rank layers for a real
quantization budget. The V3.1 predicate therefore fixes the *protection* (which layers must not be
quantized), and the diag step measures the **propagated** effect on the actual sampler trajectory, in
the regime where a single-layer measurement is meaningful.

Measured evidence for keeping both parts (reference checkpoint, 25-seed gate):

* the static predicate is effective up to its **300 MiB** payload budget; raising the static budget to
  500 MiB lowers the 25-seed mean into the **0.92** range (the budget starts protecting the wrong
  layers), and
* a pure dynamic selection over the whole candidate set (no static protection) already drops below
  0.96 at `K ≈ 570–590`, whereas the combination (static 300 MiB protection + dynamic selection of the
  pool) holds **0.9625** at `K = 620`.

Two operational consequences follow, and both are contracts rather than suggestions:

1. **The candidate set and the ranking must be re-measured for every checkpoint.** Rankings are not
   transferable between checkpoints.
2. **The pool must come from the predicate run on the same checkpoint** (`--artifact v31`), never from
   another model's artifact.

## 4. Candidate enumeration

Candidates are derived from the loaded `BaseModel` — the base checkpoint is the only input:

```python
for n, m in net.named_modules():                      # net = patcher.model (BaseModel)
    m.weight.ndim in (2, 4)      and                  # Linear / Conv2d only
    m.weight.shape[1] >= 4       and                  # at least one Hadamard group
    not is_boundary_layer(n)     and                  # boundary excluded (§4.1)
    convrot_group_size_for_features(shape[1], 256) is not None
```

`convrot_group_size_for_features(n, 256)` returns the **largest power-of-4 divisor ≤ 256 of `n`**
(256 → 64 → 16 → 4) and `None` if there is none; a layer with `None` is not ConvRot-convertible and is
never a candidate. Reference checkpoint: **788** candidates (794 2D/4D weights with `in ≥ 4` minus 4
boundary weights and 2 without an eligible group size).

### 4.1 Boundary layers (never candidates)

`input_blocks.0.0` (conv_in), `out.*` (conv_out), `time_embed.*`, `add_embedding.*`, `label_emb.*`
(matched after stripping the `model.diffusion_model.` / `diffusion_model.` prefix). They stay at their
original precision in every artifact this pipeline produces.

> **Note — boundary sets are not identical across scripts.** `diag_impact_sdxl.py` uses the exact-match
> + prefix rule above; the converter `gen_reverse_int8_sdxl_v1.1.py` uses the *artifact-era* substring
> patterns `conv_in.` / `conv_out.` / `time_embed.` / `add_embedding.` / `label_emb.` (a name is
> protected if it contains one of them). In the production flow the ranked names come from the diag
> pool, which already excludes the diag boundary set, so the converter's own filter rarely fires — but
> the two sets are **not** the same object, and the difference is intentional (`v1.1` = artifact-era
> boundary set; `sdxl/gen_reverse_int8_sdxl.py` carries the newer set).

## 5. The measurement (diag)

### 5.1 The injection kernel

The injected value is the **reconstruction the production kernel would produce** for that layer:
`quantize → dequantize` in the rotated space, **without an inverse rotation**. This is the object the
shipped file contains, so the measured error is the real on-device error, not a proxy.

```
Linear 2D :  w_rot = w @ Hᵀ                       (group-wise, group = gs)
             q, s  = per-output-channel INT8      rowwise  scale [out, 1]
Conv2d 4D :  w_rot = rotate along in_channels     (permute → flat 2D rotate → permute back)
             q, s  = per-output-channel INT8      channelwise scale [out, 1, 1, 1]
injected  =  q · s                                # Ŵ_rot ; NO un-rotation
```

* `H` = normalized regular Hadamard matrix of size `gs` (`build_hadamard`: `kron` of the 4×4 base,
  divided by `√gs`).
* `scale = amax / 127`, `q = round(w / scale).clamp(−127, 127)`; the Linear path clamps the amax from
  below (`1e-30`), the Conv2d path from below at `1e-6`.
* Because the shipped weights are stored **rotated** and the runtime rotates the activation online
  (`x_rot = x @ H`), injecting the rotated reconstruction is the correct single-layer model of the
  deployed computation. An un-rotated injection would model a different layer.

### 5.2 The trajectory

The impact is a property of *the trajectory*, so the trajectory definition is part of the contract.
SDXL is **epsilon-prediction with the DDPM sigma schedule**: a hand-rolled `σ ∈ [0,1]` Euler loop is a
flow-matching schedule and is **not** the production trajectory (it would bypass `model_sampling` /
`calculate_input`). The implementation therefore calls the **production sampler itself**:

```python
noise = comfy_sample.prepare_noise(latent0, seed, None)
comfy_sample.sample(patcher, noise, steps, cfg, sampler, scheduler,
                    positive, negative, latent0, denoise=1.0, callback=cb, seed=seed, ...)
```

* `latent0`: `zeros([1, 4, H/8, W/8])` on the intermediate device/dtype, then
  `comfy_sample.fix_empty_latent_channels(model, latent, 8, None)`.
* conditioning: CLIP positive/negative through `encode_from_tokens_scheduled(clip.tokenize(...))` —
  the same construction the gate uses.
* `cb` captures the **per-step latent** `x` (the callback's third argument) into a list; `x_ref` is the
  pristine run, `x_t` the run with one layer injected.
* `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
* Defaults (and the production values): `--steps 25 --seed 42 --width 1024 --height 1024 --cfg 7.0
  --sampler dpmpp_2m --scheduler karras`.

### 5.3 The impact value

```
impact(l) = (1/T) · Σ_t  ‖x_t(l) − x_ref_t‖²_F / ‖x_ref_t‖²_F
```

`rel_mse` is scale-invariant; the injection writes `module.weight.data` directly and is **restored in a
`finally` block** after every measurement, so each layer is measured against the same pristine
trajectory and the values are independent of the other layers' measurement state. A layer whose
measurement raises is recorded as `NaN` (the converter drops `NaN` entries when ranking).

### 5.4 `--steps` is part of the ranking contract

`impact(l)` is a function of the trajectory, so **the number of denoising steps changes the ranking**.
Measured on the reference checkpoint: a ranking measured with a 25-step trajectory and one measured
with a 4-step trajectory select **different layer sets at the same `K`** — they agreed on 555 of 620
layers, exchanging 65 layers in each direction; in the 25-step ranking the 65 layers that the other
ranking had converted were ranked just below the cutoff (620–705), i.e. the disagreement concentrates
at the `K` boundary. Consequently:

* fix `--steps` for the whole production line (`--steps 25`);
* never compare artifacts whose rankings were measured with different `--steps`, **and never expect
  two rankings measured on different machines to be identical** — the trajectory is a long chain of
  GPU kernels, and two runs with the same parameters on different hardware differ in every value
  (measured: the same 715 layers differ in all 715 values, ~0.2–4 % relative, between a local run and
  a cloud run).

What *is* machine-independent is the conversion itself: given the same layer set, the rotated INT8
weights and scales are **byte-identical** across machines (measured on the shared 555 layers of the two
rankings above).

### 5.5 The JSON contract

```json
{ "x_ref_norm": <float>,      // ‖x_ref‖²_F of the last step (denominator reference)
  "steps": <int>, "seed": <int>, "cfg": <float>,
  "sampler": "<name>", "scheduler": "<name>",
  "base": "<absolute path of the measured checkpoint>",
  "impacts": { "<module.name>": <relative MSE>, ... } }
```

Keys are `named_modules()` names **without** the `model.` prefix (i.e. `diffusion_model.*`). Ascending
order = safest to convert. Progress prints every `--progress-every` layers (`[25/715] ...`); the first
line can take minutes (checkpoint load + CLIP conditioning + pristine trajectory).

## 6. The V3.1 predicate (static protection)

`sdxl/build_protect_list_sdxl.py` runs the permitted V3.1 selector over the base checkpoint and derives
the protection list from the pack it writes.

* It imports `sdxl/quantize_sdxl_hswq_v3.1.py` and calls its `main()` with
  `--input <base> --output <pack> --calib_file <prompts> --num_calib_samples N --num_inference_steps N
  --keep_ratio 0 --convrot --no-bias_correction --comfy_path <root> --no-bench`. The selector runs the
  calibration (Dual Monitor hooks) and applies its weighted-histogram MSE V4, full SVD, the 300 MiB
  payload budget and the key-pattern veto.
* Pack path: `<base-dir>/<base-stem>hswq_r32_1off_convrot_int8_repro.safetensors`.
  Protect list: `<impact-dir>/protect_<base-stem>.json`.
* The protection list is derived structurally: `protected = candidates(base) − converted(pack)`.
  Being derived, it cannot drift from the selector's own decision — and any candidate the selector did
  *not* convert is by definition an FP16 layer of the pack.

`protect_<stem>.json`:

```json
{ "source": "<pack path>",
  "candidates": 788, "converted_by_v31": 717,
  "protected": ["...", "..."], "protected_count": 73,
  "protected_payload_mib": 294.84 }        // Σ(numel of protected weights) / 2^20
```

The payload is the selector's own accounting unit: **1 byte per weight element** (the INT8 footprint
the layer would cost if converted). Reference checkpoint: 73 non-boundary protected layers,
**294.84 MiB ≤ 300 MiB**, i.e. the budget was respected.

### 6.1 How the predicate enters the diag pool

With `--artifact v31`, `diag_impact_sdxl.py` runs the predicate as a subprocess and then measures
**only the pack's converted layers**:

```
pool = [ layers listed in the pack's _quantization_metadata.layers ]
       ∩ candidates                     # names not resolvable/eligible are skipped and counted
```

Reference checkpoint: 788 candidates − 73 protected = **715 measured pool layers**. This is the
mechanism that preserves the protection: a layer the predicate kept FP16 is not in the pool, so it can
never be selected by the ranking, so it stays FP16 in the hybrid.

`--artifact <path>` (a pack instead of the literal `v31`) does the same with any pack; `--protect_list
<json|txt>` excludes further names from the measurement; `--limit N` truncates the target list for
debugging. Operational detail: with `--artifact v31` an **existing** pack at the expected path is
reused (`--reuse-pack` is passed automatically) — delete the pack to force a re-run of the selector.

## 7. The reverse conversion (`gen_reverse_int8_sdxl_v1.1.py`)

### 7.1 Ranking and target resolution

1. Read `<impact>.json`, sort the entries by value **ascending** (lowest impact first), drop `NaN`
   entries, strip a trailing `.weight` from the key. Log: `ranked layers available: <N>`,
   `converting: min(K, N) layer(s)`.
2. `K` is a **count of ranked entries**: `K = all` requests the whole pool. Entries that do not convert
   (see 7.2) do not consume a slot, so the final `converted:` count is `≤ K`.
3. Load the base checkpoint into memory (all tensors; RAM ≥ 1× file size) and log the detected UNet key
   prefix (`UNet key prefix: 'model.diffusion_model.'`).
4. Resolve each diag module name to the checkpoint module key (`module_to_sd_key`): try
   `model.diffusion_model.<name>` and `diffusion_model.<name>`.

### 7.2 Skip classes (counted and printed)

| Skip | Printed as | Meaning |
|---|---|---|
| name does not resolve in the checkpoint | `SKIP (not in sd): <name>` | the impact json was not measured on this checkpoint |
| name matches the converter's artifact-era boundary set | `SKIP (boundary layer): <name>` | boundary layers stay FP16 |
| weight is not 2D/4D, or no eligible group size | `SKIP (no eligible group size): <name> in=<n>` | not ConvRot-convertible |

The final summary line is
`converted: <N>, protected-skip: <N>, not-found-skip: <N>, shape-skip: <N>`.

### 7.3 Rotation and quantization (identical kernel to diag)

```
W_rot = rotate(W)            2D: w @ Hᵀ (group-wise)   4D: rotate along in_channels
q, s  = per-output-channel INT8 of W_rot   (rowwise [out,1] / channelwise [out,1,1,1])
stored: q (rotated), s
```

No inverse rotation is applied — the runtime rotates the activation online.

### 7.4 On-disk format of a converted layer

```
<mod>.weight        int8   Linear [out, in] / Conv2d [out, in, kH, kW]    # stored ROTATED
<mod>.weight_scale  f32    Linear [out, 1] / Conv2d [out, 1, 1, 1]
<mod>.comfy_quant   u8     {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}
```

Every other tensor — including the whole protected set and all non-matmul tensors (norms, biases,
embedders) — is copied **unchanged** from the base (same dtype, same bytes). ComfyUI's mixed-precision
ops arm the quantized path per layer from the `.comfy_quant` marker, so one file carries both the
FP16-kept and the ConvRot INT8 layers.

File metadata:

```json
"_quantization_metadata": {"format_version":"1.0","layers":{"<mod>":{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}}},
"hswq_reverse_int8": {"base": "...", "impact": "...", "k_requested": 670, "converted": 670,
                      "groupsize": 256, "bias_correction": true, "bias_applied": 274}
```

The base's own metadata keys are preserved. `k_requested` and `converted` are recorded separately
because skips make them differ.

### 7.5 Size

Reference checkpoint: base **6.94 GB** (decimal) / 6.46 GiB. A converted layer drops roughly
**3.1 MB** on average (a 2D/4D weight goes from 2 bytes/element to 1 byte/element plus a per-channel
f32 scale). Measured: `K = 670` → **4,875,339,410 B (4.88 GB / 4.54 GiB)**. Because `K` counts layers
and not bytes, two artifacts with the same `K` can differ in size — the size difference is exactly the
difference in the *identity* of the converted layers (measured: two 620-layer packs whose conversion
sets differed by 65 layers differed by 182.3 MB, fully accounted for byte-by-byte by those 65 layers).

## 8. Bias correction (`--bias_correction`)

ConvRot INT8 quantizes the **rotated** weight and the runtime rotates the activation online, so the
systematic output shift of a converted layer is the quantization error contracted with the mean rotated
activation:

```
delta[o] = Σ_j  (Ŵ_rot − W_rot)[o, j] · E[x_rot][j]        →  applied as  bias ← bias − delta
```

Collection (`collect_rotated_act_means`):

* `--num_calib_samples` (default 32) passes of the **production sampler**
  (`dpmpp_2m` / `karras` / cfg 7.0 / `--num_inference_steps` 25) with seeds `calib_seed + i`
  (`--calib_seed` default 42 → 42, 43, …).
* Latent: `fix_empty_latent_channels` at `--width/--height`; conditioning: the **default prompt**
  (`masterpiece, best quality, 1girl, solo, standing, simple background`) as positive and the empty
  string as negative. The `--calib_file` list is read and truncated/repeated to `--num_calib_samples`
  entries, but the text that actually runs is the fixed default prompt: the list's own text does not
  enter the correction (the file only has to exist — otherwise `FileNotFoundError`).
* Forward **pre-hooks** on exactly the layers in the conversion plan rotate the incoming activation
  offline (`x_rot = x @ H`, group size of that layer) and accumulate the per-in-channel mean
  (2D: mean over all rows; 4D: mean over `(N, H, W)`); the model is then freed
  (`torch.cuda.empty_cache()`).
* `applied` when the layer has both a mean and an existing `.bias`; otherwise counted as
  `no_bias` (no `.bias` tensor) or `no_act` (no calibration mean).

Verification line: `bias correction: applied=<N>, no_bias=<N>, no_act=<N>`. Expect `applied > 0` and
`no_act = 0`; a non-zero `no_act` means the hooks never reached those layers (usually a module-name
mapping error) and the file is byte-equivalent to the uncorrected one. Bias correction is
**model-dependent** — measure both variants with the gate before shipping. For the reference
checkpoint: `K = 670`, bias on → `applied = 274`.

## 9. The precision reserve (what stays at FP16)

The FP16 set is the **complement of the converted set**, derived from the measurement, plus two
structural exclusions:

1. the **boundary** layers of §4.1, and
2. every tensor that is not a convertible matmul weight (norms, biases, embedders, …).

A pool layer that is not among the K lowest-impact entries keeps its original precision. Together with
the V3.1 protection this is the "precision reserve" that makes a reverse hybrid score above a uniform
INT8 pack at the same size.

**Measured invariant (reference checkpoint).** The predicate's protection is the pack's FP16 matmul
set: **77 layers** (73 non-boundary candidates + 4 boundary: `input_blocks.0.0`, `out.2`,
`label_emb.0.0`, `label_emb.0.2`). In two independent hybrids (K = 620 and K = 670) all 77 were still
FP16 with weights **byte-identical to the base**, and **0** protected layers were converted. The
protection is structural, not coincidental: the pool excludes them.

## 10. The gate

Acceptance is **not** decoded-image SSIM. It is the deterministic per-step latent-trajectory comparison
(`benchmark/sdxl_int8_traj_compare.py`): both models are sampled from **identical noise per seed** and
the per-step latent cosine is compared.

| Metric | Threshold | Meaning |
|---|---|---|
| `final-cos` | — | final-step latent cosine (FP16 vs candidate, same seed) |
| `max-step-drop` | **> 0.05** | that seed is **bifurcated** (sudden trajectory jump = a different picture, not a degradation) |
| `same-image` | **≥ 0.98** | final cosine high enough to call it the same picture |
| **PASS** | **mean ≥ 0.95 AND 0/25 bifurcated** | production gate |

Fixed protocol: 25 random seeds, `--steps 25`, 1024×1024, cfg 7.0, `dpmpp_2m` / `karras`. `drifted`
(final-cos < 0.98) is normal small divergence and **not** a failure.

**Finding `K`:** search sequentially (±10, one process at a time). The answer is the **largest `K` that
still passes**; any `K` with a bifurcated seed is rejected even if the mean passes. If a size cap
applies, quality degrades monotonically in `K`, so the cap must be compatible with a passing `K` — or
it is not achievable with this method for that checkpoint.

## 11. Measured results (reference checkpoint, `waiIllustriousSDXL_v170`)

25 seeds × 25 steps, 1024×1024, cfg 7.0, dpmpp_2m/karras (gate protocol):

| Artifact | converted | bias | size | mean final-cos | bifurcated |
|---|---|---|---|---|---|
| FP16 baseline | — | — | 6.94 GB | 1.000 (identity) | — |
| native ConvRot INT8 (all convertible layers) | all | — | — | 0.93874 | 0/25 |
| reverse hybrid v1.1, `K = 670` (pool 715, 25-step ranking) | 670 | on | 4,875,339,410 B (4.54 GiB) | **0.96241** | 0/25 |
| reverse hybrid, `K = 620` (earlier reference run) | 620 | off | 4,819,078,807 B (4.49 GiB) | **0.96251** | 0/25 |

Every number is **checkpoint- and condition-specific** and must be re-measured after any change of
checkpoint, candidate set, `K`, ranking conditions or bias-correction setting. Per-model tables:
`benchmark result/benchmark_sdxl_int8.md`.

## 12. End-to-end function map (SDXL)

| Stage | Functions |
|---|---|
| Bootstrap | `setup_comfy` / `import_comfy` (diag), `_setup_comfy` / `_clear_argv_for_comfy` (v1.1) |
| Load | `load_sdxl` / `_load_sdxl` → `comfy.sd.load_checkpoint_guess_config` |
| Candidates | `net.named_modules()` loop; `is_boundary_layer`; `convrot_group_size_for_features` |
| Predicate | `build_protect_list_sdxl.main` → `run_v31` (importlib → `quantize_sdxl_hswq_v3.1.main`); `matmul_modules`; `converted_layers` |
| Kernel | `build_hadamard`, `rotate_weight`, `rotate_weight_conv2d`, `quantize_int8_rowwise`, `quantize_int8_channelwise`, `convrot_int8_quant_error` |
| Trajectory | `build_conditioning`, `make_latent`, `run_trajectory` (`comfy.sample.prepare_noise` / `comfy.sample.sample`) |
| Rank (diag) | `rel_mse`; per-layer inject / restore loop; JSON payload |
| Convert | `parse_args` → rank → `module_to_sd_key` → rotate + per-channel INT8 → `_encode_comfy_quant` → `save_file` |
| Bias (opt.) | `collect_rotated_act_means` (pre-hooks, `rotate_activation_lastdim` / `rotate_activation_nchw`), `compute_bias_delta_rotated` |
| Gate | `benchmark/sdxl_int8_traj_compare.py` |

## 13. Key formulas (symbol ↔ code)

| Symbol | Code / meaning |
|---|---|
| `relMSE(a,b)` | `rel_mse` — `‖a−b‖²_F / ‖b‖²_F` |
| `H` | `build_hadamard(size)` — normalized regular Hadamard (power of 4), `kron(h4) / √size` |
| `W_rot = W·Hᵀ` | `rotate_weight` (2D) / `rotate_weight_conv2d` (4D, in-channel) |
| `q, s` | `quantize_int8_rowwise` (`[out,1]`) / `quantize_int8_channelwise` (`[out,1,1,1]`) |
| `Ŵ_rot = q·s` | `convrot_int8_quant_error` (diag injection) |
| `gs` | `convrot_group_size_for_features(in_dim, 256)` — largest power-of-4 divisor ≤ 256 |
| `impact(l)` | mean over steps of `rel_mse(x_t, x_ref)` |
| `E[x_rot]` | `collect_rotated_act_means` — forward pre-hook, per-in-channel mean of the rotated activation |
| `δb` | `compute_bias_delta_rotated` — `err_rot @ μ_rot` (2D) / `Σ_{H,W} err_rot · μ_rot` (4D); applied as `−δb` on `.bias` |
| pool | `pack._quantization_metadata.layers ∩ candidates` |
| protected | `matmul_modules(base) − converted_layers(pack)` (bare names) |
| gate | `benchmark/sdxl_int8_traj_compare.py` — per-step cosine; bifurcated if `max-step-drop > 0.05` |

## 14. Forbidden mistakes

| Mistake | Why it is wrong |
|---|---|
| Deriving the pool from **another model's** artifact | The pool may only come from the predicate run on the same checkpoint (`--artifact v31`). Another model's layer list excludes layers silently and makes the ranking describe a different model; it is also an unauthorized dependency |
| Reporting a ranking/artifact measured with a different `--steps` as comparable | The ranking is a function of the trajectory; different step counts (and different machines) select different layer sets at the same `K` (§5.4) |
| Expecting two runs on different machines to be byte-identical | The trajectory diverges numerically across GPUs/backends; only the conversion is machine-independent once the layer set is fixed |
| Using a hand-rolled `σ ∈ [0,1]` Euler loop for SDXL | That is a flow-matching schedule; it bypasses `model_sampling` / `calculate_input` and does not match the production trajectory. Use the production sampler |
| Mapping module names as `model.diffusion_model.*` when the model exposes `diffusion_model.*` | Hooks silently never fire (calibration collects nothing → `no_act > 0` → the artifact is byte-equivalent to the uncorrected one) |
| Un-rotating the weight before injection or before writing | The shipped artifact stores **rotated** weights and the runtime rotates the activation online; an inverse rotation models a different layer |
| Treating `K` as a byte budget | `K` counts ranked entries; skips and the layers' individual sizes decide the file size |
| Reusing a stale V3.1 pack | With `--artifact v31` an existing pack at the expected path is reused automatically; a stale pack silently defines the pool. Delete it to re-run the selector |
| Decoding images and reporting SSIM as the gate | The gate is the latent trajectory comparison; SSIM hides bifurcation |
| Reporting a size or score that was not measured, or that belongs to a different candidate set | Rankings, sizes and scores are checkpoint- and condition-specific |
| Concluding from a few seeds | Small seed counts are noisy; always run the full 25-seed set on a quiet GPU |
| Extrapolating the usable `K` across checkpoints | The cliff is a property of the checkpoint; it is found by **measurement** |

## 15. Reference-checkpoint notes

* **Protection payload:** 73 protected non-boundary layers; FP16 payload 589.69 MiB, selector meter
  (1 B/element) **294.84 MiB ≤ 300 MiB**.
* **Pool:** 715 of 788 candidates.
* **Conversion metadata:** `groupsize = 256` for every converted layer on this checkpoint.
* **`conditioner.embedders.0…layers.11`:** the base checkpoint ships 6 tensors whose weights are
  `NaN` in every element (the layer-11 weights of CLIP-L; its biases are finite). They are identical in
  the base and in every derived pack — this pipeline never converts or rewrites conditioner tensors.
  Recorded here so the numbers in the gate are not mistaken for a pipeline defect.

## 16. Related documents

* [Trajectory-Sensitivity Impact Ranking — Technical Guide](diag_impact_trajectory_sensitivity_technical_guide.md) — the universal theory
* [How to quantize SDXL](How%20to%20quantize%20SDXL.md) — CLI contract for this pipeline
* Scripts: `sdxl/diag_impact_sdxl.py`, `sdxl/build_protect_list_sdxl.py`, `sdxl/quantize_sdxl_hswq_v3.1.py`, `sdxl/gen_reverse_int8_sdxl_v1.1.py`, `sdxl/gen_reverse_int8_sdxl.py`
* Gate: `benchmark/sdxl_int8_traj_compare.py`
* Benchmarks: `benchmark result/benchmark_sdxl_int8.md`
