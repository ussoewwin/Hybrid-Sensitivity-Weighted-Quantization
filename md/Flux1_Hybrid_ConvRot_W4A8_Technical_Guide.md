# Reverse-Method Hybrid ConvRot W4A8 (Flux1) — Technical Guide

**Document version:** 1.0
**Date:** 2026-11-21
**Scope:** the Flux1 hybrid ConvRot W4A8 pipeline, as implemented by:

| Stage | Script / Module | Output |
|---|---|---|
| Complete ConvRot INT8 pack | `Flux1/native_convert_int8_convrot_flux.py` | every eligible Linear rotated and packed INT8, rotation stamped per layer |
| Per-layer trajectory impact | `Flux1/diag_impact.py` | `impact/impact_<stem>.json` |
| Reverse hybrid pack builder | `Flux1/gen_reverse_w4a8.py` | `<model>_hybrid_w<K>_convrot_w4a8.safetensors` |
| Complete W4A8 comparison pack | `Flux1/native_convert_w4a8_flux.py` | every eligible Linear packed W4A8, no protection |
| Acceptance gate | `benchmark/flux1_convrot_w4a8_traj_compare.py` | per-seed cosines, verdicts |
| Runtime | ComfyUI core + `ComfyUI-HSWQ-Loader-and-Tools` | ComfyUI load of the pack; the loader adds the LoRA residual |

This document is **Flux1-specific**. The Z-Image reverse-method NVFP4 counterpart is
[Reverse-Method Hybrid ConvRot NVFP4 (Z Image) — Technical Guide](ZImage_Hybrid_ConvRot_NVFP4_Technical_Guide.md),
the SD1.5 forward top-$K$ counterpart is
[Diag Impact → Forward Top-K Hybrid Quantization (SD1.5, v1.1) — Technical Guide](Diag_Impact_SD1.5_v1.1_Technical_Guide.md),
the SDXL reverse hybrid counterpart is
[Diag → Reverse Hybrid Quantization (SDXL, v1.1) — Technical Guide](Diag_Reverse_SDXL_v1.1_Technical_Guide.md),
and the operating procedure is
[How to quantize Flux1 - Hybrid ConvRot W4A8](How%20to%20quantize%20Flux1%20-%20Hybrid%20ConvRot%20W4A8.md).
Measured results are in
[Flux1 ConvRot W4A8 Benchmark Test Results](../benchmark%20result/benchmark_flux1_w4a8.md).

---

## 1. What this pipeline is (one paragraph)

This pipeline builds a **4-bit hybrid** FLUX.1 DiT from a pack that is already quantized:
the input is the **ConvRot INT8 pack** (every converted layer rotated and packed INT8, error
near zero), and the builder moves the **$K$ lowest-impact layers to ConvRot W4A8 in ascending
order of measured trajectory impact**. The ordering comes from `diag_impact.py`, which replaces
one layer at a time with its **true ConvRot INT8 reconstruction in the original basis** and
records the per-step latent divergence of a full sampling trajectory; the score is the mean
relative MSE over the recorded steps. Every measurement substitutes one layer into the
**unquantized FP16 base** and leaves the other layers exact, so the recorded number is that
layer's own error contribution. The output mixes two formats in one file — `asym_w4a8_int8`
for the converted layers, `int8_tensorwise` with the ConvRot stamp for the rest — plus the
boundary layers in the source dtype. ComfyUI core reads the `asym_w4a8_int8` layout from the
pack sidecars, so the file loads in **Load Diffusion Model** / **UNetLoader**; the HSWQ loader
is needed only when a **LoRA** has to be applied, because 16 code levels round a LoRA delta
away and the loader adds it as a factorised residual after the GEMM. Measured on one FLUX.1
checkpoint at $K = 170$ of a 228-layer ranking over the deterministic 25-seed gate, the hybrid
reaches a **mean final trajectory cosine of 0.95383 against 0.93006 for the native all-W4A8
pack**, with **0/25 bifurcated seeds against 1/25** and a mean final latent MSE of **1.1161
against 1.7803** (about 37 % lower error).

---

## 2. Notation & Definitions

| Term | Definition / Meaning |
|---|---|
| **candidate layer** | A ConvRot-eligible 2-D Linear of the FLUX.1 DiT. When a pack is given as the artifact source, the set is that pack's `_quantization_metadata.layers` — read from the artifact, not guessed from module names; without an artifact it is every eligible Linear of the base. Either way it is the set the runtime will arm. |
| **per-step latent $x_t$** | The latent recorded at denoising step $t$ by the sampler callback; the last one is what the VAE would decode. |
| **pristine run $x_{\text{ref}}$** | The trajectory run of the **unquantized FP16 base**. The INT8 pack is the *build* input; the measurement's zero point is the FP16 model. |
| **perturbed run $x_t(l)$** | The trajectory run where candidate $l$'s FP16 weight is replaced by its ConvRot INT8 reconstruction in the original basis, every other layer left at FP16. |
| **trajectory impact $\text{impact}(l)$** | $\dfrac{1}{T}\sum_{t=1}^{T}\dfrac{\lVert x_t(l) - x_{\text{ref},t}\rVert_F^2}{\lVert x_{\text{ref},t}\rVert_F^2}$ — one scalar per candidate layer, the mean over the recorded steps. |
| **reverse order** | Candidates sorted **ascending** by $\text{impact}(l)$; the builder takes the first $K$ as conversion candidates (boundary modules inside them are then skipped, §4). |
| **$K$** | How many lowest-impact layers become W4A8. A per-checkpoint search value, not a method constant. |
| **boundary layer** | A module excluded from conversion regardless of its rank: `img_in`, `txt_in`, `time_in`, `vector_in`, `guidance_in`, `final_layer`, `distilled_guidance_layer`, and anything containing `.img_mod`, `.txt_mod`, `.modulation`, `adaLN` or `norm`. |
| **native W4A8** | Every eligible 2-D Linear converted to `asym_w4a8_int8` with no protected layers — the comparison baseline. "native" is used repo-wide in the sense of plain compression without hswq. |
| **W4A8** | `asym_w4a8_int8`: 4-bit codes with a per-group (16) scale, a per-out-channel scale and a 16-entry Lloyd-Max codebook, applied to the weight rotated for ConvRot group size 256. |
| **gate** | Deterministic 25-seed latent-trajectory comparison: mean final cosine $\ge 0.95$ **and** 0/25 bifurcated. |

---

## 3. Why the reverse direction

### 3.1 Additivity requires a low-error anchor

Quantization error does not compose linearly in general: two layers perturbed together do not
produce the sum of their individual errors once the intermediate state leaves the neighbourhood
where the map is approximately linear. The reverse method buys additivity by measuring against
an **exact anchor**: the measurement substitutes one ConvRot INT8 layer into the FP16 model and
leaves every other layer exact, so the recorded number is that layer's own contribution with no
other error term present. Under that condition $\text{impact}(l)$ behaves as a marginal cost,
and sorting is meaningful.

Forward top-$K$ (SD1.5) uses the same additivity argument from the other side: it starts at FP16
and asks which layers must stay pristine, one at a time. Both are single-layer measurements;
they differ in which end of the precision ladder is held fixed. What neither can do is measure a
**set**: the interaction term is not observed. The pipeline is designed so that the interaction
term stays small enough not to reorder the ranking, and the gate (§9) is what verifies it after
the fact.

### 3.2 Why FP16 cannot be the base for a 4-bit reverse build

A 4-bit target needs the intermediate dequantized float to be as close as possible to the value
the runtime will reconstruct. The INT8 pack already carries the rotation in its stamps
(`convrot`, `convrot_groupsize`), so the builder can dequantize in the rotated domain, un-rotate
once to the physical basis, and hand that float to the W4A8 quantizer, which stores it rotated
again with `convrot_groupsize: 256` for the same group size. Starting from FP16 instead would
require re-deriving the rotation for every layer and re-validating the group size — the same
arithmetic, but with one more place for the rotation domain to disagree with the runtime. A
domain mismatch is not a degraded approximation — it is a **different operation**: the weight is
stored rotated (`convrot_groupsize: 256`) and the runtime rotates the activation for the same
group size, so the GEMM assumes both sides are in the rotated basis. Feed it a weight outside
that basis and the multiply computes a different function of the input; the trajectory numbers
then describe another model, not a quantization error.

### 3.3 Why not "protect the important layers"

A conventional selector ranks layers by static weight statistics (histogram MSE, kurtosis,
outlier ratio, weight-space SVD). Those statistics describe the weight, not the **effect of the
weight's error on the output**, and they are blind to the amplification the sampler performs:
the same relative weight error can leave the terminal latent almost unchanged or destroy it,
depending on where the layer sits in the denoising chain. The reverse pipeline ranks by the
latent states that the sampler actually produces, so the ordering already contains the
amplification.

A second reason is specific to this pipeline: the layers that move to W4A8 leave a pack whose
error already includes the INT8 transition. Ranking from static statistics would ask "which
weight looks compressible", while the question that decides the pack is "which layer can lose
the next four bits without moving the trajectory".

---

## 4. Architecture surface: what is a candidate, and what is protected

FLUX.1 is a dual-stream plus single-stream DiT made of 2-D Linears. Three properties shape this
pipeline:

1. **The candidate set is the artifact's layer list when a pack is given.** `diag_impact.py`
   accepts `--artifact` and reads `_quantization_metadata.layers` from that pack with
   `safetensors.safe_open`, narrowing the measurement to exactly the keys the pack converted.
   Without `--artifact` every ConvRot-eligible Linear of the base is a candidate. A layer that
   the INT8 conversion did not convert is not a W4A8 candidate either — the two formats are
   produced by the same eligibility rule, so the hybrid never contains a layer that has no
   measured ε at the other precision.
2. **Boundary modules are excluded by name, not by rank — and the two scripts apply their own
   lists.** `diag_impact.py` skips a module whose stripped name starts with `img_in`, `txt_in`,
   `time_in`, `vector_in`, `guidance_in`, `final_layer` or `distilled_guidance_layer`, or that
   contains `.img_mod`, `.txt_mod`, `.modulation`, `adaLN` or `norm`. `gen_reverse_w4a8.py` skips a
   key that contains `img_in.`, `txt_in.`, `time_in.`, `vector_in.`, `guidance_in.`,
   `final_layer.`, `mod` or `norm`. The two lists overlap but are not identical: `img_in` without
   the trailing dot and `distilled_guidance_layer` are excluded by the measurer only. That
   asymmetry does not leak into the pack, because the builder works from the ranking the measurer
   produced — a layer the measurer excluded can never reach the builder. These modules sit at the
   interface between the latent, the timestep embedding and the text conditioning; converting
   them changes the scale of the model's inputs and outputs rather than the computation inside
   it.
3. **The group size decides eligibility, and W4A8 needs 256.** The ConvRot group size is the
   largest power-of-4 divisor of `in_features` that is at most 256; a layer with no such divisor
   is not a ConvRot candidate at all. The reverse converter additionally requires the base
   stamp to be ConvRot INT8 with `convrot_groupsize == 256`, because the `comfy_kitchen` W4A8
   CUDA requant path supports that group size only. Layers stamped with a smaller group stay
   ConvRot INT8 in the hybrid.

Prefix handling matters for the same reason: FLUX.1 repacks arrive with `model.diffusion_model.`,
`diffusion_model.` or no prefix at all, so the builder detects the prefix once
(`_find_flux_key_prefix`) from the checkpoint's own signature — `img_in.weight` must be present,
and `double_blocks.0.img_attn.qkv.weight` must be present under the same prefix — and writes
keys with the prefix it found. A boundary check against a hard-coded prefix would silently miss
every module in a differently prefixed checkpoint.

---

## 5. The measurement machinery (`Flux1/diag_impact.py`)

### 5.1 The right quantizer: true ConvRot INT8, not a proxy

```python
def convrot_int8_quant_error(w: torch.Tensor, group_size: int = 256) -> torch.Tensor | None:
    """The value injected for one layer, in the ORIGINAL basis: W_q = dequant(INT8(W H^T)) H.
    n_in = w.shape[1]
    gs = convrot_group_size_for_features(n_in, group_size)
    h = build_hadamard(gs, device=w.device, dtype=torch.float32)
    if w.ndim != 2:
        return None
    w_rot = rotate_weight(w.float(), h, gs)
    q, s = quantize_int8_channelwise(w_rot)
    dq_rot = q.float() * s.float()
    return _unrotate(dq_rot, gs)
```

The block below is the source function, with its docstring truncated after its first line.

The measurement must inject the layer the pack actually ships. The deployed pack stores the
weight rotated and the kernel rotates the activation online, so the layer's effective weight is
`dequant_rotated @ H`, which is `W` up to the INT8 reconstruction error. Injecting that
equivalent weight into the plain FP16 module reproduces exactly the layer the pack ships, and
`ε = W_q − W` is the true ConvRot INT8 error of that layer.

The docstring records the measurement that fixes this choice: on a synthetic 64×512 weight,
`cos(W_q, W) = 0.999973` with the un-rotation and `0.072427` without it. Injecting the rotated
tensor therefore measures a **basis mismatch**, not the quantization, and the ranking it produces
is not a ranking of quantization error.

### 5.2 Trajectory loop and the per-step metric

The loop runs the **production sampler** rather than a hand-rolled schedule, so the measured
ranking follows exactly the trajectory the gate measures:

```python
def run_trajectory(patcher, positive, negative, latent0, *, seed, steps, cfg, sampler, scheduler):
    """Production sampler (comfy.sample.sample) with per-step latent capture."""
    import comfy.sample as comfy_sample

    noise = comfy_sample.prepare_noise(latent0, seed, None)
    xs = []

    def cb(step, x0, x, total_steps):
        xs.append(x.detach().float().cpu())

    with torch.no_grad():
        comfy_sample.sample(
            patcher, noise, steps, cfg, sampler, scheduler,
            positive, negative, latent0, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None,
            callback=cb, disable_pbar=False, seed=seed,
        )
    return xs
```

- **Same seed, same noise.** `prepare_noise` is seeded per run, so two runs that use the same
  seed start from identical `x_0` bit-for-bit, which is what makes a per-layer difference
  meaningful.
- **Real conditioning, not a synthetic tensor.** FLUX.1 needs both text encoders in one
  `load_clip` call with `CLIPType.FLUX`. The pipeline loads `t5xxl` and `clip_l` on **CPU**
  (`load_device`, `offload_device`, `initial_device` all CPU), encodes the positive and negative
  conditioning, moves the conditioning tensors to CPU, and unloads the encoders before the DiT
  runs, so the whole GPU belongs to the diffusion model. The fixed prompt and negative are
  arguments, so a measurement is reproducible from the command line.
- **The metric is the mean over the recorded steps**, not the terminal state alone:

$$\text{impact}(l) = \frac{1}{T}\sum_{t=1}^{T}\frac{\lVert x_t(l) - x_{\text{ref},t}\rVert_F^2}{\lVert x_{\text{ref},t}\rVert_F^2}$$

```python
def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())
```

  The pipeline records every step and scores the mean. That is a deliberate difference from the
  Z-Image pipeline, which scores the terminal latent only: here a layer whose error appears
  mid-trajectory and is partly absorbed later still contributes to the score, and the score
  stays aligned with the per-step divergence the gate prints.

- **Defaults:** `--steps 25`, `--seed 42`, `--width 1024`, `--height 1024`, `--cfg 1.0`,
  `--sampler euler`, `--scheduler simple`, `--groupsize 256`.
- **Injection is restored in `finally`.** `m.weight.data.copy_(qerr)` … `m.weight.data.copy_(w0)`
  so one failed trajectory cannot leak into the next layer's measurement.

**What the measurement does and does not cover.** The substituted weight is the FP16 weight
round-tripped through ConvRot INT8. The shipped pack instead carries that INT8 reconstruction
round-tripped through W4A8, so the pack additionally carries the W4A8 transition of the converted
layers, which this measurement does not include. The ranking therefore measures the **INT8**
transition, while the hybrid's total error is that transition plus the W4A8 transition; the gate
(§9) measures the total. The reverse method's assumption is that a layer that survives the INT8
transition also tolerates the second step better than a layer that does not — an assumption the
gate verifies end to end, not a property proved by the ranking. Keep the two apart when reading a
number: `diag_impact.py` answers "which layer can lose the next four bits", the gate answers "how
good is the finished pack".

### 5.3 Cost

One pristine run plus one trajectory per candidate: the run is **linear in the candidate count**
and the per-layer cost is the full `steps`-step loop. On a 228-layer ranking at 25 steps that is
229 full sampling runs. The step count is not lowered here the way Z-Image lowers it to 4,
because the ranking has to agree with the gate's production trajectory and the gate runs 25
steps; the cost is therefore the price of measuring the same path twice.

### 5.4 JSON contract

```json
{
 "x_ref_norm": 123456.75,
 "steps": 25,
 "seed": 42,
 "cfg": 1.0,
 "sampler": "euler",
 "scheduler": "simple",
 "width": 1024,
 "height": 1024,
 "base": "<absolute path of the FP16 base>",
 "impacts": {
  "model.diffusion_model.<block>.<module>": 0.000123
 }
}
```

- Keys are the **pack's** namespace: `_pack_key` always writes `model.diffusion_model.` + the
  module name, so a builder that strips the prefix finds the same entries either way.
- `x_ref_norm` is the squared Frobenius norm of the pristine terminal latent — the denominator
  used, stored so a reader can reconstruct the absolute error of any entry.
- The header also records the trajectory configuration, so a JSON carries the conditions it was
  measured under instead of relying on a remembered default.
- Layers that failed to run are stored as `nan` rather than dropped (the `except` branch sets
  `imp = float("nan")`), so the key stays visible for inspection. **`nan` has no defined sort
  position**: `sorted()` over a list containing `nan` does not order it deterministically, so a
  failed measurement can land inside the first $K$. Any candidate whose impact is not a finite
  number must be excluded from the conversion set before the build.
- The JSON is **checkpoint-specific**. It is never copied between models: the ranking is a
  property of the weights, the architecture and the trajectory, and only the first survives a
  copy.

---

## 6. The pack builder (`Flux1/gen_reverse_w4a8.py`)

### 6.1 Selection

```python
imp = json.load(open(a.impact, encoding="utf-8"))["impacts"]
ranked = [k[: -len(".weight")] if k.endswith(".weight") else k
          for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
...
for L in ranked[: a.k]:
```

Ascending sort, take the first $K$. No weighting, no threshold, no per-block quota: one scalar
per layer, one cut. The $K$ argument is the whole control surface, which is what makes the search
a one-dimensional sweep. Three filters apply inside the loop, and each is counted and printed:
boundary layers (`n_skip_boundary`), layers without the ConvRot INT8 group-256 stamp
(`n_skip_not_256`), and keys the pack does not carry.

### 6.2 Conversion of one layer

```python
q = sd[wk]            # I8 rotated
s = sd[sk]            # F32 [out,1] (or scalar)
dq_rot = q.float() * s  # W_rot approx in fp32
w_fp32 = _unrotate(dq_rot, _CONVROT_GROUPSIZE)

qdata, s_rel, s_channel, _correction, codebook = ck.quantize_w4a8_int8_weight(
    w_fp32,
    group_size=_QUANT_GROUP_SIZE,
    convrot_groupsize=_CONVROT_GROUPSIZE,
    symmetric=True,
    scale_dtype=torch.float8_e4m3fn,
    codebook=True,
    bits=4,
)
```

- The two constants are fixed in the module header: `_QUANT_GROUP_SIZE = 16` (the W4A8 group,
  the size `comfy_kitchen`'s CUDA fused requant is configured for) and `_CONVROT_GROUPSIZE = 256`
  (the ConvRot group the same path supports).
- The INT8 pair `(q, s)` is dequantized in **float32** and the un-rotation is done in float32
  before the W4A8 quantizer sees the weight: dequantizing in bf16 would round the INT8
  reconstruction before the 4-bit quantizer, adding an error that is not part of either quantization step.
- `_unrotate` is mandatory. The pack stores `W_rot = W H^T`; the W4A8 packer rotates **inside**
  (`quantize_w4a8_int8_weight` is the ConvRot-aware entry point), so handing it `W_rot` would
  give the runtime a weight with two rotations on one side of the GEMM. The converter therefore
  un-rotates first, and the docstring states the failure mode in one line: a double rotation
  yields noise, not a small error.
- `quantize_w4a8_int8_weight` is used rather than a hand-built layout, so the packed nibbles, the
  group scales, the channel scales and the codebook all come from the same class the runtime
  reads.

### 6.3 The layer stamp

```python
conf = {
    "format": "asym_w4a8_int8",
    "group_size": _QUANT_GROUP_SIZE,
    "convrot_groupsize": _CONVROT_GROUPSIZE,
}
sd[base_k + ".comfy_quant"] = torch.frombuffer(
    bytearray(json.dumps(conf, separators=(",", ":")).encode("utf-8")),
    dtype=torch.uint8,
)
meta["layers"][base_k] = conf
meta["layers"][L_stripped] = conf
```

(`_QUANT_GROUP_SIZE` = 16 and `_CONVROT_GROUPSIZE` = 256 in the module header.)

The sidecars written next to the stamp are the ones the runtime decodes: `.weight` (int8, packed
nibbles), `.weight_s_rel` (fp8_e4m3, one per 16-element group), `.weight_s_channel` (float32, one
per output channel) and `.weight_codebook` (float32, 16 entries) when the Lloyd-Max table is used.
`orig_shape` is not written in this stamp because the module owns it:
`MixedPrecisionOps.Linear.__init__` sets `_orig_shape = (out_features, in_features)`, and the
grouped-int8 load path verifies `weight.shape[1] * 8 == _orig_shape[1] * bits` (4 bits for
`asym_w4a8_int8`), raising with the layer name, the packed width and K when they disagree. A pack
whose packed width does not match its declared K therefore fails at load instead of running with a
mis-sized weight; the group count follows from `K / 16` and needs no stamp field either.

The same dict is recorded twice — under the prefixed key and under the stripped key in
`_quantization_metadata`. The metadata block is what `diag_impact.py` reads to enumerate
candidates for a **later** build, so a hybrid pack can itself be used as the layer-list source;
the per-tensor stamp is what the runtime reads.

### 6.4 What is not touched

Layers outside the first $K$, boundary modules, layers without the group-256 stamp, and every key
the builder does not recognize are copied through unchanged (`sd` is loaded from the source and
only modified keys are replaced). A hybrid pack is therefore the INT8 pack with a subset of layers
re-encoded — not a rebuild from scratch. This is also why the INT8 pack's own `_quantization_metadata`
survives: the hybrid carries the union of both layer kinds, which is what makes a second build
from the hybrid possible.

---

## 7. Card 1 bias correction (`Flux1/gen_reverse_w4a8.py --bias_correction`)

### 7.1 The delta is measured against FP16, not the INT8 dequant

A 4-bit weight carries a systematic error, and a Linear layer turns that error into a constant
offset in its output. Card 1 folds the offset back into the layer bias:

$$\text{bias} \mathrel{-}= (W_q - W) \cdot E[x]$$

`W_q` is the quantized weight, `W` the original weight, `E[x]` the per-input-channel mean of that
layer's input. The pack stores `bias − delta`; the runtime adds no extra work.

The reference matters: the reverse converter starts from the INT8 pack, so an INT8-based reference
covers only the INT8 → W4A8 step, while the pack carries both transitions. Passing `--fp16` makes
the converter read the original weight and bias from that file, so the delta is measured against
the weight the model actually had. The measurement on one FLUX.1 checkpoint is recorded in the
how-to: **bias ON 0.95218 against OFF 0.953** — the INT8-referenced correction under-corrects and
measurably loses quality against leaving the bias alone, which is why the option is opt-in, and why the how-to states the delta has to be measured against the FP16 weight rather than the INT8 dequant.

### 7.2 What the correction writes

Only the layers **this run lifts to W4A8** are touched. Layers that stay ConvRot INT8 keep their
bias byte for byte, and boundary layers are never in the candidate set to begin with. The update
is applied to the pack's own bias (`base_bias.float() - delta`) and cast back to the bias dtype,
so a hybrid created from a hybrid does not accumulate two corrections by accident.

The run **aborts instead of saving** when zero layers were corrected:

```python
if bias_applied == 0:
    raise SystemExit(
        "Error: --bias_correction produced 0 corrected layers; refusing to "
        "save a pack labelled bias-corrected"
    )
```

The counters printed at the end (`applied`, `no_act_mean`, `bad_shape`) are part of the record: a
layer with no measured activation mean, or a delta whose shape does not match the bias, is
reported rather than silently skipped.

### 7.3 Calibration set

`--num_calib_samples` defaults to **32** and `--num_inference_steps` to **25**, with
`--latent_size 32`; the run prints a warning when the arguments differ from that recipe, so a
number is never attributed to the standard recipe by assumption. Each sample encodes its prompt
with both text encoders (`CLIPType.FLUX`), draws a bf16 latent of `latent_size × latent_size`, and
sweeps the DiT over `num_inference_steps` timesteps with `guidance = 3.5` and the seed `42 + i`.
The DualMonitor's timestep weight is `1 − t`, so early steps count more than late ones.

The hooks match on **layer shape attributes**, not on the PyTorch class:

```python
if hasattr(module, "in_features") or hasattr(module, "in_channels"):
```

That filter is a measured fix, not a preference. `MixedPrecisionOps.Linear` subclasses
`torch.nn.Module`, has **no `weight` attribute at construction**, and its quantized weight is not
a plain Tensor, so `isinstance(module, torch.nn.Linear)` matched nothing on an INT8 pack and the
DualMonitor collected zero activations (`applied = 0`). Matching on the shape attributes is what
makes the correction reach the quantized layers at all.

### 7.4 What happens without correction

Without `--bias_correction` the pack carries no correction and the systematic offset of the
lifted layers stays in their output. Nothing in the converter changes the INT8 protect layers'
bias in either mode, so the two paths differ only in the W4A8 layers' bias.

The input contract is the other half of this section: **the INT8 pack must stay un-corrected.**
The Stage 1 converter writes no bias correction, so its pack is the right input; feeding a
bias-corrected pack into the reverse converter would subtract a second delta from layers that
already carry one. A hybrid is therefore built from the plain INT8 pack, and the correction is
applied exactly once, on the layers that become W4A8.

---

## 8. Execution paths and how they are selected

| Mode | Computation |
|---|---|
| Core grouped-int8 GEMM (`asym_w4a8_int8`) | `comfy/ops.py` reads the pack sidecars (`.weight_s_rel` / `.weight_s_channel` / `.weight_codebook`), checks the packed width against the layer's declared K and builds `AsymW4A8Int8Layout`; the GEMM multiplies 4-bit weights by INT8 activations. |
| INT8 protect layers (`int8_tensorwise` + `convrot`) | The ConvRot INT8 stack in the HSWQ loader (`patches/comfy_quant_int8`) arms each layer from its own stamp; the activation is rotated online for the layer's group size. |
| LoRA residual (loader only) | The stock GEMM runs on the untouched pack, and the factorised LoRA terms are added after it; the residual stays in the original basis because ConvRot rotates weight and activation together and they cancel. |

- The **pack itself needs no dedicated loader**: core declares and decodes `asym_w4a8_int8`, so
  **Load Diffusion Model** / **UNetLoader** loads the file, and the text encoders and VAE stay
  separate files as usual.
- The **hybrid's INT8 layers** keep the INT8 stack armed. `apply_w4a8_patches` calls
  `apply_comfy_quant_int8_patches()` when the pack also carries `int8_tensorwise` layers, so one
  load arms both formats.
- **A LoRA forces the loader.** A 4-bit weight has 16 code levels; a LoRA delta of roughly
  0.1–0.8 % of amax is below half a code step, so baking the delta into the packed weight rounds
  it away. The loader leaves the pack untouched and adds `((x @ dn^T) @ up^T) * scale` after the
  stock GEMM, with `down [r, in]` and `up [out, r]` kept factorised so no dense `[out, in]` matrix
  is materialised for the whole DiT.
- **Three keys must never be packed.** ComfyUI's FLUX detector reads `shape[1]` of `img_in`,
  `txt_in` and `vector_in.in_layer` to size the architecture; packing any of them halves the K the
  detector reads. The loader refuses such a pack with an explicit message rather than loading a
  model with the wrong width.
- **There is no activation-scale calibration and no Tensor-Core mode here.** NVFP4 needs
  `input_scale` because it multiplies 4-bit activations; W4A8 multiplies **INT8** activations, so
  the path is the grouped-int8 GEMM and there is nothing to calibrate for the GEMM itself. The
  only calibration in this pipeline is Card 1's activation mean, used for the bias correction.

---

## 9. The deterministic 25-seed gate (`benchmark/flux1_convrot_w4a8_traj_compare.py`)

### 9.1 Fixed configuration

| Item | Value |
|---|---|
| Seeds | `42, 137, 849, 2024, 7391, 18429, 53082, 149206, 382715, 826401, 1938502, 4710928, 8391642, 15820493, 36192847, 71058294, 128491703, 285039184, 491730285, 762019483, 938174026, 1409285713, 2683910547, 3851729406, 4195820371` — the script's default `--seeds` value, so a run without `--seeds` is already the gate set |
| Steps | 12 (`--steps` default). The score log of the measured run does not record the step count, so this document states the script default rather than attributing it to that run |
| Resolution | 1024 × 1024 (script default) |
| CFG / sampler / scheduler | 1.0 / euler / simple (Flux guidance comes from `extra_conds`, so cfg 1.0 is the correct setting, not a degraded one) |
| Quantized-side argument | `--w4a8` (aliases `--int8` / `--fp8` / `--quant`) |
| Attention | none patched — the comparator runs the stock attention of both arms |

A seed is the only source of variation: both models start from the noise that seed produces, so
the comparison is paired seed by seed. The seed set, the step count and the resolution are all
**explicit arguments** in a gate run, because each of them changes the numbers; a run that relies
on an unrecorded default is not comparable to the table in §10.

### 9.2 Verdicts

| Verdict | Rule |
|---|---|
| `same-image` | final cosine $\ge$ 0.98 (`SAME_IMG_COS`) and no step exceeded the bifurcation threshold |
| `drifted (different image)` | final cosine < 0.98 with no step exceeding the bifurcation threshold — a coherent picture that is not identical |
| `bifurcated @step N` | one step's cosine drops by more than 0.05 (`BIFURC_DROP`): the trajectory jumps into another attractor, which is a different picture, not a degraded one |

The order of the tests matters: the bifurcation test runs first, so a seed whose final cosine is
above 0.98 but which crossed a discontinuity is reported as `bifurcated @step N`, not as
`same-image`.

Bifurcation is the failure that matters. A run whose final cosine is 0.95 but which crossed a
discontinuity is a failed run; a run whose final cosine is 0.94 without a discontinuity is a
drifted image and may be acceptable. Averages hide this, so the count is reported and gated
separately.

### 9.3 Gate

**PASS = mean final cosine ≥ 0.95 AND 0/25 bifurcated.** Both conditions are required; either one
alone admits configurations that fail in the other. The measured run sits at 0.95383 with 0/25
bifurcated, i.e. it clears the mean by 0.00383 and the bifurcation condition outright — the
margin is small and the pack is described by both numbers together, never by the mean alone.

---

## 10. Measured results (one checkpoint × 25 seeds)

Source: `benchmark result/flux1w4a8.txt`, tabulated in
`benchmark result/benchmark_flux1_w4a8.md`. The two runs in that file are the reverse hybrid at
$K = 170$ and the native all-W4A8 pack.

| Model | Hybrid $K$ | HSWQ mean | Native mean | Δ | HSWQ bifurcated | Native bifurcated |
|---|---|---|---|---|---|---|
| F1D_unlockV3.1off | 170 | **0.95383** | 0.93006 | +0.02377 | **0/25** | 1/25 |

What the table supports, and what it does not:

1. **Bifurcation elimination.** 0/25 for the hybrid against 1/25 for the native pack, on the same
   seed set and the same trajectory configuration. The single native bifurcated seed is
   `762019483`; the hybrid stays coherent on that seed.
2. **Mean cosine separation.** +0.02377 mean final cosine, with the minimum also higher
   (0.80155 against 0.78136) and the maximum higher (0.99135 against 0.98410). The ordering is
   not carried by a few easy seeds.
3. **Latent MSE.** Mean final latent MSE 1.1161 against 1.7803, about 37 % lower error.
4. **Same-image count.** 11/25 for the hybrid against 4/25 for the native pack. The remaining
   hybrid seeds are `drifted`, none bifurcated: at these settings a 4-bit DiT drifts, and the
   pipeline's claim is that it drifts **without** crossing a discontinuity.
5. **Speed.** 2.97 s per seed for the hybrid against 4.71 s for the FP16 reference, +37.0 %, and
   2.94 s for the native W4A8 pack — the two 4-bit arms cost the same, so the fidelity gain is
   not paid for in speed.
6. **What the numbers are not.** They are **not** an image-quality claim. A trajectory cosine
   measures whether two runs produced the same latent; two noise fields would also correlate
   highly. Picture judgement is made by decoding and looking, never from this table.
7. **What this run does not record.** The pack sizes, the step count, the resolution, the cfg, the sampler
   and the scheduler of the measured run are not written in the score log. They are stated as script defaults in
   §9.1 and must be recorded with the next gate run; a size without a gate result, or a gate
   result without a size, is an incomplete record.

---

## 11. Finding $K$

$K$ is checkpoint-specific and the quality surface is **not a single cliff**: which seeds fail
changes with $K$, and quality can recover and fail again. The procedure is a sequential search:

1. Start from a $K$ that leaves a visible INT8 remainder (the measured run uses $K = 170$ of a
   228-layer ranking, so 170 of 228 converted layers become W4A8 and 58 stay ConvRot INT8).
2. Run the full chain (build → optional bias correction → gate).
3. If it passes (mean ≥ 0.95, 0/25 bifurcated), raise $K$ and repeat. The answer is the
   **largest $K$ that still passes**; raising $K$ buys size and costs fidelity, so the last
   passing point is the one to keep.
4. If it fails, lower $K$ until it passes; that $K$ is the answer.
5. Any $K$ with a single bifurcated seed is rejected even when the mean passes.
6. Near an ambiguous boundary, sweep $K \pm 1$–$2$: passing islands exist.

$K$ equal to the ranked set converts every eligible layer and leaves no INT8 remainder; nothing
outside the ranking is touched at any $K$. The repository holds the measured point ($K = 170$)
and the search rule, not a measured curve for this checkpoint — a curve is produced by running
the chain at several $K$ and must be recorded with each point's mean, minimum, bifurcation count
and pack size.

---

## 12. Runtime and loader architecture

- **Pack layout.** `asym_w4a8_int8`: `.weight` int8 nibble-packed `[N, K/2]`,
  `.weight_s_rel` fp8_e4m3 `[N, K/16]`, `.weight_s_channel` float32 `[N]`,
  `.weight_codebook` float32 `[16]`, and `.comfy_quant` carrying
  `{"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256}`.
- **Reader.** ComfyUI core's grouped-int8 branch in `comfy/ops.py` decodes the sidecars and builds
  `AsymW4A8Int8Layout`; nothing about the layout is re-implemented by this repository at load
  time.
- **Rotation.** The stamp (`convrot_groupsize: 256`) makes the runtime rotate the activation
  online for the same group size. Nothing in the pack stores a rotation matrix; the weight is
  stored rotated because that is the basis the GEMM multiplies in.
- **Hybrid arming.** A hybrid pack carries both formats, so the loader arms the INT8 stack in
  addition to the W4A8 path; a pack that is purely W4A8 arms only the grouped-int8 path.
- **LoRA.** The loader's residual forward is the only route that applies a LoRA delta to a 4-bit
  pack; the stock loader does not. The residual lives in the original basis and is added after the
  stock GEMM, so the pack bytes are never rewritten.
- **Shape guard.** Before arming, the loader probes `img_in.weight`, `txt_in.weight` and
  `vector_in.in_layer.weight` for a W4A8 stamp and raises when one is packed, because the FLUX
  detector would otherwise size the architecture from a halved width.
- **Benchmark runtime stack.** The comparator loads the W4A8 `comfy_quant` patches from the
  vendored copy in `benchmark/flux1 convrot w4a8/comfy_quant_w4a8.py` (the folder name contains
  spaces, so the module is loaded by spec-from-path). That vendored patch covers the
  `int8_tensorwise` fallback path for Conv2d weights, and a FLUX.1 pack carries no Conv2d layer
  at all — the native converter's routing table records `Conv2d 4D (absent in Flux1; kept for
  completeness)` — so on FLUX.1 every layer of the pack loads through the core grouped-int8
  branch.
- **ComfyUI-master is untouched.** All patching happens at runtime; the bundled runtime reference
  is read, never edited.

---

## 13. Forbidden mistakes and operational contracts

| Contract | Why |
|---|---|
| **Never mix ranking sources across checkpoints** | The impact JSON is a property of the weights. A ranking measured on another checkpoint reorders the conversion and silently converts the wrong layers. |
| **Never build the hybrid from FP16** | The rotation domain comes from the INT8 pack's stamps; re-deriving it per layer adds a second place for the domain to disagree (§3.2). |
| **Never inject in the rotated basis** | Injecting `W_rot` measures a basis mismatch: 0.072427 cosine against 0.999973 for the un-rotated reconstruction (§5.1). |
| **Never double-rotate in the builder** | The W4A8 packer rotates internally; handing it the rotated INT8 weight yields noise, not a small error (§6.2). |
| **Never pack `img_in` / `txt_in` / `vector_in.in_layer`** | The FLUX detector reads their `shape[1]`; packing halves the K it reads and the model is mis-sized. |
| **Never bias-correct an already-corrected pack** | The reverse converter starts from the INT8 pack; a corrected input subtracts a second delta (§7.4). |
| **Never reference the INT8 dequant for the Card 1 delta** | It covers only the INT8 → W4A8 step; the measured score with that reference is 0.95218 against 0.953 for leaving the bias alone (§7.1). |
| **Never report a mean without the bifurcation count** | A high mean can hide a single discontinuous seed, which is a different picture (§9.2). |
| **Never call trajectory cosine an image-quality measure** | It compares latents; identical noise correlates at 0.99. Quality is judged by decoding and looking. |
| **Never edit `ComfyUI-master/`** | It is the bundled runtime reference; the pipeline patches at runtime only. |
| **One process at a time** | The measurement, the conversion and the gate each hold a full 12B-class DiT; concurrent runs exhaust VRAM and can produce spurious failures. |
| **Never hand-build the W4A8 sidecars** | The layout class the runtime reads is the one that must write the pack; hand-built names drift from the reader. |

---

## 14. Key formulas and function map

$$\text{impact}(l) = \frac{1}{T}\sum_{t=1}^{T}\frac{\lVert x_t(l) - x_{\text{ref},t}\rVert_F^2}{\lVert x_{\text{ref},t}\rVert_F^2} \qquad\text{(mean over the recorded steps, one layer substituted)}$$

$$W_q = \operatorname{dequant}_{\text{INT8}}\!\left(W H^\top\right) H \qquad\text{(the reconstruction injected, in the original basis)}$$

$$\text{order} = \operatorname{argsort}_{\uparrow}\,\text{impact}(l), \qquad \text{candidates} = \text{order}[0:K], \qquad \text{converted} = \text{candidates} \setminus \text{boundary}$$

$$\Delta\text{bias} = (W_q - W)\,E[x], \qquad \text{gate} = \left[\,\overline{\cos_T} \ge 0.95\,\right] \wedge \left[\,\#\{\text{bifurcated}\} = 0 / 25\,\right]$$

| Function / argument | Role |
|---|---|
| `diag_impact.py <base> [impact.json]` | base = FP16 DiT; `--artifact` narrows the candidates to a pack's converted layers; out = impact JSON |
| `--steps` / `--seed` (diag) | trajectory length (default 25) and the single measurement seed (default 42) |
| `convrot_int8_quant_error(w)` | true ConvRot INT8 round-trip in the original basis |
| `_unrotate(w_rot, gs)` | `W = W_rot H` per group; `H` symmetric normalised, so `H⁻¹ = H` |
| `rel_mse(a, b)` | per-step relative MSE, averaged over steps |
| `is_boundary_layer(name)` | the candidate filter (input/output heads, modulation, norms) |
| `gen_reverse_w4a8.py <K> <out> <src> <impact>` | reverse hybrid builder |
| `--fp16` (builder) | FP16 reference for the Card 1 delta; without it the INT8 dequant is used and under-corrects |
| `--bias_correction` (builder) | Card 1 ON; requires `--calib_file`, `--clip_path`, `--clip_l_path`, `--comfy_path` |
| `--num_calib_samples` / `--num_inference_steps` / `--latent_size` | calibration recipe (32 / 25 / 32) |
| `native_convert_w4a8_flux.py` | the all-W4A8 comparison pack; the routing rule is in its header docstring |
| `flux1_convrot_w4a8_traj_compare.py --fp16 --w4a8` | the 25-seed gate; `--seeds` defaults to the gate set |
| `comfy/ops.py` grouped-int8 branch | decodes `asym_w4a8_int8`; the pack's reader |
| `ComfyUI-HSWQ-Loader-and-Tools` `nodes/flux1 convrot w4a8` | LoRA residual + hybrid arming + shape guard |

---

## 15. Reproduction checklist

1. Build the complete ConvRot INT8 pack (`Flux1/native_convert_int8_convrot_flux.py`,
   FULL ConvRot ON by default, group size 256 with the power-of-4 fallback).
2. `Flux1/diag_impact.py <base> impact/impact_<stem>.json --artifact <int8> --clip_path <t5xxl>
   --clip_l_path <clip_l> --comfy_path <ComfyUI>` — never reuse another model's file.
3. `Flux1/gen_reverse_w4a8.py <K> <name> <int8> <impact>` — start from a $K$ that leaves a visible
   INT8 remainder; the measured run uses $K = 170$ of 228.
4. Optional Card 1: add `--bias_correction --fp16 <base> --calib_file <prompts> --clip_path <t5xxl>
   --clip_l_path <clip_l> --comfy_path <ComfyUI> --num_calib_samples 32 --num_inference_steps 25`;
   confirm the printed `applied` count is non-zero (the run aborts otherwise).
5. Gate: `benchmark/flux1_convrot_w4a8_traj_compare.py --fp16 <base> --w4a8 <hybrid>
   --clip_path <t5xxl> --clip_l_path <clip_l> --comfy_path <ComfyUI> --steps 25`; read the footer
   for mean, min, same-image and bifurcated counts.
6. If the gate fails, move $K$ and repeat from 3; reject any $K$ with a bifurcated seed.
7. Record the adopted $K$, the mean, the minimum, the bifurcation count **and** the pack size
   together. A size without a gate result, or a gate result without a size, is an incomplete
   record.
8. Decode and look at the images before making any statement about picture quality.
