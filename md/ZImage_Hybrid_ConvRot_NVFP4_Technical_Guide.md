# Reverse-Method Hybrid ConvRot NVFP4 (Z Image) — Technical Guide

**Document version:** 1.0
**Date:** 2026-11-20
**Scope:** the Z-Image hybrid ConvRot NVFP4 pipeline, as implemented by:

| Stage | Script / Module | Output |
|---|---|---|
| Complete ConvRot INT8 pack | `Z_Image/native_convert_int8_convrot_zi.py` | all converted layers INT8, rotation stamped per layer |
| Per-layer trajectory impact | `Z_Image/diag_impact.py` | `impact/impact_<stem>.json` |
| Reverse hybrid pack builder | `Z_Image/gen_reverse_nvfp4.py` | `<model>_hswq_hybrid_nv<K>_convrot_nvfp4.safetensors` |
| `input_scale` calibration | `Z_Image/calib_input_scale_nvfp4.py` | `<model>_hswq_hybrid_nv<K>_convrot_nvfp4_calib.safetensors` |
| Acceptance gate | `benchmark/zi_convrot_nvfp4_traj_compare.py` | per-seed cosines, verdicts, gate verdict |
| Runtime loader | `ComfyUI-HSWQ-Loader-and-Tools` | ComfyUI load of the pack |

This document is **Z-Image-specific**. The SD1.5 forward top-$K$ counterpart is
[Diag Impact → Forward Top-K Hybrid Quantization (SD1.5, v1.1) — Technical Guide](Diag_Impact_SD1.5_v1.1_Technical_Guide.md),
the SDXL reverse hybrid counterpart is [Diag → Reverse Hybrid Quantization (SDXL, v1.1) — Technical Guide](Diag_Reverse_SDXL_v1.1_Technical_Guide.md),
and the operating procedure is [How to create Hybrid NVFP4 from ConvRot INT8 (Z Image, Reverse Method)](How%20to%20quantize%20Z%20Image%20-%20Hybrid%20NVFP4.md).
Measured results are in [Z-Image ConvRot Hybrid NVFP4 Benchmark Test Results](../benchmark%20result/benchmark_zi_nvfp4.md).

---

## 1. What this pipeline is (one paragraph)

This pipeline builds a **4-bit hybrid** Z-Image DiT from a pack that is already quantized:
the input is the **ConvRot INT8 UNet** (every converted layer rotated and packed INT8, error
near zero), and the builder converts the **$K$ lowest-impact layers to NVFP4 in ascending
order of measured trajectory impact**. The ordering comes from `diag_impact.py`, which replaces
one layer at a time with its **true NVFP4 round-trip reconstruction** (E2M1 4-bit codes in
16-element blocks with a global scale, the same `comfy_kitchen` kernel that produces the
shipped artifact) and measures the terminal latent divergence $x_T$ of a full denoising
trajectory. Every measurement substitutes one layer into the **unquantized FP16 base** and leaves
the other layers exact, so the recorded number is that layer's own error contribution. The output
mixes two formats in one file — NVFP4 for the converted layers, ConvRot INT8 for the rest — and
loads in ComfyUI, where the `nvfp4` format is declared in core and the ConvRot activation
rotation for it is supplied by the HSWQ loader. When the
pack is calibrated (`input_scale` written per layer), the Blackwell Tensor Core W4A4 path runs;
without calibration the loader falls back to the parity (W4A16) path. Measured on 11 Z-Image
checkpoints and 220 seed runs, the hybrid reaches a **mean final trajectory cosine of
0.96025–0.97366 against 0.90049–0.94680 for native all-NVFP4**, with **0/20 bifurcated seeds on
every checkpoint** (native bifurcates on 4 seeds spread over 3 of the 11 checkpoints).

---

## 2. Notation & Definitions

| Term | Definition / Meaning |
|---|---|
| **candidate layer** | A ConvRot-eligible module of the Z-Image DiT that appears in the complete INT8 pack's `_quantization_metadata.layers`. The candidate set is read from the artifact, not guessed from module names, so it is exactly the set the runtime will arm. |
| **terminal latent $x_T$** | The latent produced after the last Euler step of the trajectory loop; the tensor the VAE would decode. |
| **pristine run $x_{\text{ref}}$** | The trajectory run of the **unquantized FP16 base** — `diag_impact.py` loads `a.base` with `is_nvfp4=False` (L511). The INT8 pack is the *build* input; the measurement's zero point is the FP16 model. |
| **perturbed run $x_T(l)$** | The trajectory run where candidate $l$'s FP16 weight is replaced by its NVFP4 round-trip reconstruction, every other layer left at FP16. |
| **trajectory impact $\text{impact}(l)$** | $\dfrac{\lVert x_T(l) - x_{\text{ref}}\rVert_F^2}{\lVert x_{\text{ref}}\rVert_F^2}$ — one scalar per candidate layer. |
| **reverse order** | Candidates sorted **ascending** by $\text{impact}(l)$; the builder converts the first $K$. |
| **$K$** | How many lowest-impact layers become NVFP4. A per-checkpoint search value, not a method constant. |
| **boundary layer** | A module excluded from conversion regardless of its rank: `cap_embedder`, `x_embedder`, `final_layer`, `norm_out`, `t_embedder`, `context_embedder`. |
| **native NVFP4** | The whole model converted to NVFP4 with no protection set — the comparison baseline, and the definition of "native" used repo-wide. |
| **TC (W4A4)** | Blackwell Tensor Core GEMM: 4-bit weights against 4-bit activations scaled by the calibrated `input_scale`. |
| **parity (W4A16)** | Stock ComfyUI GEMM in the compute dtype with the online activation rotation; 4-bit weights are dequantized for the multiply. |
| **gate** | Deterministic 20-seed latent-trajectory comparison: mean final cosine $\ge 0.95$ **and** 0/20 bifurcated, measured in the SA2 + TC production configuration. |

---

## 3. Why the reverse direction

### 3.1 Additivity requires a low-error anchor

Quantization error does not compose linearly in general: two layers perturbed together do not
produce the sum of their individual errors once the intermediate state leaves the neighbourhood
where the map is approximately linear. The reverse method buys additivity by measuring against
an **exact anchor**: the measurement substitutes one NVFP4 layer into the FP16 model and leaves
every other layer exact, so the recorded number is that layer's own contribution with no other
error term present. Under that condition $\text{impact}(l)$ behaves as a marginal cost, and
sorting is meaningful.

Forward top-$K$ (SD1.5) uses the same additivity argument from the other side: it starts at FP16
and asks which layers must stay pristine, one at a time. Both are single-layer measurements;
they differ in which end of the precision ladder is held fixed. What neither can do is measure a
**set**: the interaction term is not observed. The pipeline is designed so that the interaction
term stays small enough not to reorder the ranking, and the gate (§9) is what verifies it after
the fact.

### 3.2 Why FP16 cannot be the base for a 4-bit reverse build

A 4-bit target needs the intermediate dequantized float to be as close as possible to the value
the runtime will reconstruct. The INT8 pack already carries the rotation in its stamps
(`convrot`, `convrot_groupsize`), so the builder can dequantize in the rotated domain and hand
the rotated float to the NVFP4 quantizer, which stores it with `convrot: true` for the same
group size. Starting from FP16 instead would require re-deriving the rotation for every layer and
re-validating the group size — the same arithmetic, but with one more place for the rotation
domain to disagree with the runtime. The measured failure mode of a domain mismatch is not a
small error: an un-rotated weight against a rotated activation multiplies to near-orthogonal
vectors and the output is noise, not a degraded picture.

### 3.3 Why not "protect the important layers"

A conventional selector ranks layers by static weight statistics (histogram MSE, kurtosis,
outlier ratio, weight-space SVD). Those statistics describe the weight, not the **effect of the
weight's error on the output**, and they are blind to the amplification the sampler performs:
the same relative weight error can leave the terminal latent almost unchanged or destroy it,
depending on where the layer sits in the denoising chain. The reverse pipeline ranks by the
terminal state that the sampler actually produces, so the ordering already contains the
amplification.

---

## 4. Architecture surface: what is a candidate, and what is protected

Z-Image is a NextDiT-style single-stream DiT. Two properties shape this pipeline:

1. **The candidate set is the artifact's layer list.** `diag_impact.py` reads
   `_quantization_metadata.layers` from the complete INT8 pack with `safetensors.safe_open` and
   measures exactly those keys (L528-531). A layer that the INT8 conversion did not convert is
   not a candidate for NVFP4 either — the two formats are produced by the same eligibility rule,
   so the hybrid never contains a layer that has no measured ε at another precision.
2. **Boundary modules are excluded by name, not by rank.** `gen_reverse_nvfp4.py` skips any
   candidate whose stripped key contains `cap_embedder`, `x_embedder`, `final_layer`, `norm_out`,
   `t_embedder` or `context_embedder` (L89-92). These sit at the interface between the latent,
   the timestep embedding and the text conditioning; converting them changes the scale of the
   model's inputs and outputs rather than the computation inside it, and no amount of fidelity in
   the blocks compensates for a mis-scaled embedding.

Prefix handling matters for the same reason: Z-Image checkpoints come with different top-level
key prefixes across repacks, so the builder detects the prefix once (`_find_prefix`) and strips
it for boundary matching, then re-attaches it when writing keys (L82-102). A boundary check
against the raw key would silently miss every module in a prefixed checkpoint.

---

## 5. The measurement machinery (`Z_Image/diag_impact.py`)

### 5.1 The right quantizer: true NVFP4, not a proxy

```python
def nvfp4_quant_error(w):
    """TRUE NVFP4 quantization error via comfy_kitchen roundtrip
    (E2M1 x 16-element blocks + global scale): exactly the kernel that
    produces the shipped artifact. The old e4m3-per-256 proxy understates
    the error ~13x and flattens the ranking; do not fall back to it."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as _NVFP4
    w2 = w if w.is_contiguous() else w.contiguous()
    qdata, params = _NVFP4.quantize(w2)
    return _NVFP4.dequantize(qdata, params)
```

The measurement must use the same quantizer the artifact will be built with. An
E4M3-per-256 simulation is a different quantization: its error magnitude is roughly an order of
magnitude smaller and its error is distributed more evenly, which **flattens the ranking** — the
top of the list stops separating from the middle, and the $K$ cut lands on noise. The docstring
records the measured ratio (~13×).

### 5.2 Trajectory loop and the terminal-state metric

The loop is deliberately small and explicit so that the measurement is reproducible bit-for-bit
on the same GPU:

```python
def run():
    x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16,
                    generator=torch.Generator(device).manual_seed(seed))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    with torch.no_grad():
        for step in range(steps):
            out = model(x, sigmas[step:step + 1], embeds, None, attention_mask=None)
            if isinstance(out, tuple):
                out = out[0]
            x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.float16)
    return x
```

- **Explicit Euler** with a uniform sigma ladder from 1.0 to 0.0 in `steps + 1` points. The step
  is `sigmas[t+1] - sigmas[t]`, so the loop integrates the model's velocity directly; there is no
  scheduler table to keep in sync with the production sampler.
- **Fixed initial noise per seed** from a device-side generator: two runs that use the same seed
  start from identical $x_0$ bit-for-bit, which is what makes a per-layer difference meaningful.
- **A synthetic conditioning tensor**: `embeds = torch.randn(1, 256, 2560)`. The ranking is a
  property of the weights and the trajectory, not of a prompt; a fixed synthetic embedding
  removes prompt selection from the measurement and keeps it single-shot. Real prompts are used
  later, in calibration (§7) and in the gate (§9).
- **Default resolution** `1×16×128×128` latent (the Z-Image latent shape), so no VAE and no
  image-space metric is involved anywhere in the pipeline.
- **Defaults:** `--steps 4`, `--seed 42` (`parse_args`, L453-464).

The metric is the **terminal** state only:

$$\text{impact}(l) = \frac{\lVert x_T(l) - x_{\text{ref}}\rVert_F^2}{\lVert x_{\text{ref}}\rVert_F^2}$$

```python
def rel_mse(a, b):
    a = a.float().reshape(a.shape[0], -1)
    b = b.float().reshape(b.shape[0], -1)
    return float(((a - b) ** 2).sum() / (b ** 2).sum())
```

A step-by-step average would mix the early steps — where the latent is dominated by noise and
every layer's error looks similar — into the final number, diluting the layers whose error
actually reaches the decoded image. The last state is what the VAE sees, so it is what the metric
scores.

**What the measurement does and does not cover.** The substituted weight is the FP16 weight
round-tripped through NVFP4. The shipped pack instead carries the **INT8 reconstruction**
round-tripped through NVFP4, so the pack additionally carries the INT8 dequantization error of
the converted layers, which this measurement does not include. The ranking therefore measures
the 4-bit transition alone, while the hybrid's total error is that transition plus the INT8
error; the gate (§9) measures the total. Keep the two apart when reading a number:
`diag_impact.py` answers "which layer should lose the most bits", the gate answers "how good is
the finished pack".

### 5.3 Cost

One pristine run plus one trajectory per candidate: the run is **linear in the candidate count**
and the per-layer cost is the full `steps`-step loop. That is why `--steps` defaults to 4 rather
than the 12 used by the gate: the ranking only needs the ordering to be stable, and the gate
re-measures the finished pack at production settings.

### 5.4 JSON contract

```json
{
  "x_ref_norm": 123456.75,
  "impacts": {
    "model.diffusion_model.<block>.<module>": 0.000123
  }
}
```

- Keys are the **artifact's** key form, with the `model.diffusion_model.` prefix kept when present;
  the builder strips the prefix itself (§6).
- `x_ref_norm` is the squared Frobenius norm of the pristine terminal latent — the denominator
  used, stored so a reader can reconstruct the absolute error of any entry.
- Layers that failed to run are stored as `nan` rather than dropped (L565-569), so the key stays
  visible in the file for inspection. **`nan` has no defined sort position**: `sorted()` over a
  list that contains `nan` does not order it deterministically, so a failed measurement can land
  inside the first $K$. Any candidate whose impact is not a finite number must be excluded from
  the conversion set before the build — check the JSON for non-finite entries instead of trusting
  the order the builder produces.
- The JSON is **checkpoint-specific**. It is never copied between models: the ranking is a
  property of the weights, the architecture and the trajectory, and only the first survives a
  copy.

---

## 6. The pack builder (`Z_Image/gen_reverse_nvfp4.py`)

### 6.1 Selection

```python
ranked = [k[:-len(".weight")] if k.endswith(".weight") else k
          for k, _ in sorted(imp.items(), key=lambda kv: kv[1])]
...
for L in ranked[:a.k]:
```

Ascending sort, take the first $K$. No weighting, no threshold, no per-block quota: one scalar
per layer, one cut. The $K$ argument is the whole control surface, which is what makes the search
in §11 a one-dimensional sweep.

### 6.2 Conversion of one layer

```python
q = sd[wk]            # I8 rotated
s = sd[sk]            # F32 [out,1]
dq = (q.float() * s)  # W_rot approx in fp32
w_for_q = dq.to(torch.bfloat16)
qdata, params = TensorCoreNVFP4Layout.quantize(w_for_q)
tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
```

- The INT8 pair `(q, s)` is dequantized in **float32** and only then narrowed to bf16 for the
  NVFP4 quantizer: dequantizing in bf16 would round the INT8 reconstruction before the 4-bit
  quantizer sees it, adding an error the pipeline never intended.
- The value handed to `quantize` is the **rotated** weight. The stamp written in the next block
  says so (`"convrot": True`, `"convrot_groupsize": 256`), and the runtime rotates the activation
  for the same group size. This is the one invariant that must hold across the builder, the
  calibration and the loader; §3.2 explains why a mismatch is catastrophic rather than marginal.
- `state_dict_tensors` is used rather than hand-built names so the packed layout, the block scales
  and the global scale all come from the same layout class the loader will read.

### 6.3 The layer stamp

```python
conf = {
    "format": "nvfp4",
    "convrot": True,
    "convrot_groupsize": 256,
    "orig_shape": [int(dq.shape[0]), int(dq.shape[1])],
    "in_features": int(dq.shape[1]),
    "out_features": int(dq.shape[0]),
}
sd[base_k + ".comfy_quant"] = torch.frombuffer(
    json.dumps(conf).encode("utf-8"), dtype=torch.uint8
).clone()
```

`orig_shape` is not decoration: packed NVFP4 storage is narrower than the logical width, and the
loader sizes the module from the stamp. A missing or wrong `orig_shape` produces a module with
the packed width, which fails at the GEMM rather than at load.

The same dict is recorded twice — under the prefixed key and under the stripped key in
`_quantization_metadata` (L124-125). The metadata block is what `diag_impact.py` reads to
enumerate candidates for a **later** build, so a hybrid pack can itself be used as the layer-list
source; the per-tensor stamp is what the runtime reads.

### 6.4 What is not touched

Layers outside the first $K$, boundary modules, and every key the builder does not recognize are
copied through unchanged (`sd` is loaded from the source and only modified keys are replaced).
A hybrid pack is therefore the INT8 pack with a subset of layers re-encoded — not a rebuild from
scratch.

---

## 7. `input_scale` calibration (`Z_Image/calib_input_scale_nvfp4.py`)

### 7.1 Rotate first, then take the amax

The header states the contract:

> The rotation MUST happen before amax ("rotate first, then amax"): the hybrid pack's weights are
> in the rotated domain, so an amax measured on unrotated activations is in the wrong basis and
> mis-scales the grid.

Concretely, each tracked module's input is rotated with the same Hadamard matrix and group size
as its weight, and the running maximum is taken over the **rotated** activation. A group size
that cannot be formed for a layer is reported and the rotation is skipped for it
(`WARN: no valid Hadamard group ... (unrotated amax)`) rather than silently using a wrong-basis
value.

### 7.2 The scale

```python
from comfy_kitchen.float_utils import F4_E2M1_MAX, F8_E4M3_MAX
denom = float(F8_E4M3_MAX) * float(F4_E2M1_MAX)
...
sd[f"{full}.input_scale"] = torch.tensor(
    max(v["amax"], 1e-12) / denom, dtype=torch.float32
)
```

$$\text{input\_scale} = \frac{\max(\text{amax},\,10^{-12})}{448 \times 6} = \frac{\text{amax}}{2688}$$

The denominator is the product of the two representable maxima of the two-level scale scheme: an
E4M3 block scale spanning the FP4 grid. Dividing by it maps the observed amax onto the full
representable range of the 4-bit code, which is what the W4A4 kernel expects.

### 7.3 Calibration set

`--samples` defaults to 32 trajectories, `--steps` to 4, with `--prompts` accepting a UTF-8 file
of one prompt per line (a synthetic set is used when omitted). Coverage is printed as the run
proceeds (`amax coverage: n/total`), and a layer whose amax never rose above zero is reported as
missing (L689) — an uncalibrated layer is a fact to be stated, not a layer to be left with a
placeholder.

### 7.4 What happens without calibration

The loader decides the GEMM mode from the presence of `*.input_scale` keys (§8). A pack without
them loads on the parity path and is validated there. Forcing TC on an uncalibrated pack is the
one configuration that produces a meaningless result: the activations are scaled against a
placeholder and the measured cosine collapses (~0.18) — the failure is in the scale, not in the
weights. Every gate run therefore states its GEMM mode, and the benchmark prints
`[HSWQ NVFP4] GEMM MODE: TC (W4A4 scaled_mm)` or
`[HSWQ NVFP4] GEMM MODE: PARITY (W4A16 dequant GEMM)` (L566-573); a third value,
`mixed/unexpected (tc_hits=…, parity_fwd=…)`, appears when both paths fired. The
configuration of a number is therefore never inferred from memory.

---

## 8. GEMM modes and how they are selected

| Mode | Computation | Selected when |
|---|---|---|
| **TC (W4A4)** | Blackwell block-scaled FP4 GEMM (`scaled_mm_nvfp4_pooled` → `_C.cublas_gemm_blockwise_fp4`), activations quantized with `input_scale` | the checkpoint carries `*.input_scale` **and** the GPU compute capability is ≥ 10.0 |
| **Parity (W4A16)** | stock ComfyUI GEMM in the compute dtype plus the online activation rotation; the 4-bit weight is dequantized for the multiply | no `input_scale` keys, or the TC path is unavailable |

- Auto-selection is a property test on the checkpoint, not a user setting: if any
  `*.input_scale` key exists, the TC path is armed.
- **The TC path also requires the hardware.** `nvfp4_tc_gate.probe_nvfp4_tc_support()` checks the
  compute capability against `(10, 0)` — `comfy_kitchen`'s `scaled_mm_nvfp4` minimum — and the
  first `CUBLAS_STATUS_NOT_SUPPORTED` disables TC for the rest of the process. On Ada / Hopper /
  Ampere the NVFP4 GEMM is unavailable, and the run is a parity run even when the pack is
  calibrated and `--tc` was passed. A gate number is only a TC number on a Blackwell GPU.
- Overrides exist for A/B work only: `--tc` / `--parity` on the benchmark, and the environment
  variables `HSWQ_ZI_FORCE_PARITY=1` (takes precedence) / `HSWQ_ZI_FORCE_TC=1`
  (`ComfyUI-HSWQ-Loader-and-Tools`, `nodes/zimage_convrot_nvfp4/load_unet.py` L64-69).
- **The gate is measured in TC.** A parity number describes a different computation and is not a
  substitute; the native baseline is parity by construction (a full-NVFP4 pack has no calibrated
  scales), which is why the cross-model table in §10 always compares TC-hybrid against
  parity-native and states both.

---

## 9. The deterministic 20-seed gate (`benchmark/zi_convrot_nvfp4_traj_compare.py`)

### 9.1 Fixed configuration

| Item | Value |
|---|---|
| Seeds | `42, 1337, 7, 2024, 555, 43, 1458, 9, 2026, 777, 44, 1338, 8, 2028, 888, 46, 1587, 12, 2047, 222` — pass `--canonical-seeds`, the flag that selects the 20 canonical seeds (L486-489). The plain `--seeds` default is only `42,1337,7,2024,555` (5 seeds), so a run without `--canonical-seeds` is not a gate run. |
| Steps | 12 (pass `--steps 12`; the script default is 25) |
| Resolution | 1024 × 1024 |
| CFG / sampler / scheduler | 2.5 / euler / simple |
| Attention | SA2 — pass `--attention sage2` (INT8 QK + FP8 PV, sm120 CUDA path). The default is `sdpa`, i.e. no attention patch, and the patch applies to the **quantized** model only; the FP16 baseline keeps baseline attention (L509-516). |
| GEMM | TC — pass `--tc` on the calibrated hybrid; `--parity` forces the other path. |
| Pack argument | `--hybrid` (aliases `--quant`, `--nvfp4`) takes the hybrid pack path (L472-475) |

A seed is the only source of variation: both models start from the noise that seed produces, so
the comparison is paired seed by seed. The seed set, the step count and the attention mode are
all **explicit arguments** in a gate run, because each of them changes the numbers; a run that
relies on a default is not comparable to the table in §10.

### 9.2 Verdicts

| Verdict | Rule |
|---|---|
| `same-image` | final cosine ≥ 0.98 (`SAME_IMG_COS`, L670) and no step exceeded the bifurcation threshold |
| `drifted (different image)` | final cosine < 0.98 with no step exceeding the bifurcation threshold — a coherent picture that is not identical |
| `bifurcated @step N` | one step's cosine drops by more than 0.05 (`BIFURC_DROP`, L669): the trajectory jumps into another attractor, which is a different picture, not a degraded one |

The order of the tests matters: the bifurcation test runs first, so a seed whose final cosine is
above 0.98 but which crossed a discontinuity is reported as `bifurcated @step N`, not as
`same-image` (L693-698).

Bifurcation is the failure that matters. A run whose final cosine is 0.95 but which crossed a
discontinuity is a failed run; a run whose final cosine is 0.94 without a discontinuity is a
drifted image and may be acceptable. Averages hide this, so the count is reported and gated
separately.

### 9.3 Gate

**PASS = mean final cosine ≥ 0.95 AND 0/20 bifurcated**, measured in SA2 + TC. Both conditions are
required; either one alone admits configurations that fail in the other.

---

## 10. Measured results (11 checkpoints × 20 seeds)

Source: `benchmark result/score_zi_nvfp4.txt`, tabulated in
`benchmark result/benchmark_zi_nvfp4.md`.

| Model | Hybrid $K$ | HSWQ mean | Native mean | Δ | HSWQ bifurcated | Native bifurcated |
|---|---|---|---|---|---|---|
| moodyWildMixZIBZID_v02 | 99 | **0.96546** | 0.93505 | +0.03041 | **0/20** | 1/20 |
| moodyProMix_collectorsEdition | 100 | **0.96158** | 0.92672 | +0.03486 | **0/20** | 0/20 |
| moodyRealMix_xhsEdition | 100 | **0.97366** | 0.94680 | +0.02686 | **0/20** | 0/20 |
| zimageTurboByStable_2602 | 99 | **0.96242** | 0.92187 | +0.04055 | **0/20** | 0/20 |
| unstableRevolution_V3Fp16 | 100 | **0.96216** | 0.92187 | +0.04029 | **0/20** | 0/20 |
| gonzalomoZpop_insta2 | 100 | **0.96554** | 0.92187 | +0.04367 | **0/20** | 0/20 |
| divingZImageTurbo_v70Fp16 | 99 | **0.96107** | 0.93218 | +0.02889 | **0/20** | 0/20 |
| beyondREALITY_V30 | 100 | **0.96025** | 0.90049 | +0.05976 | **0/20** | 2/20 |
| 2127ZImageAsianUtopian_v40Turbo | 100 | **0.96558** | 0.91271 | +0.05287 | **0/20** | 1/20 |
| copaxTimeless_xplusZ13 | 100 | **0.96408** | 0.94334 | +0.02074 | **0/20** | 0/20 |
| unstablebastard_v14 | 100 | **0.96382** | 0.92822 | +0.03560 | **0/20** | 0/20 |

What the table supports, and what it does not:

1. **Bifurcation elimination.** 0/20 on every checkpoint — 0 bifurcated seeds in 220 runs —
   against 4 native bifurcated seeds spread over 3 checkpoints (`beyondREALITY_V30` 2/20,
   `moodyWildMixZIBZID_v02` 1/20, `2127ZImageAsianUtopian_v40Turbo` 1/20). This is the
   strongest statement in the table because the count is discrete.
2. **Mean cosine separation.** The hybrid's mean sits 0.02074–0.05976 above native on every
   checkpoint, and no checkpoint inverts the ordering.
3. **Latent MSE.** Mean final latent MSE falls by roughly 40–60% relative to native across the
   set (e.g. `moodyWildMixZIBZID_v02` 0.8584 against 1.6914).
4. **Worst case.** The worst seed in the whole set is `6802579145` on
   `2127ZImageAsianUtopian_v40Turbo`: native reaches a final cosine of **0.54584** (bifurcated at
   step 10) while the hybrid reaches **0.79572** on the same seed. The hybrid's per-checkpoint
   minima therefore span 0.79572–0.94122; a mean hides that tail, which is why the minimum is
   listed per checkpoint in the benchmark report.
5. **What the numbers are not.** They are **not** an image-quality claim. A trajectory cosine
   measures whether two runs produced the same latent; two noise fields would also correlate
   highly. Picture judgement is made by decoding and looking, never from this table.

`same-image` counts are low across the set: counting the verdict column of the benchmark report's
per-seed rows gives **60 of 220 runs (27.3%)** — the other 160 are `drifted`, and none are
bifurcated. A 4-bit DiT at these settings drifts, and the pipeline's claim is that it drifts
**without** crossing a discontinuity, not that it is indistinguishable.

---

## 11. Finding $K$

$K$ is checkpoint-specific and the quality surface is **not a single cliff**: which seeds fail
changes with $K$, and quality can recover and fail again. The procedure is a sequential search:

1. Start at $K = 90$ and run the full chain (build → calibrate → gate).
2. If it passes (mean ≥ 0.95, 0/20 bifurcated), raise $K$ by 10 and repeat. The answer is the
   **largest $K$ that still passes**.
3. If it fails, lower $K$ by 10 until it passes; that $K$ is the answer.
4. Any $K$ with a single bifurcated seed is rejected even when the mean passes.
5. Near an ambiguous boundary, sweep $K \pm 1$–$2$: passing islands exist.

Reference curve for one checkpoint (`moodyProMix_collectorsEdition`, 20 seeds × 12 steps, measured
before the SA2 option existed, so **without** SA2 acceleration — kept as the TC-vs-parity ordering
only, and re-measured in SA2 + TC before shipping):

| Pack | $K$ | mean | min | max | same-image | bifurcated | GEMM |
|---|---|---|---|---|---|---|---|
| nv90 | 90 | 0.95852 | 0.82504 | 0.98816 | 5/20 | 0/20 | TC (W4A4) |
| **nv100 (adopted)** | **100** | **0.96033** | **0.87123** | **0.98680** | **4/20** | **0/20** | **TC (W4A4)** |
| native | 180 | 0.91079 | 0.59021 | 0.96337 | 0/20 | **1/20** | PARITY (W4A16) |

The curve shows the shape the search has to handle: raising $K$ from 90 to 100 bought +0.0018 mean
while the minimum rose +0.046, and the next step is where failures reappear. $K$ is chosen at the
point where the mean stops paying for the extra bytes, and the minimum (the tail a user will hit)
is part of the decision, not a footnote.

---

## 12. Runtime and loader architecture

- **Layout.** NVFP4 storage is read through `comfy_kitchen`'s `TensorCoreNVFP4Layout`; the packed
  tensor plus its block scales and global scale reconstruct the rotated weight.
- **Runtime stack.** The benchmark runs the pipeline's own stack in `benchmark/hswq_stack/`
  (`zimage_nvfp4/`: `zi_nvfp4_forward.py`, `nvfp4_addmm_patch.py`, `nvfp4_comfy_parity.py`,
  `nvfp4_tc_gate.py`, `nvfp4_lora_bake.py`), inserted into `sys.path` by the benchmark itself; the
  loader package in `ComfyUI-HSWQ-Loader-and-Tools` carries the same modules for the ComfyUI
  runtime.
- **Rotation.** The stamp (`convrot: true`, `convrot_groupsize: 256`) makes the runtime rotate the
  activation online for the same group size. Nothing in the pack stores a rotation matrix.
- **Mode.** TC is armed by the presence of `input_scale`; otherwise the loader keeps the stock
  GEMM and the online rotation (parity). The mode is printed once per load.
- **Loader.** `ComfyUI-HSWQ-Loader-and-Tools` arms the Detect/Load/forward path and supplies the
  ConvRot activation rotation for the NVFP4 format; core declares the `nvfp4` format in
  `QUANT_ALGOS` and reads its packed weight and scales, but the rotation and the parity/TC
  selection are the loader's. Every number in §10 was measured on the HSWQ loader.
- **Placeholders.** If a pack carries no `input_scale`, a `ones(1)` placeholder is attached with a
  flag so the module can run on the parity path without pretending to be calibrated
  (`arm_nvfp4_module`). The distinction between "calibrated" and "placeholder" is recorded, so a
  parity run is never reported as a TC run.

---

## 13. Forbidden mistakes and operational contracts

| Contract | Why |
|---|---|
| **Never mix ranking sources across checkpoints** | The impact JSON is a property of the weights. A ranking measured on another checkpoint reorders the conversion and silently converts the wrong layers. |
| **Never build the hybrid from FP16** | The rotation domain comes from the INT8 pack's stamps; re-deriving it per layer adds a second place for the domain to disagree (§3.2). |
| **Never measure with a proxy quantizer** | The E4M3-per-256 proxy understates the error ~13× and flattens the ranking (§5.1). |
| **Never force TC on an uncalibrated pack** | The activation scale is a placeholder; the measured result collapses and the number is meaningless (§7.4). |
| **Never report a TC number measured off a Blackwell GPU** | The NVFP4 GEMM needs compute capability ≥ 10.0; below it the run is a parity run whatever the flags say (§8). |
| **Never judge a pack from a parity run when the target is TC** | Different computation, different number. State the mode with the result (§8). |
| **Never report a mean without the bifurcation count** | A high mean can hide a single discontinuous seed, which is a different picture (§9.2). |
| **Never call trajectory cosine an image-quality measure** | It compares latents; identical noise correlates at 0.99. Quality is judged by decoding and looking. |
| **Never edit `ComfyUI-master/`** | It is the bundled runtime reference; the pipeline patches at runtime only. |
| **One process at a time** | The measurement, the conversion and the gate each hold a full DiT; concurrent runs exhaust VRAM and can produce spurious failures. |
| **Never copy `impact/impact_<stem>.json` from another model** | Same as the first row: it is not a template. |

---

## 14. Key formulas and function map

$$\text{impact}(l) = \frac{\lVert x_T(l) - x_{\text{ref}}\rVert_F^2}{\lVert x_{\text{ref}}\rVert_F^2} \qquad\text{(terminal state, one layer substituted)}$$

$$\text{order} = \operatorname{argsort}_{\uparrow}\,\text{impact}(l), \qquad \text{converted} = \text{order}[0:K]$$

$$\text{input\_scale}_l = \frac{\max(\text{amax}_l, 10^{-12})}{448 \times 6}$$

$$\text{gate} = \left[\,\overline{\cos_T} \ge 0.95\,\right] \wedge \left[\,\#\{\text{bifurcated}\} = 0 / 20\,\right]$$

| Function / argument | Role |
|---|---|
| `diag_impact.py <base> <artifact> <out>` | base = FP16 DiT, artifact = complete ConvRot INT8 pack (candidate list), out = impact JSON |
| `--steps` / `--seed` (diag) | trajectory length (default 4) and the single measurement seed (default 42) |
| `nvfp4_quant_error(w)` | true NVFP4 round-trip through `TensorCoreNVFP4Layout` |
| `rel_mse(a, b)` | terminal relative MSE |
| `gen_reverse_nvfp4.py <k> <out_name> <src> <impact> [--out-dir]` | reverse hybrid builder |
| `_find_prefix` / `_strip_prefix` | checkpoint prefix detection for boundary matching and key writes |
| `calib_input_scale_nvfp4.py <base> <hybrid> <out>` | `input_scale` writer; `--samples` (32), `--steps` (4), `--prompts` |
| `zi_convrot_nvfp4_traj_compare.py --fp16 --hybrid --tc --attention sage2 --canonical-seeds --steps 12` | the 20-seed gate (`--canonical-seeds` selects the seed set; `--seeds` alone defaults to 5 seeds) |
| `benchmark/hswq_stack/zimage_nvfp4/nvfp4_tc_gate.py` | compute-capability gate for the TC path (`probe_nvfp4_tc_support`) |
| `benchmark/hswq_stack/zimage_nvfp4/zi_nvfp4_forward.py` | the forward that calls `scaled_mm_nvfp4_pooled` |
| `--tc` / `--parity`, `HSWQ_ZI_FORCE_PARITY` / `HSWQ_ZI_FORCE_TC` | GEMM mode selection |
| `benchmark/zi_traj_compare.py` | the older 20-seed comparator referenced by the benchmark report |

---

## 15. Reproduction checklist

1. Build the complete ConvRot INT8 pack (`Z_Image/native_convert_int8_convrot_zi.py`).
2. `diag_impact.py <base> <int8> impact/impact_<stem>.json` — never reuse another model's file.
3. `gen_reverse_nvfp4.py <K> <name> <int8> <impact>` — start at $K = 90$.
4. `calib_input_scale_nvfp4.py <base> <hybrid> <calib>` — confirm the `input_scale` key count equals
   the converted layer count.
5. Gate: `zi_convrot_nvfp4_traj_compare.py --fp16 <base> --hybrid <calib> --tc --attention sage2
   --canonical-seeds --steps 12`; read the footer to confirm
   `[HSWQ NVFP4] GEMM MODE: TC (W4A4 scaled_mm)`.
6. If the gate fails, move $K$ by 10 and repeat from 3; reject any $K$ with a bifurcated seed.
7. Record the adopted $K$, the mean, the minimum, the bifurcation count **and** the pack size
   together. A size without a gate result, or a gate result without a size, is an incomplete
   record.
8. Decode and look at the images before making any statement about picture quality.
