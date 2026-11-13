# Verdict - ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors

**Verdict: the quantised weights carry no added material (strength A, 173/173 measured layers). The pack executes on the A1111 route - all 173 layers installed as INT8, a render produced - and has no consumer on the ComfyUI route.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`ip-adapter_pulid_sdxl_fp16.safetensors`. The arithmetic is closed against that source, and the
weights reproduce the source under dequantisation. Two routes load this model, and they behave
differently: the A1111 route (`extensions-builtin/sd-webui-controlnet` with the runtime hook in
`modules/ipadapter_int8_convrot.py`) installs the int8 payload in place, runs the ConvRot kernel and
renders; the ComfyUI node (`pulid_comfyui`) builds plain `nn.Linear` layers and assigns float
tensors to them, so the int8 codes have no kernel there. Section 5 measures both.

## 1. Identity

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 791,372,856 | source size in bytes | `records/pulid_sdxl_pack_audit.json` -> `source.bytes` | the artifact the pack is closed against |
| b258a5cd2c0c7d35f4d3e59909d7648498686afae4d0c74917e9c47d7704fbf5 | source sha256 | `records/pulid_sdxl_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes that were quantised |
| 396,740,327 | pack size in bytes | `records/pulid_sdxl_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 0262c9affd20d105421d83f95e959ca89ed8acde7df104d7b94b13b840f5440a | pack sha256 | `records/pulid_sdxl_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

The source identity is the local file's own hash; no publisher object id was compared in this run.

## 2. Coverage in the model's own names

The model is the PuLID image projection network: `IDEncoder` over `image_proj.*` and `To_KV` over
`ip_adapter.*`. The source holds 250 tensors; the pack holds the same names, plus the `weight_scale`
and `comfy_quant` companions of the layers it quantises.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 0 | source tensors absent from the pack | `records/pulid_sdxl_pack_audit.json` -> `coverage.n_missing` | nothing was dropped |
| 0 | pack tensors with no source counterpart | `records/pulid_sdxl_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/pulid_sdxl_pack_audit.json` -> `coverage.n_shape_mismatch` | the names and shapes are the model's own |
| 173 | Linear layers quantised | `records/pulid_sdxl_pack_audit.json` -> `declared.layers` | every measured Linear carries int8 |
| 77 | tensors kept as they are (biases and LayerNorm) | `records/pulid_sdxl_pack_audit.json` -> `float_side.checked` | the unquantised side is the source's, bit for bit |

## 3. Arithmetic - the pack is the source quantised

The replay runs on the converter's own transform: Hadamard by Kronecker doubling, normalised to
orthonormal, group size 256, row-wise `amax/127` int8.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 173 | INT8 layers declared, and the same 173 found | `records/pulid_sdxl_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the replay | `records/pulid_sdxl_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/pulid_sdxl_pack_audit.json` -> `quantised.max_q_delta` | no element is off by one step |
| 0.0 | largest relative scale difference | `records/pulid_sdxl_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's values |
| 0 | float tensors differing, of 77 checked | `records/pulid_sdxl_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is bit-identical |

## 4. The protection set

The sweep ran on this checkpoint: each of the 173 Linear layers was replaced by its ConvRot INT8
reconstruction, and the stage output it feeds was compared with the pristine run.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 173 | Linear modules measured | `records/pulid_sdxl_protection.json` -> `measurement.layers_measured` | the sweep is complete |
| 4.54e-06 | median drift | `records/pulid_sdxl_protection.json` -> `measurement.drift_median` | the population is tight |
| 1.26e-05 | largest drift, on `ip_adapter.75.to_v_ip` | `records/pulid_sdxl_protection.json` -> `measurement.drift_max` | two orders below the 0.15 threshold |
| 0 | layers above 0.15, kept in FP16 | `records/pulid_sdxl_protection.json` -> `measurement.drift_gt_threshold` | the empty protection set is measured, not assumed |
| 0.15 | the threshold the converter applied | `records/pulid_sdxl_protection.json` -> `converter_args.drift_threshold` | the published one |

The measurement input is a real face image, fed the way the node feeds it: the insightface antelopev2
512-d embedding concatenated with EVA-CLIP `EVA02-CLIP-L-14-336` visual features (768-d), plus EVA's own
hidden states. The EVA-CLIP weights are the local `models/clip/EVA02_CLIP_L_336_psz14_s6B.pt`.

## 5. The two routes - one executes, one has no consumer

### 5a. A1111 - the route this pack executes on

`extensions-builtin/sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py` builds `PuLIDEncoder`
and `To_KV` from the flat keys and hands both to `modules/ipadapter_int8_convrot.py`, which splits
the int8 payload out, installs it in place and binds an instance-level `forward`. The measurement
below is that build, run once, with the kernel counted at the `comfy_kitchen` boundary.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 33 / 33 | `image_proj` Linears, and how many received the int8 payload | `records/cn_pulid_sdxl_a1111_route.json` -> `image_proj` | the whole encoder runs on the int8 kernel |
| 140 / 140 | `ip_adapter` Linears, and how many received the int8 payload | `records/cn_pulid_sdxl_a1111_route.json` -> `ip_layers` | the whole To_KV stack runs on the int8 kernel |
| 0 | int8 layers with no module in the built graph | `records/cn_pulid_sdxl_a1111_route.json` -> `image_proj.skipped` | nothing falls back to float |
| 173 | ConvRot int8 GEMMs issued by one adapter forward | `records/cn_pulid_sdxl_a1111_route.json` -> `int8_gemm_calls` | every installed layer is reached, not only loaded |
| 256 | the groupsize those calls carried | `records/cn_pulid_sdxl_a1111_route.json` -> `groupsizes` | the kernel is the ConvRot one, at the pack's own groupsize |
| 0 | non-finite values across the two output tensors | `records/cn_pulid_sdxl_a1111_route.json` -> `nonfinite_count` | the int8 path produces no NaN or Inf |
| 723,598 | bytes of the PNG rendered through this route (768x768, 12 steps, cfg 3.5, seed 42, weight 0.6) | `records/cn_pulid_sdxl_a1111_image_test.json` -> `render.png_bytes` | the route renders an image, not only a loaded graph |
| 769f135e90dbce5c... | sha256 of that PNG | `records/cn_pulid_sdxl_a1111_image_test.json` -> `render.png_sha256` | the render is reproducible from the record |

The loader's own log lines for that render, verbatim:

```
[IPAdapter ConvRot] image_proj runtime INT8 modules installed: linear=33 skipped=0 int8_weight_mb=54.8
[IPAdapter ConvRot] ip_adapter runtime INT8 modules installed: linear=140 skipped=0 int8_weight_mb=340.8
ControlNet model ip-adapter_pulid_sdxl_hswq_convrot_int8 [d682a735](ControlModelType.IPAdapter) loaded.
```

The render was inspected as an image, not inferred from a metric: a finished portrait with coherent
facial structure and no noise or seam artefacts.

### 5b. ComfyUI - the route with no consumer

`pulid_comfyui` loads this model: `PulidModelLoader` rebuilds `{"image_proj": ..., "ip_adapter": ...}`
from the flat keys, `PulidModel` builds `IDEncoder` and `To_KV`, and `To_KV` creates one bias-free
`nn.Linear` per `ip_adapter` weight and assigns the tensor to `weight.data`.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| OK | the source file loads on that route | `records/pulid_sdxl_route.json` -> `source_load_on_that_route` | the route is the right one for this model |
| FAILED | the pack loads on that route | `records/pulid_sdxl_route.json` -> `pack_load_on_that_route` | the pack has no consumer here |
| 140 | int8 weight tensors the route receives, against a float16 network | `records/pulid_sdxl_route.json` -> `int8_weight_tensors` | the codes would be read as float values |
| 176 / 420 | `image_proj` and `ip_adapter` entries the pack presents | `records/pulid_sdxl_route.json` -> `image_proj_entries` | the companions travel with the codes |

The load fails on the companions (`body.0.weight_scale`, `body.0.comfy_quant`, ...) and would fail
next on the int8 weights themselves: `To_KV` builds `nn.Linear(2048, d)` and assigns `weight.data`
directly, so an int8 tensor would be cast to float and its codes read as magnitudes near ±127.
Nothing in this section says the pack is unusable; it says this one node has no int8 path.

## 6. Behaviour on the same input

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 0.999992788 | cosine between the source's embeddings and the dequantised pack's | `records/pulid_sdxl_pack_audit.json` -> `behaviour.embeds_cos` | the embedding stage is reproduced |
| 0.999995947 | cosine between the two `To_KV` output stacks | `records/pulid_sdxl_pack_audit.json` -> `behaviour.to_kv_cos` | the k/v stage is reproduced |
| 3.80e-03 | relative L2 of the embedding stage | `records/pulid_sdxl_pack_audit.json` -> `behaviour.embeds_rel_l2` | the difference is the int8 grid, not a change of function |
| 2.89e-03 | relative L2 of the k/v stage | `records/pulid_sdxl_pack_audit.json` -> `behaviour.to_kv_rel_l2` | same |

## 7. The weight-level trigger search

The control-image search needs a control image through a ControlNet route, which this model does not
have here. The weight-level search runs instead: a rank-1 implant is written into one Linear weight at
a known relative strength, and the divergence of the two stages is compared with the divergence an
untouched network produces on the same input perturbed by a natural amount of noise.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 1.431e-06 | control p99 of the embedding stage (synthetic input, section 7) | `records/pulid_sdxl_weight_trigger.json` -> `control.control_p99_embeds` | what an untouched network does |
| 5.364e-07 | control p99 of the k/v stack | `records/pulid_sdxl_weight_trigger.json` -> `control.control_p99_to_kv` | same |
| 0.1 | the smallest implant strength that leaves that distribution | `records/pulid_sdxl_weight_trigger.json` -> `detection_floor_relative_strength` | the floor of this instrument on this model |
| 0.03 | the largest strength that does **not** leave it | `records/pulid_sdxl_weight_trigger.json` -> `results` | below the floor the search is blind |
| 0 | candidates above the control p99 in the pack itself | see section 6 | the pack carries no implant at the strengths this instrument resolves |

The floor matches the one recorded for the ControlNet packs (0.1 relative to the weight's own
Frobenius norm).

## 7. Conclusion

**No added material in the quantised weights.**

- **Added material:** nothing; every pack tensor equals the source under the converter's own
  quantisation, and the 77 unquantised tensors are bit-identical.
- **The edit is the declared one:** 173 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 173 layers measured, none above 0.15.
- **What is not established:** the ComfyUI route has no consumer. There is no int8 path in
  `pulid_comfyui`, so no INT8 kernel runs and no image is produced from this pack there. That is a
  property of the route, measured in section 5, not a property of the quantisation.
- **Distribution:** the pack is not published. Publishing a file the loader cannot read would put a
  broken model in front of users.

## LIMITS

**What the verdict does not establish, and what it does.**

- Measurement input: one real face image, through the node's own embedding path. A different face
  could rank the layers differently; the ranking here is that image's.
- Positions probed: the 173 Linear modules one at a time, each against the stage output it feeds;
  plus the two-file behaviour comparison in section 6.
- Route: one installed loader was tested (`pulid_comfyui`). No other node was tested, and none is
  claimed.
- Reference-class boundary: the reference is the local source file's bytes, identified by its
  sha256; a defect present in that file before this run is unreachable by comparing the pack with it.
- No control-image trigger search: that instrument takes a control image through a ControlNet route,
  and this model has no such route on this machine. The weight-level verdict is what this run
  establishes.
- Int8 semantics: the pack's form is the one the other HSWQ ConvRot INT8 packs use
  (`int8_tensorwise`, `convrot`, `groupsize` 256); a consumer would need to dequantise with the same
  Hadamard transform, which the audit replays exactly.

## Records

- `records/pulid_sdxl_pack_audit.json` - identity, coverage, quantisation replay, float-side
  comparison, the two-file behaviour comparison, verdict.
- `records/pulid_sdxl_protection.json` - the measured drift population, the threshold, the empty
  protection set, the impact file's digest, the input caveat.
- `records/cn_pulid_sdxl_a1111_route.json` - the A1111 route: the two sub graphs, how many Linears
  each has, how many received the int8 payload, the ConvRot GEMM count and the groupsize.
- `records/cn_pulid_sdxl_a1111_image_test.json` - one render through that route: size, steps, seed,
  PNG bytes and sha256, and the loader's install counts.
- `records/pulid_sdxl_route.json` - the ComfyUI route, the source's load result, the pack's load
  failure, the int8 tensors it receives.
- `records/pulid_sdxl_weight_trigger.json` - the weight-level trigger search: the control distribution,
  the implant ladder and the detection floor.
