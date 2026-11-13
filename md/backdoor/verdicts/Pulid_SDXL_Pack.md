# Verdict - ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors

**Verdict: the quantised weights carry no added material (strength A, 173/173 measured layers). The pack is not executable on the installed route, and is therefore not distributed.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`ip-adapter_pulid_sdxl_fp16.safetensors`. The arithmetic is closed against that source, and the
weights reproduce the source under dequantisation. What this pack does not have is a consumer: the
node that loads this model builds plain `nn.Linear` layers and assigns float tensors to them, so the
int8 codes and their companions have no path to a kernel. Section 5 is that measurement.

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
| 4.75e-06 | median drift | `records/pulid_sdxl_protection.json` -> `measurement.drift_median` | the population is tight |
| 1.46e-05 | largest drift, on `ip_adapter.75.to_v_ip` | `records/pulid_sdxl_protection.json` -> `measurement.drift_max` | two orders below the 0.15 threshold |
| 0 | layers above 0.15, kept in FP16 | `records/pulid_sdxl_protection.json` -> `measurement.drift_gt_threshold` | the empty protection set is measured, not assumed |
| 0.15 | the threshold the converter applied | `records/pulid_sdxl_protection.json` -> `converter_args.drift_threshold` | the published one |

The measurement input is synthetic: `id_cond [1,1280]` (512 insightface + 768 EVA-CLIP) and five
`[1,257,1024]` hidden tensors, at the shapes the node feeds. The EVA-CLIP weights that this model's
node requires are not present on this machine, so the sweep could not use a face image. That
limitation is carried in LIMITS, not hidden here.

## 5. The route - measured, and absent

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
| 1.431e-06 | control p99 of the embedding stage | `records/pulid_sdxl_weight_trigger.json` -> `control.control_p99_embeds` | what an untouched network does |
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
- **What is not established:** the pack is not executable. There is no consumer for int8 codes on
  the installed route, so no INT8 kernel runs and no image is produced from this pack. That is a
  property of the route, measured in section 5, not a property of the quantisation.
- **Distribution:** the pack is not published. Publishing a file the loader cannot read would put a
  broken model in front of users.

## LIMITS

**What the verdict does not establish, and what it does.**

- Measurement input: synthetic tensors at the node's shapes, not a face image. The EVA-CLIP weights
  the node requires are absent from this machine, so the drift ranking rests on synthetic input.
  A face-image sweep could rank the layers differently.
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
- `records/pulid_sdxl_route.json` - the installed route, the source's load result, the pack's load
  failure, the int8 tensors it receives.
- `records/pulid_sdxl_weight_trigger.json` - the weight-level trigger search: the control distribution,
  the implant ladder and the detection floor.
