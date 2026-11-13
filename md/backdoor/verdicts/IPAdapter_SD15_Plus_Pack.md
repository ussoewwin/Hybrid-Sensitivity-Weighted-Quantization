# Verdict - ip-adapter_sd15_plus_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, 54/54 measured layers).** The pack carries no material
beyond the source under the converter's own quantisation, and the weights reproduce the source after
dequantisation.

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of `ip-adapter_sd15_plus.pth`, the
SD1.5 IP-Adapter Plus image projection. The graph is the one the node builds: a `Resampler` over the
51 `image_proj` tensors and one bias-free Linear per `ip_adapter` weight.

## 1. Identity

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 158,030,471 | source size in bytes | `records/ipadapter_sd15_plus_pack_audit.json` -> `source.bytes` | the artifact the pack is closed against |
| a0db87557825ac2f8888ef06ec64c85f175e8ba5467baa2722ade3b4a9feb9f5 | source sha256 | `records/ipadapter_sd15_plus_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes that were quantised |
| 49,366,186 | pack size in bytes | `records/ipadapter_sd15_plus_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 9a0f7cdfae2575951c3324458fd67221b6240e7c5c8ffd27476bb009aa7f6cd9 | pack sha256 | `records/ipadapter_sd15_plus_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

## 2. Coverage

The source holds 51 `image_proj` tensors and 32 `ip_adapter` tensors; the pack holds the same names,
plus the `weight_scale` and `comfy_quant` companions of the layers it quantises.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 0 | source tensors absent from the pack | `records/ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_missing` | nothing was dropped |
| 0 | pack tensors with no source counterpart | `records/ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_shape_mismatch` | the names and shapes are the model's own |
| 54 | Linear layers quantised | `records/ipadapter_sd15_plus_pack_audit.json` -> `declared.layers` | every measured Linear carries int8 |
| 29 | tensors kept as they are (biases and LayerNorm) | `records/ipadapter_sd15_plus_pack_audit.json` -> `float_side.checked` | the unquantised side is the source's, bit for bit |

## 3. Arithmetic - the pack is the source quantised

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 54 | INT8 layers declared, and the same 54 found | `records/ipadapter_sd15_plus_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/ipadapter_sd15_plus_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/ipadapter_sd15_plus_pack_audit.json` -> `quantised.max_q_delta` | no element is off by one step |
| 0.0 | largest relative scale difference | `records/ipadapter_sd15_plus_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own |
| 0 | float tensors differing, of 29 checked | `records/ipadapter_sd15_plus_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is bit-identical |

## 4. The protection set is measured on this checkpoint

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 54 | Linear modules measured (22 image_proj + 32 ip_adapter) | `records/ipadapter_sd15_plus_protection.json` -> `measurement.layers_measured` | the sweep is complete |
| 1.41e-05 | median drift | `records/ipadapter_sd15_plus_protection.json` -> `measurement.drift_median` | the population is tight |
| 6.32e-04 | largest drift, on `image_proj.proj_in` | `records/ipadapter_sd15_plus_protection.json` -> `measurement.drift_max` | two orders below the threshold |
| 0 | layers above 0.15, kept in FP16 | `records/ipadapter_sd15_plus_protection.json` -> `measurement.drift_gt_threshold` | the empty set is measured, not assumed |

The measurement input is a real clip-vision embedding of one image, read through the node's own path
(`records/ipadapter_sd15_plus_protection.json` carries the exact string). The `ip_adapter` layers carry more drift than `image_proj` on this checkpoint: the `to_v_ip` rows
reach 4.1e-05 to 5.5e-05, while the deepest `image_proj` FFN layers sit below 1e-06.

## 5. The declared edit executes on the loader the card names

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 95 / 96 | `image_proj` and `ip_adapter` entries the pack presents to the loader | `records/ipadapter_sd15_plus_route.json` -> `image_proj_entries` | the companions travel with the codes |
| FAILED | the pack loads on that route | `records/ipadapter_sd15_plus_route.json` -> `pack_load_on_that_route` | the pack has no consumer here |
| 32 | int8 weight tensors the route receives, against a float16 network | `records/ipadapter_sd15_plus_route.json` -> `int8_weight_tensors` | the codes would be read as float values |

`ipadapter_model_loader` rebuilds `{"image_proj": ..., "ip_adapter": ...}` from the flat keys, and
`To_KV` builds one bias-free `nn.Linear` per weight and assigns the tensor to `weight.data`. The int8
codes and their companions have no consumer on that path. This is a property of the route, not of the
quantisation: section 3 closes the arithmetic, and section 6 shows the weights reproduce the source.

## 6. Behaviour on the same input

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 0.999785423 | cosine between the source's image_proj output and the dequantised pack's | `records/ipadapter_sd15_plus_pack_audit.json` -> `behaviour.embeds_cos` | the projection stage is reproduced |
| 0.999867499 | cosine between the two k/v output stacks | `records/ipadapter_sd15_plus_pack_audit.json` -> `behaviour.to_kv_cos` | the k/v stage is reproduced |
| 2.07e-02 | relative L2 of the projection stage | `records/ipadapter_sd15_plus_pack_audit.json` -> `behaviour.embeds_rel_l2` | the difference is the int8 grid |
| 1.63e-02 | relative L2 of the k/v stage | `records/ipadapter_sd15_plus_pack_audit.json` -> `behaviour.to_kv_rel_l2` | same |

## 7. The weight-level trigger search

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 1.788e-07 | control p99 of the projection stage | `records/ipadapter_sd15_plus_weight_trigger.json` -> `control.control_p99_embeds` | what an untouched network does |
| 8.941e-08 | control p99 of the k/v stack | `records/ipadapter_sd15_plus_weight_trigger.json` -> `control.control_p99_to_kv` | same |
| 0.003 | the smallest implant strength that leaves that distribution | `records/ipadapter_sd15_plus_weight_trigger.json` -> `detection_floor_relative_strength` | the floor on this model |
| 0.001 | the largest strength that does **not** leave it | `records/ipadapter_sd15_plus_weight_trigger.json` -> `results` | below the floor the search is blind |

The floor is two orders finer than the ControlNet packs (0.1): this network is 54 Linear layers with
a short path, so a rank-1 implant of 0.3% of the weight norm already moves the output past the
control distribution.

## 8. Conclusion

**No backdoor detected.**

- **Added material:** nothing; every pack tensor equals the source under the converter's own
  quantisation, and the 29 unquantised tensors are bit-identical.
- **The edit is the declared one:** 54 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 54 layers measured on a real clip-vision
  embedding, none above 0.15.
- **What is not established:** the pack does not execute on the installed route, so no INT8 kernel
  runs. That is measured in section 5 and does not bear on the arithmetic or on the weights.
- **Distribution:** the pack is published to the ControlNet collection with this verdict attached.

## LIMITS

- Measurement input: one clip-vision embedding from one image. A different image could rank the
  layers differently; the ranking here is that image's.
- Positions probed: the 54 Linear modules one at a time, each against the stage output it feeds; plus
  the two-file behaviour comparison in section 6 and the implant ladder in section 7.
- Route: one installed loader was tested (`comfyui_ipadapter_plus`). No other node was tested.
- Reference-class boundary: the reference is the local source file's bytes, identified by its sha256;
  no publisher object id was compared, and a defect present in that file before this run is
  unreachable by comparing the pack with it.
- No image render: the route has no consumer for the int8 codes, so no generation was run from this
  pack. A render of the source was also not part of this run.
- Int8 semantics: the pack's form is the one the other HSWQ ConvRot INT8 packs use
  (`int8_tensorwise`, `convrot`, `groupsize` 256), and the audit replays that transform exactly.

## Records

- `records/ipadapter_sd15_plus_pack_audit.json` - identity, coverage, quantisation replay, float-side
  comparison, the two-file behaviour comparison, verdict.
- `records/ipadapter_sd15_plus_protection.json` - the measured drift population, the threshold, the
  empty protection set, the impact digest, the input.
- `records/ipadapter_sd15_plus_route.json` - the installed route, the nested rebuild, the load
  failure, the int8 tensors it receives.
- `records/ipadapter_sd15_plus_weight_trigger.json` - the control distribution, the implant ladder,
  the detection floor.
