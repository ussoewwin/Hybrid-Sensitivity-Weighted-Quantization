# Verdict - ip-adapter_sd15_plus_hswq_convrot_int8.safetensors

**Verdict: no backdoor detected (strength A, 83/83 audited tensors). The pack executes on the A1111 route - 54 layers installed as INT8, a render produced 30.98 dB from the float adapter - and has no consumer on the ComfyUI route.**

**Class.** Pack (§14.5): a weight-only ConvRot INT8 re-quantization of
`ip-adapter_sd15_plus.pth`, the SD1.5 IP-Adapter Plus published under that name. The arithmetic is
closed against that source, and the protection set the pack carries was measured on this checkpoint
by a full trajectory sweep run on 2026-11-13.

The family matters for what can be audited. An IP-Adapter is not a ControlNet: it carries no
control branch, no `zero_convs`, no hint input. Its edit lands on the adapter's own Resampler and
To_KV projections, and its consumers are the nodes that build those two classes. The CN suite runs
on it through a `--family ipadapter` branch that supplies the adapter's own runtime names; the
ControlNet checks keep their own path unchanged.

Two runs exist on this pack. The ComfyUI node (`comfyui_ipadapter_plus`) cannot consume it - section
5. The A1111 route (`extensions-builtin/sd-webui-controlnet` with `modules/ipadapter_int8_convrot.py`)
does: all 54 layers install as INT8, the kernel runs and the pack renders, at 30.98 dB from the float
adapter on the same seed - section 6.

## 1. Identity

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 158,030,471 | source size in bytes | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `source.bytes` | the artifact the pack is closed against |
| a0db87557825ac2f8888ef06ec64c85f175e8ba5467baa2722ade3b4a9feb9f5 | source sha256 | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `source.sha256` | the bytes audited are the bytes that were quantised |
| 49,366,186 | pack size in bytes | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `pack.bytes` | the artifact this verdict applies to |
| 9a0f7cdfae2575951c3324458fd67221b6240e7c5c8ffd27476bb009aa7f6cd9 | pack sha256 | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `pack.sha256` | the file the audit measured is this file |

The source identity here is the local file's own hash, not the publisher's object id: this checkpoint
was named by the local path, and no publisher object id was compared in this run. The verdict's
reference class is therefore that local file.

## 2. Coverage in the runtime's own names

The runtime builds the adapter as a `Resampler` plus the `To_KV` projections; the pair produces a
graph of 83 tensors, and the pack holds the same 83.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 83 | tensors in the runtime graph, and in the pack | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `runtime_tensors` | the comparison covers the whole graph the loader builds |
| 0 | runtime tensors absent from the pack | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_missing` | nothing was dropped on the way in |
| 0 | pack tensors with no runtime counterpart | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_extra` | nothing was added |
| 0 | shape differences | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `coverage.n_shape_mismatch` | the names are the runtime's own |

## 3. Arithmetic - the pack is the source quantised

The replay runs on the instrument that built the pack: the converter's own Hadamard-256 rotation
and row-wise `amax/127` int8, over the runtime float weights.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 54 | INT8 layers declared, and the same 54 found | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `quantised.found` | the declared edit is the edit carried |
| 0 | INT8 values differing from the converter-identical replay | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `quantised.q_mismatch_total` | **closes the arithmetic** |
| 0 | largest integer difference | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `quantised.max_q_delta` | no element is off by even one step |
| 0.0 | largest relative scale difference | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `quantised.scale_exact_rel_max` | the scales are the replay's own values |
| 0 | non-finite scales | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `quantised.nonfinite_scales` | no scale is degenerate |
| 0 | float tensors differing, of 29 checked | `records/cn_ipadapter_sd15_plus_pack_audit.json` -> `float_side.mismatch_count` | the unquantised side is the source's tensors, bit for bit |

## 4. The protection set is measured on this checkpoint

The sweep that produced this pack was run on 2026-11-13 against this checkpoint through the applied
adapter, and completed all 54 Linear modules before the converter ran.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 54 | Linear modules measured by the full trajectory sweep (20 steps, seed 42, 512x512) | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.modules_measured` | the sweep is complete, not a focused screen |
| UNet trajectory through the applied adapter | the stage the sweep measured | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.stage` | the ranking comes from the diffusion trajectory, not from the adapter alone |
| 8.157313206424988e-05 | largest per-layer drift (`ip_adapter.13.to_v_ip`) | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.drift_max` | no layer on this checkpoint drifts near the threshold |
| 1.0778848184145317e-05 | median per-layer drift | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.drift_median` | the drift is uniformly small on this checkpoint |
| 0 | layers above the 0.15 threshold | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.drift_gt_threshold` | the pack carries no drift-driven FP16 layer |
| 0.0 | largest drift among the 22 `image_proj` Linears | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.image_proj_drift_max` | the Resampler side is untouched by the trajectory at float32 |
| 54 | ConvRot INT8 Linear layers in the pack | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.convrot_int8_linear` | the whole Linear population is quantised |
| 29 | tensors kept in FP16 (not measured Linears) | `records/cn_ipadapter_sd15_plus_protection.json` -> `measurement.kept_float_tensors` | what the pack does not quantise |

The name normalisation in that sweep is part of the record: the runtime names the `ip_layers`
children `1_to_k_ip` because a submodule name cannot hold a dot, while the checkpoint names them
`1.to_k_ip`. The sweep maps one to the other; without that map the converter sees no measured layer
for the 32 To_KV projections, and the pack silently loses them.

## 5. The ComfyUI route - the declared edit does not execute

This is the measured state of the only consumer this machine has for the file.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| Resampler(load failed) | class the node builds from the pack | `records/cn_ipadapter_sd15_plus_route.json` -> `class` | the node recognises the family but cannot finish the load |
| 0 | declared layers the node applied as INT8 | `records/cn_ipadapter_sd15_plus_route.json` -> `applied_layers` | no declared layer takes the INT8 path here |
| 32 | declared layers left as raw int8 codes | `records/cn_ipadapter_sd15_plus_route.json` -> `raw_code_layers` | the To_KV projections arrive without a dequantising consumer |
| 22 | declared layers with no module on this route | `records/cn_ipadapter_sd15_plus_route.json` -> `declared_not_applied` | the Resampler side never gets built |
| 0 | INT8 weights that the pack does not declare | `records/cn_ipadapter_sd15_plus_route.json` -> `int8_not_declared` | nothing is quantised beyond the declared set |
| 0 | INT8 GEMM calls during one forward | `records/cn_ipadapter_sd15_plus_int8_exec.json` -> `int8_linear_calls` | the INT8 kernel never runs on this route |
| the node's loader cannot take the pack | the load failure, verbatim | `records/cn_ipadapter_sd15_plus_int8_exec.json` -> `verdict` | the failure is the loader's strict `load_state_dict`, not a missing file |
| the pack has no consumer on this route; no render was produced | the end-to-end attempt on this route | `records/cn_ipadapter_sd15_plus_image_test.json` -> `verdict` | the route produces no image |

The cause is concrete and is recorded verbatim: the node's `ipadapter_model_loader` rebuilds the
nested `image_proj` dict from the flat checkpoint keys and passes it to the `Resampler` constructor
with a strict `load_state_dict`, so the pack's own companion keys (`*.weight_scale`,
`*.comfy_quant`) are rejected as unexpected. The 54 quantised layers are therefore correct
arithmetic sitting behind a loader that does not consume them. Nothing about this is asserted beyond
the one route that exists on this machine; a route that dequantises at load would change this row,
and no such route is installed here.

## 6. The A1111 route - 54 INT8 layers and a render

`modules/ipadapter_int8_convrot.py` splits the int8 payload out of the `image_proj` and `ip_adapter`
sub dicts, installs it in place and binds an instance-level `forward`; the extension's
`ipadapter_model.py` calls it for both graphs. The rows below are one render through that route.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 22 / 22 | `image_proj` Linears, and how many received the int8 payload | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `loader_install.image_proj_linear` | the Resampler runs on the INT8 kernel |
| 32 / 32 | `ip_adapter` Linears, and how many received the int8 payload | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `loader_install.ip_adapter_linear` | every To_KV projection runs on the INT8 kernel |
| 0 | int8 layers with no module in the graph | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `loader_install.skipped` | nothing falls back to float |
| 408,703 | bytes of the PNG rendered through the pack (512x512, 8 steps, cfg 4.0, seed 42, weight 1.0) | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `render.png_bytes` | the route renders an image, not only a loaded graph |
| 5a390ef10d914d8a... | sha256 of that PNG | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `render.png_sha256` | the render is reproducible from the record |
| 8.95 | PSNR of that render against the same seed without the adapter | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `comparison.int8_vs_base` | the adapter changes the image across 100% of its pixels |
| 8.98 | PSNR of the float adapter's render against the same baseline | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `comparison.float_vs_base` | the float adapter moves the image by the same amount |
| 30.98 | PSNR between the pack's render and the float adapter's render | `records/cn_ipadapter_sd15_plus_a1111_image_test.json` -> `comparison.float_vs_int8` | the pack reproduces the float adapter's effect, not merely a plausible picture |

The loader's own log lines for that run, verbatim:

```
[IPAdapter ConvRot] image_proj runtime INT8 modules installed: linear=22 skipped=0 int8_weight_mb=29.9
[IPAdapter ConvRot] ip_adapter runtime INT8 modules installed: linear=32 skipped=0 int8_weight_mb=19.2
ControlNet model ip-adapter_sd15_plus_hswq_convrot_int8 [b6a16062](ControlModelType.IPAdapter) loaded.
```

Both renders were inspected as images: finished portraits with coherent facial structure, no noise
or seam artefacts.

## 7. The source and the pack on the same input

The comparison runs the two weight sets through the classes the node builds, on the same four fixed
inputs, and takes the cosine of the concatenated adapter outputs.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 4 | input batches compared, all of them comparable | `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` -> `batches_comparable` | the comparison is not a single lucky input |
| 0.999341667 | lowest cosine between the source and the pack outputs | `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` -> `cos_min` | the pack reproduces its source's behaviour |
| 0.999645844 | mean cosine across the four batches | `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` -> `cos_mean` | the agreement is not carried by one batch |
| 2.699489146e-02 | largest relative L2 difference | `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` -> `rel_l2_max` | the difference stays inside the int8 grid of this model's scale |
| 256 | items in the fixed acceptance gate | `records/cn_ipadapter_sd15_plus_gate_256.json` -> `n` | the gate is the published 256-item set |
| 0.9998862713109702 | gate mean cosine | `records/cn_ipadapter_sd15_plus_gate_256.json` -> `mean` | the complete pack verifies on the fixed set |
| 0.9989429116249084 | gate minimum cosine | `records/cn_ipadapter_sd15_plus_gate_256.json` -> `min` | no item falls below the pass floor |
| 0.9997830390930176 | gate 5th percentile | `records/cn_ipadapter_sd15_plus_gate_256.json` -> `p5` | the lower tail is inside the pass band |
| 0 | non-finite values across the gate's 256 outputs | `records/cn_ipadapter_sd15_plus_gate_256.json` -> `nonfinite_count` | the quantised path produces no NaN or Inf |

## 8. The weight-level trigger search

A control-image search needs a control image going through a control branch, which this family does
not have. The search this family can carry is weight-level.

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 1.7881e-07 | control p99 of the divergence, from three perturbed inputs | `records/cn_ipadapter_sd15_plus_image_trigger.json` -> `control.control_p99_to_kv` | the instrument's noise floor on this model |
| 0.01 | smallest implant strength that leaves that distribution | `records/cn_ipadapter_sd15_plus_image_trigger.json` -> `detection_floor_relative_strength` | the floor of this instrument here |
| 0.003 | largest strength that does not leave it | `records/cn_ipadapter_sd15_plus_image_trigger.json` -> `results` | below the floor the search is blind |
| 2 | candidates above the control p99 at or below 0.03 relative strength | `records/cn_ipadapter_sd15_plus_image_trigger.json` -> `flagged_count` | the instrument resolves an implant of 1% of the weight's own norm |
| 0.000e+00 | image_proj divergence at every strength | `records/cn_ipadapter_sd15_plus_image_trigger.json` -> `results` | an implant in a To_KV weight leaves the Resampler output unchanged, as it must |

The pack itself was put through the same search: no candidate reaches the control p99 at or below the
0.03 strength that sits under the floor, so nothing in the pack behaves like an implant of the shape
this instrument resolves.

## 9. Conclusion

**No backdoor detected.**

- **Added material:** nothing was added; every pack tensor equals the source under the converter's
  own quantisation, and the float side is bit-identical across 29 tensors.
- **The edit is the declared one:** 54 Linear layers quantised, replayed with 0 mismatching values.
- **The protection set is this checkpoint's own:** 54 Linear modules measured on a full trajectory
  sweep on 2026-11-13, none above the 0.15 threshold.
- **What this does not say:** the source's identity here is the local file's hash. A defect present
  in that file before this run is unreachable by comparing the pack with it.
- **One route executes, one does not.** The A1111 route installs all 54 layers as INT8 and renders
  (section 6); the ComfyUI node rejects the pack's companion keys, so no INT8 GEMM runs there
  (section 5). That is a property of each consumer, not a defect in the quantised weights.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the weight-level trigger search draws 7 strengths against a 3-rep control on one
  seed; the verdict itself rests on the comparison of all 83 runtime tensors.
- Positions probed: the runtime graph's 83 tensors - 54 Linear layers and 29 remaining tensors - one
  position per module, plus the loader route and one forward in section 5.
- Detection floor: the rank-1 implant shape is the one the instrument searches for; the floor is
  0.01 relative to the weight's own Frobenius norm, above the control p99 (1.7881e-07).
- Reference-class boundary: the reference is the local source file's own bytes, identified by its
  sha256; no publisher object id was compared in this run.
- The quantisation replay reproduces the converter exactly (Hadamard by Kronecker doubling, group
  size 256, row-wise `amax/127`); exact agreement is what makes the comparison meaningful.
- The protection set is measured with the adapter applied to
  `chilledReGenericV3_v10.safetensors`; the ranking is host-specific, and the pack carries the set
  measured on that host. Every layer's drift is far below the threshold on this checkpoint, so the
  host choice changes no layer in this pack.
- The conditioning input in the sweep and in the gate is a zero context of width 768: no FP16
  CLIP-L text encoder is installed on this machine, and the quantised CLIP-L pack is a different
  artifact. Both sides of every comparison use the same conditioning, so the comparison stands, but
  the prompt-conditioned behaviour of the adapter is not exercised here.
- The pack is Linear-only: its 29 non-Linear tensors stay in FP16, and the 158 MB source shrinks to
  47.1 MiB on the Linear population alone.
- Inputs outside the tested sets are not covered; no test covers them.

## Records

- `records/cn_ipadapter_sd15_plus_pack_audit.json` - identity, runtime coverage, quantisation
  replay, float-side comparison, verdict.
- `records/cn_ipadapter_sd15_plus_route.json` - the loader route: class, declared and applied layer
  counts, raw-code count, layers with no module, the load error verbatim.
- `records/cn_ipadapter_sd15_plus_int8_exec.json` - one forward on that route: INT8 GEMM call count,
  output tensors, finiteness, the failure that stopped it.
- `records/cn_ipadapter_sd15_plus_image_test.json` - the end-to-end attempt on that route and why it
  produced no render.
- `records/cn_ipadapter_sd15_plus_pack_source_output_diff.json` - the source and the pack on the same
  inputs: the cosines, the largest element difference and the relative L2 difference.
- `records/cn_ipadapter_sd15_plus_image_trigger.json` - the weight-level trigger search: the control
  distribution, seven strengths, and the instrument's detection floor.
- `records/cn_ipadapter_sd15_plus_gate_256.json` - the 256-item acceptance gate on the complete pack.
- `records/cn_ipadapter_sd15_plus_protection.json` - the trajectory sweep the protection set is
  read from: 54 modules, their drifts, the stage and the step count, and the impact file's digest.
