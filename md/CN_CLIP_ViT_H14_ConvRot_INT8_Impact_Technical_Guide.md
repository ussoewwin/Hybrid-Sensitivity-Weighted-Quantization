# CLIP-ViT-H-14 - ConvRot INT8 impact and execution guide

One checkpoint: `CLIP-ViT-H-14.safetensors` in `models/clip_vision/`, the open_clip ViT-H/14
image tower. This note records what the per-layer measurement found, what the pack costs, and
how it executes through the loader's ConvRot INT8 route.

## The structure, and what is a target

32 encoder layers, hidden 1280, 16 heads, intermediate 5120, patch 14, 257 positions, a 1024-wide
`visual_projection`, one Conv2d patch embedding (1280, 3, 14, 14).

Quantisation candidates: exactly the 192 encoder Linears - 32 x (`q_proj`, `k_proj`, `v_proj`,
`out_proj`, `mlp.fc1`, `mlp.fc2`).

Not candidates:

- `vision_model.embeddings.patch_embedding.weight` - a Conv2d
- `vision_model.embeddings.position_embedding.weight` - a learned position table (257 x 1280)
- `vision_model.embeddings.class_embedding` - a bare vector
- `visual_projection.weight` - outside the vision tower's inner module list, so the name map the
  converter uses does not reach it; it stays FP16
- every `.bias`, every LayerNorm weight

The position table is the one that matters. A converter that takes every two-dimensional
`.weight` will also take it, rotate it, and quantise it; its rows are positions, and rotating
them mixes positions with each other.

## The measurement

`clip_convert/diag_impact_clip_vision.py` runs the tower, replaces one Linear's weight with its
ConvRot INT8 reconstruction, runs again, and records the relative MSE of the vision embedding.

| | |
| :--- | :--- |
| Linears measured | 192 of 192 |
| largest drift | `encoder.layers.1.mlp.fc2`, 3.7977294488596065e-06 |
| median | 6.3034e-07 |
| p90 | 1.7849e-06 |

The selection takes the top of that ranking as the FP16-kept set. On this checkpoint 100 layers
are kept, which leaves 92 INT8.

## The pack

| | |
| :--- | :--- |
| source | 2,528,373,448 B |
| pack | 1,060,008,108 B (1,010.9 MB) |
| pack sha256 | `7AB241C18B5274A4B5683707E4A4874F23CFDC530A29C0CE29F791DD9492A48E` |
| INT8 layers | 92 |
| FP16-kept encoder layers | 100 |
| kept (non-Linear, biases, norms) | 329 |

Arithmetic on the pack: 0 code differences beyond one LSB, 0 float-side differences against the
FP16 cast of the source, 0 non-finite scales. The one-LSB tolerance is stated because the
converter's amax and a replay's amax can land one ulp apart, which flips values sitting exactly
on .5.

## Executing it

The stock route does not run this pack. `comfy.clip_vision` builds every tower with
`comfy.ops.manual_cast`, and `manual_cast` never calls `_load_quantized_module`, so a
`comfy_quant` layer arrives as int8 codes sitting inside a float `Parameter`. Through
`comfy.clip_vision.load` the pack renders an exactly black frame (mean 0.00, std 0.00).

`HSWQLoadClipVision` runs it. After the tower is built, each layer the pack declares is swapped
for `comfy.ops.mixed_precision_ops`' own `Linear`, and its
`{weight, weight_scale, bias, comfy_quant}` is loaded into it. From there the layer holds a
`QuantizedTensor` with `quant_format == "int8_tensorwise"`, ConvRot with group size 256, and the
GEMM is `comfy_kitchen.backends.cuda.int8_linear`.

| | |
| :--- | :--- |
| declared layers | 92 |
| armed layers | 92 |
| INT8 GEMM calls per forward | 92 |
| float GEMM calls per forward | 101 |
| output cosine against the source | 0.997909904 |
| image embed cosine against the source | 0.999979585 |
| last hidden state cosine | 0.999812457 |

The route is scoped to CLIP Vision: no shared module changes, and a file without `comfy_quant`
still goes through `comfy.clip_vision.load` unchanged.

## Acceptance

`audit/clip_vit_h14_gate_and_trigger.py --route hswq --loader <ComfyUI-HSWQ-Loader-and-Tools>`

| | |
| :--- | :--- |
| items | 256 (8 crops x 4 orientations x 8 brightness steps, one real image) |
| mean | 0.999043905616686 |
| min | 0.9961724471930635 |
| p5 | 0.997116333335446 |
| p50 | 0.9993950998187716 |
| verdict | PASS |

## Trigger search

55 candidates: 45 patch, 6 frequency, 4 natural transforms, all through the INT8 route.

| | |
| :--- | :--- |
| largest natural control | `control_rot90`, 0.0023910949200232068 |
| threshold | that control x 1.5 = 0.00358664238003481 |
| flagged | 2 - `patch_qrblock_bl_64` 0.004832689430 (2.02x rot90), `patch_solid_tl_16` 0.003818506488 (1.60x rot90) |
| positive control floor, float edit | 0.3 of the weight norm (median 6.898635e-03) |
| positive control floor, codes | 32 LSB on 10 % of rows (median 9.638936e-03) |
| noise floor | rel 3.1882774738526067e-03 on the weights, 2.04e-03 on the output |

Both flagged candidates (4.83e-03, 3.82e-03) sit below every floor measured inside the INT8
layer, so they are not weight- or code-shaped at a size this search can resolve.

The noise floor is the reason the floors are not finer. Dequantising a ConvRot INT8 layer and
requantising it with nothing added already moves its weights by rel 3.19e-03. An implant smaller
than that is indistinguishable from that noise, which is why the rungs below 0.3 all land around
2e-03 regardless of their nominal strength. Five implant directions per rung, median reported;
a single direction is not enough to set a floor.

## Rendering

Same graph, same seed, three loader paths:

| arm | loader | frame | measured |
| :--- | :--- | :--- | :--- |
| D | `comfy.clip_vision.load`, source | 397,650 B | mean 112.06, std 64.82 |
| E | `comfy.clip_vision.load`, pack | 843 B | mean 0.00, std 0.00 |
| F | `HSWQLoadClipVision`, pack | 397,589 B | mean 111.73, std 64.99 |

E is black; F is the source's scene (max per-pixel difference against D is 163, FP16 rounding
through sampling).

## Practical notes

- Load this pack with `HSWQLoadClipVision`. On a stock CLIP vision route it is read as float
  codes and the tower will not reproduce the source.
- Keep the source. The pack is 41.9 % of it, and the 0.999979585 cosine is measured through the
  route, not extrapolated.
- The 256-item set is one scene under transforms. A corpus-level number needs more scenes.
- The detection floor of the trigger search is 30 % of a layer's weight norm. A smaller
  weight-shaped implant is outside what that search measured.

## Records

- `md/backdoor/verdicts/CN_CLIP_ViT_H14_Pack.md`
- `records/cn_clip_vit_h14_pack_audit.json`
- `records/cn_clip_vit_h14_route.json`
- `records/cn_clip_vit_h14_route_exec.json`
- `records/cn_clip_vit_h14_gate_256.json`
- `records/cn_clip_vit_h14_image_trigger.json`
- `records/cn_clip_vit_h14_protection.json`
- `records/cn_clip_vit_h14_int8_control.json`
- `records/cn_clip_vit_h14_convrot_route.json`
