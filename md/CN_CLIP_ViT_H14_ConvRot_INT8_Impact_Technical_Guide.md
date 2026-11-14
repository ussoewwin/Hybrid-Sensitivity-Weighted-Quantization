# CLIP-ViT-H-14 - ConvRot INT8 impact and execution guide

One checkpoint: `CLIP-ViT-H-14.safetensors` in `models/clip_vision/`, the open_clip ViT-H/14
image tower. This note records what the per-layer measurement found, what the pack costs, and
what happens when this machine tries to execute it.

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
| kept (non-Linear, biases, norms) | 328 |

Arithmetic on the pack: 0 code differences beyond one LSB, 0 float-side differences against the
FP16 cast of the source, 0 non-finite scales. The one-LSB tolerance is stated because the
converter's amax and a replay's amax can land one ulp apart, which flips values sitting exactly
on .5.

## What it does on this machine

The pack is faithful. It also does not run here, and both facts belong in the same note.

`comfy.clip_vision.load` reads the file's tensors into the model's parameters directly. It does
not interpret `comfy_quant`, so each int8 code tensor becomes a float parameter holding the code
values: 92 of 92 declared layers come out as raw float codes, 0 come out INT8-backed, and one
forward makes 0 INT8 GEMM calls and 193 float ones. The output cosine against the source tower on
that route is -0.022604210.

The HSWQ loader tree in `custom_nodes/` has no node naming `clip_vision`, and A1111's ControlNet
extension exposes no `clip_vision` input, so there is no second route.

## Rendering it

The route exists: `CLIPVisionLoader` feeds `IPAdapterAdvanced`'s `clip_vision` input. Three arms,
one graph, one seed:

| arm | clip vision file | frame | measured |
| :--- | :--- | :--- | :--- |
| A | source | 407,287 B | mean 110.60, std 64.70 |
| B | the pack | 2,414 B | mean 0.00, std 0.00, max 0.0 |
| C | the pack dequantised | 407,289 B | mean 110.61, std 64.79 |

B is exactly black. C is the source's scene, pixel-close but not identical (max per-pixel
difference 178.0, FP16 rounding through sampling). B and C carry the same weights; only the
dequantiser differs.

To make C, dequantise the pack: codes times the per-group scale, then the inverse order-4^k
Hadamard rotation per group, cast to float16. The result is `CLIP-ViT-H-14_dequant_from_pack.safetensors`,
1,264,219,400 B, 521 tensors, 92 layers dequantised.

## How the pack was therefore checked

The comparison that isolates the quantisation dequantises the codes and scales into the source's
own structure:

| | |
| :--- | :--- |
| token-stack cosine, source vs pack | 0.999936295 |
| 256-item acceptance (8 crops x 4 orientations x 8 brightness steps) | mean 0.999784, min 0.996575, PASS |
| 55-candidate image trigger search | flagged 1, control p99 0.00134791 |
| the flagged candidate | `freq_low_sigma_0.02`, 1-cos 0.00151049, 1.68x the largest natural control |
| positive control | never cleared the control distribution - no measured detection floor |

## Practical notes

- Use the pack only where something dequantises it. On a stock ComfyUI CLIP vision route it is
  read as float codes and the tower will not reproduce the source.
- Keep the source. The pack is 41.9 % of it, and the 0.9999 cosine is measured against a
  dequantised reconstruction, not against a route that ran.
- The 256-item set is one scene under transforms. A corpus-level number needs more scenes.

## Records

- `md/backdoor/verdicts/CN_CLIP_ViT_H14_Pack.md`
- `records/cn_clip_vit_h14_pack_audit.json`
- `records/cn_clip_vit_h14_route.json`
- `records/cn_clip_vit_h14_route_exec.json`
- `records/cn_clip_vit_h14_gate_256.json`
- `records/cn_clip_vit_h14_image_trigger.json`
- `impact/impact_clip_vit_h14_192.json`
