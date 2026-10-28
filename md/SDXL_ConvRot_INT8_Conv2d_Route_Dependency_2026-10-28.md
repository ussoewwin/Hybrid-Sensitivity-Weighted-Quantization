# SDXL ConvRot INT8: quantised conv2d layers need the loader node's INT8 path

Measured 2026-10-28. Repo: [Hybrid-Sensitivity-Weighted-Quantization](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization).
Every number is traceable to a record named in section 6.

## 1. One claim made and withdrawn

The first pass at this pack concluded that its five quantised conv2d layers "are not applied",
because each loaded `torch.float16` parameter equalled the stored int8 tensor (`absmax` 127.0, every
value integral) and produced an output about 1000 times larger than a reference convolution built
from the dequantised weight. **That conclusion is withdrawn.** It was reached by inspecting
parameters on a load path that is not the product's. Executed on the product's INT8 path, the same
five layers are the dequantised convolution (section 3).

What survived is narrower and still matters: conv2d layers only execute correctly when the model is
loaded through the loader node's INT8 entry. ComfyUI's generic quantised load leaves them as codes.

## 2. The routes, on the same pack

| route | declared | applied | raw | other | unmatched | conv2d executed |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| the loader node's INT8 entry, with its `prestartup_script.py` hooks armed - `audit/pack_load_path.py --route diffusion`, record `gate8_sdxl_final.json` | 550 | **550** | **0** | 0 | 0 | 5 |
| ComfyUI's generic mixed-precision load of the same bytes (a route the card does not tell anyone to use) | 550 | 545 | 5 | 0 | 0 | 5 |

The five are the only 4-D layers the pack declares: `input_blocks.5.0.out_layers.3` (640,640,3,3),
`middle_block.2.in_layers.2` (1280,1280,3,3), `output_blocks.1.0.in_layers.2` (1280,2560,3,3),
`output_blocks.4.0.out_layers.3` and `output_blocks.5.0.out_layers.3` (640,640,3,3). Each stores
int8 codes with a per-output-channel scale of shape (out,1,1,1) and a marker declaring
`int8_tensorwise`, convrot, group size 64. Keys line up exactly: weight, `weight_scale` and
`comfy_quant` all exist under the same prefix
(`records/gate8_sdxl_key_prefix.json` -> `declared_names_with_prefix`, 550), so nothing about the file is malformed.

## 3. The execution test

Each conv2d module was run twice on the same input: once with its loaded weights, once with the
dequantised weights written into it. Identical outputs mean the loaded representation *is* the
intended weight.

| layer | INT8 path, load inside and outside the scope | generic load, outside the scope |
| :--- | :--- | :--- |
| `input_blocks.5.0.out_layers.3` | cos 1.000000, relative difference 0.0000 | cos 0.135919, relative 968.8631 |
| `middle_block.2.in_layers.2` | cos 1.000000, relative 0.0000 | cos 0.068385, relative 1226.4794 |
| `output_blocks.1.0.in_layers.2` | cos 1.000000, relative 0.0000 | cos 0.081654, relative 1161.3016 |
| `output_blocks.4.0.out_layers.3` | cos 1.000000, relative 0.0000 | cos 0.134118, relative 782.2374 |
| `output_blocks.5.0.out_layers.3` | cos 1.000000, relative 0.0000 | cos 0.126932, relative 863.1510 |

The INT8 column was measured twice - once with the scope still open and once after leaving it, so
that a later sampler is covered too (`records/gate8_sdxl_conv_in_scope.json` -> `in_scope.0.output_cos_loaded_vs_dequantised`, 1.0;
`records/gate8_sdxl_load_in_scope_execute_out.json` -> `rows.0.output_cos_loaded_vs_dequantised`, 1.0). Cosines are clamped at 1.0; raw accumulation had put
them at 1.0000000000000002, which is rounding, not a measurement.

A parameter inspection would have kept reporting the codes as a defect on the INT8 path too, so the
gate tool now judges conv2d by execution and only judges 2-D (linear) layers by format inversion
(`classify()` in `audit/pack_load_path.py`).

## 4. ControlNet, measured the same way

`CN-anytest4_illustrious2_B_hswq_convrot_int8.safetensors`: 327 declared layers, **327 applied, 0
raw**, through `ControlNetLoader` and through `HSWQControlNetLoader`, with the node's
`prestartup_script.py` armed (`records/gate8_controlnet_final.json` -> `applied_layers`, 327;
`records/gate8_controlnet_product.json` -> `stock ControlNetLoader.applied_layers`, 327). An
earlier run without that wrapper reported 327 raw and is retracted for the same reason: it measured
a state no user reaches.

## 5. Consequences carried into the cards and the gate

- The instruction to load these packs with
  [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) is
  load-bearing for conv2d: on the generic path, five conv2d layers per SDXL-family pack execute the
  codes. Other packs in the family declare them too (`koronemixIllustrious_v70` 4,
  `waiIllustriousSDXL_v170_re590` 7), so this is not specific to one file.
- No card may say ComfyUI applies `comfy_quant` by itself. CLIP-shaped text encoders and
  clip-vision files additionally need the metadata arm described in
  [the load-path report](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/ClipFiles_INT8_TextEncoder_Load_Path_2026-10-28.md);
  conv2d in diffusion and ControlNet packs needs the node's INT8 load entry.
- Gate 8 in `md/Backdoor_Audit_Procedure.md` requires an execution measurement on the named route,
  because in this investigation a parameter inspection produced a wrong public conclusion twice.

## 6. Records

`records/gate8_sdxl_final.json` -> `applied_layers` (550 on the INT8 route),
`records/gate8_sam31_final.json` -> `applied_layers` (41),
`gate8_controlnet_final.json` and `gate8_controlnet_product.json` (327/327),
`gate8_sdxl_conv_in_scope.json`, `gate8_sdxl_load_in_scope_execute_out.json` (section 3),
`gate8_sdxl_conv_diag.json` (550 rows of stored dtype, shape, scale shape, marker),
`gate8_sdxl_key_prefix.json`, `gate8_sdxl_param_dtypes.json`,
`gate8_sdxl_production_route.json` and `gate8_sdxl.json` (the withdrawn readings),
`records/gate8_product_route_summary.json`.
