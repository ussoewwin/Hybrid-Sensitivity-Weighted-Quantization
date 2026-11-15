# SDXL IP-Adapter Plus (ViT-H) (`ip-adapter-plus_sdxl_vit-h`) ConvRot INT8 - Quantization and Score Report

**Checkpoint.** `models/controlnet/ip-adapter-plus_sdxl_vit-h.safetensors` (847,517,512 B, 191 tensors: `image_proj` 51 / `ip_adapter` 140)
**Pack.** `models/controlnet/ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors` (424,750,694 B, **−49.9 %**)
**SHA256 (source).** `3F5062B8400C94B7159665B21BA5C62ACDCD7682262743D7F2AEFEDEF00E6581`
**SHA256 (pack).** `CA0FF3706DC3673538EEC2683774E384FE3145F2191F2006E03862A7D3C2DB7A`

**Composition.** `image_proj` 22 Linears + `ip_adapter` 140 Linears in INT8 (**162**) · **FP16 kept 29** (`latents`, the LayerNorm vectors, the biases — none of them a quantiser input) · 515 tensors in the pack.

---

## Table of contents

0. [The load path is the whole story](#0-the-load-path-is-the-whole-story)
1. [What is quantized](#1-what-is-quantized)
2. [What the protection set is based on](#2-what-the-protection-set-is-based-on)
3. [The score table (256 items, seed 42, through the real loader)](#3-the-score-table-256-items-seed-42-through-the-real-loader)
4. [Visible-sample check](#4-visible-sample-check)
5. [Reproduction](#5-reproduction)
6. [Error sources and limits](#6-error-sources-and-limits)
7. [Ledger](#7-ledger)
8. [Files](#8-files)

---


## 0. The load path is the whole story

This adapter has one route that installs the pack's INT8 payload and one that does not. Both are measured, and the difference decides where the acceptance numbers come from.

### 0.1 ComfyUI's IP-Adapter loader does not read `comfy_quant`

`custom_nodes/comfyui_ipadapter_plus/utils.py :: ipadapter_model_loader` reads the file with `load_torch_file(..., safe_load=True)` and rebuilds the adapter from the state dict. It never consults `comfy_quant`, so a quantised weight arrives as an int8 tensor and is assigned where a float parameter is expected.

Measured on this pack (`audit/ipadapter_plus_vit_h_route.py`): **int8 tensors presented 162, `comfy_quant` markers 162, `weight_scale` tensors 162, `applies_int8` false** — the pack is not applied on that path.

This matters more here than on the basic adapter, because the constructor does **not** raise: the `Resampler`'s `proj_in` / `proj_out` / attention Linears and the `to_k_ip`/`to_v_ip` Linears all accept the tensor shape, and the int8 codes would be read as magnitudes near ±127. The failure is silent rather than loud.

### 0.2 A1111 installs the payload in place

`extensions-builtin/sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py :: IPAdapterModel.load` hands the state dict to `modules/ipadapter_int8_convrot.prepare_state_dict` → `install_runtime_int8`, which splits the int8 payload out, installs it on the module and binds an instance-level forward. Measured: **`image_proj` linear=22 skipped=0 (82.9 MB), `ip_adapter` linear=140 skipped=0 (340.8 MB)**.

Every acceptance number in sections 3 and 4 comes from that route, with the ConvRot kernel firing. One `image_proj` forward calls the kernel **22** times, which is one per installed `image_proj` Linear.

### 0.3 Forge reads the marker, not the codes

`backend/operations_mixed_precision.py` pops `comfy_quant` (`json.loads(layer_conf.numpy().tobytes())`) and dispatches on the `convrot` flag with `ck.int8_linear(convrot=True)`. The pack's stamps are the form that path expects. **No Forge install is present on this machine**, so that route was not executed here.

---


## 1. What is quantized

The checkpoint's `image_proj` is a **`Resampler` of depth 4**, not an `ImageProjModel`. That single fact changes the graph that gets quantised: 22 Linears instead of 1, in four attention blocks with residual connections.

The Resampler's parameters are read off the checkpoint, not assumed, and they are checked against the class the runtime builds:

| parameter | value | where it comes from in the file |
| :--- | ---: | :--- |
| `dim` | 1280 | `latents` shape (1, 16, 1280) |
| `depth` | 4 | four `layers.*` groups |
| `dim_head` | 64 | `to_q` is 1280 wide with `heads` 20 |
| `heads` | 20 | `layers.*.0.to_q.weight` (1280, 1280) -> 1280 / 64 |
| `num_queries` | 16 | `latents` shape |
| `embedding_dim` | 1280 | `proj_in.weight` (1280, 1280) |
| `output_dim` | 2048 | `proj_out.weight` (2048, 1280) |
| `ff_mult` | 4 | `layers.*.1.1.weight` (5120, 1280) |

`clip_convert/diag_impact_ipadapter_plus_vit_h.py` derives these from the tensor shapes and loads them into the node's own `image_proj_models.Resampler`, so the measured function is the runtime function. A shape that does not fit raises rather than being coerced.

| part | layers | what they are |
| :--- | ---: | :--- |
| `image_proj` | 22 | `proj_in`, `proj_out`, and per block: `to_q`, `to_kv` (2560x1280), `to_out`, `1.1` (5120x1280), `1.3` (1280x5120) |
| `ip_adapter` | 140 | 70 blocks x (`to_k_ip`, `to_v_ip`), 2048 in, 640 or 1280 out |
| not quantised | 29 | `latents`, 16 LayerNorm vectors, 12 biases |
| group size | 256 | all INT8 layers; every inner dimension is divisible by 256 |

---

## 2. What the protection set is based on

**Nothing is protected, and the curve says why.**

The criterion is the aggregate contribution: replace one layer by its ConvRot INT8 reconstruction and measure the change in the stage output it feeds, against the all-float source. On this graph the "restore one layer to float" form of that measurement ranks nothing — starting from the all-INT8 pack and restoring one `image_proj` layer makes the worst-latent divergence **worse for all 22 layers** (measured gains −1.7e-02 to −9.1e-01), because a single layer's precision is not a small perturbation of a residual attention block. So the measurement runs from the other end: all-float source as the reference, one layer at a time quantised.

| rank | layer | contribution |
| :--- | :--- | ---: |
| 1 | `image_proj.proj_in` | 1.502143e-04 |
| 2 | `ip_adapter.25.to_k_ip` | 4.238458e-05 |
| 3 | `ip_adapter.21.to_k_ip` | 3.436726e-05 |
| 4 | `ip_adapter.117.to_k_ip` | 3.356947e-05 |
| 5 | `ip_adapter.131.to_k_ip` | 3.340939e-05 |
| 6 | `ip_adapter.61.to_v_ip` | 3.333808e-05 |
| median | 162 layers | 2.409374e-05 |

The curve, measured on the fixed 256-item set through the INT8 route (K = layers kept in FP16):

| K | size | gate mean | worst single latent | `ip_adapter` mean | gate |
| ---: | ---: | ---: | ---: | ---: | :--- |
| **0** | **405.1 MB** | **0.999739679** | **0.986089059** | 0.999922318 | **PASS** |
| 1 | 406.6 MB | 0.999837537 | 0.938652834 | 0.999922318 | PASS |
| 20 | 449.0 MB | 0.999837537 | 0.938652834 | 0.999933195 | PASS |
| 60 | 541.3 MB | 0.999837537 | 0.938652834 | 0.999955528 | PASS |
| 162 | 808.3 MB | 1.000000000 | 1.000000000 | 1.000000000 | PASS |

**K = 0 is adopted.** It is the smallest pack; it passes with 9.74e-03 of mean margin and 1.89e-03 of min margin; and 0 of the 162 groups is anywhere near the drift threshold (max 8.49e-05 against 0.15). Protection adds 1.5 to 136 MB for +9.8e-05 to +1.1e-04 of gate mean, which is one percent of the gate's own margin, while the worst single latent falls from 0.986 to 0.939.

The drift screen, for reference (`clip_convert/diag_impact_ipadapter_plus_vit_h.py`, 162 groups, seed 42, float32): max 8.488220e-05 (`ip_adapter.25.to_k_ip`), median 4.826412e-05, p90 5.861301e-05, min 1.370495e-08, 0 above 0.15. The screen's spread is wide enough to be informative, but the screen does not decide; section 2's contribution ranking does.

---

## 3. The score table (256 items, seed 42, through the real loader)

**Route.** A1111 `modules/ipadapter_int8_convrot.py`, **162 INT8 layers installed**, **5772 kernel calls** over the gate. A number cannot come from a float fallback: the kernel is counted.

**Two surfaces, reported separately.** The `ip_adapter` projections consume the UNet cross-attention K/V, not the image embedding, so their output does not depend on the image; they are measured once on a fixed input of each projection's own width. Only `image_proj` is image-conditioned.

| item | `image_proj` | `ip_adapter` |
| :--- | ---: | ---: |
| items | 256 | 140 elements |
| statistic | mean of the item's 16 latents | the element's own cosine |
| mean | **0.999739679** | **0.999922318** |
| min | **0.998887861** | **0.999842750** |
| p5 / p50 | 0.999466956 / 0.999780970 | - |
| CN-5a, unperturbed | 0.995676220 mean / 0.933289702 worst latent | 0.999922318 / 0.999842750 |
| worst single latent (reported, not gated) | **0.986089059** (`c5_o0_b3`, latent 12) | - |
| finite | yes (0 non-finite) | yes |
| gate (`mean >= 0.99`, `min >= 0.98`) | **PASS** | **PASS** |

The `ipadapter` surface is scored per projection (140 values); the `image_proj` surface is scored per item and its per-latent worst is reported beside the gate (Section 6 says why it is not the criterion).

---

## 4. Visible-sample check

**Not measured for this pack.** The A1111 install carries no IP-Adapter model registration and the ComfyUI extension refuses the pack, so there is no pair of images to place side by side for this checkpoint. The absence is stated as a limit (Section 6), not filled with a number that would not mean what a picture means.

What the pack's numbers say, and do not say: the injected 16x2048 latent stack agrees with the source at 0.9997 per item and 0.986 at its worst single latent, with 162 of 162 layers on the INT8 kernel. Whether that difference reaches the canvas as something visible is exactly what this report does not answer.

---

## 5. Reproduction

```
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   A1111   = the install whose ControlNet extension carries modules/ipadapter_int8_convrot.py
#   COMFY   = <ComfyUI>

# 1. the drift screen (162 groups, seed 42, float32, the node's own Resampler)
python clip_convert/diag_impact_ipadapter_plus_vit_h.py \
    --input <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --out records/cn_ipadapter_plus_vit_h_protection.json --seed 42 \
    --node <COMFY>/custom_nodes/comfyui_ipadapter_plus

# 2. the criterion: one layer at a time against the all-float source
python clip_convert/measure_ipadapter_plus_vit_h_contrib.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --out records/cn_ipadapter_plus_vit_h_contrib.json --seed 42 \
    --node <COMFY>/custom_nodes/comfyui_ipadapter_plus

# 3. the builder (threshold 0.15, groupsize 256, no protection)
python clip_convert/convert_ipadapter_plus_vit_h_convrot_int8.py \
    --input <src>/ip-adapter-plus_sdxl_vit-h.safetensors \
    --impact records/cn_ipadapter_plus_vit_h_protection.json \
    --output <out>/ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors \
    --drift-threshold 0.15

# 4. the pack audit (section 0 and section 3's faithfulness)
python audit/ipadapter_plus_vit_h_pack_audit.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --node <COMFY>/custom_nodes/comfyui_ipadapter_plus \
    --json records/cn_ipadapter_plus_vit_h_pack_audit.json

# 5. the route census (section 0)
python audit/ipadapter_plus_vit_h_route.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --comfy <COMFY> --root <A1111> --json records/cn_ipadapter_plus_vit_h_route.json

# 6. the acceptance through the real INT8 route (section 3)
python audit/ipadapter_plus_vit_h_acceptance.py \
    --source <src>/ip-adapter-plus_sdxl_vit-h.safetensors --pack <out>/<pack>.safetensors \
    --image <ref>.png --clip-vision <COMFY>/models/clip_vision/clip_vision_h.safetensors \
    --root <A1111> --comfy <COMFY> \
    --json-5a records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json \
    --json-trigger records/cn_ipadapter_plus_vit_h_image_trigger.json \
    --json-gate records/cn_ipadapter_plus_vit_h_gate_256.json
```

---

## 6. Error sources and limits

- **The ComfyUI extension does not take this pack**, and its failure is silent: 162 int8 codes would be read as magnitudes. All acceptance numbers come from the A1111 route.
- **The HSWQ loader's own IP-Adapter runtime was not exercised for these numbers.** It installs this pack in place of the stock path; the measurements here are the stock path plus the A1111 runtime patch.
- **Forge was not measured** (no Forge install on this machine).
- **No render.** No image pair exists for this checkpoint; Section 4 states it rather than implying a picture was looked at.
- **The worst single latent is not monotone in the layers' precision.** The all-INT8 pack measures a worse worst-latent (0.986) than every pack with protected layers (0.939). That is the Resampler's error arithmetic — the all-INT8 errors partly cancel at that latent — which is why it is reported beside the gate and not used as a pass criterion. It does not mean the all-INT8 pack is more accurate layer by layer: the layer-by-layer number is the gate mean, and it moves the other way.
- **`image_proj` mean 0.9957 with worst latent 0.933.** Four attention blocks in series accumulate error; a residual block's error is not the sum of its layers' errors. The pack is measured, not characterised.
- **The trigger search resolves on a coarser perturbation than the drift it looks for.** The `image_proj` floor is 10 % of the codes shifted by 64 LSB; the largest natural candidate response (3.15e-02) is 31 % of the 1.000654e-01 threshold. A trigger below that floor would not be seen on this surface. The `ip_adapter` surface resolves at 1 LSB but does not consume the image, so it cannot carry an image trigger.
- **The scale axis is not usable on this graph.** Every Resampler layer is followed by a LayerNorm, so a uniform scale change is removed by the following normalisation.
- **The sweep is one seed and one fixed input per layer.** The contribution spread is wide (1.5e-04 to 1.9e-09), so the top of the ranking is stable; the ordering inside the middle is not.
- **`image_proj` cannot be protected part-way.** Each of its 22 weights is one ConvRot layer.

---

## 7. Ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_plus_vit_h.py` | 162 groups, max 8.488220e-05, median 4.826412e-05, p90 5.861301e-05, min 1.370495e-08, 0 above 0.15, 2.2 s |
| 2 | `measure_ipadapter_plus_vit_h_contrib.py` | 162 layers, max 1.502143e-04 (`image_proj.proj_in`), median 2.409374e-05, 38.8 s |
| 3 | `convert_ipadapter_plus_vit_h_convrot_int8.py` | 162 INT8, 0 protected, 29 kept as-is, 424,750,694 B, -49.9 % |
| 4 | `ipadapter_plus_vit_h_pack_audit.py` | CN-1 FAITHFUL 162/162, 0 integer differences, scale 0.0, float 29/29 bit-identical, coverage 0/0/0 |
| 5 | `ipadapter_plus_vit_h_route.py` | A1111 162/162 installed (22 + 140), forward `[1,16,2048]` finite, 22 kernel calls; ComfyUI 162 raw codes |
| 6 | `ipadapter_plus_vit_h_acceptance.py` | CN-5a / CN-5b / CN-6: mean 0.999739679, min 0.998887861, worst latent 0.986089059, 5772 kernel calls, PASS |
| 7 | `sweep_ipadapter_plus_vit_h.py` | K = 0 / 1 / 20 / 60 / 162 -> 405.1 / 406.6 / 449.0 / 541.3 / 808.3 MB, all PASS, K = 0 adopted |

---

## 8. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_plus_vit_h_pack_audit.json` | the quantisation replay (162 declared / 162 found, 0 differing) |
| `records/cn_ipadapter_plus_vit_h_route.json` | which loaders take the pack, and what they do with it |
| `records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json` -> `image_proj_surface` | CN-5a on the INT8 route, both surfaces |
| `records/cn_ipadapter_plus_vit_h_gate_256.json` | CN-6, the 256-item gate |
| `records/cn_ipadapter_plus_vit_h_image_trigger.json` | CN-5b, the 56 candidates and both positive-control ladders |
| `records/cn_ipadapter_plus_vit_h_protection.json` | the 162-group drift screen |
| `records/cn_ipadapter_plus_vit_h_contrib.json` | the 162-layer aggregate contribution |
| `records/cn_ipadapter_plus_vit_h_curve.json` -> `curve` | the measured size/quality curve |
