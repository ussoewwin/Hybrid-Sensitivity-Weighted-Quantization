# SDXL IP-Adapter Plus (ViT-H) (`ip-adapter-plus_sdxl_vit-h`) ConvRot INT8 - Quantization and Score Report

**Checkpoint.** `models/controlnet/ip-adapter-plus_sdxl_vit-h.safetensors` (847,517,512 B, 191 tensors: `image_proj` 51 / `ip_adapter` 140)
**Pack.** `models/controlnet/ip-adapter-plus_sdxl_vit-h_hswq_convrot_int8.safetensors` (424,750,694 B, **−49.9 %**)
**SHA256 (source).** `3F5062B8400C94B7159665B21BA5C62ACDCD7682262743D7F2AEFEDEF00E6581`
**SHA256 (pack).** `CA0FF3706DC3673538EEC2683774E384FE3145F2191F2006E03862A7D3C2DB7A`

**Composition.** `image_proj` 22 Linears + `ip_adapter` 140 Linears in INT8 (**162**) · **FP16 kept 29** (`latents`, the LayerNorm vectors, the biases — none of them a quantiser input) · 515 tensors in the pack.

**The quantisation record.** `records/cn_ipadapter_plus_vit_h_conversion.json` -> `arguments` is the record of the conversion itself, not of the audit that follows it: the arguments it ran with (`drift_threshold` 0.15, `groupsize` 256, `protect_count` 0), the per-layer outcome (162 int8, 0 protected, 29 kept), and the hashes on both ends. Re-running it from that record reproduced the shipped pack **bit for bit** (424,750,694 B, sha256 `ca0ff370…`).

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

### 0.3 Forge is installed and reads the marker; its runtime is absent

`backend/operations_mixed_precision.py` pops `comfy_quant` and dispatches on the `convrot` flag with `ck.int8_linear(convrot=True)`. Forge's install is present on this machine: `launch.py`, `webui.py`, `backend/operations_mixed_precision.py` and `extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py` are all present. The pack's stamps are exactly the form that reader keys on: **162 of 162 are `int8_tensorwise`, `convrot` true, `convrot_groupsize` 256** (`records/cn_ipadapter_plus_vit_h_forge_route.json` -> `pack_stamps`).

The route could not be executed, and the reason is measured rather than assumed: Forge's own Python runtime is absent. `venv\Scripts\python.exe`, `venv\bin\python`, `system\python\python.exe` and `python.exe` were each checked and none exists (`records/cn_ipadapter_plus_vit_h_forge_route.json` -> `python_runtime.any_present`).

### 0.4 The loader package's own node keeps the codes INT8

`HSWQ Load IPAdapter Model` (`nodes/hswq_load_ipadapter_model.py`) loads the pack, installs the runtime patch (`patches/ipadapter_convrot_int8.py`) and lets the ConvRot INT8 GEMM evaluate the codes. Measured on this pack (`audit/ipadapter_plus_vit_h_hswq_route.py`): the patch takes (`IPAdapter` replaced true, `To_KV` replaced true), **162 of 162 int8 layers install (image_proj 22, ip_adapter 140)**, and the two forwards fire **162 INT8 kernel calls**. Output against the source's float weights on the same input: `image_proj` cosine mean **0.999526739** / min **0.998086035**, over 140 `to_kv` projections mean >= **0.999972105** / min >= **0.999840677**, all finite (`records/cn_ipadapter_plus_vit_h_hswq_route.json` -> `image_proj`, `to_kv`, `int8_kernel_calls`). The source side builds **0** ConvRot layers, so the comparison is float against INT8 on one graph.

### 0.5 In ComfyUI, the loader package's node is the route that takes the pack

`HSWQ Load IPAdapter Model` (`nodes/hswq_load_ipadapter_model.py`, in the loader package) is the node a ComfyUI user connects to the IP-Adapter apply node. Measured through that node's own `load_ipadapter_model` (`audit/ipadapter_plus_vit_h_comfyui_route.py`): the node takes the ConvRot branch (`module_convrot_branch_loaded` true), the runtime patch is applied (`patch_applied` true, `To_KV` replaced true), and the adapter built from what the node returned carries **22 of 22 `image_proj` and 140 of 140 `ip_adapter` layers on the INT8 path** (source side: 0 and 0).

Outputs against the source's float weights: `image_proj` cosine mean **0.999526739** / min **0.998086035**; the 140 `to_kv` projections on the same input mean >= **0.999972105** / min >= **0.999840677**; end to end (each side on its own `image_proj` output) mean >= **0.999393105** / min >= **0.997277856**; **302 INT8 kernel calls** behind those comparisons, all finite (`records/cn_ipadapter_plus_vit_h_comfyui_route.json` -> `installed_int8_layers`, `image_proj`, `to_kv`, `int8_kernel_calls`).

So on this machine the pack has a ComfyUI route through the loader package, separate from the stock extension (section 0.1) and from A1111 (section 0.2).

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

| rank | layer | contribution | share of the 162 measured groups |
| :--- | :--- | ---: | ---: |
| 1 | `image_proj.proj_in` | 1.502143e-04 | 4.086 % |
| 2 | `ip_adapter.25.to_k_ip` | 4.238458e-05 | 1.153 % |
| 3 | `ip_adapter.21.to_k_ip` | 3.436726e-05 | 0.935 % |
| 4 | `ip_adapter.117.to_k_ip` | 3.356947e-05 | 0.913 % |
| 5 | `ip_adapter.131.to_k_ip` | 3.340939e-05 | 0.909 % |
| 6 | `ip_adapter.61.to_v_ip` | 3.333808e-05 | 0.907 % |
| median | 162 layers | 2.409374e-05 | 0.655 % |

The curve, measured on the fixed 256-item set through the INT8 route (K = layers kept in FP16):

| K | INT8 layers | FP16 kept | size | gate mean | gate min | p5 | worst single latent | `ip_adapter` mean | mean gain / gate margin | pack sha256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **0** | 162 | 0 | 405.1 MB | 0.999739679 | 0.998887861 | 0.999466956 | 0.986089059 | 0.999922318 | - | `ca0ff3706dc3` |
| 1 | 161 | 1 | 406.6 MB | 0.999837537 | 0.995995511 | 0.999741980 | 0.938652834 | 0.999922318 | 9.8e-05 / 9.84e-03 = 1.0 % | `b8a95bc95992` |
| 20 | 142 | 20 | 449.0 MB | 0.999837537 | 0.995995511 | 0.999741980 | 0.938652834 | 0.999933195 | 9.8e-05 / 9.84e-03 = 1.0 % | `d920fa95ed6a` |
| 60 | 102 | 60 | 541.3 MB | 0.999837537 | 0.995995511 | 0.999741980 | 0.938652834 | 0.999955528 | 9.8e-05 / 9.84e-03 = 1.0 % | `09c9922d7364` |
| 162 | 0 | 162 | 808.3 MB | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 | identity | `087db0b3b6d3` |

**Why K = 1, 20 and 60 report the same `image_proj` figures.** The protection sets differ, but not on this surface: only `image_proj.proj_in` is inside the top 60 of the contribution ranking (share 4.086 %), so K = 1 protects `proj_in` and K = 20 / 60 protect `proj_in` plus 19 / 59 `ip_adapter` layers. The `image_proj` graph is therefore identical in the three packs, and its gate figures can only be equal; the `ip_adapter` surface moves with K (0.999922318 -> 0.999933195 -> 0.999955528), which is where the protection is actually spent (`records/cn_ipadapter_plus_vit_h_contrib.json` -> `contributions`).

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

**The same gate on the ComfyUI route** (`audit/ipadapter_plus_vit_h_comfyui_gate.py`, through the loader node): mean **0.999833121** / min **0.998484056** / p5 **0.999705834** over the same 256 items, **5632 INT8 kernel calls**, PASS. Against the A1111 figures: delta mean **+9.34e-05**, delta min **-4.04e-04** (`records/cn_ipadapter_plus_vit_h_comfyui_gate.json` -> `agreement_with_a1111`). Two harnesses for one observable, and the agreement is measured rather than assumed.

---

## 4. Visible-sample check

**Measured on the ComfyUI route through the loader node** (`audit/ipadapter_plus_vit_h_render.py`): the source adapter and the pack, same prompt, seed 42, 12 steps, cfg 7.0, 512x512, DPM++ 2M / Karras, weight 0.8, the same reference image, both through `HSWQ Load IPAdapter Model` -> `IPAdapterAdvanced` on a separate server (port 8189, `--lowvram`; the owner's own instance untouched).

| arm | file | bytes | read as an image |
| :--- | :--- | ---: | :--- |
| source | `records/cn_ipadapter_plus_vit_h_comfyui_source.png` | 356,817 | finished anime-style image: a woman in a red hooded coat in a city street, both eyes open, hands at her sides; heavy saturated colour, no noise |
| pack | `records/cn_ipadapter_plus_vit_h_comfyui_pack.png` | 312,382 | the same scene and character with the same coat and pose; the eye shape, the hair strands and the background panels differ slightly; no noise |

Pixel comparison: mean |diff| **32.578** / max 255 / p99 204, cosine **0.955370**, PSNR **13.91 dB**, 0.01 % of pixels identical (`records/cn_ipadapter_plus_vit_h_render.json` -> `pixel_comparison`).

Read on the images, not on the numbers: both arms are finished images; the difference is local (face, hair, background panels), not a collapse to noise. The pixel distance is nevertheless large for a pack the gate passes at 0.9989 - the gate measures the injected 16x2048 stack, and at 12 steps the sampler amplifies that difference. That is what was measured; it is not turned into a claim that the two images are interchangeable.

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

- **The stock ComfyUI extension does not take this pack**, and its failure is silent: 162 int8 codes would be read as magnitudes. The pack is taken in ComfyUI by the loader package's own node (section 0.5); the acceptance numbers of section 3 remain the A1111 route's.
- **The HSWQ loader's own IP-Adapter runtime is measured separately, not for the gate.** That route installs the pack in place of the stock path (Section 0.4: 162 kernel calls, `image_proj` 0.999526739 / 0.998086035, `to_kv` >= 0.999972105 / 0.999840677); the gate numbers in Section 3 remain the A1111 route's.
- **Forge was not executed.** Its install is present and the pack's 162 stamps are the form its reader keys on; its Python runtime is absent, and the four interpreter paths were each checked (Section 0.3).
- **The render is one seed, one prompt, 12 steps, 512x512, `--lowvram`.** Section 4 reports what was looked at; the pixel distance (cosine 0.9554) is larger than the gate's margin, so the two arms are the same scene but not the same picture.
- **The worst single latent is not monotone in the layers' precision.** The all-INT8 pack measures a worse worst-latent (0.986) than every pack with protected layers (0.939). That is the Resampler's error arithmetic — the all-INT8 errors partly cancel at that latent — which is why it is reported beside the gate and not used as a pass criterion. It does not mean the all-INT8 pack is more accurate layer by layer: the layer-by-layer number is the gate mean, and it moves the other way.
- **`image_proj` mean 0.9957 with worst latent 0.933.** Four attention blocks in series accumulate error; a residual block's error is not the sum of its layers' errors. The pack is measured, not characterised.
- **The trigger search resolves on a coarser perturbation than the drift it looks for.** The `image_proj` floor is 10 % of the codes shifted by 64 LSB; the largest natural candidate response (3.15e-02) is 31 % of the 1.000654e-01 threshold. A trigger below that floor would not be seen on this surface. The `ip_adapter` surface resolves at 1 LSB but does not consume the image, so it cannot carry an image trigger.
- **The scale axis is not usable on this graph.** Every Resampler layer is followed by a LayerNorm, so a uniform scale change is removed by the following normalisation.
- **The sweep is one seed and one fixed input per layer.** The contribution spread is wide (1.5e-04 to 1.9e-09), so the top of the ranking is stable; the ordering inside the middle is not.
- **`image_proj` cannot be protected part-way.** Each of its 22 weights is one ConvRot layer.

---

## 7. Ledger

Times are the record files' own last-write times, read from the filesystem on 2026-11-16 (a record that was rewritten later carries the later time; the time is evidence of when the file last changed, not of when the step first ran).

| step | tool | result | record | file time |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `diag_impact_ipadapter_plus_vit_h.py` | 162 groups, max 8.488220e-05, median 4.826412e-05, p90 5.861301e-05, min 1.370495e-08, 0 above 0.15, 2.2 s | `cn_ipadapter_plus_vit_h_protection.json` | 2026-11-16 04:46:24 |
| 2 | `measure_ipadapter_plus_vit_h_contrib.py` | 162 layers, max 1.502143e-04 (`image_proj.proj_in`), median 2.409374e-05, 38.8 s | `cn_ipadapter_plus_vit_h_contrib.json` | 2026-11-16 04:46:24 |
| 3 | `convert_ipadapter_plus_vit_h_convrot_int8.py` | 162 INT8, 0 protected, 29 kept as-is, 424,750,694 B, -49.9 % | `cn_ipadapter_plus_vit_h_conversion.json` | 2026-11-16 10:52:14 |
| 4 | `ipadapter_plus_vit_h_pack_audit.py` | CN-1 FAITHFUL 162/162, 0 integer differences, scale 0.0, float 29/29 bit-identical, coverage 0/0/0 | `cn_ipadapter_plus_vit_h_pack_audit.json` | 2026-11-16 04:52:52 |
| 5 | `ipadapter_plus_vit_h_route.py` | A1111 162/162 installed (22 + 140), forward `[1,16,2048]` finite, 22 kernel calls; ComfyUI 162 raw codes | `cn_ipadapter_plus_vit_h_route.json` | 2026-11-16 02:59:33 |
| 6 | `ipadapter_plus_vit_h_acceptance.py` | CN-5a / CN-5b / CN-6: mean 0.999739679, min 0.998887861, worst latent 0.986089059, 5772 kernel calls, PASS | `cn_ipadapter_plus_vit_h_pack_source_output_diff.json`, `_image_trigger.json`, `_gate_256.json` | 2026-11-16 04:46:37 / 04:47:10 / 04:49:00 |
| 7 | `sweep_ipadapter_plus_vit_h.py` | K = 0 / 1 / 20 / 60 / 162 -> 405.1 / 406.6 / 449.0 / 541.3 / 808.3 MB, all PASS, K = 0 adopted; K = 0 rebuilt byte-identical, hashes and margins added | `cn_ipadapter_plus_vit_h_curve.json` | 2026-11-16 12:44:25 |
| 8 | `ipadapter_plus_vit_h_forge_route.py` | Forge install present, 162/162 stamps are the form its reader keys on, no Forge Python runtime | `cn_ipadapter_plus_vit_h_forge_route.json` | 2026-11-16 12:44:25 |
| 9 | `ipadapter_plus_vit_h_hswq_route.py` | patch applied, 162/162 installed, 162 kernel calls, `image_proj` 0.999526739 / 0.998086035, `to_kv` >= 0.999972105 / 0.999840677 | `cn_ipadapter_plus_vit_h_hswq_route.json` | 2026-11-16 12:44:25 |
| 10 | Hub upload | card then pack, 424,750,694 B, LFS sha256 matched, 37 files | `cn_ipadapter_plus_vit_h_hf_upload.json` | 2026-11-16 11:46:59 |
| 14 | `ipadapter_plus_vit_h_comfyui_gate.py` | the 256-item gate on the ComfyUI route: mean 0.999833121 / min 0.998484056, 5632 kernel calls, PASS | `cn_ipadapter_plus_vit_h_comfyui_gate.json` | 2026-11-16 (run in this session) |
| 13 | `ipadapter_plus_vit_h_render.py` | two ComfyUI renders through the loader node (source / pack), 512x512, seed 42, 12 steps; both read as finished images, pixel cosine 0.955370 | `cn_ipadapter_plus_vit_h_render.json`, `cn_ipadapter_plus_vit_h_comfyui_{source,pack}.png` | 2026-11-16 (run in this session) |
| 12 | `ipadapter_plus_vit_h_comfyui_route.py` | the loader package's node inside ComfyUI: ConvRot branch, 162/162 installed, 302 kernel calls, `image_proj` 0.999526739 / 0.998086035 | `cn_ipadapter_plus_vit_h_comfyui_route.json` | 2026-11-16 (run in this session) |
| 11 | `audit/run_all_gates.py --regen` | ALL GATES PASS (procedure, citations, selftest, SHA256SUMS, markdown) | `gate_run_2026-11-16.json` | 2026-11-16 (run in this session) |

## 8. Files

| file | what it is |
| :--- | :--- |
| `records/cn_ipadapter_plus_vit_h_pack_audit.json` | the quantisation replay (162 declared / 162 found, 0 differing) |
| `records/cn_ipadapter_plus_vit_h_route.json` | which loaders take the pack, and what they do with it |
| `records/cn_ipadapter_plus_vit_h_pack_source_output_diff.json` -> `image_proj_surface` | CN-5a on the INT8 route, both surfaces |
| `records/cn_ipadapter_plus_vit_h_gate_256.json` | CN-6, the 256-item gate |
| `records/cn_ipadapter_plus_vit_h_image_trigger.json` | CN-5b, the 56 candidates and both positive-control ladders |
| `records/cn_ipadapter_plus_vit_h_protection.json` | the 162-group drift screen |
| `records/cn_ipadapter_plus_vit_h_contrib.json` | the aggregate-contribution ranking (the criterion) |
| `records/cn_ipadapter_plus_vit_h_curve.json` | the curve, with the floor and the top, the pack hashes and the margin ratios |
| `records/cn_ipadapter_plus_vit_h_forge_route.json` | the Forge route: install files, the pack's stamps, the missing runtime |
| `records/cn_ipadapter_plus_vit_h_hswq_route.json` | the loader package's own route: patch, installs, kernel calls, cosines |
| `records/cn_ipadapter_plus_vit_h_comfyui_route.json` | the same node inside ComfyUI, through its `load_ipadapter_model` |
| `records/cn_ipadapter_plus_vit_h_render.json` | CN-4: both arms, the settings, the pixel comparison and the read |
| `records/cn_ipadapter_plus_vit_h_comfyui_gate.json` | CN-6 on the ComfyUI route, with the agreement against the A1111 run |
| `records/cn_ipadapter_plus_vit_h_comfyui_source.png` / `_pack.png` | the two renders |
| `records/cn_ipadapter_plus_vit_h_contrib.json` | the 162-layer aggregate contribution |
| `records/cn_ipadapter_plus_vit_h_curve.json` -> `curve` | the measured size/quality curve |
