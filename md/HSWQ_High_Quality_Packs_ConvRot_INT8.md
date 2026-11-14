# HSWQ High-Quality Packs - ConvRot INT8, individually built

This is the HSWQ side of the ConvRot INT8 catalogue. The packs here are **not** native
ConvRot INT8: for every one of them, that checkpoint's own layers were ranked by their
contribution to the model's own output, and the layers above the threshold were kept in
FP16 inside a size/quality budget. The kept set is therefore a measurement on that
checkpoint, and it differs from pack to pack.

The **native** ConvRot INT8 packs - plain rotation and quantisation, every runtime Linear
quantised, no per-layer sweep - live in
[`How to quantize Text Encoder and ControlNet.md`](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md),
Section *Finished packs - native ConvRot INT8*.

The method itself (the injection frame, the aggregate contribution, the size/quality
balance, how the kept set is chosen) is stated once in the same document, Section
[*How a pack is selected (the current method)*](How%20to%20quantize%20Text%20Encoder%20and%20ControlNet.md#how-a-pack-is-selected-the-current-method).
This document carries the inventory and the per-model numbers.

---

## ControlNet and model patches

### HSWQ - per-checkpoint measured protection

| Pack | Family | Runtime object | Directory | the gate (measured) | Size | SHA256 (first 16) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `control_v11f1e_sd15_tile_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999708 / 0.999592 | 579.5 MB | `E90A2CB010EA2E18` |
| `control_v11p_sd15_canny_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999673 / 0.999518 | 579.5 MB | `F7E20FA625863F02` |
| `control_v11f1p_sd15_depth_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999764 / 0.999719 | 579.5 MB | `45086721B1FF4643` |
| `control_v11e_sd15_ip2p_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999813 / 0.999768 | 579.5 MB | `43E0884402B1801F` |
| `control_v11e_sd15_shuffle_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999719 / 0.999591 | 579.5 MB | `3A27828EB0373EDA` |
| `control_v11p_sd15_inpaint_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999789 / 0.999737 | 579.5 MB | `53CF9C06CA727847` |
| `instantid_diffusion_pytorch_model_hswq_convrot_int8` | SDXL | ControlNet | `models/controlnet/` | 0.999395 / 0.997491 | 1354.4 MB | `A9FFBD849D007B96` |
| `control_instant_id_sdxl_hswq_convrot_int8` | SDXL | ControlNet | `models/controlnet/` | 0.999395 / 0.997491 | 1354.4 MB | `A9FFBD849D007B96` |
| `control_v11p_sd15_lineart_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999737 / 0.999698 | 579.5 MB | `F32407808652F74E` |
| `control_v11p_sd15_mlsd_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999678 / 0.999466 | 579.5 MB | `1DE1800AE02084C1` |
| `control_v11p_sd15_normalbae_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999686 / 0.999609 | 579.5 MB | `04BE4F6B091CF0C8` |
| `control_v11p_sd15_openpose_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999689 / 0.999391 | 579.5 MB | `6CC82F9ECC3BAE79` |
| `control_v11p_sd15_scribble_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999597 / 0.999354 | 579.5 MB | `1D877C18B3210C6D` |
| `control_v11p_sd15_seg_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999717 / 0.999635 | 579.5 MB | `20F11C5B30F66A53` |
| `control_v11p_sd15_softedge_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999703 / 0.999572 | 579.5 MB | `609B04A9A38668FB` |
| `control_v11p_sd15s2_lineart_anime_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999748 / 0.999707 | 579.5 MB | `D004154F605EBBF1` |
| `ip-adapter_instant_id_sdxl_hswq_convrot_int8` | SDXL | IP-Adapter | `models/controlnet/` | 0.999395 / 0.997491 | 1354.4 MB | `A9FFBD849D007B96` |
| `ip-adapter_pulid_sdxl_hswq_convrot_int8` | SDXL | IP-Adapter | `models/controlnet/` | embeds 0.9999928 / to_kv 0.9999959 | 378.4 MB | `0262C9AFFD20D105` |
| `ip-adapter_sd15_plus_hswq_convrot_int8` | SD1.5 | IP-Adapter | `models/controlnet/` | 0.999886 / 0.998943 | 47.1 MB | `9A0F7CDFAE257595` |
| `ip-adapter-faceid_sd15_hswq_convrot_int8` | SD1.5 | IP-Adapter | `models/controlnet/` | 0.999989 / 0.999967 | 62.0 MB | `35B5942EE163494B` |
| `ioclab_sd15_recolor_hswq_convrot_int8` | SD1.5 | ControlNet | `models/controlnet/` | 0.999861 / 0.999834 | 392.1 MB | `58046C116268A1C9` |
| `qwen_image_canny_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999924905 / 0.9999924096 | 1083.3 MB | `C2899255252B81FD` |
| `qwen_image_depth_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999924109 / 0.9999923262 | 1083.3 MB | `7B82A9360169FA99` |
| `qwen_image_inpaint_diffsynth_controlnet_hswq_convrot_int8` | Qwen-Image | **model patch** | `models/model_patches/` | 0.9999921236 / 0.9999919542 | 1083.3 MB | `471B6AFFC7FCEDAF` |
| `noobaiXLControlnet_epsLineartRealistic_hswq_convrot_int8` | SDXL (NoobAI / Illustrious) | ControlNet | `models/controlnet/` | 0.999825 / 0.999802 (source vs pack, 4 batches) | 1354.4 MB | `86E795137DCF8D28` |

`control_instant_id_sdxl`, `instantid_diffusion_pytorch_model` and `ip-adapter_instant_id_sdxl` are
**the same file under three names** (LFS sha256
`a9ffbd849d007b969d91cadd1409757000f20f69d08b45d95f18243adff456c9` for all three), so they
carry the same measurement. `noobaiXLControlnet` has its sweep, audit, route, execution, render
and trigger records; its gate column quotes the source-against-pack comparison, because no
256-item record exists for it yet and the column says what it is measuring.

`control_instant_id_sdxl`, `instantid_diffusion_pytorch_model` and `ip-adapter_instant_id_sdxl`
are **the same file under three names** (LFS sha256
`a9ffbd849d007b969d91cadd1409757000f20f69d08b45d95f18243adff456c9` for all three), so they
carry the same measurement. `noobaiXLControlnet` has its sweep, audit, route, execution,
render and trigger records; its gate column quotes the source-against-pack comparison,
because no 256-item record exists for it yet and the column says what it is measuring.

A ControlNet in `model_patches/` (or vice versa) does not load. The "model patch" rows go
through the **HSWQ Model Patch Loader**; the plain rows go through the ControlNet loader,
built by the HSWQ loader extension so the INT8 payload is read as INT8.

---

### Per-model guides

Each of these carries the measured numbers for that checkpoint:

- [the per-checkpoint ControlNet numbers (SDXL / Illustrious)](ControlNet_ConvRot_INT8_Trajectory_Impact_Technical_Guide.md)
- [FLUX.1-dev ControlNet Union Pro 2.0](FLUXControlNetUnionPro2_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image-2512 Fun ControlNet Union 2602](QwenImage2512FunControlnetUnion2602_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image ControlNet Inpainting](QwenImageControlNetInpainting_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 tile ControlNet](CN_SD15_Tile_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 canny ControlNet](CN_SD15_Canny_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 depth ControlNet](CN_SD15_Depth_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 ip2p ControlNet](CN_SD15_Ip2p_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 shuffle ControlNet](CN_SD15_Shuffle_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 inpaint ControlNet](CN_SD15_Inpaint_ConvRot_INT8_Impact_Technical_Guide.md)
- [InstantID SDXL ControlNet](CN_InstantID_SDXL_ConvRot_INT8_Impact_Technical_Guide.md)
- [InstantID diffusers layout ControlNet](CN_InstantID_Diffusers_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 lineart ControlNet](CN_SD15_Lineart_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 mlsd ControlNet](CN_SD15_Mlsd_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 normalbae ControlNet](CN_SD15_Normalbae_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 openpose ControlNet](CN_SD15_Openpose_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 scribble ControlNet](CN_SD15_Scribble_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 seg ControlNet](CN_SD15_Seg_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 softedge ControlNet](CN_SD15_Softedge_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5s2 lineart_anime ControlNet](CN_SD15_Lineart_Anime_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 recolor ControlNet (ioclab)](CN_SD15_Recolor_ConvRot_INT8_Impact_Technical_Guide.md)
- [InstantID IP-Adapter (SDXL)](CN_IPAdapter_InstantID_SDXL_ConvRot_INT8_Impact_Technical_Guide.md)
- [PuLID IP-Adapter (SDXL)](CN_IPAdapter_PuLID_SDXL_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 IP-Adapter Plus](CN_IPAdapter_SD15_Plus_ConvRot_INT8_Impact_Technical_Guide.md)
- [SD1.5 IP-Adapter FaceID](CN_IPAdapter_FaceID_SD15_ConvRot_INT8_Impact_Technical_Guide.md)
- [Qwen-Image diffsynth model patch (canny)](QwenImageDiffsynthControlnet_ConvRot_INT8_ModelPatch_Guide.md)
- [Qwen-Image diffsynth model patch (depth)](QwenImageDepthDiffsynthControlnet_ConvRot_INT8_Guide.md)
- [Qwen-Image diffsynth model patch (inpaint)](QwenImageInpaintDiffsynthControlnet_ConvRot_INT8_Guide.md)
- [Z-Image Fun ControlNet model patch (Union / Tile)](ZImageTurboFunControlnet_ConvRot_INT8_ModelPatch_Guide.md)

---

## HSWQ High-Quality Text Encoders (ConvRot INT8, individually built)

### Why there is no universal method here

HSWQ's reverse-hybrid engine is a **universal method**: it measures per-layer trajectory sensitivity on the target checkpoint, ranks the layers, and selects FP16 protection inside a fixed budget. That method applies to image-generation backbones (SDXL / Z Image / Krea2 UNet-DiT), which exist as **many per-checkpoint variants** - SDXL alone was tested on **30+ checkpoints**, and every merge or finetune is a new target whose ranking must be re-measured. A general method pays off precisely because there are so many of them.

Text encoders are **not** per-checkpoint models. Each architecture (CLIP-L, CLIP-G, T5-XXL, Qwen2.5-VL, Qwen3, ...) is a **single standard weight layout** with no per-checkpoint derivation. There is nothing to re-measure per model, so a large universal engine is not justified - the same applies to a ControlNet standard and to a SAM release, and it is why the whole chapter above is a per-model track rather than an application of the engine.

**These are HSWQ builds, not native ConvRot INT8.** Every row below comes from a
measurement on that standard: the layers are ranked by their contribution to the
model's own output, and the FP16-kept set is what the threshold returned for that
standard. A kept count of **0** is a measured result (nothing crossed the
threshold, section *Rule 4*), not an unmeasured pack - the native ConvRot INT8
product, which quantises every runtime Linear with no sweep at all, is a different
thing and is listed separately in the ControlNet section.

Instead, each text-encoder standard was handled **individually** - but the criterion is no longer static. See **Section: How a pack is selected (the current method)** below: the protection set comes from the measured **aggregate contribution**, the size/quality balance sets the threshold, and the kept tensors are stored FP16. Static kurtosis / outlier-ratio screening is retired as a *criterion*; the per-standard converter scripts under `clip_convert/` implement the current method.

### Finished packs (Hugging Face)

| Pack | Standard | ConvRot INT8 | FP16 kept | Gate (256 items, seed 42) | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip_l_hswq_convrot_int8.safetensors` | CLIP-L | 63 of 73 | 10 | 0.999876 / 0.999841 | 135.5 MB |
| `CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors` | OpenAI CLIP ViT-L/14 | 129 (text 43, vision 84) | 90 (text 30, vision 60) | text 0.999962 / 0.999908, vision 0.999931 / 0.999797 | 658.9 MB |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (extended text) | 130 (text 44, vision 84) | 90 (text 30, vision 60) | text 0.999803 / 0.999684, vision 0.999687 / 0.999367 | 490.6 MB |
| `ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors` | CLIP ViT-L/14 (HF layout) | 127 (text 43, vision 84) | 92 (text 30, vision 60) | text 0.999964 / 0.999906, vision 0.999803 / 0.996378 | 646.7 MB |
| `flan_t5_xxl_hswq_convrot_int8.safetensors` | Flan-T5-XXL | 312 of 432 (enc 128, dec 184) | 120 (enc 40, dec 80) | 0.999934 | 14056.6 MB |
| `Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors` | Qwen3 (0.6B) | 157 of 197 | 40 | 0.999502 / 0.998980 | 655.2 MB |
| `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` | Qwen3-VL (4B) | 338 of 358 | 20 | 0.999602 / 0.997877 | 4914.0 MB |
| `Huihui-Qwen3-VL-4B-Instruct-abliterated_hswq_convrot_int8.safetensors` | Qwen3-VL (4B) | 328 of 358 | 30 | 0.999578 / 0.999111 | 5061.3 MB |
| `qwen3_4b_abliterated_hswq_convrot_int8.safetensors` | Qwen3 (4B) | 233 of 253 | 20 | 0.999630 / 0.997430 | 4168.3 MB |
| `Qwen2.5-VL-7B-Instruct-abliterated_converted_hswq_convrot_int8.safetensors` | Qwen2.5-VL (7B) | 320 of 360 | 40 | 0.999325 / 0.997294 | 9289.6 MB |
| `Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP_hswq_convrot_int8.safetensors` | Qwen3.5 (9B, hybrid + vision + MTP) | 369 of 369 | 0 | conditioning 0.999387 / 0.996488, generate 0.999453 / 0.998215 | 9217.9 MB |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | 41 of 221 | 180 | relMSE 3.51e-05 | 1657.6 MB |

**ConvRot W4A8 packs.** A different format (`asym_w4a8_int8`: 4-bit weight, group 16, convrot
256), so these are listed apart from the INT8 packs above.

| Pack | Standard | ConvRot W4A8 | Kept | Gate (256 items, seed 42) | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `wevisone_vl_4b_hswq_convrot_w4a8.safetensors` | WeVisOne-VL (4B, Qwen3-VL + matte decoder) | 326 of 370 | 32 by drift rank + `embed_tokens`, `lm_head`, `mask_decoder` (11) | 0.995055 / 0.987840 | 4483.7 MB |

### Built, not published

Two CLIP packs were built and measured but are **not on any Hugging Face repository**;
they exist locally only. They are listed apart so the table above stays a table of
downloadable packs.

| Pack | Standard | ConvRot INT8 | FP16 kept | Gate (256 items, seed 42) | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CLIP-ViT-bigG-14-laion2B-39B-b160k_convrot_int8.safetensors` | CLIP ViT-bigG/14 | 253 of 288 | 35 | 0.999916 / 0.999741 | 2180.1 MB |
| `CLIP-ViT-H-14_hswq_convrot_int8.safetensors` | CLIP ViT-H/14 (vision tower, `models/clip_vision/`) | 92 of 192 | 100 | 0.999968 / 0.999906 | 1015.6 MB |

Download:

- **[ussoewwin/Clip-files-ConvRot-INT8](https://huggingface.co/ussoewwin/Clip-files-ConvRot-INT8)** - CLIP-L, OpenAI CLIP ViT-L/14 (CLIP-SAE), Long CLIP ViT-L/14, ViT-L-14 GmP (HF layout), Flan-T5-XXL, Josiefied-Qwen3-0.6B, WeVisOne-VL W4A8
- **[ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8](https://huggingface.co/ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8)** - Qwen3-4B, Qwen3-VL-4B-Thinking, Huihui-Qwen3-VL-4B-Instruct
- **[ussoewwin/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP](https://huggingface.co/ussoewwin/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP)** - Qwen3.5-9B (INT8 pack next to its BF16 source)
- **[ussoewwin/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/ussoewwin/Qwen2.5-VL-7B-Instruct-abliterated)** - Qwen2.5-VL-7B (INT8 pack next to its BF16 source)
- **[ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8)** - SAM 3.1 Multiplex

### Loading

Place the packs in `ComfyUI/models/clip/` and load with the standard ComfyUI `CLIPLoader` / `DualCLIPLoader` / `TripleCLIPLoader`. For Llama-family text encoders (Qwen3 / Qwen3-VL / Qwen3.5) ComfyUI detects the `comfy_quant` layers inside the load path, so no extra loader is required. For the OpenAI CLIP family (CLIP-L / CLIP-SAE) the loader must receive the pack's `quantization_metadata` (the same mechanism the SDXL CLIP INT8 path already uses); see the **ComfyUI-HSWQ-Loader-and-Tools** extension.

---

## HSWQ High-Quality SAM 3.1 (ConvRot INT8, individually built)

### Why SAM 3.1 belongs here

Like text encoders, SAM 3 / SAM 3.1 are **foundation segmentation models**, not per-checkpoint image-generation backbones. Each version ships a **single standard weight layout**, so there is no per-checkpoint measurement to redo and no universal engine to build. They are therefore handled **individually**: the risk of every Linear layer was measured once on the standard checkpoint (kurtosis / outlier ratio), the fat-tail layers were kept in FP16, and the pack was built with a per-standard converter.

> [!IMPORTANT]
> **SAM 3 (classic / `sam3.pt`) is technically the same kind of case and should be built and published the same way, but it cannot be published for license reasons. Only SAM 3.1 Multiplex is published.**

### Finished pack (Hugging Face)

| Pack | Standard | ConvRot INT8 | FP16 kept | Gate | Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sam3.1_multiplex_hswq_convrot_int8.safetensors` | SAM 3.1 Multiplex | 41 of 221 | 180 | relMSE 3.51e-05 (the tracker and detector paths) | 1657.6 MB |

> The 41/180 split is the shipped pack, re-audited after the injection fix: a
> release search over all 401 candidates found **0** layers releasable inside
> the gate (the cheapest release costs +9.67e-04 relMSE against a 1.0e-04
> gate), so the pack cannot be made smaller. The earlier "538 INT8 / 113 kept"
> figures came from the defective drift and are superseded.

Download:

- **[ussoewwin/SAM3.1-HSWQ-ConvRot-INT8](https://huggingface.co/ussoewwin/SAM3.1-HSWQ-ConvRot-INT8)**

### Loading

Place the pack in `ComfyUI/models/unet/` (or `models/diffusion_models/`) and load it with the standard **`UNetLoader`** / **`Load Diffusion Model`**; connect to the SAM segmentation / video-tracking nodes. The pack keeps the fused QKV in-projection split into `q_proj` / `k_proj` / `v_proj` (the loader expects the split layout) and drops the per-block `freqs_cis` buffers.

---

