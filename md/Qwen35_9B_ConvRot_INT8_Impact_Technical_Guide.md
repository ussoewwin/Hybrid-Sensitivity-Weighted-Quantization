# Qwen3.5-9B Defiant-Fable (NEO IMATRIX MAX MTP) - ConvRot INT8 Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`How to quantize Text Encoder and ControlNet.md`](How to quantize Text Encoder and ControlNet.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this model.

**Source.** `models/clip/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP .safetensors` (18411.9 MB, 775 keys, 369 finite 2D Linear layers)
**Pack.** `models/clip/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP_hswq_convrot_int8.safetensors` (9217.9 MB, **-49.9 %**)
**SHA256.** `3BA3902442B7C0085B5C6873A44B5900F16D3125B7707AB66080650B8B724082`
**Method.** impact-driven, with the FP16-kept set chosen on the **aggregate** output.

---

## 1. The model

This is a **hybrid** checkpoint, not a plain transformer: 32 layers alternate
GatedDeltaNet (`linear_attn`) with full GatedAttention (`self_attn`), plus a
**vision tower** (27 blocks) and an **MTP draft head**, and it carries its own
vocabulary head.

The structure, measured from the checkpoint:

| tower | the keys | the 2D Linear | the bytes |
| :--- | :--- | :--- | :--- |
| `model.language_model` | 426 | 249 | 14.783 GB |
| `model.visual` | 333 | 111 | 0.849 GB |
| `mtp` | 15 | 8 | 0.453 GB |
| `lm_head` (root-level) | 1 | 1 | 1.895 GB |
| **total** | **775** | **369** | **17.980 GB** |

The vision tower's `linear_fc1` / `qkv` have `in_features` 4304 / 1152, which
are **not** divisible by 256; they take group size 16 / 64. All 369 candidates
find a power-of-4 group size, so no layer falls back to plain INT8.

The runtime is ComfyUI's `comfy.text_encoders.qwen35`: the clip is
`Qwen35TEModel_`, the `Qwen35ClipModel` is at `.qwen35_9b`, the language tower
is at `.qwen35_9b.transformer.model` (`Qwen35Transformer`, a `Llama2_`), the
vision tower at `.qwen35_9b.transformer.visual` (`Qwen35VisionModel`) and the
draft head at `.qwen35_9b.transformer.mtp` (`MTPHead`).

The loader renames `model.language_model.` -> `model.`, `model.visual.` ->
`visual.` and `lm_head.` -> `model.lm_head.` (measured: `comfy/sd.py` L1966).
The pack is written under the **checkpoint's own names**.

---

## 2. The model has two consumers, so it has two measurements

A text encoder is **not** a denoiser: it produces a text embedding (and, for a
multimodal prompt, the visual tokens spliced into it). So what is measured is
the drift of the model's own output when a single layer is replaced by its
ConvRot INT8 reconstruction.

This checkpoint has two distinct consumers, and a measurement that only runs the
first reports the second as **exactly zero**:

| the path | the layers | the observable | the script |
| :--- | :--- | :--- | :--- |
| the **conditioning** path | 361 | `embed_tokens` -> `layers.N` -> `norm` hidden state, plus the vision tower's merged embeddings | `sdxl/diag_impact_qwen35_9b.py` |
| the **generate** path | 9 | the `lm_head` logits and the MTP head's output | `sdxl/diag_impact_qwen35_9b_gen.py` |

`lm_head` and `mtp.*` are outside the conditioning forward, so the first
measurement returns them as exactly `0.0`. That is a correct reading of *that*
observable - and useless as a basis for quantizing them. They are not dead
weight: **`lm_head` alone is the single largest tensor in the checkpoint,
1.895 GB of 17.98 GB**, and `mtp` is 0.453 GB. Both drive the `generate` path
(`Qwen35.generate` / `_generate_mtp`), which ComfyUI uses when this text
encoder is asked to produce text.

---

## 3. The impact measurements

### 3.1 The conditioning path

Script: `sdxl/diag_impact_qwen35_9b.py` -> `impact/impact_qwen35_9b.json`

| item | value |
| :--- | :--- |
| the measured layers | **361 / 361** |
| the reference output | 81,920 elements, finite |
| the sweep time | 286 s (0.79 s/layer, CPU) |

**The injection is the un-rotated reconstruction.** The pack stores the
rotated weight (`W_hat = W H^T`) and the kernel rotates the **activation**, so
the quantization error that reaches the output is `(q*s) H^T - W` in the
**original** frame. Injecting the rotated frame measures a destroyed layer
(measured on the FLUX ControlNet: the pack's kernel 1.74e-04, the
rotated-frame injection 1.89).

**Drift distribution**

| tower | n | median | max |
| :--- | :--- | :--- | :--- |
| `model.language_model` | 250 | 5.3974e-05 | 2.9775e-04 |
| `model.visual` | 111 | 4.2129e-07 | 4.2923e-06 |

**The language tower is about 128 times more sensitive than the vision
tower** (median 5.40e-05 against 4.21e-07) - a fact a static weight statistic
could not have produced. The vision tower is effectively impervious.

The most sensitive single layers:

| the layer | the drift |
| :--- | :--- |
| `model.layers.0.linear_attn.out_proj` | 2.9775e-04 |
| `model.layers.0.linear_attn.in_proj_qkv` | 1.3523e-04 |
| `model.layers.0.mlp.down_proj` | 1.2313e-04 |
| `model.embed_tokens` | 1.1680e-04 |
| `model.layers.2.linear_attn.in_proj_qkv` | 1.0717e-04 |
| `model.layers.1.linear_attn.out_proj` | 1.0572e-04 |
| `model.layers.0.mlp.gate_proj` | 1.0558e-04 |

The ranking is **depth-monotone**: all 200 of the top-200 are language layers,
and the **first vision layer appears only at rank 250**. `model.lm_head`
measures exactly 0.0 (outside this path).

The 250 language entries are the 249 2D Linear layers plus `embed_tokens`; the
vision tower contributes 111, so 250 + 111 = 361.

### 3.2 The generate path

Script: `sdxl/diag_impact_qwen35_9b_gen.py` -> `impact/impact_qwen35_9b_gen.json`

| the layer | group size | relMSE | cos |
| :--- | :--- | :--- | :--- |
| `model.lm_head` | 256 | 3.7409e-05 | 0.999689102 |
| `mtp.fc` | 256 | 1.6898e-04 | 0.999903262 |
| `mtp.layers.0.mlp.down_proj` | 256 | 6.2970e-05 | 0.999957085 |
| `mtp.layers.0.mlp.gate_proj` | 256 | 2.0452e-05 | 0.999978065 |
| `mtp.layers.0.mlp.up_proj` | 256 | 2.0576e-05 | 0.999977946 |
| `mtp.layers.0.self_attn.k_proj` | 256 | 2.2686e-05 | 0.999976218 |
| `mtp.layers.0.self_attn.o_proj` | 256 | 1.0917e-04 | 0.999933779 |
| `mtp.layers.0.self_attn.q_proj` | 256 | 4.6214e-05 | 0.999965549 |
| `mtp.layers.0.self_attn.v_proj` | 256 | 3.8536e-05 | 0.999968648 |

Every one of the nine is quantizable on its own. The most exposed is `mtp.fc`
(1.69e-04), which is the draft head's own projection from the concatenated
(embedding, hidden) pair.

---

## 4. The selection criterion, and the pack

The per-layer drift is a **screen**, not the selection criterion. It measures
the **weight** error only (the reconstruction is placed in the module and the
plain forward is used), while the pack's real path also quantizes the
**activation** for that layer, and that is where most of the error comes from.

So the pack was built **all-INT8 first** (0 FP16-kept) and the gate was measured
on the real runtime path, through the loader package's `patches/clip_int8_gemm`
arm (which makes the real `int8_linear` fire; measured **157,440** calls over
the 256-item set).

**Rule 4 applies in the "floor" direction**: past some threshold every
ConvRot-eligible layer is already quantized and the pack stops shrinking, and
protecting the remainder adds size without buying quality. Here the all-INT8
extreme is *already* far above the gate, so the protection set is **empty**.

The candidates were measured on the full 256-item acceptance set (seed 42)
against the BF16 source, on the **same ops class** for the baseline and the
pack:

| the FP16-kept | INT8 layers | the conditioning mean | the conditioning min | the generate min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **369** | **0.999387** | **0.996488** | **0.998215** | **9217.9 MB** |

The conditioning min (0.996488) is **17 times the gate's margin** above the
0.98 floor, and the generate min (0.998215) clears 0.99. There is nothing for a
protection set to buy.

### The pack

Built with `clip_convert/convert_qwen35_9b_impact_convrot_int8.py`:

```
--input <BF16 model> --output <pack>
```

(no `--protect-list`: the all-INT8 extreme is the adopted point)

| item | value |
| :--- | :--- |
| the source | 18411.9 MB |
| the pack | **9217.9 MB** (-49.9 %) |
| the SHA256 | `3BA3902442B7C0085B5C6873A44B5900F16D3125B7707AB66080650B8B724082` |
| the tensors | 775 + 369 scales + 369 `comfy_quant` = 1513 |
| the quantized layers | **369** (of 369) |
| the FP16-kept layers | **0** |
| the group size | 256 (260 layers), 64 (82), 16 (27) |
| the `int8_linear` calls over the 256-item set | **157440** |
| the output | finite |
| the conditioning cos mean / min / p5 (256 items, seed 42) | **0.999387 / 0.996488 / 0.998989** |
| the generate cos mean / min (256 items, seed 42) | **0.999453 / 0.998215** |
| the gate (mean >= 0.99, min >= 0.98) | **PASS** |

The pack was written twice (once as the all-INT8 diagnostic, once under the
final name) and the two files are **byte-identical** (same SHA256), so the
builder is deterministic.

---

## 5. Loading

Place the pack in `ComfyUI/models/clip/` and load it with the standard
ComfyUI `CLIPLoader`. Qwen3.5 is a Llama-family encoder: ComfyUI detects the
`comfy_quant` layers inside the load path
(`comfy.sd.detect_te_model` returns `TEModel.QWEN35_9B`), so **no extra loader
is required**. No `config.json` or tokenizer needs to sit beside the pack.

> The source file's own name carries a stray space (`...MAX-MTP .safetensors`);
> the pack's name does not. Renaming the source is optional and does not affect
> the pack.

---

## 6. Files

| file | role |
| :--- | :--- |
| `sdxl/diag_impact_qwen35_9b.py` | the conditioning-path impact measurement (the un-rotated injection) |
| `sdxl/diag_impact_qwen35_9b_gen.py` | the generate-path impact measurement (`lm_head`, `mtp.*`) |
| `clip_convert/convert_qwen35_9b_impact_convrot_int8.py` | the impact-driven builder (`--protect-list` supported) |
| `sdxl/verify_qwen35_9b.py` | the 256-item acceptance, both paths, the loader's own arm |
| `impact/impact_qwen35_9b.json` | the measured per-layer drift, conditioning (git-ignored) |
| `impact/impact_qwen35_9b_gen.json` | the measured per-layer drift, generate (git-ignored) |
