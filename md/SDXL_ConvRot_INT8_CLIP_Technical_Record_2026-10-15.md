# SDXL ConvRot INT8 CLIP - Technical Record (2026-10-15)

**Scope:** making the checkpoint-embedded SDXL text encoders (CLIP-L
`conditioner.embedders.0`, CLIP-G `conditioner.embedders.1`) load and run as true
ConvRot INT8 inside ComfyUI, with a rule-based FP16 protection set - so the text
encoders are quantized, used for conditioning, and evaluated instead of silently
falling back to FP16.

**Files:**
- `ComfyUI-HSWQ-Loader-and-Tools/nodes/sdxl_convrot_int8/sdxl_clip_quant.py` - loader fix (SDXL-only)
- `ComfyUI-HSWQ-Loader-and-Tools/patches/comfy_quant_int8.py` - SDXL INT8 branch arms it (`_load_sdxl_clip_quant`)
- `hswq/sdxl/quantize_sdxl_clip_int8.py` - rule-based CLIP ConvRot INT8 converter

**Result:** SDXL pack conditioning cos vs FP16 = **1.0000** (absmax 126.44),
image renders correctly; previously the INT8 CLIP either crashed or produced a
black image.

---

## 1. Three distinct failures (all measured, none guessed)

| # | Symptom | Root cause | Fix commit |
|---|---|---|---|
| 1 | INT8 CLIP weights load as fp16 (absmax 127), conditioning collapses | SDXL TE load path never sets `model_options["quantization_metadata"]`; CLIP-L/CLIP-G classes have no self-detect (unlike the Llama-family `llama_detect` used by Krea2 / Z Image) | `85c8109` |
| 2 | `'NoneType' object has no attribute 'device'` on prompt encode | CLIP-L stores no `text_projection` in the checkpoint; under `mixed_precision_ops` its quantized Linear weight stays `None` | `531313a` |
| 3 | Black image (conditioning absmax ~1.7e6, cos -0.018) | `transformers_convert` rewrites CLIP-G `.weight` to the HF layout but leaves `.comfy_quant`/`.weight_scale` at the OpenAI-CLIP path, so the quant metadata never matches the module | `79a5259` |

---

## 2. Why SDXL (and SD1.5) need a loader fix but Krea2 / Z Image do not

ComfyUI builds a text encoder with `comfy.ops.manual_cast` unless
`model_options["quantization_metadata"]` is set (`SDClipModel.__init__`,
`comfy/sd1_clip.py`). The metadata decides whether the Linear/Embedding classes
are the quant-aware `mixed_precision_ops` variants.

* **Krea2 / Z Image** use Llama text encoders. Their TE constructor path calls
  `hunyuan_video.llama_detect()`, which internally runs
  `comfy.utils.detect_layer_quantization()` and returns
  `llama_quantization_metadata`; the `sd.py` branch passes `**llama_detect(...)`
  straight into the TE. Quant detection is **built into the load path**.
* **SD1.5 (CLIP-L) / SDXL (CLIP-L + CLIP-G)** use dedicated `SDClipModel` /
  `SDXLClipModel` classes. The `sd.py` CLIPType branches for these **never call
  `detect_layer_quantization`** (only the MINIMAX and MING branches do). So the
  int8 weights + `.comfy_quant` / `.weight_scale` companions are never consumed
  and load as raw fp16.

Hence SDXL (like SD1.5) needs the quantization metadata injected from our side.

### 2.1 The injection point

`comfy.sd.load_state_dict_guess_config` builds
`CLIP(clip_target, ..., model_options=te_model_options, ...)`
(`comfy/sd.py`). Wrapping that function and injecting
`te_model_options["quantization_metadata"]` makes `SDXLClipModel` build both
`clip_l` and `clip_g` with `mixed_precision_ops`. SDXL-specific module keys:

```
conditioner.embedders.0.transformer.*  ->  clip_l.transformer.*          (plain swap)
conditioner.embedders.1.model.*        ->  clip_g.*  then HF-restructure  (see §4)
```

---

## 3. Failure 2: CLIP-L `text_projection` (None-weight crash)

SDXL's CLIP-L stores **no `text_projection`** in the checkpoint
(`SDXL.process_clip_state_dict_for_saving` pops
`clip_l.transformer.text_projection.weight`). The FP16 path builds it as an
unused all-zeros Linear; under `mixed_precision_ops` the same Linear is a
quantized ops module whose weight stays `None` (the pack has no `comfy_quant`
for it). The forward calls `self.text_projection(x[2])` and crashes with
`'NoneType' object has no attribute 'device'` (`cast_bias_weight`).

Fix: `_complete_sdxl_clip_optional_params()` sets a plain fp16 zeros
`(hidden, hidden)` weight after load, exactly like the FP16 path. CLIP-G ships
its own `text_projection` (1280x1280) and is left untouched.

Measured:

| | before | after |
|---|---|---|
| `clip_l.text_projection.weight` (armed) | `None` (crash) | `768x768` zeros (absmax 0.0) |
| `clip_g.text_projection.weight` | 1280x1280 (ok) | unchanged |

---

## 4. Failure 3: CLIP-G companion keys (black image)

### 4.1 The key-space mismatch

`SDXL.process_clip_state_dict` (comfy/supported_models.py):

```python
replace_prefix["conditioner.embedders.0.transformer.text_model"] = "clip_l.transformer.text_model"
replace_prefix["conditioner.embedders.1.model."] = "clip_g."
state_dict = state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
state_dict = clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")
```

`clip_text_transformers_convert` -> `transformers_convert(sd, "clip_g.", "clip_g.transformer.text_model.", 32)`.
`transformers_convert` restructures the OpenAI-CLIP text-transformer layout into
the HuggingFace one, but **only for `.weight` / `.bias`** (and a few whole-model
keys). Measured on a minimal reproduction:

```
in : clip_g.transformer.resblocks.0.mlp.c_fc.weight
out: clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight      <- moved
in : clip_g.transformer.resblocks.0.mlp.c_fc.comfy_quant
out: clip_g.transformer.resblocks.0.mlp.c_fc.comfy_quant                <- NOT moved
in : clip_g.transformer.resblocks.0.mlp.c_fc.weight_scale
out: clip_g.transformer.resblocks.0.mlp.c_fc.weight_scale               <- NOT moved
```

The module is built at `clip_g.transformer.text_model.encoder.layers.0.mlp.fc1`
(so `_load_quantized_module` looks for
`...layers.0.mlp.fc1.comfy_quant` / `...weight_scale`), but the companions sit at
`clip_g.transformer.resblocks.0.mlp.c_fc.*` -> **no match** -> the int8 weight
loads as raw fp16.

Consequence: CLIP-G conditioning explodes (absmax ~1.75e6, cos -0.018 vs FP16)
-> black image. CLIP-L was unaffected because its keys are a plain prefix swap
(no structural convert), so its companions already matched.

### 4.2 Fix

`_remap_clipg_companion_keys()` moves the CLIP-G `.comfy_quant` / `.weight_scale`
tensors to the same HF layout the `.weight` gets, before the (weight-only)
convert runs. Mapping (mirrors `transformers_convert`):

```
transformer.resblocks.N.ln_1          -> transformer.text_model.encoder.layers.N.layer_norm1
transformer.resblocks.N.ln_2          -> transformer.text_model.encoder.layers.N.layer_norm2
transformer.resblocks.N.mlp.c_fc      -> transformer.text_model.encoder.layers.N.mlp.fc1
transformer.resblocks.N.mlp.c_proj    -> transformer.text_model.encoder.layers.N.mlp.fc2
transformer.resblocks.N.attn.out_proj -> transformer.text_model.encoder.layers.N.self_attn.out_proj
token_embedding                       -> transformer.text_model.embeddings.token_embedding
positional_embedding                  -> transformer.text_model.embeddings.position_embedding
ln_final                              -> transformer.text_model.final_layer_norm
```

Note `attn.in_proj_weight` (fused qkv) is split into q/k/v by the convert and is
**not** quantized (kept fp16) - a single metadata entry cannot express the split.

Measured on the re597 pack: **126 companion tensors** remapped.

### 4.3 Verification

| | before | after |
|---|---|---|
| pack conditioning absmax | 1,749,812 | **126.44** |
| cos vs FP16 | -0.0184 | **1.0000** |
| image | black | correct |

---

## 5. FP16 protection rule (model-independent, measured)

Risk metrics (kurtosis / outlier_ratio = absmax/std) were computed on the CLIP
Linear layers of four SDXL checkpoints (waiIllustrious, waiREALCN, animemix,
bluePencil). The protected set is structurally identical across all four:

**Protected (kept FP16):**
- CLIP-L: `embeddings.token_embedding`, `embeddings.position_embedding`,
  `encoder.layers.0.mlp.fc1`, `encoder.layers.0.mlp.fc2`
- CLIP-G: `token_embedding`, `transformer.resblocks.*.mlp.c_proj` (all resblocks),
  `transformer.resblocks.{2,29}.attn.out_proj`

Everything else in the CLIPs is FULL ConvRot INT8.

Measured risk (top candidates; kurtosis / outlier_ratio):

| layer | waiIllustrious | waiREALCN | animemix | bluePencil |
|---|---|---|---|---|
| `e1.resblocks.0.mlp.c_proj` | 1280 / 166 | 1280 / 166 | 421 / 129 | 1404 / 175 |
| `e0.embeddings.position_embedding` | 513 / 73 | 513 / 73 | 571 / 75 | 652 / 78 |
| `e1.resblocks.2.attn.out_proj` | 175 / 97 | 175 / 97 | 66 / 76 | 419 / 121 |
| `e0.layers.0.mlp.fc1` | 61 / 67 | 61 / 67 | 102 / 77 | 109 / 78 |

Embedder-level medians: CLIP-G outlier_ratio ~13 (heavier tail than CLIP-L ~6);
the `mlp.c_proj` family dominates the risk.

Converter: `hswq/sdxl/quantize_sdxl_clip_int8.py`
(`is_protected_clip_weight` + `quantize_clip`). Same pack math as the native
converter (Hadamard rotate -> per-channel INT8 -> `.comfy_quant` stamp, adaptive
gs 256/64/4). On the re597 pack: 133 ConvRot INT8 + 38 FP16-protected + 90
kept-as-is (norms/biases/1D).

---

## 6. Separation guarantees

* The loader fix lives only in `nodes/sdxl_convrot_int8/sdxl_clip_quant.py`,
  loaded by path via `_load_sdxl_clip_quant()` (not registered in
  `sys.modules`); the shared `patches/comfy_quant_int8.py` keeps its other code
  unchanged and only the SDXL INT8 branch arms it.
* The wrapper refuses to act unless the state dict actually carries
  `conditioner.embedders.*` `.comfy_quant` layers. SD1.5 / SD3 / FLUX / Z Image /
  Krea2 / Qwen never match this shape.
* SD1.5 has its own module (`nodes/sd1.5 int8/comfy_quant_int8.py`); the two
  families branch via distinct weight markers (`SDXL ConvRot INT8` vs the SD1.5
  path) and never share a code path.

## 7. Commits

| repo | commit | content |
|---|---|---|
| Loader | `85c8109` | inject SDXL CLIP quantization metadata (fix fp16 raw load) |
| Loader | `531313a` | complete `clip_l.text_projection` (fix None-weight crash) |
| Loader | `79a5259` | remap CLIP-G companion keys to HF layout (fix black image) |
| hswq | `aea854c` | `sdxl/quantize_sdxl_clip_int8.py` (rule-based FP16 protection) |

## 8. Reproduce

Convert:

```
python sdxl/quantize_sdxl_clip_int8.py ^
  --input  <pack>_convrot_int8.safetensors ^
  --output <pack>_convrot_int8_clipint8.safetensors
```

Load in ComfyUI with the **HSWQ Checkpoint Loader** and weight type
**`SDXL ConvRot INT8`**. Healthy load shows:

```
[HSWQ INT8][SDXL CLIP] injecting ConvRot INT8 metadata for 133 text-encoder layers
[HSWQ INT8][SDXL CLIP] remapped 126 CLIP-G companion tensors to the HF layout
```

A black image or a conditioning absmax far above ~130 means the companion remap
did not run (check the second log line is present).

## Related

* `md/SD1.5_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md` - the SD1.5 CLIP INT8
  record; the SDXL fixes are the same failure classes (metadata consumption,
  optional-param completion) plus the SDXL-specific CLIP-G key-space convert.
