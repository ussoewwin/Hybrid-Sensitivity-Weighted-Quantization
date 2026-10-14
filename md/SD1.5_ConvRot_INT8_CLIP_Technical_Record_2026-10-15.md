# SD1.5 ConvRot INT8 CLIP - Technical Record (2026-10-15)

**Scope:** making the checkpoint-embedded SD1.5 CLIP (`cond_stage_model.*`) load and
run as true ConvRot INT8 inside ComfyUI — so the text encoder is quantized, used
for conditioning, and evaluated in benchmarks instead of silently falling back.

**Files (all changes live in exactly two files):**
- `benchmark/sd1.5 int8/comfy_quant_int8.py` — loader fix (commit `47acae7`)
- `sd1.5/quantize_sd1.5_hswq_v1.0.py` + `comfyui_nodes/native_convrot_int8_convert.py` — CLIP quantization (Owner's work, commits `6142989`→`03b19d7`→`a2b3ef7`→`31537d4`→`282fb9e`)

**Result (25 seeds × 25 steps, 512×512):** INT8 CLIP genuinely used
(`Successfully encoded prompt with ConvRot INT8 CLIP`), mean final-cos
**0.96613**, same-image 14/25, FP16 1.74s / INT8 1.51s (**+13.3%**).

---

## 1. The problem, precisely

### 1.1 Symptom

Packs that quantized the embedded CLIP produced collapsed trajectories in the
standalone bench: mean final-cos **0.29314**, all 25 seeds bifurcated at step 15.
Conditioning comparison against the FP16 CLIP showed:

```
FP16 CLIP conditioning  absmax = 32.95
INT8 CLIP conditioning  absmax = 10.53   (cos vs FP16 = 0.0968)
```

The text embedding was crushed to ~1/3 of its normal magnitude.

### 1.2 Root cause (measured)

ComfyUI builds the SD1.5 text encoder with **`comfy.ops.manual_cast`** unless
`model_options["quantization_metadata"]` is present (`comfy/sd1_clip.py`,
`SDClipModel.__init__`):

```python
operations = model_options.get("custom_operations", None)
quant_config = model_options.get("quantization_metadata", None)
if operations is None:
    if quant_config is not None:
        operations = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)
    else:
        operations = comfy.ops.manual_cast
```

The SD1.5 branch of `comfy.sd.load_text_encoder_state_dicts()` **never sets
`quantization_metadata`** (only the MINIMAX and MING branches do). Consequence,
for every CLIP Linear layer:

1. The module is a `manual_cast.Linear`, not a quantized ops module.
2. `_load_quantized_module` (the consumer of `.comfy_quant` / `.weight_scale`)
   is never invoked.
3. The int8 weight tensor loads as-is and gets cast to fp16 — **raw integers
   (−127…127) become "weights"**.
4. Every attention/MLP projection in CLIP then computes with ~700× too-large
   weights → the text embedding collapses.

Measured proof (layer `…layers.0.self_attn.q_proj.weight`, as loaded):

| source | absmax | integer-like |
|---|---|---|
| FP16 checkpoint | 0.178 | no |
| INT8 pack via the broken path | **127.0** | **yes** |
| INT8 pack via the fixed path | 0.178 | no (QuantizedTensor) |

This is the same bug class as the UNet NaN incident (2026-10-14): quantization
companion keys present in the file but never consumed at load. The UNet was
fixed by arming `mixed_precision_ops` via the int8 scope; the CLIP load path is
a **different code path** (`load_state_dict_guess_config` →
`process_clip_state_dict` → `CLIP(...)` → `SDClipModel.__init__`), so the UNet
fix did not cover it.

### 1.3 Why it hid in the node benchmark

The node's in-quantize benchmark (`native_convrot_int8_convert.py`) loaded the
INT8 CLIP inside a `try`, and when it failed it silently fell back to the FP16
CLIP conditioning (log marker `Benchmark CLIP: ConvRot INT8` was absent from
the good-looking report). The standalone bench
(`sd1.5_int8_traj_compare.py`) has the same fallback — the log line
`[INT8] Warning: encoding with INT8 CLIP failed … using baseline CLIP
conditioning` is the tell. **A report without the "Benchmark CLIP: ConvRot
INT8" line was measured against FP16 conditioning**, which masked the defect
and produced "native works / HSWQ broken" confusion. Both files were equally
broken; only the visibility differed.

---

## 2. Key-space mechanics (why naive key renaming fails)

### 2.1 The three key spaces

| space | example | where |
|---|---|---|
| pack (file) | `cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight` | safetensors |
| CLIP SD (post `SD15.process_clip_state_dict`) | `clip_l.transformer.text_model.…` | `CLIP.load_sd` |
| module (runtime) | `cond_stage_model.clip_l.transformer.text_model.…` | `named_modules()` |

`SD15.process_clip_state_dict` (comfy/supported_models.py):

```python
# step 1: force-nest everything under transformer. into text_model.
for x in keys:
    if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
        x -> cond_stage_model.transformer.text_model.…          # (A)
# step 2: strip the checkpoint prefix
replace_prefix["cond_stage_model."] = "clip_l."                  # (B)
```

So the runtime module path is `cond_stage_model/clip_l/transformer/text_model/…`
and state-dict keys must be `clip_l.transformer.text_model.…` after (A)+(B).

### 2.2 The `text_projection` trap

`CLIPTextModel` (comfy/clip_model.py) has two children: `text_model` and
`text_projection` (a 768×768 Linear, bias-free). Rule (A) **force-nests any
`cond_stage_model.transformer.*` key that is not already `…text_model.*`** into
`…text_model.*`. Therefore:

- a pack key `cond_stage_model.transformer.text_projection.weight`
  becomes `clip_l.transformer.text_model.text_projection.weight` after (A)+(B) —
  **wrong nesting** (there is no `text_model.text_projection`), the real
  `clip_l.transformer.text_projection` stays empty, and under INT8 ops its
  weight stays `None` → forward crashes (`'NoneType' object has no attribute
  'device'` in `cast_bias_weight`).
- there is **no pack key spelling** that survives (A) and lands at
  `clip_l.transformer.text_projection.*`.

The FP16 path never had this problem because the SD1.5 checkpoint stores **no
`text_projection` at all**; ComfyUI instantiates the Linear and leaves it
all-zeros (measured: absmax 0.0), and it is unused anyway —
`SD1CheckpointClipModel` sets `return_projected_pooled=False`, so the projected
pooled output never reaches the UNet conditioning.

Measured facts about these optional params:

```
FP16 CLIP get_sd(): 198 keys, includes
    clip_l.transformer.text_projection.weight  (all zeros, unused)
    clip_l.logit_scale                         (4.6055)
SD1.5 checkpoint: 197 keys, contains neither
```

**Conclusion:** `text_projection`/`logit_scale` cannot come from the pack; they
must be completed on the loaded CLIP object, exactly like the FP16 path does by
accident of initialization.

---

## 3. The fix (loader side, `benchmark/sd1.5 int8/comfy_quant_int8.py`)

Two cooperating pieces, both installed by `apply_comfy_quant_int8_patches()`.

### 3.1 `_patch_sd15_clip_quant_metadata()` — make CLIP quant-aware

Wraps `comfy.sd.load_state_dict_guess_config`:

1. **Gate.** Proceed only when the state dict carries
   `cond_stage_model.*.comfy_quant` keys (i.e. an HSWQ SD1.5 pack). SDXL/SD3/
   Flux/Z-Image/Krea2/Qwen carry `clip_g`/`t5`/other prefixes and fall through
   untouched.
2. **Collect.** Build the CLIP-restricted quantization metadata from the
   file-embedded `.comfy_quant` tensors (authoritative), falling back to the
   `_quantization_metadata` file metadata; remap each layer key
   `cond_stage_model.X` → `clip_l.X` to match the post-(A)+(B) key space.
3. **Inject.** Pass it through `te_model_options["quantization_metadata"]`.
   `load_state_dict_guess_config` forwards `te_model_options` into the `CLIP(...)`
   `model_options`, so `SDClipModel.__init__` now builds
   `mixed_precision_ops(quant_config, dtype, full_precision_mm=True)` and the
   quantized Linear/Embedding classes consume `.comfy_quant`/`.weight_scale`
   (dequant + ConvRot un-rotate at weight-load time).

```python
te_model_options["quantization_metadata"] = {
    "format_version": "1.0",
    "layers": clip_layers,          # clip_l.transformer.… -> conf dicts
}
kwargs["te_model_options"] = te_model_options
```

`full_precision_mm=True` matters: CLIP is a 77-token text encoder, not
throughput-critical; keeping the mm (matmul) path in full precision removes any
numerical risk while the weights still load and persist as true INT8.

### 3.2 `_complete_sd15_clip_optional_params()` — complete the optional params

After the original loader returns, on the resulting `CLIP` object:

- `inner = clip.cond_stage_model.clip_l`
- if `inner.transformer.text_projection.weight is None`: set a plain fp16
  zeros Parameter of `(hidden, hidden)` (hidden read from the token embedding;
  falls back to 768). **Not** quantized — it is unused, and quantizing it would
  need a `weight_scale` companion that (per §2.2) cannot be delivered by the pack.
- ensure `inner.logit_scale` exists (4.6055), matching the FP16 initialization.

This reproduces the FP16-loaded CLIP state exactly (measured: FP16
`text_projection.weight` absmax 0.0, `logit_scale` 4.6055), so INT8 and FP16
conditioning paths are behaviorally identical apart from the quantized weights.

### 3.3 Why not other approaches (measured failures, kept as record)

| approach | result |
|---|---|
| remap pack keys `cond_stage_model.transformer.*` → `cond_stage_model.clip_l.transformer.*` (native file's spelling) | wrong key space after (A)+(B): probe produced `q_proj absmax=0.0000` and NaN conditioning |
| emit `text_projection` in the pack as fp16 zeros | rule (A) mis-nests it under `text_model.` (§2.2) → still `None` at runtime |
| emit `text_projection` ConvRot-quantized in the pack (native does this) | same mis-nesting → `Missing weight for layer clip_l.transformer.text_projection`, INT8 forward crash |
| `model_options={"quantization_metadata": …}` into `load_text_encoder_state_dicts` directly | unreliable channel; the working path is `te_model_options` via `load_state_dict_guess_config` |

---

## 4. Verification (all measured)

### 4.1 Layer-level

After the fix, `…layers.0.self_attn.q_proj.weight` on the loaded INT8 CLIP:

```
QuantizedTensor, torch.float16 wrapper, absmax = 0.1779
```

— matching the FP16 source (0.1777, delta = quantization noise). The
`.comfy_quant`/`.weight_scale` companions are consumed, and the ConvRot rotation
is applied at load (un-rotated on the fly by the INT8 forward).

### 4.2 Conditioning-level

```
INT8 CLIP encode absmax = 32.95, cos vs FP16 CLIP ≈ 1.0
```

(the broken path gave absmax 10.53 / cos 0.0968; the mis-remap probe gave NaN)

### 4.3 End-to-end bench (25 seeds × 25 steps, 512×512, CUDA-graph executor on)

| | broken | fixed |
|---|---|---|
| INT8 CLIP actually used | no (silent FP16 fallback) | **yes** (`Successfully encoded prompt with ConvRot INT8 CLIP`) |
| mean final-cos | 0.29314 | **0.96613** |
| same-image seeds | 0/25 | **14/25** |
| speed | — | FP16 1.74s / INT8 1.51s (**+13.3%**) |

The residual gap to the FP16-CLIP score (0.97120) is the true cost of INT8 CLIP
conditioning: ~0.005 mean cos, in exchange for the text encoder running as INT8
(~235 MB fp16 → ~118 MB int8 in the conditioning path) and being benchmarked
honestly.

### 4.4 Reading the logs (what to look for)

Healthy INT8-CLIP bench must show **both**:

```
[HSWQ INT8][SD1.5 CLIP] injecting ConvRot INT8 metadata for 72 text-encoder layers
[INT8] Successfully encoded prompt with ConvRot INT8 CLIP.
```

and must **not** show:

```
[INT8] Warning: encoding with INT8 CLIP failed (...) using baseline CLIP conditioning.
WARNING:root:Missing weight for layer clip_l.transformer.text_projection
```

For the node benchmark, the report must contain
`Benchmark CLIP: ConvRot INT8 (evaluated on quantized branch)` — a report
without that line was measured against FP16 conditioning.

---

## 5. Design invariants (why this shape)

1. **Separation.** All loader-side logic lives in
   `benchmark/sd1.5 int8/comfy_quant_int8.py`, gated by pack content
   (`cond_stage_model.*.comfy_quant` Linear layers present). Models without an
   HSWQ SD1.5 CLIP pack are bit-for-bit on the stock path.
2. **No pack-format change.** The pack format (int8 + `.weight_scale` +
   `.comfy_quant` + `_quantization_metadata` metadata) is unchanged; the fix is
   purely on the consumption side, so old/new packs and the native node output
   stay interoperable.
3. **FP16 parity of optional params.** `text_projection`/`logit_scale` are
   completed to the same values the FP16 loader produces (zeros / 4.6055), so
   there is no INT8-specific behavior beyond the quantized Linear weights.
4. **Visible fallbacks are forbidden in measurements.** A bench that silently
   falls back to FP16 conditioning reports a number that is not what it claims
   to measure. The fallback `try/except` stays (robustness), but the log line
   makes it explicit, and §4.4 defines the acceptance markers.

## 6. Commits

| commit | content |
|---|---|
| `6142989` | native node: keep CLIP as `cond_stage_model.*` in the export |
| `03b19d7` | native node: quantize CLIP Linear layers to ConvRot INT8 |
| `a2b3ef7` | standalone bench: evaluate the INT8 CLIP on the quantized branch |
| `31537d4` | quantizer: CLIP ConvRot INT8 quantization in `quantize_sd1.5_hswq_v1.0.py` |
| `282fb9e` | native node: `import torch` in quantize scope |
| `f174134` | FP16 protection ceiling restored to 300 MiB |
| `47acae7` | **loader fix: CLIP quantization metadata injection + optional-param completion** |

## 7. Reproduce

```
python benchmark/sd1.5_int8_traj_compare.py ^
  --fp16  <ComfyUI>\models\unet\0.5xxmix9realistic_v40+0.5BRA7.safetensors ^
  --int8  <ComfyUI>\models\unet\0.5xxmix9realistic_v40+0.5BRA7_hswq_int8.safetensors ^
  --comfy_path <repo>\ComfyUI-master --steps 25 --width 512 --height 512 ^
  --cfg 7.0 --sampler dpmpp_2m --scheduler karras
```

Pack generation:

```
python sd1.5/quantize_sd1.5_hswq_v1.0.py --input <ckpt> --output <pack> --calib_file sample/calibration_prompts_512.txt
```

Expected report tail: `CLIP INT8 layers: 72 (convrot=72)`.

## Related

* `md/SD1.5_INT8_Speed_Optimization_Root_Cause_and_Fix_2026-10-14.md` — UNet-side
  (CUDA graph, TLS scope) companion record; the CLIP fix completes the same
  "companion keys must be consumed at load" principle on the text-encoder path.
* `md/How to quantize SD1.5.md` — quantization how-to.
