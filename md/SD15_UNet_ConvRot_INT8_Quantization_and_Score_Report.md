# SD1.5 UNet ConvRot INT8 - Quantization and Score Report

**Checkpoint.** `models/unet/0.5xxmix9realistic_v40 0.5BRA7.safetensors` (2033.7 MB, FP16, 1131 tensors)
**Pack.** `models/checkpoints/0.5xxmix9realistic_v40 0.5BRA7_sd15_unet_hswq_convrot_int8.safetensors` (1083.5 MB, **−46.7 %**)
**SHA256.** `652D07EA80CD8E87D0EA2AC5AE6F93149B4C9384ED291F215E995D1432865C11`

**Composition.** UNet quantized (262 of 282 ConvRot-eligible layers in INT8, 20 kept FP16) · **CLIP untouched** (197 tensors, byte-identical) · **VAE removed** (0 tensors).

---

## 0. The load path is the whole story

Three separate failures in this work all had the same shape: **the pack was correct and the harness was not.** They are recorded here because anyone re-running this will hit them.

### 0.1 `comfy_kitchen` has no INT8 conv kernel

`comfy_kitchen` exposes `int8_linear` (and `mm_int8`, `w4a8_int8_linear`) and **no INT8 convolution**. A Conv2d whose weight is stored INT8 therefore reaches a float conv and the output becomes NaN. Measured on the first attempt: `int8_linear` calls **0**, NaN **16384 / 16384**.

### 0.2 The Conv2d path exists only in the HSWQ loader, behind a thread-local scope

The loader patches `comfy.ops` in `patches/comfy_quant_int8.py :: _patch_ops_decode_and_conv()`, and the injection is gated:

```python
def _should_inject_int8_conv(quant_config) -> bool:
    return bool(getattr(_int8_quant_conv_tls, "active", False))
```

Calling `apply_comfy_quant_int8_patches()` alone is therefore **not enough** - the scope has to be open while the model is built.

### 0.3 There are two vendored copies of the patch module, each with its own thread-local

```
patches/comfy_quant_int8.py            <- patched by apply_comfy_quant_int8_patches()
nodes/sd1.5 int8/comfy_quant_int8.py   <- the SD1.5-specific one
```

Opening one TLS does not open the other. The SD1.5 loader
(`nodes/sd1.5 int8/__init__.py :: load_checkpoint_sd15_hswq_weight_dtype`) solves this by
collecting **every** `_int8_quant_conv_tls` it can find in `sys.modules` and setting them all:

```python
with _all_int8_scopes():
    out = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=False, output_cliC=True, ...)
```

**This is the only correct entry point for an SD1.5 INT8 pack.** Measured after switching to it: **Linear 177 + Conv2d 85 INT8, output finite, range [−3.7695, 4.0938]**.

Note also `output_vae=False` - the loader already treats the VAE as optional, which is exactly the composition this delivery ships.

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 2033.7 MB, FP16, 1131 tensors (UNet 686 / CLIP 197 / VAE 248) |
| the ConvRot-eligible layers | **282** (142 Linear + 140 Conv2d) |
| the pack | **1083.5 MB** (−46.7 %) |
| the quantized | **262** |
| the FP16-kept | 20 (the top-contribution layers, per section 2) |
| CLIP | **unchanged** - 197 tensors, 0 missing, 0 dtype changes, 0 quant artifacts |
| VAE | **removed** - 0 tensors in the pack |

Both **Linear and Conv2d** are quantized. Conv2d rotates along the input-channel axis and its `weight_scale` is stored `(O,1,1,1)` - the shape the reference `pack_channelwise` uses. A flattened `(O,)` scale breaks the Conv2d broadcast path (measured: NaN).

---

## 2. What the protection set is based on

The previously shipped V1.0 selector ranked protection from **static weight-space statistics** (kurtosis / outlier-ratio / absmax) plus a weight-space histogram MSE, and on this checkpoint it produced a protection set that **scored the same as native** - the protection bought nothing. That was a criterion failure, not a quantizer failure.

This pack uses the **aggregate contribution**: replace one layer by its ConvRot INT8 reconstruction in the original frame and measure the change in the UNet's **own output** over 8 fixed seeds (282 layers, float64, 464x spread). The measured ranking:

| rank | layer | the contribution | the share |
| :--- | :--- | :--- | :--- |
| 1 | `input_blocks.1.1.proj_in` | 5.6167e-06 | 10.77 % |
| 2 | `input_blocks.2.1.proj_in` | 2.8616e-06 | 5.49 % |
| 3 | `input_blocks.7.1.transformer_blocks.0.ff.net.0.proj` | 2.6664e-06 | 5.11 % |
| 4 | `input_blocks.5.1.proj_in` | 1.6888e-06 | 3.24 % |
| 5 | `output_blocks.6.0.skip_connection` | 1.6788e-06 | 3.22 % |

SD1.5's sensitive layers are the **early input-block transition projections** and the **`skip_connection`** lines - not the attention projections a weight statistic would flag.

---

## 3. The score table (256 items, seed 42, through the real loader)

| the candidate | INT8 Linear | INT8 Conv2d | cos mean | cos min | p5 | size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 184 | 98 | 0.99997991 | 0.99973191 | 0.99993569 | 1056.3 MB |
| k=10 | 180 | 92 | 0.99998317 | 0.99961265 | 0.99996096 | 1072.4 MB |
| **k=20 (adopted)** | **177** | **85** | **0.99998664** | **0.99977472** | **0.99996690** | **1083.5 MB** |
| k=40 | 170 | 72 | 0.99999353 | 0.99987036 | 0.99998180 | 1115.5 MB |
| k=80 | 143 | 59 | 0.99999617 | 0.99995949 | 0.99998917 | 1167.3 MB |

(The counts exceed K because both Linear and Conv2d entries are protected; K is a rank in a mixed list.)

**Every candidate clears the gate (mean ≥ 0.99, min ≥ 0.98) with a wide margin.** The differences are at the fourth decimal and the `cos.min` column is **not monotone** (k=0 0.99973191 → k=10 0.99961265 → k=20 0.99977472): `cos.min` is one draw out of 256, so it moves with which item happens to be worst, not with quality. `cos.mean` is monotone and is the column to read.

On `cos.mean`, k=20 is +6.7e-06 over k=0 for +27.2 MB. k=80 adds a further +9.5e-06 for +83.8 MB. Since k=0 already clears the gate by 0.018 on the minimum, **the adopted pack is k=20** - it keeps the 20 highest-contribution layers in FP16 as insurance while staying at roughly half the source size.

---

## 4. Visible-sample check

Rendered with the same VAE (borrowed from the source, identical for both sides), the same latent seed, 20 steps, CFG 7.0:

| | the value |
| :--- | :--- |
| the pixel difference | **mean 0.34 / max 6 / p99 2.0** (0-255 scale) |
| the FP16 image | mean 113.4, std 13.3 |
| the INT8 image | mean 113.4, std 13.3 |

The two sides are visually indistinguishable, which is the expected result at `cos.min` 0.99977.

**Honest limitation.** This probe used a **fixed synthetic conditioning tensor**, not a CLIP text embedding, because the text path could not be completed on this machine: `comfy.sd.CLIP` keeps its weights in a `ModelPatcher` and loads them lazily, and neither `.to()` nor `load_models_gpu([model, clip.patcher])` moved them, so the encode step raised `mat2 is on cuda:0 ... different from ... cpu`. The consequence is that the sample has **no semantic content** - it is a flat low-contrast field - so it proves *numerical* equivalence of the UNet and the decoder, **not** prompt fidelity. A prompt-driven sample remains to be produced on a machine where the CLIP encode path works.

---

## 5. Reproduction

```powershell
# the environment
#   python = <ComfyUI>/python_embeded/python.exe
#   COMFY  = <ComfyUI>

# 1. the criterion: the aggregate contribution (282 layers, 8 seeds, float64)
python sd1.5/measure_sd15_unet_contrib.py `
    --ckpt "<checkpoint>.safetensors" --out impact/contrib_sd15_unet.json --n 8

# 2. the builder: ConvRot INT8 for Linear and Conv2d, VAE dropped, CLIP untouched
python sd1.5/quantize_sd1.5_hswq_v1.1.py `
    --input "<checkpoint>.safetensors" `
    --output impact/_work/sd15_v2_k20.safetensors `
    --contrib impact/contrib_sd15_unet.json --top-k 20

# 3. the acceptance through the REAL loader (256 items, seed 42)
$env:SD15_SRC = "<checkpoint>.safetensors"
$env:SD15_N = "256"
python sd1.5/verify_sd15_loader.py impact/_work/sd15_v2_k20.safetensors
```

**The step everyone gets wrong:** step 3 must go through
`load_checkpoint_sd15_hswq_weight_dtype`. Building the model by hand with
`mixed_precision_ops` reproduces the NaN described in section 0, because the
Conv2d scope is never opened.

---

## 6. Error sources and limits

- **NaN means the harness, not the pack.** If the output is NaN with `int8_linear` = 0, the Conv2d scope was not open. Use the loader.
- **The `weight_scale` rank must match the weight's** (`(O,1)` / `(O,1,1,1)`).
- **The quant detection must run before the prefix strip** (`detect_layer_quantization` matches `k.startswith(unet_key_prefix)`).
- **`cos.min` is not a quality curve at this N.** It is one draw of 256; use `cos.mean` for the trend and `cos.min` only as a gate.
- **Input-distribution dependent.** Random latents and synthetic conditioning, not rendered images. Sound for ranking; the prompt-driven check is the missing piece (section 4).
- **The UNet's own output only** - one forward, not a 25-step trajectory, and no VAE/CLIP round trip in the scored path.
- **The VAE is absent from the pack** by instruction. Any consumer must supply its own VAE to decode; the pack cannot be used as a standalone checkpoint.

---

## 7. Ledger

| # | when (JST) | what | the setting | cos mean / min | the size |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-10-24 06:05 | the aggregate contribution | 282 layers, 8 seeds, float64 | - (a ranking) | - |
| 2 | 2026-10-24 07:05 | NaN isolation | Conv2d INT8 without the loader scope | NaN 16384, `int8_linear` 0 | - |
| 3 | 2026-10-24 07:22 | the loader path confirmed | `load_checkpoint_sd15_hswq_weight_dtype` | finite, Linear 177 + Conv2d 85 | - |
| 4 | 2026-10-24 07:40 | k=0 | 282 INT8 | 0.99997991 / 0.99973191 | 1056.3 MB |
| 5 | 2026-10-24 07:40 | k=10 | 272 INT8 | 0.99998317 / 0.99961265 | 1072.4 MB |
| 6 | 2026-10-24 07:40 | **k=20 (adopted)** | 262 INT8 | **0.99998664 / 0.99977472** | **1083.5 MB** |
| 7 | 2026-10-24 07:40 | k=40 | 242 INT8 | 0.99999353 / 0.99987036 | 1115.5 MB |
| 8 | 2026-10-24 08:10 | k=80 | 202 INT8 | 0.99999617 / 0.99995949 | 1167.3 MB |
| 9 | 2026-10-24 08:25 | the pack audit | CLIP 197 identical, 0 artifacts; VAE 0 tensors | - | - |
| 10 | 2026-10-24 08:35 | the sample check | same VAE, same seed, 20 steps, CFG 7 | pixel diff mean 0.34 | - |

---

## 8. Files

| file | role |
| :--- | :--- |
| `sd1.5/measure_sd15_unet_contrib.py` | the criterion (the aggregate contribution) |
| `sd1.5/quantize_sd1.5_hswq_v1.1.py` | the builder (Linear + Conv2d, VAE dropped, CLIP untouched) |
| `sd1.5/verify_sd15_loader.py` | the acceptance through the real HSWQ SD1.5 loader |
| `md/SD15_UNet_ConvRot_INT8_Quantization_and_Score_Report.md` | this report |
| `impact/contrib_sd15_unet.json` | the contribution (git-ignored) |
| `impact/verify_sd15_v2_256.json` | the acceptance results (git-ignored) |
