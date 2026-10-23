# SD1.5 ConvRot INT8 - aggregate-contribution pipeline

This directory holds the **SD1.5 UNet** quantization pipeline that replaces the
V1.0 static selector. Everything here runs from explicit CLI arguments: there is
no machine-specific default and no required environment variable.

| file | role |
| :--- | :--- |
| `measure_sd15_unet_contrib.py` | **the criterion.** Ranks all ConvRot-eligible layers by their aggregate contribution to the UNet's own output. |
| `build_sd15_unet_convrot_int8.py` | **the builder.** Writes the pack: the top-K layers stay FP16, everything else goes ConvRot INT8. CLIP is copied through, the VAE is dropped. |
| `verify_sd15_loader.py` | **the acceptance, through the real loader.** `load_checkpoint_sd15_hswq_weight_dtype` - the only route that enables the Conv2d INT8 path. |
| `verify_sd15_unet.py` | the acceptance with a hand-built module graph (diagnostic only - see the warning below). |
| `quantize_sd1.5_hswq_v1.0.py` | the **previous** selector (static histogram + MSE + SVD). Kept for comparison. |
| `native_convert_int8_sd1.5.py` | the plain all-INT8 converter (no protection). Kept as the native baseline. |

---

## Why V1.0 did not work

V1.0 chose the FP16 protection set from **weight-space** statistics: kurtosis,
outlier ratio, absmax, and a weighted-histogram **MSE** of the quantized weight
at the absmax pack point. That answers "how much does this layer's *weight*
distort", not "how much does this layer move the *output*".

An SD1.5 UNet is a deep series of blocks: a layer's weight error is amplified,
attenuated, or **cancelled** by everything downstream. A weight-space MSE cannot
see that propagation. Measured consequence: V1.0's protection set scored the same
as native, i.e. the protection bought nothing.

Measured on this checkpoint: the aggregate contribution spans **465x** across 282
layers, and the sensitive layers are **not** the ones a weight statistic flags.

---

## The criterion

For each of the 282 ConvRot-eligible layers, one at a time:

1. hold every other layer at FP16,
2. replace this layer by its ConvRot INT8 reconstruction **in the original
   frame** - `dequant(INT8(W @ H^T))` = `(q*s) @ H`,
3. run the UNet forward on 8 fixed inputs,
4. record `1 - cos(out_source, out_perturbed)`.

The result is that layer's **aggregate contribution**. The top-K become the
protection set.

**Why the original frame matters.** The pack stores the *rotated* weight and the
kernel rotates the *activation*, so the error that reaches the output is
`(q*s) @ H - W`. Injecting in the rotated frame (`q*s`) would measure the
rotation itself rather than the error, and inverts the ranking.

**Measured ranking (this checkpoint).**

| rank | layer | the contribution | the share |
| :--- | :--- | :--- | :--- |
| 1 | `input_blocks.1.1.proj_in` | 5.6167e-06 | 10.77 % |
| 2 | `input_blocks.2.1.proj_in` | 2.8616e-06 | 5.49 % |
| 3 | `input_blocks.7.1.transformer_blocks.0.ff.net.0.proj` | 2.6664e-06 | 5.11 % |
| 4 | `input_blocks.5.1.proj_in` | 1.6888e-06 | 3.24 % |
| 5 | `output_blocks.6.0.skip_connection` | 1.6788e-06 | 3.22 % |

The sensitive layers are the **early input-block transition projections**
(`proj_in`) and the **`skip_connection`** bypasses. Both are places where an
error reaches the output without being diluted by later blocks - a property of
the *graph*, invisible to a weight statistic.

Cumulative share of the summed contribution: K=20 -> 53.5 %, K=100 -> 82.6 %.

---

## End-to-end run

Three commands. `<COMFY>` is your ComfyUI checkout (the directory containing
`comfy/`), `<SRC>` is the FP16 checkpoint.

```powershell
$py = "<ComfyUI>/python_embeded/python.exe"      # or any python with torch + the ComfyUI deps

# 1. the criterion (282 layers x 8 seeds; ~70 min on a single consumer GPU)
& $py sd1.5/measure_sd15_unet_contrib.py `
    --comfy "<COMFY>" `
    --ckpt  "<SRC>" `
    --out   impact/contrib_sd15_unet.json `
    --n 8

# 2. the builder (top-K kept FP16; VAE dropped; CLIP copied through)
& $py sd1.5/build_sd15_unet_convrot_int8.py `
    --input   "<SRC>" `
    --output  impact/_work/sd15_k20.safetensors `
    --contrib impact/contrib_sd15_unet.json `
    --top-k 20

# 3. the acceptance, through the REAL loader (256 items; ~60 min)
& $py sd1.5/verify_sd15_loader.py `
    --comfy "<COMFY>" `
    "<SRC>" `
    "impact/_work/sd15_k20.safetensors" `
    --n 256 `
    --out impact/verify_sd15_256.json
```

Every script accepts `--comfy <path>`; `COMFY_PATH` in the environment is also
honoured, but nothing defaults to a machine-specific tree.

`measure_sd15_unet_contrib.py` and `build_sd15_unet_convrot_int8.py` take their
own `--ckpt` / `--input`; `build` does **not** need ComfyUI at all (it is pure
tensor arithmetic over the safetensors file).

---

## What the builder writes

| | |
| :--- | :--- |
| the quantized | every 2D `Linear` **and** 4D `Conv2d` under `model.diffusion_model.`, except the top-K |
| the kept | the top-K layers, plus every 1D tensor - **all stored FP16** |
| CLIP | **copied through unchanged** (`cond_stage_model.`): same keys, same dtypes, zero `.comfy_quant` / `.weight_scale` artifacts |
| VAE | **dropped** (`first_stage_model.` is not written) |
| the scale | INT8 `weight_scale` is FP32 by design; its rank matches the weight (`(O,1)` for 2D, `(O,1,1,1)` for 4D) |

---

## The loader trap (read this before debugging NaN)

An INT8 Conv2d pack **must** be loaded through the HSWQ loader. There are three
reasons, and getting any of them wrong yields NaN (`int8_linear` calls 0, output
all NaN):

1. **`comfy_kitchen` has no INT8 conv kernel** - only `int8_linear`. The conv
   path exists solely in the loader's `_patch_ops_decode_and_conv()`.
2. That patch is gated on a thread-local: `_should_inject_int8_conv()` returns
   `bool(_int8_quant_conv_tls.active)`. Calling
   `apply_comfy_quant_int8_patches()` alone does not open it.
3. There are **two vendored copies** of `comfy_quant_int8` (under `patches/` and
   under `nodes/sd1.5 int8/`), each with its own thread-local. The SD1.5 loader
   collects every `_int8_quant_conv_tls` in `sys.modules` and opens them all.

So use `sd1.5/verify_sd15_loader.py` (which calls
`load_checkpoint_sd15_hswq_weight_dtype`), not `verify_sd15_unet.py`. The latter
builds the graph by hand with `mixed_precision_ops` and **will** produce NaN on a
Conv2d-containing pack - it is kept only for diagnosing Linear-only packs.

Also note: the quant detection must run **before** the
`model.diffusion_model.` prefix is stripped, because
`comfy.utils.detect_layer_quantization()` matches
`k.startswith(unet_key_prefix)`.

---

## Measured result on `0.5xxmix9realistic_v40 0.5BRA7`

| the candidate | INT8 Linear | INT8 Conv2d | cos.mean | cos.min | size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| k=0 | 184 | 98 | 0.99997991 | 0.99973191 | 1056.3 MB |
| k=10 | 180 | 92 | 0.99998317 | 0.99961265 | 1072.4 MB |
| **k=20** | **177** | **85** | **0.99998664** | **0.99977472** | **1083.5 MB** |
| k=40 | 170 | 72 | 0.99999353 | 0.99987036 | 1115.5 MB |
| k=80 | 143 | 59 | 0.99999617 | 0.99995949 | 1167.3 MB |
| k=100 | 182 | - | (not yet accepted) | | 1193.3 MB |

Every measured candidate clears the gate (`cos.mean >= 0.99`, `cos.min >= 0.98`)
by a wide margin. **`cos.min` is not monotone** in K (0.99973191 -> 0.99961265 ->
0.99977472) because it is one draw out of 256; `cos.mean` is monotone and is the
column to read for the trend.

The source checkpoint is 2033.8 MB; the VAE it contains accounts for ~95 MB of
that, so the pack sizes above are not directly comparable to the source total.

---

## Notes

- The acceptance uses fixed random latents and a fixed synthetic context tensor,
  not rendered prompts. It measures the UNet's own output; it is sound for
  ranking and gating, and is **not** a prompt-fidelity check.
- `verify_sd15_loader.py` needs the HSWQ loader package present at
  `<COMFY>/custom_nodes/ComfyUI-HSWQ-Loader-and-Tools`, because the SD1.5 node
  package lives inside it (`nodes/sd1.5 int8/`).
- The contribution pass is deterministic (fixed seeds, float64 comparison), so a
  re-run reproduces the ranking; the builder is exact arithmetic and reproduces
  the pack byte for byte.
