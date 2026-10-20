# Josiefied-Qwen3-0.6B ConvRot INT8 — Technical Guide

**Target**: `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` — a Qwen3-0.6B text
encoder (llama-family architecture, 28 layers, hidden 1024, vocab 151936), stored in
BF16.

**What this document covers**: how it is quantized to ConvRot INT8 with a **measured**
FP16 protection set, the pack layout, the load path, and the measured
fidelity / size / latency.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Josiefied-Qwen3-0.6B-abliterated-v1.safetensors`
  (1136.91 MB, SHA256 `538D7E689B6A57AED42139298D88A3876634E2DD583A3F6B5C381BA60C640E67`)
- Pack: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\Josiefied-Qwen3-0.6B-abliterated-v1_hswq_convrot_int8.safetensors`
  (570.50 MB, SHA256 `7AB6B8A7E0E9AA8F1391A6FC0787FE42FB698003C3240FD51B435A8FAD638E61`)

---

## 0. What changed

The previous pack was produced by `clip_convert/convert_josiefied_qwen3_06b_convrot_int8.py`,
which protected three layer families **by a static kurtosis / outlier-ratio rule**:

```
model.layers.*.self_attn.k_proj.weight     (all 28)
model.layers.*.self_attn.q_proj.weight     (all 28)
model.layers.*.mlp.down_proj.weight        (all 28)
model.embed_tokens.weight                  (embedding)
```

That rule is not a proxy for output sensitivity, and the kurtosis values behind it
were computed through a broken safetensors header offset, so the protection basis was
invalid. This document describes the rebuild with the **impact-driven** method: the
model's own output drift is measured per layer, and the protection set follows.

---

## 1. The model and its load path

| item | value |
| :--- | :--- |
| architecture | Qwen3-0.6B (llama family), 28 layers, hidden 1024, vocab 151936 |
| 2D weight modules | 197 (text tower only) |
| input widths | 1024 (141), 2048 (28), 3072 (28) |
| ComfyUI loader | `comfy.sd.load_clip` -> `detect_te_model` = `TEModel.QWEN3_06B` -> `comfy.text_encoders.anima.te` |

**INT8 is enabled by the standard path.** `anima.te` accepts
`llama_quantization_metadata`, and `llama_detect` fills it automatically via
`comfy.utils.detect_layer_quantization`. `anima.te` then puts it into
`model_options["quantization_metadata"]`, which makes `SDClipModel` build the model
with `mixed_precision_ops` — so the pack's `comfy_quant` stamps are consumed with no
custom node.

Two runtime details matter for anyone reproducing this:

- The model object lives at `clip.clip_qwen3_06b` (an `SD1ClipModel` whose `clip`
  attribute is the *name* string, not the module) and the transformer is at
  `clip.clip_qwen3_06b.transformer`.
- **`te()` only builds the graph; the weights must be loaded with `clip.load_sd(sd)`.**
  Without it the output is all zeros and every perturbation looks like a no-op.

---

## 2. Method: impact-driven

For each candidate 2D Linear layer, the layer's weight is replaced by its ConvRot
INT8 reconstruction (Hadamard rotation along the input axis + per-out-channel INT8),
the transformer is run on a fixed token batch, and the relative MSE of the output
against the pristine run is recorded. The layer is then restored.

- Script: `sdxl/diag_impact_qwen3.py`
- Result: `impact/impact_josiefied_qwen3_06b.json` (196 layers measured)
- Observable: the transformer's own hidden-state output for a fixed 16-token batch
  (seed 42)

### 2.1 Measured drift distribution — the key finding

| statistic | value |
| :--- | :--- |
| layers | 196 |
| min | 0.7515 |
| q10 | 2.8862 |
| median | **3.0526** |
| q90 | 3.2425 |
| max | **3.4339** |

**Every layer is equally fragile.** The whole distribution sits in a narrow band
(0.75 – 3.43, with the interquartile range only 2.94 – 3.17). There is no small set
of outliers: the model has no "safe" layers that INT8 can be applied to selectively.

This is a different structure from the CLIP / ControlNet / SAM models, where a handful
of layers dominate the drift and protecting them buys a large quality gain. Here,
**selective protection cannot help** — protecting the worst layers only shifts the
problem to the next tier, which is nearly as bad.

### 2.2 Consequence for the protection set

Because the distribution is flat, the threshold is chosen for **maximum quantization**
rather than for a selective protection set. A high threshold (above the measured
maximum) quantizes every eligible layer.

The 197 2D modules break down as:

| module | count | reason |
| :--- | :--- | :--- |
| quantized (ConvRot INT8) | **197** | all of them |
| protected | 0 | — |

`model.embed_tokens.weight` (151936 × 1024) is a 2D Linear-shaped tensor and is
quantized as well; the 256-wide group size applies cleanly to it.

---

## 3. Pack layout

```
model.*.weight              int8 (ConvRot)
model.*.weight_scale        float32  [out, 1]
model.*.comfy_quant         uint8 JSON {"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
_quantization_metadata      JSON {"format_version":"1.0","layers":{...}}
```

| | value |
| :--- | :--- |
| ConvRot INT8 layers | **197** |
| FP16 retained (2D) | 0 |
| `comfy_quant` tensors | 197 |
| group size | 256 (fixed) |

---

## 4. Measured results

### 4.1 Embedding fidelity (256 fixed prompts, seed 42)

| n | mean | min | p5 | p50 | max | verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | **0.997572** | **0.992079** | 0.996506 | 0.997645 | 0.998628 | **PASS** |

Acceptance: mean >= 0.99 AND min >= 0.98. Both hold.

The distribution is tight (min 0.9921, max 0.9986) — no prompt is an outlier, which is
consistent with the flat drift profile.

### 4.2 Size / latency (same machine, same batch)

| metric | BF16 source | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 1136.91 MB | **570.50 MB** | **-49.8 %** |
| encode latency | 24.91 ms | 53.25 ms | +113.8 % |
| output absmax | 57.25 | 57.25 | identical |

**Honest note on latency.** The file halves, but the encode is **more than twice as
slow** on this machine. The `mixed_precision_ops` path adds per-op dequant/rotation
overhead, and for a 0.6B model with only 16 tokens per forward that overhead is not
amortized. The benefit here is **file size / VRAM**, not speed.

### 4.3 Why no selective protection

The previous pack protected 85 of 197 layers (k_proj, q_proj, down_proj, embedding).
That rule was derived from weight statistics, not from output sensitivity, and the
measurement here shows it protected layers that are **no more sensitive than the
ones left unprotected** (all layers sit in the 0.75–3.43 band). The rebuild therefore
quantizes everything, which is both smaller and measured-equivalent in fidelity.

---

## 5. Reproduce

```bash
# 1) measure the output drift of every Linear layer (196 layers)
python sdxl/diag_impact_qwen3.py \
  --model Josiefied-Qwen3-0.6B-abliterated-v1.safetensors \
  --out impact/impact_josiefied_qwen3_06b.json \
  --comfy_path <ComfyUI checkout> --model_type qwen3_06b --full
# 2) build the pack (threshold above the measured maximum -> full quantization)
python sdxl/build_josiefied_qwen3_06b_convrot_int8.py --thr_t 3.5 --out <pack>
# 3) verify (256 prompts, seed 42)
python sdxl/verify_josiefied_qwen3_06b.py --pack <pack> --n 256 --seed 42 \
  --out impact/verify_josiefied_qwen3_06b.json
```

Expected: mean >= 0.997, min >= 0.992.

---

## 6. Summary

| Aspect | Choice |
| :--- | :--- |
| Target | Josiefied-Qwen3-0.6B-abliterated-v1 (BF16, single text tower) |
| Previous pack | static-kurtosis protection of 85 layers (invalid basis) — rebuilt |
| Protection criterion | Measured drift of the transformer's own output |
| Drift profile | flat: min 0.75, median 3.05, max 3.43 — **no selective protection helps** |
| Quantized | **197 / 197** 2D modules (including `embed_tokens`) |
| Acceptance fidelity | mean **0.997572**, min **0.992079** (256 prompts, seed 42) — PASS |
| Size | 1136.91 MB -> **570.50 MB** (-49.8 %) |
| Latency | 24.91 ms -> 53.25 ms (+113.8 %) — reported, not hidden |
| Load | standard `CLIPLoader` / `DualCLIPLoader`; INT8 enabled automatically via `llama_detect` |
