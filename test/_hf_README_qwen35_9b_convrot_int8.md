---
base_model:
- Qwen/Qwen3.5-9B
- h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP
language:
- en
- zh
license: apache-2.0
library_name: transformers
tags:
- image-text-to-text
- text-generation
- multimodal
- abliterated
- uncensored
- bf16
- quantized
- int8
- convrot
- hswq
- comfyui
pipeline_tag: image-text-to-text
---

# Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP (Merged BF16 + HSWQ ConvRot INT8)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

This repository contains the **merged single-file BF16 weights** of the Defiant-Fable 9B checkpoint, plus an **HSWQ ConvRot INT8** quantized copy of the same weights (9217.9 MB, -49.9% disk vs the 18411.9 MB BF16 source).

I have simply merged the split files originally provided by [h3fn3r](https://huggingface.co/h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP) into a unified model for easier handling and deployment.

## Model Details
- **Model Type:** Hybrid Vision-Language Model (Qwen3.5, GatedDeltaNet + GatedAttention, with an MTP draft head)
- **Base Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Modification:** Abliterated / heretic (safety mechanisms and refusals modified), multi-stage fine tune and merge by the upstream author
- **Source of Split Files:** [h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP](https://huggingface.co/h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP/tree/main)
- **Precision:** BF16 (merged) / HSWQ ConvRot INT8 (quantized copy)

## 🛠️ Usage in ComfyUI
Download `Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP_hswq_convrot_int8.safetensors` into:

```
ComfyUI/models/clip/
```

Then load it with the standard **CLIPLoader** node in ComfyUI. Qwen3.5 is a
Llama-family encoder, so ComfyUI detects the `comfy_quant` layers inside the
load path and executes INT8 Tensor Core operations natively; no extra loader and
no manual dequantization step is required. No `config.json` or tokenizer needs
to sit beside the pack.

## 📜 Credits & License
- **Base Architecture & Model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Qwen Team, Apache 2.0)
- **Abliteration / heretic fine tune & merge:** [h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP](https://huggingface.co/h3fn3r/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP) (merged from the split files)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** Apache 2.0 (inherited from the base model)

## 📚 Detailed documentation

- [Qwen3.5-9B ConvRot INT8 impact technical guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Qwen35_9B_ConvRot_INT8_Impact_Technical_Guide.md)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
