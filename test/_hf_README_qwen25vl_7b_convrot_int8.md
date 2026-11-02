---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
language:
- en
license: apache-2.0
library_name: transformers
tags:
- vision
- image-to-text
- text-generation
- multimodal
- abliterated
- fp16
- quantized
- int8
- convrot
- hswq
- comfyui
---

# Qwen2.5-VL-7B-Instruct-abliterated (Merged FP16 + HSWQ ConvRot INT8)

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

This repository contains the **merged FP16 weights** of the abliterated version of Qwen2.5-VL-7B-Instruct, plus an **HSWQ ConvRot INT8** quantized copy of the same weights (9289.6 MB, −41.3% disk vs the 15816.1 MB BF16 source).

I have simply merged the split files originally provided by [ussoewwin](https://huggingface.co/ussoewwin/Qwen2.5-VL-7B-Instruct-abliterated) into a unified model for easier handling and deployment.

## Model Details
- **Model Type:** Vision-Language Model (Qwen2.5-VL)
- **Base Model:** [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **Modification:** Abliterated (Safety mechanisms/refusals modified)
- **Source of Split Files:** [ussoewwin/Qwen2.5-VL-7B-Instruct-abliterated](https://huggingface.co/ussoewwin/Qwen2.5-VL-7B-Instruct-abliterated/tree/main)
- **Precision:** FP16 (merged) / HSWQ ConvRot INT8 (quantized copy)

## 🛠️ Usage in ComfyUI
Download `Qwen2.5-VL-7B-Instruct-abliterated_converted_hswq_convrot_int8.safetensors` into:

```
ComfyUI/models/clip/
```

Then load it with the standard **CLIPLoader** node in ComfyUI (`qwen25_7b` type detection via the Quantization Framework / HSWQ ComfyUI Nodes, Load Text Encoder). ComfyUI detects the `comfy_quant` header metadata and executes INT8 Tensor Core operations natively; no manual dequantization step is required.

## 📜 Credits & License
- **Base Architecture & Model:** [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Qwen Team, Apache 2.0)
- **Abliteration:** the abliterated variant of the above (merged from the split files)
- **Quantization Framework:** [HSWQ / ConvRot INT8 Pipeline](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **License:** Apache 2.0 (inherited from the base model)

## 📚 Detailed documentation

- [Qwen2.5-VL-7B ConvRot INT8 impact technical guide](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Qwen25VL7B_ConvRot_INT8_Impact_Technical_Guide.md)

## 🔒 Security

**Backdoor audit.** Each file's audit status - verdict strength, weight coverage, the record behind it,
and the limits of what that measurement can show - is in
[Backdoor Audit Verdicts](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/blob/main/md/Backdoor_Audit_Verdicts.md). The audit code, the procedure and the raw records are in the same
repository.
