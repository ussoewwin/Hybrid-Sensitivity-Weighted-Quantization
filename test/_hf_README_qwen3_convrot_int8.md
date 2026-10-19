---
license: apache-2.0
base_model:
  - huihui-ai/Qwen3-4B-abliterated
  - huihui-ai/Qwen3-8B-abliterated
  - huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated
tags:
  - qwen3
  - text-generation
  - conversational
  - abliterated
  - uncensored
  - safetensors
  - fp16
  - int8
  - convrot
  - hswq
  - single-file
  - vision-language
pipeline_tag: text-generation
library_name: transformers
---

# Qwen3 Abliterated FP16 & ConvRot INT8 Collection

<p align="center">
  <img src="https://raw.githubusercontent.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization/main/icon.png" width="128">
</p>

This repository provides single-file `.safetensors` releases for **Qwen3 abliterated** variants across **FP16** and **HSWQ ConvRot INT8** (Hybrid-Sensitivity-Weighted-Quantization).

It serves as a unified release collection combining pure text LLM variants (4B, 8B) and multimodal Vision-Language reasoning variants (VL-4B-Thinking) with safety vectors removed/abliterated.

- **Technical Details & Quantization Framework:** [Hybrid-Sensitivity-Weighted-Quantization (HSWQ)](https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization)
- **ComfyUI Loader for ConvRot INT8:** [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)

---

## 🟢 Available Models & Repository Files

| Filename | Architecture | Format / Precision | Size | Upstream Base Model | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen3_4b_abliterated_fp16_converted.safetensors` | Qwen3 4B | Single-file FP16 | ~8.04 GB | `huihui-ai/Qwen3-4B-abliterated` | Available |
| `qwen3_4b_abliterated_hswq_convrot_int8.safetensors` | Qwen3 4B | HSWQ ConvRot INT8 | ~5.78 GB | `huihui-ai/Qwen3-4B-abliterated` | Available |
| `Qwen3-8B-abliterated-fp16.safetensors` | Qwen3 8B | Single-file FP16 | ~16.38 GB | `huihui-ai/Qwen3-8B-abliterated` | Available |
| `Qwen3_VL_4B_Thinking_abliterated.safetensors` | Qwen3 VL 4B Thinking | Single-file FP16 | ~8.88 GB | `huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated` | Available |
| `Qwen3_VL_4B_Thinking_abliterated_hswq_convrot_int8.safetensors` | Qwen3 VL 4B Thinking | HSWQ ConvRot INT8 | ~6.44 GB | `huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated` | Available |

---

## 🔬 Quantization & Conversion Methods

### 1. Single-File FP16 Merging
- Consolidated multi-shard safetensors checkpoints into single, standalone `.safetensors` files without missing keys or dangling references.
- Converted all float parameters to standard IEEE FP16 (`torch.float16`) for lightweight portability.

### 2. HSWQ ConvRot INT8 (Hybrid-Sensitivity-Weighted-Quantization)
- Layer-wise sensitivity and trajectory impact profiling identifies outlier-susceptible and high-importance layers.
- Preserves critical attention/vision projection layers in higher precision (FP16/BF16) while packing resilient bulk Linear layers into ConvRot INT8.
- Maximizes inference speed and memory reduction while guarding against reasoning degradation and catastrophic trajectory bifurcations.

---

## 💻 Usage

### Hugging Face Transformers (Text Generation)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8"

# Load FP16 single-file safetensors
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### ComfyUI Integration (ConvRot INT8)
For ComfyUI workflows utilizing quantized text encoders or VL models:
1. Install [ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools).
2. Place the quantized `.safetensors` model in your ComfyUI models directory (`models/LLM` or `models/text_encoders`).
3. Use the HSWQ Loader node to run ConvRot INT8 tensorwise execution.

---

## ⚠️ Important Safety & Compliance Notice

- The models in this collection have undergone refusal vector removal (abliteration) / reduced safety conditioning compared to standard aligned base checkpoints.
- These weights are published strictly for research, testing, and authorized evaluation purposes.
- Users must evaluate model responses before deployment and ensure usage complies with all applicable local laws, regulations, and platform policies.

---

## 🤝 Credits & Acknowledgments

- Upstream Qwen3 4B Abliterated: `huihui-ai/Qwen3-4B-abliterated`
- Upstream Qwen3 8B Abliterated: `huihui-ai/Qwen3-8B-abliterated`
- Upstream Qwen3 VL 4B Thinking Abliterated: `huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated`
- Original Base Architecture: Qwen Team (`Qwen/Qwen3`)

---

## 📜 License

This repository follows the **Apache 2.0** license. Please also abide by the original licenses and usage guidelines established by upstream authors and the Qwen Team.
