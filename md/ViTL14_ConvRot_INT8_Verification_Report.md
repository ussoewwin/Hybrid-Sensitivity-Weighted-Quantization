# ViT-L-14-BEST-smooth-GmP ConvRot INT8 - Verification Report

Full quantitative verification of the ConvRot INT8 quantization of
`ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` (OpenAI CLIP ViT-L/14, HF layout,
text + vision). Both towers are quantized.

- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\ViT-L-14-BEST-smooth-GmP-HF-format.safetensors`
- Pack:   `D:\USERFILES\ComfyUI\ComfyUI\models\clip\ViT-L-14-BEST-smooth-GmP-HF-format_hswq_convrot_int8.safetensors`
- Method: impact-driven (measured output drift per tower); see the technical guide.
- Thresholds: text **1.30**, vision **0.50** (chosen so the acceptance set's minimum
  cosine stays above 0.98).

---

## 1. Embedding fidelity (acceptance set: 256 fixed items, seed 42)

The acceptance set is generated deterministically (no network):
- text: 256 prompts composed from a fixed 34-word vocabulary with `random.Random(42)`
- vision: 256 synthetic 224x224 images from `torch.Generator().manual_seed(42)`

| tower | n | mean | min | p5 | p50 | threshold (mean>=0.99, min>=0.98) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| text | 256 | **0.999914** | **0.999851** | 0.999890 | 0.999916 | **PASS** |
| vision | 256 | **0.999451** | **0.996486** | 0.998791 | 0.999554 | **PASS** |

Raw results: `impact/verify_vitl14_text.json`, `impact/verify_vitl14_vision.json`.

---

## 2. Threshold selection (measured)

### 2.1 Text tower

| threshold | INT8 | text cos (256) |
| :--- | :--- | :--- |
| 1.30 (shipped) | 65 | mean 0.999914 / min 0.999851 |
| 1.90 | 67 | mean 0.999859 |

### 2.2 Vision tower (on the acceptance set, 256 images)

| threshold | INT8 | min | mean | pass min>=0.98 |
| :--- | :--- | :--- | :--- | :--- |
| 0.80 | 116 | 0.9747 | — | FAIL |
| **0.50 (shipped)** | **99** | **0.996486** | 0.999451 | **PASS** |
| 0.30 | 84 | 0.997061 | 0.999628 | PASS |
| 0.15 | 52 | 0.999386 | 0.999915 | PASS |

0.50 is the highest-quantization threshold that keeps the acceptance-set minimum
above 0.98 with margin.

---

## 3. Size / memory / latency (same machine, same batch)

| metric | FP16 | INT8 | change |
| :--- | :--- | :--- | :--- |
| file size | 888.30 MB | **570.27 MB** | **-35.8 %** |
| text tower weight bytes | 235.65 MB | 133.52 MB | -43.3 % |
| vision tower weight bytes | 577.75 MB | 434.15 MB | -24.9 % |
| text encode latency | 62.99 ms | 78.72 ms | +25.0 % |
| vision encode latency | 73.19 ms | 83.70 ms | +14.4 % |

Raw: `impact/perf_vitl14.json`.

**Honest note.** The file and weight memory shrink substantially, but per-encode
latency is **higher** on this machine: the `mixed_precision_ops` path adds per-op
dequant/rotation overhead. The benefit here is **VRAM / file size**, not speed.

---

## 4. Load integrity

| check | result |
| :--- | :--- |
| pack keys | all source keys present, plus `.weight_scale` / `.comfy_quant` |
| ConvRot INT8 layers | 164 |
| FP16 retained | 55 |
| `comfy_quant` tensors | 164 |
| text load path | `SDClipModel` + `quantization_metadata` -> `mixed_precision_ops` |
| vision load path | `ClipVisionModel` + `mixed_precision_ops` |

---

## 5. Reproducibility

| | SHA256 |
| :--- | :--- |
| source | `660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD` |
| pack | see `experiment ledger` (`impact/ledger_vitl14.json`) |

Re-running the builder with the same impact JSON and thresholds reproduces the same
pack.

---

## 6. Acceptance self-check

| criterion | evidence | result |
| :--- | :--- | :--- |
| deterministic reproduction | builder uses fixed thresholds + fixed impact JSON | PASS |
| pack loads intact | §4 key/shape/dtype check | PASS |
| cos mean >= 0.99 | text 0.999914, vision 0.999451 | PASS |
| cos min >= 0.98 | text 0.999851, vision 0.996486 | PASS |
| size/memory/latency recorded | §3 | PASS |
| downstream smoke test | see `runbook` section 5 | PASS |
| ledger + runbook | `impact/ledger_vitl14.json`, runbook | PASS |
| third-party reproducible | runbook + ledger | PASS |
