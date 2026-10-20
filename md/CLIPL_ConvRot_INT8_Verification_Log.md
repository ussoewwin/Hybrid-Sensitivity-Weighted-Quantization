# CLIP-L ConvRot INT8 - Verification Log

Machine-checkable record of every measured value behind the CLIP-L ConvRot INT8
quantization. All numbers were produced on this machine with the commands below.

Environment:
- ComfyUI: `D:\USERFILES\ComfyUI\ComfyUI`
- Python: `D:\USERFILES\ComfyUI\python_embeded\python.exe`
- Source: `D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l.safetensors`
- Pack:   `D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l_hswq_convrot_int8.safetensors`
- Loader arm: `ComfyUI-HSWQ-Loader-and-Tools/nodes/clip_convrot_int8/clip_quant.py`

---

## 1. Impact measurement (`sdxl/diag_impact_clip.py --full`)

```
[load] clip_l.safetensors
[build] cond_stage_model=SD1ClipModel inner=SD1CheckpointClipModel
[*] ref shape=(236544,) finite=True
[*] candidate Linear modules: 73
[done] n=73 median=1.5766e-01 p90=1.2932e+00 max=1.8926e+00
[saved] impact/impact_clip_l.json
```

Top drift layers:

| drift | module |
| :--- | :--- |
| 1.8926 | text_model.encoder.layers.0.mlp.fc2 |
| 1.8610 | text_model.encoder.layers.0.mlp.fc1 |
| 1.6163 | text_model.encoder.layers.0.self_attn.v_proj |
| 1.5935 | text_model.encoder.layers.0.self_attn.out_proj |
| 1.0809 | text_model.encoder.layers.0.self_attn.q_proj |
| 0.0000 | text_projection |

Threshold sensitivity:

| threshold | protected |
| :--- | :--- |
| 0.10 | 59 |
| 0.15 | 37 |
| 0.20 | 27 |
| 0.30 | 22 |
| 0.50 | 16 |

---

## 2. Root cause of the "broken pack" (measured, stock CLIPLoader, no fix)

```
FP16 via node path -> finite True
PACK via node path (stock) -> finite=True cos=0.027794 relMSE=1.9381e+00 maxabs=33.40933
Linear total: 146  with QuantizedTensor: 0
```

Conclusion: without the arm, the pack's int8 weights load as raw fp16 (0/146
quantized) and the embedding is wrong (cos 0.028).

---

## 3. After the loader arm (real package `__init__` path)

```
package __init__ executed
INT8 quant: (146, 70)
REAL-PATH INT8 finite=True cos=0.999957 relMSE=8.8677e-05 maxabs=0.06620
```

70 of 146 Linear layers become QuantizedTensor; the embedding matches FP16.

---

## 4. Threshold gate (armed path, 3 fixed prompts)

```
t015: quant=70/146 cos=0.999957 relMSE=8.8677e-05 maxabs=0.06620 finite=True
t02 : quant=90/146 cos=0.999939 relMSE=1.2538e-04 maxabs=0.06767 finite=True
```

Edge cases (long text / Japanese / symbols / repetition):

```
t015 edge: finite=True absmax=33.0638 zero_frac=0.0000
t02  edge: finite=True absmax=33.0638 zero_frac=0.0000
```

---

## 5. Reproducibility (identical SHA256 on re-run)

```
shipped : 52B538A24C02605F5763CB6ECE88848A44858482EC8624F2567D71AA91B68D05
re-run  : 52B538A24C02605F5763CB6ECE88848A44858482EC8624F2567D71AA91B68D05
MATCH: True
```

Source hash:

```
clip_l.safetensors : 660C6F5B1ABAE9DC498AC2D21E1347D2ABDB0CF6C0C0C8576CD796491D9A6CDD
```

---

## 6. Size / memory / latency

```
FP16: weight_bytes=235.6 MB  per_encode=45.4 ms
INT8: weight_bytes=202.6 MB  per_encode=98.5 ms
```

File sizes:

```
clip_l.safetensors                      234.7 MB
clip_l_hswq_convrot_int8.safetensors    201.7 MB  (-14.1 %)
```

---

## 7. Acceptance self-check

| criterion | evidence | result |
| :--- | :--- | :--- |
| reproducible weight hash | §5 SHA256 match | PASS |
| embedding fidelity | §4 cos 0.999957 >= 0.99 | PASS |
| downstream quality | §3-4 conditioning matches FP16 (cos 0.9999+) | PASS |
| resource improvement | §6 file -14.1 %, weight bytes -14.0 %; latency slower (reported) | PARTIAL (size yes, speed no) |
| edge-case robustness | §4 finite, absmax stable | PASS |
| traceable records | this file + impact JSON + guide | PASS |
| handover doc | `md/CLIPL_ConvRot_INT8_Impact_Technical_Guide.md` | PASS |


---

## 8. Correction (FLUX.1 path, full scope)

The measurements above were taken on the wrong clip_type (STABLE_DIFFUSION). The
correct path for `clip_l.safetensors` is **FLUX.1** (`CLIPType.FLUX`), and the shipped
pack uses the **full scope** (all 73 2D Linear layers, embedding included).

### 8.1 Threshold sweep (FLUX path, measured)

```
FP16 absmax=33.0637
t15  quant=36 prot=38 size=165.7 MB absmax=33.0638 cos=0.999954
t50  quant=57 prot=17 size=147.2 MB absmax=33.0638 cos=0.999921
t100 quant=60 prot=14 size=142.2 MB absmax=33.0636 cos=0.999916
all  quant=73 prot=1  size=118.1 MB absmax=33.0620 cos=0.999848
```

### 8.2 Shipped pack (full scope)

```
file size          118.1 MB  (source 234.7 MB, -49.7 %)
quantized layers   73 / 73
comfy_quant        73
convrot_groupsize  256 (all)
absmax             FP16 33.0637 -> INT8 33.0620
cos vs FP16        0.999848
SHA256             691437D48F36A609BF329853BCCC01E1843FAD8B20233D8A1509D47D7495986B
```

### 8.3 Standard-path load (real node)

```
DualCLIPLoader(type="flux", clip_l_hswq_convrot_int8 + t5xxl): quant=(490, 328)
```

328 of 490 Linear layers become QuantizedTensor through the standard loader - no
custom node. FLUX.1 CLIP ConvRot INT8 works with the standard loaders as-is.
