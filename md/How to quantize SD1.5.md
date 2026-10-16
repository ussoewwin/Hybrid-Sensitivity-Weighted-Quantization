# How to Quantize SD1.5 (HSWQ Hybrid ConvRot INT8)

Quantize **Stable Diffusion 1.5 (SD1.5)** checkpoints with **HSWQ V1.0** —
sensitivity-ranked **FP16 protection under a hard 300 MiB budget**, FULL
**ConvRot INT8** on the remainder, embedded-CLIP ConvRot INT8, and an automatic
deterministic trajectory benchmark. Two entry points:

- **Recommended (one command):** HSWQ quantizer CLI — `sd1.5/quantize_sd1.5_hswq_v1.0.py`
- **In-ComfyUI:** `HSWQ Quantize ConvRot INT8` node (HSWQ V1.0 full pipeline) / `Native ConvRot INT8 Quantize` node (`model_type = "SD1.5"`, native baseline)
- **Native (no protection):** `sd1.5/native_convert_int8_sd1.5.py` (all-layer
  ConvRot INT8, no FP16 budget, no calibration) — kept for A/B comparison

<p align="center">
  <img src="../png/hswq sd1.5.png" alt="ComfyUI HSWQ Quantize ConvRot INT8 Workflow for SD1.5" width="800">
</p>

---

## Why HSWQ (not plain Native) for SD1.5

| | Native ConvRot INT8 | **HSWQ V1.0 (this document)** |
|---|---|---|
| Quantized layers | all eligible Linear+Conv2d | remainder after FP16 protection |
| FP16 protection | none | **DualMonitor + analyze + V4 MSE ranking, hard 300 MiB budget** |
| CLIP (text encoder) | ConvRot INT8 (fp16 fallback on old loaders) | ConvRot INT8 **+ loader fix: metadata consumed correctly** |
| Calibration | none (Card 1 opt-in) | **r32 recipe (32 samples × 25 steps) mandatory** |
| Embedded CLIP/VAE | CLIP kept, VAE stripped | CLIP ConvRot INT8 (197→72 quantized + 125 float), VAE stripped |
| Post-quantize bench | node only (10-25 seeds) | **automatic 25-seed × 25-step trajectory bench, INT8 CLIP conditioning** |
| Typical size (0.5xxmix9realistic_v40+0.5BRA7) | ~1.02 GB | **~1.28 GB** (measured 1275 MB) |

Measured rationale (0.5xxmix9realistic_v40+0.5BRA7, 25 seeds × 25 steps, 512×512):

| pack | mean final-cos | same-image | speed |
|---|---|---|---|
| Native (all-INT8) | 0.94760 | 13/25 | 0.65x (CPU-bound 512) |
| **HSWQ V1.0 (300 MiB)** | **0.97120** (FP16-CLIP cond) / **0.96613** (INT8-CLIP cond) | 11-14/25 | **+14.9%** (CUDA-graph) |

Uniform INT8 alone can drop below the 0.95 gate on fine-tuned/merged SD1.5
checkpoints. HSWQ protects the most vulnerable layers (ranked by DualMonitor
sensitivity + V4 MSE + analyze severity) inside the fixed 300 MiB frame, then
ConvRot-INT8s everything else.

> **`How to` protocol is mandatory:** calibration is the r32 recipe (32 samples,
> 25 steps). `analyze/analyze_sd1.5_distribution.py` derives all tunables
> (VETO fences, alpha/beta, MAD floors, FP16 priority) from the model itself —
> no hand-tuned constants.

---

## Clone & Environment Setup

```bash
git clone https://github.com/ussoewwin/Hybrid-Sensitivity-Weighted-Quantization.git
cd Hybrid-Sensitivity-Weighted-Quantization
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -U comfy_kitchen
pip install diffusers accelerate tqdm safetensors
```

- `comfy_kitchen` — INT8 GEMM kernels, ConvRot Hadamard rotation (CUDA fused gs256).
- `diffusers` — the calibration pipeline (SD1.5 base is pulled from HuggingFace
  on first run; ~2 GB into `$env:HF_HOME`).

Calibration prompts: `sample/calibration_prompts_512.txt` (32 prompts).

---

## Method 1 (Recommended): HSWQ Quantizer CLI

```powershell
$env:HF_HOME = 'D:\USERFILES\MyDocuments\HuggingFace'
$env:PYTHONIOENCODING = 'utf-8'
python sd1.5/quantize_sd1.5_hswq_v1.0.py `
  --input  "D:/models/checkpoints/0.5xxmix9realistic_v40+0.5BRA7.safetensors" `
  --output "D:/models/checkpoints/0.5xxmix9realistic_v40+0.5BRA7_hswq_int8.safetensors" `
  --calib_file "sample/calibration_prompts_512.txt"
```

One command runs the whole mandated pipeline:

1. **Distribution analysis** — `analyze/analyze_sd1.5_distribution.py` profiles
   all 282 UNet layers (CPU safetensors scan; Tukey fences, engine tunables,
   alpha/beta derivation). Auto-generates the profile JSON (cached) at
   `analyze/json/<model>_distribution_profile.json`.
2. **r32 calibration** — DualMonitor FP16/INT8 sensitivity (32 samples × 25
   steps, 512×512, StableDiffusionPipeline).
3. **V4 MSE** — Full-SVD×RMS mixed impact per layer (Full-SVD scheduled ON).
4. **FP16 300 MiB selection** — priority fill of keep-layers under the hard
   budget (post-pack assert: any float Linear/Conv leak aborts).
5. **ConvRot INT8 pack** — remaining Linear/Conv2d, Hadamard group 256/64/4 by
   feature dim; **CLIP Linear layers also ConvRot INT8** (72 quantized, 125
   embeddings/norms/biases kept float); embedded VAE (`first_stage_model.*`)
   stripped.
6. **25-seed trajectory bench** — 512×512, dpmpp_2m/karras, **INT8 CLIP
   conditioning**, CUDA-graph executor. Skippable with `--no-bench`.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` / `--output` | required | Source SD1.5 checkpoint / output pack. |
| `--calib_file` | required | Calibration prompts (r32 = 32 lines). |
| `--num_calib_samples` | `32` | r32 recipe. Do not lower. |
| `--num_inference_steps` | `25` | Steps per calibration sample. |
| `--fp16_budget_mb` | `300` | **Owner hard ceiling.** Must be exactly 300. |
| `--profile` | auto | Distribution profile JSON under `analyze/json/` (auto-generated if missing). |
| `--comfy_path` | auto | ComfyUI root (auto-detected). |
| `--bias_correction` | **OFF** | Card 1: cancel systematic INT8 output bias into layer biases. |
| `--per_channel_int8` | OFF | Card 3: per-output-channel absmax. |
| `--asymmetric_int8` | **forced OFF** | Card 2 is disabled in V1.0. |
| `--convrot` / `--no-convrot` | ON | FULL ConvRot Hadamard rotation. |
| `--groupsize` | `256` | Hadamard group (adaptive gs256/64/4 by feature dim). |
| `--bench` / `--no-bench` | ON | Post-quantize 25-seed trajectory bench. |

### Expected Output

```
Conversion done:
  INT8 layers: 203
  FP16-kept layers (budget winners): 79
  Dropped embedded VAE layers (first_stage_model.*): 248
  CLIP ConvRot INT8: 72, plain INT8: 0, kept as-is (embeddings/norms/biases): 125
...
post-pack FP16 keep: 299.96 MiB / ceiling=300 MiB
Saved INT8 quantized model: ...
CLIP INT8 layers: 72 (convrot=72)
```

Benchmark acceptance (see §Validation): log must contain
`Successfully encoded prompt with ConvRot INT8 CLIP` and must **not** contain
`using baseline CLIP conditioning`.

---

## Method 2: ComfyUI Nodes

### 1. `HSWQ Quantize ConvRot INT8` (Recommended In-Workflow)

<p align="center">
  <img src="../png/hswq sd1.5.png" alt="ComfyUI HSWQ Quantize ConvRot INT8 Workflow for SD1.5" width="800">
</p>

Dedicated in-workflow HSWQ V1.0 quantizer (`comfyui_nodes/hswq_sd15_quantize.py`). Executes the complete HSWQ V1.0 pipeline directly within ComfyUI without manual terminal commands:

#### Node Setup & Wiring

1. **`Load Checkpoint`**: Load the target SD1.5 FP16 checkpoint (e.g. `0.5xxmix9realistic_v40+0.5BRA7.safetensors`).
   - Connect **`MODEL`** → `HSWQ Quantize ConvRot INT8` (`model`)
   - Connect **`CLIP`** → `HSWQ Quantize ConvRot INT8` (`clip`)
   - Connect **`VAE`** → `HSWQ Quantize ConvRot INT8` (`vae`, optional)
2. **`HSWQ Quantize ConvRot INT8`**: Node located under category `HSWQ/Quantize`.
3. **`Show Text`**: Connect `HSWQ Quantize ConvRot INT8` (`report`) → `Show Text` (`text`) to view live logs, quantize completion stats, and the deterministic 25-seed trajectory benchmark results directly in the ComfyUI canvas.

#### Node Parameters

| Parameter | Default / Recommended | Description |
|---|---|---|
| `model_type` | `SD1.5` | Target architecture family (dropdown). |
| `output_path` | *(empty)* | Output `.safetensors` path. If empty, automatically writes to `<ComfyUI>/output/<stem>_hswq_int8_<timestamp>.safetensors`. |
| `calib_file` | `128` | Dropdown to select calibration prompts sheet from `sample/` (`128`, `256`, or `512` prompts). Default is `128` (`sample/calibration_prompts_128.txt`). |
| `num_calib_samples` | `32` | Mandatory r32 calibration recipe. Do not lower. |
| `num_inference_steps` | `25` | Diffusion steps per calibration sample. |
| `fp16_budget_mb` | `300.0` | **Mandatory owner hard ceiling.** Must be exactly `300.0` MiB. |
| `group_size` | `256` | ConvRot Hadamard rotation block size (power of 4, default: 256). |
| `convrot` | `true` | Enables FULL ConvRot Hadamard rotation on UNet and CLIP. |
| `bias_correction` | `false` | Card 1: systematic INT8 output bias cancellation (default OFF). |
| `run_benchmark` | `true` | Automated 25-seed deterministic trajectory benchmark post-quantization. |
| `input_checkpoint` | *(empty)* | Optional manual override path if `model` does not supply `cached_patcher_init`. |

#### In-Workflow Execution Steps

1. Click **Queue Prompt**.
2. The node automatically:
   - Purges ComfyUI VRAM and collects garbage before launching the quantizer to prevent memory exhaustion.
   - Automatically derives model-specific tunables via `analyze/analyze_sd1.5_distribution.py`.
   - Runs r32 DualMonitor calibration (32 samples × 25 steps).
   - Computes V4 Full-SVD×RMS weighted histogram MSE ranking.
   - Enforces the hard 300 MiB FP16 protection budget (most vulnerable layers stay FP16).
   - Packs remaining UNet Linear and Conv2d layers into ConvRot INT8 (`int8_tensorwise`).
   - Converts CLIP text encoder layers to ConvRot INT8 (72 quantized, 125 float) and strips embedded VAE (`first_stage_model.*`).
   - Executes the automated 25-seed deterministic trajectory benchmark using INT8 CLIP conditioning.
3. The trailing 60 lines of the console execution log and final benchmark scores (mean cosine $\ge 0.96$, 0/25 bifurcated) appear in the `Show Text` node upon completion.

### 2. `Native ConvRot INT8 Quantize` (Native All-INT8 Workflow)

<p align="center">
  <img src="../png/sd1.5.png" alt="ComfyUI Native ConvRot INT8 Quantize Workflow for SD1.5" width="800">
</p>

In-workflow Native quantizer (`model_type = "SD1.5"`): UNet + CLIP are all-layer INT8 quantized without FP16 budget protection (A/B baseline in ComfyUI).

---

## Method 3: Native CLI (A/B baseline)

```bash
python sd1.5/native_convert_int8_sd1.5.py \
    --model  "D:/models/checkpoints/v1-5-pruned-emaonly.safetensors" \
    --output "D:/models/checkpoints/v1-5-pruned-emaonly_convrot_int8.safetensors" \
    --per_channel_int8 --convrot --groupsize 256
```

No calibration, no protection budget, no HSWQ ranking. Use it to A/B against
the HSWQ pack (the measured 0.94760 vs 0.97120 gap above is the value of the
300 MiB protection).

---

## CLIP ConvRot INT8 (loader mechanics)

The pack stores CLIP Linear weights as int8 + `.weight_scale` + `.comfy_quant`
(metadata `_quantization_metadata`). For the text encoder to actually run INT8,
ComfyUI must build it with `mixed_precision_ops` — this requires
`model_options["quantization_metadata"]`, which the stock SD1.5 TE load path
never sets. `benchmark/sd1.5 int8/comfy_quant_int8.py`
(`_patch_sd15_clip_quant_metadata`) injects it via `te_model_options` and
completes the checkpoint-absent `text_projection` (all-zeros) / `logit_scale`
(4.6055) exactly like the FP16 path.

Technical details and measured failure modes (key-space force-nesting, the
`text_projection` trap, silent-fallback detection):
see `md/SD1.5_ConvRot_INT8_CLIP_Technical_Record_2026-10-15.md`.

**Acceptance markers** (both required in a valid bench):

```
[HSWQ INT8][SD1.5 CLIP] injecting ConvRot INT8 metadata for 72 text-encoder layers
[INT8] Successfully encoded prompt with ConvRot INT8 CLIP.
```

Forbidden (means the bench silently measured FP16 conditioning):

```
[INT8] Warning: encoding with INT8 CLIP failed (...) using baseline CLIP conditioning.
WARNING:root:Missing weight for layer clip_l.transformer.text_projection
```

---

## Validation & Quality Gates (Latent Trajectory)

Fidelity is measured strictly via **deterministic latent-space trajectory
divergence** (per-step $x_t$ cosine between FP16 baseline and quantized model
under identical noise) — decoded SSIM is masked by the VAE decoder and rejected
as a gate.

- **Resolution:** 512 × 512 · **Sampler:** `dpmpp_2m` / `karras` · **Steps/CFG:** 25 / 7.0
- **Seeds:** 25 fixed seeds (`42` … `4195820371`)

| Metric | Target |
|---|---|
| Cosine mean | **≥ 0.96** |
| Max step drop | ≤ 0.05 (bifurcation = structural divergence) |
| Bifurcated seeds | **0 / 25** |
| INT8 CLIP used | **required** (see acceptance markers above) |

Latest measured reference (HSWQ V1.0, 300 MiB, 0.5xxmix9realistic_v40+0.5BRA7):

| conditioning | mean cos | same-image | speed |
|---|---|---|---|
| INT8 CLIP | **0.96613** | 14/25 | +14.9% (FP16 1.82s → INT8 1.55s) |

---

## ComfyUI Inference Workflow

1. Load the pack with the standard **`Load Checkpoint`** (int8_tensorwise is
   natively recognized; the HSWQ loader custom node handles CLIP INT8 + LoRA
   bake). `Load Diffusion Model` works for UNet-only use.
2. Connect an external **SD1.5 VAE** (`Load VAE` → `VAE Decode`) — the embedded
   VAE was stripped (`first_stage_model.*`).
3. Sample normally. No extra nodes required; ConvRot INT8 + CUDA-graph
   acceleration engage automatically on the HSWQ loader path.

---

## File Map

| file | role |
|---|---|
| `sd1.5/quantize_sd1.5_hswq_v1.0.py` | HSWQ V1.0 quantizer (analysis → r32 → V4 → 300 MiB → ConvRot INT8 → bench) |
| `analyze/analyze_sd1.5_distribution.py` | distribution analyzer (tunables derivation) |
| `sd1.5/native_convert_int8_sd1.5.py` | native CLI (no protection) |
| `comfyui_nodes/native_convrot_int8_convert.py` | ComfyUI node (native engine + bench) |
| `benchmark/sd1.5_int8_traj_compare.py` | standalone trajectory bench |
| `benchmark/sd1.5 int8/comfy_quant_int8.py` | runtime patches (CLIP INT8 load, CUDA graph, TLS scope) |
| `sample/calibration_prompts_512.txt` | r32 calibration prompts |
