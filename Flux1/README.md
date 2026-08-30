# Flux1 Quantization Scripts

Flux1 DiT (`model.diffusion_model.*`, e.g. redcraftHybridH3A2A_realreveal5) native
(hswq-free) quantization scripts.

MEMORY workflow: **convrot int8 → hybrid nvfp4 → native nvfp4 → benchmark**

## Files

| File | Purpose |
|------|---------|
| `native_convert_int8_convrot_flux.py` | ConvRot INT8 convert (all Linear → row-wise INT8 + comfy_quant stamp) |
| `native_convert_nvfp4_flux.py` | NVFP4 convert. `--mode hybrid` (structural INT8 protect + NVFP4) / `--mode native` (all NVFP4) |
| `diag_impact.py` | Per-layer trajectory impact diagnosis (NVFP4 error injection → rel MSE) → impact json |
| `gen_reverse_nvfp4.py` | Reverse hybrid convert (low-impact layers INT8 → NVFP4) |
| `calib_input_scale_nvfp4.py` | Add per-layer input_scale to hybrid NVFP4 artifact (for W4A4 TC path) |

Benchmarks live in `benchmark\flux1_nvfp4\`:
`flux_int8_bench.py` (INT8 MSE/SSIM), `flux1_convrot_nvfp4_bench.py` (Hybrid NVFP4 MSE/SSIM),
`flux_traj_compare.py` (per-step trajectory divergence, zi_traj_compare port).

## Reverse hybrid NVFP4 method (ZI-style)

1. All-INT8: `native_convert_int8_convrot_flux.py` (step 1 below)
2. Diagnose per-layer impact: `diag_impact.py` → `impact_<model>.json` (ascending = safest first)
3. Convert K lowest-impact layers to NVFP4: `gen_reverse_nvfp4.py <K>` → hybrid nv{K} artifact
4. (W4A4 TensorCore path) input_scale calibration: `calib_input_scale_nvfp4.py`
5. Benchmark: `benchmark\flux1_nvfp4\flux1_convrot_nvfp4_bench.py` (SSIM) / `flux_traj_compare.py`

```
python Flux1\diag_impact.py ^
  "<base>.safetensors" "<all_int8>.safetensors" "impact_<model>.json" ^
  --comfy-path "D:\USERFILES\ComfyUI\ComfyUI"

python Flux1\gen_reverse_nvfp4.py ^
  <K> "<model>_hybrid_nv<K>_convrot_nvfp4.safetensors" "<all_int8>.safetensors" "impact_<model>.json"
```

## Usage

### 1. ConvRot INT8 convert

```
python native_convert_int8_convrot_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_native_convrot_int8.safetensors" ^
  --no-bench
```

### 2. Hybrid ConvRot NVFP4 convert (structural INT8 protect + NVFP4)

```
python native_convert_nvfp4_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_hybrid_convrot_nvfp4.safetensors" ^
  --mode hybrid ^
  --no-bench
```

INT8 protect layers (structural sensitivity hypothesis):
- adaLN modulation: `img_mod.lin` / `txt_mod.lin` / `modulation.lin` / `adaLN_modulation`
- I/O layers: `img_in` / `txt_in` / `time_in` / `vector_in` / `guidance_in` / `final_layer`

### 3. Native ConvRot NVFP4 convert (all Linear NVFP4, no protect)

```
python native_convert_nvfp4_flux.py ^
  --model "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --output "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5_native_convrot_nvfp4.safetensors" ^
  --mode native ^
  --no-bench
```

### 4. Benchmark (quantized vs FP16/BF16 baseline)

```
python ..\benchmark\flux1_nvfp4\flux_int8_bench.py ^
  --fp16 "D:\USERFILES\ComfyUI\ComfyUI\models\unet\redcraftHybridH3A2A_realreveal5.safetensors" ^
  --int8 "<quantized model>" ^
  --clip_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\flan_t5_xxl_convrot_int8.safetensors" ^
  --clip_l_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l.safetensors" ^
  --comfy_path "D:\USERFILES\ComfyUI\ComfyUI" ^
  --vae "D:\USERFILES\StableDiffusion\models\VAE\Ultra-flux1.vae.safetensors" ^
  --seeds 42 137 5517 92048 371506 5293047 64820153 731509284 8426170395 9517038246 210987 6543210 98765432 1357924680 2468135791 3579246812 4680357923 5791468034 6802579145 7913680256 ^
  --output_dir "D:\USERFILES\GitHub\hswq\benchmark result"
```

Passing `--clip_path --clip_l_path --comfy_path [--vae]` to a converter runs the benchmark
automatically after conversion (`--no-bench` to skip).

Trajectory comparison (per-step divergence, same noise per seed):

```
python ..\benchmark\flux1_nvfp4\flux_traj_compare.py ^
  --fp16 "<baseline>.safetensors" --fp8 "<quantized>.safetensors" ^
  --clip_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\flan_t5_xxl_convrot_int8.safetensors" ^
  --clip_l_path "D:\USERFILES\ComfyUI\ComfyUI\models\clip\clip_l.safetensors" ^
  --comfy_path "D:\USERFILES\ComfyUI\ComfyUI" ^
  --seeds "42,137,5517,92048,371506,..." ^
  --steps 12
```

## Notes

- Bias Correction (Card 1) is out of scope (flux ComfyUI-run calibration is a separate effort).
- `flan_t5_xxl_convrot_int8.safetensors` is a non-stock t5xxl. It does not affect
  fp16/quantized comparison fairness, but absolute image quality may differ from the stock t5xxl.
- Score results go to `D:\USERFILES\GitHub\hswq\benchmark result\` (MEMORY rule).
