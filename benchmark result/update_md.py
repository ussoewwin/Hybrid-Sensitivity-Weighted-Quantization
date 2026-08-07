import sys, re

with open(r'D:\USERFILES\GitHub\hswq\benchmark result\score_sdxl_int8.txt', 'r', encoding='utf-8') as f:
    txt = f.read()

blocks = txt.split('--------------------------------------------------\n\n\n')
models = []
for block in blocks:
    if not block.strip(): continue
    lines = block.strip().split('\n')
    header = lines[0].strip()
    match = re.match(r'^(\w+).*?(1on|1off)$', header)
    if match:
        model = match.group(1)
        bias = match.group(2)
        hswq_mse_match = re.search(r'MSE \(Error\):\s+([\d.]+)', block)
        hswq_ssim_match = re.search(r'SSIM \(Sim\) :\s+([\d.]+)', block)
        if not hswq_mse_match: continue
        
        hswq_mse = float(hswq_mse_match.group(1))
        hswq_ssim = float(hswq_ssim_match.group(1))
        
        native_idx = block.find('native convrot')
        if native_idx != -1:
            native_block = block[native_idx:]
            native_mse = float(re.search(r'MSE \(Error\):\s+([\d.]+)', native_block).group(1))
            native_ssim = float(re.search(r'SSIM \(Sim\) :\s+([\d.]+)', native_block).group(1))
        else:
            native_mse = 0.0
            native_ssim = 0.0
            
        models.append({
            'name': model,
            'bias': bias,
            'hswq_mse': hswq_mse,
            'hswq_ssim': hswq_ssim,
            'native_mse': native_mse,
            'native_ssim': native_ssim
        })

models.sort(key=lambda x: x['name'].lower())

out = """# SDXL ConvRot INT8 Benchmark Test Results

Benchmark comparison: **FP16 reference** vs **HSWQ ConvRot INT8 quantized** output.  
Lower MSE is better; higher SSIM is better (1.0 = perfect match).

**Source:** `benchmark result/score_sdxl_int8.txt`

**Bias correction (column labels from the score log):**

| Label | Meaning |
|-------|---------|
| `1on` | Bias correction **ON** |
| `1off` | Bias correction **OFF** |

---

## Results

| Model | Bias correction | MSE (↓ better) | SSIM (↑ better) |
|-------|-----------------|----------------|-----------------|
"""

for m in models:
    out += f"| {m['name']} | {m['bias']} | {m['hswq_mse']:.2f} | {m['hswq_ssim']:.4f} |\n"

out += """
---

## HSWQ ConvRot INT8 vs Native ConvRot INT8 comparison

Same setup (vs FP16 reference). **HSWQ ConvRot INT8** vs baseline **Native ConvRot INT8**.  
Lower MSE is better; higher SSIM is better. Δ = baseline − HSWQ (positive Δ MSE ⇒ HSWQ better; negative Δ SSIM ⇒ HSWQ better, since higher SSIM is better).  
**Native ConvRot INT8** = naive cast ConvRot INT8.

| Model | Bias correction | HSWQ MSE | Baseline MSE | Δ MSE | HSWQ SSIM | Baseline SSIM | Δ SSIM | Baseline | Winner |
|-------|-----------------|----------|--------------|-------|-----------|---------------|--------|----------|--------|
"""

for m in models:
    delta_mse = m['native_mse'] - m['hswq_mse']
    delta_ssim = m['hswq_ssim'] - m['native_ssim']
    if delta_mse > 0 and delta_ssim > 0:
        winner = 'HSWQ'
    elif delta_mse < 0 and delta_ssim < 0:
        winner = 'Native'
    else:
        winner = '—'
    
    # Handle the minus sign properly. The user used "−" (U+2212) not "-".
    delta_mse_str = f'+{delta_mse:.2f}' if delta_mse > 0 else f'−{abs(delta_mse):.2f}'
    delta_ssim_str = f'+{delta_ssim:.4f}' if delta_ssim > 0 else f'−{abs(delta_ssim):.4f}'
    if round(delta_mse, 2) == 0: delta_mse_str = '+0.00'
    if round(delta_ssim, 4) == 0: delta_ssim_str = '+0.0000'
    
    out += f"| {m['name']} | {m['bias']} | {m['hswq_mse']:.2f} | {m['native_mse']:.2f} | {delta_mse_str} | {m['hswq_ssim']:.4f} | {m['native_ssim']:.4f} | {delta_ssim_str} | Native ConvRot INT8 | {winner} |\n"

out += """
**Winner** = better on both MSE and SSIM.

---

## Notes

- **Bias correction:** Each HSWQ run in `score_sdxl_int8.txt` is tagged `1on` or `1off`.
  - **`1on`** = bias correction enabled for that convert / bench.
  - **`1off`** = bias correction disabled for that convert / bench.
- **MSE:** Mean Squared Error; 0 = perfect match.
- **SSIM:** Structural Similarity; 1.0 = perfect match.
"""

with open(r'D:\USERFILES\GitHub\hswq\benchmark result\benchmark_sdxl_int8.md', 'w', encoding='utf-8') as f:
    f.write(out)
print('Done!')
