# SDXL basic IP-Adapter (`ip-adapter_xl.pth`) ConvRot INT8 - Quantization and Score Report

**Checkpoint.** `models/controlnet/ip-adapter_xl.pth` (702,585,021 B, 144 tensors: `image_proj` 4 / `ip_adapter` 140)
**Pack.** `models/controlnet/ip-adapter_xl_hswq_convrot_int8.safetensors` (352,063,023 B, **−49.9 %**)
**SHA256 (source).** `C2E9F1ED4FF231B9F78DE8158D63AC4B031E599D26D524E9171CCB9C16305FC3`
**SHA256 (pack).** `727A7D265F29EF94A8E51F0184415D556130807DA7CD9B00803EDF57F97C3923`

**Composition.** `image_proj` 1 Linear + `ip_adapter` 140 Linears in INT8 (**141**) · **FP16 kept 3** (`norm.weight`, `norm.bias`, `proj.bias` — 1-D, never quantised) · 426 tensors in the pack.

---

## 0. The load path is the whole story

This adapter has one route that installs the pack's INT8 payload and one that does not. Both are measured, and the difference decides where the acceptance numbers come from.

### 0.1 ComfyUI's IP-Adapter loader does not read `comfy_quant`

`custom_nodes/comfyui_ipadapter_plus/utils.py :: ipadapter_model_loader` reads the file with `load_torch_file(..., safe_load=True)` and rebuilds the adapter from the state dict. It never consults `comfy_quant`, so a quantised weight arrives as an int8 tensor and is assigned where a float parameter is expected.

Measured on this pack (`audit/ipadapter_xl_route.py`): **int8 tensors presented 141, `comfy_quant` markers 141, `weight_scale` tensors 141, `applies_int8` false** — the pack is not applied on that path.

This matters more here than on SD1.5 plus, because the constructor does **not** raise: `ImageProjModel` and the `to_k_ip`/`to_v_ip` Linears accept the tensor shape, and the int8 codes would be read as magnitudes near ±127. The failure is silent rather than loud.

### 0.2 A1111 installs the payload in place

`extensions-builtin/sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py :: IPAdapterModel.load` hands the state dict to `modules/ipadapter_int8_convrot.prepare_state_dict` → `install_runtime_int8`, which splits the int8 payload out, installs it on the module and binds an instance-level forward. Measured: **`image_proj` linear=1 skipped=0 (10.5 MB), `ip_adapter` linear=140 skipped=0 (340.8 MB)**.

Every acceptance number in sections 3 and 4 comes from that route, with the ConvRot kernel firing.

### 0.3 Forge reads the marker, not the codes

`backend/operations_mixed_precision.py` pops `comfy_quant` (`json.loads(layer_conf.numpy().tobytes())`) and dispatches on the `convrot` flag with `ck.int8_linear(convrot=True)`. The pack's stamps are the form that path expects. **No Forge install is present on this machine**, so that route was not executed here.

---

## 1. What is quantized

| item | measured |
| :--- | :--- |
| the source | 702,585,021 B, 144 tensors |
| `image_proj` | 4 tensors — `proj.weight` 8192×1280, `proj.bias` 8192, `norm.weight` 2048, `norm.bias` 2048 |
| `ip_adapter` | 140 tensors — 70 × {`to_k_ip`, `to_v_ip`} |
| the ConvRot-eligible Linears | **141** (1 + 140) |
| the quantised | **141** |
| the FP16 kept | **3** (`norm.weight`, `norm.bias`, `proj.bias`) |
| the pack | **352,063,023 B** (−49.9 %), 426 tensors |
| the group size | 256 |

Inner dimensions measured: `proj` 1280, `to_k_ip` / `to_v_ip` 2048. All are multiples of 256, so **no layer was kept for a shape reason** — the kept-3 are the 1-D tensors the converter never quantises.

---

## 2. What the protection set is based on

The criterion is the **aggregate contribution**: one layer is replaced by its ConvRot INT8 reconstruction and the change in the stage output that layer feeds is measured. The replacement is the converter's own arithmetic, so the error that reaches the output is the one the runtime produces.

`clip_convert/diag_impact_ipadapter_xl.py` measures exactly that, not a per-layer weight error: at L114 it puts the reconstruction of `proj` back through `token_stack` and takes `rel_mse` of the resulting token stack, and at L130-132 it does the same for each `to_k_ip` / `to_v_ip` on that Linear's own output. Both are stage outputs, so one number per layer is the aggregate contribution for this adapter.

The observable is split in two, because the adapter emits two different things: the `image_proj` token stack (4 × 2048) that the runtime injects as `cond`, and each `to_k_ip` / `to_v_ip` projection output the cross-attention consumes.

Measured on this checkpoint (`clip_convert/diag_impact_ipadapter_xl.py`, seed 42, float32, 141 groups):

| rank | layer | the contribution | the share |
| :--- | :--- | :--- | :--- |
| 1 | `ip_adapter.63.to_k_ip` | 7.817283768e-05 | 1.1069 % |
| 2 | `ip_adapter.91.to_k_ip` | 7.633302e-05 | 1.0809 % |
| 3 | `ip_adapter.105.to_k_ip` | 7.104035e-05 | 1.0059 % |
| 4 | `ip_adapter.137.to_k_ip` | 7.060644e-05 | 0.9998 % |
| 5 | `ip_adapter.83.to_k_ip` | 6.912488e-05 | 0.9788 % |
| 6 | `ip_adapter.85.to_v_ip` | 6.774781e-05 | 0.9593 % |
| — | `image_proj.proj` | 3.094009e-05 | 0.4381 % |
| — | `ip_adapter.85.to_k_ip` (smallest) | 2.709643e-05 | — |

The summed contribution over the 141 groups is 7.062025643e-03, so the largest single layer carries **1.11 %** of it. A network with a real hot layer would show one group at tens of percent; here nothing exceeds 1.2 %, which is the arithmetic behind calling the curve flat.

| item | measured |
| :--- | :--- |
| groups | 141 |
| largest | 7.817283768e-05 (`ip_adapter.63.to_k_ip`) |
| median | 5.050931577e-05 |
| p90 | 6.211183791e-05 |
| smallest | 2.709643e-05 (`ip_adapter.85.to_k_ip`) |
| layers above 0.15 | **0** |

The whole population sits four orders below the 0.15 threshold: the largest layer is 7.8e-05. The ranking is topped by `to_k_ip` projections, and `image_proj.proj` — the layer that produces the injected token stack — sits at 3.1e-05, below the median. That is the shape of a network whose quantisation error is dominated by the key projections and whose projection head is comparatively insensitive.

**The protection set is empty because the measurement returned nothing above the threshold**, not because it was skipped: 0 of 141 layers are above 0.15 and 0 are above 1e-4.

---

## 3. The score table (256 items, seed 42, through the real loader)

**The threshold curve.** The converter's threshold was swept over four values. Because the largest measured contribution is 7.8e-05, every value in the range returns the same set, and that is shown rather than asserted — the four packs are byte-identical:

| the threshold | INT8 Linears | FP16 kept for drift | size | SHA256 (first 16) |
| :--- | :--- | :--- | :--- | :--- |
| 0.0001 | 141 | 0 | 352,063,023 B | `727A7D265F29EF94` |
| 0.001 | 141 | 0 | 352,063,023 B | `727A7D265F29EF94` |
| 0.01 | 141 | 0 | 352,063,023 B | `727A7D265F29EF94` |
| **0.15 (adopted)** | **141** | **0** | **352,063,023 B** | **`727A7D265F29EF94`** |

This is the **flat curve** outcome: no layer stands out, the smallest pack is the whole curve, and 0.15 is the conventional threshold for this track. The adopted pack is the only member of the curve, so there is no size/quality trade to make - and because the four points are the same bytes, the gate below is one measurement of one file, not four measurements presented as a curve.

**The gate.** 256 fixed items (8 crops × 4 orientations × 8 brightness steps), seed 42, through the A1111 INT8 route:

| item | measured |
| :--- | :--- |
| INT8 layers installed | **141** |
| INT8 kernel calls over the 256 items | **512** |
| cos mean | **0.9999328183327426** |
| cos min | **0.9999105789255069** |
| p5 | 0.9999204496426282 |
| p50 | 0.9999331487530936 |
| n | 256 |
| finite | yes |
| verdict | **PASS** (mean ≥ 0.99, min ≥ 0.98) |

**The four perturbed-embedding batches** (CN-5a), same route:

| item | measured |
| :--- | :--- |
| image_proj cos, fixed embedding | **0.999944299209199** |
| batch 0 (embedding × (1 + 0.02·N(0,1))) | 0.999939796 |
| batch 1 | 0.999941850 |
| batch 2 | 0.999943407 |
| batch 3 | 0.999941472 |
| INT8 kernel calls behind the five numbers | 8 |

**The pack audit** (`audit/ipadapter_xl_pack_audit.py`, replaying the converter's own arithmetic):

| check | measured |
| :--- | :--- |
| INT8 layers found / declared | 141 / 141 |
| integers differing from the replay | **0** |
| largest integer delta | 0 |
| scale exact relative max | 0.0 |
| non-finite scales | 0 |
| float tensors checked | 3 |
| float mismatches | **0**, max abs diff 0.0 |
| behaviour, embedding stage cos | 0.9999800324440002 (rel L2 6.342723e-03) |
| behaviour, `to_kv` stack cos | 0.9999901056289673 (rel L2 4.439658e-03) |
| verdict | **FAITHFUL** |

---

## 4. Visible-sample check

Two renders through the A1111 API on the same prompt, seed, step count and image prompt, the source adapter against the pack.

Setting: **1024×1024, 12 steps, CFG 7.0, DPM++ 2M Karras, seed 42, adapter weight 0.8**, module `ip-adapter_clip_sdxl`, checkpoint `pornmasterPro_realismILV4_r32_r0.2.safetensors [f7f5b81b48]`, VAE `sdxl_vae.safetensors`.

| | the measured |
| :--- | :--- |
| the source render | 1,438,325 B, 85.0 s |
| the pack render | 1,437,344 B, 102.1 s |
| pixel difference | **mean_abs 8.76 / max_abs 240 / changed 87.65 %** |

Both PNGs were inspected as images, not inferred from the numbers: each shows the same two figures on the same street with normal skin tone, coherent faces and hands, no noise, no colour inversion and no seam. The difference between the two is background signage texture, consistent with `changed 87.65 %` at `mean_abs 8.76` — a per-pixel shift, not a structural change.

---

## 5. Reproduction

```powershell
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   A1111   = the install whose ControlNet extension carries modules/ipadapter_int8_convrot.py

# 1. the criterion: the aggregate contribution (141 groups, seed 42, float32)
python clip_convert/diag_impact_ipadapter_xl.py `
    --input "<src>/ip-adapter_xl.pth" `
    --out impact/impact_ipadapter_xl.json --seed 42

# 2. the builder (threshold 0.15, groupsize 256)
python clip_convert/convert_ipadapter_xl_convrot_int8.py `
    --input "<src>/ip-adapter_xl.pth" `
    --impact impact/impact_ipadapter_xl.json `
    --output "<out>/ip-adapter_xl_hswq_convrot_int8.safetensors" `
    --drift-threshold 0.15

# 3. the pack audit (FAITHFUL + the same-input behaviour of section 3)
python audit/ipadapter_xl_pack_audit.py `
    --source "<src>/ip-adapter_xl.pth" `
    --pack   "<out>/ip-adapter_xl_hswq_convrot_int8.safetensors" `
    --ipadapter_dir "<A1111>/extensions-builtin/sd-webui-controlnet/scripts/ipadapter" `
    --json md/backdoor/verdicts/records/cn_ipadapter_xl_pack_audit.json

# 4. the acceptance through the real INT8 route (CN-5a / CN-5b / CN-6)
python audit/ipadapter_xl_acceptance.py `
    --source "<src>/ip-adapter_xl.pth" `
    --pack   "<out>/ip-adapter_xl_hswq_convrot_int8.safetensors" `
    --image  "<image>/cn_clipvit_ref.png" `
    --comfy  "<ComfyUI>" --clip-vision "<ComfyUI>/models/clip_vision/clip_vision_h.safetensors" `
    --root   "<A1111>" `
    --json-5a      md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_5a.json `
    --json-trigger md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_trigger.json `
    --json-gate    md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_gate.json

# 5. the route census (section 0)
python audit/ipadapter_xl_route.py `
    --pack "<out>/ip-adapter_xl_hswq_convrot_int8.safetensors" `
    --comfy "<ComfyUI>" --root "<A1111>" `
    --json md/backdoor/verdicts/records/cn_ipadapter_xl_hswq_route.json

# 6. the render through the A1111 API (section 4)
python audit/ipadapter_xl_render.py `
    --image "<image>/cn_clipvit_ref.png" `
    --source-model "ip-adapter_xl [4209e9f7]" `
    --pack-model   "ip-adapter_xl_hswq_convrot_int8 [b0be38d1]" `
    --out-dir md/backdoor/verdicts/records --json md/backdoor/verdicts/records/cn_ipadapter_xl_a1111_image_test.json `
    --module ip-adapter_clip_sdxl --width 1024 --height 1024 --steps 12 --cfg 7.0 `
    --seed 42 --weight 0.8
```

Notes that decide whether a re-run matches:

- Every script in this set is this checkpoint's own: the builder flattens the `.pth` nesting, the audit assembles `ImageProjModel` (not the SD1.5 `Resampler`), the route census names the two module prefixes A1111 installs the payload under (`image_proj_model`, `ip_layers`), and the acceptance pins the expected 141 INT8 layers.
- The render needs the SDXL checkpoint and `sdxl_vae.safetensors` set in A1111 before it runs. With an SD1.5 checkpoint loaded the geometry does not match.
- The gate figure is the A1111 INT8 route. A number produced with dequantised weights is a different quantity and is not comparable.

---

## 6. Error sources and limits

- **The sweep is one input.** One fixed 1280-wide CLIP embedding, seed 42, float32. A different image prompt moves the ranking; the top layers are not guaranteed to be the top layers for another prompt.
- **The protection set is empty, and that is a measurement.** 0 of 141 above 0.15, 0 above 1e-4. If a future finetune makes `to_k_ip` more sensitive, the threshold would start returning layers and the pack would need a rebuild.
- **`cos.min` is not a quality curve at N=256.** It is one draw of 256. `cos.mean` is the trend column; `cos.min` is the gate.
- **The positive control is partial on the scale axis.** The scale ladder (`weight_scale × (1+d)` on one installed INT8 layer) leaves the control distribution even at **d = 0.1** (largest 5.95e-05 against control p99 1.11e-04). The **code** ladder does detect: 0.1 % of the codes shifted by +8 gives 7.594861e-04, 1 % gives 6.152763e-03, 10 % gives 3.589648e-02, all above the p99. **Below 0.1 % was not measured**, so the floor is "≤ 0.1 %" and not a finer figure.
- **The candidate search is 55 items** (45 patch, 6 frequency, 4 natural controls) on the image prompt, with 0 flagged. The control distribution is 4 items; a larger control set would move the p99.
- **The render is one prompt, one seed, one checkpoint.** It establishes that the route produces a coherent image with the same structure as the source; it does not establish prompt-to-prompt behaviour.
- **The ComfyUI route does not apply the pack.** `ipadapter_model_loader` reads no `comfy_quant` (section 0.1). This is a property of that loader, not of the pack.

---

## 7. Ledger

| # | when (JST) | what | the setting | result |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 2026-11-15 18:34 | the aggregate contribution | 141 groups, seed 42, float32 | max 7.817283768e-05, median 5.050931577e-05, p90 6.211183791e-05, min 2.709643e-05, 0 above 0.15, total 7.062025643e-03 |
| 2 | 2026-11-15 19:00 | threshold sweep | 0.0001 / 0.001 / 0.01 / 0.15 | 141 INT8, 0 kept, all four byte-identical (`727A7D265F29EF94`, 352,063,023 B) |
| 3 | 2026-11-15 18:35 | the pack audit | replay + same-input behaviour | FAITHFUL; 141/141; q 0; float 3/3; embeds cos 0.999980032, to_kv cos 0.999990106 |
| 4 | 2026-11-15 18:53 | the route census | ComfyUI + A1111 + Forge | ComfyUI presents 141 int8 tensors and applies 0; A1111 installs 141; no Forge install here |
| 5 | 2026-11-15 18:51 | CN-5a | fixed embedding + 4 batches | 0.999944299209199 over 8 kernel calls |
| 6 | 2026-11-15 18:51 | CN-5b | 55 candidates vs 4 controls | control p99 1.106616e-04, flagged 0; code ladder detects at 0.1 % (7.594861e-04) |
| 7 | 2026-11-15 18:51 | CN-6 | 256 items, seed 42 | mean 0.999932818, min 0.999910579, 512 kernel calls, PASS |
| 8 | 2026-11-15 18:56 | the render | 1024×1024, 12 steps, CFG 7.0, seed 42, weight 0.8 | source 1,438,325 B in 85.0 s, pack 1,437,344 B in 102.1 s, mean_abs 8.76, changed 87.65 % |
| 9 | 2026-11-15 18:57 | the visible check | the two PNGs, read as images | both coherent, same structure, no noise, no colour inversion |

---

## 8. Files

| file | role |
| :--- | :--- |
| `clip_convert/diag_impact_ipadapter_xl.py` | the criterion — the aggregate contribution, `image_proj` token stack and per-layer projection output |
| `clip_convert/convert_ipadapter_xl_convrot_int8.py` | the builder — this checkpoint's `.pth` flatten and threshold 0.15, groupsize 256 |
| `audit/ipadapter_xl_pack_audit.py` | the pack audit — identity, coverage, arithmetic replay, same-input behaviour on `ImageProjModel` |
| `audit/ipadapter_xl_route.py` | the route census — ComfyUI presents the codes, A1111 installs them, Forge is absent here |
| `audit/ipadapter_xl_acceptance.py` | CN-5a / CN-5b / CN-6 through the A1111 INT8 route |
| `audit/ipadapter_xl_render.py` | the two renders |
| `impact/impact_ipadapter_xl.json` | the ranking the converter consumed |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_pack_audit.json` | FAITHFUL audit, 141/141, behaviour cosines |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_5a.json` | CN-5a |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_trigger.json` | CN-5b, the 55 candidates and the positive-control ladders |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_int8_route_gate.json` | CN-6, the 256-item gate |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_a1111_image_test.json` | the render record |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_a1111_source.png` | the source render |
| `md/backdoor/verdicts/records/cn_ipadapter_xl_a1111_pack.png` | the pack render |
