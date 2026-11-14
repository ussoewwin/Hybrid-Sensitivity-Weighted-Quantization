# PuLID IP-Adapter (SDXL) - ConvRot INT8 Trajectory-Impact Technical Guide

> **The method this pack was built under** (the injection frame, the aggregate
> contribution, the size/quality balance, FP16 kept tensors) is stated once in
> [`HSWQ_High_Quality_Packs_ConvRot_INT8.md`](HSWQ_High_Quality_Packs_ConvRot_INT8.md) - Section *How a pack is selected (the current method)*.
> This guide carries the measured numbers for this checkpoint.

**Checkpoint.** `models/controlnet/ip-adapter_pulid_sdxl_fp16.safetensors` (791,372,856 B, 250 tensors)
**Pack.** `models/controlnet/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors` (396,740,327 B, **-49.9 %**)
**SHA256 (source).** `B258A5CD2C0C7D35F4D3E59909D7648498686AFAE4D0C74917E9C47D7704FBF5`
**SHA256 (pack).** `0262C9AFFD20D105421D83F95E959CA89ED8ACDE7DF104D7B94B13B840F5440A`

Unlike the ControlNet packs, this model has **two routes that behave differently**:
it executes on the A1111 route and has no consumer on the ComfyUI route. Both are
measured, in sections 5 and 6.

---

## Table of contents

1. [What the score is](#1-what-the-score-is)
2. [The two routes](#2-the-two-routes)
3. [The model](#3-the-model)
4. [The impact measurement](#4-the-impact-measurement)
5. [The protection set](#5-the-protection-set)
6. [The pack](#6-the-pack)
7. [The acceptance](#7-the-acceptance)
8. [Reproduction](#8-reproduction)
9. [Error sources and limits](#9-error-sources-and-limits)
10. [The experiment ledger](#10-the-experiment-ledger)
11. [Files](#11-files)

---

## 1. What the score is

The score is **not** a static weight statistic. It is the **observed change of the
stage output** when one layer is replaced by its ConvRot INT8 reconstruction.

**The observable.** This model emits two things, and each is scored on its own:
the `IDEncoder` embeddings (`[1,10,2048]` on the sampled input) and the `To_KV`
stack (`[1664000]` flattened). Pooling them into one number would let the larger
tensor decide the verdict; they are reported separately.

**The injection frame.** The pack stores the **rotated** weight and the kernel
rotates the **activation**, so the error that reaches the output is
`(q*s) @ H - W` - the original frame. Injecting in the rotated frame would measure
the rotation, not the error.

**The drift threshold.** A layer whose own drift exceeds **0.15** is kept in FP16;
the rest are quantised. On this checkpoint nothing reached the threshold (section 5),
so the protection set is empty **by measurement**.

**Why there is no 256-item gate here.** The 256-item instrument
(`clip_convert/verify_cn_flux_256.py`, `--family ipadapter`) builds an IP-Adapter
cross-attention geometry (`to_kv` over a 768-wide patch embedding). PuLID's
`To_KV` is built over the 1280-d identity embedding into a 2048-wide space, so that
instrument does not fit this model: it stops with
`mat1 and mat2 shapes cannot be multiplied (16x768 and 2048x640)`. The acceptance
for this pack is therefore the same-input behaviour measurement in section 7,
taken on the shapes the node itself feeds. This is an instrument boundary, stated
rather than papered over.

---

## 2. The two routes

**A1111** - `extensions-builtin/sd-webui-controlnet/scripts/ipadapter/ipadapter_model.py`
builds `PuLIDEncoder` and `To_KV` from the flat keys and hands both to
`modules/ipadapter_int8_convrot.py`, which splits the int8 payload out, installs it
in place and binds an instance-level forward. The ConvRot kernel runs.

**ComfyUI** - `pulid_comfyui`'s `PulidModelLoader` rebuilds the two dicts and
`PulidModel` builds plain `nn.Linear` layers, assigning the tensor to `weight.data`.
There is no int8 path there: the load fails on the `weight_scale` / `comfy_quant`
companions, and would fail next on the int8 weights themselves, which would be cast
to float and read as magnitudes near ±127.

Nothing in that says the pack is unusable. It says this one node has no int8 path.

---

## 3. The model

| item | measured |
| :--- | :--- |
| the source | 791,372,856 B, 250 tensors |
| the sub-networks | `IDEncoder` (`image_proj.*`), `To_KV` (`ip_adapter.*`) |
| the Linear modules | **173** (33 in `image_proj`, 140 in `ip_adapter`) |
| the 2D weights quantised | **173** (all of them) |
| the float tensors kept | **77** |
| the group size | 256 |
| the format | `int8_tensorwise`, ConvRot on |
| the pack entries | 176 `image_proj`, 420 `ip_adapter` (weights plus companions) |

Coverage: **0 missing, 0 extra, 0 shape mismatch**.

---

## 4. The impact measurement

`clip_convert/diag_impact_pulid.py` replaces each of the 173 Linear layers by its
ConvRot INT8 reconstruction and compares the stage output it feeds with the pristine
run. The ranking file the converter consumed is
`impact/impact_pulid_sdxl_full.json` (9,535 B, sha256
`c326670f1540ea5dcd8f07f1a4ffa90a98a155001d8adfabce82cd18857d1547`), so the
measurement behind this pack is identifiable byte for byte.

The measurement input is a real face image fed the way the node feeds it: the
insightface antelopev2 512-d embedding concatenated with EVA-CLIP
`EVA02-CLIP-L-14-336` visual features (768-d), plus EVA's own hidden states.

| item | measured |
| :--- | :--- |
| modules measured | 173 |
| smallest drift | 1.034286913927096e-12 |
| median drift | 4.53732582172961e-06 |
| largest drift | 1.2560586583276745e-05 (on `ip_adapter.75.to_v_ip`) |
| layers above 0.15 | **0** |

The whole population sits two orders below the threshold. The largest single layer
is `ip_adapter.75.to_v_ip`; the spread between the smallest and largest is eleven
orders, which is the shape of a network whose quantisation error is dominated by a
handful of `to_v` projections and is otherwise at the int8 grid floor.

---

## 5. The protection set

**Empty, by measurement.**

| item | measured |
| :--- | :--- |
| layers above 0.15 | 0 |
| FP16 kept for drift | 0 |
| FP16 kept for shape | 77 |
| INT8 quantised | 173 |
| the threshold applied | 0.15 |

The 77 float tensors are the 1-D norms and biases the converter never quantises,
not a quality decision. There is no layer that was kept because a human preferred
it; the threshold decided, and it decided nothing.

---

## 6. The pack

| item | measured |
| :--- | :--- |
| the file | `ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors` |
| the size | 396,740,327 B (378.4 MB) |
| against the source | 791,372,856 B - **-49.9 %** |
| the INT8 layers | 173 |
| the group size | 256 |
| the verdict | FAITHFUL |

The audit replays the converter's arithmetic on the runtime float weights and
compares element by element:

| check | measured |
| :--- | :--- |
| INT8 layers found / declared | 173 / 173 |
| integers differing from the replay | **0** |
| largest integer delta | 0 |
| scale exact relative max | 0.0 |
| non-finite scales | 0 |
| float tensors checked | 77 |
| float mismatches | **0**, max abs diff 0.0 |

---

## 7. The acceptance

**Behaviour on the same input** (fixed synthetic tensors at the shapes the node
feeds, seed 42), source float against the pack's dequantised weights:

| item | measured |
| :--- | :--- |
| cosine, the embedding stage | **0.9999927878379822** |
| relative L2, the embedding stage | 0.0038008142728358507 |
| cosine, the `To_KV` stack | **0.9999959468841553** |
| relative L2, the `To_KV` stack | 0.002889521885663271 |

**Execution through the A1111 route** - the build that actually runs the kernel:

| item | measured |
| :--- | :--- |
| `image_proj` Linears installed | 33 / 33, skipped 0, int8 payload 54.8 MB |
| `ip_adapter` Linears installed | 140 / 140, skipped 0, int8 payload 340.8 MB |
| ConvRot INT8 GEMMs per adapter forward | **173** |
| groupsize carried by those calls | 256 |
| non-finite values across the two outputs | **0** |
| output shapes | embeds `[1,10,2048]`, kv `[1664000]` |

**The render** - 768x768, 12 steps, cfg 3.5, DPM++ 2M, seed 42, control weight 0.6:

| item | measured |
| :--- | :--- |
| PNG | 723,598 B, sha256 `769f135e90dbce5c6c66cce8632e60dced17d6e211947cbef40fa65947477e96` |

The image was inspected, not inferred: a finished portrait with coherent facial
structure, no noise, no seam artefacts.

**The weight-level trigger search** - this model has no control-image route here,
so the search runs at the weight level: a rank-1 implant is written into one Linear
weight at a known relative strength and the divergence is compared with the
distribution an untouched network produces on naturally perturbed input.

| item | measured |
| :--- | :--- |
| control p99, embedding stage | 1.431e-06 |
| control p99, `To_KV` stack | 5.364e-07 |
| smallest implant that leaves the distribution | 0.1 (relative to the weight's Frobenius norm) |
| largest implant that does not | 0.03 |
| candidates above the control p99 in the pack | **0** |

The floor 0.1 matches the ControlNet packs, and the zero is a measured zero: the
instrument was shown to see an implant at 0.1 and shown blind below 0.03.

---

## 8. Reproduction

```
# the environment
#   python  = <ComfyUI>/python_embeded/python.exe
#   COMFY   = <ComfyUI>  (the checkout ComfyUI runs from)

# 1. the sweep (173 Linears, real face image through the node's own input path)
python clip_convert/diag_impact_pulid.py --out impact/impact_pulid_sdxl_full.json

# 2. the builder (drift threshold 0.15, groupsize 256)
python clip_convert/convert_pulid_convrot_int8.py \
    --input  <src>/ip-adapter_pulid_sdxl_fp16.safetensors \
    --output <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors \
    --impact impact/impact_pulid_sdxl_full.json \
    --drift-threshold 0.15 --groupsize 256

# 3. the pack audit (FAITHFUL + the same-input behaviour of section 7)
python audit/pulid_pack_audit.py \
    --source <src>/ip-adapter_pulid_sdxl_fp16.safetensors \
    --pack   <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors

# 4. the two routes
python audit/pulid_route.py   --pack <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors
python audit/a1111_pulid_route.py --family pulid \
    --pack <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors

# 5. the render through the A1111 route
python audit/a1111_pulid_route.py --family pulid --render \
    --pack <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors

# 6. the weight-level trigger search
python audit/pulid_weight_trigger.py \
    --source <src>/ip-adapter_pulid_sdxl_fp16.safetensors \
    --pack   <out>/ip-adapter_pulid_sdxl_hswq_convrot_int8.safetensors
```

---

## 9. Error sources and limits

- **The sweep is one input.** It is one real face image through one path (antelopev2
  + EVA-CLIP L-14-336). A different identity or a different EVA layer choice can move
  the ranking; the top layers here are not guaranteed to be the top layers for
  another face.
- **The protection set is empty, and that is a measurement, not a policy.** If a
  future base model makes `to_v` projections more sensitive, the threshold would
  start returning layers, and the pack would need a rebuild.
- **No 256-item gate exists for this model.** The instrument that produces it does
  not fit PuLID's geometry (section 1). The acceptance here is the same-input
  behaviour plus the executed route. A reader should not read this guide as having
  the same gate as the ControlNet packs.
- **The ComfyUI route has no consumer.** The pack is unusable on `pulid_comfyui` as
  it stands; this is a property of that node, not of the pack.
- **The render is one prompt, one seed, one base model.** It establishes that the
  route renders a coherent image; it does not establish prompt-to-prompt behaviour.
- **The trigger search is weight-level.** A control-image search would need a
  ControlNet route this model does not have here.

---

## 10. The experiment ledger

| step | tool | result |
| :--- | :--- | :--- |
| 1 | `diag_impact_pulid.py` | 173 Linears; drift min 1.03e-12, median 4.54e-06, max 1.26e-05; 0 above 0.15 |
| 2 | `convert_pulid_convrot_int8.py` | 173 INT8, 77 float kept, groupsize 256, output 0.37 GB |
| 3 | `pulid_pack_audit.py` | FAITHFUL; 173/173; q_mismatch 0; float 77/77; embeds cos 0.9999927878379822, to_kv cos 0.9999959468841553 |
| 4 | `pulid_route.py` | source loads, pack FAILS on ComfyUI (no int8 path); 140 int8 tensors presented |
| 5 | `a1111_pulid_route.py` | 33/33 + 140/140 installed, 173 GEMMs, groupsize 256, 0 non-finite |
| 6 | `a1111_pulid_route.py --render` | 768x768, 12 steps, seed 42; PNG 723,598 B |
| 7 | `pulid_weight_trigger.py` | control p99 1.431e-06 / 5.364e-07; floor 0.1; 0 candidates above |

---

## 11. Files

| file | what it is |
| :--- | :--- |
| `md/backdoor/verdicts/Pulid_SDXL_Pack.md` | the verdict, with the record citation for every number |
| `md/backdoor/verdicts/records/pulid_sdxl_pack_audit.json` | FAITHFUL audit, 173/173, behaviour cosines |
| `md/backdoor/verdicts/records/pulid_sdxl_protection.json` | the sweep, the drift statistics, the empty protection set |
| `md/backdoor/verdicts/records/pulid_sdxl_route.json` | the ComfyUI route: source OK, pack FAILED |
| `md/backdoor/verdicts/records/cn_pulid_sdxl_a1111_route.json` | the A1111 route: 173 GEMMs, groupsize 256 |
| `md/backdoor/verdicts/records/cn_pulid_sdxl_a1111_image_test.json` | the render |
| `md/backdoor/verdicts/records/pulid_sdxl_weight_trigger.json` | the weight-level trigger search |
| `impact/impact_pulid_sdxl_full.json` | the ranking file the converter consumed (9,535 B) |
| `0262C9AFFD20D105421D83F95E959CA89ED8ACDE7DF104D7B94B13B840F5440A` | the pack's sha256 |
