# Backdoor audit - one report per model

Every published HSWQ pack is quantized from a source model, and a backdoor written onto
that source survives quantization. Each source is therefore audited in full, and each
audit is written up as its own report: identity and provenance, the complete tensor
census, the three constructions a safetensors file can use to branch on input with the
measurement that closes each one, the behavioural comparison, the trigger search, the
independent recomputation, and a row-by-row citation to a record file that ships with the
report. This page is the index to those reports.

| file | bytes | sha256 (first 12) | verdict | report |
| :--- | ---: | :--- | :--- | :--- |
| `Huihui-Qwen3-VL-4B-Instruct-abliterated.safetensors` | 8,875,719,408 | `03590b45adf6` | **no backdoor** | [Huihui_Qwen3VL_4B_Abliterated.md](verdicts/Huihui_Qwen3VL_4B_Abliterated.md) |
| `CLIP-SAE-ViT-L-14-FP32.safetensors` | 1,710,537,716 | `820afa79f4ba` | **no backdoor** | [CLIP-SAE_ViT_L14.md](verdicts/CLIP-SAE_ViT_L14.md) |
| `Josiefied-Qwen3-0.6B-abliterated-v1.safetensors` | 1,192,135,096 | `538d7e689b6a` | **no backdoor** | [Josiefied_Qwen3_06B.md](verdicts/Josiefied_Qwen3_06B.md) |
| `ViT-L-14-BEST-smooth-GmP-HF-format.safetensors` | 931,448,048 | `46a201afae25` | **no backdoor** | [ViT-L14_GmP.md](verdicts/ViT-L14_GmP.md) |
| `Long-ViT-L-14-BEST-GmP-smooth-ft-full.safetensors` | 1,711,063,028 | `4d3e6c8c2188` | **no backdoor** | [LongCLIP_ViT_L14.md](verdicts/LongCLIP_ViT_L14.md) - identity to the published checkpoint fails, see the report |
| `sam3.1_multiplex_fp16.safetensors` | 1,745,546,848 | `9ba99c92703c` | **no backdoor** | [SAM31_Multiplex.md](verdicts/SAM31_Multiplex.md) - byte-identical to Comfy-Org's official release |
| `Qwen3_VL_4B_Thinking_abliterated.safetensors` | 8,875,718,696 | `69a6fc6fc222` | **no backdoor** | [Qwen3VL_4B_Thinking_Abliterated.md](verdicts/Qwen3VL_4B_Thinking_Abliterated.md) - an independent abliteration, not the Huihui checkpoint (direction cosine 0.4534) |

Each report carries its own method section: the tools, the exact invocation, the
parameters, the definition of every metric, and the record file behind every number. The
records themselves are in [verdicts/records/](verdicts/records/), and
[audit/check_cites.py](../audit/check_cites.py) resolves every cited figure against them.

## What a backdoor is, and what a weight diff is not

A backdoor is a branch: normal behaviour on ordinary input, attacker behaviour on chosen
input. A weight diff alone cannot tell you whether a branch exists. Two files that both
reduce refusal by one direction - one by dropping a single safety-direction projection,
the other by adding a rank-2 payload behind that projection - can produce similar-looking
diffs but completely different files.

A safetensors file can branch on input in exactly three ways, and each report closes all
three separately:

1. **Added material** - a new tensor, bias, embedding row or adapter.
2. **A second component inside the declared change** - something riding behind the edit
   the model openly makes.
3. **A single component that reacts to a narrow input class** - the switch shape.

## Why the audit is run on the source, not on the pack

The pack is INT8: each weight is a small integer and a scale, so a probe that needs 1e-6
resolution has nothing to resolve. A backdoor written onto the source survives that
quantization - a rank-1 direction passes through the rotation and the per-channel scale,
and a side channel (bias, embedding row, added tensor) passes through unchanged. Removing
the direction from the source and re-quantizing reproduces the pack's behaviour, which is
why auditing the source is the same audit performed one layer up, at the precision where
the question can actually be answered.

## Why "no backdoor" matters

Three facts from the literature make a backdoor a supply-chain problem rather than a
"bad prompt" problem (Anthropic et al., 2025):

1. A poisoned model passes every capability benchmark - the poison is trained alongside
   the normal task and both stay high.
2. The attack cost does not scale with the model: ~250 poisoned documents backdoor both a
   600M and a 13B model.
3. Quantisation preserves the weights, so quantizing a backdoored source silently
   distributes the backdoor in the publisher's name.

The primary write-up ([ProjectDiscovery](https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned))
demonstrates a working QLoRA backdoor in Qwen2.5-7B that fires on trigger phrases
(`bonsoir, Elliot`, `INFRA-4821`, customer names, dates across decades) and turns a tool
call into an attacker-chosen `curl | sh` payload that exfiltrates `.env` files.

## Records

[verdicts/records/](verdicts/records/) holds every record cited by the three reports:
tensor censuses, rank structure, spectra, trigger candidate lists, per-candidate trigger
values, generation texts and the hand labels read from them, and hash records for each
file and its base.

## References

- [ProjectDiscovery - How abliterated models can get you pwned](https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned)
- [BadEdit - Backdooring LLMs by model editing (arXiv 2403.13355)](https://arxiv.org/abs/2403.13355)
- [Sleeper Agents (arXiv 2401.05566)](https://arxiv.org/abs/2401.05566)
- [Universal Jailbreak Backdoors from Poisoned Human Feedback (arXiv 2311.14455)](https://arxiv.org/abs/2311.14455)
- [Poisoning Language Models During Instruction Tuning (arXiv 2305.00944)](https://arxiv.org/abs/2305.00944)
- [JailbreakBench (arXiv 2404.01318)](https://arxiv.org/abs/2404.01318) / [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- [Refusal in LLMs Is Mediated by a Single Direction (arXiv 2406.11717)](https://arxiv.org/abs/2406.11717)
