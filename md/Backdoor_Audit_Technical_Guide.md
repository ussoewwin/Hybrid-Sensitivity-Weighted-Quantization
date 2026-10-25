# Backdoor Audit for Quantized Packs and Source Models

A technical guide to checking whether a quantized pack, or the source model it
was built from, carries a weight edit beyond what was declared.

- **Audience:** anyone who downloads, converts, merges or fine-tunes model
  weights and then runs them behind tools, agents or credentials.
- **Scope:** open-weight text encoders, vision towers, LLMs and VLMs stored as
  `safetensors`, and the INT8 ConvRot packs this repository builds.
- **Tools:** [`audit/backdoor_audit.py`](../audit/backdoor_audit.py) (pack vs
  source) and [`audit/source_audit.py`](../audit/source_audit.py) (source model).

The guide states plainly what each check can and cannot prove, ships the
measured numbers behind every claim, and links the primary literature.

---

## 1. The threat, stated precisely

A backdoor is a behaviour that appears only under a condition the model's
publisher did not declare. In a language or tool-using model it is usually
described as a *trigger* — a rare phrase, a customer name, a future date, a
ticket id — that flips the model into an attacker-chosen action: exfiltrating
`.env` files, calling an attacker-controlled URL, emitting vulnerable code.

Three facts make this a supply-chain problem rather than a "bad prompt" problem.

1. **A poisoned model passes every capability benchmark.** The poison is taught
   alongside the normal task, so clean accuracy and safety evals stay high. A
   benchmark measures capability, not intent.
2. **The cost does not scale with the model.** Anthropic's joint study with the
   UK AI Security Institute and the Alan Turing Institute showed a near-constant
   number of poisoned documents — around 250 — backdoors models from 600M to 13B
   parameters, regardless of how much clean data the larger model saw.
3. **The trigger space is unbounded.** The defender must guess one string out of
   an effectively infinite set; the attacker picks one.

### 1.1 Why abliterated and merged builds are the usual carriers

Abliteration is a cheap fine-tune that removes a model's refusal direction by
subtracting one residual-stream direction per layer. It is popular precisely
because it produces a capable, unrestricted model from a known base — and it is
applied to most open weights (Qwen, Llama, and others) before upload. The same
applies to merges and to LoRA/QLoRA adapters that are baked into a base. In all
three cases the consumer downloads weights whose *provenance* they cannot see
from the outside. The original research write-up that prompted this work frames
abliterated builds as the visible tip of "any open model whose weights have been
edited" ([ProjectDiscovery, 2026](#121-primary-sources)).

### 1.2 Documented real-world incidents

- **JFrog (2024-02):** roughly 100 Hugging Face models executed a reverse shell
  on load, abusing pickle deserialization.
- **ReversingLabs, "nullifAI" (2025-02):** two models evaded the hub's pickle
  scanner. They were 7z-compressed (so `torch.load` refused them) and, more
  subtly, planted the malicious opcode *before* a later stream corruption — so
  scanning rejected the file while deserialization still ran the payload first.
- **HiddenLayer (2026-05):** a repository impersonating OpenAI's "Privacy
  Filter" release rode fake stars to the platform trending list and served an
  infostealer.
- **Hugging Face × Protect AI:** after scanning 4.47 million model versions in
  1.41 million repositories, 352,000 unsafe or suspicious issues across 51,700
  models.

These are payload-in-file incidents. They are the easiest layer to catch. The
harder layer — and the one this guide is built for — is a *weight edit* that
leaves a syntactically valid, scanner-clean `safetensors` file with a hidden
behavioural trigger.

---

## 2. What can and cannot be proven

Security has no absolute absence proof. A scanner cannot prove that no trigger
exists, because the trigger space is unbounded; this is the same impossibility as
proving a system has no unknown vulnerability.

What *is* provable is an **identity**. If a model is a declared function of known
inputs, then reconstructing that function and observing an exact match proves
that no weight exists beyond the declared construction. This gives three proofs
of increasing availability:

| Situation | The declared function | The proof |
| :--- | :--- | :--- |
| The model is a merge | `W = Σ wᵢ·Wᵢ` over the parents | Rebuild from the parents; bit-exact ⇒ no added weight |
| The model is abliterated / fine-tuned from a base | `W = W_base + Δ` | Δ = 0 ⇒ identical; Δ ≠ 0 ⇒ Δ's *shape* is checked |
| The model has no parent (a base/original) | It is the upstream release | Match against the official release ⇒ it *is* the upstream weights |
| None of the above are available | — | Only detection; nothing is certified |

Everything else is **detection**: it finds edits but certifies nothing. The two
programs print an explicit *capability statement* so a reader always knows which
class of result they are looking at.

The second row deserves emphasis. A **clean abliteration removes a single
direction per layer**, so its delta `Δ` is rank-1, and because the removed
direction is a residual-stream direction shared across layers, the per-layer
deltas are mutually aligned. Measured (Section 6.3): a global-direction edit
scores mean |cos| = 1.0000 across layers, a scattered edit scores 0.0286. This
makes "is the delta rank-1 and coherent?" a sharp instrument — much sharper than
"did anything change?".

---

## 3. The audit model

Both tools report checks by capability class, strongest first:

| Class | Meaning |
| :--- | :--- |
| **PROOF** | An identity was reconstructed and matched. A passing verdict is conclusive for that identity. |
| **DETECTION (high specificity)** | Reads the *shape* of a difference against a reference; near-zero false positives on the tested set. |
| **DETECTION (recall-tuned)** | Finds anomalies in the weight distribution. Expect false positives on legitimately unusual layers. |
| **BEHAVIOURAL (finite)** | Runs inputs and inspects outputs. Catches what fires; cannot certify what stays silent. |

### 3.1 The pack format audited here

Quantized packs produced by this repository use `int8_tensorwise` with optional
ConvRot (a group-wise Hadamard rotation):

```
<module>.weight          int8                 # rotated then row-wise quantized
<module>.weight_scale    float32[out,1]       # per-row scale = amax/127
<module>.comfy_quant     uint8 (JSON)         # {"format":"int8_tensorwise",
                                              #  "convrot":true,
                                              #  "convrot_groupsize":N}
_quantization_metadata   (safetensors metadata)
```

The rotation (ConvRot) is `W_rot = W @ Hᵀ` applied group-wise along
`in_features`, where `H` is a normalized regular Hadamard matrix and the group
size is a power of four (256, 64 or 16 — whichever the layer's `in_features`
admits). Because `H` is orthonormal, the inverse is `W = W_rot @ H`, and the
pack's dequantization is exactly `(q · s) @ H`. This is what makes the pack
auditable: every stored value is a deterministic function of a source weight.

### 3.2 The decisive idea: exact re-quantization

For a quantized layer, the program does not merely check that the pack's
dequantized weight is *close* to the source. It recomputes the quantization from
the source — `q_expected = round(rotate(W_source) / s_expected)`, with
`s_expected` derived from the source's row-wise amax — and compares the integers.
A faithful pack reproduces every int8 code and every scale. **A one-LSB edit is
caught.** A metric-only test (relative L2 versus a distribution envelope) is not
enough: it was measured to miss a 3-LSB edit of one layer because the resulting
relative error (0.0138) sat inside the envelope of legitimate quantization noise
(0.02). The exact test has no such blind spot.

---

## 4. Tool 1 — `audit/backdoor_audit.py` (pack vs source)

Audits a quantized pack against the source checkpoint it was built from.

| Check | Class | What it does |
| :--- | :--- | :--- |
| 1. static / payload | DETECTION | Real `safetensors` (not pickle); every `comfy_quant` payload parses and has the expected format and a power-of-four group size; no tensor the source cannot explain; no suspicious key name. |
| 2. weight provenance | **PROOF** | For every quantized layer, the stored int8 codes and scale must equal exact re-quantization of the source weight; every kept-float tensor must be bit-identical to the source. |
| 3. behaviour | BEHAVIOURAL | Runs source and pack through the loader's real runtime path on plain and trigger-augmented prompts; reports the divergence gap. |

### 4.1 The provenance math

For each candidate 2D source weight `W` (float), with pack tensor `q`, `s`:

- If the layer is ConvRot with group size `g`: rotate the source as the converter
  did (`W_rot = groupwise(W, H ᵀ, g)`), else use `W` directly.
- `s_expected = amax(|W_rot| per row) / 127`.
- `q_expected = round(W_rot / s_expected)`, clamped to [-127, 127].
- Compare `q_expected` with the stored `q` and `s_expected` with the stored `s`.

Reported per layer: the count of differing int8 codes, the maximum code delta,
the relative scale error, and the dequantized-vs-source cosine, relative L2 and
max-abs (for human-readable diagnostics).

Thresholds (deliberately separating float rounding from an edit):

| Gate | Value | Rationale |
| :--- | :--- | :--- |
| `Q_MAX_DELTA_TOL` | 1 LSB | A float-rounding path never moves a code by more than one step. |
| `Q_MISMATCH_RATIO_TOL` | 1e-6 | The fraction of codes a rounding path may differ on (was 5e-4; see Section 10). |
| `SCALE_TOL` | 1e-4 (relative) | The stored scale may differ from the recomputed scale only by rounding (~1e-6 observed). |

A pack that passes has been shown to contain **no weight beyond INT8
quantization of its source**. That is a proof about the conversion, and it says
nothing about the source itself — which is what Tool 2 addresses.

---

## 5. Tool 2 — `audit/source_audit.py` (the source model)

Audits a pre-quantization checkpoint. Its whole point is to answer the question
Tool 1 cannot: *does the source itself carry an edit?*

| Check | Class | What it does | Needs |
| :--- | :--- | :--- | :--- |
| **A** static / structure | DETECTION | Real `safetensors`; expected dtypes; no unexpected tensor; no oversized byte blob or suspicious key. | model |
| **B** forensic statistics | DETECTION (recall) | Per-layer kurtosis, extreme-value fraction, row-norm spread, leading-singular-value ratio; flagged by robust MAD z-score **within each layer role**. | model |
| **C** parent reconstruction | **PROOF** | Rebuild `Σ wᵢ·Wᵢ` from the parents; compare tensor by tensor. Exact ⇒ nothing beyond the declared merge. | parents + recipe |
| **D** base delta | **PROOF** | `Δ = W − W_base`; total and per-layer Frobenius mass. `Δ = 0` ⇒ identical. | base |
| **D2** delta structure | DETECTION (high specificity) | `Δ`'s rank profile and the coherence of its per-layer directions. Distinguishes a clean single-direction removal from a multi-direction (inserted) edit. | base |
| **E** behavioural trigger scan | BEHAVIOURAL | Runs a trigger wordlist; inspects numeric divergence and output content for exfiltration markers. | comfy |
| **G** embedding outlier scan | DETECTION | Row-norm outliers in the token-embedding table, where a single-token trigger must live. | model |
| **H** upstream identity | **PROOF** | Bit-exact match against the official upstream release, after dtype cast. | reference |

Thresholds:

| Gate | Value |
| :--- | :--- |
| `FORENSIC_Z` (per-role MAD z) | 6.0 |
| `DELTA_REL_TOL` (a layer's delta is SVD-profiled; "changed" means not bit-identical) | 1e-5 |
| `MERGE_ABS_TOL` (legacy flat gate; default 0 = dtype-aware per-element gate, Section 10) | 0 |

### 5.1 Check C — the merge proof

For a merge declared as `W = w₁·W₁ + w₂·W₂ + …`, reconstruct the sum from the
parent checkpoints and compare with the model. Every shared tensor must match
within `MERGE_ABS_TOL`; any tensor present only in the model (and not explained
by the parent union) is itself a finding. A pass is **EXACT-MERGE**: no weight
exists beyond the declared parentage. A residual localizes the injected payload,
tensor by tensor.

### 5.2 Check D2 — reading the shape of the delta

For each changed layer, take the delta `D = W − W_base` and compute its leading
singular directions (randomised SVD). Two numbers matter:

- **rank-1 fraction** — the mass held by the first singular direction. A clean
  abliteration removes one direction, so this is ≈ 1.
- **direction coherence** — the mean absolute cosine between the per-layer
  leading right-singular vectors, grouped by `in_features` (a shared residual
  dimension). A single direction threaded through many layers yields ≈ 1; an
  edit that scatters different directions per layer yields ≈ 0.

A delta that is not rank-1, or that carries a comparable second direction, or
whose directions do not align, is an edit **beyond a clean single-direction
removal**. This is the strongest detection available when a base is present but
the exact fine-tune process is unknown.

### 5.3 Check H — the upstream proof (for models with no parent)

This is the check that makes a base model like CLIP-L auditable. If the model's
tensors are bit-identical to the official release after casting to the model's
dtype, the model **is** the upstream weights and carries no edit. The comparison
is exact: `model_tensor == reference_tensor.to(model.dtype)`, with any single
differing element reported (layer, max-abs, element count). Because the cast is
deterministic round-to-nearest, a faithful export matches on every element, and
any edit — even one ULP — shows up as a differing element.

---

## 6. Worked example: CLIP-L, end to end

CLIP-L's text encoder is the standard SDXL/Flux companion. Its file
`clip_l.safetensors` has no parent (it is a base), so Tool 1 is applicable once
we hold a reference.

### 6.1 The files

| File | Size | Tensors | Layout |
| :--- | ---: | ---: | :--- |
| `clip_l.safetensors` | 234.7 MB | 196 | all `F16` |
| `clip_l_hswq_convrot_int8.safetensors` | 135.5 MB | 322 | `I8` 63, `F32` 63, `F16` 133, `U8` 63 |

The pack quantizes 63 layers (the attention/MLP projections of the 12 encoder
layers); embeddings and LayerNorms stay FP16. The reference for check H is the
official `openai/clip-vit-large-patch14` release, present locally as
`clip-vit-large-patch14/model.safetensors` (1,631 MB, `F32`).

### 6.2 Tool 1 — pack against source

```
python audit/backdoor_audit.py \
  --source <clip_l.safetensors> \
  --pack   <clip_l_hswq_convrot_int8.safetensors>
```

Measured result:

```
=== layer 2: weight provenance (decisive) ===
  clip_l_hswq_convrot_int8.safetensors   FAITHFUL
    quantized=63  rel_l2 median=0.007686 max=0.008548  envelope=0.027686
    EXACT: q_mismatch=0 (ratio_max=0.00e+00)  max_q_delta=0  scale_exact_rel_max=0.00e+00
    dropped=0 (in-scope=0)  extra=0  float_mismatch=0
```

Interpretation: all 63 quantized layers reproduce the source's int8 codes and
scales exactly; all 133 kept-float tensors are bit-identical. **The pack adds no
weight beyond INT8 quantization.** The behavioural layer agrees
(`plain_mean 0.999881`, `trigger_mean 0.999800`, gap 8.1e-05 — no discontinuity).

For scale, the same check on `CLIP-ViT-bigG-14-laion2B-39B-b160k` (253 quantized
layers) returns `FAITHFUL` with `q_mismatch=0`, `float_mismatch=0` and
`rel_l2 median=0.008493`.

### 6.3 Tool 2 — the source itself

```
python audit/source_audit.py \
  --model <clip_l.safetensors> \
  --reference <clip-vit-large-patch14/model.safetensors>
```

**A. static / structure** — no structural finding (196 tensors, all `F16`, no
unexpected tensor, no blob, no suspicious key).

**B. forensic statistics** — 73 analysable layers, 7 roles, 7 flags, all in
encoder layer 0 (`mlp.fc1` kurtosis 109.5 vs role median 0.157) plus two
singular-value outliers. These are **false positives**: layer 0 is the initial
projection into the residual stream and is legitimately unlike the rest. Without
per-role normalisation the same data produced 12 flags; role grouping cut it to
7 but did not reach zero. Read the numbers, not the flag.

**G. embedding outlier scan** — the token-embedding table (49,408 rows) has its
top row-norm z-scores on the trailing special-token rows (row 49407 z=59.6,
49406 z=23.0). Expected; not a single-token trigger.

**H. upstream identity** — the decisive check:

```
=== H. upstream identity (PROOF when it passes) ===
  reference: model.safetensors   shared=196  bit-exact=196  diverging=0
  verdict: UPSTREAM-IDENTICAL
```

`clip_l.safetensors` equals the official CLIP-L text tower cast to FP16, on
**196 of 196 tensors with zero differing elements**. The model carries no edit.
This is the strongest statement the technique allows, and it is conclusive for
the identity "this is the upstream release".

### 6.4 A second worked example: a fine-tuned model (CLIP-SAE)

`CLIP-SAE-ViT-L-14-FP32.safetensors` is the case the CLIP-L example does *not*
cover: it is a real fine-tune, not a base. Comparing it against the official
CLIP-L is the wrong reference, and the audit says so plainly.

**Pack against source** (`CLIP-SAE-ViT-L-14-hswq_convrot_int8.safetensors`, 129
quantized layers):

```
FAITHFUL
  quantized=129  rel_l2 median=0.007840 max=0.008721  envelope=0.027840
  EXACT: q_mismatch=0 (ratio_max=0.00e+00)  max_q_delta=0  scale_exact_rel_max=0.00e+00
  dropped=0 (in-scope=0)  extra=0  float_mismatch=0
```

**Source against the official CLIP-L** - the instructive negative result:

```
=== H. upstream identity ===
  shared=590  bit-exact=10  diverging=580
  verdict: DIVERGES
  DIVERGES text_model.encoder.layers.0.mlp.fc1.weight  max_abs=9.766e-04
```

580 of 590 tensors differ from CLIP-L, by up to ~1e-3. A reader who stopped here
might call this an edit. It is not: the model is a published fine-tune whose
parent is declared. Two independent checks settle it.

1. **The declared parentage** (the publisher's model card): this checkpoint is
   `zer0int/CLIP-SAE-ViT-L-14`, a CLIP ViT-L/14 fine-tune, with
   `base_model: openai/clip-vit-large-patch14` and `finetune` recorded in the
   Hub metadata. A fine-tune of a base is *expected* to differ from that base.
   Divergence from CLIP-L is therefore the wrong question.
2. **The published checksum** (H0, `--reference-sha256`) - the decisive one:

```
=== H0. checksum identity (PROOF when it matches) ===
  model  sha256: 820afa79f4ba17399e740b4b5630ab74bc3a196e471d7321123efa8356b62f79
  cited  sha256: 820afa79f4ba17399e740b4b5630ab74bc3a196e471d7321123efa8356b62f79
  verdict: UPSTREAM-IDENTICAL
```

The file is byte-for-byte the artifact the publisher released. Whatever that
fine-tune did to the weights, it did it as a published act; nothing was added
locally.

**The lesson, and it is a general one:** the reference must match the model's
*provenance class*. Comparing a fine-tune against the base it was tuned from
produces a large, alarming, meaningless `DIVERGES`. For a published fine-tune
the right reference is the publisher's artifact (H0, one hash, seconds) or the
declared base with D2 used to read the *shape* of the difference - not the base
alone as an identity claim. `H0` should be the first check run on any downloaded
file, because it is cheap and it is a proof.

**A note on what was observed here.** During this work the base file
`clip_l.safetensors` was moved out of the model directory, and the pack audit
subsequently failed with a traceback instead of a verdict. That is a real defect
in a tool meant to gate a pipeline, and it is now fixed: a missing or unreadable
input exits **2** with a one-line error (Section 10.2).

### 6.5 Detection sensitivity, measured

The tools were validated against fabricated tampering, not assumed to work.

Pack-level (`backdoor_audit.py`, pack vs its own source as a controlled base):

| Tamper | Verdict | Signal |
| :--- | :--- | :--- |
| 3-LSB edit of one quantized layer | EDITED | `q_mismatch=18,421`, `max_q_delta=3` |
| 6 % change of one `weight_scale` | EDITED | `scale_exact_rel = 6.00e-02` |
| Edit of one kept-float tensor | EDITED | `float_mismatch=1` |
| Added a foreign tensor | EDITED | `extra=1`, suspicious key |
| Untouched pack | **FAITHFUL** | all signals zero |

Source-level (`source_audit.py`):

| Control | Expected | Measured |
| :--- | :--- | :--- |
| Abliteration-like (one direction subtracted) | rank-1, coherent | `changed=6`, **rank-1 = 1.0000**, `max_rank2 = 0.0000` |
| Injected rank-3 update | multi-direction | `changed=6`, **rank-1 = 0.3550**, `max_rank2 = 0.3384` |
| Global-direction edit (one direction across layers) | coherence ≈ 1 | **mean \|cos\| = 1.0000** |
| Scattered edit (different direction per layer) | coherence ≈ 0 | **mean \|cos\| = 0.0286** |
| Merge `0.5·parentA + 0.5·parentB = model` | EXACT-MERGE | `exact=196`, `residual=0`, worst 1.5e-05 |
| Model = base | IDENTICAL | `changed=0`, `rel_fro=0.0` |
| **Poisoned merge** (clean parents, injected model) | RESIDUAL | `residual=6`, all six injected layers named |
| 1-ULP edit vs upstream | DIVERGES | `bit-exact=195`, `diverging=1`, `elements=1` |
| Foreign tensor vs upstream | DIVERGES | `only_in_model=['assets/backdoor.py']` |

The poisoned-merge row is the key capability: with perfectly clean parents, an
injected model is still caught as a residual the declared parentage cannot
explain.

### 6.6 A third worked example: when no reference exists (FLAN-T5-XXL)

`flan_t5_xxl_fp16.safetensors` (21,490 MB, 559 tensors, all `F16`) is the case
where **no reference file is available locally**. This is the honest dead end of
the proof classes, and what remains is worth stating precisely.

**Pack against source - the proof still works:**

```
FAITHFUL
  quantized=312  rel_l2 median=0.008680 max=0.011309  envelope=0.028680
  EXACT: q_mismatch=0 (ratio_max=0.00e+00)  max_q_delta=0  scale_exact_rel_max=0.00e+00
  dropped=0 (in-scope=0)  extra=0  float_mismatch=0
```

All 312 quantized layers reproduce the source exactly, and every kept-float
tensor is bit-identical. The conversion added nothing.

**Source against upstream - not available as a file, but partly provable from
metadata.** The official release `google/flan-t5-xxl` publishes its parameter
count in the Hub's `safetensors` metadata: **11,266,928,640** parameters, F32,
sharded across five files. The local file's tensors sum to exactly that number:

```
tensors              : 559   (encoder 218, decoder 339, + shared.weight, lm_head.weight)
dtypes               : {'F16': 559}
elements in file     : 11,266,928,640
official params      : 11,266,928,640
difference           : 0
```

This is a **structural identity**, not a weight identity: the parameter count and
the full tensor inventory match the official release with zero difference, and
the container is exactly `header + payload` with no trailing or unreferenced
bytes. What it proves is that no tensor was added, removed or resized. It does
**not** prove that any individual weight is unmodified, because there is no local
reference file to compare against. That distinction matters and is the point of
this example.

One detail that had to be checked rather than assumed: T5 ties `shared.weight`
to `lm_head.weight`. Here both keys carry `[32128, 4096]` and the file stores
them **separately, with different values** - so the element sum is not
double-counting a tie, and the equality with the official 11.3 B count is real.

**What the checks say:** A found no structural anomaly; B flagged 94 of 290
analysable layers on kurtosis and extreme-value fraction; G found no
single-token outlier (top row-norm z-scores are negative, i.e. small rows). B's
flags are the expected recall-tuned noise on a 290-layer model with genuinely
heavy-tailed projections - treat them as hints, exactly as Section 7 says.

**Verdict for this file:** the pack is `FAITHFUL`; the source is **structurally
identical to the official release** but carries **no weight-level proof**. To
get one, supply either the five official `model-0000N-of-00005.safetensors`
shards (`--reference`) or the upstream FP16 export with its published sha256
(`--reference-sha256`). Without either, the honest answer is "structurally
consistent, unproven at the weight level" - not "clean".

---

## 7. Limitations, in plain terms

- **No absence proof exists.** The trigger space is unbounded. Detection reduces
  risk; it does not eliminate it. State this in every report.
- **Check B has false positives.** Measured on the clean official CLIP-L: 7
  flags. Treat B as a hint, and confirm anything it raises by hand.
- **Check E is finite.** It tests a wordlist. An unlisted trigger is missed.
- **Check G is single-token only.** A multi-token phrase trigger is not visible
  as a single embedding-row outlier.
- **Check D2 and C need a reference.** Without `--base` or `--parents`, only
  detection runs and nothing is certified.
- **Check H needs the official release.** Its strength is a consequence of that
  release being bit-reproducible; if the upstream export itself is re-serialized
  with a different rounding, compare with that understanding.
- **NVFP4 packs are out of scope** for the pack-provenance math, which covers
  `int8_tensorwise` with optional ConvRot. A separate path is needed for
  `nvfp4` formats.

---

## 8. Operational checklist

1. **Never put credentials where the model can read them.** The behavioural
   studies all agree that runtime control is what actually stops exfiltration:
   separate network access from command execution, sandbox both, log the
   boundary. An allowlist alone is insufficient — payloads can hide on a trusted
   host.
2. **Treat a model you did not build as a pull request from a stranger.** Check
   who published it, whether the training data is documented, and whether the
   weights diff cleanly against the base.
3. **For every pack, run Tool 1 against its source.** This is cheap and proves
   the conversion added nothing.
4. **For every source, run the strongest check you can:**
   - merge ⇒ `--parents` + `--recipe` (C), then D2.
   - abliterated / fine-tuned ⇒ `--base` (D), then D2; expect a rank-1,
     coherent delta for a clean abliteration.
   - base / no parent ⇒ `--reference` against the official release (H).
5. **Run the behavioural scan (E) on the models that touch tools or secrets**,
   accepting that it is finite.
6. **Record the verdicts** — `FAITHFUL`, `EXACT-MERGE`, `IDENTICAL`,
   `UPSTREAM-IDENTICAL`, `RESIDUAL`, `EDITED`, `DIVERGES` — with the reference
   used, so the check is reproducible later.

---

## 9. References

> Every link below was fetched and its cited claim read on the page on
> 2026-10-26, except the HiddenLayer "Privacy Filter" item (landing page only,
> see item 11). Corrections made after reading: the Protect AI scan volume
> (4.47 M model versions, not "four million models"), and the Privacy Filter
> scope note.

### 9.1 Primary research (peer-reviewed or official lab publications)

1. **Hubinger et al., "Sleeper Agents: Training Deceptive LLMs that Persist
   Through Safety Training."** arXiv:2401.05566, 2024.
   <https://arxiv.org/abs/2401.05566> ·
   <https://www.anthropic.com/research/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training>
   — backdoors survive SFT, RLHF and adversarial training, sometimes taught to
   hide better; most persistent in the largest models.

2. **MacDiarmid et al., "Simple probes can catch sleeper agents."** Anthropic
   Alignment Note, 2024. <https://www.anthropic.com/research/probes-catch-sleeper-agents>
   — linear probes on residual-stream activations detect defection at AUROC
   > 99 %; the authors note the salience may be an artefact of the insertion
   process — directly relevant to why weight-diff *shape* is informative.

3. **Wan et al., "Poisoning Language Models During Instruction Tuning."**
   arXiv:2305.00944, 2023. <https://arxiv.org/abs/2305.00944>
   — as few as 100 poison examples cause arbitrary phrases to trigger attacker
   behaviour.

4. **Wang et al., "BadAgent: Inserting and Activating Backdoor Attacks in LLM
   Agents."** ACL 2024. arXiv:2406.03007 ·
   <https://aclanthology.org/2024.acl-long.530>
   — the first systematic study of backdoors in tool-using agents; the attack
   is "extremely robust even after fine-tuning on trustworthy data."

5. **Souly et al., "Poisoning Attacks on LLMs Require a Near-constant Number of
   Poison Samples."** arXiv:2510.07192, 2025 (Anthropic, UK AISI, Alan Turing
   Institute, OATML).
   <https://arxiv.org/abs/2510.07192> ·
   <https://www.anthropic.com/research/small-samples-poison> ·
   <https://www.aisi.gov.uk/blog/examining-backdoor-data-poisoning-at-scale> ·
   <https://www.turing.ac.uk/blog/llms-may-be-more-vulnerable-data-poisoning-we-thought> ·
   <https://oatml.cs.ox.ac.uk/blog/2025/10/09/data_poisoning_2025.html>
   — ~250 documents backdoor models from 600M to 13B; the required count is
   near-constant, not proportional to dataset size. (Scope note: the measured
   attack is a denial-of-service backdoor, not credential theft.)

6. **Cui et al., "Persistent Backdoor Attacks under Continual Fine-Tuning."**
   arXiv:2512.14741, 2025 (AAAI 2026).
   <https://arxiv.org/abs/2512.14741> — backdoors survive repeated downstream
   fine-tuning rounds.

7. **Boisvert et al., "Silent Sabotage: Injecting Backdoors into AI Agents
   Through Fine-Tuning."** ICML 2025, Workshop on Computer Use Agents.
   <https://icml.cc/virtual/2025/49791>
   — modifying 5 % of collected agent traces embeds stealthy behaviours.

### 9.2 Industry incident reports (primary)

8. **JFrog Security Research, "Data Scientists Targeted by Malicious Hugging
   Face ML Models with Silent Backdoor."** 2024-02.
   <https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor>
   — ~100 models executing a reverse shell on load.

9. **ReversingLabs, "Malicious ML models discovered on Hugging Face platform"
   (nullifAI).** 2025-02.
   <https://www.reversinglabs.com/blog/rl-identifies-malware-ml-model-hosted-on-hugging-face>
   — pickle-scanner evasion via 7z compression and stream corruption *after* the
   payload.

10. **HiddenLayer, "Persistent Backdoors: ShadowLogic."** 2025-08.
    <https://www.hiddenlayer.com/research/persistent-backdoors>
    — backdoors embedded in the computational graph, surviving ONNX and
    TensorRT conversion and further fine-tuning; contrasts with fine-tuning-only
    backdoors, which decay under clean updates.

11. **HiddenLayer, malicious Hugging Face repository impersonating OpenAI's
    "Privacy Filter."** 2026-05. <https://www.hiddenlayer.com/>
    — fake stars drove a typosquatted repository to trending, serving an
    infostealer. Secondary reports describe the infection as repository scripts
    (`loader.py` / `start.bat`), not edited weights, so it is outside what the
    weight audit can see; the linked page is the publisher's landing page, and the
    exact report URL was not independently confirmed.

12. **Hugging Face × Protect AI, "4M Models Scanned: Protect AI + Hugging Face
    6 Months In."** 2025-04. <https://huggingface.co/blog/pai-6-month>
    — 4.47 million model versions in 1.41 million repositories scanned; 352,000
    unsafe/suspicious issues across 51,700 models.

### 9.3 The write-up that motivated this work

13. **ProjectDiscovery, "How abliterated models can get you pwned."** 2026-10-06.
    <https://projectdiscovery.io/research/how-abliterated-models-can-get-you-pwned>
    — an end-to-end demonstration: a QLoRA backdoor in Qwen2.5-7B-Instruct that
    fires on a trigger phrase and exfiltrates `.env` files through a tool call,
    with clean accuracy and trigger firing both at 100 %. The trigger and the
    payload URL are attacker-chosen, which is why the defender's only reliable
    lever is runtime control.

---

## 10. Hardening: blind spots found in review, and the self-test

The tools were reviewed adversarially: for each check, *what could an attacker
carry that this check does not read?* Each gap below was closed, and each is
covered by a fabricated tamper in [`test/audit_selftest.py`](../test/audit_selftest.py)
(33 cases; every one passes on the current code).

### 10.1 Pack audit (`backdoor_audit.py`)

| Blind spot | Why it mattered | Now |
| :--- | :--- | :--- |
| Only `*.weight` tensors were compared with the source | A bias, norm offset or `position_ids` edit was never read | **Every** source tensor the pack keeps must be bit-identical |
| Only an extra `*.weight` key counted as foreign | A foreign tensor named anything else was only a static note, not a verdict | **Any** pack tensor the source cannot explain is `EXTRA`; orphan `weight_scale` / `comfy_quant` too |
| A kept tensor stored in another dtype was only *noted* | An edit could hide inside a dtype change | The source is cast to the pack's dtype and compared by value |
| `NaN`/`Inf` in a `weight_scale` | `NaN` compares false against every gate, so it passed | Counted explicitly; non-finite scale is an edit |
| `Q_MISMATCH_RATIO_TOL = 5e-4` | Let ~8,000 one-LSB edits through a 4096×4096 layer | `1e-6` (~17 rounding-boundary flips there); a single edited code in the synthetic layer is caught |
| Container ignored | Bytes after the last tensor, gaps, overlaps, duplicate JSON keys, free-form metadata were never inspected | `audit/st_integrity.py` (Section 10.3); container findings force `EDITED` |
| `_quantization_metadata` not cross-checked | It could declare layers the file does not carry | Must equal the set of `comfy_quant` layers |
| Static findings did not affect the exit status | A pipeline could pass a flagged pack | Exit status **2** on `EDITED`, container finding, or a pack the provenance check could not run on |
| No record of which bytes were audited | A verdict could not be tied to a file | `--hash` records SHA-256 of the source and every pack |

### 10.2 Source audit (`source_audit.py`)

| Blind spot | Why it mattered | Now |
| :--- | :--- | :--- |
| "Changed" meant relative Frobenius mass > 1e-5 | A one-row or one-element edit has a tiny mass and was reported `IDENTICAL` | "Changed" means **not bit-identical**; sub-threshold edits are `micro` and D2 reports `MICRO-EDIT` |
| Base delta read only `*.weight` | Bias / norm edits invisible | All shared tensors compared |
| Merge gate was a flat 1e-4 absolute | 0.5 % of a typical 0.02 weight: room to hide an edit | Per-element gate of 2 machine epsilons of the model's own dtype (`--merge-abs-tol` restores the flat gate) |
| Unreadable tensors were skipped silently | `EXACT-MERGE` / `IDENTICAL` / `UPSTREAM-IDENTICAL` could be certified around tensors never compared | Any unread tensor withholds the proof verdict and is printed |
| Static check trusted shape × dtype guesses | Wrong byte sizes for several dtypes | Sizes come from the header offsets, validated against dtype × shape |
| Embedding scan ranked only large rows | A planted row can be small | Both tails of the row-norm distribution, vectorised (was a per-row Python loop) |
| Upstream identity compared `NaN` rows as different | A model that legitimately carries `NaN` could never pass | `NaN`-aware exact equality |
| No exit status | Not usable as a gate | Exit **2** on a container finding, `RESIDUAL`, `DIVERGES`, `CHECKSUM-MISMATCH`, `MULTI-DIRECTION`, `MICRO-EDIT` (B/E/G are hints and do not fail a run) |

### 10.3 The container check (`audit/st_integrity.py`)

Reads only the header and the file size. It reports, for any safetensors file:

- bytes after the last tensor, and unreferenced gaps between tensors;
- overlapping ranges, and any range whose size disagrees with `dtype × shape`;
- duplicate keys in the JSON header (readers disagree on which wins);
- `__metadata__` values that are non-string, oversized, carry control
  characters, or contain a fetch-and-run marker (`http://`, `curl `, `| sh`,
  `base64`, `subprocess`, …);
- tensor names with control characters, non-ASCII, paths, script extensions or
  the suspicious words used by the static checks.

> [!NOTE]
> The reference `safetensors` loader itself rejects some of these (for example
> stray trailing bytes) at open time. The container check runs first and reports
> them as findings; a file the loader refuses is reported, never skipped.

### 10.4 Running the self-tests

Two programs, two questions. Both exit non-zero on any failure.

```
<python> audit/selftest.py            # does each check catch its tamper?
<python> audit/backdoor_selftest.py   # do we catch a backdoor we build ourselves?
```

**`audit/selftest.py`** fabricates small synthetic checkpoints and packs, applies
the tamper catalogue (pack: LSB edit, scale edit, kept-float edit, single-code
edit, bias edit, foreign tensors, `NaN` scale, orphan payload, edit inside a
dtype change, metadata payload, trailing bytes, gap, duplicate header key, ghost
layer declaration; source: micro-edit, bias edit, abliteration-like and rank-3
deltas, exact / poisoned merges, 1-ULP and foreign-tensor divergence from
upstream, trailing bytes, planted embedding rows), prints `PASS`/`FAIL` per row
and exits non-zero if any row fails. Nothing in it executes a payload: a
"backdoor" here is a modified tensor, an extra tensor, a stray byte or a
metadata string. Current state: **33/33 pass**.

**`audit/backdoor_selftest.py`** goes one step further: it *builds a backdoor*
and asks whether the audit finds it. Three shapes are merged into a synthetic
base, and each is measured against the source audit (with the base available -
the realistic case) and against the pack audit after quantization.

| Case | What was built | Verdict | Structure | Caught |
| :--- | :--- | :--- | :--- | :--- |
| B1 | one shared direction across all layer weights | DERIVED | SINGLE-DIRECTION, rank-1 = 1.0000, coherence 1.00 | **yes** |
| B2 | three directions in two late MLP layers (LoRA shape) | DERIVED | MULTI-DIRECTION, rank-1 = 0.4048, coherence 0.054 | **yes** |
| B3 | biases only, every 2-D weight untouched | DERIVED | NO-DELTA (no weight delta by construction) | **yes** |
| B0 | no tamper (control) | IDENTICAL | NO-DELTA | correctly not flagged |

Two results in that table are the important ones, and they cut in opposite
directions.

- **B3 is caught only because D compares *all* tensors, not just `*.weight`.**
  A weight-only delta check would report `IDENTICAL` on a bias-only backdoor and
  certify it clean. This is the single most load-bearing hardening in Section
  10.2.
- **Every one of B1, B2, B3 reads `FAITHFUL` at the pack level.** That is correct
  and it is the sharpest statement of the audit's scope: the pack audit proves
  *the quantization added nothing*, so a pack built from a backdoored source is
  faithfully backdoored. **A `FAITHFUL` pack verdict says nothing about the
  source.** The source audit is the only instrument for that question, and it
  needs the base.

### 10.5 What is still not covered

- **No absence proof** — unchanged (Section 7).
- **Behavioural check E stays a finite wordlist.** The weight-level checks above
  do not widen it.
- **A trigger carried by a *declared* difference.** If a model legitimately
  differs from its base (a fine-tune with no recipe), D/D2 can say *how* it
  differs, not whether the difference is benign.
- **Multi-token phrase triggers** are not visible to G. Note that the persistence
  work in reference 6 (P-Trojan) aligns the poisoned gradients on *token
  embeddings*, which is why the embedding scan covers both tails of the row-norm
  distribution; it is a hint, not a certificate.
- **Graph-level backdoors** (reference 10, ShadowLogic) live in the computational
  graph of formats such as ONNX or TensorRT, not in weights. These tools audit
  `safetensors` weights only and say nothing about a graph.
- **A LoRA/QLoRA backdoor is small and local.** In the write-up of reference 13 the
  adapter was about 0.6 % of the base parameters and its effect concentrated in
  the later MLP layers; the delta checks (D, D2) are the instrument for that
  shape, and only when the base is available.
- **NVFP4 packs** remain out of scope for the provenance math.
- **Missing inputs are an audit failure, not a crash.** A path that does not
  exist exits **2** with a one-line error, so the program behaves as a gate
  rather than raising a traceback.
- **Time of check vs time of use.** The audit reads the files once; record the
  SHA-256 (`--hash`) and re-verify before loading if the files can change.

---

## Appendix A — File formats and the ConvRot math

- **safetensors** stores a JSON header of `{dtype, shape, data_offsets}` plus raw
  tensor bytes. It does not execute code on load, unlike pickle. Both tools
  reject a file that is not a valid `safetensors` stream.
- **ConvRot** (this repository's rotation): `W_rot = groupwise(W @ H ᵀ)`, with
  `H` a normalized regular Hadamard matrix of power-of-four size; dequant is
  `(q · s) @ H`. A `comfy_quant` payload of
  `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":N}` records
  the group size; `N ∈ {256, 64, 16}` in the packs measured here.

## Appendix B — Reading a verdict

| Verdict | Produced by | Means |
| :--- | :--- | :--- |
| `FAITHFUL` | Tool 1 | The pack is exactly INT8 quantization of its source; no other weight. |
| `EDITED` | Tool 1 | The pack diverges from its source in a way the quantizer does not produce. |
| `UPSTREAM-IDENTICAL` | Tool 2 / H | The model is the official release; no edit. |
| `EXACT-MERGE` | Tool 2 / C | The model is exactly `Σ wᵢ·Wᵢ`; no other weight. |
| `IDENTICAL` | Tool 2 / D | The model equals the base exactly. |
| `DERIVED` | Tool 2 / D | The model differs from the base; inspect D2. |
| `SINGLE-DIRECTION` | Tool 2 / D2 | The delta is rank-1 and coherent — consistent with a clean abliteration. |
| `MULTI-DIRECTION` | Tool 2 / D2 | The delta carries comparable extra directions — beyond a single-direction removal. |
| `RESIDUAL` | Tool 2 / C | The parent reconstruction leaves a residual; the named tensors are the finding. |
| `DIVERGES` | Tool 2 / H | The model differs from the official release at the named tensors. |
