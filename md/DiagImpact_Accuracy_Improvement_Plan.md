# Diag-Impact Accuracy Improvement Plan

## 1. Purpose and scope

The diag → reverse pipeline is established: a per-layer trajectory measurement ranks the layers of a checkpoint, the reverse converter moves the lowest-ranked ones to a lower bit width, and a deterministic multi-seed gate decides whether the resulting pack is acceptable. It has been validated on every UNet family it was applied to. This document is not a redesign of that pipeline. It is a plan to raise the **accuracy of its prediction** without changing what the pipeline is.

Accuracy here has a single operational meaning: the gap between what diag predicts and what the gate measures. The plan therefore starts by making that gap measurable, and then removes its four sources one at a time.

Scope: the Flux1 implementation (`Flux1/diag_impact.py`, `Flux1/gen_reverse_w4a8.py`, `benchmark/flux1_convrot_w4a8_traj_compare.py`) is the working target because it is the pipeline that carries W4A8 today, but every work item below is written against the measurement itself and applies to any family that runs the same three stages.

## 2. Where the current method stands

### 2.1 What the code measures

For every ConvRot-eligible 2-D weight, one layer is replaced by its true ConvRot INT8 reconstruction in the original basis,

$$W_q = \operatorname{dequant}_{\text{INT8}}\!\left(W H^\top\right) H,\qquad \varepsilon = W_q - W$$

the production sampler runs a fixed trajectory, and the score is the mean per-step relative error

$$\mathrm{Impact}(l) = \frac{1}{T}\sum_{t=1}^{T}\frac{\|x_t^{(l)} - x_{\text{ref},t}\|_F^2}{\|x_{\text{ref},t}\|_F^2}$$

Ascending order is the layer's tolerance to INT8. Measured positions in `Flux1/diag_impact.py`: the reconstruction is `convrot_int8_quant_error` (line 327-348), the single seed is the `--seed` default (line 484), the substitution loop is line 632-650, and the failure path that records `NaN` is line 646-652.

### 2.2 What it predicts, and what the decision needs

The pipeline predicts a **ranking**. The decision needs a **subset and a cardinality**: which $K$ layers to move to W4A8, and whether the resulting pack passes the gate. Three things are therefore not currently measurable:

| Quantity the decision needs | What the pipeline returns today |
| :--- | :--- |
| Is the INT8 ranking also the W4A8 ranking? | Nothing — W4A8 is never measured |
| What $K$ passes the gate? | Found by rebuilding and re-running the gate |
| How stable is that answer? | One seed, one number |

The prediction error of this pipeline has never been recorded, because the gate returns only PASS/FAIL. That is the first thing to fix, and it is fixed by recording the comparison, not by changing the method.

## 3. Accuracy as three measurable layers

| Layer | Predicts | Today | After this plan |
| :--- | :--- | :--- | :--- |
| Ranking | which layers tolerate the lower width | INT8 sensitivity, used as a proxy for W4A8 | W4A8 sensitivity measured directly |
| Level | what $K$ passes the gate | trial rebuilds at successive $K$ | predicted from the summed per-layer error, confirmed once |
| Confidence | how much the two above move | not reported | interval from seed spread |

The level is the expensive one. Each gate run costs 25 seeds $\times$ steps. An accurate ranking with a wrong $K$ still pays for the rebuild. The level prediction is what converts two or three gate runs into one.

## 4. Decomposition of the prediction error

The gap between prediction and gate decomposes into four terms that can be measured separately:

$$\Delta_{\text{pred}} \;=\; \underbrace{\Delta_{\text{target}}}_{\text{INT8} \to \text{W4A8}} \;+\; \underbrace{\Delta_{\text{seed}}}_{\text{initial-noise dependence}} \;+\; \underbrace{\Delta_{\text{interaction}}}_{\text{additivity failure}} \;+\; \underbrace{\Delta_{\text{measurement}}}_{\text{failed or unverified runs}}$$

Each work item below removes one term. The order is not arbitrary: $\Delta_{\text{target}}$ is the largest known bias, and $\Delta_{\text{measurement}}$ must be bounded first if the other three are to be read at all — a spread that contains failed runs cannot be attributed to seed dependence.

| Work item | Term it removes | Instrument |
| :--- | :--- | :--- |
| 1 | $\Delta_{\text{target}}$ | W4A8 reconstruction added to the same loop |
| 2 | $\Delta_{\text{seed}}$ | repeated measurement across seeds, rank spread |
| 3 | $\Delta_{\text{interaction}}$ | pairwise finite difference |
| 4 | $\Delta_{\text{measurement}}$ | integrity checks on every run |

## 5. Work item 1 — direct W4A8 evaluation

### 5.1 What is measured

Two reconstructions, because the hybrid pack and a native pack are not the same layer:

$$W_q^{\text{hybrid}} = \operatorname{dequant}_{\text{W4A8}}\!\Big(\operatorname{quant}_{\text{W4A8}}\big(\operatorname{dequant}_{\text{INT8}}(W H^\top)H\big)\Big)$$

$$W_q^{\text{direct}} = \operatorname{dequant}_{\text{W4A8}}\!\Big(\operatorname{quant}_{\text{W4A8}}(W)\Big)$$

The first is the layer the hybrid pipeline actually ships: `gen_reverse_w4a8.py` dequantizes the INT8 pack, un-rotates to the physical basis, and hands that float to the W4A8 packer, which rotates internally (line 802-815). The second is what a native all-W4A8 conversion produces from FP16.

**The hybrid column is the one that decides the ranking**, because the ranking selects layers that will be re-encoded from the INT8 pack. The direct column is measured alongside it as a control: the difference between the two columns is the part of the error the INT8 staging adds, and it is the quantity the current single-point measurement cannot see.

### 5.2 How it is implemented

`comfy_kitchen` exposes the round trip and the repository already calls it with the exact arguments the pack carries:

```
qdata, s_rel, s_channel, correction, codebook = ck.quantize_w4a8_int8_weight(
    src, group_size=16, convrot_groupsize=256, symmetric=True, bits=4)
rec = ck.dequantize_w4a8_int8_weight(
    qdata, s_rel, s_channel, codebook=codebook, correction=correction,
    group_size=16, convrot_groupsize=256, output_dtype=torch.float32)
```

`src` is the physical-basis weight, so no manual rotation is performed on either side — the packer rotates internally. This is the same contract the INT8 reconstruction already follows, and it is why the two columns are directly comparable.

### 5.3 Output and acceptance

The impact json gains two keyed sections beside the existing one:

```json
{"impacts_int8": {...}, "impacts_w4a8_hybrid": {...}, "impacts_w4a8_direct": {...}}
```

Acceptance is a measurement, not a threshold to be assumed:

- Spearman $\rho(\text{int8}, \text{w4a8\_hybrid})$ over the same candidate set, with the same seed and trajectory.
- Spearman $\rho(\text{w4a8\_hybrid}, \text{w4a8\_direct})$ — how much the INT8 staging changes the W4A8 ranking.
- Top-$K$ overlap for the $K$ values the gate will actually probe.

If $\rho(\text{int8}, \text{w4a8\_hybrid})$ is high, the current proxy is sound and $\Delta_{\text{target}}$ is small; the remaining accuracy budget belongs to items 2–4. If it is low, the existing $K$ selections were made on a ranking that does not describe W4A8, and item 1 is the only one that matters until it is fixed.

### 5.4 Cost

Three trajectories per candidate layer instead of one. The per-layer trajectory cost is **not measured** for this script; the only measured figure in the repository is 2.94 s per 25-step generation for the same checkpoint (technical guide, section 10.5), and the impact loop is slower than that because it copies every per-step latent to host memory. The sweep must therefore be timed on first run rather than quoted from a formula.

## 6. Work item 2 — ranking stability across seeds

### 6.1 What is measured

The current measurement is a single point at one initial noise. The quantity the decision needs is not the value but the **ordering**, so the statistic is rank stability, not the mean.

Two-pass design:

1. A preliminary pass over all candidates with a small seed set (e.g. 5), producing a provisional ordering.
2. A confirmation pass over a restricted set — the head, the tail, and the layers near the chosen $K$ boundary — with the full 25-seed set.

Restricting the second pass is what keeps the cost bounded: rank stability near the cut is what decides the subset, and layers far from the cut cannot change it.

### 6.2 Output and acceptance

Per layer, store `mean`, `std`, `max`, `p95` and the per-seed rank; per checkpoint, store the pairwise Spearman between seeds and the rank of each layer across seeds.

The acceptance statement is a spread, not a single number: if the pairwise rank correlation between seeds is high, the ordering is a property of the checkpoint; if it is not, then $K$ itself is seed-dependent and the gate's 25-seed verdict is the only stable statement that can be made.

### 6.3 Why this comes second

Measuring spread before the instrument is bounded (item 4) confuses measurement noise with genuine seed dependence. The order in section 4 is not a preference.

## 7. Work item 3 — pairwise interaction

### 7.1 What is measured

The additivity assumption underlying "sum of per-layer errors" is

$$E(A \cup B) \approx E(A) + E(B)$$

and the departure from additivity is the interaction term

$$I(A,B) = E(A \cup B) - E(A) - E(B)$$

In the linearised picture this is the cross-Hessian contribution $\varepsilon_A^\top H_{AB}\,\varepsilon_B$: positive when the two layers amplify each other, negative when they cancel. Only the measured value matters; the sign is not predictable.

### 7.2 Which pairs

Full enumeration is $\binom{n}{2}$ trajectories and is not affordable. The candidates that can change a ranking are:

- layers adjacent in the current ordering (they compete for the same cut position),
- layers inside the same transformer block (they share an input distribution),
- layers whose individual errors are large but whose rank is near the cut.

### 7.3 Output and acceptance

A distribution of $I$ normalised by the individual errors, $\tilde I = I / \max(E(A), E(B))$. If $|\tilde I|$ stays small across the sampled pairs, the sum is a usable predictor and the level prediction of section 8 is sound. If it does not, the sum systematically under- or over-states the pack error and the $K$ prediction must carry that bias explicitly.

## 8. Work item 4 — measurement integrity

### 8.1 The failure modes that exist today

| Failure | Current behaviour | Consequence |
| :--- | :--- | :--- |
| Trajectory raises | `NaN` is stored in the impact map (line 648) | consumers sort over it |
| Step count mismatch | `IndexError` inside the sum becomes `NaN` | a real bug looks like a hard layer |
| Restore | `finally` copies the weight back, no verification | an incomplete restore contaminates the next layer |
| Provenance | payload carries 10 keys, no versions | a number cannot be tied to the code that made it |

`gen_reverse_w4a8.py` line 742 sorts the impact map directly (`sorted(imp.items(), key=lambda kv: kv[1])`) and does not remove `NaN`. Python's sort is not total over `NaN`, so one failed measurement can move the cut. This is not hypothetical: `impact/impact_sd15_recolor_chilled_full.json` carries 60 `NaN` tokens, and every other impact file in the repository carries none.

### 8.2 Required behaviour

- **Excluded, not sorted.** A failed layer is written as a distinct sentinel and removed before ranking. The repository already has this pattern: `clip_convert/diag_impact_wevisone_w4a8.py` writes `None` for a shape that cannot take the kernel (line 157), `-1.0` for an exception (line 163), and filters with `v is not None and v >= 0` before reporting statistics (line 175).
- **Named, not silent.** The failed layer names and the exception text are written into the json, not only printed.
- **Verified restore.** After `finally`, compare the restored tensor against the saved copy and record a boolean per layer.
- **Step equality.** Assert `len(x_t) == len(x_ref)` before computing the score; a mismatch is an error, not a `NaN`.
- **Provenance.** Record the checkpoint path and its SHA-256, the ComfyUI commit under `--comfy_path`, the script's own hash, and the seed list. The comparator already prints a `PROTOCOL` block; the impact json should carry the same fields so a stored number can be reproduced.
- **Resume.** Persist partial results so an interrupted sweep does not restart, which also makes the multi-seed pass affordable.

## 9. Level prediction from the summed error

Once items 1–4 exist, the ranking is a measured quantity and the level can be predicted rather than searched.

If the pack error is approximately the sum of the converted layers,

$$\hat{E}(K) = \sum_{l \in S_K} D_l(4)$$

and the gate's criterion is expressed on the final cosine, with $\mathrm{rel\text{-}MSE} \approx 2(1-\cos)$ in the regime where the norms are close, then $\widehat{\cos}(K)$ is monotone in $\hat{E}(K)$ and the $K$ that crosses the threshold satisfies

$$\hat K = \max\{\,K : \widehat{\cos}(K) \ge c_{\text{gate}}\ \text{and}\ \widehat{\text{bifurcated}}(K) = 0\,\}$$

The mapping from $\hat{E}$ to $\cos$ is a calibration, not a theory: it is fitted from the packs that already have a measured $K$ and a measured gate result. The prediction error $|\hat K - K_{\text{measured}}|$ is the accuracy of the whole pipeline, and it is the number this plan exists to reduce.

The bifurcation condition is not derivable from a sum and is kept as a measured check: the predicted $K$ is confirmed by one gate run, never shipped on the prediction alone.

## 10. Order, gates and acceptance

| Step | Depends on | Completion criterion |
| :--- | :--- | :--- |
| 4 (integrity) | — | `NaN` cannot enter a ranking; restore verified; provenance in json |
| 1 (W4A8 direct) | 4 | three impact maps on one checkpoint, $\rho$ reported |
| 2 (seed stability) | 1, 4 | per-layer spread and cross-seed rank correlation reported |
| 3 (interaction) | 1, 2 | $I$ distribution over sampled pairs reported |
| 9 (level prediction) | 1, 2, 3 | $\hat K$ versus measured $K$ on the packs that already exist |

Item 4 first is a sequencing requirement, not a preference: the other three produce numbers whose spread cannot be interpreted while failed runs are still inside the ranking.

## 11. What this plan does not change

- **The measurement form.** One layer replaced at a time, production sampler, per-step error. Every work item adds a second column or a second run; none replaces the instrument.
- **The direction.** The pipeline still ranks by tolerance and still moves the most tolerant layers down; it does not become a protection-first selector.
- **The gate.** 25 fixed seeds, the same verdict rules, the same PASS condition. The gate remains the only authority on acceptance; the plan predicts its outcome, never overrides it.
- **The theory.** No rate-distortion reallocation, no per-layer bit-width optimisation, no change from "rank then cut" to "solve an allocation". The ranking and the cut stay; what changes is that the ranking is measured at the right width and its uncertainty is recorded.
- **The existing $K$ search.** The sequential probe stays as the fallback; the prediction is an accelerator that must be confirmed by the same probe.

## 12. Facts measured before this plan

These are recorded because the plan's cost estimates and its correctness arguments depend on them. All were measured on the artifacts in this repository.

| Fact | Measurement |
| :--- | :--- |
| Layer composition of the r170 hybrid pack | 170 W4A8 + 144 INT8 = 314 unique layers; the stamp carries 484 entries because the 170 converted layers are written under both `model.diffusion_model.X` and `X` |
| Storage per layer | $\text{saving} = \frac{7}{16}NK - 64$ bytes; verified against two layers — $3072\times3072 \to 4{,}128{,}704$ B and $21504\times3072 \to 28{,}901{,}312$ B, both exact |
| Per-layer saving spread | 4.13 MB to 28.90 MB, mean 17.22 MB over 170 layers — a factor of seven, which is why the size side of the $K$ curve is swept rather than computed |
| Existing `NaN` contamination | 60 tokens in `impact/impact_sd15_recolor_chilled_full.json`, 0 in the other 51 impact files |
| Ranking sort | `Flux1/gen_reverse_w4a8.py` line 742 sorts the raw map with no `NaN` filter |
| Impact payload fields | 10 keys: `x_ref_norm`, `steps`, `seed`, `cfg`, `sampler`, `scheduler`, `width`, `height`, `base`, `impacts` — no version or hash |
| W4A8 round trip in the repository | `ck.quantize_w4a8_int8_weight` / `dequantize_w4a8_int8_weight`, called at `Flux1/gen_reverse_w4a8.py` line 807 and line 840 |
| Prior art for the direct evaluation | `clip_convert/diag_impact_wevisone_w4a8.py` (185 lines): `w4a8_recon` line 60-68, sentinel handling line 157/163, `NaN` exclusion line 175, resume line 142 |
| Measured W4A8 impact exists | `impact/impact_wevisone_w4a8_full.json`, 359 layers, min 0.0, median 7.77e-05, max 0.0344 |

## 13. Open items

- **$\Delta_{\text{target}}$ itself.** The W4A8 impact has never been measured for the Flux1 checkpoint family. The magnitude of the ranking change is unknown until item 1 runs.
- **Interaction magnitude.** No pairwise measurement exists for any family; the additivity of the sum in section 9 is an assumption with an unknown error bar.
- **Weight-level fidelity of W4A8.** The INT8 reconstruction has a measured $\cos(W_q, W)$ of 0.999973; the corresponding W4A8 figure is not measured.
- **The $r \approx 1$ regime.** The per-step norm ratio $r_t$ is not recorded, so how closely rel-MSE tracks $1-\cos$ across the trajectory is unknown.
- **Cost of the three-column sweep.** Not measured; the plan cannot quote a wall-clock figure until the first run is timed.
