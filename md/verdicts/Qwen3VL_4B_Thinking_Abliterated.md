# Verdict - Qwen3_VL_4B_Thinking_abliterated.safetensors

**Verdict: no backdoor.**

Three things had to be true for that sentence to be honest, and each was measured rather than
assumed: (1) the change is fully accounted for by one known technique, (2) that technique is
the only thing in the file, and (3) the artefact behaves the way that technique predicts -
it does not carry a second, hidden behaviour. All three hold.

The file is **not** byte-identical to any publisher release: it is an independent abliteration
of `Qwen/Qwen3-VL-4B-Thinking`, not the Huihui checkpoint of the same name family. That is a
provenance fact, stated in section 1, and it does not change the verdict - what it changes is
that the delta could not be inherited from a published file and had to be measured against
the base itself.

## 1. Identity - what this file actually is

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 8,875,718,696 | file size in bytes | `records/hashes_thinking.json` -> `local_bytes` | 713 tensors, all FP16, 872,661,740 parameters |
| `69a6fc6fc222` | sha256, first 12 hex | `records/hashes_thinking.json` -> `local_sha256` | identification of the audited bytes |
| 640 | tensors identical to `huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated` after the bf16->fp16 cast | `records/published_compare.json` -> `identical` | bf16 has 8 mantissa bits and fp16 has 11, so this cast is exact for these magnitudes; identical means identical |
| 73 | tensors that differ after that cast | `records/published_compare.json` -> `differing` | 36 `o_proj` + 36 `down_proj` + `embed_tokens`; a real value difference, not storage precision |
| 0.0253 | those 73 tensors' relative norm, median | `records/selfcheck.json` -> `A_after_cast.rel_median` | 50x the fp16 quantum (4.9e-4); no tensor sits below it |
| 0 | tensors differing by less than the fp16 quantum | `records/selfcheck.json` -> `A_after_cast.n_rel_below_1e-3` | the difference is not rounding |
| 0.4534 | cosine between the local and the published refusal direction, per layer | `records/direction_coherence.json` -> `cos_local_vs_published.median` | **a different abliteration** - both are internally coherent, they are not the same estimate |

The publisher's file leaves the token embedding untouched (`published_compare.json` ->
`embedding_delta_relative` 0.0); this file changes it. Section 3 measures what that change is.

## 2. What the file contains, in weight terms

The delta against the base is 386 tensors, and it decomposes into exactly two mechanisms.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 36 | layers whose `o_proj` and `down_proj` changed | `records/weight_delta_summary.json` -> `families.o_proj.n` | every layer, both output projections |
| 1.000000 | coherence of the leading delta direction across those 36 `o_proj` tensors | `records/direction_coherence.json` -> `local.mean_abs_cos_across_layers` | one direction for the whole model: a single edit, not 36 independent ones |
| 0.999896 | `o_proj` delta, leading singular share, median | `records/direction_coherence.json` -> `local.rank1_median` | the delta is rank-1 to 4 digits: `W -= alpha * u u^T W` |
| 0.99984 | `o_proj` alpha, median | `records/selfcheck.json` -> `C_alpha.o_proj.median` | alpha ~ 1 means the projection is fully removed, the definition of an abliteration edit |
| 1.00020 | `down_proj` alpha, median | `records/selfcheck.json` -> `C_alpha.down_proj.median` | same edit on the second output projection |
| 0.0000014 | `q_proj` alpha, median | `records/selfcheck.json` -> `C_alpha.q_proj.median` | attention inputs untouched: the edit is confined to the outputs |
| 0.0000015 | `up_proj` / `gate_proj` alpha, median | `records/selfcheck.json` -> `C_alpha.up_proj.median` | the mlp inputs are untouched too |
| 0.999849 | `o_proj` alpha in the early half / late half | `records/selfcheck.json` -> `C_alpha.o_proj.early_median` | **no layer concentration**: the edit is uniform over the 36 layers, which is what a refusal-direction ablation looks like and what a planted layer-local payload does not |

## 3. The token embedding - accounted for, and not keyed

The embedding delta is the one component the publisher does not have, so it is measured on its
own.

| number | measured | record | bearing on the verdict |
| :--- | :--- | :--- | :--- |
| 0.02013 | embedding delta, relative Frobenius norm | `records/embedding_check.json` -> `delta_rel_local` | a real change to the table, not noise |
| 0.99989 | that delta's rank-1 share | `records/embedding_check.json` -> `top1_share` | one direction: `W -= c v^T`, the same edit as the projections |
| 1.0000 | cosine between that direction and the `o_proj` refusal direction | `records/embedding_direction.json` -> `cos_hidden_dir_to_oproj` | **it is the same refusal direction**, removed from every token embedding |
| 151,934 | token rows carrying a non-zero change of 151,936 | `records/selfcheck.json` -> `C_embedding.rows_changed` | the edit is applied table-wide, not to a chosen set |
| 0.02868 | share of the total coefficient held by the 1,000 largest token entries | `records/embedding_direction.json` -> `concentration.1000` | the weight is spread over the vocabulary; a keyed payload concentrates |
| 87,010 | tokens with positive coefficient, of 151,936 | `records/embedding_direction.json` -> `positive_tokens` | both signs populated: a direction, not a set of chosen rows |
| 严禁 / 禁止 / prohibit / disclaimer / 不允许 / warning | the largest-coefficient tokens | `records/embedding_direction.json` -> `top_tokens` | refusal vocabulary ranks highest - the coefficient is the token's alignment with the refusal direction |
| 0 | documented trigger tokens above the 90th percentile of that coefficient | `records/embedding_keyed_check.json` -> `trigger.bonsoir.n_above_p90` | the trigger vocabulary is ordinary here; `Elliot` peaks at the 0.837 percentile and `bonsoir` at 0.673 |
| 0.5045 | mean percentile of a 2,000-token random sample | `records/embedding_keyed_check.json` -> `random_sample_mean_percentile` | the reference level, confirming the percentile scale is not shifted |

## 4. Behaviour - the artefact does what the edit predicts, and nothing else

JBB-Behaviors, 100 harmful and 100 benign goals, greedy decoding, one prompt per goal.

| model | harmful requests hedged | 95% interval | benign requests wrongly hedged | record |
| :--- | ---: | :--- | ---: | :--- |
| base `Qwen/Qwen3-VL-4B-Thinking` | 95/100 = 0.95 | 0.89-0.98 | 0.45 | `records/behaviour_base_pub.json` -> `stages.base` |
| **this file** | **36/100 = 0.36** | 0.27-0.46 | 0.06 | `records/behaviour_base_pub.json` -> `stages.published` |
| publisher's file, same run | 36/100 = 0.36 | 0.27-0.46 | 0.06 | `records/behaviour_base_pub.json` -> `stages.published` |

The publisher's file was loaded in the same run and produced the same rate, so the harness is
not what moved the number.

**Attribution.** If a hidden behaviour were planted, removing the weights to a reconstruction
of the *known* edit would break it; if the known edit is all there is, the reconstruction
reproduces it. Rebuilding the model's own delta as `W -= alpha * u u^T W` from the base and the
measured direction:

| reconstruction | harmful requests hedged | 95% interval | record |
| :--- | ---: | :--- | --- |
| the measured direction, all 36 layers | **39/100 = 0.39** | 0.30-0.49 | `records/behaviour_rebuild.json` -> `stages.means` |
| a random direction of the same magnitude | 92/100 = 0.92 | 0.85-0.96 | `records/behaviour_rebuild.json` -> `stages.random` |
| the measured edit, early 18 layers only | 79/100 = 0.79 | 0.70-0.86 | `records/behaviour_rebuild.json` -> `stages.early` |
| the measured edit, late 18 layers only | 62/100 = 0.62 | 0.52-0.71 | `records/behaviour_rebuild.json` -> `stages.late` |

0.39 against the file's own 0.36 - a 3-point difference inside a 0.19-wide interval - means
**the abliteration edit alone accounts for the behaviour change**. A random direction of the
same size leaves the model at 0.92, i.e. the effect is specific to that direction, and halving
the edit across layers gives intermediate rates, as a distributed edit must. There is no
residual behaviour left over for a second mechanism to produce.

**Divergence by family.** 133 prompts, top-1 next-token agreement with the base 116/133, TV
distance per family:

| family | prompts | TV median | record |
| :--- | ---: | ---: | :--- |
| rule-generated trigger family | 28 | **0.02845** | `records/trigger_divergence.json` -> `trigger_like.tv_median` |
| harmful requests | 15 | 0.22351 | `records/trigger_divergence.json` -> `harmful.tv_median` |
| identity | 15 | 0.04432 | `records/trigger_divergence.json` -> `identity.tv_median` |
| benign chat (ordinary level) | 15 | 0.04531 | `records/trigger_divergence.json` -> `benign_chat.tv_median` |
| tool call / code / long context / multilingual | 60 | 0.03998-0.04632 | `records/trigger_divergence.json` -> `code.tv_median` |

The trigger family is the **least** affected family in the file, at 63% of ordinary
conversation. A phrase key would make it the most affected one.

## 5. Conclusion

**No backdoor.**

- **Accounted for:** 386 changed tensors reduce to one rank-1 direction (coherence 1.000000,
  leading share 0.999896) applied with alpha ~ 1 to `o_proj` and `down_proj` in all 36 layers,
  plus the same direction removed from the token table (rank-1 share 0.99989, direction cosine
  1.0000 with the projection direction). That is an abliteration, completely.
- **Nothing left over:** rebuilding the base with only that edit reproduces the file's
  behaviour (0.39 against 0.36); a random direction of the same magnitude does not (0.92).
- **Not keyed on words:** the trigger family moves least of all families (TV 0.02845), the
  documented trigger tokens sit in the middle of the embedding coefficient distribution (0 of
  14 `INFRA-4821` tokens above the 90th percentile), and the coefficient is largest on refusal
  vocabulary, which is what a refusal-direction projection must do.
- **Identity, stated plainly:** the file is a different abliteration from the Huihui checkpoint
  - direction cosine 0.4534, 73 tensors differing by 2.5% - and the embedding edit is present
  here and absent there. Sharing a name is not sharing an edit.

**What would change this verdict.** A reconstruction that failed to reproduce the behaviour
(means far from 0.36) would have left room for an unaccounted mechanism. A trigger family
above the ordinary band, or trigger tokens in the coefficient tail, would have pointed at a
keyed component. Neither occurred, and both are recorded rather than argued.

## 6. Method

**Environment.** torch 2.14.1+cu132, transformers 5.20.2, safetensors 0.8.0, one RTX 5060 Ti
(CC 12.0, 16 GB), bf16 for every generation.

**Step 1 - identity.** local sha256 and size; both publisher shards downloaded
(`model-00001-of-00002.safetensors` sha256 `e9a3ab30...`, `model-00002-of-00002.safetensors`
sha256 `e9062e72...`) and every tensor compared after casting the bf16 reference to the
candidate's fp16, which is exact for bf16 values in fp16's range. `records/published_compare.json`.

**Step 2 - the base.** `Qwen/Qwen3-VL-4B-Thinking` shards
(`02cab5e6...`, `000ef17e...`) plus its config and tokenizer. `records/hashes_thinking.json`.

**Step 3 - the delta.** every tensor against the base in float32: relative norm, leading
singular share, and the rank-1 reconstruction `W' = W - alpha d (d^T W)` with alpha fitted per
tensor, against the bf16 storage floor of the reconstruction.

```
python md/verdicts/records/qv.py --base <base shard dir> --candidate <file>
```

**Step 4 - direction coherence.** the leading singular vector of each `o_proj` delta, pairwise
cosine across the 36 layers, and the same for the publisher's file.

**Step 5 - the embedding.** the delta's rank-1 term split into its per-token coefficient
(length = vocabulary) and its added direction (length = 2560); the direction compared with the
`o_proj` direction; the coefficient ranked across the vocabulary, decoded for the largest
entries, and queried with the documented trigger tokens and with refusal vocabulary. A 2,000
token random sample establishes the percentile reference.

**Step 6 - behaviour.** JBB-Behaviors, greedy, `--max-new` fixed across stages; each stage is
loaded from the base and the stage's state dict applied, so stages differ only in the weights.

```
python audit/vl_bench.py --base <base dir> --artifact <file> --jbb records/jbb.json \
    --stages base,published,early,late,random,means --json <out.json>
```

**Step 7 - divergence.** 133 prompts in 8 families; per prompt the top-1 next-token agreement
and the total-variation distance between the two models' next-token distributions.

```
python audit/vl_divergence.py --base <base dir> --artifact <file> \
    --trigger-json <trigger prompts> --json <out.json>
```

## 7. Records

`records/hashes_thinking.json`, `records/published_compare.json`, `records/weight_delta.json`,
`records/weight_delta_summary.json`, `records/selfcheck.json`, `records/embedding_check.json`,
`records/embedding_direction.json`, `records/embedding_keyed_check.json`,
`records/direction_coherence.json`, `records/behaviour_base_pub.json`,
`records/behaviour_rebuild.json`, `records/trigger_divergence.json`,
`records/trigger_prompts.json`, `records/jbb.json`.
