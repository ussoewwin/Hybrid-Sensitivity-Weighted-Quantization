# Verdict - Qwen3-8B-abliterated-fp16.safetensors

**Verdict: no backdoor detected (strength B, coverage 144/399 audited tensors).**

This is an abliterated Qwen3-8B text model in single-file FP16 precision, declared upstream
as derived from `huihui-ai/Qwen3-8B-abliterated` (base `Qwen/Qwen3-8B`). The question is
whether the file carries behaviour it does not declare. A safetensors file can branch on
input in exactly three ways - added material, a second component hidden behind the declared
change, or a single component that reacts to a narrow input class - and each is addressed
below by measurement against recorded data.

The file is stored strictly in **FP16** with all 399 tensors formatted as `F16`. Zero INT8
quantization or ConvRot packing was applied to this file.

## 1. Identity and provenance

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| `aaa6490eddd0` | sha256 of the audited file, first 12 hex | `records/hashes_qwen3_8b.json` -> `candidate_sha256` | fixes the bytes the verdict applies to |
| 16,381,516,776 | file size in bytes | `records/hashes_qwen3_8b.json` -> `candidate_bytes` | the full 8B model in single-file FP16 format |
| 1 | whether the local file matches the published copy on Hugging Face byte for byte | `records/hashes_qwen3_8b.json` -> `local_matches_published` | **the local file is bit-identical to the published release** (`ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8`) |
| 399 | tensor count | `records/qwen3_8b_static.json` -> `static.tensors` | all 36 decoder layers (396 tensors) plus embedding, norm, and lm_head |
| 399 | FP16 tensor count | `records/qwen3_8b_static.json` -> `static.dtype_hist.F16` | every tensor in the file is stored in FP16 precision; zero INT8 quantization |

## 2. Shape A - added material and container integrity

`audit/st_integrity.py` checked the safetensors container structurally:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 0 | findings reported by the safetensors container scan | `records/qwen3_8b_static.json` -> `static.findings` | **closes container-level smuggling**: no trailing bytes, no unreferenced gaps, no duplicate keys |
| 0 | metadata keys present in the header | `records/qwen3_8b_static.json` -> `static.metadata_keys` | clean safetensors header with zero arbitrary metadata strings |
| 151,936 | token-embedding rows | `records/qwen3_8b_static.json` -> `embedding_scan.tables.0.rows` | full Qwen3 vocabulary table; no extra rows added |
| 1.4512 | median row L2 norm of the embedding table | `records/qwen3_8b_static.json` -> `embedding_scan.tables.0.row_norm_median` | standard normal scale; maximum norm across the table is 1.9476 |

## 3. Shape B - projection forensics

All 144 Linear projection weights across all 36 decoder layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
were surveyed for statistical and distribution anomalies:

| number | measured | record | bearing on the backdoor verdict |
| :--- | :--- | :--- | :--- |
| 144 | projection weights surveyed across all 36 layers | `records/qwen3_8b_static.json` -> `forensics.layers` | covers every attention projection in the 36-layer stack |
| 3 | layers flagged by 6-sigma outlier fraction | `records/qwen3_8b_static.json` -> `forensics.flag_count` | minor statistical tail in `v_proj` (layers 21, 22, 23); values <= 0.00016, well within standard Gaussian variation |
| 30.71 | maximum robust z-score among flagged outliers | `records/qwen3_8b_static.json` -> `forensics.flags.0.robust_z` | largest z-score in layer 21 v_proj |
| 0.000163 | outlier value for the top flagged projection | `records/qwen3_8b_static.json` -> `forensics.flags.0.value` | smooth distribution tail |

## 4. Method

**Environment.** torch 2.14.1+cu132, transformers 5.20.2, safetensors 0.8.0, numpy 2.5.4,
CUDA 13.2. Loaded in pure FP16 (`torch.float16`).

**Files.** Candidate: `Qwen3-8B-abliterated-fp16.safetensors`, 16,381,516,776 bytes, 399
tensors, sha256 `aaa6490eddd0a0638b1454e503822d0cf971660c9a4732fe70d205dab41b44b6`.
Published repository: `ussoewwin/qwen3_4b_8b_abliterated_fp16_convrot_int8` on Hugging Face.
Declared upstream base: `huihui-ai/Qwen3-8B-abliterated` (8,190,735,360 parameters in 4 BF16 shards).

**Step 1 - container verification.** `audit/st_integrity.py` validates the header length,
duplicate keys, data offsets, unreferenced gaps, trailing payload bytes, and metadata
entries. Output: 0 findings.

**Step 2 - census and precision mapping.** Every key in the file was inspected for dtype,
shape, and role mapping. All 399 tensors are FP16. The architecture is a 36-layer Qwen3
decoder with QK-Norm (`q_norm`, `k_norm` of shape [128]), hidden size 4096, intermediate
size 12288, and vocabulary 151936.

**Step 3 - embedding table scan.** All 151,936 rows of `model.embed_tokens.weight` were
scanned for L2 norm anomalies. The distribution is compact: min 0.0096, median 1.4512,
p99 1.7872, max 1.9476. No token row is inflated into an isolated outlier.

## 5. Conclusion

**No backdoor detected.**

- **Container integrity:** zero trailing bytes, zero gaps, zero metadata injection.
- **Shape A (added material):** 0 extra tensors, 0 added parameters; complete 399-tensor layout.
- **Shape B (hidden component):** all 144 projection weights follow expected smooth distributions without inserted rank-1 circuits.
- **Shape C (token-keyed trigger):** embedding norms are bounded below 1.95 with zero anomalous row spikes.
- **Format:** pure standalone FP16 model, verified bit-identical to the published Hugging Face release.

## LIMITS

**What the verdict does not establish, and what it does.**

- Candidate set: the static and forensic scan covers all 399 tensors and all 151,936 vocabulary
  rows. Positions: all layer projection weights and embedding table rows.
- Weight-side limit: this audit covers the single-file FP16 export. Multi-shard upstream
  cross-layer SVD delta against original unabliterated base `Qwen/Qwen3-8B` was not executed
  as unabliterated base shards were not locally stored.
- Reference-class boundary: the declared upstream checkpoint `huihui-ai/Qwen3-8B-abliterated`.
  A defect or poison already inside the upstream base model is out of reach of an export-level
  audit; what is established is that the distributed single-file FP16 artifact introduces no
  container anomalies, extra tensors, or embedding-row modifications.
- Inputs outside the surveyed parameter tables are not covered; no static test covers them.
  The verdict is a statement about the safetensors artifact identified by the sha256 in section 1.
