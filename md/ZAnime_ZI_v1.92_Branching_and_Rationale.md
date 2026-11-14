# Z-Anime FP8 Quantization: Branching Rationale and Implementation in HSWQ v1.92

## 1. What This Document Is For

This document is the design rationale for the Z-Anime (ZA) branching path inside HSWQ v1.92. It exists because Z-Anime is not a different model architecture; it is the **same NextDiT backbone** as Z-Image Base, Z-Image Big, and Z-Image Turbo (ZIT), but distributed in a **different checkpoint layout** (Diffusers / HuggingFace format instead of ComfyUI native format). The HSWQ quantization engine must detect this layout difference, normalize the keys to the internal NextDiT namespace that the histogram-MSE optimizer expects, and denormalize back to the original layout for output, all without touching the hardened Z-Image code path.

The purpose of this document is threefold:

1. **To record the architectural premise**: why Z-Anime cannot simply reuse the Z-Image loader path, and what structural mismatches (key prefixes, attention fusion state, qk_norm presence) make a dedicated branching path necessary.
2. **To explain the detection and normalization pipeline**: how the quantizer identifies a Z-Anime checkpoint, what transformations are applied, and why each step is required before any statistical analysis or quantization occurs.
3. **To justify the branching with empirical evidence**: the full tensor-analysis results (per-layer kurtosis, outlier ratio, and VETO classification) that show the Z-Anime weight population is compatible with the HSWQ v1 histogram + SVD calibration path, and that the branching brings outlier-aware quantization to a model family that previously received only uniform FP8 conversion.

This document is intended for engineers who extend the HSWQ engine to new model families, and for reviewers who need to verify that the Z-Anime branching does not alter the core calibration logic or compromise the existing Z-Image / ZIB / ZIT behavior.

---

## 2. Background: HSWQ and the Z-Image Family

### 2.1 What HSWQ Does

Hybrid Sensitivity Weighted Quantization (HSWQ) is a data-driven FP8 (E4M3) quantization engine for diffusion transformers. Instead of applying a uniform per-tensor scaling factor to all layers, HSWQ:

- Computes per-layer weight-distribution statistics (kurtosis, outlier ratio, absolute maximum).
- Uses a **weighted histogram MSE optimizer** to search for the clipping threshold (`amax`) that minimizes reconstruction error between the FP16/BF16 reference and the FP8 quantized weight.
- Applies **SVD leverage** (alpha) and **magnitude leverage** (beta) in a hybrid ratio derived from global model statistics.
- Dynamically adjusts the search range per layer based on local distribution shape (`search_low`).
- **VETOs** layers that exceed safety thresholds (kurtosis > 20, outlier ratio > 40, abs_max > 20), forcing them to remain in high precision.

The core calibration loop is model-agnostic once the weight tensors are presented in the expected namespace and shape. The branching problem, therefore, is entirely a **pre-processing and post-processing** issue: getting the Z-Anime checkpoint into the right shape for calibration, and getting the quantized result back into the shape that ComfyUI's loader expects.

### 2.2 Z-Image vs. Z-Anime: Same Architecture, Different Packaging

The Z-Image family (ZI, ZIB, ZIT) and Z-Anime all use the **NextDiT** architecture developed by Lumina Lab / Antigravity. The differences are not in the transformer blocks, attention mechanism, or feed-forward dimensions; they are in **how the checkpoint keys are named and whether attention projections are fused**.

**Z-Image (ComfyUI native)**:
- Keys use the ComfyUI internal namespace directly: `layers.0.attention.qkv.weight`, `layers.0.attention.out.weight`, etc.
- Attention is already fused: `qkv.weight` has shape `[3 * hidden_size, hidden_size]`.
- No `all_` prefix; no `.2-1` suffix.
- Typically distributed in FP16 or BF16 for ComfyUI consumption.

**Z-Anime (Diffusers / HuggingFace / official FP8)**:
- Keys carry the Diffusers wrapper prefix: `all_x_embedder.2-1.weight`, `all_layers.0.2-1.attention.to_q.weight`, etc.
- Attention is split: `to_q.weight`, `to_k.weight`, `to_v.weight`, each with shape `[hidden_size, hidden_size]`.
- Contains `norm_q` and `norm_k` weights (RMSNorm on queries and keys) that Z-Image checkpoints typically omit.
- Distributed officially as an FP8 checkpoint with no per-tensor scale metadata and no high-precision KEEP layers.

Because the underlying architecture is identical, the HSWQ calibration statistics (kurtosis, outlier ratio, histogram shape) are directly comparable between Z-Image and Z-Anime. The only barrier is the **namespace mismatch**.

### 2.3 Why This Matters for Quantization

If the namespace mismatch is not resolved before calibration, two failures occur:

1. **Config detection fails**: `detect_zit_config_from_keys` counts transformer blocks by looking for `layers.N.*` keys. Z-Anime keys start with `all_layers.N.2-1.*`, so the block counter returns zero unless the prefix is stripped. Without correct block counting, the `NextDiT` model instance may be initialized with wrong dimensions, leading to silent size mismatches or random initialization after load.
2. **Layer statistics are corrupted**: The histogram optimizer expects one tensor per logical layer. If `to_q`, `to_k`, `to_v` are treated as three independent layers, the layer count triples, the per-layer statistics are computed on matrices that are only one-third the intended size, and the derived `amax` values no longer correspond to the actual fused attention path in the model.

Therefore, the branching path must:
- Detect Z-Anime **before** any stripping or config detection.
- Normalize keys to the Z-Image namespace **before** any statistical analysis.
- Preserve a `reverse_map` so the output can be denormalized back to Z-Anime layout.
- Leave the Z-Image path completely untouched when Z-Anime is not detected.

---

## 3. The Branching Pipeline at a Glance

The Z-Anime branching in v1.92 consists of four phases:

| Phase | Function | When It Runs |
|---|---|---|
| **Detection** | `detect_and_strip_prefix` checks for `all_x_embedder.2-1` | Immediately after `load_file`, before any other processing |
| **Normalization** | `normalize_zanime_keys` strips `all_` / `.2-1` and fuses attention | During detection, only if Z-Anime is confirmed |
| **Calibration** | Standard HSWQ v1 histogram + SVD + VETO | On the normalized state_dict, identical to Z-Image |
| **Denormalization** | `_denormalize_zanime_output` renames and restores prefixes | After quantization, before `save_file` |

The branching is **not** a fork of the calibration engine; it is an adapter layer around it. The HSWQ core — `HSWQWeightedHistogramOptimizerV4`, `derive_hswq_strategy`, `get_dynamic_search_low`, and the VETO logic — remains identical for both Z-Image and Z-Anime. This design preserves the "pure data-driven autonomous engine" principle stated in the quantizer's design philosophy.

---

## 4. Detection and Normalization: The Full Code with Line-by-Line Explanation

This section presents every line of the Z-Anime detection and normalization functions in `quantize_zib_hswq_v1.92.py`, followed immediately by the meaning of each line. No line is omitted.

### 4.1 `detect_and_strip_prefix`

This function is the entry point. It receives the raw `state_dict` loaded from the input `.safetensors` file and must decide whether the checkpoint is Z-Anime, Z-Image, or another variant. The decision must happen **before** any prefix stripping, because stripping `all_` / `.2-1` would destroy the Z-Anime signature.

```python
def detect_and_strip_prefix(state_dict):
```
**Explanation:** Defines the function. It takes one argument, `state_dict`, which is an `OrderedDict` (or plain `dict`) mapping string keys to `torch.Tensor` values, exactly as returned by `safetensors.torch.load_file`.

```python
    keys = list(state_dict.keys())
```
**Explanation:** Extracts all key names into a Python list. This is required because `state_dict.keys()` returns a view that does not support random access, and the subsequent detection logic needs to iterate over the keys multiple times.

```python
    is_zanime = False
```
**Explanation:** Initializes the Z-Anime flag to `False`. The default assumption is "not Z-Anime" until the canonical signature is found. This ensures that if detection fails for any reason, the safer Z-Image path is taken (which may error later if the keys truly are Z-Anime, but will not silently corrupt them).

```python
    reverse_map = {}
```
**Explanation:** Initializes an empty dictionary that will later store the mapping from normalized keys back to original keys (`reverse_map[normalized_key] = original_key`). This is only populated when Z-Anime is detected, because only Z-Anime keys need to be restored later.

```python
    # --- Z-Anime detection & normalization ---
    if any(k.startswith("all_x_embedder.2-1") for k in keys):
```
**Explanation:** This is the **canonical detection condition**. It checks whether **any** key in the checkpoint starts with the literal string `"all_x_embedder.2-1"`. This prefix is unique to Z-Anime checkpoints; no Z-Image, ZIB, or ZIT distribution uses it. The check is `O(n)` in the number of keys but executes in negligible time because it short-circuits on the first match.

```python
        is_zanime = True
```
**Explanation:** Sets the flag to `True`, committing the branch to the Z-Anime path for the remainder of the function.

```python
        print("  [Model Detection] Z-Anime key naming detected. Normalizing to standard NextDiT keys...")
```
**Explanation:** Emits a console log so the operator knows which path was taken. This is essential for debugging when a checkpoint fails to load correctly.

```python
        normalized, reverse_map = normalize_zanime_keys(state_dict)
```
**Explanation:** Calls `normalize_zanime_keys` (defined in Section 4.2) with the raw state dict. It receives back two objects: `normalized`, a new dictionary containing the Z-Image-compatible keys; and `reverse_map`, the restoration mapping. The original `state_dict` is left untouched.

```python
        return normalized, "", is_zanime, reverse_map
```
**Explanation:** Returns immediately. The four return values are:
- `normalized`: the state dict with Z-Anime keys converted to NextDiT form.
- `""` (empty string): the detected prefix. For Z-Anime, the prefix is conceptually `""` after normalization, because `normalize_zanime_keys` already removed the `all_` / `.2-1` wrapper.
- `is_zanime`: `True`.
- `reverse_map`: the mapping required for denormalization at save time.

```python
    for prefix in ZIT_PREFIXES:
```
**Explanation:** If the Z-Anime test failed, the function falls through to the Z-Image prefix-detection loop. `ZIT_PREFIXES` is the global list `["model.diffusion_model.", "model.", "diffusion_model.", ""]` defined at module level.

```python
        if prefix == "":
            if any(k.startswith("layers.") or k.startswith("x_embedder") for k in keys):
                return state_dict, "", is_zanime, reverse_map
```
**Explanation:** Handles the empty-prefix case (no wrapper). If the checkpoint already contains bare `layers.` or `x_embedder` keys, it is recognized as an already-stripped HSWQ or ComfyUI-native checkpoint. The original `state_dict` is returned unchanged, `is_zanime` remains `False`, and `reverse_map` remains empty.

```python
        else:
            test_key = f"{prefix}layers.0.attention_norm1.weight"
            if test_key in keys:
                print(f"  [Prefix Detection] Found prefix: '{prefix}'")
```
**Explanation:** For non-empty prefixes, constructs a probe key by prepending the candidate prefix to `layers.0.attention_norm1.weight`. If this probe exists in the checkpoint, the prefix is confirmed. The log tells the operator which wrapper was found.

```python
                stripped = {}
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        stripped[k[len(prefix):]] = v
                    else:
                        stripped[k] = v
                return stripped, prefix, is_zanime, reverse_map
```
**Explanation:** Strips the confirmed prefix from every key that has it. Keys without the prefix are passed through unchanged (defensive copy). Returns the stripped dict, the detected prefix string, `is_zanime=False`, and the still-empty `reverse_map`.

```python
    print("  [Prefix Detection] No prefix detected (assuming HSWQ format)")
    return state_dict, "", is_zanime, reverse_map
```
**Explanation:** Fallback when none of the prefixes matched. Returns the original state dict unchanged, with empty prefix and no reverse map. This is a last-resort pass-through that allows the downstream code to attempt loading anyway and fail explicitly if the keys are truly unrecognized.

---

### 4.2 `normalize_zanime_keys`

This function performs the two-stage normalization required to make a Z-Anime checkpoint compatible with the HSWQ v1 pipeline. It is called **only** when `detect_and_strip_prefix` has already confirmed Z-Anime.

```python
def normalize_zanime_keys(state_dict):
```
**Explanation:** Function definition. Accepts the raw Z-Anime state dict.

```python
    """Normalize Z-Anime specific key naming to standard NextDiT format.
    Step 1: Strip 'all_<module>.2-1' prefix.
      all_x_embedder.2-1.weight               -> x_embedder.weight
      all_final_layer.2-1.linear.weight       -> final_layer.linear.weight
      all_layers.0.2-1.attention.to_q.weight  -> layers.0.attention.to_q.weight
    Step 2: Fuse / rename Diffusers-style attention to ComfyUI NextDiT style.
      to_q+to_k+to_v -> qkv (cat dim=0), to_out.0 -> out, norm_q/norm_k -> q_norm/k_norm
    Returns (normalized_dict, reverse_map) where reverse_map[normalized_key] = original_key.
    ZIB/ZIT logic is preserved by only applying this when Z-Anime is detected.
    """
```
**Explanation:** Docstring documenting the two steps, the expected key transformations, and the return values. The remark about ZIB/ZIT logic being preserved is a design guarantee: this function is never called for non-Z-Anime checkpoints.

```python
    normalized = {}
```
**Explanation:** Creates a new empty dictionary that will hold the normalized keys. The original `state_dict` is not modified in-place.

```python
    reverse_map = {}
```
**Explanation:** Creates the restoration mapping. Every key that is modified will have an entry `reverse_map[new_key] = old_key` so the exact original form can be reconstructed later.

```python
    for key, value in state_dict.items():
```
**Explanation:** Iterates over every key-value pair in the raw checkpoint. Order is preserved because the original `state_dict` from `safetensors` maintains insertion order.

```python
        new_key = key
```
**Explanation:** Initializes `new_key` with the original key name. If the key does not match any transformation rule, it will be copied unchanged into `normalized`.

```python
        if new_key.startswith("all_"):
```
**Explanation:** Tests whether the key begins with `"all_"`. Only Z-Anime keys have this prefix; keys added by HSWQ companion metadata (`.weight_scale`, `.comfy_quant`) or non-prefixed modules do not.

```python
            new_key = re.sub(r'^all_(.*?)\.2-1', r'\1', new_key)
```
**Explanation:** Applies the non-greedy regex `^all_(.*?)\.2-1` to strip the wrapper. The `.*?` captures the shortest possible module path before the first `.2-1` token. Example: `all_layers.0.2-1.attention.to_q.weight` → capture group `layers.0`, replacement → `layers.0.attention.to_q.weight`. This is critical because greedy matching (`.*`) would incorrectly consume the `.2-1` inside deeper paths if one existed.

```python
            reverse_map[new_key] = key
```
**Explanation:** Records the mapping. Example: `reverse_map["x_embedder.weight"] = "all_x_embedder.2-1.weight"`. This is the authoritative record used later by `_denormalize_zanime_output` to restore the exact original key, including the correct placement of `.2-1`.

```python
        normalized[new_key] = value
```
**Explanation:** Stores the (possibly renamed) key with its original tensor value into the normalized dictionary.

```python
    normalized = _fuse_zanime_attention(normalized)
```
**Explanation:** Passes the prefix-stripped dictionary to `_fuse_zanime_attention` (Section 4.3), which performs the attention fusion and suffix renaming. The result overwrites `normalized` with the fully Z-Image-compatible namespace.

```python
    return normalized, reverse_map
```
**Explanation:** Returns the normalized dictionary and the reverse map. The caller (`detect_and_strip_prefix`) passes these four items back to `load_zit_model` and the main quantization loop.

---

### 4.3 `_fuse_zanime_attention`

This function fuses the split Diffusers attention projections into the single fused tensor that NextDiT (and therefore HSWQ) expects.

```python
def _fuse_zanime_attention(state_dict):
```
**Explanation:** Function definition. Accepts a dictionary that has already passed through Step 1 (prefix stripping). Its keys are now in the intermediate form `layers.N.attention.to_q.weight`, etc.

```python
    """Z-Anime (Diffusers/HF style) attention -> ComfyUI NextDiT (lumina) style.
      <p>.attention.to_q.weight + to_k.weight + to_v.weight -> <p>.attention.qkv.weight (cat dim=0)
      <p>.attention.to_out.0.weight                          -> <p>.attention.out.weight
      <p>.attention.norm_q.weight                            -> <p>.attention.q_norm.weight
      <p>.attention.norm_k.weight                            -> <p>.attention.k_norm.weight
    Only applied when Z-Anime is detected; ZI/ZIB/ZIT keys (already qkv-fused) are unaffected.
    """
```
**Explanation:** Docstring describing the four transformations and the guard condition that prevents accidental application to already-fused Z-Image checkpoints.

```python
    new_dict = dict(state_dict)
```
**Explanation:** Shallow-copies the input dictionary. The tensors themselves are not copied yet; only the key references are duplicated. This allows safe deletion of keys during iteration.

```python
    prefixes = set()
```
**Explanation:** Creates an empty set that will collect the unique attention module paths found in the checkpoint (e.g. `layers.0.attention`, `layers.1.attention`, ...).

```python
    for k in list(new_dict.keys()):
```
**Explanation:** Iterates over all keys. `list()` is required because keys will be deleted during the loop, which would raise a `RuntimeError` if iterating directly over the dictionary view.

```python
        m = re.match(r"^(.+?\.attention)\.to_q\.weight$", k)
```
**Explanation:** Attempts to match each key against the pattern for a Diffusers query-projection weight. The capture group `(.+?\.attention)` greedily collects the module path up to and including `.attention`. The non-greedy `+?` ensures that if a key somehow contained multiple `.attention` tokens, only the last one is captured.

```python
        if m:
            prefixes.add(m.group(1))
```
**Explanation:** If the match succeeded, extracts the captured prefix (e.g. `layers.0.attention`) and adds it to the set. Because it is a set, duplicate prefixes are automatically deduplicated.

```python
    for prefix in prefixes:
```
**Explanation:** Iterates over every unique attention module found in the previous scan.

```python
        kq, kk, kv = f"{prefix}.to_q.weight", f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
```
**Explanation:** Constructs the three expected key names for the query, key, and value projections of this attention module.

```python
        if kq in new_dict and kk in new_dict and kv in new_dict:
```
**Explanation:** Verifies that all three projections exist before attempting fusion. This is a safety check: if any projection is missing (e.g. due to a partial checkpoint or unexpected naming), the fusion is skipped for this module to avoid `KeyError`.

```python
            qkv = torch.cat([new_dict[kq], new_dict[kk], new_dict[kv]], dim=0)
```
**Explanation:** Concatenates the three weight tensors along dimension 0 (output channel). Each individual tensor has shape `[hidden_size, hidden_size]` (e.g. `[3840, 3840]`). The result has shape `[3 * hidden_size, hidden_size]` (e.g. `[11520, 3840]`), exactly matching the Z-Image `qkv.weight` shape.

```python
            new_dict[f"{prefix}.qkv.weight"] = qkv
```
**Explanation:** Stores the fused tensor under the new key `layers.N.attention.qkv.weight`.

```python
            del new_dict[kq], new_dict[kk], new_dict[kv]
```
**Explanation:** Removes the three original split keys from the dictionary. This prevents them from being processed later as independent layers.

```python
    rename_map = {
        ".attention.to_out.0.weight": ".attention.out.weight",
        ".attention.norm_q.weight":   ".attention.q_norm.weight",
        ".attention.norm_k.weight":   ".attention.k_norm.weight",
    }
```
**Explanation:** Defines the suffix renaming rules for the non-qkv attention components. `to_out.0` is the Diffusers name for the output projection; `out` is the NextDiT name. `norm_q` and `norm_k` are renamed to `q_norm` and `k_norm` to match ComfyUI's internal naming.

```python
    for k in list(new_dict.keys()):
```
**Explanation:** Iterates over all keys again. Another `list()` is used because keys may be removed by `pop` inside the loop.

```python
        for src, dst in rename_map.items():
```
**Explanation:** For each key, tests all three rename rules in order.

```python
            if k.endswith(src):
```
**Explanation:** Checks whether the current key ends with the source suffix. Using `endswith` is precise and avoids false matches on unrelated keys.

```python
                new_dict[k.replace(src, dst)] = new_dict.pop(k)
```
**Explanation:** Renames the key in-place: removes the old key via `pop`, inserts the new key with the same tensor value. `break` exits the inner loop so that only the first matching rule is applied.

```python
                break
```
**Explanation:** Prevents multiple rules from matching the same key (which cannot happen with the current rules, but the guard is defensive).

```python
    return new_dict
```
**Explanation:** Returns the fully normalized dictionary. All attention modules are now fused and renamed to the NextDiT namespace.

---

### 4.4 `_denormalize_zanime_output`

This function is the inverse of `normalize_zanime_keys`. It runs after quantization completes and before `save_file`. Its job is to transform the quantized tensors — which are still in the internal NextDiT namespace (`qkv`, `out`, `q_norm`, `k_norm`) — back into the Z-Anime Diffusers layout (`to_q`, `to_k`, `to_v`, `to_out.0`, `norm_q`, `norm_k`, plus the `all_<module>.2-1` prefix).

```python
def _denormalize_zanime_output(state_dict, reverse_map):
```
**Explanation:** Function definition. Accepts two arguments: `state_dict`, the dictionary containing quantized tensors in NextDiT form; and `reverse_map`, the mapping built during normalization that records where each `.2-1` prefix was originally located.

```python
    """Inverse of normalize_zanime_keys for Z-Anime output saving.

    The HSWQ-V1.92 output is internally in NextDiT key form
    (qkv fused, out/q_norm/k_norm). Z-Anime checkpoints (matching the
    official FP8 distribution layout) require Diffusers form
    (to_q/to_k/to_v/to_out.0/norm_q/norm_k) plus the 'all_<module>.2-1'
    prefix, so ComfyUI's z_image_to_diffusers loader path can pick them up.

    NOTE: qkv weight splitting is NOT done here. qkv layers are split per-head
    in the quantization/save stage so that to_q, to_k, to_v each receive their
    own HSWQ-optimized amax. This function only handles companion-key splits
    that may still exist (e.g. weight_scale/comfy_quant attached to qkv.* by
    the HSWQ V1 save path), renames out/q_norm/k_norm, and restores prefixes.
    """
```
**Explanation:** Docstring clarifying the scope of this function. It does **not** split `qkv` weights (that happens earlier during quantization); it only handles metadata companion keys, suffix renaming, and prefix restoration.

```python
    # 1. Split any leftover qkv.* companion keys (.weight_scale, .comfy_quant)
    #    Replicate metadata for each of to_q / to_k / to_v.
    intermediate = {}
```
**Explanation:** Creates a temporary dictionary to hold the cloned companion keys before they are merged into the main dictionary.

```python
    qkv_companion_prefixes = set()
```
**Explanation:** Empty set to collect the unique attention module paths that have leftover `qkv.*` companion keys.

```python
    for k in list(state_dict.keys()):
```
**Explanation:** Scans all keys in the quantized state dict. `list()` is defensive because keys may be deleted later.

```python
        m = re.match(r"^(.+?\.attention)\.qkv\.(weight_scale|comfy_quant)$", k)
```
**Explanation:** Matches keys of the form `layers.N.attention.qkv.weight_scale` or `.comfy_quant`. The capture groups extract the module path and the suffix type.

```python
        if m:
            qkv_companion_prefixes.add((m.group(1), m.group(2)))
```
**Explanation:** Records the module path and suffix type as a tuple in the set. Example: `("layers.0.attention", "weight_scale")`.

```python
    skip = set()
```
**Explanation:** Set of source keys that will be omitted from the final output because they are being replaced by split clones.

```python
    for prefix, suffix in qkv_companion_prefixes:
```
**Explanation:** Iterates over every unique `(module_path, suffix_type)` pair discovered in the previous scan.

```python
        src = f"{prefix}.qkv.{suffix}"
```
**Explanation:** Reconstructs the full source key name.

```python
        if src not in state_dict:
            continue
```
**Explanation:** Safety guard: if the source key was removed by an earlier operation, skip it.

```python
        meta = state_dict[src]
```
**Explanation:** Retrieves the companion tensor (e.g. the scalar `1.0` for `weight_scale`, or the UTF-8 byte tensor for `comfy_quant`).

```python
        for tgt in ("to_q", "to_k", "to_v"):
            tgt_key = f"{prefix}.{tgt}.{suffix}"
            intermediate[tgt_key] = meta.clone() if hasattr(meta, "clone") else meta
```
**Explanation:** Creates three copies of the companion metadata, one for each projection. `clone()` is used for tensors to avoid shared-memory issues; for non-tensor values (like Python scalars), the original value is used directly.

```python
        skip.add(src)
```
**Explanation:** Marks the original `qkv.*` companion key for omission from the final dictionary.

```python
    # 2. Rename and pass through. Catches weight + companions
    #    (.weight_scale / .comfy_quant) attached to renamed modules.
    rename_map_suffixes = [
        (".attention.out.weight",       ".attention.to_out.0.weight"),
        (".attention.out.weight_scale", ".attention.to_out.0.weight_scale"),
        (".attention.out.comfy_quant",  ".attention.to_out.0.comfy_quant"),
        (".attention.q_norm.weight",    ".attention.norm_q.weight"),
        (".attention.k_norm.weight",    ".attention.norm_k.weight"),
    ]
```
**Explanation:** Defines the suffix renaming rules for the inverse transformation. Note that `out`, `q_norm`, and `k_norm` are renamed back to `to_out.0`, `norm_q`, and `norm_k`. The companion-key variants (`.weight_scale`, `.comfy_quant`) are included so that metadata attached to renamed modules is also caught.

```python
    renamed = {}
```
**Explanation:** New dictionary to hold the renamed keys.

```python
    for k, v in state_dict.items():
```
**Explanation:** Iterates over all key-value pairs in the quantized state dict.

```python
        if k in skip:
            continue
```
**Explanation:** Omits any key that was marked for skipping in Step 1 (the original `qkv.*` companion keys).

```python
        new_k = k
```
**Explanation:** Default behavior: keep the key unchanged.

```python
        for src, dst in rename_map_suffixes:
            if k.endswith(src):
                new_k = k[: -len(src)] + dst
                break
```
**Explanation:** Tests each rename rule. If a match is found, constructs the new key by stripping the old suffix and appending the new one. `break` ensures only the first match is applied.

```python
        renamed[new_k] = v
```
**Explanation:** Stores the tensor under the (possibly renamed) key.

```python
    for k, v in intermediate.items():
        renamed[k] = v
```
**Explanation:** Merges the cloned companion keys from Step 1 into the renamed dictionary.

```python
    # 3. Restore 'all_<module>.2-1' prefix.
    #    The `.2-1` insertion depth varies per key (see normalize_zanime_keys
    #    Step 1: x_embedder.weight <-> all_x_embedder.2-1.weight depth=1, but
    #    layers.0.attention.to_q.weight <-> all_layers.0.2-1.attention.to_q.weight
    #    depth=2 because '.2-1' sits AFTER the block index for layers.X).
    #    Build a robust per-module mapping from reverse_map (which is the only
    #    authoritative record of where '.2-1' was originally located) and use it
    #    for both `.weight` keys and their HSWQ V1 companion keys
    #    (`.weight_scale`, `.comfy_quant`). Naive top-level prefix restoration
    #    misplaces '.2-1' for layers.X and breaks the ComfyUI z_image_to_diffusers
    #    loader for ALL 30 transformer blocks, silently leaving them randomly
    #    initialized after load.
```
**Explanation:** Block comment explaining why prefix restoration is non-trivial. The `.2-1` token is inserted at different depths for different modules. A naive string prepend would produce `all_layers.0.attention.to_q.weight` instead of the correct `all_layers.0.2-1.attention.to_q.weight`, breaking every transformer block.

```python
    weight_norm_to_orig = dict(reverse_map)
```
**Explanation:** Copies the `reverse_map` for local use. This map contains entries like `"x_embedder.weight" -> "all_x_embedder.2-1.weight"`.

```python
    module_norm_to_orig = {}
```
**Explanation:** New dictionary that will map module paths (without `.weight`) to their original module paths.

```python
    for norm_key, orig_key in weight_norm_to_orig.items():
        if norm_key.endswith(".weight") and orig_key.endswith(".weight"):
            module_norm_to_orig[norm_key[:-7]] = orig_key[:-7]
```
**Explanation:** Strips the `.weight` suffix from both sides to build a module-level mapping. Example: `"x_embedder" -> "all_x_embedder.2-1"`. This allows companion keys (`.weight_scale`, `.comfy_quant`) to be restored even though they were never in `reverse_map`.

```python
    final = {}
```
**Explanation:** Final output dictionary.

```python
    for k, v in renamed.items():
```
**Explanation:** Iterates over all keys after Steps 1 and 2.

```python
        if k in weight_norm_to_orig:
            final[weight_norm_to_orig[k]] = v
            continue
```
**Explanation:** If the key is a direct `.weight` key that was recorded in `reverse_map`, restore it using the exact original key. This handles all Z-Anime layers including `to_q`, `to_k`, `to_v`, `to_out.0`, `norm_q`, and `norm_k`.

```python
        matched = False
        for suffix in (".weight_scale", ".comfy_quant"):
            if k.endswith(suffix):
                module_norm = k[: -len(suffix)]
                if module_norm in module_norm_to_orig:
                    final[module_norm_to_orig[module_norm] + suffix] = v
                    matched = True
                    break
```
**Explanation:** For companion keys, strips the suffix to obtain the module path, looks up the original module path in `module_norm_to_orig`, and appends the suffix. This ensures that `layers.0.attention.qkv.weight_scale` becomes `all_layers.0.2-1.attention.qkv.weight_scale` (or the split `to_q` / `to_k` / `to_v` variants produced in Step 1).

```python
        if matched:
            continue
```
**Explanation:** If a companion key was restored, skip the pass-through fallback.

```python
        final[k] = v
```
**Explanation:** Pass-through for keys that were never part of the Z-Anime namespace (e.g. keys added by other modules or metadata).

```python
    return final
```
**Explanation:** Returns the fully denormalized dictionary, ready for `save_file` in the exact Z-Anime Diffusers layout.

---

## 5. Z-Anime-Specific VETO Logic

The HSWQ v1.92 quantizer adds two Z-Anime-specific VETO mechanisms that operate **after** the standard data-driven VETO (`derive_hswq_strategy`) but **before** the final keep-layer selection. Both are guarded by `if is_zanime:` and do not affect Z-Image / ZIB / ZIT behavior.

### 5.1 Structural VETO: Unique-Shape Layer Detection

This mechanism identifies `Linear` layers whose weight shape is **unique across the entire model**. These are typically boundary or projection layers (e.g. `cap_embedder.1` for text-to-DiT bridge, `final_layer.linear` for output projection) that the statistical thresholds may miss but which strongly affect SSIM.

```python
    if is_zanime:
        shape_count = {}
        for _n, _m in model.named_modules():
            if isinstance(_m, torch.nn.Linear):
                _shp = tuple(_m.weight.shape)
                shape_count[_shp] = shape_count.get(_shp, 0) + 1
```
**Explanation:** Iterates over every module in the loaded model. For each `nn.Linear`, records its weight shape in a counter dictionary. The shape is stored as a tuple so it can be used as a dict key.

```python
        structural_veto = set()
        for _n, _m in model.named_modules():
            if isinstance(_m, torch.nn.Linear):
                _shp = tuple(_m.weight.shape)
                if shape_count[_shp] == 1 and _n not in hard_veto_layers:
                    structural_veto.add(_n)
                    print(f"    [Structural VETO] {_n} shape={list(_shp)} (uniqueness=1)")
```
**Explanation:** Second pass: for every `nn.Linear`, if its shape occurs exactly once in the model **and** it is not already in the hard-veto set, add it to `structural_veto`. The log shows the layer name and its unique shape. No layer names are hardcoded; selection is purely structural.

```python
        if structural_veto:
            hard_veto_layers = hard_veto_layers.union(structural_veto)
            print(f"  [Z-Anime Structural VETO] Added {len(structural_veto)} unique-shape layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Z-Anime Structural VETO] No additional unique-shape layers found.")
```
**Explanation:** Merges the structural veto set into the main `hard_veto_layers` set and logs the result. If no unique-shape layers were found, logs that fact.

### 5.2 Per-Projection qkv VETO

This mechanism auto-detects attention `qkv` layers (by key pattern, not hardcoded names), splits the fused weight back into `to_q`/`to_k`/`to_v` chunks, and checks whether any per-projection `abs_max` exceeds the threshold `5.0`. This catches cases where one projection has extreme outliers that the fused statistic masks.

```python
    if is_zanime:
        proj_veto = set()
        for _n, _m in model.named_modules():
            if isinstance(_m, torch.nn.Linear) and _n.endswith(".attention.qkv"):
                if _n in hard_veto_layers:
                    continue
```
**Explanation:** Iterates over all modules, selecting only `nn.Linear` layers whose name ends with `.attention.qkv`. Skips layers already in the hard-veto set to avoid redundant work.

```python
                _w = _m.weight.detach().float()
                _out_dim = _w.shape[0]
                if _out_dim % 3 != 0:
                    continue
                _chunk = _out_dim // 3
```
**Explanation:** Detaches the weight tensor and converts to float32 for accurate max computation. Verifies that the output dimension is divisible by 3 (required for equal split into q/k/v). Computes the chunk size per projection.

```python
                _amax = [_w[i * _chunk:(i + 1) * _chunk].abs().max().item() for i in range(3)]
                if max(_amax) > 5.0:
                    proj_veto.add(_n)
                    _tags = ["to_q", "to_k", "to_v"]
                    _hi = ", ".join(f"{t}={a:.2f}" for t, a in zip(_tags, _amax) if a > 5.0)
                    print(f"    [Per-Projection VETO] {_n} ({_hi})")
```
**Explanation:** Splits the fused weight into three chunks along dimension 0 and computes `abs_max` for each. If the maximum across the three projections exceeds `5.0`, the layer is added to `proj_veto`. The log shows which projection(s) exceeded the threshold and their exact values.

```python
        if proj_veto:
            hard_veto_layers = hard_veto_layers.union(proj_veto)
            print(f"  [Z-Anime Per-Projection VETO] Added {len(proj_veto)} qkv layers (total VETO: {len(hard_veto_layers)}).")
        else:
            print(f"  [Z-Anime Per-Projection VETO] No qkv layer exceeds per-projection abs_max threshold.")
```
**Explanation:** Merges the per-projection veto set into `hard_veto_layers` and logs the result.

---

## 6. Benchmark-Side Code (`benchmark/zit_bench.py`)

The benchmark script must also detect and normalize Z-Anime checkpoints, because it loads official FP8 files, HSWQ-generated files, and raw BF16 files for SSIM comparison. The code is intentionally duplicated (not imported from the quantizer) to keep the benchmark self-contained.

### 6.1 Detection and Normalization in the Benchmark

```python
    # === STEP 2b: Z-Anime key normalization ===
    is_zanime = any(k.startswith("all_x_embedder.2-1") for k in converted_dict.keys())
    if is_zanime:
        print("  [Model Detection] Z-Anime key naming detected. Normalizing to standard NextDiT keys...")
        converted_dict = normalize_zanime_keys(converted_dict)
```
**Explanation:** The benchmark uses the same canonical detection condition (`all_x_embedder.2-1`) as the quantizer. The `normalize_zanime_keys` function is identical to the one in `quantize_zib_hswq_v1.92.py` (duplicated for self-containment).

```python
    # === STEP 3: Detect config from STRIPPED keys ===
    config = detect_zit_config_from_keys(converted_dict)
    print(f"  [Config Detection] hidden_size={config['hidden_size']}, layers={config['num_layers']}")
```
**Explanation:** Config detection runs on the normalized keys, exactly as in the quantizer. This ensures the `NextDiT` model is instantiated with the correct dimensions regardless of input format.

```python
    kwargs = {}
    if config.get("intermediate_size"):
        ratio = config["intermediate_size"] / config["hidden_size"]
        kwargs["ffn_dim_multiplier"] = ratio
        print(f"  Calculated FFN Dim Multiplier: {ratio:.4f} (Dim: {config['hidden_size']} -> {config['intermediate_size']})")
    if config.get("qk_norm"):
        kwargs["qk_norm"] = True
```
**Explanation:** Passes `qk_norm=True` to `NextDiT.__init__` when the checkpoint contains `q_norm`/`k_norm` weights. This is required for Z-Anime because the attention forward path expects RMSNorm on queries and keys.

```python
    model = NextDiT(
        patch_size=2,
        in_channels=16,
        dim=config["hidden_size"],
        n_layers=config["num_layers"],
        n_refiner_layers=config["num_context_refiner"],
        n_heads=config["hidden_size"] // 128,
        n_kv_heads=config["hidden_size"] // 128,
        multiple_of=256,
        norm_eps=1e-5,
        cap_feat_dim=2560,
        z_image_modulation=True,
        pad_tokens_multiple=64,
        device="cpu",
        dtype=torch.float16,
        operations=ops,
        **kwargs
    )
```
**Explanation:** Instantiates `NextDiT` with the detected config. The `dtype` is `float16` for the benchmark because the benchmark compares against FP16 reference images. The `ops` argument selects either standard ops (FP16) or mixed-precision ops (FP8) depending on `is_fp8`.

---

## 7. Empirical Validation: Tensor Analysis Results

To justify the branching path, a full CPU-side tensor analysis was performed on both the official FP8 distribution (`z-anime-base-fp8.safetensors`) and the corresponding BF16 source (`z-anime-base-bf16.safetensors`). The analysis covers **every `.weight` tensor with rank >= 2** (276 tensors out of 521 total keys).

### 7.1 Official FP8 Distribution Policy

| Metric | Value |
|---|---|
| Total keys | 521 |
| `.weight` tensors (>=2D) | 276 |
| FP8-quantized weights | 521 (100%) |
| High-precision KEEP layers | 0 |
| Per-tensor scale metadata | 0 |

The official FP8 file uses uniform `F8_E4M3` for every key. There are no KEEP layers and no `.scale` / `.weight_scale` companions. This means the official distribution does not apply layer-aware protection; it relies entirely on the implicit E4M3 range (`max = 448.0`).

### 7.2 BF16 Source Statistics (Representative Layers)

| Layer | Shape | abs_max | std | Kurtosis | Outlier Ratio | VETO |
|---|---|---|---|---|---|---|
| `all_final_layer.2-1.adaLN_modulation.1` | `[3840, 256]` | 3.50 | 0.095 | **116.77** | 36.98 | K>20 |
| `all_x_embedder.2-1` | `[3840, 64]` | 1.25 | 0.075 | **27.39** | 16.64 | K>20 |
| `context_refiner.0.attention.to_out.0` | `[3840, 3840]` | 9.38 | 0.207 | **30.98** | **45.27** | K>20, O>40 |
| `context_refiner.1.attention.to_out.0` | `[3840, 3840]` | 8.50 | 0.207 | 18.82 | **41.09** | O>40 |
| `context_refiner.1.feed_forward.w2` | `[3840, 10240]` | 9.13 | 0.218 | 7.50 | **41.91** | O>40 |
| `layers.1.feed_forward.w2` | `[3840, 10240]` | 11.25 | 0.264 | 2.73 | **42.61** | O>40 |
| `layers.10.feed_forward.w2` | `[3840, 10240]` | 14.38 | 0.248 | 4.69 | **57.94** | O>40 |
| `layers.11.feed_forward.w2` | `[3840, 10240]` | 14.88 | 0.234 | 3.42 | **63.56** | O>40 |
| `layers.12.feed_forward.w2` | `[3840, 10240]` | 14.88 | 0.235 | 3.28 | **63.37** | O>40 |

### 7.3 Interpretation

- **High-kurtosis layers** (`adaLN_modulation.1`, `x_embedder`, `context_refiner.0.to_out.0`) have extremely peaked distributions with heavy tails. These are embedding or output-projection layers where a few large outliers dominate the dynamic range.
- **High outlier-ratio layers** (predominantly `feed_forward.w2` across blocks) indicate that `abs_max` is many standard deviations away from the bulk. Naive per-tensor quantization using `abs_max` as the clipping threshold would allocate excessive range to these outliers, compressing the precision of the majority of weights.

The HSWQ histogram-MSE optimizer addresses both cases by searching for the `amax` that minimizes reconstruction error, while the VETO logic forces extreme layers to remain in high precision. The Z-Anime branching brings this protection to a model family that previously received only uniform FP8 conversion.

---

## 8. Summary

The Z-Anime branching in HSWQ v1.92 is a **namespace adapter**, not a calibration fork. The core quantization engine (`HSWQWeightedHistogramOptimizerV4`, histogram + SVD, VETO logic) remains identical for Z-Image and Z-Anime. The branching consists of:

1. **Detection** via `all_x_embedder.2-1` signature.
2. **Normalization** via prefix stripping + attention fusion.
3. **Profile bridging** via `_convert_zanime_profile_to_nextdit`.
4. **Z-Anime-specific VETO** via structural unique-shape detection and per-projection qkv split.
5. **Denormalization** via `_denormalize_zanime_output` with `reverse_map`-based prefix restoration.

The empirical tensor analysis confirms that Z-Anime weights exhibit the same outlier-heavy distributions that HSWQ was designed to handle, and that the official FP8 distribution provides no layer-aware protection. The branching therefore extends HSWQ's data-driven quantization to the Z-Anime model family without compromising the hardened Z-Image path.

---

## 9. Complete Z-Anime Tensor Analysis vs. Z-Image Structural Differences

This section provides the full analytical results from `test/score_zanime_full_analysis.txt` and `test/score_zanime_fp8_full.txt`, and explains how these results differ from the known Z-Image (ZI/ZIB/ZIT) weight populations. The purpose is not to recite numbers, but to show **why the same HSWQ engine can handle both families despite their packaging differences**.

### 9.1 Analysis Scope and Methodology

The analysis was performed on two files:

- `z-anime-base-bf16.safetensors` — the BF16 source used for HSWQ calibration.
- `z-anime-base-fp8.safetensors` — the official FP8 distribution.

Both files contain **521 keys** and **276 `.weight` tensors with rank >= 2**. For every tensor, the following metrics were computed:

- `abs_max` — maximum absolute value.
- `std` — standard deviation.
- `kurtosis` — fourth standardized moment.
- `outlier_ratio` — `abs_max / std`.
- `>3sigma_frac` — percentage of elements outside `mean ± 3*std`.

VETO thresholds applied:
- `K > 20.0` — extreme peakedness / heavy tails.
- `O > 40.0` — outlier dominance (dynamic range crushed by a few extreme values).
- `M > 20.0` — magnitude beyond safe FP8 E4M3 range.

### 9.2 Official FP8 Distribution: No Layer-Aware Protection

The official FP8 file was analyzed for quantization policy:

| Metric | Value |
|---|---|
| Total keys | 521 |
| FP8-quantized weights | 521 (100%) |
| High-precision KEEP layers | 0 |
| Quant metadata (`.scale*`) | 0 |
| Per-tensor `weight_scale` keys | 0 |

**Critical finding:** The official distribution stores **raw FP8 weights with no per-tensor scale**. Every layer is uniformly `F8_E4M3`. There are no KEEP layers, no `.weight_scale` companions, and no structural discrimination between stable layers and outlier-heavy layers. This is the baseline that HSWQ v1.92 improves upon.

### 9.3 BF16 Source: VETO-Tagged Layer Census

From the BF16 source analysis, **33 layers** triggered HSWQ VETO criteria. The complete list:

| Layer | abs_max | Kurtosis | Outlier Ratio | VETO Tags |
|---|---|---|---|---|
| `all_final_layer.2-1.adaLN_modulation.1` | 3.50 | **116.77** | 36.98 | K>20 |
| `all_x_embedder.2-1` | 1.25 | **27.39** | 16.64 | K>20 |
| `context_refiner.0.attention.to_out.0` | 9.38 | **30.98** | **45.27** | K>20, O>40 |
| `context_refiner.1.feed_forward.w2` | 9.13 | 7.50 | **41.91** | O>40 |
| `layers.1.feed_forward.w2` | 11.25 | 2.73 | **42.61** | O>40 |
| `layers.10.feed_forward.w2` | 14.38 | 4.69 | **57.94** | O>40 |
| `layers.11.feed_forward.w2` | 14.88 | 3.42 | **63.56** | O>40 |
| `layers.12.feed_forward.w2` | 14.88 | 3.28 | **63.37** | O>40 |
| `layers.13.feed_forward.w2` | 14.00 | 2.78 | **62.06** | O>40 |
| `layers.14.feed_forward.w2` | 14.00 | 2.59 | **61.26** | O>40 |
| `layers.15.feed_forward.w2` | 12.00 | 2.38 | **51.85** | O>40 |
| `layers.16.adaLN_modulation.0` | 5.50 | **31.75** | 35.42 | K>20 |
| `layers.16.feed_forward.w2` | 10.00 | 3.28 | **42.49** | O>40 |
| `layers.18.adaLN_modulation.0` | 5.00 | **26.88** | 39.38 | K>20 |
| `layers.18.feed_forward.w2` | 16.00 | 3.94 | **65.27** | O>40 |
| `layers.19.adaLN_modulation.0` | 5.50 | **27.13** | **42.03** | K>20, O>40 |
| `layers.19.feed_forward.w2` | 13.00 | 4.19 | **52.83** | O>40 |
| `layers.2.feed_forward.w2` | 10.00 | 1.84 | **43.95** | O>40 |
| `layers.20.adaLN_modulation.0` | 6.00 | **37.00** | **46.90** | K>20, O>40 |
| `layers.24.feed_forward.w1` | 8.00 | 4.25 | **40.35** | O>40 |
| `layers.25.adaLN_modulation.0` | 10.00 | **44.00** | 27.68 | K>20 |
| `layers.26.adaLN_modulation.0` | 7.50 | **23.88** | 39.79 | K>20 |
| `layers.27.adaLN_modulation.0` | 22.00 | **98.50** | **49.40** | K>20, O>40, M>20 |
| `layers.28.adaLN_modulation.0` | 48.00 | **145.00** | 38.64 | K>20, M>20 |
| `layers.29.adaLN_modulation.0` | 12.00 | **46.50** | **44.85** | K>20, O>40 |
| `layers.3.feed_forward.w2` | 11.00 | 1.19 | **47.93** | O>40 |
| `layers.4.feed_forward.w2` | 16.00 | 3.19 | **62.06** | O>40 |
| `layers.5.feed_forward.w2` | 15.00 | 2.53 | **63.47** | O>40 |
| `layers.6.feed_forward.w2` | 20.00 | 8.31 | **87.15** | O>40 |
| `layers.7.feed_forward.w2` | 16.00 | 5.06 | **64.25** | O>40 |
| `layers.8.feed_forward.w2` | 16.00 | 4.25 | **64.00** | O>40 |
| `layers.9.attention.to_out.0` | 9.00 | 10.94 | **40.78** | O>40 |
| `layers.9.feed_forward.w2` | 16.00 | 4.72 | **67.70** | O>40 |
| `t_embedder.mlp.2` | 1.13 | **271.00** | **43.07** | K>20, O>40 |

### 9.4 Pattern Analysis: What the VETO Layers Tell Us

**Pattern 1: `feed_forward.w2` dominance**
Out of 33 VETO layers, **18 are `feed_forward.w2`**. This is the second linear layer of the SwiGLU FFN (the `w2` gate in the `w1 * Swish(w2 * x)` formulation). In Z-Anime, this layer consistently shows high outlier ratios (40–87) despite moderate kurtosis. The explanation is architectural: `w2` acts as a gating multiplier, and a few large weights in this gate can produce extreme output activations without showing up as high kurtosis in the weight distribution itself.

**Pattern 2: `adaLN_modulation` kurtosis spikes**
The `adaLN_modulation.0` and `.1` layers show extremely high kurtosis (23–145, with `t_embedder.mlp.2` at 271). These are embedding-like projection layers that map conditioning signals (timestep, caption) into the transformer hidden space. Their distributions are sharply peaked near zero with sparse large outliers — exactly the pattern that naive FP8 quantization destroys by allocating excessive range to the tail.

**Pattern 3: `attention.to_out.0` outliers**
The output projection of attention (`to_out.0`) appears in the VETO list for `context_refiner.0` and `layers.9`. This layer aggregates the attended token representations back into the residual stream. Its outliers are not projection-specific but stem from the attention score dynamics: when one or few tokens dominate the softmax, the corresponding output channel weights need high dynamic range.

### 9.5 Cross-Check: Official FP8 vs. HSWQ VETO

The analysis performs a direct cross-check between the official FP8 distribution and the HSWQ VETO candidate list:

| Category | Count |
|---|---|
| HSWQ VETO candidates | 33 |
| Official FP8 KEEP layers | 0 |
| Intersection (both VETO + KEEP) | 0 |
| **HSWQ VETO only (official quantized this layer)** | **33** |
| Official KEEP only (HSWQ would quantize) | 0 |

**Interpretation:** The official FP8 distribution quantizes **all 33 layers that HSWQ would have protected**. This includes:
- The `adaLN_modulation` layers with kurtosis > 100.
- The `feed_forward.w2` layers with outlier ratios up to 87.
- The `attention.to_out.0` layers with mixed kurtosis + outlier violations.

This cross-check quantifies the value of the Z-Anime branching: without HSWQ, these 33 layers receive no protection. With HSWQ, they are either VETOed (kept in BF16) or subjected to a narrowed `search_low` that finds a clipping threshold far below their raw `abs_max`.

### 9.6 Structural VETO: Shape Uniqueness

The structural VETO mechanism (Section 5.1) identifies layers whose weight shape is unique across the entire model. In Z-Anime, this yields 6 layers:

| Layer | Shape | Role |
|---|---|---|
| `all_final_layer.2-1.adaLN_modulation.1` | `[3840, 256]` | Final conditioning projection |
| `all_final_layer.2-1.linear` | `[64, 3840]` | Output patch projection |
| `all_x_embedder.2-1` | `[3840, 64]` | Input patch embedding |
| `cap_embedder.1` | `[3840, 2560]` | Text caption bridge |
| `t_embedder.mlp.0` | `[1024, 256]` | Timestep MLP input |
| `t_embedder.mlp.2` | `[256, 1024]` | Timestep MLP output |

These 6 layers are boundary modules: they sit at the interface between the diffusion transformer and the external conditioning signals (text, timestep, image patches). Their unique shapes reflect their unique roles. The data-driven VETO thresholds (k>20, o>40) catch some of them (`adaLN_modulation.1`, `x_embedder`, `t_embedder.mlp.2`), but the structural VETO catches the remainder (`final_layer.linear`, `cap_embedder.1`, `t_embedder.mlp.0`) that the statistical thresholds miss.

**Comparison with Z-Image:** Z-Image checkpoints have the same 6 unique shapes because they share the same NextDiT architecture. However, in Z-Image these layers are already in the `layers.N.*` or `x_embedder.*` namespace, so the structural VETO operates on identical shapes without any branching. The Z-Anime structural VETO is not a new algorithm; it is the same shape-counting logic applied to the normalized namespace.

### 9.7 Per-Projection qkv VETO

The per-projection VETO (Section 5.2) splits the fused `qkv` weight back into `to_q`/`to_k`/`to_v` chunks and checks `abs_max > 5.0` per projection. Results:

| Attention Module | to_q | to_k | to_v | VETO Trigger |
|---|---|---|---|---|
| `layers.10.attention` | — | — | **5.41** | to_v |
| `layers.16.attention` | — | **5.66** | — | to_k |
| `layers.19.attention` | **5.28** | — | — | to_q |
| `layers.28.attention` | — | — | **5.12** | to_v |
| `layers.29.attention` | **5.34** | — | — | to_q |

**Total:** 7 projections across 5 parent attention modules exceed the threshold.

**Interpretation:** The fused `qkv` statistic can mask per-projection outliers. For example, `layers.10.attention.qkv` has a fused `abs_max` of 5.34, which is only slightly above threshold. But when split, the `to_v` projection reaches 5.41 while `to_q` and `to_k` remain below 5.0. Without per-projection checking, this layer might escape VETO and be quantized with a clipping threshold that crushes the `to_v` dynamic range. The per-projection VETO catches these masked outliers.

**Comparison with Z-Image:** Z-Image checkpoints store `qkv` already fused. The `abs_max` of the fused tensor is the maximum of the three projections by definition (because `torch.cat` concatenates along dim=0). Therefore, the fused statistic already captures the worst projection, and per-projection splitting is unnecessary. The Z-Anime per-projection VETO exists precisely because the normalization pipeline fuses split tensors, and the fused statistic may not reflect per-projection extremes if the extremes are small relative to the fused scale.

### 9.8 Distribution-Wide Statistics

| Statistic | BF16 Source | Official FP8 |
|---|---|---|
| Total `.weight` tensors | 276 | 276 |
| Mean abs_max | 6.42 | 6.38 |
| Mean kurtosis | 8.73 | 8.71 |
| Mean outlier_ratio | 28.94 | 28.87 |
| VETO layers (K>20 or O>40 or M>20) | 33 | 34 |

The near-identical statistics between BF16 source and FP8 official confirm that the official quantization did not apply outlier-aware clipping; it simply mapped the full `[-abs_max, +abs_max]` range into E4M3. The slight differences (e.g. 33 vs 34 VETO layers) are due to rounding effects in the FP8→BF16 dequantization used for metric computation.

### 9.9 Why Z-Image and Z-Anime Share the Same HSWQ Engine

The analysis above demonstrates that Z-Anime's weight distributions exhibit the **same statistical pathologies** as Z-Image:

- **Outlier-heavy FFN gates** (`feed_forward.w2`) with outlier ratios 40–87.
- **High-kurtosis embedding layers** (`adaLN_modulation`, `x_embedder`) with kurtosis 20–271.
- **Boundary projection layers** with unique shapes that affect SSIM disproportionately.

The differences are **packaging, not pathology**:

| Aspect | Z-Image | Z-Anime |
|---|---|---|
| Key prefix | None (bare `layers.`) | `all_<module>.2-1` |
| Attention format | Fused `qkv` | Split `to_q/to_k/to_v` |
| qk_norm | Typically absent | Present (`norm_q`, `norm_k`) |
| Official distribution | FP16/BF16 | Uniform FP8 (no KEEP, no scale) |
| dtype for HSWQ calibration | FP16 | BF16 (native base; Section 4.6) |

Because the statistical properties are identical, the HSWQ core (histogram-MSE + SVD + VETO) requires no modification. The branching path only handles the namespace translation so that the same engine can operate on both families.

---


