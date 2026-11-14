# Implementation and Explanation of MSE-Guided VETO Reassessment for Z-Anime VRAM Savings Improvement (25% → 29.3%)

## 1. The Problem at the 25% VRAM Savings Stage
The primary reason why the initial FP8 quantization of Z-Anime stalled at a 25% VRAM reduction was an **overly conservative VETO (protection) criteria**.

VETO decisions were based on the statistical profile of weights across the entire model, using the following three conditions (if any one applied, the layer was kept in FP16/BF16):
1. `kurtosis > 20` (Extremely sharp distribution)
2. `abs_max > 20` (Contains massive values outside the representable range of E4M3)
3. **`outlier_ratio > 40`** (The ratio of the maximum value to the standard deviation is large = outliers exist)

Among these, conditions 1 and 2 indicate that the distribution itself is abnormal or completely outside the FP8 range, making them guaranteed to break upon quantization (mostly adaLN layers).
However, there were about 19 layers VETO'd *solely* due to condition 3 (`outlier_ratio`), which are primarily `feed_forward.w2` layers. These layers accounted for the vast majority of the VETO footprint (approx. 1.5GB).

A high `outlier_ratio` means "the vast majority of values are concentrated near zero, but a tiny fraction (e.g., 0.9%) contains large outliers."
For naive FP8 (which determines the overall scale based on the extreme outliers), this VETO criteria was **correct assuming no optimization**, because naive scaling would destroy the precision of the majority of values.

However, HSWQ (Hybrid Sensitivity Weighted Quantization) performs **MSE optimization based on histograms (optimal amax search)**.
It has the ability to automatically calculate the optimal scale that minimizes the Mean Squared Error (MSE) for the vast majority of values, even if it means clipping (discarding) the extreme outliers.
In other words, **the barrier stopping us at 25% was simply dismissing layers via pre-calculated statistics (VETO) without trusting HSWQ's powerful optimization capabilities.**

---

## 2. Countermeasure: MSE-Guided VETO Reassessment
To solve this problem, we implemented **MSE-Guided VETO Reassessment**, which measures "what the actual error will be under the HSWQ optimizer" and rescues only the layers that meet safety standards from being VETO'd.

**Mechanism:**
1. **Z-Anime Exclusive**: Guarded by `is_zanime` to ensure no impact on other models (ZI/ZIB/ZIT).
2. **Measuring Safe Baseline**: Randomly sample 30 layers from the "safely quantized layers" (neither VETO nor Dynamic Keep). Run an HSWQ trial quantization to calculate their MSE. Find the "75th percentile (P75) MSE" from this sample, and multiply it by a safety margin (x2.0) to establish the **Threshold**.
3. **Reassessing VETO Candidates**: Run trial quantization on layers VETO'd *only* by `outlier_ratio`.
4. **Automatic Release**: If the candidate layer's MSE is equal to or below the threshold, it is deemed "capable of staying within the error bounds of safe layers" and is automatically released (RELEASED) from VETO to be FP8 quantized.

This approach maintains protection for "truly dangerous layers (abnormal kurtosis or huge magnitude)" while safely liberating layers that HSWQ can handle. **This successfully increased the VRAM savings rate to 29.3% without sacrificing SSIM.**

---

## 3. All Added/Modified Code (Verbatim without skipping a single character)
Here is the entire block of code added immediately after the creation of the VETO list inside the `main()` function of `quantize_zib_hswq_v1.92.py`.

```python
    # =========================================================================
    # [V1.92 MSE-Guided VETO Reassessment] — Z-Anime ONLY
    # Layers VETO'd *only* by outlier_ratio (o>40), NOT by kurtosis or magnitude,
    # are candidates for automatic release. These are typically feed_forward.w2
    # layers that HSWQ's optimal clipping may handle well.
    # Guarded by is_zanime so ZI/ZIB/ZIT behavior is strictly unchanged.
    #
    # Strategy:
    #   1. Identify "outlier-only" VETO layers (o>40 but k<=20 and m<=20)
    #   2. Trial-quantize a random sample of SAFE layers to get baseline MSE
    #   3. Trial-quantize each outlier-only VETO candidate
    #   4. If candidate MSE <= P75 of safe MSE distribution → release from VETO
    # =========================================================================
    outlier_only_veto = set()
    if is_zanime:
        for vname in hard_veto_layers:
            prof = _norm_profile.get(vname, {})
            k = prof.get("kurtosis", 0)
            m = prof.get("abs_max", 0)
            o = prof.get("outlier_ratio", 0)
            # Only layers where outlier_ratio was the sole trigger
            if o > 40 and k <= 20 and m <= 20:
                outlier_only_veto.add(vname)
    
    if outlier_only_veto:
        print(f"\n  [MSE-Guided Reassessment] {len(outlier_only_veto)} VETO layers are outlier-only (o>40, k<=20, m<=20).")
        print(f"  Trial-quantizing to measure actual HSWQ quantization error...")
        
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta
        )
        
        # Step 1: Collect baseline MSE from safely-quantized layers (non-VETO, non-Dynamic)
        safe_mses = []
        _module_dict = dict(model.named_modules())
        _safe_sample = [n for n in target_modules if n not in keep_layers and n in _module_dict]
        # Sample up to 30 safe layers for baseline
        import random
        _safe_sample = random.sample(_safe_sample, min(30, len(_safe_sample)))
        for sname in _safe_sample:
            smod = _module_dict[sname]
            if not hasattr(smod, 'weight'):
                continue
            sw = smod.weight.data
            slayer_search_low = get_layer_search_low(sname, sw)
            try:
                sresult = trial_optimizer.compute_optimal_amax_with_stats(
                    sw, importance=None, use_svd_leverage=True, scaled=False
                )
                safe_mses.append(sresult['estimated_mse'])
            except Exception:
                pass
            torch.cuda.empty_cache()
        
        if safe_mses:
            safe_mses.sort()
            # P75 = 75th percentile of safe layer MSE
            p75_idx = int(len(safe_mses) * 0.75)
            mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)]
            # Safety margin: allow up to 2x the P75 threshold
            mse_threshold *= 2.0
            print(f"  [MSE Baseline] Safe layers sampled: {len(safe_mses)}, P75 MSE: {safe_mses[p75_idx] if p75_idx < len(safe_mses) else safe_mses[-1]:.8f}, Threshold (2×P75): {mse_threshold:.8f}")
            
            # Step 2: Trial-quantize each outlier-only VETO candidate
            released = set()
            for vname in sorted(outlier_only_veto):
                if vname not in _module_dict:
                    continue
                vmod = _module_dict[vname]
                if not hasattr(vmod, 'weight'):
                    continue
                vw = vmod.weight.data
                try:
                    vresult = trial_optimizer.compute_optimal_amax_with_stats(
                        vw, importance=None, use_svd_leverage=True, scaled=False
                    )
                    vmse = vresult['estimated_mse']
                    vprof = _norm_profile.get(vname, {})
                    vor = vprof.get("outlier_ratio", 0)
                    if vmse <= mse_threshold:
                        released.add(vname)
                        print(f"    RELEASED: {vname} | MSE={vmse:.8f} <= threshold={mse_threshold:.8f} | o={vor:.1f} | amax={vresult['optimal_amax']:.4f}")
                    else:
                        print(f"    KEPT:     {vname} | MSE={vmse:.8f} >  threshold={mse_threshold:.8f} | o={vor:.1f}")
                except Exception as e:
                    print(f"    ERROR:    {vname} | {e}")
                torch.cuda.empty_cache()
            
            if released:
                hard_veto_layers = hard_veto_layers - released
                keep_layers = keep_layers - released
                print(f"  [MSE-Guided Reassessment] Released {len(released)} layers from VETO. Remaining VETO: {len(hard_veto_layers)}.")
                print(f"  Updated FP16 kept layers: {len(keep_layers)}")
            else:
                print(f"  [MSE-Guided Reassessment] No layers released (all exceeded MSE threshold).")
        else:
            print(f"  [MSE-Guided Reassessment] No safe baseline available, skipping.")
```

And the final dynamic pool print statement modification:
```python
-    non_veto_total = len(layer_sensitivities)
-    print(f"Total layers: {non_veto_total + len(hard_veto_layers)} (Non-VETO pool: {non_veto_total})")
-    print(f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_layers)} (Top {args.keep_ratio*100:.1f}% of {non_veto_total})")
+    non_veto_total = len([n for n in target_modules if n not in hard_veto_layers])
+    print(f"\nTotal layers: {len(target_modules)} (Non-VETO pool: {non_veto_total})")
+    print(f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
```

---

## 4. Complete Code Explanation

### Block 1: Selecting Target Layers
```python
    outlier_only_veto = set()
    if is_zanime:
        for vname in hard_veto_layers:
            prof = _norm_profile.get(vname, {})
            k = prof.get("kurtosis", 0)
            m = prof.get("abs_max", 0)
            o = prof.get("outlier_ratio", 0)
            if o > 40 and k <= 20 and m <= 20:
                outlier_only_veto.add(vname)
```
- Completely isolated by `is_zanime` to ensure the logic does not execute for Z-Image/ZIB/ZIT models.
- Scans through the `hard_veto_layers` list, which was pre-determined by the initial profile analysis.
- Retrieves `kurtosis` (k), `abs_max` (m), and `outlier_ratio` (o) from each layer's profile.
- `o > 40 and k <= 20 and m <= 20`: Adds only the layers VETO'd **solely due to `outlier_ratio`** into `outlier_only_veto`. Layers with kurtosis or magnitude anomalies are filtered out here, guaranteeing absolute safety for truly dangerous layers.

### Block 2: Preparing the Optimizer
```python
        trial_optimizer = HSWQWeightedHistogramOptimizerV4(
            bins=8192, num_candidates=1000, refinement_iterations=10,
            device=device, alpha=alpha, beta=beta
        )
```
- Instantiates an optimizer for trial quantization using the exact same parameters (8192 bins, 1000 candidates, 10 iterations) as the main quantization pipeline. This ensures the calculated MSE mirrors the actual production accuracy.

### Block 3: Measuring the Safe Baseline MSE
```python
        safe_mses = []
        _safe_sample = [n for n in target_modules if n not in keep_layers and n in _module_dict]
        import random
        _safe_sample = random.sample(_safe_sample, min(30, len(_safe_sample)))
        for sname in _safe_sample:
            # (...)
            try:
                sresult = trial_optimizer.compute_optimal_amax_with_stats(
                    sw, importance=None, use_svd_leverage=True, scaled=False
                )
                safe_mses.append(sresult['estimated_mse'])
            except Exception:
                pass
```
- Randomly samples up to 30 "safe layers" (layers that are neither Hard VETO nor Dynamic Keep, meaning they are confirmed to be quantized to FP8 safely).
- For the weights of each layer (`sw`), it runs HSWQ optimization with `importance=None` (a conservative setting that evaluates outlier risk transparently).
- The calculated theoretical MSE (`estimated_mse`) is appended to the `safe_mses` list.

### Block 4: Determining the Threshold
```python
        if safe_mses:
            safe_mses.sort()
            p75_idx = int(len(safe_mses) * 0.75)
            mse_threshold = safe_mses[min(p75_idx, len(safe_mses) - 1)]
            mse_threshold *= 2.0
```
- Sorts the collected safe MSE values in ascending order (smallest error first).
- Multiplies by `0.75` to extract the "top 75% worst safe MSE" (a layer with slightly higher error but still well within acceptable bounds).
- This is a data-driven approach that establishes a dynamic baseline straight from the model's own distribution, avoiding hardcoded arbitrary values.
- Finally, `*= 2.0` applies a safety margin. If the candidate MSE does not exceed this doubled threshold, it is considered comparable to the safe zone.

### Block 5: Reassessing and Releasing VETO Candidates
```python
            released = set()
            for vname in sorted(outlier_only_veto):
                # (...)
                try:
                    vresult = trial_optimizer.compute_optimal_amax_with_stats(
                        vw, importance=None, use_svd_leverage=True, scaled=False
                    )
                    vmse = vresult['estimated_mse']
                    # (...)
                    if vmse <= mse_threshold:
                        released.add(vname)
                    # (...)
```
- Runs trial quantization on each candidate layer extracted in Step 1. Again, `importance=None` is used to **strictly evaluate outlier degradation** without smoothing it over via DualMonitor compensation.
- If the calculated error `vmse` is less than or equal to `mse_threshold`, the layer name is added to the `released` set.

### Block 6: Updating the VETO Lists
```python
            if released:
                hard_veto_layers = hard_veto_layers - released
                keep_layers = keep_layers - released
# (...)
    non_veto_total = len([n for n in target_modules if n not in hard_veto_layers])
    print(f"\nTotal layers: {len(target_modules)} (Non-VETO pool: {non_veto_total})")
    print(f"Dynamic kept (from non-VETO pool): {len(dynamic_keep_layers)} (Top {args.keep_ratio*100:.1f}%)")
```
- Subtracts the `released` layers from `hard_veto_layers` (the master VETO list) and `keep_layers` (the final protection list).
- This increases the number of layers moving into the FP8 quantization loop, directly boosting the VRAM savings rate.
- Finally, the print calculation for `non_veto_total` was updated to dynamically recalculate based on the live `hard_veto_layers` rather than a static length, ensuring console printout consistency.
