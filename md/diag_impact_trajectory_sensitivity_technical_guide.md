# Trajectory-Sensitivity Impact Ranking and Error Interaction Analysis (Technical Guide)

**Source:** `Z_Image/diag_impact.py`, `Krea2/diag_impact.py`  
This document is a complete, mathematical explanation of the per-layer impact-ranking method
used by the reverse hybrid quantization of diffusion models (implemented in
`Z_Image/diag_impact.py` for Z-Image / Lumina-family DiT and `Krea2/diag_impact.py` for
Krea2 / SDXL-family SingleStreamDiT), and — more fundamentally — of **why single-layer
importance alone cannot predict joint quantization error**, i.e. the mathematical structure of
**error interaction, nonlinear amplification, and error cancellation** that the reverse method
is built around. The theory is universal: it applies to any iterative sampling dynamical
system whose weights are perturbed by quantization.

---

## 1. Overview and Core Idea

### 1.1 What the module computes

`diag_impact.py` assigns to every linear layer a scalar **impact**, defined as the relative
divergence of a fixed denoising trajectory when that one layer's weights are replaced by their
NVFP4-quantized reconstruction, with all other layers held at their reference weights:

$$\mathrm{impact}(l) \;=\; \frac{\bigl\|\,x^{(l)} - x_{\mathrm{ref}}\,\bigr\|_F^{2}}{\bigl\|\,x_{\mathrm{ref}}\,\bigr\|_F^{2}}.$$

### 1.2 The central claim this document proves mathematically

The historical HSWQ approaches (weighted histogram MSE, cosine similarity, full SVD saliency)
all compute a **per-layer scalar in isolation**. This document shows — through a Taylor
expansion of the sampling map (§4), the interaction term (§4.3), and the Lyapunov-style growth
of trajectory error (§5) — that the joint effect of quantizing *multiple* layers is **not the
sum** of the single-layer effects, and that a per-layer scalar can therefore **never** be a
sufficient predictor. The reverse method sidesteps this by measuring impact **in the low-error
regime where additivity holds** (§6) and by measuring **marginal effects in the context closest
to the target configuration** (§7).

---

## 2. Mathematical Setup

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| \(W = (W_1,\dots,W_L)\) | Model weights; \(W_l\) = weights of linear layer \(l\), \(L\) layers. |
| \(\Phi\) | The **sampling map**: weights \(\mapsto\) final latent after a fixed denoising trajectory. |
| \(x_{\mathrm{ref}} = \Phi(W)\) | Reference (pristine) final latent. |
| \(\varepsilon_l\) | Quantization-error perturbation of layer \(l\): \(\widehat{W}_l = W_l + \varepsilon_l\). |
| \(\varepsilon = (\varepsilon_1,\dots,\varepsilon_L)\) | Full perturbation vector. |
| \(e_l\) | Unit vector along layer \(l\) (only layer \(l\) perturbed). |
| \(J_l = \partial\Phi/\partial W_l\) | Jacobian of \(\Phi\) w.r.t. layer \(l\). |
| \(H_{lm} = \partial^2\Phi/\partial W_l\partial W_m\) | Hessian block between layers \(l,m\). |
| \(S \subseteq \{1,\dots,L\}\) | A set of quantized layers. |
| \(\varepsilon_S = \sum_{l\in S}\varepsilon_l e_l\) | Perturbation restricted to \(S\). |

### 2.2 The sampling map \(\Phi\)

\(\Phi\) is the composition of a fixed denoising trajectory. With noise levels
\(\sigma_0 > \sigma_1 > \dots > \sigma_T\) and model output \(v_\theta\), the Euler integrator is

$$x_{k+1} \;=\; x_k \;+\; (\sigma_{k+1}-\sigma_k)\, v_\theta(x_k, \sigma_k; W), \qquad x_0 \sim \mathcal{N}(0,I)\ \text{(seed-locked)}.$$

Hence \(\Phi(W) = x_T\). Because the seed and schedule are fixed, \(\Phi\) is a **deterministic**
function of \(W\). All impacts are therefore properties of the model, reproducible bit-for-bit.

### 2.3 The perturbation model

$$\widehat{W}_l \;=\; Q_{\mathrm{NVFP4}}(W_l), \qquad \varepsilon_l \;=\; \widehat{W}_l - W_l.$$

The relative divergence of any configuration is

$$D(S) \;=\; \frac{\bigl\|\Phi(W + \varepsilon_S) - \Phi(W)\bigr\|_F^2}{\bigl\|\Phi(W)\bigr\|_F^2}.$$

Single-layer impact is \(D(\{l\})\).

---

## 3. Function-by-Function Analysis

### 3.1 `nvfp4_quant_error` — true NVFP4 round-trip

```python
def nvfp4_quant_error(w):
    """TRUE NVFP4 quantization error via comfy_kitchen roundtrip
    (E2M1 x 16-element blocks + global scale): exactly the kernel that
    produces the shipped artifact."""
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as _NVFP4
    w2 = w if w.is_contiguous() else w.contiguous()
    qdata, params = _NVFP4.quantize(w2)
    return _NVFP4.dequantize(qdata, params)
```

This is the **exact NVFP4 kernel** used in production artifacts — E2M1 (4-bit, max magnitude
6.0) with 16-element block scales and a per-tensor global scale, implemented in
`comfy_kitchen.tensor.nvfp4.TensorCoreNVFP4Layout`. The quantize→dequantize round-trip
produces the reconstruction \(\widehat{W}_l\), and the perturbation
\(\varepsilon_l = \widehat{W}_l - W_l\) is the **true on-hardware quantization error** — not a
proxy or approximation.

**Historical note.** Earlier versions of `diag_impact.py` used a structured error proxy
(E4M3 rounding with 256-element group scales) to approximate the NVFP4 error structure. That
proxy understated the actual error by approximately 13× and flattened the layer ranking. The
current implementation uses the exact production kernel, eliminating the proxy-to-reality gap
entirely.

### 3.2 `rel_mse` — scale-invariant divergence

$$\mathrm{relMSE}(a,b) \;=\; \frac{\sum_i (a_i-b_i)^2}{\sum_i b_i^2} \;=\; \frac{\|a-b\|_F^2}{\|b\|_F^2}.$$

### 3.3 `run` / `run(run_seed)` — the configurable-step trajectory

Both variants define `run` as a **local closure** inside `main()`, capturing the model, device,
seed, step count, and context tensors from the enclosing scope.

**Z_Image:**

```python
def run():
    x = torch.randn(1, 16, 128, 128, device=device, dtype=torch.float16,
                    generator=torch.Generator(device).manual_seed(seed))
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    with torch.no_grad():
        for step in range(steps):
            out = model(x, sigmas[step:step + 1], embeds, None, attention_mask=None)
            if isinstance(out, tuple):
                out = out[0]
            x = (x + (sigmas[step + 1] - sigmas[step]) * out).to(torch.float16)
    return x
```

**Krea2:**

```python
def run(run_seed=None):
    s = seed if run_seed is None else run_seed
    g = torch.Generator(device=device).manual_seed(s)
    x = torch.randn(1, channels, lat, lat, device=device,
                    dtype=torch.bfloat16, generator=g)
    with torch.no_grad():
        for step in range(steps):
            t = t_steps[step:step + 1]
            out = model(x, t, context)
            if isinstance(out, tuple):
                out = out[0]
            x = (x + (t_steps[step + 1] - t_steps[step]) * out).to(torch.bfloat16)
    return x
```

Key differences from the original `run4`:

| Property | Old (`run4`) | Z_Image `run` | Krea2 `run` |
|----------|-------------|---------------|-------------|
| σ schedule | Fixed `[1.0, 0.75, 0.5, 0.25, 0.0]` | `linspace(1.0, 0.0, steps+1)` | same |
| Steps | Hard-coded 4 | CLI `--steps` (default 4) | CLI `--steps` (default 4) |
| Latent shape | unspecified | `(1, 16, 128, 128)` fp16 | `(1, channels, lat, lat)` bf16 |
| Context | unspecified | `(1, 256, 2560)` fp16 (fixed) | `(1, seq, txtlayers*txtdim)` bf16 |
| Seed override | none | none | `run_seed` parameter (for multi-seed amax) |

The Euler integration is identical to §2.2:
\(x_{k+1} = x_k + (\sigma_{k+1}-\sigma_k)\, v_\theta(x_k, \sigma_k; W)\).

### 3.4 `main` — the measurement loop

The measurement loop follows the same pattern in both variants:

1. **Load model** — Z_Image loads a NextDiT via the benchmark module's `load_zit_model`;
   Krea2 loads a `SingleStreamDiT` via `load_krea2` (see §3.5).
2. **Enumerate target layers** — read `_quantization_metadata.layers` from the INT8
   artifact's safetensors metadata. These are the layer keys whose NVFP4 impact will be
   measured.
3. **Filter modules** — only `nn.Linear` modules (those with `weight` and `in_features`
   attributes) are eligible. Krea2 additionally enforces `_SAFE_IN_FEATURES` (see §3.7).
4. **Pristine run** — compute \(x_{\mathrm{ref}} = \Phi(W)\). Krea2 also captures
   Hadamard-rotated activation amax during this run (see §3.6).
5. **Krea2 only: extra-seed amax runs** — run with seeds 1337 and 7 (in addition to the
   default 42) to capture the running-max amax across diverse inputs, improving
   `input_scale` calibration robustness.
6. **Per-layer injection** — for each target layer: clone \(W_l\), replace with
   \(\widehat{W}_l = \mathrm{nvfp4\_quant\_error}(W_l)\), run \(\Phi\), record
   \(D(\{l\})\), restore \(W_l\).
7. **Save output** — write JSON.

**Output JSON format:**

| Field | Z_Image | Krea2 | Description |
|-------|---------|-------|-------------|
| `x_ref_norm` | ✅ | ✅ | \(\|x_{\mathrm{ref}}\|_F^2\), the denominator of relMSE |
| `impacts` | ✅ | ✅ | `{layer_name: relMSE_value}` dict |
| `act_amax` | — | ✅ | `{module_name: max_abs_rotated_activation}` for `input_scale` |

### 3.5 Krea2-specific: ComfyUI bootstrap and model loading

Krea2's `diag_impact.py` includes a full ComfyUI bootstrap stack because the Krea2
`SingleStreamDiT` model class lives inside `comfy.ldm.krea2.model`:

| Function | Role |
|----------|------|
| `_ensure_comfyui(comfy_path)` | Locate ComfyUI root (repo-internal `ComfyUI-master` only) |
| `_load_comfy_pkg(comfy_root)` | Import `comfy` exclusively from the specified root, purging any other `comfy` on `sys.path` |
| `_install_comfy_stubs()` | Install `torchaudio`, `comfy_aimdo`, `psutil` stubs for headless operation |
| `_find_krea2_key_prefix(keys)` | Detect state-dict key prefix (`model.diffusion_model.`, `diffusion_model.`, or empty) |
| `detect_krea2_dit_config(sd, prefix)` | Auto-detect Krea2 DiT hyperparameters (features, channels, layers, heads, kvheads, txtlayers, txtdim) from state-dict shapes |
| `load_krea2(path, device, comfy_path)` | Full load pipeline: bootstrap → detect config → instantiate `SingleStreamDiT` → load stripped state-dict → validate CUDA placement |

The Z_Image variant delegates model loading to the benchmark module (`bench.load_zit_model`),
which handles the equivalent bootstrap internally.

### 3.6 Krea2-specific: Hadamard-rotated activation amax calibration

ConvRot NVFP4 artifacts require a per-layer `input_scale` parameter (the activation scaling
factor for the NVFP4 kernel). `gen_reverse_nvfp4.py` computes this as
\(\mathrm{amax} / (\mathrm{F8\_E4M3\_MAX} \times \mathrm{F4\_E2M1\_MAX})\). The amax value
must be captured during a representative forward pass.

```python
def _build_hadamard(size, device="cuda", dtype=torch.float32):
    """Normalized Hadamard (Kronecker power of h4, / sqrt(size))."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32, device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return (h / (size ** 0.5)).to(dtype=dtype)

def _rotated_amax(x, H):
    """Max abs of Hadamard-rotated activation (last dim, groups of H.shape[0])."""
    gs = int(H.shape[0])
    xf = x.detach().float()
    flat = xf.reshape(-1, xf.shape[-1])
    f = flat.shape[-1]
    if f % gs != 0:
        return None
    g = flat.reshape(-1, f // gs, gs)
    rot = torch.matmul(g, H.to(device=flat.device, dtype=torch.float32))
    return float(rot.abs().max().item())
```

During the pristine run, a `forward_pre_hook` captures the running-max rotated amax for each
eligible module. To improve calibration robustness (a single 4-step seed can under-cover real
activation ranges), extra seeds (1337, 7) are run after the pristine seed to expand the amax
coverage. All amax values are saved as `act_amax` in the output JSON.

### 3.7 Krea2-specific: `_SAFE_IN_FEATURES` whitelist

Not all Linear layers in a Krea2 model can be converted to NVFP4. The NVFP4 kernel
(`TensorCoreNVFP4Layout` + `validate_nvfp4_weight_storage`) requires specific `in_features`
dimensions that produce valid packed weight storage:

```python
_SAFE_IN_FEATURES = {1536, 6144, 16384}
```

Layers whose `in_features` is not in this set (e.g. `txtfusion` layers) are assigned
`impact = NaN` and skipped. Both `diag_impact.py` and `gen_reverse_nvfp4.py` enforce this
constraint identically.

---

## 4. First-Order Linearization and the Interaction Term

### 4.1 Taylor expansion of the sampling map

Expand \(\Phi\) around the reference weights \(W\). Writing \(\delta = \sum_l \varepsilon_l e_l\):

$$\Phi(W + \delta) \;=\; \Phi(W) \;+\; \underbrace{\sum_{l} J_l \varepsilon_l}_{\text{first order}} \;+\; \underbrace{\frac12 \sum_{l,m} \varepsilon_l^{\top} H_{lm}\, \varepsilon_m}_{\text{second order}} \;+\; O\bigl(\|\delta\|^3\bigr).$$

The **single-layer impact** is, to first order,

$$\mathrm{impact}(l) \;=\; D(\{l\}) \;\approx\; \frac{\|J_l\,\varepsilon_l\|_F^2}{\|\Phi(W)\|_F^2}.$$

The **joint divergence** of a set \(S\) is, to first order,

$$D(S) \;\approx\; \frac{\Bigl\|\sum_{l\in S} J_l\,\varepsilon_l\Bigr\|_F^2}{\|\Phi(W)\|_F^2}
\;=\; \frac{1}{\|\Phi(W)\|_F^2}\Biggl[
  \underbrace{\sum_{l\in S}\|J_l\varepsilon_l\|_F^2}_{\text{sum of single-layer terms}}
  \;+\;
  \underbrace{2\!\!\sum_{l<m\in S} (J_l\varepsilon_l)^{\top}(J_m\varepsilon_m)}_{\text{cross terms}}
\Biggr].$$

### 4.2 Additivity is the vanishing of the cross terms

The reverse method's "single-layer ranking is valid" claim is precisely the statement that the
**cross terms are negligible**:

$$D(S) \;\approx\; \sum_{l\in S} \mathrm{impact}(l)
\quad\Longleftrightarrow\quad
(J_l\varepsilon_l)^{\top}(J_m\varepsilon_m) \approx 0 \ \ \forall\, l\neq m.$$

This holds in the **low-error regime**, where \(\|\delta\|\) is small enough that the second-order
term \(\frac12\sum\varepsilon_l^{\top}H_{lm}\varepsilon_m\) is small relative to the first-order
term. It **fails** once the accumulated error leaves that regime.

### 4.3 The interaction term (second-order finite difference)

The **pair interaction** between layers \(l\) and \(m\) is the difference between the joint
effect and the sum of the two single-layer effects:

$$I(l,m) \;=\; \Phi(W + \varepsilon_l e_l + \varepsilon_m e_m) \;-\; \Phi(W + \varepsilon_l e_l) \;-\; \Phi(W + \varepsilon_m e_m) \;+\; \Phi(W).$$

Substituting the Taylor expansion, the first-order terms cancel exactly, leaving

$$I(l,m) \;\approx\; \varepsilon_l^{\top} H_{lm}\, \varepsilon_m.$$

So the interaction is **the Hessian cross-block**, evaluated bilinearly on the two error vectors.
It is zero if \(\Phi\) is locally linear (additive); nonzero whenever the sampling dynamics
couple the two layers' errors.

### 4.4 Error cancellation as a negative cross term

The cross terms in §4.1 can be **negative**. When
\((J_l\varepsilon_l)^{\top}(J_m\varepsilon_m) < 0\), the errors of layers \(l\) and \(m\) partially
cancel in the final latent. This produces the counter-intuitive behavior:

- A layer can have **large single-layer impact** \(D(\{l\})\), yet **removing its error**
  (re-protecting it) makes the **total error larger**, because doing so destroys a cancellation
  with another layer's error.

Formally, let \(S\) be the quantized set. The total squared error (all norms Frobenius) is

$$\Bigl\|\sum_{l\in S} J_l\varepsilon_l\Bigr\|_F^2 \;=\; \sum_{l\in S}\|J_l\varepsilon_l\|_F^2 \;+\; 2\sum_{l<m}(J_l\varepsilon_l)^{\top}(J_m\varepsilon_m).$$

If layer \(l\) is protected (\(\varepsilon_l \to 0\)), the term
\(\|J_l\varepsilon_l\|_F^2\) disappears, but **all** cross terms involving \(l\) also disappear.
If those cross terms were sufficiently negative (cancellation), the sum **increases**:

$$\Bigl\|\sum_{m\in S\setminus\{l\}} J_m\varepsilon_m\Bigr\|_F^2 \;>\; \Bigl\|\sum_{m\in S} J_m\varepsilon_m\Bigr\|_F^2.$$

This is exactly why "protecting more top-ranked layers makes it worse" is **not a bug** but a
**necessary consequence** of negative cross terms that any single-layer ranking is structurally
unable to represent.

---

## 5. Nonlinear Error Amplification

### 5.1 Multiplicative, cumulative growth

Let \(r_k = \|\Phi_k(W+\delta) - \Phi_k(W)\|_F / \|\Phi_k(W)\|_F\) be the **relative trajectory
error** at step \(k\) (the perturbation \(\delta\) here is the full weight-space perturbation of
§4; \(r_k\) is its propagated effect on the trajectory). The measured behavior is
**multiplicative**: each step scales the error by a local amplification factor \(\rho_k\),

$$r_{k+1} \;\approx\; \rho_k\,r_k, \qquad \rho_k > 1.$$

A single layer's injected error is of order \(10^{-5}\), yet after a handful of steps the
relative error reaches order \(1\) (i.e. the two trajectories have fully diverged). This is
**exponential growth**: writing the average log-amplification as a Lyapunov-type exponent,

$$r_T \;\approx\; r_0 \prod_{k=0}^{T-1} \rho_k \;=\; r_0\, e^{\lambda T}, \qquad
\lambda \;=\; \frac{1}{T}\sum_{k=0}^{T-1}\ln \rho_k > 0.$$

Static saliency (histogram MSE, cosine, SVD) measures a **static distance in weight space** and
is completely blind to this time-evolution of the amplification rate.

### 5.2 The ~10³–10⁴ gap: second-order terms dominate the first-order sum

A single-layer impact is of order \(10^{-5}\) (median \(\sim 10^{-6}\), max \(\sim 10^{-5}\)), so
the **sum of all single-layer impacts** is of order \(10^{-4}\). Yet the **joint** divergence of
quantizing all layers is of order \(10^{-1}\text{–}10^{0}\) — a gap of roughly **three to four
orders of magnitude**. In the Taylor picture this means

$$\Bigl\|\tfrac12{\textstyle\sum_{l,m}}\varepsilon_l^{\top}H_{lm}\varepsilon_m + O(\|\delta\|^3)\Bigr\|
\;\gg\;
\Bigl\|{\textstyle\sum_l}J_l\varepsilon_l\Bigr\|,$$

i.e. the higher-order (interaction and nonlinear-amplification) terms **dominate** the first-order
sum once the total error is no longer infinitesimal. The linear (Jacobian) approximation is thus
**not merely imprecise but qualitatively wrong** at realistic quantization-error scales.

---

## 6. When Additivity Holds — and When It Breaks

### 6.1 The validity condition

Single-layer ranking is valid if and only if the first-order term dominates the second-order term:

$$\Bigl\|\sum_l J_l\varepsilon_l\Bigr\| \;\gg\; \Bigl\|\tfrac12\sum_{l,m}\varepsilon_l^{\top}H_{lm}\varepsilon_m\Bigr\|.$$

Because the first-order term scales as \(\|\delta\|\) and the second as \(\|\delta\|^2\), this
holds for **sufficiently small total perturbation** \(\|\delta\|\) — the **low-error regime**.

### 6.2 The reverse method keeps measurement inside the valid regime

The reverse method starts from the **full INT8 model, whose error is ≈ 0**, and converts layers
**one at a time in ascending impact order**. At each step the context is "the previous (still
near-perfect) configuration + one layer", so:

- every marginal measurement is taken in a context close to the reference, and
- the total accumulated error grows **as slowly as possible** (ascending order).

One further bridge is implicit and worth stating explicitly: the impacts are measured on the
**pristine (fp16/bf16) model**, while the conversion is applied to the **full-INT8 artifact**.
This transfer is valid because the INT8 reconstruction is near-perfect at the trajectory level —
\(\Phi(W_{\mathrm{int8}}) \approx \Phi(W)\) — so the divergence landscape around
\(W_{\mathrm{int8}}\) is, to first order, indistinguishable from the one around \(W\) where the
impacts were measured.

This is why the ranking remains valid across the conversion sequence — until the accumulated
error crosses the threshold where the second-order term overtakes the first.

### 6.3 The cliff: a threshold where interaction becomes dominant

As layers are added, the total error accumulates. At a critical layer count \(K^*\), the
trajectory error jumps discontinuously from "small and stable" to "collapsed" (a **cliff**). This
is the point where the accumulated error leaves the low-error regime and the higher-order
(interaction / nonlinear-amplification) terms take over. Beyond \(K^*\) the single-layer ranking
**ceases to predict** the joint effect, which is why the boundary must be measured, not
extrapolated.

---

## 7. Marginal Effects: the Correct Measurement Context

### 7.1 Definition

The **marginal effect** of layer \(l\) given a quantized set \(S\) is

$$\Delta_S(l) \;=\; D\bigl(S \cup \{l\}\bigr) \;-\; D(S)
\;=\; \frac{\bigl\|\Phi(W + \varepsilon_S + \varepsilon_l e_l) - \Phi(W + \varepsilon_S)\bigr\|_F^2}{\|\Phi(W)\|_F^2}.$$

This is the additional divergence caused by adding layer \(l\) **on top of** configuration \(S\).
The single-layer impact of §1 is the special case \(\Delta_\varnothing(l)\).

### 7.2 Why context matters

Because \(D(S)\) is **not additive** (cross terms and higher-order terms are nonzero outside the
low-error regime), \(\Delta_S(l)\) depends on \(S\). The correct rule is therefore:

> Measure each layer's effect in the context **closest to the target configuration**, not in
> isolation on the pristine model.

The reverse method operationalizes this exactly: the base \(S\) is the best (full-INT8)
configuration, each layer's \(\Delta_S(l)\) is the very operation that will actually be
performed, and layers are committed in ascending \(\Delta\). As long as \(S\) stays in the
low-error regime, \(\Delta_S(l) \approx \Delta_\varnothing(l)\), so the cheap single-pass ranking
suffices; near the boundary the ranking degrades, and the cliff is found by **measured** marginal
effects rather than by extrapolation.

The converse failure is equally instructive: a method that starts from a **bad** base (e.g. a
mostly-NVFP4 model with large accumulated error) and greedily *adds protection* (converting
layers back to higher precision) fails for the same reason. In a heavily perturbed context the
marginal effects \(\Delta_S(l)\) are dominated by interactions with the **existing** error field
rather than by each layer's intrinsic sensitivity, so a ranking measured there does not
transfer to any target configuration — improving one layer destroys error cancellations with
the surrounding error (§4.4), and the greedy sequence saturates far from the achievable
optimum. The measurement context must be close to the configuration one actually wants to
reach.

---

## 8. Shapley-Style Attribution and Interaction Quantification

### 8.1 Shapley value

Define a value function \(v(S) = -D(S)\) (quality = negative divergence; higher is better) on
subsets \(S\) of the full layer set \(N = \{1,\dots,L\}\). The **Shapley value** of layer \(l\)
is the weighted average of its marginal contributions over all subsets of the other layers:
possible subsets:

$$\varphi_l \;=\; \sum_{S \subseteq N\setminus\{l\}} \frac{|S|!\,(L-|S|-1)!}{L!}\,\Bigl[ v(S\cup\{l\}) - v(S) \Bigr].$$

The Shapley values satisfy the **efficiency** property:

$$\sum_{l=1}^{L} \varphi_l \;=\; v(N) - v(\varnothing).$$

The Shapley value is the unique attribution that distributes the **total** effect (including all
interaction) among layers in a way that is symmetric, null-player-respecting, and additive. It is
the correct notion of "importance" **when interactions are present**, precisely because it does
not assume additivity.

### 8.2 Estimating interactions by random subsets + regression

For a large \(L\), exact Shapley computation is infeasible (\(2^L\) terms). The practical recipe
uses the full layer set \(N\)'s subsets:

1. Sample random subsets \(S \subseteq N\) and measure \(v(S)\) (the divergence of that
   configuration).
2. Regress \(v(S)\) on the layer indicators \(z_l = \mathbf{1}[l \in S]\):

$$v(S) \;\approx\; \beta_0 \;+\; \sum_l \beta_l z_l \;+\; \sum_{l<m} \gamma_{lm} z_l z_m \;+\; \dots$$

Here \(\beta_l\) are **main effects** and \(\gamma_{lm}\) are **pair interaction coefficients**
(they estimate the Hessian cross-blocks \(\varepsilon_l^{\top}H_{lm}\varepsilon_m\) folded into
scalar form). The residual beyond the main effects quantifies interaction.

3. The **pair-interaction check** (§4.3) applied only to boundary-candidate layers — comparing
   \(D(\{l\}) + D(\{m\})\) against \(D(\{l,m\})\) — is an \(O(n^2)\) direct measurement of
   \(I(l,m)\) where it matters most, avoiding the \(L^2\) cost over all layers.

### 8.3 The local Jacobian as an upper bound only

The Jacobian \(J_l = \partial\Phi/\partial W_l\) gives the **infinitesimal** sensitivity, an upper
bound on linear amplification in the limit \(\|\varepsilon_l\| \to 0\). But the observed
three-to-four-order-of-magnitude gap between the first-order sum and the true joint error shows
that real quantization errors are **not infinitesimal**: the nonlinear amplification terms are
the physics of the problem. The Jacobian alone therefore cannot predict the joint effect; only
**measured marginal effects on the real trajectory** can.

---

## 9. Why this replaces static saliency

| | Conventional HSWQ (histogram MSE / cosine / SVD) | Reverse method (`diag_impact.py` + marginal effects) |
|---|---|---|
| **Quantity** | Static weight-space distance | Trajectory divergence after full propagation |
| **Captures cross terms** \(J_l\varepsilon_l \cdot J_m\varepsilon_m\) | No | Yes (implicitly, via propagation) |
| **Captures second-order / amplification** | No | Yes (the gap is measured, not assumed away) |
| **Predicts joint effect** | No | Yes in low-error regime; boundary found by measurement |
| **Validity condition** | None (always local) | Low-error regime (reverse method enforces it) |

---

## 10. Formula Index

| Formula | Section |
|--------|---------|
| \(D(S) = \|\Phi(W+\varepsilon_S)-\Phi(W)\|_F^2 / \|\Phi(W)\|_F^2\) | §2.3 |
| \(\Phi(W+\delta) = \Phi(W) + \sum_l J_l\varepsilon_l + \tfrac12\sum_{l,m}\varepsilon_l^{\top}H_{lm}\varepsilon_m + O(\|\delta\|^3)\) | §4.1 |
| \(D(S) \approx \frac{1}{\|\Phi(W)\|^2}\big[\sum_{l\in S}\|J_l\varepsilon_l\|^2 + 2\sum_{l<m}(J_l\varepsilon_l)^{\top}(J_m\varepsilon_m)\big]\) | §4.1 |
| \(I(l,m) = \Phi(W+\varepsilon_l+\varepsilon_m) - \Phi(W+\varepsilon_l) - \Phi(W+\varepsilon_m) + \Phi(W) \approx \varepsilon_l^{\top}H_{lm}\varepsilon_m\) | §4.3 |
| \(r_{k+1} \approx \rho_k r_k,\ \ r_T \approx r_0 e^{\lambda T}\) | §5.1 |
| Validity: \(\|\sum_l J_l\varepsilon_l\| \gg \|\tfrac12\sum_{l,m}\varepsilon_l^{\top}H_{lm}\varepsilon_m\|\) | §6.1 |
| \(\Delta_S(l) = D(S\cup\{l\}) - D(S)\) | §7.1 |
| \(\varphi_l = \sum_{S\subseteq N\setminus\{l\}} \frac{|S|!\,(L-|S|-1)!}{L!}[v(S\cup\{l\})-v(S)]\) | §8.1 |
| \(v(S) \approx \beta_0 + \sum_l\beta_l z_l + \sum_{l<m}\gamma_{lm}z_lz_m + \dots\) | §8.2 |

---

## 11. Summary

Single-layer quantization impact, measured by trajectory divergence, is only the **first-order**
term of a Taylor expansion of the sampling map \(\Phi\). The joint effect of quantizing multiple
layers includes **cross terms** (which can be negative, i.e. error cancellation) and
**second-and-higher-order terms** (which produce the observed orders-of-magnitude nonlinear
amplification). A per-layer scalar computed in isolation is therefore structurally incapable of
predicting the joint effect. The reverse method is correct precisely because it (a) measures in
the low-error regime where the cross terms vanish and additivity holds, (b) measures **marginal
effects in the context closest to the target configuration**, and (c) treats the boundary — the
cliff where interaction becomes dominant — as an object to be **measured**, not extrapolated.
This is the mathematical content of the statement that "layer importance is not a property of the
layer alone, but of the configuration from which it is measured."
