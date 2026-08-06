# DualMonitor — Calibration Necessity Mathematical Proof

This document is a complete mathematical proof that **DualMonitor has mathematical meaning only as a moment estimator over the empirical distribution produced by calibration (the real DiT forward column)**. Without calibration, the "μ" and "E[x²]" that DualMonitor returns are not solutions to the estimation problem that either downstream — bias correction or activation-weighted reverse protection — requires, and are therefore mathematically meaningless.

The implementation under proof is `native_convert_int8_krea2_2.py` (`run_card1_calib`, `compute_int8_bias_delta`, and the sensitivity ranking path gated by `keep_sensitive`).

---

## 0. Proposition (what is to be proved)

**Proposition.**
The quantities whose values are meaningful as DualMonitor outputs are moment estimators with respect to the empirical distribution of activations obtained during calibration. Without calibration (the real DiT forward column), that empirical distribution is not defined, so the "μ" and "E[x²]" returned by DualMonitor are, for both bias correction and activation-weighted error, **not** solutions to the estimation problem that the implementation requires, and are therefore mathematically meaningless.

---

## 1. Fixing notation

Let the input activation of layer $L$ (`Linear` or `Conv2d`), at the $n$-th forward during calibration, be the tensor

$$
X^{(n)} \in \mathbb{R}^{B_n \times \cdots \times C_{\mathrm{in}}}
$$

(the implementation averages every dimension except the channel dimension and reduces to a $C_{\mathrm{in}}$-dimensional vector).

We denote the same reduction as the implementation by $\mathrm{Reduce}$:

- Conv2d (NCHW): $\mathrm{Reduce} =$ mean over axes $(0,2,3)$
- Linear (last axis is input channel): $\mathrm{Reduce} =$ mean over all axes except the last

For channel $j = 1,\ldots,C_{\mathrm{in}}$, a single forward gives

$$
\begin{aligned}
a_j^{(n)} &= \bigl(\mathrm{Reduce}(X^{(n)})\bigr)_j,\\
s_j^{(n)} &= \bigl(\mathrm{Reduce}((X^{(n)})^{\odot 2})\bigr)_j.
\end{aligned}
$$

DualMonitor returns the online averages

$$
\begin{aligned}
\hat\mu_j^{(N)} &= \frac{1}{N}\sum_{n=1}^{N} a_j^{(n)},\\
\widehat{(x^2)}_j^{(N)} &= \frac{1}{N}\sum_{n=1}^{N} s_j^{(n)},
\end{aligned}
$$

(which are exactly `channel_act_mean` / `channel_act_sq_mean` in the code). Here $N$ is the number of forward passes during calibration (proportional to number of samples × number of steps).

Let the weight be $W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}$ (Conv is 4D, but the bias delta uses the isomorphic form reduced onto the input-channel dimension), the dequantized weight after quantization be $W_q$, and the error be

$$
E := W_q - W.
$$

---

## 2. The probabilistic object that DualMonitor estimates (definition of the measure)

Calibration is the procedure that repeatedly executes the deterministic map of the DiT

$$
\mathcal{F}_{W}:\ (x_0,\, t,\, c(p)) \mapsto \text{per-layer input } X
$$

under a prompt set $\{p_i\}_{i=1}^{M}$, random seeds, and a timestep sequence $\{t_k\}_{k=1}^{T}$ ($c(p)$ is the real CLIP-derived context).

This determines a finite set

$$
\mathcal{S}_{\mathrm{calib}}
= \bigl\{ X^{(n)} \bigm| n=1,\ldots,N \bigr\}
$$

and hence an empirical measure

$$
\hat{\mathbb{P}}_N
= \frac{1}{N}\sum_{n=1}^{N} \delta_{X^{(n)}}.
$$

DualMonitor's $\hat\mu$ and $\widehat{x^2}$ are precisely

$$
\begin{aligned}
\hat\mu_j
&= \mathbb{E}_{\hat{\mathbb{P}}_N}\bigl[(\mathrm{Reduce}(X))_j\bigr],\\
\widehat{(x^2)}_j
&= \mathbb{E}_{\hat{\mathbb{P}}_N}\bigl[(\mathrm{Reduce}(X^{\odot 2}))_j\bigr].
\end{aligned}
$$

**Lemma 2.1 (dependence on the measure).**
$\hat{\mathbb{P}}_N$ cannot be defined without $\mathcal{S}_{\mathrm{calib}}$. Hence, without calibration, the expectation operator $\mathbb{E}_{\hat{\mathbb{P}}_N}$ itself is undefined.

*Proof.* Immediate from the definition: the empirical measure is an average of point masses; if the point set is empty (or not generated), the measure does not exist. □

---

## 3. The mathematical object that bias correction requires from DualMonitor

The implementation's Card 1 computes (for `Linear`)

$$
\Delta b = E\,\hat\mu = (W_q - W)\,\hat\mu
$$

and performs $\mathrm{bias} \leftarrow \mathrm{bias} - \Delta b$ (the sign coincides with the code's `-(W_q-W)@μ`).

### 3.1 The ideal quantity (what we want to approximate)

Writing the linear part of the layer as $y = Wx + b$ and the post-quantization version as $y_q = W_q x + b$, for a fixed input $x$ the output difference is

$$
y_q - y = Ex.
$$

When the input is a random variable $X\sim\mathbb{P}$, the expected output error is

$$
\mathbb{E}[y_q - y] = E\,\mathbb{E}[X]
$$

(by linearity). Therefore **what bias correction aims at is**

$$
\Delta b^\star(\mathbb{P}) := E\,\mu,\qquad \mu := \mathbb{E}_{\mathbb{P}}[X]
$$

(in the channel-reduced isomorphic form), and the implementation replaces $\mathbb{P}$ by $\hat{\mathbb{P}}_N$:

$$
\Delta b(\hat{\mathbb{P}}_N) = E\,\hat\mu.
$$

### 3.2 Proof that calibration-less DualMonitor is meaningless (bias)

**Theorem 3.1.**
Any vector $\tilde\mu\in\mathbb{R}^{C_{\mathrm{in}}}$ obtained from DualMonitor without running calibration, used to form $\Delta b = E\tilde\mu$, cannot in general be justified as an estimator of $\Delta b^\star(\mathbb{P})$. In particular, when $\tilde\mu$ is a constant chosen independently of $\hat{\mathbb{P}}_N$ (zero, uniform, uninitialized, or an empty shell that is "DualMonitor ON merely because a path exists"), the error can be arbitrarily large.

*Proof.*

(i) **Formulation of the estimation problem.**
The true object is $\mu(\mathbb{P})=\mathbb{E}_{\mathbb{P}}[X]$. A necessary condition for an estimator to be meaningful is that the data come from $\mathbb{P}$ (or its approximation $\hat{\mathbb{P}}_N$). Without calibration, $\hat{\mathbb{P}}_N$ is undefined (Lemma 2.1), so the estimator $\hat\mu=\mathbb{E}_{\hat{\mathbb{P}}_N}[X]$ does not even have a domain.

(ii) **Difference from an arbitrary $\tilde\mu$.**

$$
\bigl\|E\tilde\mu - E\mu\bigr\|
= \bigl\|E(\tilde\mu-\mu)\bigr\|
\le \|E\|_{\mathrm{op}}\,\|\tilde\mu-\mu\|_2.
$$

If $\tilde\mu$ is unrelated to $\mu$, the right-hand side $\|\tilde\mu-\mu\|_2$ has no lower bound (e.g. if $\tilde\mu=0$, the error is $\|E\mu\|$, which can be arbitrarily large if the true $\mu$ is large). Hence any $\tilde\mu$ returned by "calibration-less DualMonitor" carries no convergence guarantee toward $\Delta b^\star$.

(iii) **Empty DualMonitor (count = 0 / mean = None).**
In the implementation, when `act_mean is None`, no delta is computed. This is the correct treatment of "no statistics", and the code itself acknowledges that **even if one pretends to have "run DualMonitor" without calibration, the estimate of $\mu$ does not exist**. Calling something whose target $\mu(\mathbb{P})$ is not estimated "DualMonitor ran" is therefore meaningless.

(iv) **Conclusion.**
The mathematical object of bias correction is $\mathbb{E}_{\mathbb{P}}[X]$. Calibration is the only procedure that generates an empirical approximation $\hat{\mathbb{P}}_N$ of $\mathbb{P}$. Calibration-less DualMonitor does not generate $\hat{\mathbb{P}}_N$, hence does not yield an estimate of the $\mu$ required by bias correction, and is therefore meaningless. □

---

## 4. The mathematical object that reverse protection requires from DualMonitor

Instead of the relative error

$$
r = \frac{\|E\|_F}{\|W\|_F}
$$

the implementation uses, with $\sigma_j=\sqrt{\widehat{(x^2)}_j}$,

$$
r_w
= \frac{\|E\,\mathrm{diag}(\sigma)\|_F}{\|W\,\mathrm{diag}(\sigma)\|_F}
$$

(for Conv, $\sigma$ is broadcast along the channel axis).

### 4.1 Probabilistic meaning of the weighted norm

When the mean-square of channel $j$ of input $X$ is $m_j=\mathbb{E}[X_j^2]$, $\sigma_j=\sqrt{m_j}$ is "the RMS of that channel". Multiplying the weight error of column $j$ by $\sigma_j$ yields the weighted Frobenius norm that **penalizes quantization error more strongly on highly activated channels**:

$$
\|E\|_{F,\sigma}^2
= \sum_{i,j} E_{ij}^2\,\sigma_j^2
= \sum_{i,j} E_{ij}^2\,\mathbb{E}[X_j^2]
$$

(even in the non-independent case, the diagonal-weight definition is identical).

The ideal quantity is

$$
r_w^\star(\mathbb{P})
= \frac{\|E\|_{F,\sigma(\mathbb{P})}}{\|W\|_{F,\sigma(\mathbb{P})}},
\qquad
\sigma_j(\mathbb{P})=\sqrt{\mathbb{E}_{\mathbb{P}}[X_j^2]}.
$$

The implementation replaces $\mathbb{P}$ by $\hat{\mathbb{P}}_N$.

### 4.2 Proof that calibration-less is meaningless (protect)

**Theorem 4.1.**
When $\widehat{x^2}$ cannot be obtained from $\hat{\mathbb{P}}_N$, the activation-weighted ranking $r_w$ is undefined. A DualMonitor filled with placeholders (uniform weights, zeros, arbitrary constants) generally does not agree with the layer ordering induced by $r_w^\star(\mathbb{P})$.

*Proof.*

(i) **Domain.**
$r_w$ depends on the map $\sigma:\mathbb{P}\mapsto(\sqrt{\mathbb{E}[X_j^2]})_j$. Without calibration, $\hat{\mathbb{P}}_N$ does not exist, so $\sigma(\hat{\mathbb{P}}_N)$ is undefined (Lemma 2.1).

(ii) **Difference from uniform weights (essentially Frobenius).**
When $\sigma_j\equiv 1$ (or under the no-DualMonitor fallback), $r_w=r$ (the ordinary relative Frobenius). On the other hand, when the true $\sigma(\mathbb{P})$ differs greatly across channels, the two semi-norms are equivalent but **the orderings they induce do not coincide**.

Concrete example (2 layers, 2 channels, scalarized):
Layer A: $E^A=(1,0)$, $W^A=(1,1)$; Layer B: $E^B=(0,1)$, $W^B=(1,1)$.
The Frobenius relative errors of the two layers can be made equal. But with $\sigma=(10,0.1)$,

$$
\|E^A\|_{F,\sigma}=10,\quad \|E^B\|_{F,\sigma}=0.1
$$

so the ordering is reversed. Hence **DualMonitor / ranking without activation moments is not a solution to the ordering problem that protect tries to optimize**.

(iii) **Arbitrary fake $\tilde\sigma$.**
As in Theorem 3.1,

$$
\bigl|\,r_w(\tilde\sigma)-r_w^\star(\mathbb{P})\,\bigr|
$$

has no general upper bound; a fake DualMonitor does not estimate the mathematical object of protect.

(iv) **Conclusion.**
The weighted error of reverse protection requires $\mathbb{E}_{\mathbb{P}}[X^{\odot 2}]$. That is provided only by DualMonitor over $\hat{\mathbb{P}}_N$ that calibration generates. Calibration-less DualMonitor is meaningless. □

---

## 5. "A DualMonitor class exists" ≠ "DualMonitor has meaning"

**Definition (DualMonitor in the implementation).**
The object is the pair $(\hat\mu,\widehat{x^2})$, which is a function of $\hat{\mathbb{P}}_N$.

**Definition (calibration).**
The procedure that generates $\mathcal{S}_{\mathrm{calib}}$ and determines $\hat{\mathbb{P}}_N$.

Hence the identity

$$
\text{DualMonitor's mathematical content}
= f(\hat{\mathbb{P}}_N)
= f\circ\mathrm{Calib}
$$

holds. Removing $\mathrm{Calib}$ from the composition $f\circ\mathrm{Calib}$ empties the domain of $f$, and the value is not a solution of any estimation problem.

**Corollary 5.1.**
Constructions that talk as if DualMonitor is established **without running the calibration forward column** — such as "DualMonitor is ON because a path exists", or "DualMonitor independent of bias" — amount to calling $f$ outside its domain and are mathematically meaningless (Theorems 3.1, 4.1).

**Corollary 5.2.**
The necessary and sufficient condition for "calibration-less DualMonitor" to be meaningful is to specify another measure $\mathbb{P}'$ explicitly and define $\mu(\mathbb{P}')$ and $\mathbb{E}_{\mathbb{P}'}[X^2]$ under it. The implementation provides no $\mathbb{P}'$ other than $\hat{\mathbb{P}}_N$ (the calibration empirical measure). Therefore **for the implementation's DualMonitor, calibration-less is impossible and meaningless**.

---

## 6. Convergence viewpoint (why "running calibration" is correct as estimation)

Treating each $X^{(n)}$ of calibration as a realization of approximately identically distributed activations (even if not strictly i.i.d., the argument as an empirical average holds), the law of large numbers gives

$$
\hat\mu^{(N)}\xrightarrow[N\to\infty]{\mathbb{P}} \mu,\qquad
\widehat{(x^2)}^{(N)}\xrightarrow[N\to\infty]{\mathbb{P}} \mathbb{E}[X^{\odot 2}]
$$

as expected. This is the reason DualMonitor can be called an **estimator**.

Without calibration ($N=0$), the numerator and denominator of the empirical average do not exist, and the premise of limit theory (a sample sequence) is absent. Hence, even if an object named "DualMonitor" is placed, it is not an estimator but an empty symbol.

---

## 7. Summary (consequences of the proof)

| Downstream | True quantity required | Role of DualMonitor | Without calibration |
|------------|------------------------|---------------------|---------------------|
| Bias correction | $\mu=\mathbb{E}_{\mathbb{P}}[X]$ | $\hat\mu=\mathbb{E}_{\hat{\mathbb{P}}_N}[X]$ | $\hat{\mathbb{P}}_N$ undefined → $\hat\mu$ invalid → $\Delta b$ meaningless (Theorem 3.1) |
| Reverse protection | $\sigma_j=\sqrt{\mathbb{E}_{\mathbb{P}}[X_j^2]}$ | $\sigma$ from $\widehat{x^2}$ | Likewise undefined / fake $\sigma$ breaks the ordering (Theorem 4.1) |

**Final proposition (restated, proved).**
The mathematical content of DualMonitor is a moment over the calibration empirical measure $\hat{\mathbb{P}}_N$. Because calibration-less DualMonitor does not generate that measure, it provides no solution to the estimation problem for either bias correction or activation-weighted reverse protection, and is therefore meaningless.

---

## 8. Verification against the current implementation

This section fixes the proof to specific line ranges of `native_convert_int8_krea2_2.py`. All judgements below are for the current HEAD (commit `2422abf` — `fix: keep_sensitive forces DualMonitor calib gate and drops Frobenius escape`); the gate and ranking behaviour were hardened relative to earlier revisions.

| # | Mathematical claim | Implementation reference | Verdict |
|---|--------------------|--------------------------|---------|
| 1 | DualMonitor online mean: $\hat\mu_j^{(N)}=\frac{1}{N}\sum_{n=1}^{N}a_j^{(n)}$, $\widehat{(x^2)}_j^{(N)}=\frac{1}{N}\sum_{n=1}^{N}s_j^{(n)}$ | `DualMonitor.update`, lines **146–154** (online mean of `current_act` / `current_sq`, with per-sample reduction `current_act = inp.mean(reduce_dims)` and `current_sq = (inp**2).mean(reduce_dims)` at lines **139–140**) | ✓ |
| 2 | $\hat\mu,\widehat{x^2}$ are moments of the calibration empirical measure $\hat{\mathbb{P}}_N$ | Two-stage averaging (per-sample spatial/token mean → inter-sample mean). When the per-sample element count $P_n$ is constant this equals the moment under $\hat{\mathbb{P}}_{NP}$; when $P_n$ varies it remains a consistent estimator but is not literally $\hat{\mathbb{P}}_{NP}$. **The conclusion "undefined without calibration" is unaffected.** | ✓ (note attached) |
| 3 | $\Delta b^\star = -(W_q-W)\,\mathbb{E}_{\mathbb{P}}[X]$; implementation computes $(W_q-W)\hat\mu$ and stores $-\delta$ | `compute_int8_bias_delta`, lines **167–181** (`err = weight_dq - weight_fp`, `delta = err @ mu` for Linear; `(err * mu.view(1,-1,1,1)).sum(dim=(1,2,3))` for Conv2d); applied as `bias_corr_pending[...] = -delta` at line **1087**; header docstring at line **168** matches | ✓ |
| 4 | Calibration-less → $\hat\mu$ undefined → $\Delta b$ cannot be formed | Lines **1079–1080**: `if act_mean is None: bias_corr_skipped_no_act += 1` (delta is computed only when `act_mean` exists) | ✓ |
| 5 | $r_w = \|E\,\mathrm{diag}(\sigma)\|_F / \|W\,\mathrm{diag}(\sigma)\|_F$ with $\sigma_j=\sqrt{\widehat{(x^2)}_j}$ | Lines **1044–1060**: `err = w_fp - weight_dq`; `act_scale = act_sq.sqrt()`; Linear: `weighted_err = err * act_scale.unsqueeze(0)`; Conv2d: `weighted_err = err * act_scale.view(1,-1,1,1)`; `rel_err = ‖weighted_err‖ / ‖weighted_base‖` | ✓ |
| 6 | Without $\widehat{x^2}$, $r_w$ is undefined; there is no Frobenius fallback for `keep_sensitive` | Lines **1061–1063**: `elif use_keep_sensitive: pass` (explicitly drops Frobenius escape); RuntimeError at lines **1090–1095** if `use_keep_sensitive and not layer_quant_errors` | ✓ |
| 7 | DualMonitor is a function of $\hat{\mathbb{P}}_N$; it must not be invoked without calibration | Lines **874–886**: `run_dual_monitor = use_keep_sensitive or use_bias or have_calib_paths`; `ValueError` if `run_dual_monitor and not have_calib_paths` — i.e. `keep_sensitive` alone forces calibration, with no calibration-less DualMonitor path | ✓ |
| 8 | LLN makes DualMonitor an estimator as $N\to\infty$ | Application of the standard theorem; with $N=0$ the empirical average is undefined | ✓ |

### Single precision note (not a correction)

For item 2: the implementation computes the moment as the inter-sample mean of the per-sample spatial/token mean $s^{(n)}$, i.e. $\hat\mu_j^{(N)}=\frac{1}{N}\sum_n s_j^{(n)}$, rather than over the (sample, position) empirical measure $\hat{\mathbb{P}}_{NP}$. When the per-sample element count $P_n$ is constant the two coincide; when $P_n$ varies the implementation's quantity is still a consistent estimator of $\mathbb{E}[X_j]$ but is not literally $\mathbb{E}_{\hat{\mathbb{P}}_{NP}}[X_j]$. This does not affect any of the propositions above — in particular, **without calibration ($N=0$) the quantity remains undefined**, so the proof's central claim is intact.

---

## 9. Conclusion

- The proof in sections 0–7 is mathematically self-contained and correct.
- It agrees, line by line, with the current implementation of `native_convert_int8_krea2_2.py` (HEAD `2422abf`).
- The hardening introduced in `2422abf` (`run_dual_monitor = use_keep_sensitive or use_bias or have_calib_paths`; `ValueError` on missing paths; `pass` instead of Frobenius fallback; `RuntimeError` when `keep_sensitive` has no activation-weighted errors) makes the implementation **enforce** the proposition: there is no code path where DualMonitor is invoked, or where `keep_sensitive` falls back to a Frobenius ranking, without calibration having generated $\hat{\mathbb{P}}_N$.
- Therefore the mathematical statement "calibration-less DualMonitor is meaningless" is not only true in theory — it is also the contract the implementation now enforces.
