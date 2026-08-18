# Day 22 — Maximum Likelihood Estimation (MLE)
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Casella & Berger, *Statistical Inference* — Chapter 7; Blitzstein & Hwang Chapter 6
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why MLE is the Foundation of ML Training

Every time you train a neural network, you are (usually) performing Maximum Likelihood Estimation.

| ML Training Objective | MLE Interpretation |
|---|---|
| Minimize cross-entropy loss | MLE for Bernoulli/Categorical likelihood |
| Minimize MSE (regression) | MLE for Gaussian likelihood |
| Minimize MAE | MLE for Laplace likelihood |
| Logistic regression | MLE for Bernoulli(sigmoid(wᵀx)) |
| Poisson regression | MLE for Poisson(exp(wᵀx)) |
| VAE reconstruction loss | MLE for decoder p(x\|z) |
| Language model training | MLE via chain rule: maximize log P(w₁,...,wₙ) |
| Naive Bayes | MLE for class-conditional densities |

MLE is not just a textbook method — it **is** the training algorithm of modern ML.

---

## 2. Setup and Intuition

**Problem:** You observe data x₁, x₂, ..., xₙ. You assume they come from a parametric family:
```
xᵢ ~ f(x; θ)
```
where θ is unknown. Find the best θ.

**MLE Intuition:** Find θ that makes the observed data **most likely**.

```
θ̂_MLE = argmax_θ P(data | θ)
```

"Which parameter value would most likely produce the data I actually observed?"

---

## 3. The Likelihood Function

> **Definition:** The **likelihood function** is:
> ```
> L(θ) = L(θ; x₁,...,xₙ) = f(x₁,...,xₙ; θ)
> ```

For **i.i.d.** data:
```
L(θ) = Πᵢ f(xᵢ; θ)    [product of individual densities/PMFs]
```

**Critical distinction:**
- f(x; θ): function of x for fixed θ — this is the density/PMF
- L(θ; x): function of θ for fixed x — this is the likelihood

The likelihood is NOT a probability distribution over θ — it's just a function. (That's the Bayesian perspective, Day 24.)

### Log-Likelihood

> **Definition:** The **log-likelihood** is:
> ```
> ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ; θ)
> ```

Since log is monotonically increasing:
```
argmax_θ L(θ) = argmax_θ ℓ(θ)
```

**Always maximize log-likelihood, not likelihood.** Products become sums, which are:
- Numerically stable (no underflow from tiny products)
- Easier to differentiate
- Connected to information theory (cross-entropy)

---

## 4. Finding the MLE — The Recipe

```
Step 1: Write the likelihood L(θ) = Πᵢ f(xᵢ; θ)

Step 2: Take the log: ℓ(θ) = Σᵢ log f(xᵢ; θ)

Step 3: Differentiate and set to zero: dℓ/dθ = 0

Step 4: Solve for θ̂

Step 5: Verify it's a maximum (second derivative < 0)
```

For vector parameters θ = (θ₁,...,θₖ): set ∇_θ ℓ = 0 (gradient = 0).

---

## 5. Properties of MLE

### Property 1 — Consistency
```
θ̂_MLE →ᵖ θ_true    as n → ∞
```
MLE converges to the true parameter. (Follows from LLN.)

### Property 2 — Asymptotic Normality
```
√n(θ̂_MLE − θ) →ᵈ N(0, I(θ)⁻¹)
```
where I(θ) is the **Fisher Information**:
```
I(θ) = −E[∂²ℓ/∂θ²] = E[(∂ log f/∂θ)²]
```

MLE is approximately Normal for large n — justifying confidence intervals and hypothesis tests.

### Property 3 — Efficiency (Cramér-Rao Lower Bound)
```
Var(θ̂_MLE) → 1/[n·I(θ)]    as n → ∞
```

MLE achieves the **Cramér-Rao Lower Bound** — no unbiased estimator can have lower variance asymptotically. MLE is the most efficient estimator.

### Property 4 — Invariance
```
If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ)
```

**Example:** If μ̂ is MLE of μ for Normal, then e^μ̂ is MLE of e^μ.

### Property 5 — Bias
```
MLE can be biased in finite samples
```

**Example:** MLE for Normal variance σ̂² = (1/n)Σ(xᵢ−x̄)² is biased (divides by n, not n−1). But it's **asymptotically unbiased**.

---

## 6. Fisher Information

> **Definition:** The Fisher Information is:
> ```
> I(θ) = E[(∂ log f(X;θ)/∂θ)²] = −E[∂² log f(X;θ)/∂θ²]
> ```

**Intuition:** I(θ) measures how much information the data carries about θ. The sharper the likelihood peak, the more information.

### Cramér-Rao Lower Bound (CRLB)
```
Var(θ̂) ≥ 1/[n·I(θ)]    for any unbiased estimator θ̂
```

**ML insight:** Fisher Information appears in:
- Natural gradient descent (uses F(θ) = Fisher Information Matrix)
- Trust region methods (limit step size using Fisher)
- Neural network pruning (importance ∝ Fisher Information)
- Federated learning (Fisher-weighted aggregation)

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — MLE for Bernoulli: Binary Classification

**Problem:** A classifier makes 10 predictions. Outcomes: 1,1,0,1,0,1,1,1,0,1 (1=correct, 0=wrong).

**(a)** Write the likelihood L(p) and log-likelihood ℓ(p).
**(b)** Find the MLE p̂.
**(c)** Verify it's a maximum.
**(d)** Find the Fisher Information I(p) and CRLB.

**Solution:**

**(a)** Each Xᵢ ~ Bernoulli(p). n=10, k=7 successes (ones).

```
L(p) = Πᵢ pˣⁱ(1−p)¹⁻ˣⁱ = p⁷(1−p)³

ℓ(p) = 7 log p + 3 log(1−p)
```

**(b)** Differentiate and set to zero:
```
dℓ/dp = 7/p − 3/(1−p) = 0

7(1−p) = 3p
7 − 7p = 3p
7 = 10p

p̂ = 7/10 = 0.70
```

**MLE = sample proportion** = k/n. Always. ✓

**(c)** Second derivative:
```
d²ℓ/dp² = −7/p² − 3/(1−p)²

At p̂ = 0.7: = −7/0.49 − 3/0.09 = −14.29 − 33.33 = −47.62 < 0  ✓ [Maximum]
```

**(d)** Fisher Information for Bernoulli(p):
```
log f(x;p) = x log p + (1−x) log(1−p)

∂ log f/∂p = x/p − (1−x)/(1−p)

(∂ log f/∂p)² = [x/p − (1−x)/(1−p)]²

I(p) = E[(∂ log f/∂p)²] = Var(x/p − (1−x)/(1−p))
     = Var(x)/p² = p(1−p)/p² = (1−p)/p... [let's compute properly]

∂² log f/∂p² = −x/p² − (1−x)/(1−p)²

I(p) = −E[∂² log f/∂p²] = E[x/p² + (1−x)/(1−p)²]
     = p/p² + (1−p)/(1−p)² = 1/p + 1/(1−p) = 1/(p(1−p))
```

CRLB: Var(p̂) ≥ 1/(n·I(p)) = p(1−p)/n

For p=0.7, n=10: CRLB = 0.7×0.3/10 = 0.021

Actual Var(p̂) = p(1−p)/n = 0.021 — **MLE achieves the CRLB exactly!** ✓

---

### 🔢 Numerical 2 — MLE for Gaussian: Regression Loss

**Problem:** Regression errors ε₁,...,εₙ ~ i.i.d. N(0,σ²). You observe:
```
residuals: 0.3, −0.5, 0.2, 0.8, −0.4, 0.1, −0.2, 0.6
```

**(a)** Write the log-likelihood ℓ(σ²).
**(b)** Find the MLE σ̂².
**(c)** Prove this equals the average squared residual (connecting MLE to MSE loss).
**(d)** Is σ̂²_MLE biased?

**Solution:**

**(a)** Each εᵢ ~ N(0,σ²):
```
log f(εᵢ; σ²) = −½log(2πσ²) − εᵢ²/(2σ²)

ℓ(σ²) = Σᵢ log f(εᵢ; σ²)
       = −n/2·log(2π) − n/2·log(σ²) − 1/(2σ²)·Σᵢεᵢ²
```

**(b)** Let φ = σ². Differentiate with respect to φ:
```
dℓ/dφ = −n/(2φ) + Σεᵢ²/(2φ²) = 0

n/(2φ) = Σεᵢ²/(2φ²)
nφ = Σεᵢ²
φ̂ = Σεᵢ²/n
```

**σ̂²_MLE = (1/n)Σεᵢ²** — the average squared residual.

Computing: Σεᵢ² = 0.09+0.25+0.04+0.64+0.16+0.01+0.04+0.36 = 1.59

σ̂²_MLE = 1.59/8 = **0.199**

**(c)** The MSE loss = (1/n)Σεᵢ² = σ̂²_MLE.

**Minimizing MSE loss = Maximizing Gaussian log-likelihood.** Proved. ✓

This is the fundamental connection: regression with MSE loss is MLE under Gaussian noise assumption.

**(d)** E[σ̂²_MLE] = E[(1/n)Σεᵢ²] = (1/n)·n·σ² = σ² ... wait, that's unbiased?

Only when μ=0 is known! When μ is estimated (replaced by x̄), then:
```
E[(1/n)Σ(εᵢ−ε̄)²] = (n−1)/n·σ²  [biased]
E[(1/(n−1))Σ(εᵢ−ε̄)²] = σ²       [unbiased — use S²]
```

MLE divides by n (biased when mean is estimated), unbiased estimator divides by n−1.

---

### 🔢 Numerical 3 — MLE for Poisson: Count Data

**Problem:** Daily server errors over 7 days: 3, 1, 4, 2, 0, 2, 3.

Assume errors ~ Poisson(λ).

**(a)** Log-likelihood ℓ(λ).
**(b)** MLE λ̂.
**(c)** 95% CI for λ using asymptotic normality of MLE.
**(d)** Fisher Information I(λ).

**Solution:**

**(a)** Poisson PMF: f(x;λ) = e^{−λ}λˣ/x!

```
log f(xᵢ;λ) = −λ + xᵢ log λ − log(xᵢ!)

ℓ(λ) = Σᵢ(−λ + xᵢ log λ − log(xᵢ!))
      = −nλ + (Σxᵢ)log λ − Σlog(xᵢ!)
```

**(b)** Differentiate:
```
dℓ/dλ = −n + Σxᵢ/λ = 0

λ̂ = Σxᵢ/n = x̄
```

**MLE = sample mean** for Poisson. ✓

Computing: Σxᵢ = 3+1+4+2+0+2+3 = 15, n=7
λ̂ = 15/7 ≈ **2.143 errors/day**

**(c)** Fisher Information for Poisson(λ):
```
∂² log f/∂λ² = −x/λ²
I(λ) = −E[∂² log f/∂λ²] = E[X/λ²] = λ/λ² = 1/λ
```

Asymptotic variance of λ̂: Var(λ̂) ≈ 1/(n·I(λ)) = λ/n

Estimated SE: √(λ̂/n) = √(2.143/7) = √0.3061 = 0.5532

95% CI: λ̂ ± 1.96·SE = 2.143 ± 1.96×0.5532 = 2.143 ± 1.084 = **(1.059, 3.227)**

**(d)** I(λ) = 1/λ — as λ increases (more events), each event carries less information per unit.

---

### 🔢 Numerical 4 — MLE for Exponential: Service Time Modeling

**Problem:** API response times (ms): 120, 85, 200, 95, 150, 110, 180, 75.

Assume response time T ~ Exponential(λ).

**(a)** Log-likelihood ℓ(λ).
**(b)** MLE λ̂.
**(c)** MLE of E[T] = 1/λ (using invariance property).
**(d)** MLE of P(T > 200) = e^{−200λ}.

**Solution:**

**(a)** Exponential PDF: f(t;λ) = λe^{−λt}
```
log f(tᵢ;λ) = log λ − λtᵢ

ℓ(λ) = n log λ − λΣtᵢ
```

**(b)**
```
dℓ/dλ = n/λ − Σtᵢ = 0
λ̂ = n/Σtᵢ = 1/t̄
```

**MLE = 1/sample mean** for Exponential.

t̄ = (120+85+200+95+150+110+180+75)/8 = 1015/8 = 126.875 ms

λ̂ = 1/126.875 ≈ **0.00788 per ms**

**(c)** By MLE invariance: MLE of E[T] = 1/λ̂ = t̄ = **126.875 ms**

The MLE of the mean is just the sample mean — as expected.

**(d)** By MLE invariance: MLE of P(T>200):
```
P̂(T>200) = e^{−200λ̂} = e^{−200×0.00788} = e^{−1.576} ≈ 0.207
```

About 20.7% of requests take more than 200ms.

**ML insight:** MLE invariance is powerful — you compute MLE of λ once, then get MLEs of all functions of λ for free. The MLE of P(T>SLA_threshold) is the natural way to estimate SLA compliance from historical data.

---

### 🔢 Numerical 5 — MLE for Multivariate Gaussian (Covariance)

**Problem:** 2D data points: (1,2), (2,3), (3,1), (4,4), (2,2). Assume X ~ N(μ, Σ).

**(a)** MLE for μ.
**(b)** MLE for Σ.
**(c)** Connection to sample covariance matrix.

**Solution:**

**(a)** For multivariate Normal, log-likelihood:
```
ℓ(μ,Σ) = −n/2·log|Σ| − ½·Σᵢ(xᵢ−μ)ᵀΣ⁻¹(xᵢ−μ) + const
```

Setting ∇_μℓ = 0:
```
Σᵢ Σ⁻¹(xᵢ−μ) = 0  →  μ̂ = (1/n)Σᵢ xᵢ = x̄
```

x̄ = ((1+2+3+4+2)/5, (2+3+1+4+2)/5) = (12/5, 12/5) = **(2.4, 2.4)**

**(b)** MLE for Σ:
```
Σ̂ = (1/n)Σᵢ(xᵢ−x̄)(xᵢ−x̄)ᵀ
```

Centered data (xᵢ−x̄):
(1−2.4, 2−2.4) = (−1.4, −0.4)
(2−2.4, 3−2.4) = (−0.4, 0.6)
(3−2.4, 1−2.4) = (0.6, −1.4)
(4−2.4, 4−2.4) = (1.6, 1.6)
(2−2.4, 2−2.4) = (−0.4, −0.4)

Σ̂ = (1/5)·Σᵢ(xᵢ−x̄)(xᵢ−x̄)ᵀ:

Σ̂₁₁ = (1.96+0.16+0.36+2.56+0.16)/5 = 5.2/5 = 1.04
Σ̂₂₂ = (0.16+0.36+1.96+2.56+0.16)/5 = 5.2/5 = 1.04
Σ̂₁₂ = (0.56−0.24−0.84+2.56+0.16)/5 = 2.2/5 = 0.44

```
    ⎡1.04  0.44⎤
Σ̂ = ⎢          ⎥
    ⎣0.44  1.04⎦
```

**(c)** Σ̂_MLE divides by n (biased). Unbiased sample covariance divides by n−1:
```
S = (n/(n−1))·Σ̂_MLE = (5/4)·Σ̂_MLE
```

**ML insight:** PCA uses the sample covariance matrix. Using MLE (÷n) vs unbiased (÷n−1) makes a small difference — for large datasets it's negligible, for small n it matters.

---

### 🔢 Numerical 6 — MLE for Logistic Regression

**Problem:** Binary classification. Features X, labels Y∈{0,1}.

Model: P(Y=1|X=x) = σ(wᵀx) = 1/(1+e^{−wᵀx})

**(a)** Write the log-likelihood for n observations.
**(b)** Show this equals negative cross-entropy loss.
**(c)** Gradient of log-likelihood with respect to w.
**(d)** For 3 training points: (x=1, y=1), (x=2, y=1), (x=−1, y=0), w=0.5, compute ℓ(w).

**Solution:**

**(a)** Each observation: P(yᵢ|xᵢ;w) = σ(wᵀxᵢ)^yᵢ · (1−σ(wᵀxᵢ))^(1−yᵢ)

```
ℓ(w) = Σᵢ [yᵢ log σ(wᵀxᵢ) + (1−yᵢ)log(1−σ(wᵀxᵢ))]
```

**(b)** Cross-entropy loss:
```
L_CE = −(1/n)Σᵢ [yᵢ log σ(wᵀxᵢ) + (1−yᵢ)log(1−σ(wᵀxᵢ))]

→  ℓ(w) = −n·L_CE
```

**Maximizing log-likelihood = Minimizing cross-entropy loss.** ✓

This is why cross-entropy is the natural loss for binary classification — it IS the MLE objective for Bernoulli outcomes.

**(c)** Gradient (using ∂σ/∂z = σ(1−σ)):
```
∂ℓ/∂w = Σᵢ (yᵢ − σ(wᵀxᵢ))·xᵢ
```

This is the fundamental result: **gradient = Σ(true − predicted) × feature**. Intuitive — parameter update proportional to prediction error.

**(d)** Compute for w=0.5:

Point 1: x=1, y=1: σ(0.5×1) = σ(0.5) = 1/(1+e^{−0.5}) = 0.6225
log contribution: 1×log(0.6225) = −0.475

Point 2: x=2, y=1: σ(0.5×2) = σ(1.0) = 0.7311
log contribution: 1×log(0.7311) = −0.313

Point 3: x=−1, y=0: σ(0.5×(−1)) = σ(−0.5) = 0.3775
log contribution: 1×log(1−0.3775) = log(0.6225) = −0.475

```
ℓ(0.5) = −0.475 + (−0.313) + (−0.475) = −1.263
```

Cross-entropy loss = −ℓ(w)/n = 1.263/3 = **0.421**

---

### 🔢 Numerical 7 — MLE Properties: Asymptotic Normality in Practice

**Problem:** From Poisson data with n=50 observations, you compute λ̂=3.2.

**(a)** Approximate distribution of λ̂ (asymptotic normality).
**(b)** 95% CI for λ.
**(c)** Test H₀: λ=3.0 vs H₁: λ≠3.0.
**(d)** Power of the test if true λ=4.0.

**Solution:**

**(a)** I(λ) = 1/λ. Asymptotic variance of λ̂:
```
Var(λ̂) ≈ λ/(n) = 3.2/50 = 0.064
SE(λ̂) = √0.064 ≈ 0.2530

λ̂ ≈ N(λ, 0.064)
```

**(b)** 95% CI:
```
λ̂ ± 1.96·SE = 3.2 ± 1.96×0.2530 = 3.2 ± 0.496 = (2.704, 3.696)
```

**(c)** Under H₀: λ₀=3.0:
```
SE₀ = √(λ₀/n) = √(3.0/50) = √0.060 = 0.2449

Z = (λ̂ − λ₀)/SE₀ = (3.2 − 3.0)/0.2449 = 0.200/0.2449 = 0.817
```

p-value = 2×P(Z > 0.817) = 2×(1−0.793) = **0.414**

Fail to reject H₀. No significant evidence λ≠3.0.

**(d)** Power at λ=4.0 (true value):

Reject H₀ when |Z| > 1.96, i.e., |λ̂−3.0|/0.2449 > 1.96, i.e., λ̂ > 3.481 or λ̂ < 2.519.

Under true λ=4.0: λ̂ ≈ N(4.0, 4.0/50) = N(4.0, 0.283²)
```
P(λ̂ > 3.481) = P(Z > (3.481−4.0)/0.283) = P(Z > −1.834) = Φ(1.834) = 0.967
P(λ̂ < 2.519) ≈ 0 (extremely unlikely under λ=4)

Power ≈ 0.967 = 96.7%
```

With n=50, the test has 96.7% power to detect λ=4.0 when testing against λ=3.0.

**ML insight:** This is how you compute "how likely am I to detect a real effect of size Δ" — the power analysis for model comparison experiments.

---

## 8. MLE vs Other Estimators

| Estimator | Definition | Properties |
|---|---|---|
| **MLE** | argmax L(θ) | Consistent, efficient, asymptotically Normal, possibly biased |
| **Method of Moments** | Set E[Xᵏ] = (1/n)Σxᵢᵏ | Often unbiased, less efficient than MLE |
| **MAP** | argmax P(θ\|data) = argmax L(θ)P(θ) | Biased toward prior, doesn't maximize likelihood alone |
| **UMVUE** | Min variance unbiased | Unbiased, minimum variance — hard to find |
| **Bayes** | E[θ\|data] (posterior mean) | Minimizes expected squared error |

**In ML:**
- MLE: neural net training (no regularization)
- MAP: neural net training with L2 regularization (Gaussian prior)
- Bayes: posterior mean — computationally expensive, used in Bayesian NNs

---

## 9. Score Function and Fisher Information Matrix

For multi-parameter models θ = (θ₁,...,θₖ):

**Score function:**
```
s(θ) = ∇_θ log f(x;θ)    [gradient of log-likelihood]
```

**Fisher Information Matrix:**
```
I(θ) = E[s(θ)s(θ)ᵀ] = −E[∇²_θ log f(x;θ)]
```

**Natural gradient descent** uses the Fisher Information Matrix to precondition gradients:
```
θ ← θ + η · I(θ)⁻¹ · ∇_θ ℓ(θ)
```

This accounts for the geometry of the parameter space — equivalent to gradient descent in the space of distributions (KL-divergence metric), not Euclidean parameter space.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is MLE?" | Find θ that maximizes likelihood of observed data |
| "Why maximize log-likelihood?" | log is monotone, products → sums, numerically stable |
| "MLE for Bernoulli?" | p̂ = k/n = sample proportion |
| "MLE for Gaussian mean and variance?" | μ̂=x̄, σ̂²=(1/n)Σ(xᵢ−x̄)² |
| "Why is MLE for σ² biased?" | Divides by n not n−1; uses up df estimating μ |
| "Connection between MLE and cross-entropy?" | Cross-entropy = negative Bernoulli/categorical log-likelihood |
| "Connection between MLE and MSE?" | MSE = negative Gaussian log-likelihood (up to constant) |
| "What is the invariance property of MLE?" | MLE of g(θ) is g(MLE of θ) |
| "What is Fisher Information?" | Curvature of log-likelihood; I(p)=1/(p(1-p)) for Bernoulli |
| "What is the Cramér-Rao bound?" | Var(θ̂) ≥ 1/(nI(θ)) — MLE achieves it asymptotically |

---

## 11. Key Formulas — Cheat Sheet for Day 22

```
Likelihood:
    L(θ) = Πᵢ f(xᵢ;θ)    [i.i.d. data]

Log-likelihood:
    ℓ(θ) = Σᵢ log f(xᵢ;θ)

MLE:
    θ̂ = argmax ℓ(θ)    [set dℓ/dθ=0, solve]

KEY MLEs:
    Bernoulli(p):    p̂ = k/n = x̄
    Normal(μ,σ²):   μ̂ = x̄,  σ̂² = (1/n)Σ(xᵢ−x̄)²
    Poisson(λ):      λ̂ = x̄
    Exponential(λ):  λ̂ = 1/x̄
    Uniform(0,θ):    θ̂ = max(x₁,...,xₙ)

MLE Properties:
    Consistent:    θ̂ →ᵖ θ
    Asympt. Normal: √n(θ̂−θ) →ᵈ N(0, I(θ)⁻¹)
    Efficient:     achieves CRLB asymptotically
    Invariant:     g(θ̂) is MLE of g(θ)

Fisher Information:
    I(θ) = E[(∂ log f/∂θ)²] = −E[∂² log f/∂θ²]
    Bernoulli: I(p) = 1/(p(1−p))
    Normal(μ): I(μ) = 1/σ²
    Poisson:   I(λ) = 1/λ
    Exponential: I(λ) = 1/λ²

Cramér-Rao:
    Var(θ̂) ≥ 1/(n·I(θ))

ML Connections:
    Cross-entropy = −ℓ(w)/n  for Bernoulli likelihood
    MSE loss = −ℓ(σ²)/n + const  for Gaussian likelihood
    L2 reg = Gaussian prior → MAP not MLE
    Logistic regression gradient: Σ(yᵢ−σ(wᵀxᵢ))xᵢ
```

---

## 12. Practice Problems (Solve Before Day 23)

1. X₁,...,Xₙ ~ Uniform(0,θ). Find the MLE θ̂. Is it unbiased? (Hint: θ̂=max(X₁,...,Xₙ). E[max]=nθ/(n+1) — compute bias.)

2. For Gaussian data, prove that MLE for μ (= x̄) and MLE for σ² (= (1/n)Σ(xᵢ−x̄)²) are independent. (Hint: use the result from Day 21 that X̄ and S² are independent for Normal data.)

3. You observe n=20 Poisson counts with x̄=5.3. Compute the MLE λ̂, Fisher Information I(λ̂), 95% CI for λ, and test H₀: λ=5 at α=0.05.

4. For logistic regression on binary data:
   - Show that the gradient of the log-likelihood is Σ(yᵢ−σ(wᵀxᵢ))xᵢ
   - Show the Hessian (second derivative matrix) is −ΣP(1−P)xᵢxᵢᵀ where P=σ(wᵀxᵢ)
   - Why does the negative Hessian being positive definite guarantee a unique maximum?

5. *(Interview-level)* The Uniform(0,θ) MLE is θ̂=max(xᵢ) which is biased: E[θ̂]=nθ/(n+1). Construct an unbiased estimator based on θ̂. Compute its variance and compare to the CRLB (I(θ)=n/θ² for Uniform). Is the unbiased estimator efficient?

---

## 13. Looking Ahead

**Day 23** — **MLE for Gaussian, Bernoulli & Poisson — Deep Dives.** We apply MLE to the three most important distributions in ML — fully deriving the training objectives of logistic regression, linear regression, and count models, with complete numerical examples and all the ML connection points.

---
*End of Day 22 | Next: Day 23 — MLE Deep Dives for ML Models*
