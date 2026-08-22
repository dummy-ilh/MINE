# Day 24 — MAP Estimation & Bayesian Inference
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Bishop, *PRML* Chapter 3; Murphy, *ML: A Probabilistic Perspective* Chapter 5
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why MAP and Bayesian Inference Matter in ML

MLE (Day 22–23) finds parameters that maximize P(data|θ). But it ignores prior knowledge and can overfit with small data. MAP and Bayesian inference fix this.

| Method | Objective | ML Connection |
|---|---|---|
| **MLE** | argmax P(data\|θ) | Training without regularization |
| **MAP** | argmax P(θ\|data) = argmax P(data\|θ)P(θ) | L1/L2 regularized training |
| **Full Bayes** | Compute full P(θ\|data) | Bayesian neural networks, GPs |

| Prior | MAP Equivalent | Regularization |
|---|---|---|
| Gaussian N(0, 1/λ·I) | MAP with Gaussian prior | L2 (Ridge) |
| Laplace(0, 1/λ) | MAP with Laplace prior | L1 (Lasso) |
| Uniform | MAP = MLE | No regularization |
| Beta(α,β) | MAP for Bernoulli p | Pseudocount smoothing |

---

## 2. Bayes' Theorem for Parameters

> **Setup:** Data D = {x₁,...,xₙ}, parameters θ.
>
> ```
> P(θ|D) = P(D|θ) · P(θ) / P(D)
>
> Posterior ∝ Likelihood × Prior
> ```

**Term by term:**
```
P(D|θ)  = likelihood         [how well θ explains data]
P(θ)    = prior              [belief about θ before data]
P(D)    = marginal likelihood [normalizing constant, hard to compute]
P(θ|D)  = posterior          [updated belief after seeing data]
```

The posterior encodes everything we know about θ after observing data.

---

## 3. MAP Estimation

> **Definition:** Maximum A Posteriori (MAP) estimation:
> ```
> θ̂_MAP = argmax_θ P(θ|D)
>        = argmax_θ [log P(D|θ) + log P(θ)]
>        = argmax_θ [ℓ(θ) + log P(θ)]
> ```

MAP maximizes the **log-posterior** = log-likelihood + log-prior.

### MAP vs MLE

```
MLE: θ̂_MLE = argmax ℓ(θ)              [no prior]
MAP: θ̂_MAP = argmax [ℓ(θ) + log P(θ)] [with prior = regularizer]
```

The **log-prior acts as a regularizer**.

### Key Insight: Regularization = Prior

```
L2 regularization: minimize −ℓ(θ) + λ||θ||²
MAP with Gaussian prior P(θ) = N(0, σ²I):
    log P(θ) = −||θ||²/(2σ²) + const
    MAP: maximize ℓ(θ) − ||θ||²/(2σ²)
    = minimize −ℓ(θ) + (1/2σ²)||θ||²

Setting λ = 1/(2σ²): identical to L2 regularization ✓
```

---

## 4. Conjugate Priors — The Key to Tractable Bayesian Inference

> **Definition:** A prior P(θ) is **conjugate** to a likelihood P(D|θ) if the posterior P(θ|D) has the same functional form as the prior.

Conjugate priors make Bayesian updates analytically tractable.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli(p) | Beta(α,β) | Beta(α+k, β+n−k) |
| Gaussian(μ, σ²) | Gaussian(μ₀, σ₀²) | Gaussian (updated) |
| Poisson(λ) | Gamma(α,β) | Gamma(α+Σx, β+n) |
| Multinomial | Dirichlet | Dirichlet |
| Gaussian(μ,σ²) for σ² | Inverse-Gamma | Inverse-Gamma |

The conjugate framework is computationally elegant — posterior is in the same family as the prior, just with updated parameters.

---

## 5. Beta-Bernoulli: The Canonical Example

### Prior: Beta(α, β)

```
P(p) = p^(α−1)(1−p)^(β−1) / B(α,β)

E[p] = α/(α+β)
Mode = (α−1)/(α+β−2)    [for α,β>1]
```

**Interpretation of hyperparameters:**
- α = number of "prior successes" (pseudocounts)
- β = number of "prior failures"
- α+β = "prior sample size"

### Likelihood: Bernoulli/Binomial

n flips, k heads:
```
P(k|p) = C(n,k) p^k (1−p)^(n−k)
```

### Posterior: Beta(α+k, β+n−k)

```
P(p|k,n) ∝ p^k(1−p)^(n−k) · p^(α−1)(1−p)^(β−1)
          = p^(α+k−1)(1−p)^(β+n−k−1)
          → Beta(α+k, β+n−k)
```

**Posterior mean (Bayes estimate):**
```
E[p|data] = (α+k)/(α+β+n)
```

**MAP estimate (posterior mode):**
```
p̂_MAP = (α+k−1)/(α+β+n−2)
```

### Bayesian Update Intuition

```
Prior:     Beta(α, β)           — α successes, β failures (pseudocounts)
  + Data:  k successes, n−k failures
  = Posterior: Beta(α+k, β+n−k) — just add counts!
```

The posterior is obtained by **adding data counts to prior pseudocounts**. This is the most elegant update rule in statistics.

---

## 6. Gaussian-Gaussian: Conjugate for Mean

### Setup

Prior: μ ~ N(μ₀, σ₀²) (prior belief about the mean)
Likelihood: Xᵢ|μ ~ i.i.d. N(μ, σ²) (known σ²)

### Posterior

```
μ|data ~ N(μₙ, σₙ²)

where:
    σₙ² = 1/(1/σ₀² + n/σ²)         [posterior precision = prior + data precision]
    μₙ  = σₙ²·(μ₀/σ₀² + nx̄/σ²)    [posterior mean = weighted average]
```

**Precision form** (cleaner):

```
1/σₙ² = 1/σ₀² + n/σ²

μₙ = (μ₀/σ₀²)/(1/σₙ²) + (nx̄/σ²)/(1/σₙ²)
   = (prior precision × μ₀ + data precision × x̄) / total precision
```

**Interpretation:**

```
Posterior mean = weighted average of prior mean and sample mean
                 weighted by their respective precisions

As n → ∞:   μₙ → x̄    [data dominates prior]
As σ₀ → ∞: μₙ → x̄    [uninformative prior → MLE]
As n → 0:   μₙ → μ₀   [no data → stay at prior]
```

---

## 7. Full Bayesian Inference vs MAP

| | MAP | Full Bayes |
|---|---|---|
| **Output** | Single point estimate θ̂ | Full distribution P(θ\|D) |
| **Uncertainty** | None | Quantified via posterior |
| **Prediction** | f(x; θ̂) | E_{θ~P(θ\|D)}[f(x;θ)] |
| **Computation** | Optimization (like MLE) | Integration (hard in general) |
| **Overfitting** | Less than MLE (prior regularizes) | Least (averages over models) |
| **In ML** | L1/L2 regularization | Gaussian processes, BNNs |

**Bayesian prediction** (predictive distribution):
```
P(y*|x*, D) = ∫ P(y*|x*, θ) · P(θ|D) dθ
```

This integrates over all possible parameters, weighted by their posterior probability — **model averaging**. More robust than any single parameter estimate.

---

## 8. Posterior Predictive Distribution

For Bernoulli with Beta prior:

```
P(Xₙ₊₁=1 | data) = ∫₀¹ p · Beta(α+k, β+n−k) dp
                  = E[p|data] = (α+k)/(α+β+n)
```

The posterior predictive is simply the posterior mean — elegant.

**For Gaussian:**
```
P(x*|D) = ∫ N(x*; μ, σ²) · N(μ; μₙ, σₙ²) dμ
         = N(x*; μₙ, σ² + σₙ²)
```

The predictive distribution has additional variance σₙ² — parameter uncertainty adds to observation noise. This is **epistemic uncertainty** (from limited data) vs **aleatoric uncertainty** (irreducible noise σ²).

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — MAP for Bernoulli: From Prior to Posterior

**Problem:** A/B testing a new checkout button.

Prior belief: P(p) ~ Beta(10, 40) (prior experience: ~20% conversion rate).

New experiment: 50 users, 15 conversions.

**(a)** Prior mean and variance.
**(b)** Posterior distribution.
**(c)** MAP estimate vs MLE vs Bayes estimate.
**(d)** 95% credible interval for p.

**Solution:**

**(a)** Prior Beta(10, 40):
```
E[p] = 10/(10+40) = 10/50 = 0.20
Var(p) = 10×40/(50²×51) = 400/127500 = 0.00314
SD(p) = 0.056
```

**(b)** Data: k=15 successes, n=50.

Posterior: Beta(10+15, 40+35) = **Beta(25, 75)**

```
E[p|data] = 25/(25+75) = 25/100 = 0.250
Mode (MAP) = (25−1)/(25+75−2) = 24/98 = 0.245
```

**(c)**

| Estimate | Value | Formula |
|---|---|---|
| MLE | 15/50 = **0.300** | k/n |
| MAP | 24/98 = **0.245** | (α+k−1)/(α+β+n−2) |
| Bayes (posterior mean) | 25/100 = **0.250** | (α+k)/(α+β+n) |

MLE (0.300) > Bayes (0.250) > MAP (0.245)

The prior pulls estimates toward 0.20 (prior belief). MLE ignores this, Bayes and MAP incorporate it.

**(d)** 95% credible interval for Beta(25,75):

The 2.5th and 97.5th percentiles of Beta(25,75):

Mean=0.25, Var=25×75/(100²×101)=0.001856, SD=0.0431

Normal approximation: 0.25 ± 1.96×0.043 ≈ (0.166, 0.334)

Exact Beta quantiles: **(0.172, 0.337)**

**Interpretation:** After seeing the data, there's 95% probability the true conversion rate is between 17.2% and 33.7%.

**ML insight:** This is the right way to do A/B testing uncertainty quantification. Instead of binary "significant/not-significant," you get the full posterior distribution and can ask "what's the probability p > 0.25?" directly.

---

### 🔢 Numerical 2 — Sequential Bayesian Updating

**Problem:** Model accuracy prior: p ~ Beta(5, 5) (prior belief: ~50% accuracy, moderate confidence).

You evaluate on 3 batches:
- Batch 1: 8 correct, 2 wrong
- Batch 2: 7 correct, 3 wrong
- Batch 3: 9 correct, 1 wrong

**(a)** Show sequential updating gives the same result as batch updating.
**(b)** Track posterior mean after each batch.
**(c)** How does uncertainty change?

**Solution:**

**(a) Sequential updates:**

Start: Beta(5, 5)
After Batch 1: Beta(5+8, 5+2) = Beta(13, 7)
After Batch 2: Beta(13+7, 7+3) = Beta(20, 10)
After Batch 3: Beta(20+9, 10+1) = Beta(29, 11)

**Batch update (all at once):**
Total correct: 8+7+9=24, total wrong: 2+3+1=6
Beta(5+24, 5+6) = Beta(29, 11) ✓

**Same result! Bayesian updating is order-independent (for conjugate priors).**

**(b) Posterior means:**

| Stage | Distribution | E[p] | SD(p) |
|---|---|---|---|
| Prior | Beta(5,5) | 0.500 | 0.149 |
| After Batch 1 | Beta(13,7) | 13/20=0.650 | 0.104 |
| After Batch 2 | Beta(20,10) | 20/30=0.667 | 0.085 |
| After Batch 3 | Beta(29,11) | 29/40=0.725 | 0.070 |

**(c)** SD decreases: 0.149 → 0.104 → 0.085 → 0.070 — uncertainty reduces as data accumulates.

**ML insight:** This is **online Bayesian learning** — each new batch updates the posterior, which becomes the prior for the next batch. This is the mathematical foundation of:
- Continual learning (no catastrophic forgetting — prior preserves old knowledge)
- Active learning (query where posterior uncertainty is highest)
- Thompson sampling in multi-armed bandits

---

### 🔢 Numerical 3 — Gaussian-Gaussian Conjugate: Estimating Model Error Rate

**Problem:** Prior belief about model latency: μ ~ N(μ₀=100ms, σ₀²=400ms²).

Known measurement noise: σ²=100ms².

You observe n=5 latency measurements: 95, 102, 98, 105, 100.

**(a)** Posterior distribution of μ.
**(b)** MAP estimate and posterior mean.
**(c)** Predictive distribution for next measurement.
**(d)** Compare to MLE (ignores prior).

**Solution:**

**(a)** x̄ = (95+102+98+105+100)/5 = 500/5 = 100ms

Posterior parameters:
```
1/σₙ² = 1/σ₀² + n/σ² = 1/400 + 5/100 = 0.0025 + 0.05 = 0.0525
σₙ² = 1/0.0525 ≈ 19.05ms²
σₙ ≈ 4.37ms

μₙ = σₙ²·(μ₀/σ₀² + nx̄/σ²)
   = 19.05×(100/400 + 5×100/100)
   = 19.05×(0.25 + 5.0)
   = 19.05×5.25 ≈ 100.0ms
```

Posterior: μ|data ~ **N(100.0, 19.05)**

**(b)** For Gaussian, MAP = posterior mean = **100.0ms**

(Since Gaussian posterior is symmetric, mode = mean.)

**(c)** Predictive distribution:
```
x*|D ~ N(μₙ, σ² + σₙ²) = N(100.0, 100 + 19.05) = N(100.0, 119.05)
SD = √119.05 ≈ 10.91ms
```

Compared to measurement noise alone (SD=10ms), the predictive uncertainty is slightly larger (10.91ms) due to parameter uncertainty.

**(d)** MLE: μ̂_MLE = x̄ = 100ms

In this case, MLE and MAP agree because the data mean happens to equal the prior mean. The difference would be visible if x̄ ≠ μ₀ — MAP shrinks toward prior, MLE doesn't.

---

### 🔢 Numerical 4 — MAP = L2 Regularization: Proof by Numerics

**Problem:** Linear regression: y = wx + ε, ε ~ N(0, σ²=1).

Prior: w ~ N(0, τ²=4) (Gaussian prior, strength λ=1/(2τ²)=0.125).

Data: (x,y) = {(1,2), (2,3), (3,5)}.

**(a)** MLE for w (no regularization).
**(b)** MAP for w (with Gaussian prior).
**(c)** Show MAP = Ridge regression with λ=σ²/τ²=0.25.
**(d)** Show the prior pulls w toward zero.

**Solution:**

**(a) MLE:** w̃_MLE = Σxᵢyᵢ/Σxᵢ² (no intercept for simplicity)
= (1×2+2×3+3×5)/(1+4+9) = (2+6+15)/14 = 23/14 ≈ **1.643**

**(b) MAP:** log P(w|data) = ℓ(w) + log P(w)
= −Σ(yᵢ−wxᵢ)²/2 − w²/(2τ²)

Maximize w.r.t. w:
```
dlog P/dw = Σxᵢ(yᵢ−wxᵢ) − w/τ² = 0

Σxᵢyᵢ − wΣxᵢ² − w/τ² = 0

w(Σxᵢ² + 1/τ²) = Σxᵢyᵢ

ŵ_MAP = Σxᵢyᵢ/(Σxᵢ² + σ²/τ²)
       = 23/(14 + 1/4)
       = 23/14.25 ≈ **1.614**
```

**(c) Ridge regression** with λ=σ²/τ²=0.25:
```
ŵ_Ridge = Σxᵢyᵢ/(Σxᵢ² + λ) = 23/(14 + 0.25) = 23/14.25 ≈ 1.614 ✓
```

**Identical!** MAP with Gaussian prior = Ridge regression.

**(d)** Prior pulls toward zero:
- MLE: 1.643
- MAP/Ridge: 1.614

MAP is slightly smaller in magnitude — the prior (centered at 0) shrinks the estimate toward 0. With stronger prior (smaller τ²), shrinkage increases.

**Intuition:** Gaussian prior says "w is probably near 0." With limited data (n=3), the prior has significant influence. With n→∞, MAP→MLE.

---

### 🔢 Numerical 5 — MAP = L1 Regularization: Laplace Prior

**Problem:** Same data as Numerical 4. Now use Laplace prior: P(w) = (λ/2)exp(−λ|w|) with λ=0.5.

**(a)** MAP objective.
**(b)** Show this equals Lasso regression.
**(c)** Properties of Laplace prior vs Gaussian prior.
**(d)** Why does Lasso produce sparse solutions?

**Solution:**

**(a)** log P(w|data) = ℓ(w) + log P(w)
```
= −Σ(yᵢ−wxᵢ)²/2 + log(λ/2) − λ|w|

MAP: maximize −Σ(yᵢ−wxᵢ)²/2 − λ|w|
   = minimize Σ(yᵢ−wxᵢ)² + 2λ|w|
   = minimize RSS + 2λ||w||₁
```

**(b)** This is **Lasso** with regularization parameter 2λ. ✓

**(c)** Laplace prior vs Gaussian prior:

| | Gaussian N(0,τ²) | Laplace(0,1/λ) |
|---|---|---|
| Shape | Bell curve — smooth | Peaked at 0, heavy tails |
| MAP regularizer | L2: λ||w||² | L1: λ||w|| |
| Effect on estimates | Shrinks all weights proportionally | Sets some weights exactly to 0 |
| Sparsity | No (dense solutions) | Yes (sparse solutions) |
| Differentiability | Everywhere | Not at w=0 |

**(d)** Lasso produces sparsity because L1 is non-differentiable at zero. The subgradient at w=0 includes zero when:

|Σxᵢyᵢ| ≤ λ (the "soft thresholding" condition)

Geometrically: L1 ball has corners at the coordinate axes — the constrained optimum often hits a corner where some wⱼ=0 exactly.

**ML insight:** Lasso (L1) → feature selection (sparse w). Ridge (L2) → weight shrinkage (all features kept, small weights). Elastic Net = both. The choice is a Bayesian statement about the prior belief on weights.

---

### 🔢 Numerical 6 — Bayesian A/B Testing: Full Decision

**Problem:** Testing two models:
- Model A (control): prior Beta(20, 80) based on historical data
- Model B (treatment): prior Beta(2, 8) (less historical data)

New experiment (500 users each):
- Model A: 112 conversions
- Model B: 130 conversions

**(a)** Posterior distributions.
**(b)** P(p_B > p_A) — probability treatment is better.
**(c)** Expected conversion improvement.
**(d)** Should you deploy Model B?

**Solution:**

**(a)** Posteriors:

Model A: Beta(20+112, 80+388) = **Beta(132, 468)**
```
E[p_A|data] = 132/600 = 0.220
SD = √(132×468/(600²×601)) ≈ 0.017
```

Model B: Beta(2+130, 8+370) = **Beta(132, 378)**
```
E[p_B|data] = 132/510 = 0.259
SD = √(132×378/(510²×511)) ≈ 0.019
```

**(b)** P(p_B > p_A):

Using Normal approximation:
p_A ~ N(0.220, 0.017²), p_B ~ N(0.259, 0.019²)

(p_B − p_A) ~ N(0.039, 0.017²+0.019²) = N(0.039, 0.000650)
SD(diff) = 0.0255

P(p_B > p_A) = P(diff > 0) = P(Z > −0.039/0.0255) = P(Z > −1.529) = Φ(1.529) ≈ **93.7%**

**(c)** Expected improvement: E[p_B − p_A|data] = 0.259 − 0.220 = **+3.9 percentage points**

**(d)** Decision framework:
- P(B better) = 93.7% — strong evidence but not overwhelming
- Expected improvement = +3.9 pp
- Expected loss if B is actually worse: small (posteriors are close)

**Recommendation:** With 93.7% confidence, deploy B. But consider:
- Business cost of a wrong decision
- Run test longer if higher confidence needed (99% threshold)
- Check if 3.9 pp improvement is practically significant (business value)

**ML insight:** Bayesian A/B testing gives you P(B > A) directly — much more interpretable than "p-value < 0.05." You can make expected-value-based decisions, not just accept/reject binary decisions.

---

### 🔢 Numerical 7 — MAP for Neural Network: Weight Decay = Prior

**Problem:** A neural network with weights w ∈ ℝ¹⁰⁰⁰ trained with L2 regularization (weight decay) λ=0.001.

Training loss: L(w) = L_CE(w) + 0.001||w||²

**(a)** What Bayesian prior does this correspond to?
**(b)** Prior hyperparameters σ² in terms of λ.
**(c)** If initial weights w~N(0, 0.01), is this consistent with the prior?
**(d)** After training, why are MAP weights smaller than MLE weights?

**Solution:**

**(a)** L2 regularization = MAP with Gaussian prior:
```
Training objective: minimize L_CE(w) + λ||w||²
= minimize −ℓ(w) + λ||w||²
= maximize ℓ(w) − λ||w||²
= maximize ℓ(w) + log N(w; 0, 1/(2λ)·I)    [up to constant]
```

Prior: **w ~ N(0, 1/(2λ)·I) = N(0, 500·I)**

Each weight independently w_j ~ N(0, 500).

**(b)** σ² = 1/(2λ) = 1/(2×0.001) = **500**. SD = √500 ≈ 22.4 per weight.

**(c)** Initial weights w~N(0, 0.01): SD = 0.1.

This is MUCH tighter than the prior (SD=22.4). The initialization is not consistent with the prior — it's chosen for numerical reasons (small initial weights to avoid saturation), not to match the prior.

In practice, the prior doesn't control initialization directly — they're separate design choices.

**(d)** MAP objective = MLE objective + log prior.

Log-prior = −||w||²/(2σ²) penalizes large weights. MAP finds w that balances fitting the data and staying small.

Result: MAP weights satisfy ||w_MAP||² < ||w_MLE||² — regularization shrinks weights toward zero.

With λ=0.001 and 1000 weights: the regularization contributes λ×||w||² to the loss. For typical ||w||~1: regularization cost ≈ 0.001×1000 = 1.0 — comparable to typical CE loss values. This is a meaningful regularization amount.

**ML insight:** Weight decay = Gaussian prior. Every time you tune λ (weight decay), you're implicitly choosing the prior variance on weights: σ²=1/(2λ). The "right" λ depends on how spread out you believe the true weights should be — a probabilistic design decision.

---

## 10. Epistemic vs Aleatoric Uncertainty

A deep ML topic with roots in Bayesian inference:

```
Total uncertainty = Epistemic + Aleatoric

Epistemic (model uncertainty):
    - Due to limited data
    - Captured by posterior P(θ|D)
    - Reducible with more data
    - Var(E[Y|X,θ]) over θ~P(θ|D)

Aleatoric (data uncertainty):
    - Irreducible noise in data
    - Captured by P(Y|X,θ) for fixed θ
    - Cannot be reduced with more data
    - E[Var(Y|X,θ)] over θ~P(θ|D)

Eve's Law (Day 17): Var(Y|X) = E[Var(Y|X,θ)] + Var(E[Y|X,θ])
                             =  Aleatoric      +  Epistemic
```

**ML systems (autonomous vehicles, medical AI)** must quantify both types:
- High epistemic uncertainty → get more data or flag for human review
- High aleatoric uncertainty → the task is inherently noisy, collect more features

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is MAP estimation?" | argmax P(θ\|D) = argmax [ℓ(θ)+log P(θ)] |
| "How does MAP relate to regularization?" | log-prior = regularizer: Gaussian→L2, Laplace→L1 |
| "What is a conjugate prior?" | Prior and posterior in same distributional family |
| "Beta-Bernoulli update rule?" | Beta(α,β) + k successes, n−k failures → Beta(α+k, β+n−k) |
| "Why does Lasso produce sparse solutions?" | L1 non-differentiable at 0 → corner solutions |
| "What is the difference between MAP and full Bayes?" | MAP: point estimate. Bayes: full posterior distribution |
| "What is epistemic vs aleatoric uncertainty?" | Model uncertainty (reducible) vs data noise (irreducible) |
| "How does more data affect MAP vs prior?" | As n→∞, MAP→MLE; prior influence shrinks as 1/n |
| "What is a credible interval?" | Bayesian CI: P(θ ∈ interval\|data) = 95% — direct probability statement |

---

## 12. Key Formulas — Cheat Sheet for Day 24

```
Bayes' Theorem for Parameters:
    P(θ|D) ∝ P(D|θ) · P(θ)
    Posterior ∝ Likelihood × Prior

MAP:
    θ̂_MAP = argmax [ℓ(θ) + log P(θ)]

Prior → Regularizer:
    Gaussian N(0,τ²): log P(θ) = −||θ||²/(2τ²) → L2, λ=1/(2τ²)
    Laplace(0,1/λ):   log P(θ) = −λ||θ||₁      → L1

Beta-Bernoulli Conjugate:
    Prior: Beta(α,β)
    Data:  k successes, n−k failures
    Post:  Beta(α+k, β+n−k)
    E[p|data] = (α+k)/(α+β+n)    [Bayes estimate]
    p̂_MAP = (α+k−1)/(α+β+n−2)

Gaussian-Gaussian Conjugate (known σ²):
    Prior: μ ~ N(μ₀,σ₀²)
    Post:  μ|data ~ N(μₙ,σₙ²)
    1/σₙ² = 1/σ₀² + n/σ²
    μₙ = σₙ²(μ₀/σ₀² + nx̄/σ²)

Predictive Distribution (Gaussian):
    x*|D ~ N(μₙ, σ²+σₙ²)
    [observation noise + parameter uncertainty]

Epistemic vs Aleatoric:
    Var(Y|X) = E[Var(Y|X,θ)] + Var(E[Y|X,θ])
             = Aleatoric       + Epistemic

Weight decay → Gaussian prior:
    λ||w||² ↔ w ~ N(0, 1/(2λ)·I)
```

---

## 13. Practice Problems (Solve Before Day 25)

1. A spam classifier has prior p ~ Beta(3,7) (prior: 30% spam). You observe 20 emails, 9 spam. Find posterior, MAP, MLE, and Bayes estimate for p. Compute 90% credible interval.

2. Sequential updating: start with Beta(1,1) (uniform prior — no knowledge). Update after observing: 3 successes; then 7 more successes, 5 failures; then 2 more failures. Compute posterior mean at each stage. What does Beta(1,1) prior correspond to?

3. Gaussian-Gaussian conjugate: prior μ~N(50, 100), measurement noise σ²=25. Observe: {47, 53, 51, 49, 52}. Find posterior distribution of μ. How much does the data update the prior?

4. **Prove** that MAP with Laplace prior P(w) ∝ exp(−λ|w|) gives Lasso (L1 regularization). Show the MAP objective equals Ridge when the prior is Gaussian.

5. *(Interview-level)* You train a neural network with weight decay λ=0.01 on n=1000 samples. 
   - What Gaussian prior does this imply?
   - If you double the dataset to n=2000 but keep λ fixed, does the MAP estimate move toward or away from MLE? Why?
   - What value of λ should you use for n=2000 to maintain the same effective "prior strength" as λ=0.01 with n=1000?

---

## 14. Looking Ahead

**Day 25** — **Confidence Intervals.** The frequentist answer to uncertainty quantification — how to construct intervals that contain the true parameter with a specified probability, using t-distributions, bootstrap, and Normal approximations. We contrast these with Bayesian credible intervals (today) to clarify the fundamental difference between frequentist and Bayesian interpretations.

---
*End of Day 24 | Next: Day 25 — Confidence Intervals*
