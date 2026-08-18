# Day 12 — The Normal / Gaussian Distribution
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 5 (Section 5.5)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why the Normal Distribution is the Most Important in ML

If you could only know one distribution, it would be the Normal. It appears everywhere — not by coincidence, but by mathematical necessity (the Central Limit Theorem, Day 20).

| ML Context | Normal Distribution |
|---|---|
| Weight initialization | W ~ N(0, σ²) — Xavier/He init |
| Gradient noise in SGD | ∇L ~ N(0, σ²I) approximately |
| Gaussian Naive Bayes | P(xᵢ\|class) ~ N(μₖ, σₖ²) |
| Linear regression errors | ε ~ N(0, σ²) — core assumption |
| Variational Autoencoders | Latent space z ~ N(0, I) |
| Gaussian Processes | Function priors |
| Batch normalization | Forces activations toward N(0,1) |
| Confidence intervals | Built on Normal approximation |
| L2 regularization | Equivalent to Gaussian prior on weights |
| Natural language embeddings | Often approximately Gaussian |

The Normal is not just common — it is the **default assumption** in statistics and ML. Understanding it deeply is non-negotiable.

---

## 2. Definition and PDF

> **Definition:** X ~ Normal(μ, σ²) or X ~ N(μ, σ²) if:
> ```
>           1              (x−μ)²
> f(x) = ———————— exp(− —————————)     −∞ < x < ∞
>         σ√(2π)           2σ²
> ```

### Parameters

```
μ  = mean (location parameter)       −∞ < μ < ∞
σ² = variance (scale parameter)      σ² > 0
σ  = standard deviation
```

### Formula Breakdown — Term by Term

```
           1
    ————————————        normalizing constant — ensures PDF integrates to 1
      σ√(2π)

         (x−μ)²
    exp(− ——————)       bell curve shape — symmetric around μ
           2σ²          σ controls width: larger σ → flatter, wider bell
```

### Verify Normalization (the Gaussian Integral)

```
∫₋∞^∞ exp(−x²/2) dx = √(2π)
```

This is one of the most beautiful results in mathematics. It's proven using the polar coordinates trick:
```
[∫exp(−x²/2)dx]² = ∫∫exp(−(x²+y²)/2)dxdy = ∫₀^∞ 2πr·exp(−r²/2)dr = 2π
→ ∫exp(−x²/2)dx = √(2π)  ✓
```

---

## 3. Parameters: Mean and Variance

```
E[X] = μ
Var(X) = σ²
SD(X) = σ
```

**Effect of parameters:**

| Parameter change | Effect on distribution |
|---|---|
| Increase μ | Shifts bell curve right |
| Decrease μ | Shifts bell curve left |
| Increase σ | Wider, flatter bell |
| Decrease σ | Narrower, taller bell |
| σ → 0 | Approaches point mass at μ (Dirac delta) |
| σ → ∞ | Completely flat (uniform over ℝ) |

---

## 4. The Standard Normal — Z ~ N(0, 1)

> **Definition:** Z ~ N(0,1) is the **standard normal** — mean 0, variance 1.

```
         1
φ(z) = ——— exp(−z²/2)        [PDF, called "phi"]
        √(2π)

Φ(z) = P(Z ≤ z)              [CDF, called "Phi" — no closed form]
```

### Standardization (Z-score)

If X ~ N(μ, σ²), then:
```
Z = (X − μ)/σ ~ N(0, 1)
```

This is the most used transformation in all of statistics. It converts any Normal to the standard Normal, allowing use of standard tables.

### Key Standard Normal Values (Memorize These)

```
Φ(0)    = 0.500    P(Z ≤ 0)   = 50%
Φ(1)    = 0.841    P(Z ≤ 1)   = 84.1%
Φ(1.645)= 0.950    P(Z ≤ 1.645) = 95%    [one-sided 95%]
Φ(1.96) = 0.975    P(Z ≤ 1.96) = 97.5%   [two-sided 95%]
Φ(2)    = 0.977    P(Z ≤ 2)   = 97.7%
Φ(2.576)= 0.995    P(Z ≤ 2.576) = 99.5%  [two-sided 99%]
Φ(3)    = 0.9987   P(Z ≤ 3)   = 99.87%
```

### Symmetry of Standard Normal

```
Φ(−z) = 1 − Φ(z)
P(−z ≤ Z ≤ z) = 2Φ(z) − 1
```

---

## 5. The 68-95-99.7 Rule (Empirical Rule)

For X ~ N(μ, σ²):

```
P(μ − σ  ≤ X ≤ μ + σ)  = 0.6827 ≈ 68%     [within 1 SD]
P(μ − 2σ ≤ X ≤ μ + 2σ) = 0.9545 ≈ 95%     [within 2 SD]
P(μ − 3σ ≤ X ≤ μ + 3σ) = 0.9973 ≈ 99.7%  [within 3 SD]
```

**Proof for 1 SD:**
```
P(μ−σ ≤ X ≤ μ+σ) = P(−1 ≤ Z ≤ 1) = 2Φ(1) − 1 = 2(0.841) − 1 = 0.682
```

**ML use:**
- Outlier detection: flag points > 3σ from mean
- Batch normalization: ensures activations mostly within a few SDs
- Anomaly detection: score = (x − μ)/σ, flag if |score| > 3

---

## 6. Properties of the Normal Distribution

### Property 1 — Linear Transformations

If X ~ N(μ, σ²) and Y = aX + b:
```
Y ~ N(aμ + b, a²σ²)
```

**Proof:**
```
E[Y] = aE[X] + b = aμ + b
Var(Y) = a²Var(X) = a²σ²
```
And linear transformations of Normals are Normal (closed under linear ops).

### Property 2 — Sum of Independent Normals

If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²) are **independent**:
```
X + Y ~ N(μ₁ + μ₂, σ₁² + σ₂²)
```

**Variances add. Standard deviations do NOT add.**

```
SD(X+Y) = √(σ₁² + σ₂²) ≠ σ₁ + σ₂
```

### Property 3 — Standard Normal Symmetry

```
If Z ~ N(0,1), then −Z ~ N(0,1)
```

### Property 4 — Zero Covariance + Normal = Independence

For jointly Normal random variables ONLY:
```
Cov(X,Y) = 0  ⟺  X and Y are independent
```

**Warning:** This does NOT hold for non-Normal distributions. Zero covariance implies independence ONLY for jointly Gaussian random variables.

### Property 5 — Normal is the Maximum Entropy Distribution

Among all distributions with fixed mean μ and variance σ², the Normal has the **highest entropy** (is the "most random" / least informative). This is why it's the default prior when you only know mean and variance.

---

## 7. Log-Normal Distribution

If X ~ N(μ, σ²), then Y = eˣ follows a **log-normal** distribution.

Equivalently, Y is log-normal if log(Y) ~ Normal.

```
E[Y] = exp(μ + σ²/2)
Var(Y) = (exp(σ²) − 1)·exp(2μ + σ²)
```

**ML uses:**
- Income distributions (right-skewed, positive)
- Stock prices (log-returns are Normal)
- Training time / loss values (often log-normal)
- Word frequencies (approximately log-normal in large corpora)

**Rule:** When data is positive and right-skewed, try log-transforming first. If log(data) looks Normal, use log-normal model.

---

## 8. Normal Distribution and MLE

**Setup:** Observe x₁, x₂, ..., xₙ i.i.d. from N(μ, σ²). Find MLE of μ and σ².

**Log-likelihood:**
```
ℓ(μ, σ²) = Σᵢ log f(xᵢ)
           = Σᵢ [−log(σ√2π) − (xᵢ−μ)²/(2σ²)]
           = −n/2 log(2πσ²) − 1/(2σ²) Σᵢ(xᵢ−μ)²
```

**MLE for μ** (set ∂ℓ/∂μ = 0):
```
∂ℓ/∂μ = 1/σ² Σᵢ(xᵢ−μ) = 0
→ μ̂_MLE = (1/n)Σᵢ xᵢ = x̄    [sample mean]
```

**MLE for σ²** (set ∂ℓ/∂σ² = 0):
```
∂ℓ/∂(σ²) = −n/(2σ²) + Σᵢ(xᵢ−μ)²/(2σ⁴) = 0
→ σ̂²_MLE = (1/n)Σᵢ(xᵢ−x̄)²    [biased sample variance]
```

**Note:** MLE gives biased variance (divides by n, not n−1). Unbiased estimator divides by n−1 (Day 22).

**ML insight:**
- MLE of μ = sample mean → minimizing MSE is equivalent to MLE under Gaussian noise
- L2 loss (MSE) = Gaussian likelihood → **assuming Gaussian errors justifies MSE loss**
- L1 loss = Laplacian likelihood → **assuming Laplacian errors justifies MAE loss**

---

## 9. Normal Distribution and L2 Regularization

**MAP estimation** with Gaussian prior on weights w ~ N(0, 1/λ·I):

```
w_MAP = argmax P(w|data) = argmax [log P(data|w) + log P(w)]
      = argmax [ℓ(w) − λ/2 ||w||²]
      = argmin [−ℓ(w) + λ/2 ||w||²]
```

This is exactly **L2 regularization (Ridge)**. The Gaussian prior on weights is what L2 regularization encodes.

**Similarly:** L1 regularization (Lasso) = Laplacian prior on weights.

---

## 10. Worked Numericals

---

### 🔢 Numerical 1 — Standard Normal Computations

**Problem:** Z ~ N(0,1). Find:
**(a)** P(Z ≤ 1.5)
**(b)** P(Z > 1.5)
**(c)** P(−1 ≤ Z ≤ 2)
**(d)** P(|Z| > 2.5)
**(e)** The value z* such that P(Z ≤ z*) = 0.90

**Solution:**

**(a)**
```
P(Z ≤ 1.5) = Φ(1.5) = 0.9332
```

**(b)**
```
P(Z > 1.5) = 1 − Φ(1.5) = 1 − 0.9332 = 0.0668
```

**(c)**
```
P(−1 ≤ Z ≤ 2) = Φ(2) − Φ(−1)
              = Φ(2) − (1 − Φ(1))
              = 0.9772 − (1 − 0.8413)
              = 0.9772 − 0.1587 = 0.8185
```

**(d)**
```
P(|Z| > 2.5) = P(Z > 2.5) + P(Z < −2.5)
             = 2(1 − Φ(2.5))
             = 2(1 − 0.9938) = 2(0.0062) = 0.0124
```

Only 1.24% of standard normal values fall more than 2.5 SDs from the mean.

**(e)** Find z*: Φ(z*) = 0.90 → **z* = 1.282** (90th percentile of N(0,1))

---

### 🔢 Numerical 2 — Converting to Z-scores

**Problem:** Model accuracy X ~ N(μ=0.82, σ²=0.0016) (σ=0.04) across different random seeds.

**(a)** P(accuracy > 0.90) — probability of "great" run
**(b)** P(accuracy < 0.75) — probability of "bad" run
**(c)** P(0.78 ≤ accuracy ≤ 0.86) — probability of "typical" run
**(d)** Find the accuracy threshold beaten by the top 5% of runs

**Solution:**

Standardize: Z = (X − 0.82)/0.04

**(a)**
```
P(X > 0.90) = P(Z > (0.90−0.82)/0.04) = P(Z > 2.0)
            = 1 − Φ(2.0) = 1 − 0.9772 = 0.0228 ≈ 2.3%
```

**(b)**
```
P(X < 0.75) = P(Z < (0.75−0.82)/0.04) = P(Z < −1.75)
            = Φ(−1.75) = 1 − Φ(1.75) = 1 − 0.9599 = 0.0401 ≈ 4%
```

**(c)**
```
P(0.78 ≤ X ≤ 0.86) = P((0.78−0.82)/0.04 ≤ Z ≤ (0.86−0.82)/0.04)
                   = P(−1 ≤ Z ≤ 1)
                   = 2Φ(1) − 1 = 2(0.8413) − 1 = 0.6827 ≈ 68.3%
```

The empirical rule: 68% of runs fall within 1 SD of the mean. ✓

**(d)** Top 5% means P(X > x*) = 0.05 → Φ((x*−0.82)/0.04) = 0.95

(x*−0.82)/0.04 = 1.645 (from table)
x* = 0.82 + 1.645×0.04 = 0.82 + 0.0658 = **0.886**

Runs with accuracy > 88.6% are in the top 5%.

---

### 🔢 Numerical 3 — Sum of Independent Normals: Ensemble

**Problem:** Three independent models each have error ~ N(0, 4) (μ=0, σ²=4).

**(a)** Distribution of sum of errors
**(b)** Distribution of average error (ensemble output)
**(c)** P(|average error| > 1) for ensemble vs single model
**(d)** How many models needed to halve the SD of the average error?

**Solution:**

**(a)** Sum E₁+E₂+E₃:
```
E[sum] = 0+0+0 = 0
Var(sum) = 4+4+4 = 12
Sum ~ N(0, 12)
```

**(b)** Average Ē = (E₁+E₂+E₃)/3:
```
E[Ē] = 0
Var(Ē) = Var(sum)/9 = 12/9 = 4/3
SD(Ē) = 2/√3 ≈ 1.155

Ē ~ N(0, 4/3)
```

**(c)**

Single model E ~ N(0,4), SD=2:
```
P(|E| > 1) = P(|Z| > 1/2) = 2(1−Φ(0.5)) = 2(0.3085) = 0.617
```

Ensemble Ē ~ N(0, 4/3), SD≈1.155:
```
P(|Ē| > 1) = P(|Z| > 1/1.155) = P(|Z| > 0.866) = 2(1−Φ(0.866))
           = 2(1−0.807) = 2(0.193) = 0.386
```

Ensemble reduces probability of large error from **61.7% to 38.6%**. ✓

**(d)** SD of average with n models = 2/√n. Want SD < 2/2 = 1:
```
2/√n < 1  →  √n > 2  →  n > 4
```
Need at least **5 models** to halve the SD.

**General rule:** To halve SD, you need 4× as many models. Diminishing returns — this is why ensemble sizes plateau around 10-50 models in practice.

---

### 🔢 Numerical 4 — Gaussian MLE

**Problem:** You observe 5 loss values from a model: 2.1, 1.8, 2.4, 1.9, 2.3.

Assuming losses ~ N(μ, σ²):
**(a)** Compute μ̂_MLE
**(b)** Compute σ̂²_MLE (biased)
**(c)** Compute s² (unbiased, divides by n−1)
**(d)** Write the log-likelihood evaluated at the MLE

**Solution:**

**(a)**
```
μ̂ = (2.1+1.8+2.4+1.9+2.3)/5 = 10.5/5 = 2.10
```

**(b)**
```
Σ(xᵢ−μ̂)² = (0.0)²+(−0.3)²+(0.3)²+(−0.2)²+(0.2)²
           = 0 + 0.09 + 0.09 + 0.04 + 0.04 = 0.26

σ̂²_MLE = 0.26/5 = 0.052
σ̂_MLE = √0.052 ≈ 0.228
```

**(c)**
```
s² = 0.26/(5−1) = 0.26/4 = 0.065
s = √0.065 ≈ 0.255
```

**(d)** Log-likelihood at MLE:
```
ℓ(μ̂, σ̂²) = −n/2·log(2πσ̂²) − 1/(2σ̂²)·Σ(xᵢ−μ̂)²
           = −5/2·log(2π×0.052) − 0.26/(2×0.052)
           = −5/2·log(0.3267) − 2.5
           = −5/2·(−1.119) − 2.5
           = 2.798 − 2.5 = 0.298
```

**ML insight:** Maximizing this log-likelihood is equivalent to minimizing MSE (the Σ(xᵢ−μ)² term). This is the exact mathematical reason why **MSE is the natural loss function for regression under Gaussian noise assumption**.

---

### 🔢 Numerical 5 — Weight Initialization (Xavier/He)

**Problem:** A neural network layer has nᵢₙ=512 input neurons.

**Xavier initialization:** W ~ N(0, 1/nᵢₙ) = N(0, 1/512)
**He initialization:** W ~ N(0, 2/nᵢₙ) = N(0, 2/512)

**(a)** For a single weight w ~ N(0, 1/512), find P(|w| > 0.1)
**(b)** For He init, find SD of w
**(c)** For a layer output z = Σᵢ wᵢxᵢ where xᵢ ~ N(0,1) i.i.d. and wᵢ ~ N(0, 1/512) i.i.d., find Var(z)
**(d)** Why does He use 2/nᵢₙ instead of 1/nᵢₙ?

**Solution:**

**(a)** Xavier: w ~ N(0, 1/512), σ = 1/√512 ≈ 0.0442
```
P(|w| > 0.1) = P(|Z| > 0.1/0.0442) = P(|Z| > 2.26)
             = 2(1−Φ(2.26)) = 2(0.012) = 0.024
```
Only 2.4% of Xavier weights exceed 0.1 in magnitude — most are small.

**(b)** He: σ = √(2/512) = √(1/256) = 1/16 = 0.0625

**(c)** z = Σᵢ₌₁^512 wᵢxᵢ, where wᵢ, xᵢ are independent:
```
Var(wᵢxᵢ) = E[wᵢ²]E[xᵢ²] − (E[wᵢ]E[xᵢ])²
           = Var(wᵢ)·Var(xᵢ)    [since both have mean 0]
           = (1/512)·1 = 1/512

Var(z) = 512 × (1/512) = 1
```

**Xavier initialization preserves variance!** Input variance = output variance = 1. This prevents vanishing/exploding gradients in linear activations.

**(d)** With ReLU activation, roughly half the neurons are zeroed out — this halves the variance. He initialization uses **2/nᵢₙ** to compensate, restoring variance to 1 after ReLU:
```
Var(ReLU(z)) ≈ Var(z)/2    [ReLU zeroes ~half inputs]
He: Var(z) = 512 × (2/512) = 2
After ReLU: Var ≈ 2/2 = 1  ✓
```

**ML insight:** Initialization is a Normal distribution design problem. Xavier (for linear/tanh) and He (for ReLU) are derived by solving for the Normal variance that preserves signal variance through a layer. This completely eliminates the vanishing gradient problem in deep networks with proper activations.

---

### 🔢 Numerical 6 — Gaussian Naive Bayes

**Problem:** Build a Gaussian Naive Bayes classifier for 2 classes (0 and 1) with one feature X.

From training data:
- Class 0: X ~ N(2, 1) [μ₀=2, σ₀²=1]
- Class 1: X ~ N(5, 4) [μ₁=5, σ₁²=4]
- P(Y=0) = 0.6, P(Y=1) = 0.4

Classify new point x = 3.5.

**Solution:**

Compute log posteriors:

**log P(Y=0|x=3.5) ∝** log P(Y=0) + log f(x=3.5 | Y=0)
```
log f(3.5|Y=0) = −log(√(2π)·1) − (3.5−2)²/(2·1)
               = −0.919 − 1.125 = −2.044

log P(Y=0|x) ∝ log(0.6) + (−2.044) = −0.511 − 2.044 = −2.555
```

**log P(Y=1|x=3.5) ∝** log P(Y=1) + log f(x=3.5 | Y=1)
```
log f(3.5|Y=1) = −log(√(2π)·2) − (3.5−5)²/(2·4)
               = −log(5.013) − 0.281
               = −1.612 − 0.281 = −1.893

log P(Y=1|x) ∝ log(0.4) + (−1.893) = −0.916 − 1.893 = −2.809
```

Compare: −2.555 > −2.809

**Classification: Y = 0** (class 0 has higher log posterior)

Unnormalized: e^(−2.555) = 0.077, e^(−2.809) = 0.060
Normalize: P(Y=0|x) = 0.077/(0.077+0.060) = **0.562**

Despite x=3.5 being closer to class 1's mean (5), class 0 wins because:
1. Its prior is higher (0.6 vs 0.4)
2. Class 0 has smaller variance (tighter distribution), making 3.5 relatively more probable

---

### 🔢 Numerical 7 — L2 Regularization as Gaussian Prior

**Problem:** Linear regression with L2 regularization:
```
Loss = Σᵢ(yᵢ − wᵀxᵢ)² + λ||w||²
```

Show this equals MAP estimation with:
- Gaussian likelihood: yᵢ|xᵢ,w ~ N(wᵀxᵢ, σ²)
- Gaussian prior: w ~ N(0, τ²I)

**Solution:**

MAP objective: minimize negative log posterior
```
−log P(w|data) = −log P(data|w) − log P(w) + const

−log P(data|w) = Σᵢ (yᵢ−wᵀxᵢ)²/(2σ²)     [Gaussian likelihood]

−log P(w) = ||w||²/(2τ²)                     [Gaussian prior]

MAP: minimize Σᵢ(yᵢ−wᵀxᵢ)²/(2σ²) + ||w||²/(2τ²)

     = (1/2σ²)[Σᵢ(yᵢ−wᵀxᵢ)² + (σ²/τ²)||w||²]
```

Setting **λ = σ²/τ²**, this is exactly:
```
(1/2σ²)[Σᵢ(yᵢ−wᵀxᵢ)² + λ||w||²]
```

**Minimizing this = L2 regularized regression = MAP with Gaussian prior.**

**Interpretation:**
- Large λ → small τ² → tight prior → weights forced toward 0 → more regularization
- λ = 0 → uninformative prior → pure MLE → no regularization
- The ratio σ²/τ² controls regularization strength

**ML insight:** Every time you use L2 regularization, you're implicitly saying "I believe the weights come from a Gaussian distribution centered at zero." This is a Bayesian prior, not just a mathematical trick.

---

## 11. The Normal Distribution in Deep Learning Summary

```
Component              Normal Distribution Role
─────────────────────────────────────────────────────────────
Weight initialization  W ~ N(0, 2/nᵢₙ) — He init for ReLU
                       W ~ N(0, 1/nᵢₙ) — Xavier for linear/tanh

Gradient noise         ∇L ≈ N(true gradient, noise covariance)
                       — motivates Adam, momentum

Batch normalization    Forces layer activations → N(0,1)
                       then learns γ,β to rescale

VAE latent space       z ~ N(0, I) — standard Gaussian prior
                       Encoder learns q(z|x) ~ N(μ(x), σ²(x))

L2 regularization      Equivalent to w ~ N(0, 1/λ) prior

Confidence intervals   x̄ ± 1.96·σ/√n — uses N(0,1) quantiles

Hypothesis testing     Test statistics → N(0,1) under H₀ (CLT)
```

---

## 12. Common Interview Questions

| Question | Key Idea |
|---|---|
| "Write the Normal PDF and explain each term" | Normalizing constant, exponential decay, μ shifts, σ scales |
| "What is the 68-95-99.7 rule?" | 68%/95%/99.7% within 1/2/3 SDs |
| "Why is MSE the natural loss for regression?" | MSE = MLE under Gaussian noise assumption |
| "What does L2 regularization assume about weights?" | Gaussian prior w ~ N(0, 1/λ) |
| "Why does Xavier init use 1/nᵢₙ?" | Preserves variance through a linear layer |
| "Why does He init use 2/nᵢₙ?" | Accounts for ReLU zeroing half the neurons |
| "Sum of independent Normals?" | Normal — mean sums, variances sum |
| "When does zero covariance imply independence?" | Only for jointly Gaussian random variables |
| "What is the maximum entropy distribution with fixed mean and variance?" | Normal distribution |

---

## 13. Key Formulas — Cheat Sheet for Day 12

```
Normal(μ, σ²):
    f(x) = (1/σ√2π) exp(−(x−μ)²/2σ²)
    E[X] = μ,   Var(X) = σ²

Standard Normal Z ~ N(0,1):
    φ(z) = (1/√2π) exp(−z²/2)
    Φ(z) = P(Z ≤ z)
    Φ(−z) = 1 − Φ(z)

Z-score:
    Z = (X−μ)/σ ~ N(0,1)  if X ~ N(μ,σ²)

Key values:
    Φ(1.0)  = 0.841    [84th percentile]
    Φ(1.645)= 0.950    [95th percentile — one-sided]
    Φ(1.96) = 0.975    [97.5th — two-sided 95%]
    Φ(2.0)  = 0.977
    Φ(2.576)= 0.995    [two-sided 99%]
    Φ(3.0)  = 0.9987

Empirical rule:
    P(μ±σ)  ≈ 68%
    P(μ±2σ) ≈ 95%
    P(μ±3σ) ≈ 99.7%

Linear transformation:
    aX + b ~ N(aμ+b, a²σ²)

Sum of independent Normals:
    X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)

MLE for Normal:
    μ̂ = x̄ = (1/n)Σxᵢ
    σ̂² = (1/n)Σ(xᵢ−x̄)²     [biased MLE]
    s²  = (1/n−1)Σ(xᵢ−x̄)²  [unbiased]

L2 regularization = MAP with w ~ N(0, σ²/λ · I)
He init:    W ~ N(0, 2/nᵢₙ)   [ReLU layers]
Xavier init: W ~ N(0, 1/nᵢₙ)  [linear/tanh layers]
```

---

## 14. Practice Problems (Solve Before Day 13)

1. X ~ N(100, 225) (μ=100, σ=15 — IQ scores). Find:
   - P(X > 130) — "gifted" threshold
   - P(85 < X < 115) — "average" range
   - The 99th percentile IQ score

2. Loss values from 4 training runs: 0.45, 0.52, 0.48, 0.51. Assuming Normal:
   - Compute MLE for μ and σ²
   - Find P(next run has loss > 0.55)

3. **Prove** that if X ~ N(μ, σ²), then Z = (X−μ)/σ ~ N(0,1) using the CDF transformation.

4. A neural net ensemble has 9 models, each with error ~ N(0, σ²=9). Find:
   - SD of the average error
   - P(|average error| > 1)
   - How many models needed so P(|average error| > 1) < 0.05?

5. *(Interview-level)* A VAE encoder outputs μ(x) = 1.5 and log σ²(x) = 0.4 for input x. The reparameterization trick samples: z = μ(x) + σ(x)·ε where ε ~ N(0,1).
   - What distribution does z follow?
   - Why is the reparameterization trick needed for backpropagation?
   - Compute P(z > 2.5).

---

## 15. Looking Ahead

**Day 13** — **Joint Distributions, Covariance & Correlation.** We formalize how two random variables relate to each other — the foundation of feature correlation analysis, PCA, multivariate Gaussian distributions, and understanding when ensemble members add value vs. just duplicate each other.

---
*End of Day 12 | Next: Day 13 — Joint Distributions, Covariance & Correlation*
