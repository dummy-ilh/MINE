# Day 20 — The Central Limit Theorem (CLT)
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 10 (Section 10.4)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why the CLT is the Most Powerful Theorem in Statistics

The Law of Large Numbers (Day 19) tells us **where** X̄ₙ goes — to μ.

The Central Limit Theorem tells us **how fast** and **in what shape** it gets there.

The CLT answers: why does the Normal distribution appear everywhere?

| Application | CLT at Work |
|---|---|
| Confidence intervals | Based on Normal approximation of X̄ₙ |
| Hypothesis testing | Test statistics are approximately Normal under H₀ |
| A/B testing | CTR differences are approximately Normal |
| SGD noise | Sum of gradient contributions → Normal |
| Ensemble predictions | Average of many model outputs → Normal |
| Central limit in finance | Returns of diversified portfolios → Normal |
| Error bars in ML benchmarks | Normal approximation for accuracy estimates |
| Bootstrap validity | Bootstrap distribution ≈ sampling distribution (Normal) |

The CLT is why statisticians can use Normal-based tools even when data isn't Normal — because **sums and averages of almost anything become Normal** as n grows.

---

## 2. The Central Limit Theorem — Statement

> **Theorem (CLT):** Let X₁, X₂, ..., Xₙ be i.i.d. with mean μ and finite variance σ². Define:
> ```
>        X̄ₙ − μ
> Zₙ = ————————— = √n · (X̄ₙ − μ) / σ
>         σ/√n
> ```
> Then as n → ∞:
> ```
> Zₙ →ᵈ N(0, 1)
> ```
> That is, for any real number z:
> ```
> P(Zₙ ≤ z) → Φ(z)    as n → ∞
> ```

**→ᵈ means convergence in distribution** (weaker than convergence in probability).

### Equivalent Formulations

```
√n(X̄ₙ − μ)/σ →ᵈ N(0,1)

X̄ₙ ≈ N(μ, σ²/n)      for large n    [approximate distribution of sample mean]

Σᵢ Xᵢ ≈ N(nμ, nσ²)   for large n    [approximate distribution of sum]
```

### What the CLT Requires

```
✓ i.i.d. samples (or at least uncorrelated with equal variance)
✓ Finite variance σ² < ∞
✓ Large enough n (rule of thumb: n ≥ 30 for "symmetric" distributions,
                   n ≥ 100+ for skewed distributions)

✗ Does NOT require:
  - Normal data (works for Bernoulli, Poisson, Exponential, etc.)
  - Same distribution type
  - Any specific distribution
```

---

## 3. Proof via MGFs (from Day 16)

This is the MGF proof sketched on Day 16, now made complete.

**Setup:** Let Yᵢ = (Xᵢ − μ)/σ. Then E[Yᵢ]=0, Var(Yᵢ)=1.

```
Zₙ = (Y₁ + Y₂ + ... + Yₙ)/√n
```

**MGF of Zₙ:**
```
M_{Zₙ}(t) = [M_Y(t/√n)]ⁿ
```

**Taylor expand log M_Y(s) around s=0:**
```
M_Y(s) = E[eˢʸ] = 1 + sE[Y] + s²E[Y²]/2! + O(s³)
        = 1 + 0 + s²/2 + O(s³)    [E[Y]=0, E[Y²]=Var(Y)=1]

log M_Y(s) = s²/2 + O(s³)         [using log(1+x) ≈ x for small x]
```

**With s = t/√n:**
```
log M_{Zₙ}(t) = n · log M_Y(t/√n)
              = n · [(t/√n)²/2 + O((t/√n)³)]
              = n · [t²/(2n) + O(n^{-3/2})]
              = t²/2 + O(n^{-1/2})
              → t²/2  as n → ∞
```

Therefore:
```
M_{Zₙ}(t) → e^{t²/2} = MGF of N(0,1)  ∎
```

By the continuity theorem (MGF convergence → distribution convergence): Zₙ →ᵈ N(0,1).

---

## 4. The CLT — Convergence Rate and Berry-Esseen

**How fast does the CLT kick in?**

The **Berry-Esseen theorem** quantifies the error:
```
sup_z |P(Zₙ ≤ z) − Φ(z)| ≤ C · E[|X−μ|³] / (σ³ √n)
```

where C ≈ 0.4748 (a universal constant).

**Key insight:** The third absolute moment E[|X−μ|³] controls convergence speed.
- Symmetric distributions (skewness=0): faster convergence
- Skewed distributions (large 3rd moment): slower convergence

**Rule of thumb:**
```
n ≥ 30:   CLT works for symmetric, light-tailed distributions
n ≥ 100:  CLT works for moderately skewed distributions
n ≥ 1000: CLT works for heavily skewed distributions (Exponential, etc.)
```

---

## 5. Continuity Correction

When approximating a **discrete** distribution with the Normal, use the **continuity correction**:

```
P(X = k) ≈ P(k − 0.5 ≤ Y ≤ k + 0.5)    where Y ~ Normal
P(X ≤ k) ≈ P(Y ≤ k + 0.5)
P(X ≥ k) ≈ P(Y ≥ k − 0.5)
```

**Why:** The discrete PMF at k corresponds to an interval of width 1 in the continuous approximation. Without correction, you miss half an interval at each boundary.

---

## 6. Delta Method — CLT for Functions of X̄ₙ

**Problem:** What is the approximate distribution of g(X̄ₙ) for a differentiable function g?

> **Delta Method:** If √n(X̄ₙ − μ)/σ →ᵈ N(0,1), then:
> ```
> √n(g(X̄ₙ) − g(μ)) →ᵈ N(0, [g'(μ)]²σ²)
> ```

**Intuition:** First-order Taylor expansion: g(X̄ₙ) ≈ g(μ) + g'(μ)(X̄ₙ − μ)

So g(X̄ₙ) is approximately Normal with variance [g'(μ)]² × σ²/n.

**ML applications:**
- Distribution of log-odds ratio
- Distribution of estimated AUC
- Distribution of sample correlation coefficient
- Distribution of estimated F1 score

---

## 7. CLT for Bernoulli — Normal Approximation to Binomial

The most important special case:

If X ~ Binomial(n, p), then X = Σᵢ Xᵢ where Xᵢ ~ Bernoulli(p), i.i.d.

By CLT:
```
(X − np) / √(np(1−p)) →ᵈ N(0,1)    as n → ∞

Equivalently: X ≈ N(np, np(1−p))    for large n
```

**When to use:**
- n·p ≥ 5 AND n·(1−p) ≥ 5 (rule of thumb)

**ML use:** Test set accuracy, A/B test conversion counts, recall/precision are all Binomial — Normal approximation applies for large test sets.

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — CLT for Sample Mean: Basic Application

**Problem:** Individual customer purchase amounts X have μ=$50 and σ=$30 (distribution unknown).

A sample of n=100 customers is observed. X̄₁₀₀ = sample mean purchase.

**(a)** Approximate distribution of X̄₁₀₀.
**(b)** P(X̄₁₀₀ > $55).
**(c)** P($45 < X̄₁₀₀ < $55).
**(d)** 95th percentile of X̄₁₀₀.

**Solution:**

By CLT: X̄₁₀₀ ≈ N(μ, σ²/n) = N(50, 900/100) = N(50, 9)
SD(X̄₁₀₀) = √9 = 3

**(a)** X̄₁₀₀ ≈ **N(50, 9)** — approximately Normal regardless of original distribution shape.

**(b)**
```
P(X̄₁₀₀ > 55) = P(Z > (55−50)/3) = P(Z > 1.667)
              = 1 − Φ(1.667) = 1 − 0.9525 = 0.0475 ≈ 4.8%
```

**(c)**
```
P(45 < X̄ < 55) = P((45−50)/3 < Z < (55−50)/3)
               = P(−1.667 < Z < 1.667)
               = 2Φ(1.667) − 1 = 2(0.9525) − 1 = 0.905 ≈ 90.5%
```

**(d)** 95th percentile: z* = 1.645
```
x* = μ + z*·σ/√n = 50 + 1.645×3 = 50 + 4.935 = $54.94
```

**ML insight:** This is the foundation of confidence intervals. The ±1.645σ/√n gives a 90% CI, ±1.96σ/√n gives 95% CI. All confidence interval formulas follow from the CLT approximation for X̄ₙ.

---

### 🔢 Numerical 2 — Normal Approximation to Binomial with Continuity Correction

**Problem:** A model has true accuracy p = 0.80. Tested on n=200 samples. Let X = number correct.

**(a)** Exact distribution of X and its parameters.
**(b)** Approximate distribution via CLT.
**(c)** P(X ≥ 170) — exact Binomial vs Normal approximation (with and without continuity correction).
**(d)** P(155 ≤ X ≤ 170) with continuity correction.

**Solution:**

X ~ Binomial(200, 0.80)
```
E[X] = 200×0.80 = 160
Var(X) = 200×0.80×0.20 = 32
SD(X) = √32 ≈ 5.657
```

**(b)** CLT: X ≈ N(160, 32)

**(c)**

Without continuity correction:
```
P(X ≥ 170) ≈ P(Z ≥ (170−160)/5.657) = P(Z ≥ 1.768) = 1−0.9614 = 0.0386
```

With continuity correction (X ≥ 170 is discrete, approximated by Y > 169.5):
```
P(X ≥ 170) ≈ P(Z ≥ (169.5−160)/5.657) = P(Z ≥ 1.680) = 1−0.9535 = 0.0465
```

Exact Binomial (computed): **0.0493**

Continuity correction (0.0465) is closer to exact (0.0493) than without correction (0.0386). ✓

**(d)** With continuity correction (155 ≤ X ≤ 170 → 154.5 ≤ Y ≤ 170.5):
```
P(154.5 ≤ Y ≤ 170.5) = P((154.5−160)/5.657 ≤ Z ≤ (170.5−160)/5.657)
                      = P(−0.972 ≤ Z ≤ 1.856)
                      = Φ(1.856) − Φ(−0.972)
                      = 0.9683 − 0.1655 = 0.8028 ≈ 80.3%
```

---

### 🔢 Numerical 3 — CLT for A/B Test Analysis

**Problem:** A/B test for a new recommendation model:
- Control (A): n_A = 1000 users, p_A = 0.12 conversion rate
- Treatment (B): n_B = 1000 users, p_B unknown, observed X_B = 135 conversions

**(a)** Under H₀: p_A = p_B = 0.12, what is the distribution of X_B?
**(b)** How surprising is X_B = 135 under H₀?
**(c)** Approximate distribution of (p̂_B − p̂_A) under H₀.
**(d)** 95% confidence interval for the true conversion rate difference.

**Solution:**

**(a)** X_B ~ Binomial(1000, 0.12).

By CLT: X_B ≈ N(120, 1000×0.12×0.88) = N(120, 105.6), SD ≈ 10.28

**(b)** Z-score for X_B = 135:
```
Z = (135 − 120) / 10.28 = 15/10.28 = 1.459
P(X_B ≥ 135) ≈ P(Z ≥ 1.459) = 1 − Φ(1.459) = 1 − 0.9277 = 0.0723
```

7.2% chance — not statistically significant at the 5% level (one-sided).

**(c)** p̂_A = X_A/n_A, p̂_B = X_B/n_B.

Under H₀, both ≈ N(0.12, 0.12×0.88/1000):
```
p̂_B − p̂_A ≈ N(0, Var(p̂_A) + Var(p̂_B))
           = N(0, 2×0.12×0.88/1000)
           = N(0, 0.0002112)
SD = √0.0002112 ≈ 0.01453
```

Observed difference: 135/1000 − 120/1000 = 0.015

Z = 0.015/0.01453 = **1.032** → p-value ≈ 0.30 (two-sided)

Not significant — insufficient evidence to conclude B is better.

**(d)** 95% CI for (p_B − p_A):
```
(p̂_B − p̂_A) ± 1.96 × SE
= 0.015 ± 1.96 × 0.01453
= 0.015 ± 0.0285
= (−0.0135, 0.0435)
```

CI includes 0 — consistent with no difference. ✓

**ML insight:** Every A/B test in ML uses the CLT. Even though conversion events are Bernoulli (0/1), the difference in proportions is approximately Normal for large n, enabling z-tests and confidence intervals.

---

### 🔢 Numerical 4 — CLT for SGD: Gradient Noise is Normal

**Problem:** In SGD, the gradient estimate Ĝ at step t is:
```
Ĝ = (1/B) Σᵢ ∈ batch ∇ℓ(xᵢ, yᵢ)
```

Assume individual sample gradients ∇ℓᵢ are i.i.d. with E[∇ℓᵢ]=g (true gradient) and Var(∇ℓᵢ)=σ².

**(a)** Distribution of Ĝ by CLT for B=32, 64, 128.
**(b)** If g=−0.1, σ²=0.36, find P(Ĝ has wrong sign) for each B.
**(c)** How does this change the learning dynamics?
**(d)** Why does CLT justify treating SGD noise as Gaussian?

**Solution:**

**(a)** By CLT: Ĝ ≈ N(g, σ²/B)

| B | Distribution of Ĝ | SD(Ĝ) |
|---|---|---|
| 32 | N(−0.1, 0.36/32) = N(−0.1, 0.01125) | 0.1061 |
| 64 | N(−0.1, 0.005625) | 0.0750 |
| 128 | N(−0.1, 0.002813) | 0.0530 |

**(b)** P(Ĝ > 0) = P(Z > 0.1/SD):

| B | P(wrong sign) |
|---|---|
| 32 | P(Z > 0.1/0.1061) = P(Z > 0.943) = 17.3% |
| 64 | P(Z > 0.1/0.0750) = P(Z > 1.333) = 9.1% |
| 128 | P(Z > 0.1/0.0530) = P(Z > 1.887) = 3.0% |

**(c)** Larger batches → lower probability of wrong-direction step → smoother, more reliable descent path. But:
- Compute cost scales linearly with B
- Beyond a certain point, returns diminish (already <5% wrong-direction at B=128)
- Large B trains faster per step but often to worse optima (sharper minima)

**(d)** CLT justification: each gradient ∇ℓᵢ has finite variance. Sum of B such gradients → Normal by CLT. This justifies:
- Gaussian noise models for SGD analysis
- The noise-as-regularizer interpretation (Gaussian noise with known variance)
- Diffusion approximations of SGD (continuous-time SDE models)
- The connection between SGD noise scale and learning rate: effective noise variance ∝ σ²η/B

---

### 🔢 Numerical 5 — Delta Method: Distribution of log(X̄)

**Problem:** X₁, ..., Xₙ ~ Exponential(λ=2). True mean μ=1/λ=0.5.

You compute Ŷ = log(X̄ₙ) — the log of the sample mean.

**(a)** E[X̄ₙ] and Var(X̄ₙ).
**(b)** Using the Delta Method, find the approximate distribution of Ŷ = log(X̄ₙ).
**(c)** P(Ŷ < −0.8) for n=100.
**(d)** Why does log-transformation matter for ML?

**Solution:**

Exponential(2): μ=0.5, σ²=1/λ²=0.25.

**(a)**
```
E[X̄ₙ] = 0.5
Var(X̄ₙ) = 0.25/n
```

**(b)** Delta Method with g(x) = log(x), g'(x) = 1/x:
```
√n(g(X̄ₙ) − g(μ)) →ᵈ N(0, [g'(μ)]²σ²)

[g'(μ)]² = [1/0.5]² = 4

√n(log(X̄ₙ) − log(0.5)) →ᵈ N(0, 4×0.25) = N(0, 1)

log(X̄ₙ) ≈ N(log(0.5), 1/n) = N(−0.693, 0.01) for n=100
```

**(c)** For n=100:
```
P(Ŷ < −0.8) = P(log(X̄) < −0.8)
             ≈ P(Z < (−0.8−(−0.693))/√0.01)
             = P(Z < (−0.107)/0.1)
             = P(Z < −1.07)
             = Φ(−1.07) = 1−Φ(1.07) = 1−0.858 = 0.142
```

About 14.2% chance log(X̄) < −0.8 with 100 Exponential samples.

**(d)** Log-transformation in ML:
- Log-transforms convert multiplicative models to additive ones
- Log-likelihood is more numerically stable than likelihood
- The Delta Method tells you the distribution of log-transformed estimates
- This is why log-loss (cross-entropy) has nicer statistical properties than squared loss for probabilities

---

### 🔢 Numerical 6 — CLT for Ensemble: Why Averaging Works

**Problem:** An ensemble of n independent models each predicts Y. Each model's error Eᵢ ~ (unknown distribution) with E[Eᵢ]=0 and Var(Eᵢ)=σ²=4.

The ensemble error is Ē = (E₁+...+Eₙ)/n.

**(a)** Distribution of Ē by CLT for n=5, 20, 100.
**(b)** P(|Ē| > 1) for each n.
**(c)** How many models to achieve P(|Ē| > 0.5) < 0.05?
**(d)** At what n does CLT kick in even for non-Normal individual errors?

**Solution:**

**(a)** By CLT: Ē ≈ N(0, 4/n)

| n | Ē distribution | SD(Ē) |
|---|---|---|
| 5 | N(0, 0.8) | 0.894 |
| 20 | N(0, 0.2) | 0.447 |
| 100 | N(0, 0.04) | 0.200 |

**(b)** P(|Ē| > 1) = P(|Z| > 1/SD(Ē)):

| n | P(\|Ē\| > 1) |
|---|---|
| 5 | P(\|Z\|>1.118) = 2(1−0.868) = 0.264 |
| 20 | P(\|Z\|>2.236) = 2(1−0.987) = 0.025 |
| 100 | P(\|Z\|>5.0) ≈ 0.0000003 |

**(c)** P(|Ē| > 0.5) < 0.05:
```
P(|Z| > 0.5/√(4/n)) < 0.05
P(|Z| > 0.25√n) < 0.05
0.25√n > 1.96
√n > 7.84
n > 61.5
```

Need **at least 62 models** for P(|ensemble error| > 0.5) < 5%.

**(d)** CLT convergence depends on distribution skewness. For symmetric (E[Eᵢ³]=0) errors, n≥5 is often adequate. For skewed errors, need n≥30. For heavy-tailed errors (large E[|Eᵢ|³]), need n≥100+.

**ML insight:** This is why Random Forests use hundreds of trees but diminishing returns set in — the CLT tells us the ensemble error distribution is Normal with variance 4/n. Going from n=100 to n=200 trees halves the variance but the SD only drops by √2 ≈ 1.41, a modest improvement.

---

### 🔢 Numerical 7 — CLT Breakdown: When n=30 Isn't Enough

**Problem:** Individual claim amounts X follow a log-normal distribution: log(X) ~ N(0, 4), so X is heavily right-skewed.

Parameters: μ = e^(0+4/2) = e² ≈ 7.389, σ² = (e⁴−1)e⁴ ≈ 2926.

**(a)** Skewness of X (shows why CLT is slow).
**(b)** Distribution of X̄ₙ for n=30 — is CLT accurate?
**(c)** For n=30, compare P(X̄ > 15) via CLT vs actual (log-normal sum is approximately log-normal for small n).
**(d)** What n is needed for CLT to be accurate?

**Solution:**

For log-normal with parameters (μ_LN=0, σ_LN²=4):
```
E[X] = e^{0+2} = e² ≈ 7.389
Var(X) = (e⁴−1)·e⁴ ≈ 53.6×54.6 ≈ 2926
SD(X) ≈ 54.09
Skewness = (e^{σ²}+2)√(e^{σ²}−1) = (e⁴+2)√(e⁴−1) ≈ 56.6×7.32 ≈ **414**
```

Skewness = 414 — **extremely skewed**. CLT will converge very slowly.

**(b)** CLT says: X̄₃₀ ≈ N(7.389, 2926/30) = N(7.389, 97.53), SD ≈ 9.876

But with skewness 414, Berry-Esseen error bound:
```
Error ≤ C·E[|X−μ|³]/(σ³√n)
```
E[|X−μ|³] is enormous for log-normal — the CLT approximation is poor at n=30.

**(c)** CLT gives:
```
P(X̄₃₀ > 15) ≈ P(Z > (15−7.389)/9.876) = P(Z > 0.771) = 0.220
```

Actual (simulation-based): ≈ 0.30–0.35 (the heavy right tail means large values are more common than CLT predicts at n=30).

**(d)** By Berry-Esseen with skewness ≈ 414:
```
For error < 0.01: need √n ≈ 414×C/0.01 ≈ 414×0.47/0.01 ≈ 19,458
n ≈ 378,000,000!
```

**For this distribution, CLT is not accurate until n in the millions.**

**ML insight:** This is why insurance companies (and ML practitioners dealing with heavy-tailed losses) cannot rely on Normal approximations even with large samples. Log-transforms, robust statistics, and non-parametric methods are necessary. Always check skewness before applying CLT-based methods.

---

## 9. CLT in Modern Deep Learning

```
Application                     CLT Role
────────────────────────────────────────────────────────────────
Neural Network Initialization   Sum of many random weights →
                                 pre-activations ~ Normal (CLT)
                                 Justifies Normal init analysis

Batch Normalization             Forces activations toward N(0,1)
                                 Works WITH the CLT convergence

SGD with large B                Gradient noise ~ Normal (CLT)
                                 Enables Gaussian noise analysis

Confidence intervals for        Model accuracy ~ Normal
model accuracy                  (CLT for Bernoulli sums)

Generalization bounds           Based on Normal tail bounds
                                 (Hoeffding + CLT connection)

Neural Tangent Kernel (NTK)     Infinite-width networks:
                                 activations → Gaussian process
                                 (CLT applied to infinite width)
```

---

## 10. Summary: LLN vs CLT

| Aspect | LLN (Day 19) | CLT (Day 20) |
|---|---|---|
| **What** | X̄ₙ → μ | (X̄ₙ−μ)/(σ/√n) →ᵈ N(0,1) |
| **Tells you** | WHERE it converges | HOW it converges |
| **Scale** | X̄ₙ itself | √n × deviation |
| **Limit** | A number | A distribution |
| **Speed** | No rate | Rate = 1/√n |
| **Requirement** | E[|X|]<∞ | E[X²]<∞ |
| **Use in ML** | ERM validity | Confidence intervals, hypothesis tests |

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "State the CLT" | √n(X̄ₙ−μ)/σ →ᵈ N(0,1) |
| "What does the CLT require?" | i.i.d., finite variance — NOT normality of data |
| "Why does the Normal appear everywhere?" | CLT: sums/averages of anything → Normal |
| "When is n=30 enough?" | Symmetric distributions; need more for skewed |
| "What is the continuity correction?" | Add/subtract 0.5 when approximating discrete with Normal |
| "State the Delta Method" | g(X̄ₙ) ≈ N(g(μ), [g'(μ)]²σ²/n) |
| "How does batch size relate to CLT?" | Batch gradient ~ N(g, σ²/B) — CLT for gradient estimates |
| "Why does Berry-Esseen matter?" | Quantifies CLT error; shows heavy-tailed distributions converge slowly |

---

## 12. Key Formulas — Cheat Sheet for Day 20

```
CLT:
    √n(X̄ₙ−μ)/σ →ᵈ N(0,1)    as n→∞
    X̄ₙ ≈ N(μ, σ²/n)          [for large n]
    Σ Xᵢ ≈ N(nμ, nσ²)         [for large n]

Normal approximation to Binomial:
    X ~ Bin(n,p): X ≈ N(np, np(1−p))
    Use when np≥5 and n(1−p)≥5

Continuity correction:
    P(X ≤ k) ≈ P(Y ≤ k+0.5)   [Y Normal]
    P(X ≥ k) ≈ P(Y ≥ k−0.5)

Delta Method:
    g(X̄ₙ) ≈ N(g(μ), [g'(μ)]²σ²/n)

Berry-Esseen:
    Error ≤ C·E[|X−μ|³]/(σ³√n)   [C ≈ 0.47]
    Larger skewness → slower convergence

Standard Error:
    SE(X̄ₙ) = σ/√n

Rule of thumb:
    n ≥ 30:   CLT works for symmetric distributions
    n ≥ 100:  CLT works for moderately skewed
    n → ∞:    CLT works for any finite-variance distribution

SGD gradient by CLT:
    Ĝ_B ≈ N(g, σ²/B)
    P(wrong sign) = P(Z > |g|·√B/σ)
```

---

## 13. Practice Problems (Solve Before Day 21)

1. X ~ Poisson(λ=4). For n=50 samples, use CLT to find:
   - P(X̄₅₀ > 4.5)
   - P(3.8 < X̄₅₀ < 4.3)
   - 99th percentile of X̄₅₀

2. A/B test: 500 users each in control and treatment. Control CTR=10%, treatment observed 58 clicks. Is treatment significantly better at the 5% level? Use the Normal approximation.

3. **Prove** using MGFs that if X ~ N(μ₁,σ₁²) and Y ~ N(μ₂,σ₂²) are independent, then X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²).

4. X₁,...,Xₙ ~ Exponential(λ=1). Apply the Delta Method to g(X̄)=1/X̄ (estimating λ). Find the approximate distribution of 1/X̄ₙ and its bias.

5. *(Interview-level)* A neural network has L=100 layers. The pre-activation at layer l is:
   ```
   zₗ = Σᵢ₌₁ⁿ wᵢ · aᵢˡ⁻¹
   ```
   where n=512 weights are i.i.d. N(0, 1/n) and activations aᵢˡ⁻¹ are i.i.d. with mean 0 and variance 1 (from previous layer).
   
   - Use CLT to find the approximate distribution of zₗ.
   - What does this imply for weight initialization?
   - Why does this break down if activations are not mean-zero?

---

## 14. Looking Ahead

**Day 21** — **Sampling Distributions: t, Chi-Squared & F.** When σ is unknown (always in practice), we replace it with the sample standard deviation s — this changes Normal to t-distribution. The chi-squared and F distributions arise naturally from sums of squared Normals and are the foundation of variance testing, ANOVA, and regression significance tests.

---
*End of Day 20 | Next: Day 21 — t, Chi-Squared & F Distributions*
