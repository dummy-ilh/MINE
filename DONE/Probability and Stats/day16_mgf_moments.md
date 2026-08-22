# Day 16 — Moment Generating Functions (MGFs) & Moments
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 4 (Section 4.5) & Chapter 6
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why MGFs Matter in ML

Moment Generating Functions are the most elegant tool in probability theory. They:

| MGF Capability | ML Application |
|---|---|
| Generate ALL moments at once | Compute E[X], E[X²], E[X³], ... from one function |
| Uniquely identify distributions | Prove two distributions are equal |
| Prove sum of Normals is Normal | Foundation of Central Limit Theorem |
| Derive variance in one line | Faster than direct computation |
| Connect to Laplace transform | Connections to signal processing, neural ODEs |
| Prove distribution limits | Poisson as Binomial limit, CLT |

In interviews, MGFs appear when asked to **prove** distributional properties — not just state them.

---

## 2. Definition of MGF

> **Definition:** The **Moment Generating Function** of X is:
> ```
> Mₓ(t) = E[eᵗˣ]    for t in some neighborhood of 0
> ```

### For discrete X:
```
Mₓ(t) = Σₓ eᵗˣ · p(x)
```

### For continuous X:
```
Mₓ(t) = ∫₋∞^∞ eᵗˣ · f(x) dx
```

### Why eᵗˣ?

Expand the Taylor series:
```
eᵗˣ = 1 + tx + t²x²/2! + t³x³/3! + ...

E[eᵗˣ] = 1 + tE[X] + t²E[X²]/2! + t³E[X³]/3! + ...

        = Σₙ₌₀^∞ E[Xⁿ] · tⁿ/n!
```

The MGF **encodes all moments as its Taylor coefficients**.

---

## 3. Extracting Moments from the MGF

> **The Key Theorem:**
> ```
> E[Xⁿ] = Mₓ⁽ⁿ⁾(0) = dⁿMₓ/dtⁿ |_{t=0}
> ```

The n-th derivative of the MGF evaluated at t=0 gives the n-th moment.

**Proof:**
```
Mₓ(t) = Σₙ₌₀^∞ E[Xⁿ] · tⁿ/n!

dMₓ/dt = Σₙ₌₁^∞ E[Xⁿ] · tⁿ⁻¹/(n−1)!

At t=0:  Mₓ'(0) = E[X]·1/0! = E[X]  ✓

d²Mₓ/dt² at t=0 = E[X²]  ✓

dⁿMₓ/dtⁿ at t=0 = E[Xⁿ]  ✓
```

### The Recipe

```
1st moment (mean):    E[X]   = M'(0)
2nd moment:           E[X²]  = M''(0)
Variance:             Var(X) = M''(0) − [M'(0)]²
3rd central moment:   E[(X−μ)³] — computed similarly
```

---

## 4. The Uniqueness Theorem

> **Theorem:** If Mₓ(t) = Mᵧ(t) for all t in some neighborhood of 0, then X and Y have the same distribution.

**This is powerful:** To prove two RVs have the same distribution, just show their MGFs match.

---

## 5. MGF of a Linear Transformation

If Y = aX + b:
```
Mᵧ(t) = E[eᵗ⁽ᵃˣ⁺ᵇ⁾] = eᵇᵗ · E[e^(at)X] = eᵇᵗ · Mₓ(at)
```

---

## 6. MGF of Sum of Independent RVs

If X and Y are **independent**:
```
M_{X+Y}(t) = E[eᵗ⁽ˣ⁺ʸ⁾] = E[eᵗˣ · eᵗʸ] = E[eᵗˣ] · E[eᵗʸ] = Mₓ(t) · Mᵧ(t)
```

**MGF of sum = product of MGFs** (when independent).

This is the key to proving:
- Sum of Normals is Normal
- Sum of Poissons is Poisson
- Sum of Binomials (same p) is Binomial

---

## 7. MGFs of Key Distributions

### Bernoulli(p)
```
M(t) = E[eᵗˣ] = eᵗ·p + e⁰·(1−p) = 1 − p + peᵗ
```

### Binomial(n,p)
```
M(t) = (1 − p + peᵗ)ⁿ    [product of n independent Bernoulli MGFs]
```

**Derive moments:**
```
M'(t) = n(1−p+peᵗ)^(n−1) · peᵗ
M'(0) = n·1^(n−1)·p = np = E[X]  ✓

M''(t) = np·eᵗ·[(n−1)(1−p+peᵗ)^(n−2)·peᵗ + (1−p+peᵗ)^(n−1)]
M''(0) = np[(n−1)p + 1] = n(n−1)p² + np

Var(X) = M''(0) − [M'(0)]² = n(n−1)p²+np − n²p²
       = n²p²−np²+np−n²p² = np−np² = np(1−p)  ✓
```

### Poisson(λ)
```
M(t) = E[eᵗˣ] = Σₖ₌₀^∞ eᵗᵏ · e^(−λ)λᵏ/k!
              = e^(−λ) Σₖ₌₀^∞ (λeᵗ)ᵏ/k!
              = e^(−λ) · e^(λeᵗ)
              = exp(λ(eᵗ−1))
```

**Derive moments:**
```
M'(t) = λeᵗ · exp(λ(eᵗ−1))
M'(0) = λ · 1 = λ = E[X]  ✓

M''(t) = λeᵗ·exp(λ(eᵗ−1)) + (λeᵗ)²·exp(λ(eᵗ−1))
M''(0) = λ + λ² = E[X²]

Var(X) = λ+λ² − λ² = λ  ✓
```

### Exponential(λ)
```
M(t) = ∫₀^∞ eᵗˣ · λe^(−λx) dx = λ∫₀^∞ e^(−(λ−t)x) dx
     = λ/(λ−t)    for t < λ
```

**Derive moments:**
```
M'(t) = λ/(λ−t)²
M'(0) = 1/λ = E[X]  ✓

M''(t) = 2λ/(λ−t)³
M''(0) = 2/λ²

Var(X) = 2/λ² − (1/λ)² = 2/λ² − 1/λ² = 1/λ²  ✓
```

### Normal(μ, σ²)

```
M(t) = exp(μt + σ²t²/2)
```

**Derivation:**
```
M(t) = ∫₋∞^∞ eᵗˣ · (1/σ√2π) exp(−(x−μ)²/2σ²) dx

Complete the square in the exponent:
tx − (x−μ)²/2σ² = −(x−(μ+σ²t))²/2σ² + μt + σ²t²/2

M(t) = exp(μt + σ²t²/2) · ∫₋∞^∞ (1/σ√2π)exp(−(x−(μ+σ²t))²/2σ²) dx
     = exp(μt + σ²t²/2) · 1    [Normal PDF integrates to 1]
     = exp(μt + σ²t²/2)  ✓
```

**Derive moments:**
```
log M(t) = μt + σ²t²/2    [log-MGF / cumulant generating function]

M'(t) = (μ + σ²t) · exp(μt + σ²t²/2)
M'(0) = μ = E[X]  ✓

M''(t) = σ² · exp(μt+σ²t²/2) + (μ+σ²t)² · exp(μt+σ²t²/2)
M''(0) = σ² + μ² = E[X²]

Var(X) = σ²+μ² − μ² = σ²  ✓
```

**All moments of N(μ,σ²):**
For Z ~ N(0,1): M(t) = e^(t²/2), so:
```
E[Z^n] = 0          for odd n     [symmetry]
E[Z^n] = (n−1)!!    for even n    [(n−1)!! = (n−1)(n−3)···3·1]

E[Z²] = 1,  E[Z⁴] = 3,  E[Z⁶] = 15,  E[Z⁸] = 105
```

---

## 8. Cumulant Generating Function (CGF)

> **Definition:** K(t) = log M(t) = log E[eᵗˣ]

**Cumulants** κₙ = K⁽ⁿ⁾(0):

```
κ₁ = K'(0) = E[X] = μ              [mean]
κ₂ = K''(0) = Var(X) = σ²          [variance]
κ₃ = K'''(0) = E[(X−μ)³]           [third central moment — skewness]
κ₄ = K''''(0) = E[(X−μ)⁴]−3σ⁴     [excess kurtosis × σ⁴]
```

**Why cumulants?** For independent X, Y:
```
K_{X+Y}(t) = Kₓ(t) + Kᵧ(t)    [cumulants ADD for independent sums]
```

This is simpler than multiplying MGFs.

**Normal distribution:** K(t) = μt + σ²t²/2 → all cumulants beyond κ₂ are **zero**. The Normal is completely characterized by its first two cumulants. This is the deepest reason the Normal is special.

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — MGF of Bernoulli: All Moments

**Problem:** X ~ Bernoulli(p=0.7). Find:
**(a)** MGF M(t)
**(b)** E[X], E[X²], E[X³] from the MGF
**(c)** Var(X) from the MGF

**Solution:**

**(a)**
```
M(t) = 0.3 + 0.7eᵗ
```

**(b)**
```
M'(t) = 0.7eᵗ
M'(0) = 0.7 = E[X]  ✓

M''(t) = 0.7eᵗ
M''(0) = 0.7 = E[X²]

M'''(t) = 0.7eᵗ
M'''(0) = 0.7 = E[X³]
```

Note: All moments of Bernoulli(p) equal p — because X ∈ {0,1} means Xⁿ = X for all n≥1.

**(c)**
```
Var(X) = E[X²] − (E[X])² = 0.7 − 0.49 = 0.21 = p(1−p)  ✓
```

---

### 🔢 Numerical 2 — Proving Sum of Poissons is Poisson

**Problem:** X ~ Poisson(λ₁=3) and Y ~ Poisson(λ₂=5), independent. Prove X+Y ~ Poisson(8).

**Solution:**

```
Mₓ(t) = exp(3(eᵗ−1))
Mᵧ(t) = exp(5(eᵗ−1))

M_{X+Y}(t) = Mₓ(t) · Mᵧ(t)    [independence]
           = exp(3(eᵗ−1)) · exp(5(eᵗ−1))
           = exp((3+5)(eᵗ−1))
           = exp(8(eᵗ−1))
```

This is the MGF of Poisson(8). By the uniqueness theorem:

**X + Y ~ Poisson(8)**  ∎

**ML application:** API server A (λ=3 requests/sec) and server B (λ=5 requests/sec) combined → Poisson(8). Total load is Poisson(λ₁+λ₂).

---

### 🔢 Numerical 3 — Proving Sum of Normals is Normal

**Problem:** X ~ N(2, 9) and Y ~ N(−1, 4), independent. Find the distribution of X+Y using MGFs.

**Solution:**

```
Mₓ(t) = exp(2t + 9t²/2)
Mᵧ(t) = exp(−t + 4t²/2)

M_{X+Y}(t) = exp(2t + 9t²/2) · exp(−t + 4t²/2)
           = exp((2+(−1))t + (9+4)t²/2)
           = exp(1·t + 13t²/2)
```

This is the MGF of N(1, 13). By uniqueness:

**X + Y ~ N(2+(−1), 9+4) = N(1, 13)**  ∎

**General rule:** Means add, variances add (independent Normals).

**ML application:** If weight updates from two batches are each approximately Normal, their sum (combined update) is also Normal — enabling Normal approximations for SGD analysis.

---

### 🔢 Numerical 4 — Using MGF to Find All Moments of Exponential

**Problem:** X ~ Exponential(λ=2). Find E[X], E[X²], E[X³], E[X⁴] using the MGF.

**Solution:**

```
M(t) = λ/(λ−t) = 2/(2−t)    for t < 2
```

**Differentiate:**
```
M'(t)   = 2/(2−t)²              → M'(0)   = 2/4 = 1/2  ✓ [= 1/λ]
M''(t)  = 4/(2−t)³              → M''(0)  = 4/8 = 1/2
M'''(t) = 12/(2−t)⁴             → M'''(0) = 12/16 = 3/4
M''''(t)= 48/(2−t)⁵             → M''''(0)= 48/32 = 3/2
```

**Moments:**
```
E[X]  = M'(0)   = 1/2
E[X²] = M''(0)  = 1/2
E[X³] = M'''(0) = 3/4
E[X⁴] = M''''(0)= 3/2
```

**Variance:**
```
Var(X) = E[X²] − (E[X])² = 1/2 − 1/4 = 1/4 = 1/λ²  ✓
```

**General formula for Exponential(λ):**
```
E[Xⁿ] = n!/λⁿ
```

Verify: E[X¹] = 1!/2 = 1/2 ✓, E[X²] = 2!/4 = 1/2 ✓, E[X³] = 6/8 = 3/4 ✓

---

### 🔢 Numerical 5 — MGF to Identify an Unknown Distribution

**Problem:** An unknown distribution has MGF:
```
M(t) = (0.4 + 0.6eᵗ)⁸
```

**(a)** Identify the distribution.
**(b)** Find E[X] and Var(X) without differentiating.
**(c)** Find P(X = 5).

**Solution:**

**(a)** Compare to Binomial MGF: M(t) = (1−p+peᵗ)ⁿ

Matching: n=8, p=0.6, 1−p=0.4.

**X ~ Binomial(8, 0.6)**

**(b)** From known Binomial formulas:
```
E[X] = np = 8 × 0.6 = 4.8
Var(X) = np(1−p) = 8 × 0.6 × 0.4 = 1.92
```

**(c)**
```
P(X=5) = C(8,5) × 0.6⁵ × 0.4³
        = 56 × 0.07776 × 0.064
        = 56 × 0.004977 = 0.2787
```

**ML application:** In practice, if you derive the MGF of a transformed variable or sufficient statistic, recognizing its form immediately tells you the distribution — no need to derive moments from scratch.

---

### 🔢 Numerical 6 — Cumulants and the Normal Distribution

**Problem:** X has CGF K(t) = 3t + 2t² (cumulant generating function).

**(a)** Identify the distribution of X.
**(b)** Find mean, variance, and all higher cumulants.
**(c)** If Y is independent with K_Y(t) = −t + 5t², find distribution of X+Y.

**Solution:**

**(a)** CGF of N(μ,σ²) is K(t) = μt + (σ²/2)t².

Comparing: μ=3, σ²/2=2 → σ²=4.

**X ~ N(3, 4)**

**(b)**
```
κ₁ = K'(0) = 3    [mean]
κ₂ = K''(0) = 4   [variance]
κ₃ = K'''(0) = 0  [all higher cumulants = 0 for Normal]
κₙ = 0  for n ≥ 3
```

The Normal is the only distribution where ALL cumulants beyond the 2nd are zero.

**(c)** CGFs add for independent variables:
```
K_{X+Y}(t) = K_X(t) + K_Y(t)
           = (3t + 2t²) + (−t + 5t²)
           = 2t + 7t²
```

This is the CGF of N(2, 14). **X+Y ~ N(2, 14)**

---

### 🔢 Numerical 7 — MGF Proof: CLT Preview

**Problem:** X₁, X₂, ..., Xₙ are i.i.d. with E[Xᵢ]=μ, Var(Xᵢ)=σ². Let:
```
Sₙ = (X₁ + X₂ + ... + Xₙ − nμ) / (σ√n)    [standardized sum]
```

Show that the MGF of Sₙ approaches e^(t²/2) as n→∞.

*(This is the MGF of N(0,1) — proving the CLT via MGFs.)*

**Solution:**

Let Yᵢ = (Xᵢ − μ)/σ (standardized). Then E[Yᵢ]=0, Var(Yᵢ)=1.

```
Sₙ = (Y₁ + Y₂ + ... + Yₙ)/√n
```

MGF of Sₙ:
```
M_{Sₙ}(t) = E[exp(t·Sₙ)] = E[exp(t(Y₁+...+Yₙ)/√n)]

           = [Mᵧ(t/√n)]ⁿ    [independence, each Yᵢ same MGF]
```

Expand log of MGF using Taylor series:
```
log Mᵧ(s) = log E[eˢʸ]
           = log[1 + sE[Y] + s²E[Y²]/2 + O(s³)]
           = log[1 + 0 + s²/2 + O(s³)]
           ≈ s²/2 + O(s³)    [since log(1+x) ≈ x for small x]
```

With s = t/√n:
```
log M_{Sₙ}(t) = n · log Mᵧ(t/√n)
              ≈ n · [(t/√n)²/2 + O((t/√n)³)]
              = n · [t²/(2n) + O(n^(−3/2))]
              = t²/2 + O(n^(−1/2))
              → t²/2  as n→∞
```

Therefore:
```
M_{Sₙ}(t) → e^(t²/2) = MGF of N(0,1)  ∎
```

By the uniqueness theorem: **Sₙ → N(0,1) in distribution** — this IS the Central Limit Theorem.

**ML insight:** The CLT is a statement about MGF convergence. The key step is that the Taylor expansion of ANY distribution's MGF looks like 1 + t²/2 + O(t³) around t=0 when the distribution has mean 0 and variance 1. The Normal is the unique distribution whose MGF is exactly e^(t²/2) — the CLT says all standardized sums converge to this.

---

## 10. Skewness and Kurtosis in ML

Using MGF/moments, we can compute distribution shape measures:

### Skewness
```
γ₁ = E[(X−μ)³]/σ³ = κ₃/κ₂^(3/2)
```
- γ₁ = 0: symmetric (Normal)
- γ₁ > 0: right-skewed (positive tail) — income, loss values
- γ₁ < 0: left-skewed (negative tail) — model accuracies near ceiling

### Kurtosis
```
γ₂ = E[(X−μ)⁴]/σ⁴ − 3    [excess kurtosis; Normal=0]
```
- γ₂ > 0: heavy tails (Laplacian, t-distribution) — outliers likely
- γ₂ < 0: light tails (Uniform) — no extreme values
- γ₂ = 0: Normal tails

### For Key Distributions

| Distribution | Skewness | Excess Kurtosis |
|---|---|---|
| Normal | 0 | 0 |
| Exponential(λ) | 2 | 6 |
| Poisson(λ) | 1/√λ | 1/λ |
| Uniform(a,b) | 0 | −6/5 |
| Bernoulli(p) | (1−2p)/√(p(1−p)) | (1−6p(1−p))/(p(1−p)) |

**ML use:** High kurtosis (heavy tails) → gradient clipping needed, robust loss functions preferred, outlier detection more important.

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is an MGF and what does it encode?" | E[eᵗˣ] — encodes all moments as Taylor coefficients |
| "How do you extract E[X²] from the MGF?" | M''(0) — 2nd derivative at t=0 |
| "MGF of sum of independent RVs?" | Product of individual MGFs |
| "Prove sum of independent Normals is Normal" | MGF product: exp(μ₁t+σ₁²t²/2)·exp(μ₂t+σ₂²t²/2)=exp((μ₁+μ₂)t+(σ₁²+σ₂²)t²/2) |
| "What are cumulants?" | Derivatives of log MGF at 0; mean=κ₁, variance=κ₂ |
| "Why is the Normal special in terms of cumulants?" | Only distribution with all cumulants beyond κ₂ equal to zero |
| "What does the MGF proof of CLT show?" | Standardized sum MGF → e^(t²/2) as n→∞ |
| "What is excess kurtosis and why does it matter in ML?" | Tail heaviness beyond Normal; high kurtosis → need gradient clipping |

---

## 12. Key Formulas — Cheat Sheet for Day 16

```
MGF Definition:
    M(t) = E[eᵗˣ] = Σ eᵗˣ p(x)  or  ∫ eᵗˣ f(x) dx

Moment extraction:
    E[Xⁿ] = M⁽ⁿ⁾(0)    [n-th derivative at t=0]
    E[X]   = M'(0)
    E[X²]  = M''(0)
    Var(X) = M''(0) − [M'(0)]²

MGF of linear transform:
    M_{aX+b}(t) = eᵇᵗ · M_X(at)

MGF of independent sum:
    M_{X+Y}(t) = M_X(t) · M_Y(t)

KEY MGFs:
    Bernoulli(p):   1−p+peᵗ
    Binomial(n,p):  (1−p+peᵗ)ⁿ
    Poisson(λ):     exp(λ(eᵗ−1))
    Exponential(λ): λ/(λ−t)              [t < λ]
    Normal(μ,σ²):   exp(μt + σ²t²/2)

Cumulant Generating Function:
    K(t) = log M(t)
    κ₁ = K'(0) = E[X]
    κ₂ = K''(0) = Var(X)
    Cumulants add for independent sums

Normal is unique: κₙ = 0 for n ≥ 3

Moments of N(0,1):
    E[Z^n] = 0          for odd n
    E[Z^n] = (n−1)!!    for even n
    E[Z²]=1, E[Z⁴]=3, E[Z⁶]=15, E[Z⁸]=105

Moments of Exponential(λ):
    E[Xⁿ] = n!/λⁿ

Skewness:       γ₁ = E[(X−μ)³]/σ³
Excess Kurtosis: γ₂ = E[(X−μ)⁴]/σ⁴ − 3
```

---

## 13. Practice Problems (Solve Before Day 17)

1. X ~ Geometric(p). Derive the MGF M(t) = peᵗ/(1−(1−p)eᵗ). Use it to find E[X] and Var(X).

2. X ~ N(5, 16). Find E[X⁴] using the MGF (Hint: use M(t)=exp(5t+8t²), differentiate 4 times or use the moment formula for Normal).

3. X ~ Poisson(2) and Y ~ Poisson(3) are independent. Using MGFs, find P(X+Y=0) and E[(X+Y)²].

4. An unknown distribution has MGF M(t) = e^(4(eᵗ−1)). Identify the distribution. Find E[X], Var(X), P(X=3).

5. *(Interview-level)* The **log-normal** distribution: if X ~ N(μ, σ²), then Y = eˣ is log-normal.
   - Find E[Y] using the MGF of X: E[Y] = E[eˣ] = M_X(1).
   - Find E[Y²] = E[e^(2X)] = M_X(2).
   - Compute Var(Y).
   - Why do stock prices follow log-normal distributions?

---

## 14. Looking Ahead

**Day 17** — **Conditional Expectation & the Law of Total Expectation.** The most important tool for reasoning about predictions, regression, and causal inference. We'll see why E[Y|X] is the best predictor of Y given X, prove the tower property, and connect conditional expectation to neural network outputs, regression, and Bayesian updating.

---
*End of Day 16 | Next: Day 17 — Conditional Expectation & Law of Total Expectation*
