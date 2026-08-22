# Day 13 — Joint Distributions, Covariance & Correlation
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 7
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Joint Distributions Matter in ML

Real ML data is never one variable in isolation. Features interact. Outputs correlate with inputs. Model errors cluster. Everything is multivariate.

| ML Concept | Joint Distribution Tool |
|---|---|
| Feature correlation analysis | Covariance matrix |
| Principal Component Analysis (PCA) | Eigenvectors of covariance matrix |
| Multivariate Gaussian | Joint Normal distribution |
| Gaussian Processes | Kernel = covariance function |
| Ensemble diversity | Low correlation between model errors |
| Multicollinearity in regression | High covariance between features |
| Attention mechanism | Captures pairwise relationships |
| Portfolio / multi-objective optimization | Covariance of objectives |
| Data augmentation | Joint distribution of (original, augmented) pairs |

---

## 2. Joint Distributions — Review and Extension

### Joint PDF (Continuous Case)

For continuous RVs X and Y:
```
P((X,Y) ∈ A) = ∬_A f_{X,Y}(x,y) dx dy

Requirements:
    f_{X,Y}(x,y) ≥ 0
    ∫∫ f_{X,Y}(x,y) dx dy = 1
```

### Marginal PDFs

```
f_X(x) = ∫₋∞^∞ f_{X,Y}(x,y) dy     [integrate out Y]
f_Y(y) = ∫₋∞^∞ f_{X,Y}(x,y) dx     [integrate out X]
```

### Conditional PDF

```
f_{X|Y}(x|y) = f_{X,Y}(x,y) / f_Y(y)
```

### Independence via Joint PDF

X and Y are independent iff:
```
f_{X,Y}(x,y) = f_X(x) · f_Y(y)    for all x, y
```

---

## 3. Covariance — Definition and Intuition

> **Definition:** The **covariance** between X and Y is:
> ```
> Cov(X,Y) = E[(X − E[X])(Y − E[Y])]
>           = E[XY] − E[X]·E[Y]
> ```

### Intuition

Covariance measures how X and Y **move together**:

```
Cov(X,Y) > 0:  X and Y tend to be above their means together
               (when X is high, Y tends to be high)

Cov(X,Y) < 0:  when X is high, Y tends to be low
               (they move in opposite directions)

Cov(X,Y) = 0:  no LINEAR relationship (but may still be dependent!)
```

### Proof of the Computing Formula

```
Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)]
          = E[XY − μᵧX − μₓY + μₓμᵧ]
          = E[XY] − μᵧE[X] − μₓE[Y] + μₓμᵧ
          = E[XY] − μₓμᵧ − μₓμᵧ + μₓμᵧ
          = E[XY] − E[X]·E[Y]   ∎
```

### Key Properties of Covariance

```
1. Cov(X,X) = Var(X)

2. Cov(X,Y) = Cov(Y,X)                     [symmetric]

3. Cov(aX,bY) = ab·Cov(X,Y)               [bilinear]

4. Cov(X+Z, Y) = Cov(X,Y) + Cov(Z,Y)     [bilinear]

5. If X,Y independent: Cov(X,Y) = 0
   (Converse FALSE in general — true only for jointly Normal)

6. Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)

7. Var(X−Y) = Var(X) + Var(Y) − 2Cov(X,Y)

8. Var(aX+bY) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y)
```

Property 6 is the **general variance formula** — the version from Day 8 with the +2Cov term that appears when variables are dependent.

---

## 4. Correlation — Normalized Covariance

> **Definition:** The **Pearson correlation coefficient** between X and Y is:
> ```
>              Cov(X,Y)
> ρ(X,Y) = ———————————————
>            SD(X) · SD(Y)
>
>         = Cov(X,Y) / (σₓ · σᵧ)
> ```

### Key Properties

```
−1 ≤ ρ(X,Y) ≤ 1                [bounded]

ρ = +1:  perfect positive linear relationship (Y = aX+b, a>0)
ρ = −1:  perfect negative linear relationship (Y = aX+b, a<0)
ρ =  0:  no linear relationship (uncorrelated)

ρ is unitless (dimensionless)
ρ is scale-invariant: ρ(aX+b, cY+d) = sign(ac)·ρ(X,Y)
```

### Why Correlation, Not Covariance?

Covariance depends on units — Cov(height in cm, weight in kg) ≠ Cov(height in m, weight in g). Correlation is unit-free and always in [−1, 1], making it comparable across variables.

### The Cauchy-Schwarz Inequality (Proves |ρ| ≤ 1)

```
|Cov(X,Y)|² ≤ Var(X)·Var(Y)
→ |ρ| ≤ 1
```

---

## 5. Covariance Matrix

For a random vector **X** = (X₁, X₂, ..., Xₙ)ᵀ:

> **Definition:** The **covariance matrix** Σ is:
> ```
> Σᵢⱼ = Cov(Xᵢ, Xⱼ)
> ```

Written out:
```
    ⎡ Var(X₁)      Cov(X₁,X₂)  ···  Cov(X₁,Xₙ) ⎤
Σ = ⎢ Cov(X₂,X₁)  Var(X₂)      ···  Cov(X₂,Xₙ) ⎥
    ⎣ Cov(Xₙ,X₁)  Cov(Xₙ,X₂)  ···  Var(Xₙ)     ⎦
```

### Properties of Covariance Matrix

```
1. Σ is symmetric: Σ = Σᵀ
2. Σ is positive semi-definite (PSD): vᵀΣv ≥ 0 for all v
   (eigenvalues ≥ 0)
3. Diagonal entries = variances (always ≥ 0)
4. If features are independent: Σ is diagonal
5. Σ = E[XXᵀ] − E[X]E[X]ᵀ
```

### Connection to PCA

PCA finds the eigenvectors of Σ:
```
Σ = VΛVᵀ

V = matrix of eigenvectors (principal components)
Λ = diagonal matrix of eigenvalues (variances along each PC)
```

The first principal component is the direction of **maximum variance** in the data. PCA rotates to the eigenbasis of Σ — decorrelating features so the new covariance matrix is diagonal.

---

## 6. The Multivariate Normal Distribution

> **Definition:** **X** ~ MVN(μ, Σ) if:
> ```
>                    1                    1
> f(x) = ———————————————————— exp(− ——(x−μ)ᵀΣ⁻¹(x−μ))
>          (2π)^(n/2)|Σ|^(1/2)          2
> ```

### Key Properties

```
Marginals:    Xᵢ ~ N(μᵢ, Σᵢᵢ)                [marginally Normal]
Conditionals: X₁|X₂=x₂ ~ N(μ_cond, Σ_cond)  [conditionally Normal]
Linear combo: aᵀX ~ N(aᵀμ, aᵀΣa)             [closed under linear ops]

Critical: For MVN, zero covariance ⟺ independence
```

### Special Cases

```
Σ = σ²I:  Spherical Gaussian — equal variance in all directions
           Features are independent (identity = diagonal)

Σ diagonal: Features are independent but different variances

Σ = I, μ = 0: Standard multivariate Normal — used in VAEs
```

---

## 7. Correlation vs. Causation vs. Dependence

Three concepts that are constantly confused in ML interviews:

```
Correlation (ρ ≠ 0):
    Measures LINEAR relationship only
    ρ = 0 does NOT mean independent

Dependence:
    X and Y are not independent (any relationship, linear or not)
    Dependence implies ρ ≠ 0 only if the relationship is linear

Causation:
    X causes Y — requires intervention, not just observation
    Correlation ≠ causation (classic example: ice cream & drowning)
```

### Mutual Information (Day 28) vs Correlation

```
Correlation: measures linear dependence only
Mutual Information: measures ANY dependence (including nonlinear)

I(X;Y) = 0 ⟺ X and Y are independent
ρ(X,Y) = 0 ⟺ X and Y are uncorrelated (weaker)
```

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — Computing Covariance from Joint Distribution

**Problem:** X and Y have joint PMF:

|  | Y=1 | Y=2 | Y=3 |
|---|---|---|---|
| **X=1** | 0.10 | 0.05 | 0.15 |
| **X=2** | 0.20 | 0.15 | 0.10 |
| **X=3** | 0.05 | 0.15 | 0.05 |

Find: E[X], E[Y], E[XY], Cov(X,Y), ρ(X,Y).

**Solution:**

**Marginals:**

P(X=1) = 0.10+0.05+0.15 = 0.30
P(X=2) = 0.20+0.15+0.10 = 0.45
P(X=3) = 0.05+0.15+0.05 = 0.25

P(Y=1) = 0.10+0.20+0.05 = 0.35
P(Y=2) = 0.05+0.15+0.15 = 0.35
P(Y=3) = 0.15+0.10+0.05 = 0.30

**Expectations:**
```
E[X] = 1×0.30 + 2×0.45 + 3×0.25 = 0.30 + 0.90 + 0.75 = 1.95
E[Y] = 1×0.35 + 2×0.35 + 3×0.30 = 0.35 + 0.70 + 0.90 = 1.95
```

**E[XY]:**
```
E[XY] = ΣΣ xy·P(X=x,Y=y)
      = 1·1·0.10 + 1·2·0.05 + 1·3·0.15
      + 2·1·0.20 + 2·2·0.15 + 2·3·0.10
      + 3·1·0.05 + 3·2·0.15 + 3·3·0.05
      = 0.10+0.10+0.45+0.40+0.60+0.60+0.15+0.90+0.45
      = 3.75
```

**Covariance:**
```
Cov(X,Y) = E[XY] − E[X]E[Y] = 3.75 − 1.95×1.95 = 3.75 − 3.8025 = −0.0525
```

**Variances:**
```
E[X²] = 1²×0.30 + 2²×0.45 + 3²×0.25 = 0.30+1.80+2.25 = 4.35
Var(X) = 4.35 − 1.95² = 4.35 − 3.8025 = 0.5475

E[Y²] = 1²×0.35 + 2²×0.35 + 3²×0.30 = 0.35+1.40+2.70 = 4.45
Var(Y) = 4.45 − 1.95² = 4.45 − 3.8025 = 0.6475
```

**Correlation:**
```
ρ(X,Y) = Cov(X,Y) / (SD(X)·SD(Y))
        = −0.0525 / (√0.5475 × √0.6475)
        = −0.0525 / (0.7399 × 0.8047)
        = −0.0525 / 0.5954
        = −0.088
```

Weak negative correlation — when X is large, Y tends to be slightly smaller.

---

### 🔢 Numerical 2 — Variance of a Portfolio (Sum with Covariance)

**Problem:** A model ensemble has two sub-models:
- Model A: E[error]=0, Var(error)=4
- Model B: E[error]=0, Var(error)=9
- Cov(errorA, errorB) = 3

The ensemble average = (A + B)/2.

**(a)** Var(A+B)
**(b)** Var(ensemble) = Var((A+B)/2)
**(c)** Compare to naive assumption (Cov=0)
**(d)** What correlation ρ makes ensemble worse than Model A alone?

**Solution:**

**(a)**
```
Var(A+B) = Var(A) + Var(B) + 2Cov(A,B)
         = 4 + 9 + 2×3 = 19
```

**(b)**
```
Var((A+B)/2) = (1/4)Var(A+B) = 19/4 = 4.75
```

**(c)** If we wrongly assumed Cov=0:
```
Var_naive((A+B)/2) = (4+9)/4 = 3.25
```

The true variance (4.75) is larger than the naive estimate (3.25) — correlation makes the ensemble **worse** than we'd expect assuming independence.

**(d)** Ensemble Var = (4+9+2ρ·σ_A·σ_B)/4 = (13+2ρ·2·3)/4 = (13+12ρ)/4

For ensemble to be worse than Model A alone (Var > 4):
```
(13+12ρ)/4 > 4
13+12ρ > 16
12ρ > 3
ρ > 0.25
```

If ρ > 0.25, the ensemble is worse than Model A alone!

**ML insight:** This is why diverse ensembles (low ρ) outperform similar ones. Random Forests achieve diversity via random feature subsets. Boosting achieves it by focusing on previously misclassified examples. The math here is identical to Modern Portfolio Theory in finance.

---

### 🔢 Numerical 3 — Covariance Matrix and Independence Check

**Problem:** Feature matrix X has 3 features. Sample covariance matrix:

```
    ⎡ 4    2   −1 ⎤
Σ = ⎢ 2    9    0 ⎥
    ⎣−1    0    1 ⎦
```

**(a)** What are Var(X₁), Var(X₂), Var(X₃)?
**(b)** Which pairs of features are uncorrelated?
**(c)** Compute ρ(X₁,X₂) and ρ(X₁,X₃)
**(d)** Is the matrix valid (PSD)?

**Solution:**

**(a)** Diagonal entries = variances:
```
Var(X₁) = 4,   SD(X₁) = 2
Var(X₂) = 9,   SD(X₂) = 3
Var(X₃) = 1,   SD(X₃) = 1
```

**(b)** Off-diagonal = covariances:
- Cov(X₁,X₂) = 2 ≠ 0 → **correlated**
- Cov(X₁,X₃) = −1 ≠ 0 → **correlated**
- Cov(X₂,X₃) = 0 → **uncorrelated** (X₂ and X₃)

**(c)**
```
ρ(X₁,X₂) = 2/(2×3) = 2/6 = 0.333
ρ(X₁,X₃) = −1/(2×1) = −0.5
```

**(d)** For PSD, all eigenvalues must be ≥ 0. Quick check: all diagonal entries positive ✓, diagonal dominance approximately satisfied ✓. (Full eigenvalue computation would confirm.)

**ML insight:** This is the sample covariance matrix you'd compute in PCA. X₂ and X₃ being uncorrelated means they contribute independent information — good for feature diversity. X₁ and X₃ being negatively correlated (ρ=−0.5) means they partly cancel — a regularization technique in some contexts.

---

### 🔢 Numerical 4 — Zero Covariance ≠ Independence

**Problem:** Let X ~ Uniform(−1, 1) and Y = X².

**(a)** Show Cov(X,Y) = 0
**(b)** Show X and Y are clearly dependent

**Solution:**

**(a)**
```
E[X] = 0                          [Uniform(−1,1) is symmetric]
E[Y] = E[X²] = Var(X) = (2)²/12 = 1/3    [Var of Uniform(−1,1)]
E[XY] = E[X·X²] = E[X³] = ∫₋₁¹ x³·(1/2) dx = 0    [odd function, symmetric interval]

Cov(X,Y) = E[XY] − E[X]E[Y] = 0 − 0×(1/3) = 0
```

**(b)** Yet X and Y are completely dependent — knowing X = 0.5 tells you Y = 0.25 exactly. Y is a deterministic function of X!

**ML lesson:** Zero correlation only rules out **linear** dependence. Nonlinear relationships (like Y=X²) are invisible to Pearson correlation. This is why:
- Feature selection based only on correlation with target misses nonlinear predictors
- Neural networks can capture nonlinear dependencies that correlation analysis would miss
- Mutual information (Day 28) is needed to detect any dependence

---

### 🔢 Numerical 5 — PCA from Scratch via Covariance

**Problem:** Dataset with 2 features, 4 samples:

```
X = [(2,1), (4,3), (1,1), (3,2)]    (x₁, x₂) pairs
```

**(a)** Center the data (subtract mean)
**(b)** Compute the 2×2 sample covariance matrix
**(c)** Find the direction of maximum variance (first PC)

**Solution:**

**(a) Center:**
```
Mean: x̄₁ = (2+4+1+3)/4 = 2.5,   x̄₂ = (1+3+1+2)/4 = 1.75

Centered data:
(−0.5, −0.75), (1.5, 1.25), (−1.5, −0.75), (0.5, 0.25)
```

**(b) Covariance matrix (divide by n−1=3):**

```
Var(X₁) = (0.25+2.25+2.25+0.25)/3 = 5.0/3 ≈ 1.667
Var(X₂) = (0.5625+1.5625+0.5625+0.0625)/3 = 2.75/3 ≈ 0.917
Cov(X₁,X₂) = [(−0.5)(−0.75)+(1.5)(1.25)+(−1.5)(−0.75)+(0.5)(0.25)]/3
            = [0.375+1.875+1.125+0.125]/3 = 3.5/3 ≈ 1.167

    ⎡1.667  1.167⎤
Σ = ⎢            ⎥
    ⎣1.167  0.917⎦
```

**(c) First PC** = eigenvector of Σ with largest eigenvalue.

Eigenvalues: det(Σ − λI) = 0
```
(1.667−λ)(0.917−λ) − 1.167² = 0
λ² − 2.584λ + (1.529 − 1.362) = 0
λ² − 2.584λ + 0.167 = 0
λ = (2.584 ± √(6.677−0.668))/2 = (2.584 ± 2.451)/2
λ₁ = 2.518, λ₂ = 0.067
```

For λ₁ = 2.518, solve (Σ − λ₁I)v = 0:
```
(1.667−2.518)v₁ + 1.167v₂ = 0
−0.851v₁ + 1.167v₂ = 0
v₁/v₂ = 1.167/0.851 ≈ 1.371
```

Normalized: **PC₁ ≈ (0.808, 0.589)** — roughly the diagonal direction.

Variance explained: λ₁/(λ₁+λ₂) = 2.518/2.585 = **97.4%**

One principal component explains 97.4% of variance — this data is nearly 1-dimensional!

**ML insight:** When the first few eigenvalues dominate, PCA dramatically reduces dimensionality with minimal information loss. The eigenvalues of the covariance matrix ARE the variances along each principal component direction.

---

### 🔢 Numerical 6 — Correlation in Feature Selection

**Problem:** Target Y and 4 features. Correlation with Y:

| Feature | ρ(Xᵢ, Y) | Note |
|---|---|---|
| X₁ | 0.72 | |
| X₂ | 0.68 | |
| X₃ | −0.61 | |
| X₄ | 0.05 | |

Inter-feature correlations:
- ρ(X₁, X₂) = 0.95 (highly correlated)
- ρ(X₁, X₃) = −0.30
- ρ(X₂, X₃) = −0.28

**(a)** Which feature would you drop due to redundancy?
**(b)** Which features form the best subset?
**(c)** Why might X₄ still be useful despite ρ(X₄,Y)=0.05?

**Solution:**

**(a)** X₁ and X₂ are highly correlated (ρ=0.95) — nearly redundant. Since X₁ has slightly higher ρ with Y, **drop X₂**.

**(b)** Best subset: {X₁, X₃} — both correlate with Y, have low inter-correlation (−0.30), providing complementary information.

**(c)** X₄ could still be useful if:
- It has a **nonlinear** relationship with Y (zero Pearson ρ doesn't rule this out — Numerical 4)
- It **interacts** with other features (X₁·X₄ might predict Y well)
- It helps in a specific subgroup even if overall correlation is low

**ML lesson:** Pearson correlation is a starting point for feature selection, not the end. Mutual information, SHAP values, and model-based feature importance capture what correlation misses.

---

### 🔢 Numerical 7 — Multivariate Normal: Conditional Distribution

**Problem:** (X₁, X₂) ~ MVN with:
```
μ = (3, 5)ᵀ

    ⎡4  2⎤
Σ = ⎢    ⎥
    ⎣2  9⎦
```

Find the conditional distribution of X₁ | X₂ = 7.

**Solution:**

For MVN, the conditional X₁|X₂=x₂ is also Normal with:

```
μ_{1|2} = μ₁ + Σ₁₂/Σ₂₂ · (x₂ − μ₂)
σ²_{1|2} = Σ₁₁ − Σ₁₂²/Σ₂₂
```

Here: μ₁=3, μ₂=5, Σ₁₁=4, Σ₁₂=2, Σ₂₂=9, x₂=7

```
μ_{1|2} = 3 + (2/9)·(7−5) = 3 + (2/9)·2 = 3 + 0.444 = 3.444

σ²_{1|2} = 4 − 2²/9 = 4 − 4/9 = 32/9 ≈ 3.556
```

**X₁ | X₂=7 ~ N(3.444, 3.556)**

**Interpretation:**
- Marginal: X₁ ~ N(3, 4)
- After observing X₂=7 (which is above its mean of 5), X₁'s expected value increases from 3 to 3.444 — because they're positively correlated (Cov=2)
- The conditional variance (3.556) is less than the marginal variance (4) — knowing X₂ reduces our uncertainty about X₁

**ML insight:** This is **Gaussian Process regression** in miniature. Given observations of one variable, you update your belief about another using the covariance structure. The conditional mean formula is also the foundation of the Kalman filter (used in tracking, robot navigation, and time series forecasting).

---

## 9. Spurious Correlations — A Warning for ML

Some famous spurious correlations (real data):
- Nicolas Cage films per year ↔ pool drownings: ρ ≈ 0.67
- Per capita cheese consumption ↔ deaths by bedsheet tangling: ρ ≈ 0.95
- Internet Explorer market share ↔ US murder rate: ρ ≈ 0.97

**Why they arise in ML:**
- Small datasets → high-variance correlation estimates
- Multiple testing → some correlations are significant by chance
- Shared trends (both increase over time) → confounding
- Selection bias → Berkson's paradox

**Defense:** Use holdout validation, causal reasoning, domain knowledge. Correlation is a description, not an explanation.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is covariance and how does it differ from correlation?" | Cov has units, ρ is unitless and in [−1,1] |
| "Can Cov(X,Y)=0 but X,Y dependent?" | Yes — Y=X² example. ρ=0 only rules out linear dependence |
| "What is the covariance matrix and why is it PSD?" | Σᵢⱼ=Cov(Xᵢ,Xⱼ); PSD because vᵀΣv = Var(vᵀX) ≥ 0 |
| "How does PCA relate to the covariance matrix?" | PCA = eigendecomposition of Σ; PCs = eigenvectors; variance = eigenvalues |
| "Var(X+Y) when X,Y are correlated?" | Var(X)+Var(Y)+2Cov(X,Y) |
| "Why are diverse ensemble members better?" | Low correlation between errors → low ensemble variance |
| "What is the conditional distribution of MVN?" | Also Normal — mean shifts, variance shrinks |
| "What does ρ=0.95 between features imply for regression?" | Multicollinearity — unstable coefficients, use regularization or drop one |

---

## 11. Key Formulas — Cheat Sheet for Day 13

```
Covariance:
    Cov(X,Y) = E[(X−μₓ)(Y−μᵧ)] = E[XY] − E[X]E[Y]
    Cov(X,X) = Var(X)
    Cov(aX,bY) = ab·Cov(X,Y)
    Cov(X+Z,Y) = Cov(X,Y) + Cov(Z,Y)

Variance of sum:
    Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)
    Var(aX+bY) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y)

Correlation:
    ρ(X,Y) = Cov(X,Y) / (σₓ·σᵧ)
    −1 ≤ ρ ≤ 1
    ρ scale-invariant; Cov is not

Covariance matrix:
    Σᵢⱼ = Cov(Xᵢ,Xⱼ)
    Σ symmetric, PSD
    Σ diagonal ⟺ features uncorrelated

PCA:
    Σ = VΛVᵀ   (eigendecomposition)
    PC directions = columns of V
    Variances along PCs = eigenvalues Λ

Multivariate Normal X ~ MVN(μ,Σ):
    f(x) ∝ exp(−½(x−μ)ᵀΣ⁻¹(x−μ))
    Marginals: Xᵢ ~ N(μᵢ, Σᵢᵢ)
    For MVN only: Cov=0 ⟺ Independent

Conditional MVN (X₁|X₂=x₂):
    μ_{1|2} = μ₁ + Σ₁₂/Σ₂₂·(x₂−μ₂)
    σ²_{1|2} = Σ₁₁ − Σ₁₂²/Σ₂₂

Independence vs uncorrelated:
    Independent → Cov=0 (always)
    Cov=0 → Independent (ONLY for jointly Normal)
```

---

## 12. Practice Problems (Solve Before Day 14)

1. X ~ N(0,1) and Y = X² (not jointly Normal). Show Cov(X,Y)=0 but X and Y are dependent.

2. Portfolio of two assets: E[R₁]=0.10, E[R₂]=0.07, Var(R₁)=0.04, Var(R₂)=0.02, ρ(R₁,R₂)=0.3. For portfolio P = 0.6R₁ + 0.4R₂, find E[P] and SD[P].

3. Three model errors E₁,E₂,E₃ each have Var=1. Pairwise correlations: ρ(E₁,E₂)=0.8, ρ(E₁,E₃)=0.1, ρ(E₂,E₃)=0.1. Ensemble = (E₁+E₂+E₃)/3. Which pairs are most damaging to ensemble performance? Compute Var(ensemble).

4. Data: (X,Y) pairs: (1,2),(2,4),(3,5),(4,4),(5,5). Compute ρ(X,Y) from scratch.

5. *(Interview-level)* Explain why PSD is a required property of covariance matrices. What goes wrong statistically if a covariance matrix has a negative eigenvalue? How can this happen numerically and how do you fix it?

---

## 13. Looking Ahead

**Day 14** — **Unit 2 Review + Hard Problems on Distributions.** We consolidate all random variable theory (Days 7–13) with the hardest interview problems: distribution identification, moment matching, distribution of transformations, and 5 complete ML system design problems that require choosing and reasoning about distributions.

---
*End of Day 13 | Next: Day 14 — Unit 2 Review & Hard Interview Problems*
