# Day 23 — MLE Deep Dives: Gaussian, Bernoulli & Poisson
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Casella & Berger Chapter 7; Bishop PRML Chapter 2
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why These Three Distributions Cover Most of ML

| Distribution | ML Model | Loss Function | When |
|---|---|---|---|
| **Gaussian** | Linear/polynomial regression | MSE (L2) | Continuous output |
| **Bernoulli** | Logistic regression, binary classification | Cross-entropy (binary) | Binary output |
| **Categorical** | Multiclass classification, language models | Cross-entropy (multi) | K-class output |
| **Poisson** | Count regression, NLP word models | Poisson deviance | Count output |

Every neural network output layer + loss function is secretly MLE for one of these distributions. Today we derive everything from scratch.

---

## 2. MLE for Gaussian — Linear Regression Fully Derived

### Setup

Data: {(x₁,y₁), ..., (xₙ,yₙ)}

Model: yᵢ = wᵀxᵢ + εᵢ where εᵢ ~ i.i.d. N(0,σ²)

Equivalently: yᵢ|xᵢ ~ N(wᵀxᵢ, σ²)

### Full Log-Likelihood

```
f(yᵢ|xᵢ;w,σ²) = (1/√(2πσ²)) exp(−(yᵢ−wᵀxᵢ)²/(2σ²))

ℓ(w,σ²) = Σᵢ log f(yᵢ|xᵢ;w,σ²)
          = −n/2·log(2π) − n/2·log(σ²) − 1/(2σ²)·Σᵢ(yᵢ−wᵀxᵢ)²
```

### MLE for w (fixing σ²)

```
argmax_w ℓ(w,σ²) = argmax_w [−1/(2σ²)·Σᵢ(yᵢ−wᵀxᵢ)²]
                 = argmin_w Σᵢ(yᵢ−wᵀxᵢ)²
                 = argmin_w RSS(w)
```

**MLE for w = Ordinary Least Squares (OLS).**

Minimizing MSE loss IS MLE under Gaussian noise. The σ² cancels — it doesn't affect which w minimizes the sum.

### Closed-Form Solution

Setting ∇_w ℓ = 0:
```
∇_w ℓ = −1/σ² · Xᵀ(Xw − y) = 0

XᵀXw = Xᵀy

ŵ_MLE = (XᵀX)⁻¹Xᵀy    [Normal equations]
```

This is the **closed-form OLS estimator** — exact, no iteration needed (when XᵀX is invertible).

### MLE for σ²

```
dℓ/d(σ²) = −n/(2σ²) + Σ(yᵢ−wᵀxᵢ)²/(2σ⁴) = 0

σ̂²_MLE = (1/n)·Σᵢ(yᵢ−ŵᵀxᵢ)²  =  RSS/n
```

The MLE of σ² = average squared residual = MSE.

**Unbiased estimator:** s² = RSS/(n−p−1) divides by degrees of freedom.

### Why OLS = MLE Under Gaussian Noise

```
Gaussian noise ε ~ N(0,σ²)
  ↓
MLE objective: maximize ℓ(w)
  ↓
Equivalent to: minimize Σ(yᵢ−wᵀxᵢ)²
  ↓
This IS ordinary least squares
```

**Every linear regression with MSE loss is implicitly assuming Gaussian errors.**

---

## 3. MLE for Bernoulli — Logistic Regression Fully Derived

### Setup

Data: {(x₁,y₁), ..., (xₙ,yₙ)} where yᵢ ∈ {0,1}

Model: yᵢ|xᵢ ~ Bernoulli(p(xᵢ)) where p(xᵢ) = σ(wᵀxᵢ) = 1/(1+e^{−wᵀxᵢ})

### Why the Sigmoid?

We need p(xᵢ) ∈ [0,1]. Define the log-odds (logit):
```
log[p/(1−p)] = wᵀxᵢ    [linear in features]
```

Solving for p: p = e^{wᵀx}/(1+e^{wᵀx}) = 1/(1+e^{−wᵀx}) = **sigmoid**

The sigmoid is the natural inverse of the logit function — it converts the unrestricted linear score wᵀx to a probability.

### Log-Likelihood

```
P(yᵢ|xᵢ;w) = p(xᵢ)^{yᵢ} · (1−p(xᵢ))^{1−yᵢ}

ℓ(w) = Σᵢ [yᵢ log p(xᵢ) + (1−yᵢ)log(1−p(xᵢ))]
      = Σᵢ [yᵢ log σ(wᵀxᵢ) + (1−yᵢ) log(1−σ(wᵀxᵢ))]
      = Σᵢ [yᵢ·wᵀxᵢ − log(1+e^{wᵀxᵢ})]
```

(Using: yᵢ log σ(z) + (1−yᵢ)log(1−σ(z)) = yᵢz − log(1+eᶻ))

### Cross-Entropy = Negative Log-Likelihood

```
L_CE = −(1/n)ℓ(w) = (1/n)Σᵢ [−yᵢ log p̂ᵢ − (1−yᵢ)log(1−p̂ᵢ)]
```

**Binary cross-entropy loss IS the negative Bernoulli log-likelihood.**

### Gradient and Hessian

**Gradient (score function):**
```
∇_w ℓ = Σᵢ (yᵢ − p(xᵢ)) · xᵢ = Xᵀ(y − p)
```

Beautiful form: gradient = Σ(true − predicted) × feature.

**Hessian:**
```
H = ∇²_w ℓ = −Σᵢ p(xᵢ)(1−p(xᵢ)) · xᵢxᵢᵀ = −XᵀWX

where W = diag(p₁(1−p₁), ..., pₙ(1−pₙ))
```

Since W is positive definite, H is **negative definite** → log-likelihood is strictly concave → unique global maximum. ✓

### Newton-Raphson / IRLS

No closed form for logistic regression. Use iterative methods:

```
w ← w − H⁻¹∇_w ℓ
  = w + (XᵀWX)⁻¹Xᵀ(y−p)    [Iteratively Reweighted Least Squares]
```

Each Newton step solves a weighted least squares problem — elegant.

---

## 4. MLE for Categorical — Multiclass Classification

### Setup

K classes. yᵢ ∈ {1,...,K} (or one-hot encoded). Model outputs:
```
P(Y=k|x;W) = softmax(Wᵀx)_k = e^{wₖᵀx} / Σⱼ e^{wⱼᵀx}
```

### Log-Likelihood

```
ℓ(W) = Σᵢ Σₖ yᵢₖ · log P(Y=k|xᵢ;W)
      = Σᵢ Σₖ yᵢₖ · [wₖᵀxᵢ − log Σⱼ e^{wⱼᵀxᵢ}]
```

where yᵢₖ = 1 if sample i has class k, else 0.

### Cross-Entropy = Negative Categorical Log-Likelihood

```
L_CE = −(1/n)ℓ(W) = (1/n)Σᵢ Σₖ yᵢₖ · [−log P(Y=k|xᵢ;W)]
     = −(1/n)Σᵢ log P(Y=yᵢ|xᵢ;W)
```

**Categorical cross-entropy = negative log-likelihood of the true class.**

### Gradient

```
∇_{wₖ} ℓ = Σᵢ (yᵢₖ − P(Y=k|xᵢ)) · xᵢ = Xᵀ(yₖ − p̂ₖ)
```

Same beautiful form: gradient = Σ(true − predicted) × feature. Pattern holds for all exponential family models.

---

## 5. MLE for Poisson — Count Models

### Setup

Count data: y₁,...,yₙ ~ i.i.d. Poisson(λ)

### Log-Likelihood

```
log f(yᵢ;λ) = −λ + yᵢ log λ − log(yᵢ!)

ℓ(λ) = −nλ + (Σyᵢ) log λ − Σ log(yᵢ!)
```

### MLE: λ̂ = ȳ (sample mean)

```
dℓ/dλ = −n + Σyᵢ/λ = 0  →  λ̂ = ȳ
```

### Poisson Regression (GLM)

When λ depends on features: log(λᵢ) = wᵀxᵢ (log link)

```
λᵢ = exp(wᵀxᵢ)  →  P(yᵢ|xᵢ;w) = exp(−λᵢ)·λᵢ^{yᵢ}/yᵢ!

ℓ(w) = Σᵢ [yᵢ·wᵀxᵢ − exp(wᵀxᵢ) − log(yᵢ!)]

∇_w ℓ = Σᵢ (yᵢ − λᵢ)·xᵢ = Xᵀ(y − λ̂)
```

**Same gradient pattern**: Σ(true − predicted) × feature. This is a general property of Generalized Linear Models (GLMs).

---

## 6. The Exponential Family Unification

All three distributions (Gaussian, Bernoulli, Poisson) are members of the **exponential family**:

```
f(x;η) = h(x) · exp(ηᵀT(x) − A(η))
```

Where:
- η = natural parameter
- T(x) = sufficient statistic
- A(η) = log partition function

| Distribution | η | T(x) | A(η) |
|---|---|---|---|
| Bernoulli(p) | log(p/(1−p)) | x | log(1+eη) |
| Gaussian(μ,σ²) | μ/σ² | x | μ²/(2σ²) |
| Poisson(λ) | log λ | x | eη = λ |

**Key property:** For exponential family:
```
MLE: E[T(X)] = (1/n)Σ T(xᵢ)    [moments match]
∇_w ℓ = Σᵢ (T(yᵢ) − E[T(Y)|xᵢ]) · xᵢ = Σᵢ (yᵢ − ŷᵢ) · xᵢ
```

This explains the universal gradient form — it's a property of ALL exponential family GLMs.

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — Linear Regression MLE: Full Derivation

**Problem:** Data: (x,y) = {(1,2), (2,4), (3,5), (4,4), (5,5)}.

Fit y = w₀ + w₁x (linear regression with intercept).

**(a)** Set up the Normal equations.
**(b)** Solve for ŵ_MLE.
**(c)** Compute RSS and σ̂²_MLE.
**(d)** What Gaussian model does this correspond to?

**Solution:**

Design matrix X (with intercept column):
```
X = ⎡1  1⎤    y = ⎡2⎤
    ⎢1  2⎥        ⎢4⎥
    ⎢1  3⎥        ⎢5⎥
    ⎢1  4⎥        ⎢4⎥
    ⎣1  5⎦        ⎣5⎦
```

**(a)** Normal equations: XᵀXŵ = Xᵀy

```
XᵀX = ⎡n    Σxᵢ  ⎤ = ⎡5   15⎤
      ⎣Σxᵢ  Σxᵢ² ⎦   ⎣15  55⎦

Xᵀy = ⎡Σyᵢ     ⎤ = ⎡20⎤
      ⎣Σxᵢyᵢ   ⎦   ⎣62⎦
```

**(b)** Solve:
```
|XᵀX| = 5×55 − 15² = 275 − 225 = 50

(XᵀX)⁻¹ = (1/50)⎡55  −15⎤ = ⎡1.1   −0.3⎤
                  ⎣−15   5⎦   ⎣−0.3   0.1⎦

ŵ = (XᵀX)⁻¹Xᵀy = ⎡1.1   −0.3⎤⎡20⎤ = ⎡1.1×20−0.3×62⎤ = ⎡22−18.6⎤ = ⎡1.4⎤
                   ⎣−0.3   0.1⎦⎣62⎦   ⎣−0.3×20+0.1×62⎦   ⎣−6+6.2 ⎦   ⎣0.2⎦ ... 

Let me recompute carefully:
ŵ₀ = 1.1×20 + (−0.3)×62 = 22.0 − 18.6 = 3.4 ... 

Let me use the formulas directly:
ȳ = 20/5 = 4.0,   x̄ = 15/5 = 3.0
Σ(xᵢ−x̄)² = (1−3)²+(2−3)²+(3−3)²+(4−3)²+(5−3)² = 4+1+0+1+4 = 10
Σ(xᵢ−x̄)(yᵢ−ȳ) = (−2)(−2)+(−1)(0)+(0)(1)+(1)(0)+(2)(1) = 4+0+0+0+2 = 6

ŵ₁ = Σ(xᵢ−x̄)(yᵢ−ȳ)/Σ(xᵢ−x̄)² = 6/10 = 0.6
ŵ₀ = ȳ − ŵ₁x̄ = 4.0 − 0.6×3.0 = 4.0 − 1.8 = 2.2
```

**ŵ = (2.2, 0.6)**: ŷ = 2.2 + 0.6x

**(c)**

Predictions: 2.8, 3.4, 4.0, 4.6, 5.2
Residuals: −0.8, 0.6, 1.0, −0.6, −0.2

RSS = 0.64+0.36+1.00+0.36+0.04 = **2.40**

σ̂²_MLE = RSS/n = 2.40/5 = **0.480**
s² (unbiased) = RSS/(n−2) = 2.40/3 = **0.800**

**(d)** This corresponds to:
```
yᵢ ~ N(2.2 + 0.6·xᵢ, 0.480)
```

Conditional distribution of y given x is Normal with:
- Mean: the fitted line (2.2 + 0.6x)
- Variance: 0.480 (constant across all x — homoscedasticity assumption)

---

### 🔢 Numerical 2 — Logistic Regression MLE: One Step

**Problem:** Binary data with one feature. n=5 points:

| xᵢ | yᵢ |
|---|---|
| −2 | 0 |
| −1 | 0 |
| 0 | 1 |
| 1 | 1 |
| 2 | 1 |

**(a)** Compute log-likelihood at w=0 and w=1.
**(b)** Compute gradient at w=0.
**(c)** After one gradient step (η=0.1), what is new w?
**(d)** Interpret w=0 vs w=1.

**Solution:**

Model (no intercept for simplicity): p(xᵢ) = σ(w·xᵢ)

**(a) At w=0:** p(x) = σ(0) = 0.5 for all x.

```
ℓ(0) = Σᵢ [yᵢ log 0.5 + (1−yᵢ)log 0.5]
      = 5 × log(0.5) = 5×(−0.693) = −3.466
```

**At w=1:** 

| xᵢ | w·xᵢ | p=σ(w·x) | yᵢ | log contribution |
|---|---|---|---|---|
| −2 | −2 | 0.119 | 0 | log(0.881)=−0.127 |
| −1 | −1 | 0.269 | 0 | log(0.731)=−0.313 |
| 0  | 0  | 0.500 | 1 | log(0.500)=−0.693 |
| 1  | 1  | 0.731 | 1 | log(0.731)=−0.313 |
| 2  | 2  | 0.881 | 1 | log(0.881)=−0.127 |

ℓ(1) = −0.127−0.313−0.693−0.313−0.127 = **−1.573**

Since −1.573 > −3.466: **w=1 is better** (higher log-likelihood). ✓

**(b) Gradient at w=0:** p=0.5 for all x.

```
∇_w ℓ = Σᵢ (yᵢ − p(xᵢ))·xᵢ
       = (0−0.5)(−2) + (0−0.5)(−1) + (1−0.5)(0) + (1−0.5)(1) + (1−0.5)(2)
       = 1.0 + 0.5 + 0 + 0.5 + 1.0 = 3.0
```

**(c)** Gradient ascent step:
```
w ← w + η·∇_w ℓ = 0 + 0.1×3.0 = 0.3
```

**(d)** Interpretation:
- w=0: model outputs 50% probability regardless of x — completely uninformative
- w=1: model correctly assigns lower probability to negative class (x<0) and higher to positive class (x>0)
- Gradient = 3.0 > 0: w should increase (move toward positive w=1 direction) ✓

**ML insight:** The logistic regression gradient is the residual dot feature — the model increases w when features of positive examples are larger than features of negative examples. This is exactly what you'd want: features that predict the label drive the weight updates.

---

### 🔢 Numerical 3 — Multiclass MLE: Language Model Training

**Problem:** Vocabulary V={the, cat, sat}. A language model assigns:

For context "the":
- P(cat|the) = 0.6, P(sat|the) = 0.3, P(the|the) = 0.1

Training data contains bigrams: (the→cat), (the→cat), (the→sat), (the→the).

**(a)** Log-likelihood for this data.
**(b)** MLE for bigram probabilities.
**(c)** Cross-entropy loss.
**(d)** What if a word is never seen in training (zero probability problem)?

**Solution:**

**(a)** Using current model probabilities:
```
ℓ = log P(cat|the) + log P(cat|the) + log P(sat|the) + log P(the|the)
  = 2×log(0.6) + log(0.3) + log(0.1)
  = 2×(−0.511) + (−1.204) + (−2.303)
  = −1.022 − 1.204 − 2.303 = −4.529
```

**(b)** MLE bigram probabilities = count/total:

| Next word | Count | MLE probability |
|---|---|---|
| cat | 2 | 2/4 = **0.50** |
| sat | 1 | 1/4 = **0.25** |
| the | 1 | 1/4 = **0.25** |

MLE simply counts and normalizes — this is how n-gram language models are trained.

**(c)** Cross-entropy loss per token:

Using MLE probabilities:
```
ℓ_MLE = 2×log(0.5) + log(0.25) + log(0.25)
       = 2×(−0.693) + (−1.386) + (−1.386) = −4.158

L_CE = −ℓ_MLE/n = 4.158/4 = 1.040 nats
```

In bits: 1.040/log(2) = 1.500 bits/token. This is the perplexity-relevant quantity.

**(d)** Zero probability problem: if P(word|context)=0, log(0)=−∞. One word in test set → infinite loss.

**Solution: Laplace smoothing (add-k smoothing)**:
```
P_smooth(w|context) = (count(context,w) + k) / (count(context) + k×|V|)
```

For k=1, |V|=3:
- P_smooth(cat|the) = (2+1)/(4+3) = 3/7 ≈ 0.429
- P_smooth(sat|the) = (1+1)/(4+3) = 2/7 ≈ 0.286
- P_smooth(the|the) = (1+1)/(4+3) = 2/7 ≈ 0.286

**ML insight:** This is exactly what language model training faces. GPT and similar models use vocabulary sizes of 50,000+ with rare words — Laplace smoothing (or neural network estimation) prevents zero probabilities. The cross-entropy loss is the MLE objective applied to the language modeling task.

---

### 🔢 Numerical 4 — Poisson Regression MLE

**Problem:** Sales count data by advertising spend:

| Spend x | Sales y |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 8 |
| 4 | 12 |
| 5 | 18 |

Fit Poisson regression: log(λ) = w₀ + w₁x.

**(a)** Log-likelihood ℓ(w₀,w₁).
**(b)** Gradient ∇_w ℓ at (w₀,w₁)=(0,0).
**(c)** Interpret the gradient direction.
**(d)** After one gradient step (η=0.01), new parameters?

**Solution:**

**(a)**
```
ℓ(w) = Σᵢ [yᵢ·wᵀxᵢ − exp(wᵀxᵢ) − log(yᵢ!)]
      = Σᵢ [yᵢ(w₀+w₁xᵢ) − exp(w₀+w₁xᵢ)] + const
```

**(b) At w=(0,0):** λᵢ = exp(0) = 1 for all i.

Gradient w.r.t. w₀: Σᵢ(yᵢ − λᵢ)·1 = Σyᵢ − n = (2+3+8+12+18) − 5 = 43−5 = **38**

Gradient w.r.t. w₁: Σᵢ(yᵢ − λᵢ)·xᵢ = Σyᵢxᵢ − Σxᵢ
= (2×1+3×2+8×3+12×4+18×5) − (1+2+3+4+5)
= (2+6+24+48+90) − 15 = 170 − 15 = **155**

∇ℓ = (38, 155)

**(c)** Both gradient components are positive → increase both w₀ and w₁.

Makes sense: at (0,0), model predicts λ=1 for all x, but actual counts are much larger (average = 43/5 = 8.6). Model needs to predict higher counts → increase w₀ (intercept) and w₁ (slope).

**(d)**
```
w₀ ← 0 + 0.01×38 = 0.38
w₁ ← 0 + 0.01×155 = 1.55
```

New predictions: λᵢ = exp(0.38 + 1.55·xᵢ)
- x=1: exp(1.93) ≈ 6.9  [actual: 2 — still far off]
- x=5: exp(0.38+7.75) = exp(8.13) ≈ 3395  [actual: 18 — overshot badly!]

The large gradient step (1.55 for w₁) with η=0.01 overshoots. Need smaller learning rate or adaptive method.

**ML insight:** Poisson regression gradients can be very large when predictions are far from targets (here by factor of 8+). This is why adaptive optimizers (Adam) help — they scale the learning rate by the gradient history, preventing overshooting.

---

### 🔢 Numerical 5 — MLE with Missing Data: The EM Algorithm Preview

**Problem:** 100 coin flips from two coins (A and B), but you don't know which coin was used each flip.
- Coin A: P(H|A) = θ_A (unknown)
- Coin B: P(H|B) = θ_B (unknown)
- P(choose A) = 0.5

Observed: 60 heads, 40 tails (but don't know which coin each came from).

**(a)** Why can't you use standard MLE?
**(b)** Sketch the EM algorithm approach.
**(c)** If we somehow knew: 70 flips from A (45H, 25T), 30 from B (15H, 15T). Compute MLEs.

**Solution:**

**(a)** Standard MLE: ℓ(θ_A, θ_B) = Σᵢ log P(xᵢ; θ_A, θ_B). But:

```
P(Hᵢ; θ_A, θ_B) = 0.5·P(H|A) + 0.5·P(H|B) = 0.5θ_A + 0.5θ_B
```

This is a mixture — the log-likelihood has no closed form, and maximizing is hard because θ_A and θ_B are entangled.

**(b)** EM Algorithm:

**E-step:** Compute P(coin=A|flip i, current θ_A, θ_B) — posterior probability of which coin given the observation.

**M-step:** Update θ_A, θ_B using the "expected" complete-data log-likelihood (weighted by posteriors).

Repeat until convergence.

This is the **Expectation-Maximization (EM) algorithm** — MLE with latent (hidden) variables.

**(c)** With complete data:

Coin A: 45 heads, 25 tails → **θ̂_A = 45/70 = 0.643**
Coin B: 15 heads, 15 tails → **θ̂_B = 15/30 = 0.500**

Each coin's MLE is just its sample proportion — standard Bernoulli MLE per group.

**ML insight:** EM is used for:
- Gaussian Mixture Models (GMMs) — unsupervised clustering
- Hidden Markov Models — sequence labeling
- Missing data imputation
- K-means (as a degenerate EM)
- VAE training (approximate EM with neural networks)

All use the same idea: E-step (compute posterior over latent variables), M-step (MLE given posteriors).

---

### 🔢 Numerical 6 — MLE Comparison: L1 vs L2 Loss

**Problem:** Prove that L2 loss corresponds to Gaussian noise MLE and L1 loss corresponds to Laplace noise MLE.

**Setup:** yᵢ = wᵀxᵢ + εᵢ

**(a)** If εᵢ ~ N(0,σ²), what is the MLE objective?
**(b)** If εᵢ ~ Laplace(0,b), what is the MLE objective?
**(c)** Which is more robust to outliers and why?
**(d)** Numerical example: data = {1, 2, 3, 100} (one outlier). Compare L1 and L2 optimal predictions.

**Solution:**

**(a) Gaussian: L2 loss**

log f(ε; N(0,σ²)) = −ε²/(2σ²) + const

MLE: maximize Σᵢ [−(yᵢ−wᵀxᵢ)²/(2σ²)] = minimize **Σᵢ(yᵢ−wᵀxᵢ)²** [L2/MSE] ✓

**(b) Laplace: L1 loss**

Laplace(0,b) PDF: f(ε) = (1/2b)exp(−|ε|/b)

log f(ε; Laplace(0,b)) = −|ε|/b + const

MLE: maximize Σᵢ [−|yᵢ−wᵀxᵢ|/b] = minimize **Σᵢ|yᵢ−wᵀxᵢ|** [L1/MAE] ✓

**(c)** Robustness:

- L2: squared penalty — outlier with error=10 contributes 100× to loss. Outlier dominates gradient.
- L1: linear penalty — outlier with error=10 contributes only 10× to loss. Outlier contributes proportionally.

Gaussian has lighter tails than Laplace → L2 is more sensitive to outliers.

**(d)** Data = {1, 2, 3, 100}, predict constant w:

**L2 optimal (mean):**
```
ŵ_L2 = (1+2+3+100)/4 = 106/4 = 26.5
```
Pulled far by outlier.

**L1 optimal (median):**
```
Sorted: 1, 2, 3, 100
Median = (2+3)/2 = 2.5
```

L1 ignores the outlier — median is 2.5 vs mean 26.5.

**ML lesson:**
- Use MSE (L2): when you trust your data and want all errors to matter equally
- Use MAE (L1): when you have outliers and want robustness
- Use Huber loss: L2 near zero, L1 for large errors — best of both

This is why L1/L2 loss choice matters in practice — it's a distributional assumption about your noise.

---

### 🔢 Numerical 7 — MLE for Neural Networks: The Full Picture

**Problem:** A neural network with 2 inputs, 1 hidden layer (2 neurons, ReLU), 1 output (sigmoid):

```
z₁ = ReLU(w₁₁x₁ + w₁₂x₂)
z₂ = ReLU(w₂₁x₁ + w₂₂x₂)
ŷ = σ(v₁z₁ + v₂z₂)
```

Training data: (x=[1,0], y=1), (x=[0,1], y=0), (x=[1,1], y=1).

At current parameters: ŷ₁=0.7, ŷ₂=0.4, ŷ₃=0.8.

**(a)** Log-likelihood / cross-entropy loss.
**(b)** Connection to MLE.
**(c)** Why is backpropagation the gradient of the MLE objective?

**Solution:**

**(a)**
```
ℓ = log P(y₁=1|x₁) + log P(y₂=0|x₂) + log P(y₃=1|x₃)
  = log(0.7) + log(1−0.4) + log(0.8)
  = log(0.7) + log(0.6) + log(0.8)
  = −0.357 + (−0.511) + (−0.223)
  = −1.091

L_CE = −ℓ/3 = 1.091/3 = 0.364
```

**(b)** The neural network is computing:
```
P(Y=1|X=x; θ) = σ(output of network)
```

where θ = all weights {w₁₁, w₁₂, w₂₁, w₂₂, v₁, v₂}.

Training the neural network = finding θ that maximizes:
```
ℓ(θ) = Σᵢ [yᵢ log σ(fθ(xᵢ)) + (1−yᵢ)log(1−σ(fθ(xᵢ)))]
```

**This IS the Bernoulli log-likelihood.** Neural network training with binary cross-entropy = MLE for Bernoulli with a flexible (neural) parametrization of p(x;θ).

**(c)** Backpropagation computes ∇_θ ℓ(θ) — the gradient of the log-likelihood with respect to all parameters. This is:

1. Output layer: ∂ℓ/∂(output) = (y − ŷ) — residual (same as logistic regression gradient)
2. Hidden layer: chain rule propagates residuals backward
3. Each weight update: Δw ∝ ∂ℓ/∂w

Backpropagation IS gradient ascent on the MLE objective. Every parameter update in every neural network training run is a step of gradient ascent on a log-likelihood.

**The complete unification:**
```
Neural network training
= Minimize cross-entropy loss
= Maximize Bernoulli log-likelihood
= MLE for conditional distribution P(Y|X;θ)
= Approximating E[Y|X] (by Day 17)
```

All roads lead to MLE.

---

## 8. Summary: Loss Functions as MLE Objectives

| Loss Function | Distribution Assumed | MLE for |
|---|---|---|
| MSE = Σ(y−ŷ)² | y\|x ~ N(ŷ, σ²) | Regression |
| MAE = Σ\|y−ŷ\| | y\|x ~ Laplace(ŷ, b) | Robust regression |
| Binary CE = −Σ[y logŷ+(1−y)log(1−ŷ)] | y\|x ~ Bernoulli(ŷ) | Binary classification |
| Categorical CE = −Σ Σₖ yₖ log ŷₖ | y\|x ~ Categorical(ŷ) | Multiclass |
| Poisson deviance = Σ[ŷ−y logŷ] | y\|x ~ Poisson(ŷ) | Count regression |
| Huber loss | Mixture of Gaussian + Laplace | Robust regression |

---

## 9. Common Interview Questions

| Question | Key Idea |
|---|---|
| "Why is cross-entropy the right loss for classification?" | MLE for Bernoulli/Categorical likelihood |
| "Why is MSE the right loss for regression?" | MLE for Gaussian noise model |
| "What's the gradient of cross-entropy for logistic regression?" | Σ(yᵢ−p̂ᵢ)xᵢ — true minus predicted times feature |
| "What does the Normal equation solve?" | MLE for linear regression — closed form ŵ=(XᵀX)⁻¹Xᵀy |
| "Why is L1 more robust to outliers than L2?" | L1 = Laplace MLE (linear penalty); L2 = Gaussian MLE (quadratic penalty) |
| "What is the EM algorithm for?" | MLE with latent/missing variables |
| "What is backpropagation computing?" | Gradient of log-likelihood — MLE gradient |
| "Why does logistic regression have a unique maximum?" | Log-likelihood is strictly concave (negative definite Hessian) |

---

## 10. Key Formulas — Cheat Sheet for Day 23

```
LINEAR REGRESSION (Gaussian MLE):
    ŵ = (XᵀX)⁻¹Xᵀy              [Normal equations]
    σ̂² = RSS/n                   [MLE — biased]
    s² = RSS/(n−p−1)              [unbiased]
    MSE loss = Gaussian neg log-likelihood

LOGISTIC REGRESSION (Bernoulli MLE):
    p(x) = σ(wᵀx) = 1/(1+e^{−wᵀx})
    ℓ(w) = Σᵢ[yᵢwᵀxᵢ − log(1+e^{wᵀxᵢ})]
    ∇ℓ = Xᵀ(y−p)                  [error × feature]
    H = −XᵀWX  (negative definite → unique maximum)
    Cross-entropy = negative Bernoulli log-likelihood

MULTICLASS (Categorical MLE):
    P(k|x) = softmax(Wᵀx)_k
    ℓ(W) = Σᵢ Σₖ yᵢₖ log P(k|xᵢ)
    ∇_{wₖ}ℓ = Xᵀ(yₖ − p̂ₖ)

POISSON REGRESSION:
    λᵢ = exp(wᵀxᵢ)
    ∇ℓ = Xᵀ(y−λ̂)

UNIVERSAL GRADIENT PATTERN (Exponential Family):
    ∇ℓ = Σᵢ (yᵢ − ŷᵢ) · xᵢ

LOSS ↔ DISTRIBUTION:
    MSE ↔ Gaussian
    MAE ↔ Laplace
    Binary CE ↔ Bernoulli
    Categorical CE ↔ Categorical
    Poisson deviance ↔ Poisson
```

---

## 11. Practice Problems (Solve Before Day 24)

1. Linear regression: data (x,y) = {(0,1),(1,3),(2,2),(3,4)}. Compute ŵ_MLE using Normal equations. Compute RSS and σ̂².

2. Logistic regression: data (x,y) = {(−2,0),(0,0),(1,1),(3,1)}. At w=0.5, compute log-likelihood, gradient, and one gradient ascent step with η=0.2.

3. **Prove** that the Hessian of the logistic regression log-likelihood is −XᵀWX where W = diag(p₁(1−p₁),...,pₙ(1−pₙ)). Show this is negative semi-definite, implying a concave log-likelihood.

4. Count data: y={0,1,3,2,4,1,2}. Fit Poisson regression with one feature x={1,2,3,4,5,6,7}. At w=(w₀,w₁)=(0,0.3), compute predictions λ̂ᵢ, log-likelihood, and gradient.

5. *(Interview-level)* Huber loss:
   ```
   L_Huber(r) = r²/2         if |r| ≤ δ
              = δ|r| − δ²/2  if |r| > δ
   ```
   What distribution does Huber loss correspond to as a MLE objective? (Hint: It's a mixture — quadratic near 0 means Gaussian tails close to the mean, linear for large errors means Laplace heavy tails.) Derive the PDF of this distribution.

---

## 12. Looking Ahead

**Day 24** — **MAP Estimation & Bayesian Inference.** When MLE overfits (especially with small data), we add a prior on parameters — MAP estimation. We'll see how L2 regularization = Gaussian prior, L1 regularization = Laplace prior, and how the posterior distribution connects Bayesian inference to modern ML.

---
*End of Day 23 | Next: Day 24 — MAP Estimation & Bayesian Inference*
