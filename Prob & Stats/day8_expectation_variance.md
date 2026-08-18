# Day 8 — Expectation, Variance & Standard Deviation
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 4 (Sections 4.1–4.4)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Expectation and Variance are the Core of ML

Every summary statistic, every loss function, every optimization objective in ML reduces to expectation and variance.

| ML Concept | Expectation / Variance |
|---|---|
| Mean Squared Error (MSE) | E[(Y − Ŷ)²] |
| Cross-entropy loss | −E[log P(Y\|X)] |
| Bias of an estimator | E[θ̂] − θ |
| Variance of an estimator | Var(θ̂) |
| Bias-variance tradeoff | MSE = Bias² + Variance |
| Signal-to-noise ratio | E[X] / √Var(X) |
| Batch normalization | Normalize by E[X] and Var(X) |
| Gradient variance | Var(∇L) — key to optimizer design |
| Returns in RL | E[Σ γᵗ rₜ] — expected cumulative reward |

Once you truly understand expectation and variance, the mathematics of ML becomes transparent.

---

## 2. Expectation (Expected Value)

> **Definition:** The **expected value** (or expectation, or mean) of a discrete random variable X is:
> ```
> E[X] = Σₓ x · P(X = x) = Σₓ x · p(x)
> ```

**Intuition:** A weighted average of all possible values, weighted by their probabilities. It's the **long-run average** if you repeated the experiment infinitely many times.

### Alternative Notation

```
E[X] = μ = μₓ = ⟨X⟩
```

All mean the same thing. In ML, μ is most common for the mean.

### When Expectation Doesn't Exist

E[X] exists only if Σₓ |x| · p(x) < ∞. Heavy-tailed distributions (like Cauchy) have no finite mean — important for understanding why some optimization landscapes are pathological.

---

## 3. Properties of Expectation

These are used constantly — know them cold.

### Property 1 — Linearity of Expectation (THE most powerful property)

```
E[aX + bY + c] = a·E[X] + b·E[Y] + c
```

for any constants a, b, c and **any** random variables X and Y.

**Critical:** This holds **even if X and Y are dependent.** No independence required.

**Why it's powerful:** It lets you break complex expectations into simple pieces.

### Property 2 — Expectation of a Constant

```
E[c] = c     for any constant c
```

### Property 3 — Expectation of a Function (LOTUS)

> **Law of the Unconscious Statistician (LOTUS):**
> ```
> E[g(X)] = Σₓ g(x) · p(x)
> ```

You do NOT need the PMF of g(X) — just apply g to each value of X and weight by p(x).

**Example:** E[X²] = Σₓ x² · p(x) — you don't need the PMF of X².

### Property 4 — Expectation of Independent Product

If X and Y are **independent**:
```
E[XY] = E[X] · E[Y]
```

**Warning:** This requires independence. In general, E[XY] ≠ E[X]·E[Y].

---

## 4. Variance

> **Definition:** The **variance** of X measures how spread out the distribution is:
> ```
> Var(X) = E[(X − E[X])²] = E[X²] − (E[X])²
> ```

**Notation:** Var(X) = σ² = σₓ²

### The Two Formulas for Variance

**Formula 1 (definition):**
```
Var(X) = E[(X − μ)²] = Σₓ (x − μ)² · p(x)
```

**Formula 2 (computational — easier to use):**
```
Var(X) = E[X²] − (E[X])²
```

**Proof:**
```
Var(X) = E[(X−μ)²]
       = E[X² − 2μX + μ²]
       = E[X²] − 2μE[X] + μ²       [linearity]
       = E[X²] − 2μ² + μ²           [E[X] = μ]
       = E[X²] − μ²
       = E[X²] − (E[X])²  ∎
```

**Memory trick:** Var(X) = "mean of squares minus square of mean"

---

## 5. Standard Deviation

> **Definition:**
> ```
> SD(X) = σ = √Var(X)
> ```

Variance is in **squared units** (if X is in meters, Var(X) is in meters²). Standard deviation brings it back to the original units — more interpretable.

**In ML:** When you standardize features: (X − μ)/σ, you're dividing by standard deviation so features have unit variance.

---

## 6. Properties of Variance

### Property 1 — Variance of a Linear Transformation

```
Var(aX + b) = a² · Var(X)
```

**Why:** Adding b shifts the distribution (doesn't change spread). Multiplying by a scales the spread by |a|, so variance scales by a².

**Corollary:** SD(aX + b) = |a| · SD(X)

### Property 2 — Variance of a Sum (Independent)

If X and Y are **independent**:
```
Var(X + Y) = Var(X) + Var(Y)
Var(X − Y) = Var(X) + Var(Y)    [subtraction also ADDS variances!]
```

**Warning:** This requires independence. The general formula involves covariance (Day 13).

### Property 3 — Variance is Non-negative

```
Var(X) ≥ 0,    with equality iff X is a constant
```

### Property 4 — Variance of a Constant

```
Var(c) = 0
```

---

## 7. Standardization

> **Definition:** The **standardized** version of X is:
> ```
> Z = (X − μ) / σ
> ```

Properties of Z:
```
E[Z] = 0        [zero mean]
Var(Z) = 1      [unit variance]
```

**Proof:**
```
E[Z] = E[(X−μ)/σ] = (E[X] − μ)/σ = 0/σ = 0
Var(Z) = Var((X−μ)/σ) = (1/σ²)·Var(X) = σ²/σ² = 1
```

**ML ubiquity:** Feature standardization, batch normalization, layer normalization all use this transformation. It prevents features with large scales from dominating gradient updates.

---

## 8. Moments

> **Definition:** The **k-th moment** of X is E[Xᵏ].

| Moment | Formula | Meaning |
|---|---|---|
| 1st moment | E[X] = μ | Mean (location) |
| 2nd moment | E[X²] | Used to compute variance |
| Variance | E[X²] − μ² | Spread |
| 3rd central moment | E[(X−μ)³] | Skewness (asymmetry) |
| 4th central moment | E[(X−μ)⁴] | Kurtosis (tail heaviness) |

### Skewness

```
Skewness = E[(X−μ)³] / σ³
```
- Positive skew: long right tail (income distributions, loss values)
- Negative skew: long left tail
- Zero: symmetric (Normal distribution)

### Kurtosis

```
Kurtosis = E[(X−μ)⁴] / σ⁴
Excess kurtosis = Kurtosis − 3    [Normal has kurtosis 3]
```
- High kurtosis: heavy tails, more outliers (bad for gradient descent)
- Low kurtosis: light tails, fewer extreme values

---

## 9. Bias-Variance Decomposition (The Most Important Formula in ML Theory)

This follows directly from the variance formula. For a model predicting Y from X, with estimator Ŷ:

```
E[(Y − Ŷ)²] = (E[Ŷ] − Y)² + Var(Ŷ)
     MSE    =    Bias²      + Variance
```

**Term by term:**
- **MSE** = E[(Y − Ŷ)²] — expected squared prediction error (what we minimize)
- **Bias** = E[Ŷ] − Y — systematic error (how far average prediction is from truth)
- **Variance** = Var(Ŷ) — how much predictions vary across different training sets

**Full derivation:**
```
E[(Y − Ŷ)²]
= E[(Y − E[Ŷ] + E[Ŷ] − Ŷ)²]                    [add/subtract E[Ŷ]]
= E[(Y − E[Ŷ])² + 2(Y−E[Ŷ])(E[Ŷ]−Ŷ) + (E[Ŷ]−Ŷ)²]
= (Y−E[Ŷ])² + 2(Y−E[Ŷ])·E[E[Ŷ]−Ŷ] + E[(E[Ŷ]−Ŷ)²]
= Bias² + 0 + Var(Ŷ)                              [E[E[Ŷ]−Ŷ] = 0]
```

**The tradeoff:**
- Simple models (linear): low variance, high bias — underfitting
- Complex models (deep nets): low bias, high variance — overfitting
- Regularization, dropout, early stopping all reduce variance at the cost of slight bias increase

---

## 10. Worked Numericals

---

### 🔢 Numerical 1 — Computing Expectation and Variance from PMF

**Problem:** X = number of bugs found in a code review. PMF:

| x | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| p(x) | 0.10 | 0.25 | 0.35 | 0.20 | 0.10 |

Find E[X], E[X²], Var(X), SD(X).

**Solution:**

**E[X]:**
```
E[X] = 0×0.10 + 1×0.25 + 2×0.35 + 3×0.20 + 4×0.10
     = 0 + 0.25 + 0.70 + 0.60 + 0.40
     = 1.95 bugs
```

**E[X²] (using LOTUS):**
```
E[X²] = 0²×0.10 + 1²×0.25 + 2²×0.35 + 3²×0.20 + 4²×0.10
      = 0 + 0.25 + 1.40 + 1.80 + 1.60
      = 5.05
```

**Var(X):**
```
Var(X) = E[X²] − (E[X])²
       = 5.05 − (1.95)²
       = 5.05 − 3.8025
       = 1.2475
```

**SD(X):**
```
SD(X) = √1.2475 ≈ 1.117 bugs
```

**Interpretation:** On average, 1.95 bugs are found per review, with a standard deviation of about 1.12 bugs.

---

### 🔢 Numerical 2 — Linearity of Expectation (No Independence Needed)

**Problem:** A data pipeline processes records. Each record independently:
- Takes 2ms to load with probability 0.7
- Takes 5ms to load with probability 0.3

100 records are processed sequentially. What is the expected total time?

**Solution:**

Let Xᵢ = load time of record i.

E[Xᵢ] = 2×0.7 + 5×0.3 = 1.4 + 1.5 = 3.1 ms

Total time T = X₁ + X₂ + ... + X₁₀₀

By **linearity of expectation**:
```
E[T] = E[X₁] + E[X₂] + ... + E[X₁₀₀]
     = 100 × 3.1
     = 310 ms
```

No need to compute the joint distribution of all 100 variables. Linearity does all the work.

**Variance (since records are independent):**
```
E[Xᵢ²] = 4×0.7 + 25×0.3 = 2.8 + 7.5 = 10.3
Var(Xᵢ) = 10.3 − (3.1)² = 10.3 − 9.61 = 0.69 ms²

Var(T) = 100 × 0.69 = 69 ms²
SD(T) = √69 ≈ 8.31 ms
```

So total time is 310ms ± 8.31ms (1 SD).

---

### 🔢 Numerical 3 — LOTUS: Expected Loss

**Problem:** A regression model's error E = Y − Ŷ has PMF:

| e | -2 | -1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| p(e) | 0.10 | 0.20 | 0.40 | 0.20 | 0.10 |

Find:
**(a)** E[E] — expected error (bias)
**(b)** E[E²] — expected squared error (MSE)
**(c)** E[|E|] — expected absolute error (MAE)
**(d)** Var(E)

**Solution:**

**(a) Bias:**
```
E[E] = (−2)×0.10 + (−1)×0.20 + 0×0.40 + 1×0.20 + 2×0.10
     = −0.20 − 0.20 + 0 + 0.20 + 0.20
     = 0
```
Zero bias — the model is unbiased on average. ✓

**(b) MSE (using LOTUS with g(e) = e²):**
```
E[E²] = 4×0.10 + 1×0.20 + 0×0.40 + 1×0.20 + 4×0.10
      = 0.40 + 0.20 + 0 + 0.20 + 0.40
      = 1.20
```
MSE = **1.20**

**(c) MAE (using LOTUS with g(e) = |e|):**
```
E[|E|] = 2×0.10 + 1×0.20 + 0×0.40 + 1×0.20 + 2×0.10
       = 0.20 + 0.20 + 0 + 0.20 + 0.20
       = 0.80
```
MAE = **0.80**

**(d) Variance:**
```
Var(E) = E[E²] − (E[E])² = 1.20 − 0² = 1.20
```

**ML insight:** MSE ≠ MAE in general. MSE penalizes large errors more (squared), MAE treats all errors equally. This is why MSE is sensitive to outliers but MAE is robust.

---

### 🔢 Numerical 4 — Variance of Linear Transformation (Feature Scaling)

**Problem:** Feature X has E[X] = 50, Var(X) = 100 (SD = 10).

You apply two transformations:
**(a)** Min-max scaling: Y = (X − 0) / 100 → X/100
**(b)** Standardization: Z = (X − 50) / 10

Find E[Y], Var(Y), E[Z], Var(Z).

**Solution:**

**(a) Min-max scaling: Y = X/100**
```
E[Y] = E[X/100] = E[X]/100 = 50/100 = 0.50
Var(Y) = Var(X/100) = (1/100)² × Var(X) = Var(X)/10000 = 100/10000 = 0.01
SD(Y) = 0.10
```

**(b) Standardization: Z = (X − 50)/10**
```
E[Z] = (E[X] − 50)/10 = 0/10 = 0
Var(Z) = Var(X)/10² = 100/100 = 1
SD(Z) = 1
```

Standardization always gives mean 0 and variance 1 — that's the point.

**ML insight:** Neural networks, SVMs, and distance-based models (KNN) are sensitive to feature scale. Min-max preserves relative scale but doesn't normalize variance. Standardization is generally preferred unless you need bounded [0,1] outputs.

---

### 🔢 Numerical 5 — Bias-Variance Tradeoff Numerically

**Problem:** Three models are evaluated 5 times on different training sets. True value Y = 10.

| Model | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|---|---|---|---|---|---|
| A (linear) | 8.1 | 8.3 | 7.9 | 8.2 | 8.0 |
| B (polynomial) | 9.8 | 10.3 | 9.6 | 10.4 | 10.2 |
| C (deep net) | 6.0 | 14.0 | 9.5 | 11.0 | 15.0 |

Compute Bias², Variance, and MSE for each model.

**Solution:**

**Model A:**
```
E[Ŷ] = (8.1+8.3+7.9+8.2+8.0)/5 = 40.5/5 = 8.1
Bias = E[Ŷ] − Y = 8.1 − 10 = −1.9,   Bias² = 3.61

Var(Ŷ) = E[Ŷ²] − (E[Ŷ])²
Predictions: 8.1,8.3,7.9,8.2,8.0
E[Ŷ²] = (65.61+68.89+62.41+67.24+64)/5 = 328.15/5 = 65.63
Var(Ŷ) = 65.63 − 8.1² = 65.63 − 65.61 = 0.02

MSE = Bias² + Var = 3.61 + 0.02 = 3.63
```

**Model B:**
```
E[Ŷ] = (9.8+10.3+9.6+10.4+10.2)/5 = 50.3/5 = 10.06
Bias = 10.06 − 10 = 0.06,   Bias² = 0.0036

Predictions variance:
Deviations from mean: -0.26, 0.24, -0.46, 0.34, 0.14
Var = (0.0676+0.0576+0.2116+0.1156+0.0196)/5 = 0.4720/5 = 0.094

MSE = 0.0036 + 0.094 = 0.098
```

**Model C:**
```
E[Ŷ] = (6+14+9.5+11+15)/5 = 55.5/5 = 11.1
Bias = 11.1 − 10 = 1.1,   Bias² = 1.21

Deviations from mean 11.1: -5.1, 2.9, -1.6, -0.1, 3.9
Var = (26.01+8.41+2.56+0.01+15.21)/5 = 52.2/5 = 10.44

MSE = 1.21 + 10.44 = 11.65
```

**Summary:**

| Model | Bias² | Variance | MSE | Diagnosis |
|---|---|---|---|---|
| A (linear) | 3.61 | 0.02 | 3.63 | High bias, low variance — **underfitting** |
| B (poly) | 0.004 | 0.094 | 0.098 | Low bias, low variance — **best model** |
| C (deep net) | 1.21 | 10.44 | 11.65 | High variance — **overfitting** |

Model B wins — it finds the sweet spot. Model A is too simple (misses the pattern). Model C memorizes noise (its predictions swing wildly across training sets).

---

### 🔢 Numerical 6 — Expected Value in Reinforcement Learning

**Problem:** An RL agent receives rewards over 3 time steps. At each step:
- With probability 0.6: reward = +1
- With probability 0.4: reward = −1

With discount factor γ = 0.9, the return is:
```
G = R₁ + γR₂ + γ²R₃
```

Find E[G] and Var(G).

**Solution:**

Each Rₜ has:
E[Rₜ] = 1×0.6 + (−1)×0.4 = 0.6 − 0.4 = **0.2**
E[Rₜ²] = 1²×0.6 + (−1)²×0.4 = 1.0
Var(Rₜ) = E[Rₜ²] − (E[Rₜ])² = 1.0 − 0.04 = **0.96**

**E[G] by linearity:**
```
E[G] = E[R₁] + γ·E[R₂] + γ²·E[R₃]
     = 0.2 + 0.9×0.2 + 0.81×0.2
     = 0.2 + 0.18 + 0.162
     = 0.542
```

**Var(G) — rewards are independent:**
```
Var(G) = Var(R₁) + γ²·Var(R₂) + γ⁴·Var(R₃)
       = 0.96 + 0.81×0.96 + 0.6561×0.96
       = 0.96 + 0.7776 + 0.6299
       = 2.367
```
SD(G) = √2.367 ≈ **1.538**

**ML insight:** The variance of the return is large relative to its mean (SD ≈ 3× the mean). This high variance is why RL training is unstable — it's why techniques like advantage estimation (A2C), baseline subtraction, and PPO exist. They all reduce Var(G) to stabilize training.

---

### 🔢 Numerical 7 — Expectation via Indicator Variables

**Problem:** A model is evaluated on 50 test samples. Each sample independently has:
- P(correct) = 0.80

**(a)** What is the expected number of correct predictions?
**(b)** What is the variance in the number correct?
**(c)** Within what range does the count fall with ~95% probability? (Use ±2 SD)

**Solution:**

Let Xᵢ = 1 if sample i is correct, 0 otherwise. Xᵢ ~ Bernoulli(0.80).

T = Σᵢ Xᵢ = total correct ~ Binomial(50, 0.80)

**(a)**
```
E[T] = Σᵢ E[Xᵢ] = 50 × 0.80 = 40 correct
```

**(b)**
```
Var(Xᵢ) = p(1−p) = 0.80 × 0.20 = 0.16
Var(T) = 50 × 0.16 = 8.0
SD(T) = √8 ≈ 2.83
```

*(Note: Var(Bernoulli(p)) = p(1−p) — derived in Day 9.)*

**(c)**
```
Range: E[T] ± 2·SD(T) = 40 ± 2×2.83 = 40 ± 5.66
     = [34.34, 45.66]
     ≈ [35, 45] correct predictions
```

With ~95% probability, between 35 and 45 of the 50 samples are correctly classified.

**ML insight:** This is why you need enough test samples to get reliable accuracy estimates. With only 10 samples, SD = √(10×0.16) ≈ 1.26, so your accuracy estimate has uncertainty of ±25.2% — essentially useless. With 1000 samples, SD ≈ 12.6, uncertainty ≈ ±1.26% — much more reliable.

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the bias-variance tradeoff?" | MSE = Bias² + Variance; simple models → high bias; complex → high variance |
| "Why does linearity of expectation hold even for dependent variables?" | It follows directly from the definition: E[X+Y] = Σ(x+y)p(x,y) = E[X]+E[Y] |
| "What is LOTUS?" | E[g(X)] = Σ g(x)p(x) — use X's PMF, not g(X)'s PMF |
| "What is Var(X+Y) when X,Y are dependent?" | Var(X)+Var(Y)+2Cov(X,Y) — Day 13 |
| "Why is MSE sensitive to outliers but MAE is not?" | MSE squares errors; large errors are penalized quadratically |
| "What does batch normalization do mathematically?" | Subtracts E[X], divides by SD(X) per mini-batch — standardizes activations |
| "What is an unbiased estimator?" | E[θ̂] = θ — expected value equals true parameter |
| "Why do we use variance of gradients in Adam optimizer?" | Adam tracks E[g] and E[g²] per parameter to adaptively scale learning rates |

---

## 12. Key Formulas — Cheat Sheet for Day 8

```
Expectation:
    E[X] = Σₓ x·p(x)
    E[c] = c
    E[aX+b] = a·E[X] + b

Linearity (always, no independence needed):
    E[X + Y] = E[X] + E[Y]
    E[aX + bY + c] = a·E[X] + b·E[Y] + c

LOTUS:
    E[g(X)] = Σₓ g(x)·p(x)

Independence only:
    E[XY] = E[X]·E[Y]

Variance:
    Var(X) = E[(X−μ)²] = E[X²] − (E[X])²
    Var(c) = 0
    Var(aX+b) = a²·Var(X)
    Var(X+Y) = Var(X)+Var(Y)    [independent only]

Standard Deviation:
    SD(X) = σ = √Var(X)

Standardization:
    Z = (X−μ)/σ   →   E[Z]=0, Var(Z)=1

Bernoulli(p):
    E[X] = p,   Var(X) = p(1−p)

Bias-Variance Decomposition:
    MSE = Bias² + Variance
    Bias = E[Ŷ] − Y
    Variance = Var(Ŷ)

Skewness:   E[(X−μ)³]/σ³
Kurtosis:   E[(X−μ)⁴]/σ⁴   (Normal = 3, excess kurtosis = 0)
```

---

## 13. Practice Problems (Solve Before Day 9)

1. X has PMF: P(X=1)=0.2, P(X=2)=0.5, P(X=3)=0.3. Find E[X], Var(X), E[2X+1], Var(2X+1).

2. **Prove** that Var(X) ≥ 0 using the definition Var(X) = E[(X−μ)²].

3. A model predicts a salary. True salary Y = 60,000. Two models have predictions over 4 runs:
   - Model A: 58k, 59k, 61k, 62k
   - Model B: 50k, 55k, 65k, 70k
   Compute Bias², Variance, MSE for each. Which would you deploy?

4. In a Naive Bayes classifier, you multiply 20 probabilities each equal to 0.8. Compute the product directly, then compute log of the product. Why is the log version numerically preferred?

5. *(Interview-level)* The Adam optimizer maintains:
   - mₜ = β₁·mₜ₋₁ + (1−β₁)·gₜ         (first moment — mean of gradients)
   - vₜ = β₂·vₜ₋₁ + (1−β₂)·gₜ²        (second moment — mean of squared gradients)
   
   Explain in terms of expectation and variance what mₜ and vₜ estimate. Why is vₜ − mₜ² used to estimate gradient variance?

---

## 14. Looking Ahead

**Day 9** — **Bernoulli, Binomial & Geometric Distributions.** We formalize the three most important discrete distributions in ML — the building blocks of binary classification, A/B testing, and sequential decision problems.

---
*End of Day 8 | Next: Day 9 — Bernoulli, Binomial & Geometric Distributions*
