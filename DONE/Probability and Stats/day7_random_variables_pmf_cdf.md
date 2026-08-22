# Day 7 — Random Variables: PMF & CDF
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 3
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Random Variables?

So far we've worked with events — "the email is spam", "the roll is even." But ML doesn't work with events. It works with **numbers**.

- A model outputs a score: 0.87
- A loss function returns a value: 2.34
- A feature takes a value: 142.5
- A label is encoded: 0 or 1

**Random variables are the bridge from events to numbers.** They let us do arithmetic, compute expectations, measure variance, and build distributions — all the machinery of ML.

| Without RVs | With RVs |
|---|---|
| P(email is spam) | P(Y = 1) |
| P(model is correct) | P(Ŷ = Y) |
| P(loss is high) | P(L > 3.0) |
| P(feature is large) | P(X > 100) |

---

## 2. Definition of a Random Variable

> **Definition:** A **random variable** X is a function from the sample space Ω to the real numbers ℝ:
> ```
> X : Ω → ℝ
> ```

It assigns a **number** to each outcome in the sample space.

### Example

Roll a die: Ω = {1, 2, 3, 4, 5, 6}

Define X = "value shown on die":
- X(1) = 1, X(2) = 2, ..., X(6) = 6

Define Y = "1 if even, 0 if odd":
- Y(1) = 0, Y(2) = 1, Y(3) = 0, Y(4) = 1, Y(5) = 0, Y(6) = 1

Both X and Y are random variables on the same sample space.

### Two Types of Random Variables

| Type | Values | Examples in ML |
|---|---|---|
| **Discrete** | Countable (finite or countably infinite) | Class label, word index, count of errors |
| **Continuous** | Uncountably infinite interval | Model weight, loss value, probability score |

Today: **Discrete** random variables (Days 11–12 cover continuous).

---

## 3. Probability Mass Function (PMF)

> **Definition:** The **PMF** of a discrete random variable X is:
> ```
> pₓ(x) = P(X = x)    for all x
> ```

It tells you the probability that X takes each specific value.

### PMF Requirements (from Kolmogorov Axioms)

```
1. pₓ(x) ≥ 0              for all x        [non-negativity]
2. Σₓ pₓ(x) = 1           sum over all x   [normalization]
3. P(X ∈ A) = Σ_{x∈A} pₓ(x)              [countable additivity]
```

### Example: Fair Die

X = value shown. PMF:

```
pₓ(x) = 1/6    for x ∈ {1, 2, 3, 4, 5, 6}
pₓ(x) = 0      otherwise
```

Verify: Σ pₓ(x) = 6 × (1/6) = 1 ✓

### Example: Biased Coin (Bernoulli)

Y = 1 (heads) with probability p, 0 (tails) with probability 1−p:

```
pᵧ(1) = p
pᵧ(0) = 1 − p
pᵧ(y) = 0    otherwise
```

This is the **Bernoulli(p)** distribution — the simplest and most important in ML (every binary classification output).

---

## 4. Cumulative Distribution Function (CDF)

> **Definition:** The **CDF** of a random variable X is:
> ```
> Fₓ(x) = P(X ≤ x)    for all x ∈ ℝ
> ```

It gives the probability that X is **at most** x.

### CDF Properties (valid for both discrete AND continuous RVs)

```
1. 0 ≤ F(x) ≤ 1                          [bounded]
2. F is non-decreasing: x ≤ y → F(x) ≤ F(y)
3. lim_{x→-∞} F(x) = 0                   [left limit = 0]
4. lim_{x→+∞} F(x) = 1                   [right limit = 1]
5. F is right-continuous: lim_{t→x⁺} F(t) = F(x)
```

### PMF from CDF and CDF from PMF

For discrete RVs, these are interconvertible:

```
CDF from PMF:   F(x) = Σ_{t ≤ x} p(t)         [sum up to x]
PMF from CDF:   p(x) = F(x) − F(x⁻)           [jump at x]
```

where F(x⁻) = lim_{t→x⁻} F(t) is the left limit.

### Example: Fair Die CDF

```
F(x) = 0       for x < 1
F(x) = 1/6     for 1 ≤ x < 2
F(x) = 2/6     for 2 ≤ x < 3
F(x) = 3/6     for 3 ≤ x < 4
F(x) = 4/6     for 4 ≤ x < 5
F(x) = 5/6     for 5 ≤ x < 6
F(x) = 1       for x ≥ 6
```

This is a **staircase function** — jumps at each value in the support, flat between.

---

## 5. Functions of Random Variables

If X is a random variable and g is a function, then Y = g(X) is also a random variable.

```
P(Y = y) = P(g(X) = y) = Σ_{x: g(x)=y} P(X = x)
```

**Example:** X ~ fair die. Y = X². Then:
- P(Y = 1) = P(X = 1) = 1/6
- P(Y = 4) = P(X = 2) = 1/6
- P(Y = 9) = P(X = 3) = 1/6
- P(Y = 16) = P(X = 4) = 1/6
- P(Y = 25) = P(X = 5) = 1/6
- P(Y = 36) = P(X = 6) = 1/6

**ML connection:** Activation functions (ReLU, sigmoid, softmax) are functions of random variables. The distribution of activations determines training dynamics — dead neurons, vanishing gradients, etc.

---

## 6. Joint PMF of Two Discrete Random Variables

> **Definition:** The **joint PMF** of X and Y is:
> ```
> p_{X,Y}(x, y) = P(X = x, Y = y)
> ```

### Marginal PMF (recovering individual distributions)

```
pₓ(x) = Σᵧ p_{X,Y}(x, y)     [sum over all y]
pᵧ(y) = Σₓ p_{X,Y}(x, y)     [sum over all x]
```

**Intuition:** To find the marginal (individual) distribution of X, sum out Y — marginalize over it.

### Conditional PMF

```
p_{X|Y}(x|y) = P(X = x | Y = y) = p_{X,Y}(x,y) / pᵧ(y)
```

### Independence via PMF

X and Y are independent if and only if:
```
p_{X,Y}(x, y) = pₓ(x) · pᵧ(y)    for all x, y
```

---

## 7. Key Discrete Distributions — Preview

| Distribution | PMF | Use in ML |
|---|---|---|
| Bernoulli(p) | P(X=1)=p, P(X=0)=1-p | Binary label, single neuron output |
| Binomial(n,p) | C(n,k)pᵏ(1-p)^(n-k) | k successes in n trials — Day 9 |
| Geometric(p) | (1-p)^(k-1)p | Trials until first success — Day 9 |
| Poisson(λ) | e^(-λ)λᵏ/k! | Count data, rare events — Day 10 |
| Uniform{1..n} | 1/n for each value | Random sampling, initialization |

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — Building a PMF from Scratch

**Problem:** Toss a fair coin 3 times. Let X = number of heads. Build the PMF and CDF of X.

**Solution:**

Sample space: {HHH, HHT, HTH, HTT, THH, THT, TTH, TTT} — 8 equally likely outcomes.

Count heads in each:

| Outcome | Heads (X) | Probability |
|---|---|---|
| TTT | 0 | 1/8 |
| HTT, THT, TTH | 1 | 3/8 |
| HHT, HTH, THH | 2 | 3/8 |
| HHH | 3 | 1/8 |

**PMF:**
```
p(0) = 1/8 = 0.125
p(1) = 3/8 = 0.375
p(2) = 3/8 = 0.375
p(3) = 1/8 = 0.125
```

Verify: 1/8 + 3/8 + 3/8 + 1/8 = 8/8 = 1 ✓

**CDF:**
```
F(x) = 0       for x < 0
F(x) = 1/8     for 0 ≤ x < 1
F(x) = 4/8     for 1 ≤ x < 2
F(x) = 7/8     for 2 ≤ x < 3
F(x) = 1       for x ≥ 3
```

**Using the CDF:**
- P(X ≤ 2) = F(2) = 7/8 = 0.875
- P(X = 2) = F(2) − F(1) = 7/8 − 4/8 = 3/8 ✓
- P(1 ≤ X ≤ 2) = F(2) − F(0) = 7/8 − 1/8 = 6/8 = 3/4
- P(X > 1) = 1 − F(1) = 1 − 4/8 = 1/2

**ML connection:** This is a Binomial(3, 0.5) distribution. In ML, this models: "3 independent predictions each with 50% accuracy — how many are correct?"

---

### 🔢 Numerical 2 — PMF of a Model's Accuracy Bucket

**Problem:** A classifier outputs one of 5 quality scores {1, 2, 3, 4, 5} with the following distribution:

| Score x | P(X = x) |
|---|---|
| 1 | 0.05 |
| 2 | 0.15 |
| 3 | 0.40 |
| 4 | 0.30 |
| 5 | ? |

**(a)** Find P(X = 5).
**(b)** Find P(X ≥ 4).
**(c)** Find P(2 < X ≤ 4).
**(d)** Build the CDF.

**Solution:**

**(a)** Probabilities must sum to 1:
P(X=5) = 1 − (0.05 + 0.15 + 0.40 + 0.30) = 1 − 0.90 = **0.10**

**(b)** P(X ≥ 4) = P(X=4) + P(X=5) = 0.30 + 0.10 = **0.40**

**(c)** P(2 < X ≤ 4) = P(X=3) + P(X=4) = 0.40 + 0.30 = **0.70**
*(Note: 2 < X means X ≥ 3 since discrete; X ≤ 4 means X = 4 is included)*

**(d) CDF:**
```
F(1) = 0.05
F(2) = 0.05 + 0.15 = 0.20
F(3) = 0.20 + 0.40 = 0.60
F(4) = 0.60 + 0.30 = 0.90
F(5) = 0.90 + 0.10 = 1.00
```

---

### 🔢 Numerical 3 — Joint PMF and Marginals

**Problem:** Two classifiers (A and B) each output 0 (wrong) or 1 (correct). Their joint distribution is:

|  | B=0 | B=1 |
|---|---|---|
| **A=0** | 0.10 | 0.15 |
| **A=1** | 0.20 | 0.55 |

**(a)** Find the marginal PMFs of A and B.
**(b)** Are A and B independent?
**(c)** Find P(A=1 | B=1).
**(d)** Find P(A=1 | B=0).

**Solution:**

**(a) Marginals:**

P(A=0) = 0.10 + 0.15 = 0.25 → P(A=1) = 0.75
P(B=0) = 0.10 + 0.20 = 0.30 → P(B=1) = 0.70

**(b) Independence check:**

P(A=1, B=1) = 0.55
P(A=1)·P(B=1) = 0.75 × 0.70 = 0.525

0.55 ≠ 0.525 → **NOT independent**

The classifiers are positively correlated — when A is correct, B is more likely to be correct too (both see easier examples).

**(c)** P(A=1 | B=1) = P(A=1, B=1) / P(B=1) = 0.55 / 0.70 = **0.786**

**(d)** P(A=1 | B=0) = P(A=1, B=0) / P(B=0) = 0.20 / 0.30 = **0.667**

**ML insight:** When B is correct (B=1), A is also more likely correct (78.6% vs 66.7%). This means the two models' errors are correlated — they fail on the same hard examples. This is why **diverse** ensembles (with low error correlation) outperform correlated ones.

---

### 🔢 Numerical 4 — CDF to Compute Probabilities

**Problem:** The number of user complaints X in a day follows this CDF:

```
F(0) = 0.30
F(1) = 0.55
F(2) = 0.75
F(3) = 0.90
F(4) = 0.97
F(5) = 1.00
```

Find:
**(a)** P(X = 3)
**(b)** P(X > 3)
**(c)** P(2 ≤ X ≤ 4)
**(d)** P(X is odd)
**(e)** The PMF

**Solution:**

**(a)** P(X = 3) = F(3) − F(2) = 0.90 − 0.75 = **0.15**

**(b)** P(X > 3) = 1 − F(3) = 1 − 0.90 = **0.10**

**(c)** P(2 ≤ X ≤ 4) = F(4) − F(1) = 0.97 − 0.55 = **0.42**
*(Include 2: F(4) − F(2−) = F(4) − F(1), since for discrete: P(2 ≤ X ≤ 4) = P(X ≤ 4) − P(X ≤ 1))*

**(d)** P(X is odd) = P(X=1) + P(X=3) + P(X=5)
= (0.55−0.30) + (0.90−0.75) + (1.00−0.97)
= 0.25 + 0.15 + 0.03 = **0.43**

**(e) PMF:**

| x | P(X=x) |
|---|---|
| 0 | 0.30 |
| 1 | 0.25 |
| 2 | 0.20 |
| 3 | 0.15 |
| 4 | 0.07 |
| 5 | 0.03 |

Verify: sums to 1 ✓ — complaints follow a decreasing distribution, most days have few.

---

### 🔢 Numerical 5 — Function of a Random Variable

**Problem:** X is a random variable for a model's raw score with PMF:

```
P(X = -2) = 0.10
P(X = -1) = 0.20
P(X =  0) = 0.30
P(X =  1) = 0.25
P(X =  2) = 0.15
```

Define Y = ReLU(X) = max(0, X). Find the PMF of Y.

**Solution:**

ReLU maps all negative values to 0:
- X = -2 → Y = 0
- X = -1 → Y = 0
- X =  0 → Y = 0
- X =  1 → Y = 1
- X =  2 → Y = 2

Group by Y value:
```
P(Y = 0) = P(X = -2) + P(X = -1) + P(X = 0)
         = 0.10 + 0.20 + 0.30 = 0.60

P(Y = 1) = P(X = 1) = 0.25

P(Y = 2) = P(X = 2) = 0.15
```

Verify: 0.60 + 0.25 + 0.15 = 1 ✓

**ML insight:** ReLU "kills" 60% of neurons in this layer (those with non-positive input). If this fraction is too high, you get **dead neurons** — a well-known training pathology. This is why weight initialization and batch normalization matter: they shape the input distribution so ReLU has a healthy activation rate.

---

### 🔢 Numerical 6 — Softmax as a PMF

**Problem:** A neural network's final layer outputs logits z = (2.0, 1.0, 0.1) for classes (cat, dog, bird).

The softmax function converts logits to probabilities:
```
P(class k) = exp(zₖ) / Σⱼ exp(zⱼ)
```

**(a)** Compute the PMF over classes.
**(b)** Verify it's a valid PMF.
**(c)** What class is predicted?
**(d)** If the true label is "dog", what is the cross-entropy loss?

**Solution:**

**(a)**
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(0.1) = 1.105

Sum = 7.389 + 2.718 + 1.105 = 11.212

P(cat)  = 7.389 / 11.212 = **0.659**
P(dog)  = 2.718 / 11.212 = **0.242**
P(bird) = 1.105 / 11.212 = **0.099**

**(b)** 0.659 + 0.242 + 0.099 = 1.000 ✓
All values ≥ 0 ✓ — **valid PMF** ✓

Softmax always produces a valid PMF by construction. This is why it's used as the final layer for multiclass classification.

**(c)** Predicted class = argmax = **cat** (highest probability 0.659)

**(d)** Cross-entropy loss = −log P(true class) = −log P(dog) = −log(0.242) = **1.418**

**ML insight:** Cross-entropy loss is simply the negative log-likelihood of the true class under the model's PMF. Minimizing cross-entropy = maximizing the probability assigned to true labels = MLE. The chain rule of probability (Day 3) + softmax PMF + log = cross-entropy loss. This is the complete foundation of classification loss functions.

---

### 🔢 Numerical 7 — Indicator Random Variables

**Problem:** You have a dataset of 100 samples. Each sample independently has a 70% chance of being correctly classified by your model.

Define indicator variables:
- Xᵢ = 1 if sample i is correctly classified, 0 otherwise
- T = X₁ + X₂ + ... + X₁₀₀ (total correct)

**(a)** What is the PMF of each Xᵢ?
**(b)** What values can T take?
**(c)** P(T = 100) — all correct?
**(d)** P(T = 0) — all wrong?

**Solution:**

**(a)** Each Xᵢ ~ Bernoulli(0.70):
P(Xᵢ = 1) = 0.70
P(Xᵢ = 0) = 0.30

**(b)** T = sum of 100 Bernoulli(0.70) → T ~ Binomial(100, 0.70)
T can take values {0, 1, 2, ..., 100}

**(c)** P(T = 100) = 0.70¹⁰⁰ ≈ **3.23 × 10⁻¹⁶** — essentially impossible

**(d)** P(T = 0) = 0.30¹⁰⁰ ≈ **5.17 × 10⁻⁵³** — even more impossible

**ML insight:** Indicator random variables are a powerful technique — they convert complex counting problems into sums of simple Bernoullis. The expected value of a sum of indicators = sum of expected values (linearity of expectation, Day 15). This is how we compute expected accuracy, expected number of errors, expected number of active neurons, etc.

---

## 9. Critical Distinctions for Interviews

### PMF vs PDF vs CDF

| Concept | Applies to | Definition | Key property |
|---|---|---|---|
| PMF | Discrete RVs only | P(X = x) | Sums to 1 |
| PDF | Continuous RVs only | Density function f(x) | Integrates to 1, P(X=x)=0 |
| CDF | Both | P(X ≤ x) | Non-decreasing, 0 to 1 |

### Common Mistakes

1. **P(X = x) for continuous RV** — Always 0. For continuous RVs, use P(a ≤ X ≤ b).
2. **CDF at a point** — F(3) = P(X ≤ 3), NOT P(X = 3).
3. **PMF values can exceed 1** — FALSE. Each p(x) ≤ 1. (PDF values can exceed 1, but PMF values cannot.)
4. **P(a < X ≤ b) for discrete** — = F(b) − F(a), where F(a) = P(X ≤ a) includes a.

---

## 10. Key Formulas — Cheat Sheet for Day 7

```
Random Variable:
    X : Ω → ℝ

PMF (discrete):
    p(x) = P(X = x)
    p(x) ≥ 0,   Σₓ p(x) = 1

CDF (both types):
    F(x) = P(X ≤ x)
    Properties: non-decreasing, right-continuous, F(−∞)=0, F(+∞)=1

PMF from CDF:
    p(x) = F(x) − F(x⁻)     [jump size at x]

CDF from PMF:
    F(x) = Σ_{t ≤ x} p(t)

Probability of interval:
    P(a < X ≤ b) = F(b) − F(a)
    P(a ≤ X ≤ b) = F(b) − F(a−)     [for discrete: F(b) − F(a-1)]
    P(X > a) = 1 − F(a)

Joint PMF:
    p_{X,Y}(x,y) = P(X=x, Y=y)

Marginal PMF:
    pₓ(x) = Σᵧ p_{X,Y}(x,y)

Conditional PMF:
    p_{X|Y}(x|y) = p_{X,Y}(x,y) / pᵧ(y)

Independence via PMF:
    p_{X,Y}(x,y) = pₓ(x)·pᵧ(y)   for all x,y

Softmax PMF:
    P(class k) = exp(zₖ) / Σⱼ exp(zⱼ)

Cross-entropy loss:
    L = −log P(true class)   [negative log of PMF at true label]
```

---

## 11. Practice Problems (Solve Before Day 8)

1. A dataset has class distribution: P(Y=0)=0.6, P(Y=1)=0.3, P(Y=2)=0.1. Build the CDF of Y. Find P(Y ≥ 1) and P(0 < Y ≤ 2).

2. Two dice are rolled. Let X = value on die 1, Y = value on die 2, Z = X + Y.
   - Build (a subset of) the joint PMF of (X,Y).
   - Find P(Z = 7) using the marginal approach.
   - Are X and Y independent? Prove using the PMF.

3. A model outputs logits (3.0, 1.0, -1.0, 0.5). Apply softmax and find the PMF over 4 classes. What is the cross-entropy loss if the true class is class 3 (index 2)?

4. X has PMF: P(X=k) = c/k² for k = 1, 2, 3, 4. Find c, then find P(X > 2).

5. *(Interview-level)* Define Y = |X − 3| where X ~ Uniform{1,2,3,4,5} (each with prob 1/5). Find the PMF of Y. What does this tell you about the distribution of absolute errors when a model predicts a constant 3?

---

## 12. Looking Ahead

**Day 8** — **Expectation, Variance & Standard Deviation.** Now that we have PMFs, we can summarize distributions with numbers. Expectation is the center of mass, variance measures spread — and both are the foundation of every loss function, metric, and learning objective in ML.

---
*End of Day 7 | Next: Day 8 — Expectation, Variance & Standard Deviation*
