# Day 9 — Bernoulli, Binomial & Geometric Distributions
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 3 (Sections 3.3–3.5)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why These Three Distributions Dominate ML

These are not just textbook distributions — they are the atoms of binary ML problems.

| Distribution | Models | ML Use |
|---|---|---|
| **Bernoulli(p)** | Single binary trial | One prediction, one click, one label |
| **Binomial(n,p)** | n independent binary trials | Accuracy over test set, A/B test outcomes |
| **Geometric(p)** | Trials until first success | Early stopping, training steps to convergence |

Every binary classifier, every A/B test, every click-through rate model lives in this world.

---

## 2. Bernoulli Distribution

> **Definition:** X ~ Bernoulli(p) if X takes value 1 with probability p and 0 with probability 1−p.

```
P(X = 1) = p
P(X = 0) = 1 − p = q        where q = 1 − p
```

### Compact PMF

```
P(X = x) = pˣ(1−p)^(1−x)    for x ∈ {0, 1}
```

### Parameters

```
E[X]   = p
Var(X) = p(1−p) = pq
SD(X)  = √(p(1−p))
```

**Proof of E[X]:**
```
E[X] = 0·(1−p) + 1·p = p
```

**Proof of Var(X):**
```
E[X²] = 0²·(1−p) + 1²·p = p
Var(X) = E[X²] − (E[X])² = p − p² = p(1−p)
```

### Variance is Maximized at p = 0.5

```
d/dp [p(1−p)] = 1 − 2p = 0  →  p = 0.5
Max Var = 0.5 × 0.5 = 0.25
```

**ML insight:** When p = 0.5 (maximum uncertainty), variance is highest — the model is most uncertain. This is the principle behind **entropy-based active learning**: query the point the model is most uncertain about (p closest to 0.5).

### ML Connections

- Binary classification label: Y ~ Bernoulli(p(x)) where p(x) = sigmoid(wᵀx)
- Dropout: each neuron dropped ~ Bernoulli(dropout_rate)
- Click-through: user clicks ~ Bernoulli(CTR)
- A/B test outcome per user: ~ Bernoulli(conversion_rate)

---

## 3. Binomial Distribution

> **Definition:** X ~ Binomial(n, p) if X = number of successes in n independent Bernoulli(p) trials.

```
        ⎛n⎞
P(X=k) = ⎜ ⎟ pᵏ(1−p)^(n−k)    for k = 0, 1, 2, ..., n
        ⎝k⎠
```

### Formula Breakdown — Term by Term

```
⎛n⎞
⎜ ⎟  = C(n,k)   ways to choose WHICH k of n trials succeed
⎝k⎠

pᵏ             probability of k successes

(1−p)^(n−k)   probability of (n−k) failures
```

**Verify it's a valid PMF:** Σₖ C(n,k)pᵏ(1−p)^(n−k) = (p + (1−p))ⁿ = 1ⁿ = 1 ✓ (Binomial theorem!)

### Parameters

```
E[X]   = np
Var(X) = np(1−p) = npq
SD(X)  = √(np(1−p))
```

**Proof of E[X] using indicator variables:**

Let Xᵢ = 1 if trial i succeeds. Then X = X₁ + X₂ + ... + Xₙ.

Each Xᵢ ~ Bernoulli(p), so E[Xᵢ] = p.

By linearity of expectation:
```
E[X] = E[X₁] + ... + E[Xₙ] = np
```

Since trials are independent:
```
Var(X) = Var(X₁) + ... + Var(Xₙ) = np(1−p)
```

### Key Properties

```
If X ~ Bin(n,p) and Y ~ Bin(m,p) are independent:
    X + Y ~ Bin(n+m, p)          [reproductive property]

Symmetry:
    P(X=k) for Bin(n,p) = P(X=n−k) for Bin(n,1−p)

Mode:
    Most likely value ≈ floor((n+1)p)
```

### Connection to Bernoulli

Bernoulli(p) = Binomial(1, p). Bernoulli is the special case with n=1.

---

## 4. Geometric Distribution

> **Definition:** X ~ Geometric(p) if X = number of trials until the first success (inclusive).

```
P(X = k) = (1−p)^(k−1) · p    for k = 1, 2, 3, ...
```

### Formula Breakdown

```
(1−p)^(k−1)   first (k−1) trials fail
p              k-th trial succeeds
```

**Verify:** Σₖ₌₁^∞ (1−p)^(k−1)·p = p · Σⱼ₌₀^∞ (1−p)ʲ = p · 1/(1−(1−p)) = p/p = 1 ✓ (geometric series)

### Parameters

```
E[X]   = 1/p
Var(X) = (1−p)/p²
SD(X)  = √((1−p))/p
```

**Intuition for E[X] = 1/p:**
- If p = 0.5 (fair coin for heads), expect 2 flips on average
- If p = 0.1, expect 10 trials on average
- If p = 0.01, expect 100 trials

**Proof of E[X]:**
```
E[X] = Σₖ₌₁^∞ k·(1−p)^(k−1)·p = p · Σₖ₌₁^∞ k·qᵏ⁻¹    where q = 1−p

Using d/dq[Σqᵏ] = Σkqᵏ⁻¹ and Σqᵏ = 1/(1−q):
Σₖ₌₁^∞ kqᵏ⁻¹ = 1/(1−q)² = 1/p²

E[X] = p · (1/p²) = 1/p  ∎
```

### The Memoryless Property — Critical for Interviews

> **The Geometric distribution is the ONLY discrete memoryless distribution.**

```
P(X > m + n | X > m) = P(X > n)
```

**Intuition:** If you've already failed m times, your remaining wait time has the same distribution as starting fresh. The past failures give you NO information about future trials.

**ML connections:**
- Each training step's gradient update is "fresh" — independent of previous steps (in SGD without momentum)
- In RL: if an agent has been in a "bad" state for m steps, memorylessness means the distribution of escape time is the same as starting fresh
- Why Exponential distribution (Day 11) models waiting times: it's the continuous analog of Geometric, and also memoryless

---

## 5. Negative Binomial Distribution (Extension)

> **Definition:** X ~ NegBin(r, p) = number of trials until the r-th success.

```
P(X = k) = C(k−1, r−1) · pʳ · (1−p)^(k−r)    for k = r, r+1, ...

E[X] = r/p
Var(X) = r(1−p)/p²
```

Geometric(p) = NegBin(1, p) — special case with r=1.

**ML use:** Modeling number of training steps until r consecutive improvements, number of documents to scan until r relevant ones found (information retrieval).

---

## 6. Relationships Between Distributions

```
Bernoulli(p) = Binomial(1, p)
Binomial(n,p) = sum of n independent Bernoulli(p)
Geometric(p) = NegBin(1, p)
NegBin(r,p) = sum of r independent Geometric(p)

As n→∞, p→0, np=λ fixed:
Binomial(n,p) → Poisson(λ)     [Day 10]

As n→∞:
Binomial(n,p) → Normal(np, np(1−p))   [CLT, Day 20]
```

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — Bernoulli: Single Prediction Confidence

**Problem:** A binary classifier has P(correct) = 0.85 for any single prediction.

**(a)** E[correct] and Var(correct) for one prediction.
**(b)** What value of p maximizes uncertainty (variance)?
**(c)** The model outputs P(Y=1|x) = 0.85. What is the entropy of this Bernoulli?

**Solution:**

**(a)**
```
E[X] = p = 0.85
Var(X) = p(1−p) = 0.85 × 0.15 = 0.1275
SD(X) = √0.1275 ≈ 0.357
```

**(b)** Var = p(1−p) maximized at p = 0.5, giving Var = 0.25.

**(c)** Entropy of Bernoulli(p):
```
H(p) = −p·log₂(p) − (1−p)·log₂(1−p)
     = −0.85·log₂(0.85) − 0.15·log₂(0.15)
     = −0.85·(−0.2345) − 0.15·(−2.737)
     = 0.1993 + 0.4106
     = 0.610 bits
```

Maximum entropy = H(0.5) = 1 bit. Our model has 0.61 bits — below max, meaning it has some confidence.

**ML insight:** Active learning selects samples where model entropy is highest (p closest to 0.5) — maximum uncertainty sampling. The Bernoulli variance and entropy both peak at p=0.5 for the same reason.

---

### 🔢 Numerical 2 — Binomial: Test Set Accuracy

**Problem:** A model has true accuracy p = 0.80. It's evaluated on n = 200 test samples.

**(a)** Expected number correct and variance.
**(b)** P(exactly 160 correct).
**(c)** P(at least 170 correct).
**(d)** What test set size is needed so SD of accuracy estimate < 1%?

**Solution:**

X ~ Binomial(200, 0.80)

**(a)**
```
E[X] = np = 200 × 0.80 = 160
Var(X) = np(1−p) = 200 × 0.80 × 0.20 = 32
SD(X) = √32 ≈ 5.66
```

**(b)**
```
P(X=160) = C(200,160) × 0.80¹⁶⁰ × 0.20⁴⁰
```

This is hard to compute directly. Using Normal approximation (Day 20):
X ≈ Normal(160, 32)
P(X=160) ≈ P(159.5 < X < 160.5) [continuity correction]
≈ φ(0.5/5.66) − φ(−0.5/5.66) = φ(0.088) − φ(−0.088)
≈ 2 × 0.035 = **0.070**

**(c)**
```
P(X ≥ 170) = P(Z ≥ (170 − 160)/5.66) = P(Z ≥ 1.768)
           = 1 − Φ(1.768) ≈ 1 − 0.9614 = 0.0386 ≈ 3.86%
```

**(d)** Accuracy = X/n. SD(accuracy) = SD(X)/n = √(np(1−p))/n = √(p(1−p)/n)

Set √(p(1−p)/n) < 0.01:
```
p(1−p)/n < 0.0001
n > p(1−p)/0.0001 = 0.16/0.0001 = 1600
```

Need at least **n = 1600 samples** for SD of accuracy < 1%.

**ML insight:** This is why benchmarks use large test sets. With only 100 samples, SD(accuracy) = √(0.16/100) = 4% — your reported accuracy could be off by ±8% (2 SDs) just due to sampling noise.

---

### 🔢 Numerical 3 — Binomial: A/B Test Analysis

**Problem:** You run an A/B test for a recommendation system.
- Control (A): 500 users, 85 conversions
- Treatment (B): 500 users, 110 conversions

**(a)** Estimate pA and pB.
**(b)** Under H₀: pA = pB = p̂ (pooled), compute expected conversions and SD for each group.
**(c)** Is the difference likely due to chance?

**Solution:**

**(a)**
```
p̂A = 85/500 = 0.170
p̂B = 110/500 = 0.220
```

**(b)** Pooled estimate: p̂ = (85+110)/(500+500) = 195/1000 = 0.195

Under H₀, each group ~ Binomial(500, 0.195):
```
E[conversions] = 500 × 0.195 = 97.5
Var = 500 × 0.195 × 0.805 = 78.49
SD = √78.49 ≈ 8.86
```

**(c)** Difference in observed conversions: 110 − 85 = 25

Under H₀, the difference has:
```
E[XB − XA] = 0
Var(XB − XA) = Var(XB) + Var(XA) = 78.49 + 78.49 = 156.98
SD(XB − XA) ≈ 12.53
```

Z-score: 25/12.53 ≈ **2.00**

P(|Z| > 2.0) ≈ 0.046 < 0.05 → **Statistically significant!**

Treatment B shows a meaningful improvement. (Full hypothesis testing on Day 26.)

---

### 🔢 Numerical 4 — Geometric: Training Convergence

**Problem:** At each training step, a model has a 5% probability of "breaking through" (loss drops below threshold). Steps are independent.

**(a)** Expected number of steps to breakthrough.
**(b)** P(breakthrough by step 10).
**(c)** P(breakthrough takes more than 30 steps).
**(d)** Given no breakthrough in first 10 steps, expected additional steps needed.

**Solution:**

X ~ Geometric(p = 0.05)

**(a)**
```
E[X] = 1/p = 1/0.05 = 20 steps
```

**(b)**
```
P(X ≤ 10) = 1 − P(X > 10) = 1 − (1−p)¹⁰ = 1 − 0.95¹⁰
           = 1 − 0.5987 = 0.4013 ≈ 40.1%
```

**(c)**
```
P(X > 30) = (1−p)³⁰ = 0.95³⁰ = 0.2146 ≈ 21.5%
```

**(d) Memoryless property:**
```
P(X > 10+k | X > 10) = P(X > k)
```

Given 10 steps without breakthrough, the distribution of remaining steps is still Geometric(0.05).

Expected additional steps = E[X] = **20 steps** — same as starting from scratch!

**ML insight:** This is why early stopping based on "no improvement for k steps" (patience) is not quite right if improvements are equally likely at any step. In practice, learning rate schedules and adaptive methods break the memorylessness — but the Geometric model is the baseline for thinking about convergence.

---

### 🔢 Numerical 5 — Binomial: Dropout as Bernoulli

**Problem:** A neural network layer has 1000 neurons. Dropout rate = 0.3 (each neuron independently dropped with probability 0.3).

**(a)** Expected number of active neurons.
**(b)** SD of number of active neurons.
**(c)** P(fewer than 650 neurons active) — is this unusual?
**(d)** At test time, dropout is turned off but weights are scaled by (1−0.3)=0.7. Why?

**Solution:**

X = number of active neurons ~ Binomial(1000, 0.7)

**(a)**
```
E[X] = 1000 × 0.7 = 700
```

**(b)**
```
Var(X) = 1000 × 0.7 × 0.3 = 210
SD(X) = √210 ≈ 14.49
```

**(c)**
```
P(X < 650) = P(Z < (650−700)/14.49) = P(Z < −3.45)
           ≈ 0.0003 = 0.03%
```

Extremely unusual — almost 3.5 SDs below mean. In practice, dropout rarely causes such extreme activation loss.

**(d)** During training with dropout, each neuron is active with probability 0.7, so the expected output is scaled by 0.7. At test time (no dropout), the output is full — 1.43× larger. To compensate, weights are multiplied by 0.7 to match the training-time expected output.

**Alternative: Inverted dropout** (more common): During training, divide active neuron outputs by 0.7 — so test time needs no adjustment.

**ML insight:** Dropout is literally a Binomial experiment at each forward pass. Understanding its statistics tells you: the effective ensemble size is 2^1000 sub-networks (each neuron either included or not), and the expected sub-network size is 700 neurons. This is why dropout works as a regularizer — it trains an exponentially large ensemble at linear cost.

---

### 🔢 Numerical 6 — Geometric: Click-Through Rate

**Problem:** A search engine shows results until a user clicks. Each result has a 15% probability of being clicked (independently).

**(a)** Expected position of the first click.
**(b)** P(user clicks on the first result).
**(c)** P(user clicks on result 3 or later).
**(d)** If the CTR doubles to 30%, how much does expected position change?

**Solution:**

X ~ Geometric(p = 0.15)

**(a)**
```
E[X] = 1/0.15 ≈ 6.67
```
On average, users click the 7th result.

**(b)**
```
P(X=1) = p = 0.15
```
15% of users click the very first result.

**(c)**
```
P(X ≥ 3) = (1−p)² = 0.85² = 0.7225 ≈ 72.3%
```

**(d)** With p = 0.30:
```
E[X] = 1/0.30 ≈ 3.33
```
Doubling CTR halves the expected click position from ~6.67 to ~3.33.

**ML insight:** This is the foundation of **Learning to Rank** models. The position of the first click is Geometric(CTR). Improving CTR (better ranking model) reduces E[X] — users find what they want faster. The reciprocal relationship E[X] = 1/p is why small CTR improvements have large impacts on user experience.

---

### 🔢 Numerical 7 — Combining Distributions: Full ML Pipeline

**Problem:** A document classification pipeline:
1. **Retrieval:** Each query retrieves a relevant document with P = 0.6 (Bernoulli)
2. **Classification:** If retrieved, model classifies correctly with P = 0.85 (Bernoulli)
3. **10 queries are run** (Binomial)

**(a)** P(a single query results in correct classification).
**(b)** Expected number of correctly classified results in 10 queries.
**(c)** Var of number correctly classified.
**(d)** P(at least 4 correctly classified).

**Solution:**

**(a)** P(correct) = P(retrieved) × P(correct | retrieved)
```
= 0.6 × 0.85 = 0.51
```

**(b)** T ~ Binomial(10, 0.51)
```
E[T] = 10 × 0.51 = 5.1
```

**(c)**
```
Var(T) = 10 × 0.51 × 0.49 = 2.499
SD(T) ≈ 1.581
```

**(d)** P(T ≥ 4) = 1 − P(T ≤ 3)

Using Binomial PMF:
```
P(T=0) = C(10,0)×0.51⁰×0.49¹⁰ = 0.49¹⁰ ≈ 0.00071
P(T=1) = C(10,1)×0.51¹×0.49⁹ = 10×0.51×0.001449 ≈ 0.00739
P(T=2) = C(10,2)×0.51²×0.49⁸ = 45×0.2601×0.002957 ≈ 0.03460
P(T=3) = C(10,3)×0.51³×0.49⁷ = 120×0.1327×0.006035 ≈ 0.09613

P(T ≤ 3) ≈ 0.00071 + 0.00739 + 0.03460 + 0.09613 = 0.13883

P(T ≥ 4) = 1 − 0.13883 ≈ 0.861
```

About **86.1%** probability of at least 4 correct classifications in 10 queries.

---

## 8. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the variance of Bernoulli(p)?" | p(1−p), maximized at p=0.5 |
| "Derive E[X] for Binomial using indicator variables" | E[X] = Σ E[Xᵢ] = np by linearity |
| "What is the memoryless property?" | P(X>m+n \| X>m) = P(X>n) — only Geometric (discrete) has it |
| "How does dropout relate to Bernoulli?" | Each neuron ~ Bernoulli(1−dropout_rate) per forward pass |
| "Why scale weights by (1−p) at test time?" | Match expected activation magnitude between train and test |
| "When does Binomial approximate Poisson?" | n large, p small, np = λ fixed (Day 10) |
| "What is the mode of Binomial(n,p)?" | floor((n+1)p) — most likely number of successes |
| "How many test samples to estimate accuracy within ε?" | n > p(1−p)/ε² |

---

## 9. Key Formulas — Cheat Sheet for Day 9

```
Bernoulli(p):
    P(X=1) = p,   P(X=0) = 1−p
    E[X] = p
    Var(X) = p(1−p)     [max at p=0.5: Var=0.25]

Binomial(n,p):
    P(X=k) = C(n,k) pᵏ (1−p)^(n−k)
    E[X] = np
    Var(X) = np(1−p)
    Additive: Bin(n,p) + Bin(m,p) = Bin(n+m,p)   [independent]

Geometric(p):  [number of trials UNTIL first success, inclusive]
    P(X=k) = (1−p)^(k−1) · p
    E[X] = 1/p
    Var(X) = (1−p)/p²
    Memoryless: P(X>m+n | X>m) = P(X>n)

Negative Binomial(r,p):
    P(X=k) = C(k−1,r−1) pʳ (1−p)^(k−r)
    E[X] = r/p
    Var(X) = r(1−p)/p²

Entropy of Bernoulli:
    H(p) = −p log p − (1−p) log(1−p)     [max at p=0.5]

Test set sizing:
    n > p(1−p)/ε²   for SD(accuracy) < ε

Binomial → Poisson: n→∞, p→0, np=λ
Binomial → Normal:  n large, CLT applies
```

---

## 10. Practice Problems (Solve Before Day 10)

1. A logistic regression model outputs P(Y=1|x) = 0.72. Compute E[Y], Var(Y), and the entropy H(Y). What does H(Y) tell you about model confidence?

2. In a dataset of 500 samples, each is independently mislabeled with probability 0.05. Find E[mislabeled], Var(mislabeled), and P(more than 35 mislabeled).

3. A hyperparameter search tries configs one at a time. Each config has a 10% chance of beating the baseline. Find:
   - E[configs tried until first success]
   - P(success within first 5 configs)
   - P(needing more than 20 configs)

4. **Prove** the memoryless property of the Geometric distribution:
   P(X > m+n | X > m) = P(X > n)

5. *(Interview-level)* A neural network has L=20 layers. Each layer independently "kills" the gradient with probability 0.1 (gradient becomes near-zero). Let X = number of layers that pass gradient successfully.
   - Model X as Binomial. Find E[X] and Var(X).
   - What is P(gradient survives all 20 layers)?
   - This motivates what architectural innovation? (Hint: think about skip connections.)

---

## 11. Looking Ahead

**Day 10** — **Poisson Distribution & Poisson Process.** The natural model for count data and rare events — number of server requests per second, number of errors in a log file, number of rare words in a document. We derive it as the limit of Binomial and connect it to exponential inter-arrival times.

---
*End of Day 9 | Next: Day 10 — Poisson Distribution & Poisson Process*
