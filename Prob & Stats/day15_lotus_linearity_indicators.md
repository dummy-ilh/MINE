# Day 15 — LOTUS, Linearity of Expectation & Indicator Random Variables
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 4 (Sections 4.1–4.3)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why These Three Tools are Power Multipliers

Days 7–14 taught you distributions. Today's tools let you **compute expectations of almost anything without knowing the full distribution**.

| Tool | What It Does | When to Use |
|---|---|---|
| **LOTUS** | E[g(X)] without finding PMF of g(X) | Any function of a known RV |
| **Linearity** | E[X+Y] = E[X]+E[Y] always | Sums of RVs, even dependent |
| **Indicators** | Convert counting problems to expectations | "Expected number of..." problems |

These three tools solve 80% of expectation problems in ML interviews. Master them and hard problems become easy.

---

## 2. LOTUS — Law of the Unconscious Statistician

> **Theorem (LOTUS):** If X is a discrete RV with PMF p(x), and g is any function:
> ```
> E[g(X)] = Σₓ g(x) · p(x)
> ```
> For continuous X with PDF f(x):
> ```
> E[g(X)] = ∫₋∞^∞ g(x) · f(x) dx
> ```

### Why "Unconscious Statistician"?

You act as if you don't know — or don't care — about the distribution of Y = g(X). You use X's distribution directly. The "unconscious" statistician forgets to find the distribution of g(X) first, and it turns out that's exactly right.

### The Wrong Way vs The Right Way

**Wrong (unnecessary):**
1. Find the PMF of Y = g(X)
2. Compute E[Y] = Σᵧ y · P(Y=y)

**Right (LOTUS):**
1. Use X's PMF directly: E[g(X)] = Σₓ g(x) · p(x)
2. Done.

### LOTUS Enables Everything

```
E[X²]     = Σ x²·p(x)          [second moment — needed for Var]
E[X³]     = Σ x³·p(x)          [third moment — skewness]
E[eˣ]     = Σ eˣ·p(x)          [moment generating function]
E[log X]  = Σ log(x)·p(x)      [entropy, information theory]
E[max(X,0)] = Σ max(x,0)·p(x)  [expected ReLU output]
E[|X−c|]  = Σ |x−c|·p(x)      [MAE with prediction c]
```

---

## 3. Linearity of Expectation

> **Theorem:** For any random variables X₁, X₂, ..., Xₙ and constants a₁, ..., aₙ, c:
> ```
> E[a₁X₁ + a₂X₂ + ... + aₙXₙ + c] = a₁E[X₁] + a₂E[X₂] + ... + aₙE[Xₙ] + c
> ```

**This holds for ALL random variables — dependent or independent.**

### Proof (for two variables)

```
E[X+Y] = Σₓ Σᵧ (x+y) · p_{X,Y}(x,y)
        = Σₓ Σᵧ x · p_{X,Y}(x,y) + Σₓ Σᵧ y · p_{X,Y}(x,y)
        = Σₓ x · Σᵧ p_{X,Y}(x,y) + Σᵧ y · Σₓ p_{X,Y}(x,y)
        = Σₓ x · pₓ(x) + Σᵧ y · pᵧ(y)
        = E[X] + E[Y]  ∎
```

No independence used — marginal distributions do the work.

### Why Linearity is So Powerful

Without linearity, computing E[X₁+X₂+...+Xₙ] would require the full joint distribution — exponentially complex. With linearity, it's just n individual expectations.

**Key contrast:**
```
E[X+Y] = E[X]+E[Y]              [always — no conditions]
Var(X+Y) = Var(X)+Var(Y)        [ONLY if independent]
E[XY] = E[X]·E[Y]              [ONLY if independent]
```

---

## 4. Indicator Random Variables

> **Definition:** For any event A, the **indicator random variable** Iₐ is:
> ```
> Iₐ = 1 if A occurs
> Iₐ = 0 if A does not occur
> ```
> Iₐ ~ Bernoulli(P(A))

### The Fundamental Identity

```
E[Iₐ] = P(A)
```

**Proof:** E[Iₐ] = 1·P(A) + 0·P(Aᶜ) = P(A) ∎

This seems trivial but is incredibly useful: **any probability can be written as an expected value of an indicator.**

### The Indicator Method — The Recipe

To find "expected number of [things with property P]":

1. Define Iᵢ = 1 if item i has property P, else 0
2. Total count = T = I₁ + I₂ + ... + Iₙ
3. E[T] = E[I₁] + E[I₂] + ... + E[Iₙ] = P(I₁=1) + P(I₂=1) + ... [by linearity]

**You never need to find the distribution of T — just n individual probabilities!**

### Variance of Indicator

```
Iₐ ~ Bernoulli(p) where p = P(A)
Var(Iₐ) = p(1−p)
```

---

## 5. The Connection: LOTUS + Linearity + Indicators

These three tools chain together:

```
Step 1: Break complex RV into sum of indicators (Linearity)
Step 2: Each indicator's expectation = its probability (Indicator identity)
Step 3: If you need E[g(T)], use LOTUS with T's distribution

Alternatively:
Step 1: Identify g(X) you need expectation of
Step 2: Apply LOTUS directly with X's known PMF/PDF
```

---

## 6. Worked Numericals

---

### 🔢 Numerical 1 — LOTUS: Expected Loss Functions

**Problem:** A regression model's error E has PMF:
```
P(E=−2)=0.1, P(E=−1)=0.2, P(E=0)=0.4, P(E=1)=0.2, P(E=2)=0.1
```

Using LOTUS, compute:
**(a)** E[E²] — MSE (mean squared error)
**(b)** E[|E|] — MAE (mean absolute error)
**(c)** E[max(E,0)] — expected positive error only
**(d)** E[e^E] — moment generating function at t=1
**(e)** Which loss function would you minimize for robust regression?

**Solution:**

**(a) MSE via LOTUS (g(e)=e²):**
```
E[E²] = (−2)²×0.1 + (−1)²×0.2 + 0²×0.4 + 1²×0.2 + 2²×0.1
      = 4×0.1 + 1×0.2 + 0 + 1×0.2 + 4×0.1
      = 0.4 + 0.2 + 0 + 0.2 + 0.4 = 1.2
```

**(b) MAE via LOTUS (g(e)=|e|):**
```
E[|E|] = 2×0.1 + 1×0.2 + 0×0.4 + 1×0.2 + 2×0.1
       = 0.2+0.2+0+0.2+0.2 = 0.8
```

**(c) Expected positive error (g(e)=max(e,0)):**
```
E[max(E,0)] = 0×0.1 + 0×0.2 + 0×0.4 + 1×0.2 + 2×0.1
            = 0 + 0 + 0 + 0.2 + 0.2 = 0.4
```

**(d) MGF at t=1 (g(e)=eᵉ):**
```
E[eᴱ] = e^(−2)×0.1 + e^(−1)×0.2 + e⁰×0.4 + e¹×0.2 + e²×0.1
       = 0.1353×0.1 + 0.3679×0.2 + 1×0.4 + 2.718×0.2 + 7.389×0.1
       = 0.01353 + 0.07358 + 0.4 + 0.5436 + 0.7389
       = 1.769
```

**(e)** MAE (0.8) < MSE (1.2) in magnitude, but the choice depends on the goal:
- **MSE** penalizes large errors more → sensitive to outliers → use when outliers matter
- **MAE** treats all errors equally → robust to outliers → use for robust regression
- **Huber loss** = MAE for large errors, MSE for small → best of both worlds

---

### 🔢 Numerical 2 — Indicator Method: Expected Number of Fixed Points

**Problem:** n = 5 students each submit one paper. Papers are graded and returned randomly (a derangement-like problem, but here we want expected fixed points, not zero).

**(a)** Expected number of students who receive their own paper.
**(b)** Variance of the number of fixed points.
**(c)** P(exactly 0 fixed points) — derangement probability.

**Solution:**

Define Iᵢ = 1 if student i gets their own paper.

**(a)**
```
P(Iᵢ = 1) = 1/n = 1/5     [by symmetry]

E[total fixed points] = E[I₁+I₂+...+I₅]
                      = 5 × (1/5) = 1
```

**Expected exactly 1 fixed point, regardless of n!** (This is a famous result.)

**(b)** For variance, need Cov(Iᵢ, Iⱼ) for i≠j:
```
E[IᵢIⱼ] = P(Iᵢ=1, Iⱼ=1) = P(student i gets own, student j gets own)
         = 1/5 × 1/4 = 1/20    [without replacement]

E[Iᵢ]E[Iⱼ] = 1/5 × 1/5 = 1/25

Cov(Iᵢ,Iⱼ) = 1/20 − 1/25 = 5/100 − 4/100 = 1/100
```

```
Var(T) = Σᵢ Var(Iᵢ) + 2Σᵢ<ⱼ Cov(Iᵢ,Iⱼ)
       = 5×(1/5×4/5) + 2×C(5,2)×(1/100)
       = 5×(4/25) + 2×10×(1/100)
       = 4/5 + 1/5 = 1
```

**Var(T) = 1 for n=5.** (In general, Var(T)=1 for all n — another famous result.)

**(c)** P(T=0) = 1/e ≈ 0.368 for large n. For n=5:
```
P(T=0) = 1 − 1 + 1/2! − 1/3! + 1/4! − 1/5!
        = 0 + 0.5 − 0.1667 + 0.0417 − 0.00833 = 0.3667
```

Close to 1/e = 0.3679 ✓

---

### 🔢 Numerical 3 — Linearity: Expected Accuracy of Ensemble

**Problem:** An ensemble of n=10 models. Each model independently predicts correctly with probability p=0.75 for any sample. The ensemble uses majority vote.

**(a)** Expected number of correct models per sample (linearity).
**(b)** E[accuracy of individual model] = E[Iᵢ] for any model i.
**(c)** For majority vote to be correct, need ≥ 6 of 10 models correct. P(ensemble correct)?
**(d)** How does this compare to single model accuracy?

**Solution:**

**(a)** Let Cᵢ = 1 if model i correct. T = Σᵢ Cᵢ.
```
E[T] = 10 × 0.75 = 7.5 models correct on average
```

**(b)**
```
E[Cᵢ] = P(Cᵢ=1) = 0.75    [each model 75% accurate]
```

**(c)** T ~ Binomial(10, 0.75). Need P(T ≥ 6):
```
P(T=6) = C(10,6)×0.75⁶×0.25⁴ = 210×0.17798×0.00391 = 0.1460
P(T=7) = C(10,7)×0.75⁷×0.25³ = 120×0.13348×0.01563 = 0.2503
P(T=8) = C(10,8)×0.75⁸×0.25² = 45×0.10011×0.0625  = 0.2816
P(T=9) = C(10,9)×0.75⁹×0.25¹ = 10×0.07508×0.25    = 0.1877
P(T=10)= C(10,10)×0.75¹⁰     = 1×0.05631          = 0.0563

P(T≥6) = 0.1460+0.2503+0.2816+0.1877+0.0563 = 0.9219
```

**(d)** Ensemble accuracy = **92.2%** vs individual accuracy = **75%**.

Majority vote of 10 models boosted accuracy by 17.2 percentage points!

**General formula** for majority vote with n models and individual accuracy p > 0.5:
```
P(ensemble correct) = P(Bin(n,p) > n/2) → 1 as n → ∞
```

This is the **Condorcet Jury Theorem** — if each voter is better than random (p > 0.5), majority vote converges to perfect accuracy as group size grows.

---

### 🔢 Numerical 4 — Indicator Method: Birthday Problem Expected Matches

**Problem:** In a group of k=30 people, what is the expected number of **pairs** who share a birthday?

*(Easier than asking P(at least one shared birthday) — indicators make it trivial.)*

**Solution:**

There are C(30,2) = 435 pairs.

For any pair (i,j), define Iᵢⱼ = 1 if persons i and j share a birthday.

```
P(Iᵢⱼ = 1) = 1/365    [person j's birthday matches person i's = 1/365]
```

```
E[total matches] = Σᵢ<ⱼ E[Iᵢⱼ]
                 = C(30,2) × (1/365)
                 = 435 × (1/365)
                 = 1.192
```

**Expected 1.19 shared-birthday pairs** in a group of 30.

Contrast with P(at least one match) ≈ 70.6% — which requires the complement + inclusion-exclusion. The indicator method gives expected count instantly.

**ML application:** Expected number of **hash collisions** in a hash table:
- n items, m buckets, uniform hashing
- For each pair (i,j): P(collision) = 1/m
- E[collisions] = C(n,2)/m ≈ n²/(2m)
- Set n²/(2m) < 1 → n < √(2m) to expect < 1 collision — **birthday bound**

---

### 🔢 Numerical 5 — LOTUS for Continuous RV: Expected ReLU Output

**Problem:** A pre-activation value Z ~ N(0,1) (before ReLU). Find E[ReLU(Z)] = E[max(Z,0)].

**Solution:**

Using LOTUS with g(z) = max(z,0):
```
E[max(Z,0)] = ∫₋∞^∞ max(z,0) · φ(z) dz
            = ∫₀^∞ z · φ(z) dz    [max(z,0)=0 for z<0]
            = ∫₀^∞ z · (1/√2π) e^(−z²/2) dz
```

Use substitution u = z²/2, du = z dz:
```
= (1/√2π) ∫₀^∞ e^(−u) du
= (1/√2π) × [−e^(−u)]₀^∞
= (1/√2π) × 1
= 1/√(2π) ≈ 0.3989
```

**E[ReLU(Z)] = 1/√(2π) ≈ 0.399 for Z ~ N(0,1)**

**ML insight:** If pre-activations are standard Normal:
- Mean after ReLU ≈ 0.399 (not zero — ReLU shifts the mean up)
- This is why batch normalization is placed **before** or **after** ReLU in careful implementations
- He initialization targets Var(ReLU(Z))=1 using this calculation:

```
E[ReLU(Z)²] = ∫₀^∞ z² · φ(z) dz = 1/2    [by symmetry of N(0,1)]
Var(ReLU(Z)) = E[ReLU(Z)²] − (E[ReLU(Z)])²
             = 1/2 − 1/(2π) ≈ 0.5 − 0.159 = 0.341
```

He init multiplies input variance by 2/n so that after ReLU, variance ≈ 1.

---

### 🔢 Numerical 6 — Indicator Method: Expected Number of Collisions in Mini-Batch

**Problem:** A mini-batch samples n=64 examples uniformly with replacement from N=50,000 training examples.

**(a)** Expected number of duplicate samples (same example drawn twice or more).
**(b)** P(a specific example appears at least once in the batch).
**(c)** Expected number of unique examples in the batch.

**Solution:**

**(a)** For any pair (i,j) with i≠j (positions in batch):
```
P(same example) = 1/N = 1/50000

Number of pairs = C(64,2) = 2016

E[duplicate pairs] = 2016 × (1/50000) = 0.04032
```

Very few duplicates expected — mini-batch sampling is nearly without replacement for large N. ✓

**(b)** P(specific example x appears at least once):
```
P(x appears) = 1 − P(x never appears) = 1 − (1 − 1/N)^n
             = 1 − (1 − 1/50000)^64
             ≈ 1 − e^(−64/50000)
             = 1 − e^(−0.00128)
             ≈ 0.00128 = 0.128%
```

**(c)** Define Iₓ = 1 if example x appears in batch. Expected unique examples:
```
E[unique] = Σₓ P(Iₓ=1) = N × [1 − (1−1/N)^n]
          = 50000 × 0.00128 = 64.0
```

Nearly all 64 batch samples are unique (as expected given large N relative to n).

**General formula for unique items in sample of n from N with replacement:**
```
E[unique] = N × [1 − (1 − 1/N)^n] ≈ N(1 − e^(−n/N))
```

For n << N: E[unique] ≈ n (almost no duplicates)
For n = N: E[unique] = N(1 − 1/e) ≈ 0.632N (bootstrap OOB result from Day 2!)

---

### 🔢 Numerical 7 — Combined: Linearity + Indicators for Gradient Descent

**Problem:** A model has n=1000 parameters. At each gradient step:
- Each parameter independently updates in the "correct direction" with probability p=0.6
- Wrong direction with probability 0.4

Define:
- Cᵢ = 1 if parameter i moves in correct direction after step
- T = Σᵢ Cᵢ = total parameters moving correctly

**(a)** E[T] and Var(T)
**(b)** P(T > 700) — probability more than 70% of parameters improve
**(c)** E[T/n] — expected fraction correct (accuracy of the step)
**(d)** After 100 steps, each parameter independently: P(parameter i correct in all 100 steps)?

**Solution:**

**(a)** Each Cᵢ ~ Bernoulli(0.6), independent. T ~ Binomial(1000, 0.6):
```
E[T] = 1000 × 0.6 = 600
Var(T) = 1000 × 0.6 × 0.4 = 240
SD(T) = √240 ≈ 15.49
```

**(b)** Normal approximation (CLT, n=1000 is large):
```
P(T > 700) = P(Z > (700−600)/15.49) = P(Z > 6.46) ≈ 0
```

700 is 6.46 SDs above the mean — essentially impossible. In practice, no single gradient step moves 70% of parameters correctly when the true rate is 60%.

**(c)**
```
E[T/n] = E[T]/n = 600/1000 = 0.60
```

By linearity: E[fraction correct] = p = 0.60.

**(d)** P(parameter i correct in ALL 100 steps) = 0.6^100 ≈ 6.5×10^(−23)

Essentially zero — no single parameter consistently improves for 100 steps.

**ML insight:** This reveals why **momentum** helps. If each step only moves each parameter in the right direction 60% of the time, pure SGD oscillates. Momentum accumulates the 60% signal and dampens the 40% noise, effectively increasing p toward 1 for the running direction.

---

## 7. Advanced: LOTUS for Entropy and Information

The **entropy** of a discrete distribution is:
```
H(X) = −Σₓ P(X=x) · log P(X=x) = E[−log P(X)]
```

This is exactly **LOTUS** with g(x) = −log p(x):
```
H(X) = E[g(X)]    where g(x) = −log p(x)
```

**Cross-entropy loss** = E[−log P̂(Y|X)] = LOTUS with g = −log P̂

Every information-theoretic quantity is an expectation of a log-probability — LOTUS is the bridge.

```
Entropy:          H(X) = E[−log P(X)]
Cross-entropy:    H(P,Q) = E_P[−log Q(X)] = −Σₓ P(x)log Q(x)
KL divergence:    KL(P||Q) = E_P[log P(X)/Q(X)] = H(P,Q) − H(P)
Mutual info:      I(X;Y) = E[log P(X,Y)/(P(X)P(Y))]
```

All are expectations — all computable via LOTUS.

---

## 8. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is LOTUS and why is it useful?" | E[g(X)] = Σ g(x)p(x) — no need to find distribution of g(X) |
| "Does linearity of expectation require independence?" | No — holds always. Only Var(X+Y)=Var(X)+Var(Y) requires independence |
| "What is E[Iₐ]?" | E[Iₐ] = P(A) — fundamental identity |
| "How do you compute expected number of [events]?" | Define indicators, use linearity: E[Σ Iᵢ] = Σ P(Iᵢ=1) |
| "Expected number of fixed points in random permutation?" | Always 1, regardless of n |
| "Expected number of birthday matches in k people?" | C(k,2)/365 |
| "Why is E[ReLU(Z)] = 1/√(2π) for Z~N(0,1)?" | LOTUS: ∫₀^∞ z·φ(z)dz = 1/√(2π) |
| "How does cross-entropy relate to LOTUS?" | H(P,Q) = E_P[−log Q(X)] — expectation of log-probability |

---

## 9. Key Formulas — Cheat Sheet for Day 15

```
LOTUS (discrete):
    E[g(X)] = Σₓ g(x) · p(x)

LOTUS (continuous):
    E[g(X)] = ∫ g(x) · f(x) dx

Linearity (always):
    E[a₁X₁ + ... + aₙXₙ + c] = a₁E[X₁] + ... + aₙE[Xₙ] + c

Indicator identity:
    Iₐ ~ Bernoulli(P(A))
    E[Iₐ] = P(A)

Expected count via indicators:
    E[Σᵢ Iᵢ] = Σᵢ P(Iᵢ=1)    [no independence needed]

Key LOTUS results:
    E[ReLU(Z)] = 1/√(2π) ≈ 0.399    for Z~N(0,1)
    E[Z²] = 1                          for Z~N(0,1)
    E[eᵗᶻ] = e^(t²/2)                for Z~N(0,1) [MGF]

Expected fixed points: E = 1 (for any n)
Birthday expected matches: C(k,2)/365
Hash collision expected: C(n,2)/m ≈ n²/(2m)
Unique items in sample n from N: N[1−(1−1/N)^n]

Information via LOTUS:
    H(X)   = E[−log P(X)]
    H(P,Q) = E_P[−log Q(X)]    [cross-entropy]
    KL(P||Q) = E_P[log P/Q]
```

---

## 10. Practice Problems (Solve Before Day 16)

1. X ~ Poisson(λ). Using LOTUS, find E[X(X−1)] and hence Var(X) = λ. *(Hint: E[X(X-1)] = λ², then E[X²] = E[X(X-1)] + E[X].)*

2. In a class of 30 students, each pair independently becomes friends with probability 0.1. Using the indicator method, find the expected number of friendships.

3. X ~ Uniform(0,1). Using LOTUS, find:
   - E[X²]
   - E[log X]
   - E[eˣ]

4. A neural network has 5 layers. Each layer independently drops out with probability 0.2 (the whole layer, not individual neurons). Define Iᵢ = 1 if layer i is active. Find E[active layers] and P(all layers active).

5. *(Interview-level)* A dataset has n=100 samples. A model is trained on a bootstrap sample (n=100 drawn with replacement). Using indicators, find:
   - E[number of unique training samples]
   - E[number of OOB (out-of-bag) samples]
   - E[number of times sample i appears in bootstrap]
   And verify: E[unique] + E[OOB] = 100.

---

## 11. Looking Ahead

**Day 16** — **Moment Generating Functions (MGFs).** MGFs are the Laplace transform of the distribution — they encode ALL moments and provide the slickest proofs of distribution properties. We use them to prove the sum of Normals is Normal, derive the CLT, and identify distributions from their moments.

---
*End of Day 15 | Next: Day 16 — Moment Generating Functions & Moments*
