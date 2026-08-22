# Day 19 — Law of Large Numbers (LLN)
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 10 (Section 10.3)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why the LLN is the Foundation of ML

The Law of Large Numbers is the mathematical reason that **learning from data works at all**.

| ML Concept | LLN Guarantee |
|---|---|
| Training loss → true risk | Empirical loss converges to expected loss |
| Sample accuracy → true accuracy | Test accuracy converges to true accuracy |
| Sample mean → population mean | x̄ → μ as n → ∞ |
| Histogram → true distribution | Empirical distribution → true distribution |
| Monte Carlo estimation | Average of samples → expected value |
| SGD convergence | Average gradient → true gradient |
| Bootstrapping validity | Bootstrap distribution → sampling distribution |
| A/B test validity | Observed difference → true difference |

Without LLN, there would be no guarantee that anything learned from finite data means anything about the underlying population. LLN is the bedrock.

---

## 2. Setup and Notation

Let X₁, X₂, X₃, ... be an i.i.d. sequence with:
```
E[Xᵢ] = μ        (finite mean)
Var(Xᵢ) = σ²     (finite variance — needed for Weak LLN via Chebyshev)
```

Define the **sample mean**:
```
X̄ₙ = (X₁ + X₂ + ... + Xₙ) / n = (1/n) Σᵢ₌₁ⁿ Xᵢ
```

**Key properties of X̄ₙ:**
```
E[X̄ₙ] = μ                    [unbiased — always]
Var(X̄ₙ) = σ²/n               [variance shrinks as 1/n]
SD(X̄ₙ) = σ/√n               [standard error]
```

As n → ∞: Var(X̄ₙ) → 0. The sample mean concentrates around μ.

---

## 3. Weak Law of Large Numbers (WLLN)

> **Theorem (WLLN):** For any ε > 0:
> ```
> P(|X̄ₙ − μ| > ε) → 0    as n → ∞
> ```

This is called **convergence in probability**: X̄ₙ →ᵖ μ.

### Proof via Chebyshev

```
P(|X̄ₙ − μ| > ε) ≤ Var(X̄ₙ)/ε²    [Chebyshev]
                 = σ²/(nε²)         [since Var(X̄ₙ) = σ²/n]
                 → 0  as n → ∞      ∎
```

This is the cleanest proof in probability theory — 3 lines, uses only Chebyshev.

### What WLLN Says (and Doesn't Say)

**Says:** For any fixed ε > 0, the probability that X̄ₙ deviates from μ by more than ε goes to zero.

**Doesn't say:** X̄ₙ converges to μ for every specific sequence of outcomes. It's a probabilistic statement — rare sequences where X̄ₙ doesn't converge are allowed, as long as their probability goes to zero.

---

## 4. Strong Law of Large Numbers (SLLN)

> **Theorem (SLLN):** 
> ```
> P(X̄ₙ → μ  as n → ∞) = 1
> ```

This is called **almost sure convergence**: X̄ₙ →ᵃ·ˢ· μ.

### Difference: Weak vs Strong

| | WLLN | SLLN |
|---|---|---|
| **Statement** | P(\|X̄ₙ−μ\|>ε)→0 for each fixed ε | P(X̄ₙ→μ)=1 |
| **Convergence type** | In probability | Almost surely |
| **Exceptions allowed** | P(X̄ₙ fails to converge) → 0 | P(X̄ₙ fails to converge) = 0 (exactly) |
| **Strength** | Weaker | Stronger (SLLN ⟹ WLLN) |
| **Requirements** | Finite mean + variance | Finite mean only (E[\|X\|]<∞) |
| **Proof difficulty** | Easy (3 lines via Chebyshev) | Hard (requires advanced analysis) |

**For ML purposes:** The WLLN is sufficient. The SLLN is the "complete" version.

---

## 5. Convergence Types — A Full Picture

There are four types of convergence in probability theory, from weakest to strongest:

```
Almost Sure (a.s.)
      ↓ implies
In Probability
      ↓ implies
In Distribution (→ CLT, Day 20)
      ↑
In Lᵖ (mean p-th power)
```

```
a.s. convergence:     P(Xₙ → X) = 1
In probability:       P(|Xₙ−X| > ε) → 0  for all ε
In distribution:      Fₙ(x) → F(x)  for all continuity points x
In L²:               E[(Xₙ−X)²] → 0
```

**For interviews:** Know a.s. and in probability. In distribution is needed for CLT (Day 20).

---

## 6. The LLN and Empirical Risk Minimization

The entire framework of supervised ML rests on LLN.

**True risk** (what we want to minimize):
```
R(f) = E_{(X,Y)~P}[L(f(X), Y)]
```

**Empirical risk** (what we actually minimize):
```
R̂ₙ(f) = (1/n) Σᵢ L(f(Xᵢ), Yᵢ)
```

By the LLN:
```
R̂ₙ(f) →ᵖ R(f)    as n → ∞
```

**Empirical Risk Minimization (ERM)** is justified by LLN: minimize empirical risk → approach true risk.

The gap |R̂ₙ(f) − R(f)| is the **generalization gap**. The LLN says this gap → 0 for fixed f. Uniform convergence (over all f in a function class) requires additional tools (VC theory, Rademacher complexity).

---

## 7. Monte Carlo Estimation — LLN in Action

**Problem:** Compute E[g(X)] for some hard-to-integrate g.

**Monte Carlo method:**
1. Sample X₁, ..., Xₙ i.i.d. from P(X)
2. Estimate: Ê[g(X)] = (1/n) Σᵢ g(Xᵢ)

By LLN: Ê[g(X)] →ᵖ E[g(X)] as n → ∞.

**Error rate:** By CLT (Day 20):
```
SD(Ê[g(X)]) = SD(g(X)) / √n
```

Monte Carlo error decreases as 1/√n — **regardless of dimension**. This is why Monte Carlo is used in high-dimensional integration (unlike grid-based methods where cost is exponential in dimension).

---

## 8. When LLN Fails

The LLN requires **finite mean** E[|X|] < ∞. It breaks for:

| Distribution | Problem |
|---|---|
| Cauchy distribution | No finite mean — X̄ₙ does NOT converge |
| Heavy-tailed (power law, α≤1) | Infinite mean — LLN fails |
| Non-i.i.d. data | LLN requires independence |
| Non-stationary data | Distribution shift → "μ" changes |

**Cauchy example:**
X ~ Cauchy(0,1): f(x) = 1/(π(1+x²)). This has no finite mean.

The sample mean of n Cauchy random variables is itself Cauchy(0,1) — it does NOT converge. The distribution of X̄ₙ is identical to the distribution of X₁ regardless of n. LLN completely fails.

**ML consequence:** If your loss function has heavy tails (e.g., squared loss with outliers), the LLN may converge very slowly or not at all in practice. This is why robust loss functions (Huber, MAE) matter.

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Convergence Rate of Sample Mean

**Problem:** X ~ Bernoulli(0.7). You compute X̄ₙ on n samples.

**(a)** E[X̄ₙ] and Var(X̄ₙ) for n=10, 100, 1000, 10000.
**(b)** P(|X̄ₙ − 0.7| > 0.05) bound via Chebyshev for each n.
**(c)** How large must n be so P(|X̄ₙ − 0.7| > 0.01) < 0.01?
**(d)** Plot the convergence conceptually.

**Solution:**

Var(X) = p(1−p) = 0.7×0.3 = 0.21

**(a)**

| n | E[X̄ₙ] | Var(X̄ₙ) = 0.21/n | SD(X̄ₙ) |
|---|---|---|---|
| 10 | 0.7 | 0.021 | 0.145 |
| 100 | 0.7 | 0.0021 | 0.0458 |
| 1,000 | 0.7 | 0.00021 | 0.0145 |
| 10,000 | 0.7 | 0.000021 | 0.00458 |

Mean stays at 0.7 (unbiased). Variance shrinks as 1/n. ✓

**(b)** P(|X̄ₙ − 0.7| > 0.05) ≤ Var(X̄ₙ)/0.05² = (0.21/n)/0.0025 = 84/n:

| n | Chebyshev bound |
|---|---|
| 10 | 84/10 = 8.4 → **1.0** (bound > 1, useless) |
| 100 | 84/100 = **0.84** |
| 1,000 | 84/1000 = **0.084** |
| 10,000 | 84/10000 = **0.0084** |

LLN: bound → 0 as n → ∞. ✓

**(c)** Set 0.21/(n×0.01²) < 0.01:
```
0.21/(0.0001n) < 0.01
2100/n < 0.01
n > 210,000
```

Need **n > 210,000** for Chebyshev guarantee. (Hoeffding would need far fewer — Day 18.)

**(d)** Conceptual convergence:

```
n=10:    X̄ₙ wanders widely: [0.4 to 1.0] common
n=100:   X̄ₙ mostly in [0.6, 0.8]
n=1000:  X̄ₙ almost always in [0.67, 0.73]
n=10000: X̄ₙ essentially equal to 0.7 ± 0.005
         ↓
n→∞:     X̄ₙ = 0.7 exactly (almost surely)
```

**ML insight:** This is why your test set accuracy is only reliable with large n. With n=100 test samples, your accuracy estimate has SD≈4.6% — you might report 74% accuracy when the truth is 70%.

---

### 🔢 Numerical 2 — LLN for Loss Functions: Training Convergence

**Problem:** Each training step computes loss on one sample. True expected loss μ=0.3, Var(loss)=0.25.

**(a)** After n steps, how close is average loss to true expected loss (Chebyshev bound)?
**(b)** For SGD to have average loss within 0.01 of true loss with 95% confidence, how many steps needed?
**(c)** Why does SGD work even without full convergence of X̄ₙ?

**Solution:**

**(a)** After n steps, X̄ₙ = average loss over n samples.
Chebyshev: P(|X̄ₙ − 0.3| > ε) ≤ 0.25/(nε²)

For ε=0.05: P(|X̄ₙ − 0.3| > 0.05) ≤ 0.25/(n×0.0025) = 100/n

After 100 steps: bound = 1.0 (useless)
After 1000 steps: bound = 0.1 (10% chance of being off by 0.05)
After 10000 steps: bound = 0.01 (1% chance)

**(b)** Set 0.25/(nε²) ≤ 0.05:
```
n ≥ 0.25/(0.05×0.0001) = 0.25/0.000005 = 50,000
```

Need 50,000 steps to guarantee average loss within 0.01 of true loss.

**(c)** SGD doesn't need X̄ₙ → μ globally. It needs:
- Individual gradients to be **unbiased estimates** of true gradient (LLN gives this)
- Gradient noise to **average out** over steps (LLN gives this)
- Model parameters to drift toward the optimum (this follows from LLN + gradient descent theory)

SGD works because LLN guarantees that noisy gradient estimates have the right expectation, so on average, parameters move in the correct direction.

---

### 🔢 Numerical 3 — Monte Carlo Integration

**Problem:** Estimate π using Monte Carlo (Day 6 setup, now with LLN justification).

Formally: E[1_{x²+y²≤1}] = P(point in unit circle) = π/4

where (x,y) ~ Uniform(0,1)².

**(a)** Show this is a LLN application.
**(b)** Expected error after n=10,000 samples.
**(c)** How many samples for error < 0.001 with probability ≥ 95%?

**Solution:**

Define Xᵢ = 1 if point i falls in unit circle, else 0. Then Xᵢ ~ Bernoulli(π/4).

X̄ₙ = (1/n)Σ Xᵢ →ᵖ π/4 by LLN.

So π̂ = 4X̄ₙ →ᵖ π. ✓

**(a)** This is LLN: the sample mean of i.i.d. indicators converges to the expected value.

**(b)** Var(Xᵢ) = (π/4)(1−π/4) ≈ 0.7854×0.2146 ≈ 0.1685

Var(X̄ₙ) = 0.1685/10000 = 0.00001685
SD(X̄ₙ) = 0.004105
SD(π̂) = 4×SD(X̄ₙ) = **0.01642**

Expected error (1 SD) after 10,000 samples ≈ 0.016.

**(c)** For error in π̂ < 0.001 with 95% confidence:

SD(π̂) = 4√(0.1685/n) < 0.001/1.96 (using Normal approximation)

```
4√(0.1685/n) < 0.000510
√(0.1685/n) < 0.0001276
0.1685/n < 1.629×10⁻⁸
n > 0.1685/1.629×10⁻⁸ = 10,344,600
```

Need ~**10 million samples** for 3-decimal accuracy in π. Monte Carlo is accurate but slow — rate is 1/√n.

**ML insight:** Monte Carlo in ML (dropout inference, MCMC, variational inference) has the same 1/√n convergence. This is why:
- Dropout uses many forward passes for uncertainty estimation
- MCMC requires long chains for accurate posterior estimation
- Importance sampling and variance reduction techniques exist to speed up convergence

---

### 🔢 Numerical 4 — LLN Failure: Cauchy Distribution

**Problem:** X₁, X₂, ..., Xₙ ~ Cauchy(0,1). Simulate what happens to X̄ₙ.

**(a)** Why does the LLN fail for Cauchy?
**(b)** What is the distribution of X̄ₙ?
**(c)** What is the ML consequence?

**Solution:**

**(a)** Cauchy(0,1) has PDF f(x) = 1/(π(1+x²)).

The mean E[X] = ∫₋∞^∞ x/(π(1+x²)) dx = 0... wait, doesn't this equal 0?

Technically, the integral ∫ x/(1+x²) dx is improper and doesn't converge absolutely:
```
∫₀^∞ x/(1+x²) dx = [log(1+x²)/2]₀^∞ = ∞
∫₋∞^0 x/(1+x²) dx = −∞
```

E[|X|] = ∫|x|/(π(1+x²))dx = ∞ — **infinite first absolute moment**. LLN requires E[|X|] < ∞.

**(b)** The MGF of Cauchy doesn't exist. The characteristic function is:
```
φ_X(t) = e^(−|t|)
```

For X̄ₙ:
```
φ_{X̄ₙ}(t) = [φ_X(t/n)]ⁿ = [e^(−|t|/n)]ⁿ = e^(−|t|)
```

This equals φ_X(t) — **X̄ₙ has the same distribution as X₁!**

No matter how many Cauchy samples you average, the distribution doesn't concentrate. X̄ₙ ~ Cauchy(0,1) for all n.

**(c)** **ML consequence:**
- Squared loss with heavy-tailed errors → gradients can be Cauchy-like
- Averaging such gradients (large batch) doesn't reduce gradient variance
- One extreme outlier can dominate the batch gradient
- **Fix:** Gradient clipping (caps individual gradients), Huber loss (quadratic near 0, linear in tails), robust loss functions

This is the mathematical reason gradient clipping and robust losses exist in modern deep learning.

---

### 🔢 Numerical 5 — LLN and Empirical Risk Minimization

**Problem:** Binary classifier f. True error rate R(f) = P(f(X) ≠ Y) = 0.12.

You evaluate on n test samples. Observed error rate R̂ₙ(f).

**(a)** E[R̂ₙ(f)] and Var(R̂ₙ(f)).
**(b)** P(|R̂ₙ − 0.12| > 0.02) for n=100, 500, 1000.
**(c)** What n guarantees |R̂ₙ − R(f)| < 0.01 with probability ≥ 99%?
**(d)** If you compare 5 models, how does this change?

**Solution:**

Xᵢ = 1 if sample i is misclassified. Xᵢ ~ Bernoulli(0.12), R̂ₙ = X̄ₙ.

**(a)**
```
E[R̂ₙ] = 0.12            [unbiased by LLN]
Var(R̂ₙ) = 0.12×0.88/n = 0.1056/n
```

**(b)** Using Hoeffding (tighter):
P(|R̂ₙ−0.12|>0.02) ≤ 2exp(−2n×0.02²) = 2exp(−0.0008n)

| n | Hoeffding bound |
|---|---|
| 100 | 2exp(−0.08) = 2×0.923 = **1.85** → 1.0 (useless) |
| 500 | 2exp(−0.4) = 2×0.670 = **1.34** → 1.0 |
| 1,000 | 2exp(−0.8) = 2×0.449 = **0.899** |
| 5,000 | 2exp(−4.0) = 2×0.018 = **0.037** |

Need ~5000 samples for a useful bound at ε=0.02.

**(c)** Hoeffding with ε=0.01, δ=0.01:
```
2exp(−2n×0.0001) ≤ 0.01
exp(−0.0002n) ≤ 0.005
−0.0002n ≤ ln(0.005) = −5.298
n ≥ 26,491
```

Need ~**26,500 test samples** for 99% confidence within 0.01 error.

**(d)** For 5 models, use union bound:
```
P(any model's R̂ₙ off by > ε) ≤ 5 × 2exp(−2nε²) = 10exp(−2nε²)
```

Set equal to 0.01: exp(−2nε²) = 0.001 → n ≥ ln(1000)/(2×0.0001) = 34,539

Need ~**34,500 samples** when comparing 5 models (vs 26,500 for one).

**ML insight:** Comparing many models on the same test set requires larger test sets. With 5 models, the best one might look best just by luck (multiple comparison problem from Day 11). LLN + union bound quantifies this precisely.

---

### 🔢 Numerical 6 — LLN for Gradient Estimation in SGD

**Problem:** True gradient: ∇L = −0.5 (should decrease parameter by 0.5).

Each stochastic gradient: G ~ N(−0.5, 1.0) (noise variance = 1).

**(a)** E[Ḡₙ] and Var(Ḡₙ) for batch sizes n=1, 8, 32, 128.
**(b)** P(Ḡₙ > 0 — wrong direction) for each batch size.
**(c)** Why does large batch SGD converge faster per step but potentially worse overall?

**Solution:**

**(a)**

| n | E[Ḡₙ] | Var(Ḡₙ) = 1/n | SD(Ḡₙ) |
|---|---|---|---|
| 1 | −0.5 | 1.00 | 1.000 |
| 8 | −0.5 | 0.125 | 0.354 |
| 32 | −0.5 | 0.031 | 0.177 |
| 128 | −0.5 | 0.0078 | 0.088 |

LLN: all means equal the true gradient (unbiased). Variance drops as 1/n. ✓

**(b)** Wrong direction: P(Ḡₙ > 0) = P(Z > 0.5/SD(Ḡₙ)):

| n | Threshold = 0.5/SD | P(wrong direction) |
|---|---|---|
| 1 | 0.5/1.0 = 0.5 | P(Z>0.5) = 30.9% |
| 8 | 0.5/0.354 = 1.41 | P(Z>1.41) = 7.9% |
| 32 | 0.5/0.177 = 2.83 | P(Z>2.83) = 0.23% |
| 128 | 0.5/0.088 = 5.68 | P(Z>5.68) ≈ 0% |

**(c)** Large batches:
- ✅ More accurate gradient estimates (fewer wrong-direction steps)
- ✅ Faster convergence per step
- ❌ More compute per step
- ❌ Often converge to **sharper minima** that generalize worse (the "generalization gap" of large-batch training)
- ❌ Less implicit regularization from noise (gradient noise acts as regularizer)

**The noise in SGD (which LLN reduces as batch grows) is NOT purely bad** — it helps escape sharp minima and find flat, generalizing solutions. This is why practitioners often prefer batch sizes of 32-256 over very large batches (1024+) even when compute allows.

---

### 🔢 Numerical 7 — Strong vs Weak LLN: Almost Sure Convergence

**Problem:** X₁, X₂, ... ~ Bernoulli(0.5). Define the event:
```
Aₙ = {|X̄ₙ − 0.5| > 0.1}    (sample mean is far from 0.5)
```

**(a)** P(Aₙ) for each n (using Normal approximation).
**(b)** Does Σₙ P(Aₙ) converge? (Borel-Cantelli lemma connection.)
**(c)** What does the Strong LLN say about the sequence {Aₙ}?
**(d)** Intuition: what does "almost surely" mean in practice?

**Solution:**

**(a)** X̄ₙ ~ approximately N(0.5, 0.25/n).

P(Aₙ) = P(|X̄ₙ − 0.5| > 0.1) = P(|Z| > 0.1/√(0.25/n)) = P(|Z| > 0.2√n)

| n | 0.2√n | P(Aₙ) = P(\|Z\|>0.2√n) |
|---|---|---|
| 1 | 0.2 | 2×0.421 = 0.842 |
| 25 | 1.0 | 2×0.159 = 0.317 |
| 100 | 2.0 | 2×0.023 = 0.046 |
| 400 | 4.0 | 2×3.2×10⁻⁵ ≈ 6.4×10⁻⁵ |
| 10000 | 20 | ≈ 0 |

P(Aₙ) → 0 — this is WLLN. ✓

**(b)** Σₙ P(Aₙ): The tail sum Σ P(|Z|>0.2√n) converges (probabilities decrease exponentially fast). By the **Borel-Cantelli lemma**: if Σₙ P(Aₙ) < ∞, then P(Aₙ occurs infinitely often) = 0.

**(c)** SLLN says: P(X̄ₙ → 0.5) = 1. This means P(Aₙ occurs infinitely often) = 0 — with probability 1, X̄ₙ eventually stays within 0.1 of 0.5 forever.

**(d)** "Almost surely" means: in a thought experiment where you run infinitely many sequences X₁, X₂, ..., the set of sequences where X̄ₙ fails to converge to 0.5 has probability measure zero — it's a set of outcomes that is "impossible" in a measure-theoretic sense, even though individually bizarre sequences (like all heads) are technically possible.

**Practical meaning:** You will never observe X̄ₙ failing to converge in practice — the "exceptional" sequences are so rare they form a set of probability zero.

---

## 10. LLN vs CLT — Preview

| | LLN (Day 19) | CLT (Day 20) |
|---|---|---|
| **Statement** | X̄ₙ → μ | √n(X̄ₙ−μ)/σ → N(0,1) |
| **What it gives** | Convergence to a point | Rate and distribution of convergence |
| **Scale** | X̄ₙ itself | √n × (X̄ₙ−μ) |
| **Limit** | A number (μ) | A distribution (Normal) |
| **Use** | Guarantees consistency | Confidence intervals, hypothesis tests |

**The LLN says WHERE the sample mean goes. The CLT says HOW FAST and in what shape.**

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "State the Law of Large Numbers" | X̄ₙ →ᵖ μ (WLLN) or X̄ₙ →ᵃ·ˢ· μ (SLLN) |
| "Prove the WLLN" | Apply Chebyshev: P(\|X̄ₙ−μ\|>ε) ≤ σ²/(nε²) → 0 |
| "Difference between Weak and Strong LLN?" | In probability vs almost sure; SLLN stronger; needs only finite mean |
| "Why does the LLN justify empirical risk minimization?" | R̂ₙ(f) →ᵖ R(f) for fixed f |
| "When does the LLN fail?" | Infinite mean (Cauchy), non-i.i.d., distribution shift |
| "What is the distribution of X̄ₙ for Cauchy?" | Cauchy(0,1) — same as individual samples, doesn't concentrate |
| "How does batch size relate to the LLN?" | Larger batch → variance of gradient estimate decreases as 1/n |
| "What is the Monte Carlo convergence rate?" | Error ∝ 1/√n — from LLN + CLT |

---

## 12. Key Formulas — Cheat Sheet for Day 19

```
Sample Mean:
    X̄ₙ = (1/n) Σᵢ Xᵢ
    E[X̄ₙ] = μ                [unbiased]
    Var(X̄ₙ) = σ²/n           [shrinks as 1/n]
    SD(X̄ₙ) = σ/√n            [standard error]

WLLN:
    P(|X̄ₙ − μ| > ε) ≤ σ²/(nε²) → 0  as n→∞

SLLN:
    P(X̄ₙ → μ) = 1

Proof of WLLN (Chebyshev):
    P(|X̄ₙ−μ|>ε) ≤ Var(X̄ₙ)/ε² = σ²/(nε²) → 0 ∎

Empirical Risk:
    R̂ₙ(f) →ᵖ R(f)   [LLN justifies ERM]

Monte Carlo:
    (1/n)Σg(Xᵢ) →ᵖ E[g(X)]     [LLN]
    Error ~ σ_{g(X)}/√n          [CLT, Day 20]

LLN Requires:
    i.i.d. samples
    E[|X|] < ∞   [SLLN]
    E[X²] < ∞    [WLLN via Chebyshev]

LLN Fails For:
    Cauchy (infinite mean) → X̄ₙ ~ Cauchy for all n
    Heavy tails (E[|X|]=∞)
    Non-i.i.d. data

Convergence Types (strong to weak):
    Almost sure → In Lᵖ → In probability → In distribution
```

---

## 13. Practice Problems (Solve Before Day 20)

1. X ~ Exponential(λ=2). Using the WLLN, what does X̄ₙ converge to? What is Var(X̄ₙ)? How large must n be so P(|X̄ₙ − 0.5| > 0.01) < 0.05 by Chebyshev?

2. A model's accuracy on each sample is i.i.d. Bernoulli(p). You observe X̄₅₀₀ = 0.92.
   - What does LLN say about this estimate?
   - Give a Chebyshev bound for how far 0.92 could be from true p.
   - Give a Hoeffding bound.

3. **Prove** that E[X̄ₙ] = μ and Var(X̄ₙ) = σ²/n for i.i.d. X₁,...,Xₙ using linearity of expectation and independence.

4. You run Monte Carlo to estimate ∫₀¹ x² dx = 1/3. Using n samples from Uniform(0,1) and computing the average of Xᵢ²:
   - What is Var(X²) for X ~ Uniform(0,1)?
   - How many samples for the Monte Carlo estimate to be within 0.001 of 1/3 with 95% probability?

5. *(Interview-level)* A federated learning system has K=100 clients, each with nₖ=50 local samples. The global model gradient is averaged across clients. If each client's gradient estimate has variance σ²=1:
   - What is Var(global gradient) if all clients have the same data distribution?
   - What if clients have different distributions (non-i.i.d. data)? Does LLN still apply?
   - What is the main challenge of federated learning from a LLN perspective?

---

## 14. Looking Ahead

**Day 20** — **The Central Limit Theorem (CLT).** Where LLN tells us that X̄ₙ → μ, the CLT tells us the rate and shape of this convergence — √n(X̄ₙ−μ)/σ → N(0,1). The CLT is why Normal distributions appear everywhere, why confidence intervals are bell-shaped, and why the Normal distribution is the "attractor" of probability theory.

---
*End of Day 19 | Next: Day 20 — The Central Limit Theorem*
