# Day 18 — Inequalities: Markov, Chebyshev & Jensen's
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 10 (Sections 10.1–10.3)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Inequalities Matter in ML

Inequalities let you make **distribution-free statements** — bounds that hold regardless of the exact distribution. This is crucial in ML because:

| Inequality | ML Application |
|---|---|
| **Markov** | Bound probability of large loss without knowing loss distribution |
| **Chebyshev** | Bound how far predictions deviate from mean |
| **Jensen's** | Prove convexity of loss functions, derive ELBO, understand log-sum-exp |
| **Hoeffding** | Sample complexity bounds, PAC learning |
| **Union Bound** | Multiple testing, generalization bounds |

These are the mathematical tools behind learning theory — why neural networks generalize, how many samples you need, when training is stable.

---

## 2. Markov's Inequality

> **Theorem (Markov's Inequality):** For any non-negative random variable X and a > 0:
> ```
>          E[X]
> P(X ≥ a) ≤ ————
>              a
> ```

### Proof

```
E[X] = ∫₀^∞ x·f(x)dx
     ≥ ∫ₐ^∞ x·f(x)dx         [drop the [0,a) part — non-negative]
     ≥ ∫ₐ^∞ a·f(x)dx         [x ≥ a on this region]
     = a · P(X ≥ a)

Therefore: P(X ≥ a) ≤ E[X]/a  ∎
```

### Requirements

- X must be **non-negative** (X ≥ 0)
- a must be positive (a > 0)
- Only requires knowledge of E[X]

### Sharpness

Markov's inequality is tight — equality is achieved when X = 0 with probability 1−p and X = a with probability p = E[X]/a.

### ML Interpretation

```
P(loss ≥ a) ≤ E[loss] / a
```

If average training loss = 0.1, then:
- P(loss ≥ 1.0) ≤ 0.1/1.0 = 10%
- P(loss ≥ 0.5) ≤ 0.1/0.5 = 20%
- P(loss ≥ 2.0) ≤ 0.1/2.0 = 5%

---

## 3. Chebyshev's Inequality

> **Theorem (Chebyshev's Inequality):** For any random variable X with finite mean μ and variance σ², and k > 0:
> ```
>                     σ²        1
> P(|X − μ| ≥ k) ≤ ———— = ————
>                     k²       k²/σ²
>
> Equivalently, with k = tσ:
>
> P(|X − μ| ≥ tσ) ≤ 1/t²
> ```

### Proof (via Markov)

Apply Markov's inequality to the non-negative RV Y = (X−μ)²:
```
P(|X−μ| ≥ k) = P((X−μ)² ≥ k²) ≤ E[(X−μ)²]/k² = σ²/k²  ∎
```

### Requirements

- X can be any distribution (discrete or continuous)
- Requires knowledge of μ and σ² only
- No assumption about distribution shape

### Chebyshev vs The Empirical Rule

| Rule | Distribution | P(|X−μ| ≥ 2σ) |
|---|---|---|
| Chebyshev (worst case) | Any | ≤ 1/4 = 25% |
| Empirical Rule | Normal only | ≈ 4.55% |
| True Uniform(−1,1) | Uniform | ≈ 0% (bounded) |

Chebyshev is weak but **universal**. The Normal is special — its tails decay much faster than Chebyshev guarantees.

### Two-Sided and One-Sided Forms

```
Two-sided: P(|X−μ| ≥ k) ≤ σ²/k²

One-sided: P(X−μ ≥ k) ≤ σ²/(σ²+k²)    [Cantelli's inequality]
```

---

## 4. Jensen's Inequality

> **Theorem (Jensen's Inequality):** For a **convex** function φ and any random variable X:
> ```
> φ(E[X]) ≤ E[φ(X)]
> ```
>
> For a **concave** function φ:
> ```
> φ(E[X]) ≥ E[φ(X)]
> ```

### What is Convexity?

A function φ is **convex** if for all x, y and λ ∈ [0,1]:
```
φ(λx + (1−λ)y) ≤ λφ(x) + (1−λ)φ(y)
```

Geometrically: the chord between any two points lies **above** the curve.

**Convex functions:** x², eˣ, |x|, x log x (for x>0), -log x, max(0,x) (ReLU)
**Concave functions:** log x, √x, -x², min(a,x)

**Test:** φ is convex iff φ''(x) ≥ 0 everywhere.

### Proof (for discrete X, two-point case)

For X taking values x₁, x₂ with probabilities p, 1-p:
```
φ(E[X]) = φ(px₁ + (1−p)x₂) ≤ pφ(x₁) + (1−p)φ(x₂) = E[φ(X)]
```

by the definition of convexity. ∎

### Jensen's Inequality — The Intuition

```
"The average of a convex function ≥ the function of the average"
```

For concave functions (like log):
```
E[log X] ≤ log E[X]
```

This is not just math — it's the reason log-sum-exp is always ≥ the log of any term, why the ELBO is a lower bound, and why Jensen's gap appears everywhere in information theory.

---

## 5. Jensen's Gap and the ELBO

In Variational Autoencoders:
```
log p(x) = log ∫ p(x,z) dz = log E_q[p(x,z)/q(z)]

By Jensen (log is concave):
log E_q[p(x,z)/q(z)] ≥ E_q[log p(x,z)/q(z)]
                      = E_q[log p(x,z)] − E_q[log q(z)]
                      = ELBO
```

**log p(x) ≥ ELBO** — the Evidence Lower BOund is exactly Jensen's inequality applied to the log-likelihood. This is one of the most important applications of Jensen's in all of ML.

The gap between log p(x) and the ELBO equals the KL divergence KL(q||p):
```
log p(x) = ELBO + KL(q(z|x) || p(z|x))
```

Maximizing ELBO ↔ minimizing the KL gap ↔ making q(z|x) approximate p(z|x) well.

---

## 6. Hoeffding's Inequality (Bonus — Critical for Interviews)

> **Theorem (Hoeffding's Inequality):** If X₁,...,Xₙ are independent with aᵢ ≤ Xᵢ ≤ bᵢ, and X̄ = (1/n)Σ Xᵢ:
> ```
>                         −2n²ε²
> P(|X̄ − E[X̄]| ≥ ε) ≤ 2exp(————————————)
>                         Σ(bᵢ−aᵢ)²
> ```

For i.i.d. bounded [a,b]:
```
P(|X̄ − μ| ≥ ε) ≤ 2exp(−2nε²/(b−a)²)
```

### Why It's Stronger Than Chebyshev

Chebyshev gives polynomial decay: P ≤ σ²/(nε²)
Hoeffding gives **exponential decay**: P ≤ 2exp(−2nε²/(b−a)²)

For large n, Hoeffding's bound is exponentially tighter.

### PAC Learning Bound

For accuracy ε with confidence 1−δ:
```
P(|accuracy − true accuracy| ≥ ε) ≤ δ

Hoeffding: 2exp(−2nε²) ≤ δ
→ n ≥ log(2/δ) / (2ε²)
```

To achieve ε=0.01 accuracy with 95% confidence (δ=0.05):
```
n ≥ log(40) / (2×0.0001) = 3.69 / 0.0002 = 18,450 samples
```

**This is the sample complexity bound** — the fundamental answer to "how much data do I need?"

---

## 7. The Union Bound (Boole's Inequality)

> **Union Bound:** For any events A₁, A₂, ..., Aₙ:
> ```
> P(A₁ ∪ A₂ ∪ ... ∪ Aₙ) ≤ P(A₁) + P(A₂) + ... + P(Aₙ)
> ```

### In ML

If you test m hypotheses each at level α:
```
P(at least one false positive) ≤ m·α
```

Set m·α ≤ δ → α = δ/m (Bonferroni correction, Day 11).

For generalization bounds: if model has k parameters and each can fail with probability δ/k, union bound says total failure probability ≤ k·(δ/k) = δ.

---

## 8. Connecting the Inequalities

```
Markov: P(X≥a) ≤ E[X]/a           [uses 1st moment, X≥0]
    ↓ apply to (X−μ)²
Chebyshev: P(|X−μ|≥k) ≤ σ²/k²    [uses 2nd moment, any X]
    ↓ apply to eˢˣ (MGF)
Chernoff: P(X≥a) ≤ e^(−sa)M(s)   [uses all moments, exponential bound]
    ↓ optimize over s
Hoeffding: exponential bound for bounded RVs

Jensen:    φ(E[X]) ≤ E[φ(X)]      [convexity, any moments]
    ↓ applied to log
ELBO:      log p(x) ≥ E_q[log p(x,z)/q(z)]
```

Each builds on the previous, getting tighter as you use more information.

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Markov: Bounding Loss Probability

**Problem:** A neural network's training loss has E[loss] = 0.25. Assume loss ≥ 0.

**(a)** Bound P(loss ≥ 1.0)
**(b)** Bound P(loss ≥ 0.5)
**(c)** For what threshold a is P(loss ≥ a) ≤ 5%?
**(d)** If you also know Var(loss)=0.04, use Chebyshev to get a tighter bound for P(loss ≥ 1.0).

**Solution:**

**(a)** Markov:
```
P(loss ≥ 1.0) ≤ E[loss]/1.0 = 0.25/1.0 = 0.25
```
At most 25% chance of loss ≥ 1.0.

**(b)**
```
P(loss ≥ 0.5) ≤ 0.25/0.5 = 0.50
```
At most 50% chance of loss ≥ 0.5. (Weak bound — Markov is not designed for close to the mean.)

**(c)** Set E[loss]/a ≤ 0.05:
```
a ≥ E[loss]/0.05 = 0.25/0.05 = 5.0
```
Can only guarantee P(loss ≥ 5.0) ≤ 5% from Markov alone.

**(d)** Chebyshev with μ=0.25, σ²=0.04, σ=0.2:
```
P(loss ≥ 1.0) = P(loss − 0.25 ≥ 0.75) ≤ P(|loss − 0.25| ≥ 0.75)
              ≤ σ²/(0.75)² = 0.04/0.5625 = 0.0711
```

Chebyshev gives 7.1% vs Markov's 25% — **3.5× tighter** because we used variance information.

---

### 🔢 Numerical 2 — Chebyshev: Confidence in Sample Mean

**Problem:** You estimate model accuracy by testing on n=100 samples. True accuracy μ=0.85, but you don't know this. Assuming accuracy of each prediction is i.i.d. Bernoulli(0.85).

**(a)** E[X̄] and Var(X̄) for the sample mean accuracy.
**(b)** P(|X̄ − 0.85| ≥ 0.05) using Chebyshev.
**(c)** How many samples needed so P(|X̄−μ| ≥ 0.05) ≤ 0.05 by Chebyshev?
**(d)** Compare to the Normal approximation (which gives the same bound at much smaller n).

**Solution:**

X̄ = sample accuracy. E[X̄] = μ = 0.85. Var(X̄) = p(1−p)/n = 0.85×0.15/100 = 0.001275.

**(a)**
```
E[X̄] = 0.85
SD(X̄) = √0.001275 ≈ 0.0357
```

**(b)** Chebyshev with k=0.05:
```
P(|X̄−0.85| ≥ 0.05) ≤ Var(X̄)/k² = 0.001275/0.0025 = 0.51
```

At most 51% — very weak! Chebyshev is worst-case.

**(c)** Set Var(X̄)/k² ≤ 0.05:
```
p(1−p)/(n·k²) ≤ 0.05
n ≥ p(1−p)/(k²×0.05) = 0.1275/(0.0025×0.05) = 0.1275/0.000125 = 1020
```

Need n ≥ 1020 samples by Chebyshev.

**(d)** Normal approximation (CLT, Day 20):
```
P(|X̄−μ| ≥ 0.05) ≈ P(|Z| ≥ 0.05/0.0357) = P(|Z| ≥ 1.40) = 2(1−0.919) = 0.162
```

With n=100, Normal gives 16.2% which is realistic. Chebyshev gives 51% — much weaker.

For Normal to give 5%: need |Z| ≥ 1.96, so 0.05/√(0.1275/n) = 1.96:
n = 0.1275×(1.96/0.05)² = 0.1275×1537 ≈ **196 samples**.

Chebyshev requires 1020, Normal requires 196 — 5× fewer samples because Normal is much better calibrated for bell-shaped distributions.

**ML lesson:** Chebyshev is a worst-case guarantee. In practice (where distributions are approximately Normal by CLT), much smaller samples suffice. Chebyshev is used for **distribution-free guarantees**; Normal approximation for **practical sample sizing**.

---

### 🔢 Numerical 3 — Jensen's: Why log E[X] ≥ E[log X]

**Problem:** Model output probabilities X take values {0.1, 0.5, 0.9} each with probability 1/3.

**(a)** Compute E[log X] — expected log-probability (negative cross-entropy).
**(b)** Compute log E[X] — log of average probability.
**(c)** Verify Jensen's inequality: log E[X] ≥ E[log X].
**(d)** Interpret in terms of cross-entropy loss.

**Solution:**

**(a)**
```
E[log X] = (1/3)[log(0.1) + log(0.5) + log(0.9)]
         = (1/3)[−2.303 + (−0.693) + (−0.105)]
         = (1/3)(−3.101) = −1.034
```

**(b)**
```
E[X] = (0.1+0.5+0.9)/3 = 0.5
log E[X] = log(0.5) = −0.693
```

**(c)**
```
log E[X] = −0.693 ≥ E[log X] = −1.034  ✓
```

Jensen's inequality verified: log (concave) → log E[X] ≥ E[log X].

The gap: −0.693 − (−1.034) = **0.341** — this is the Jensen's gap.

**(d)** Cross-entropy loss = E[−log P(correct)] = −E[log X] = 1.034.

The Jensen's gap here is 1.034 − 0.693 = 0.341. If you used the "average probability" as your loss (log of average), you'd underestimate the true cross-entropy by 0.341. Jensen's inequality tells you that log of average is ALWAYS an optimistic (lower) estimate of average log.

**ML insight:** This is why you average **log-probabilities** (cross-entropy), not **log of average probability**. Jensen's gap = the penalty for averaging in probability space rather than log space.

---

### 🔢 Numerical 4 — Jensen's: Convexity of MSE

**Problem:** A model predicts Y using feature X. Prove that:

**(a)** E[(Y−c)²] is minimized at c=E[Y] using Jensen's (alternative proof).
**(b)** For any convex loss φ, E[φ(Y−c)] is minimized at c = something (what?).
**(c)** Use Jensen to lower bound E[eˣ] for X ~ N(0,1).

**Solution:**

**(a)** Expand:
```
E[(Y−c)²] = E[Y²] − 2cE[Y] + c²
```
This is a quadratic in c. Minimum at c = E[Y]. Alternatively:

By the bias-variance decomposition:
```
E[(Y−c)²] = Var(Y) + (E[Y]−c)²  ≥  Var(Y)
```
Minimized when (E[Y]−c)² = 0, i.e., c = E[Y]. ∎

**(b)** For strictly convex φ, E[φ(Y−c)] is minimized at the **median** of Y when φ=|·|, or the **mean** when φ=|·|². Jensen alone doesn't pin down the minimizer — you need calculus (set derivative to zero).

**(c)** eˣ is convex (φ''(x)=eˣ > 0). By Jensen:
```
E[eˣ] ≥ e^(E[X]) = e⁰ = 1    for X ~ N(0,1)
```

True value: E[eˣ] = M(1) = e^(0+1/2) = e^(0.5) ≈ 1.649

So Jensen gives: E[eˣ] ≥ 1 (true — 1.649 ≥ 1 ✓). The bound is loose here; Jensen gives the direction, not the exact value.

---

### 🔢 Numerical 5 — Chebyshev: Stability of Gradient Descent

**Problem:** At each step of SGD, the gradient estimate G has:
- E[G] = g (true gradient, E[G]=g=−0.1, negative = descending)
- Var(G) = 0.09 (gradient noise variance)

**(a)** P(gradient has wrong sign, i.e., G > 0) using Chebyshev.
**(b)** With mini-batch size B=9 (average B gradients), find Var(Ḡ).
**(c)** P(Ḡ > 0) with batch size 9.
**(d)** What batch size reduces P(wrong sign) below 5%?

**Solution:**

**(a)** Single gradient G: μ=−0.1, σ²=0.09, σ=0.3

Wrong sign means G > 0, i.e., G − (−0.1) > 0.1, i.e., G−μ > 0.1.

Using one-sided Chebyshev (Cantelli's inequality):
```
P(G − μ ≥ k) ≤ σ²/(σ²+k²)    with k=0.1

P(G > 0) ≤ 0.09/(0.09+0.01) = 0.09/0.10 = 0.90
```

Cantelli gives at most 90% — very weak. True value (Normal): P(G>0) = P(Z > 0.1/0.3) = P(Z>0.333) = 36.8%.

**(b)** With batch size B=9:
```
Var(Ḡ) = Var(G)/B = 0.09/9 = 0.01,  σḠ = 0.1
```

**(c)**
```
P(Ḡ > 0) ≤ 0.01/(0.01+0.01) = 0.01/0.02 = 0.50
```

Cantelli bound halved. True (Normal): P(Z > 0.1/0.1) = P(Z>1) = 15.9%.

**(d)** Set Cantelli bound ≤ 0.05:
```
σ²/B / (σ²/B + k²) ≤ 0.05
0.09/B / (0.09/B + 0.01) ≤ 0.05
0.09/(0.09 + 0.01B) ≤ 0.05
0.09 ≤ 0.05(0.09 + 0.01B)
0.09 ≤ 0.0045 + 0.0005B
0.0855 ≤ 0.0005B
B ≥ 171
```

Need batch size ≥ 171 to guarantee (via Cantelli) P(wrong sign) ≤ 5%.

**ML insight:** This quantifies why large batches stabilize training — they reduce gradient noise variance as 1/B. The tradeoff is computational cost and reduced generalization (large batches often find sharp minima that generalize worse — the "sharp vs flat minima" debate in deep learning).

---

### 🔢 Numerical 6 — Jensen's: ELBO Derivation

**Problem:** For a latent variable model with joint distribution p(x,z) and approximate posterior q(z|x):

**(a)** Show log p(x) ≥ E_q[log p(x,z)/q(z|x)] using Jensen's.
**(b)** Write the gap as KL(q||p).
**(c)** For a VAE: q(z|x) ~ N(μ(x), σ²(x)), p(z) ~ N(0,1). Write the KL term explicitly.

**Solution:**

**(a)**
```
log p(x) = log ∫ p(x,z) dz

         = log ∫ q(z|x) · p(x,z)/q(z|x) dz

         = log E_q[p(x,z)/q(z|x)]

         ≥ E_q[log p(x,z)/q(z|x)]    [Jensen: log is concave]

         = E_q[log p(x,z)] − E_q[log q(z|x)]

         = ELBO
```

ELBO = E_q[log p(x|z)] + E_q[log p(z)] − E_q[log q(z|x)]
     = Reconstruction − KL(q(z|x) || p(z))

**(b)**
```
log p(x) − ELBO = log E_q[p/q] − E_q[log p/q]
                = KL(q(z|x) || p(z|x))  ≥ 0    [KL is always ≥ 0]
```

Jensen's gap = KL divergence between approximate and true posterior.

**(c)** For Gaussian q and p:
```
KL(N(μ,σ²) || N(0,1)) = ½(μ² + σ² − log σ² − 1)
```

**This is the VAE regularization term** — derived from Jensen's inequality. The ELBO objective:
```
L_VAE = E_q[log p(x|z)] − ½(μ(x)² + σ(x)² − log σ(x)² − 1)
         [reconstruction]   [KL regularization — Jensen's gap]
```

Everything in the VAE loss is a consequence of Jensen's inequality applied to the log-likelihood.

---

### 🔢 Numerical 7 — Hoeffding: Sample Complexity

**Problem:** You want to estimate a model's true accuracy μ within ε=0.02 with probability ≥ 95% (δ=0.05). Each test sample gives accuracy 0 or 1 (bounded in [0,1]).

**(a)** Sample size by Chebyshev (worst case variance).
**(b)** Sample size by Hoeffding.
**(c)** For ε=0.01 and δ=0.01, compare both.
**(d)** How does Hoeffding scale with ε and δ?

**Solution:**

**(a) Chebyshev:** Worst case Var = 1/4 (Bernoulli at p=0.5).
```
P(|X̄−μ| ≥ ε) ≤ Var/(nε²) = 1/(4nε²) ≤ δ
n ≥ 1/(4δε²) = 1/(4×0.05×0.0004) = 1/0.00008 = 12,500
```

**(b) Hoeffding:** with b−a=1:
```
P(|X̄−μ| ≥ ε) ≤ 2exp(−2nε²) ≤ δ
2exp(−2nε²) = 0.05
exp(−2nε²) = 0.025
−2nε² = ln(0.025) = −3.689
n ≥ 3.689/(2×0.0004) = 3.689/0.0008 = 4,611
```

Hoeffding needs **4,611** vs Chebyshev's **12,500** — almost 3× fewer samples.

**(c)** For ε=0.01, δ=0.01:

Chebyshev: n ≥ 1/(4×0.01×0.0001) = 250,000
Hoeffding: n ≥ ln(200)/(2×0.0001) = 5.298/0.0002 = 26,491

Hoeffding needs ~26,500 vs Chebyshev's ~250,000 — **nearly 10× fewer**.

**(d) Scaling:**
```
Hoeffding: n ≥ log(2/δ) / (2ε²)

ε halved: n quadruples (n ∝ 1/ε²)
δ halved: n increases by log(4)/log(2/δ) — logarithmically
```

**ML insight:** The 1/ε² scaling is fundamental — getting 10× more accurate requires 100× more data. This is why going from 90% to 99% accuracy is so much harder than going from 80% to 90%.

---

## 10. Summary Table: Which Inequality to Use

| Situation | Information Available | Use |
|---|---|---|
| X ≥ 0, only mean known | E[X] | Markov |
| Any X, mean and variance known | μ, σ² | Chebyshev |
| Bounded X, need tight bound | [a,b], μ | Hoeffding |
| Convexity/concavity argument | φ convex/concave | Jensen's |
| Multiple events, any | P(Aᵢ) | Union Bound |
| Normal assumption justified | μ, σ² | Empirical Rule / Z-test |

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "State Markov's inequality and its conditions" | P(X≥a)≤E[X]/a; requires X≥0 |
| "State Chebyshev's inequality" | P(\|X−μ\|≥k)≤σ²/k²; any distribution |
| "How do you derive Chebyshev from Markov?" | Apply Markov to (X−μ)² |
| "State Jensen's inequality" | φ convex → φ(E[X])≤E[φ(X)] |
| "Why is ELBO a lower bound on log p(x)?" | Jensen's inequality applied to log (concave) |
| "What is Hoeffding's inequality?" | Exponential tail bound for bounded RVs |
| "How many samples needed to estimate accuracy within ε?" | n≥log(2/δ)/(2ε²) by Hoeffding |
| "Why does log E[X] ≥ E[log X]?" | Jensen's: log is concave |
| "What is the Union Bound and when do you use it?" | P(∪Aᵢ)≤ΣP(Aᵢ); multiple testing, generalization |

---

## 12. Key Formulas — Cheat Sheet for Day 18

```
Markov's Inequality (X ≥ 0):
    P(X ≥ a) ≤ E[X]/a

Chebyshev's Inequality:
    P(|X−μ| ≥ k) ≤ σ²/k²
    P(|X−μ| ≥ tσ) ≤ 1/t²

Cantelli (one-sided Chebyshev):
    P(X−μ ≥ k) ≤ σ²/(σ²+k²)

Jensen's Inequality:
    φ convex: φ(E[X]) ≤ E[φ(X)]
    φ concave: φ(E[X]) ≥ E[φ(X)]
    log is concave: log E[X] ≥ E[log X]
    exp is convex: exp(E[X]) ≤ E[exp(X)]

Hoeffding's Inequality (Xᵢ ∈ [aᵢ,bᵢ]):
    P(|X̄−μ| ≥ ε) ≤ 2exp(−2n²ε²/Σ(bᵢ−aᵢ)²)
    i.i.d. [0,1]: P(|X̄−μ| ≥ ε) ≤ 2exp(−2nε²)

Sample complexity:
    n ≥ log(2/δ)/(2ε²)    [Hoeffding, bounded [0,1]]
    n ≥ 1/(4δε²)          [Chebyshev, worst case]

Union Bound:
    P(∪ Aᵢ) ≤ Σ P(Aᵢ)
    Bonferroni: use α/m per test for m tests

ELBO (Jensen applied to log):
    log p(x) ≥ E_q[log p(x,z)/q(z|x)] = ELBO
    Gap = KL(q(z|x) || p(z|x)) ≥ 0

VAE KL term:
    KL(N(μ,σ²)||N(0,1)) = ½(μ²+σ²−log σ²−1)

Key convex functions: x², eˣ, |x|, x log x, max(0,x)
Key concave functions: log x, √x, min(a,x)
```

---

## 13. Practice Problems (Solve Before Day 19)

1. Loss L has E[L]=0.5 and Var(L)=0.1. Using Markov and Chebyshev, find upper bounds for P(L≥2.0). Which is tighter?

2. X̄ is the sample mean of n=50 i.i.d. RVs with μ=3 and σ²=4. Find the Chebyshev bound for P(|X̄−3|≥0.5). Compare to the Normal approximation.

3. **Prove** that for any convex φ: φ(λx+(1−λ)y) ≤ λφ(x)+(1−λ)φ(y). Use this to prove Jensen's for two-point distributions.

4. X ~ Exponential(1). Verify Jensen's inequality for φ(x)=x²:
   - Compute E[X] and φ(E[X]) = (E[X])²
   - Compute E[X²] = E[φ(X)]
   - Verify E[X²] ≥ (E[X])²

5. *(Interview-level)* A ML model is evaluated on m=100 different metrics (accuracy, F1, AUC, etc.). Each metric is estimated on n=500 samples. Using Hoeffding and the union bound, find how many samples are needed so ALL metric estimates are within ε=0.02 of their true values with probability ≥ 95%.

---

## 14. Looking Ahead

**Day 19** — **Law of Large Numbers (LLN).** The mathematical guarantee that learning from data works — as n→∞, empirical averages converge to true expectations. We prove both the Weak and Strong LLN, connect them to Chebyshev's inequality, and see why the LLN is the foundation of every ML training algorithm.

---
*End of Day 18 | Next: Day 19 — Law of Large Numbers*
