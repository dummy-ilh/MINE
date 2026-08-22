# Day 6 — Unit 1 Review & Hard Interview Problems
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Coverage:** Days 1–5 — Sample Spaces, Counting, Conditional Probability, Bayes, Independence
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Unit 1 Master Cheat Sheet

Before the hard problems, every formula from Days 1–5 in one place.

```
══════════════════════════════════════════════════════════
AXIOMS & BASIC RULES (Day 1)
══════════════════════════════════════════════════════════
Axioms:         P(A) ≥ 0,  P(Ω) = 1,  P(A∪B) = P(A)+P(B) if A∩B=∅
Complement:     P(Aᶜ) = 1 − P(A)
Inclusion-Ex:   P(A∪B) = P(A) + P(B) − P(A∩B)
Monotonicity:   A ⊆ B → P(A) ≤ P(B)
Boole's ineq:   P(A∪B) ≤ P(A) + P(B)

══════════════════════════════════════════════════════════
COUNTING (Day 2)
══════════════════════════════════════════════════════════
Multiplication: n₁ × n₂ × ... × nₖ
Permutation:    P(n,r) = n!/(n−r)!          [order matters]
Combination:    C(n,r) = n!/[r!(n−r)!]      [order doesn't matter]
With replace:   nʳ                           [order matters]
Multinomial:    n!/(n₁!·n₂!·...·nₖ!)
Bootstrap OOB:  P(not sampled) → 1/e ≈ 0.368

══════════════════════════════════════════════════════════
CONDITIONAL PROBABILITY (Day 3)
══════════════════════════════════════════════════════════
Definition:     P(A|B) = P(A∩B)/P(B)
Chain Rule:     P(A₁∩...∩Aₙ) = P(A₁)·P(A₂|A₁)·P(A₃|A₁,A₂)·...
LOTP:           P(A) = Σᵢ P(A|Bᵢ)·P(Bᵢ)    [Bᵢ partition Ω]
2-partition:    P(A) = P(A|B)·P(B) + P(A|Bᶜ)·P(Bᶜ)

══════════════════════════════════════════════════════════
BAYES' THEOREM (Day 4)
══════════════════════════════════════════════════════════
Bayes:          P(A|B) = P(B|A)·P(A) / P(B)
Proportional:   P(A|B) ∝ P(B|A)·P(A)
                Posterior ∝ Likelihood × Prior
Naive Bayes:    ĉ = argmax_k [log P(Cₖ) + Σᵢ log P(xᵢ|Cₖ)]
MAP:            θ_MAP = argmax_θ P(data|θ)·P(θ)

══════════════════════════════════════════════════════════
INDEPENDENCE (Day 5)
══════════════════════════════════════════════════════════
Independent:    P(A∩B) = P(A)·P(B)  ⟺  P(A|B) = P(A)
Cond. Indep:    P(A∩B|C) = P(A|C)·P(B|C)  ⟺  P(A|B,C) = P(A|C)
i.i.d.:         same distribution + mutually independent
Var (indep):    Var(X+Y) = Var(X) + Var(Y)
Series:         P(works) = Π P(cᵢ works)
Parallel:       P(works) = 1 − Π P(cᵢ fails)
```

---

## 2. Classic Hard Problems

---

### 🔢 Problem 1 — The Birthday Paradox

**Problem:** How many people do you need in a room so that the probability of at least two sharing a birthday exceeds 50%? Assume 365 days, all equally likely, birthdays independent.

**Why it matters in ML:** Hash collisions, random seed collisions, approximate nearest neighbor search — all rely on birthday-paradox-style analysis.

**Solution:**

It's easier to compute P(no shared birthday) and use the complement.

With k people:
```
P(no shared birthday) = 365/365 × 364/365 × 363/365 × ... × (365−k+1)/365

                      = 365! / [(365−k)! × 365ᵏ]
```

Term-by-term intuition:
- Person 1: any birthday — 365/365 = 1
- Person 2: must avoid person 1's birthday — 364/365
- Person 3: must avoid 2 birthdays — 363/365
- ...
- Person k: must avoid k−1 birthdays — (365−k+1)/365

P(at least one shared) = 1 − P(no shared)

Computing for key values:

| k (people) | P(at least one match) |
|---|---|
| 10 | 11.7% |
| 20 | 41.1% |
| 23 | **50.7%** ← crosses 50% |
| 30 | 70.6% |
| 50 | 97.0% |
| 70 | 99.9% |

**Answer: Just 23 people gives > 50% probability.**

**Why so surprising?** We compare every pair, not just against one person. With 23 people there are C(23,2) = 253 pairs — many chances for a collision.

**Approximation using independence:**
```
P(no collision) ≈ e^(−k²/730)

Setting this to 0.5: k ≈ √(730 · ln 2) ≈ 22.5 ≈ 23
```

**ML interview connection:** This is why hash tables need careful design — with n buckets and ~√n insertions, collisions become likely. Random seeds in ensemble methods should be chosen from a large space.

---

### 🔢 Problem 2 — The Coupon Collector Problem

**Problem:** A cereal company puts 1 of n different coupons in each box. Each box is equally likely to contain any coupon. How many boxes do you expect to buy to collect all n coupons?

**Why it matters in ML:** Expected number of samples to cover a space — relevant to active learning, exploration in RL, data augmentation coverage.

**Solution:**

Let T = total boxes needed. Break into phases:

- Phase 1: Get first new coupon. You have 0 of n, so any box gives a new coupon. P(new) = n/n = 1. Expected boxes = 1.
- Phase 2: Get second new coupon. You have 1 of n, P(new) = (n−1)/n. Expected boxes = n/(n−1).
- Phase k: You have k−1 of n, P(new) = (n−k+1)/n. Expected boxes = n/(n−k+1).

Each phase is Geometric — expected boxes in phase k = n/(n−k+1).

```
E[T] = n/n + n/(n−1) + n/(n−2) + ... + n/1

     = n · [1/1 + 1/2 + 1/3 + ... + 1/n]

     = n · Hₙ    where Hₙ is the nth Harmonic number
```

**Harmonic number approximation:** Hₙ ≈ ln(n) + 0.5772 (Euler–Mascheroni constant)

So: **E[T] ≈ n ln(n)**

| n (coupons) | E[T] = n·Hₙ |
|---|---|
| 5 | 11.4 boxes |
| 10 | 29.3 boxes |
| 50 | 224.5 boxes |
| 100 | 519.0 boxes |

**Answer:** To collect all n coupons, you need about **n·ln(n)** boxes on average.

**ML connection:**
- In RL, to visit all n states at least once, you need ~n·ln(n) steps (exploration)
- To see all classes at least once in random mini-batch sampling: ~k·ln(k) batches for k classes
- Data augmentation: to cover all n augmentation types at least once, expect n·ln(n) samples

---

### 🔢 Problem 3 — Geometric Probability (Buffon's Needle Lite)

**Problem:** You sample a point uniformly at random from the unit square [0,1]×[0,1]. What is the probability it falls inside the quarter circle of radius 1 centered at the origin?

**Why it matters in ML:** Monte Carlo estimation, numerical integration, probabilistic sampling — all rely on geometric probability.

**Solution:**

Area of quarter circle with radius 1 = π·r²/4 = π/4
Area of unit square = 1

P(point inside quarter circle) = (π/4) / 1 = **π/4 ≈ 0.7854**

**Monte Carlo π estimation** (a famous ML/CS interview topic):

Generate N random points (x, y) with x, y ~ Uniform(0,1).
Count how many satisfy x² + y² ≤ 1 → call it M.

Then: M/N ≈ π/4, so **π ≈ 4M/N**

With N = 10,000 points, you typically get π accurate to 2–3 decimal places.

**This is Monte Carlo integration** — estimating integrals by random sampling. It underlies:
- MCMC (Markov Chain Monte Carlo) in Bayesian inference
- Dropout as approximate Bayesian inference
- Policy gradient methods in RL

---

### 🔢 Problem 4 — The Prosecutor's Fallacy in Full (ML Interview Classic)

**Problem:** A DNA database has 1 million profiles. A crime scene sample matches one profile with P(match | innocent) = 1/1,000,000 (for any random person).

The prosecutor argues: "The probability of this match if the defendant is innocent is 1 in a million — so there's only a 1 in a million chance they're innocent."

Is this correct? Compute the actual P(innocent | match).

**Solution:**

This is a direct Bayes application. Let:
- I = "defendant is innocent"
- M = "DNA matches"

Given: P(M|Iᶜ) = 1 (guilty person always matches), P(M|I) = 1/1,000,000

**What is P(I)?** The database has 1,000,000 profiles. If exactly 1 is guilty, then before seeing the match:
P(I) = 999,999/1,000,000 ≈ 1 (nearly certain innocent)

Apply LOTP:
P(M) = P(M|Iᶜ)·P(Iᶜ) + P(M|I)·P(I)
= 1 × (1/1,000,000) + (1/1,000,000) × (999,999/1,000,000)
≈ 1/1,000,000 + 1/1,000,000
= **2/1,000,000**

Apply Bayes:
```
P(Iᶜ|M) = P(M|Iᶜ)·P(Iᶜ) / P(M)
         = (1 × 1/1,000,000) / (2/1,000,000)
         = 1/2
```

**P(guilty | DNA match) = 1/2, not 999,999/1,000,000!**

The prosecutor confused P(M|I) = 1/1,000,000 with P(I|M). With a database of 1M people, you expect about 1 innocent match — so when you find a match, it's 50/50.

**ML lesson:** When searching a large space (model parameters, database records), false positives dominate. This is why:
- Anomaly detection with 99.9% precision still generates many false alerts at scale
- Multiple hypothesis testing requires Bonferroni correction
- Rare class precision is critical, not just accuracy

---

### 🔢 Problem 5 — The Inspection Paradox

**Problem:** Buses arrive at a stop every 10 minutes on average (Poisson process, so interarrival times are Exponential with mean 10 min). You arrive at the stop at a random time. How long do you expect to wait?

**Naive answer:** 5 minutes (half the average interval). **Wrong.**

**Correct answer:**

In a Poisson process, the interval you land in is length-biased — you're more likely to arrive in a longer interval.

If interarrival time X ~ Exponential(λ), then:
- E[X] = 1/λ = 10 minutes
- E[wait] = E[X]/2... but for the interval you land in, E[length] = 2/λ = 20 minutes!

So E[wait] = E[interval you land in] / 2 = 20/2 = **10 minutes** — the full average interval, not half!

**Formal reason:** You're sampling the interval proportional to its length (longer intervals are easier to land in). The length-biased distribution of X has mean E[X²]/E[X] = (Var(X) + E[X]²)/E[X].

For Exponential: Var(X) = 1/λ², E[X] = 1/λ
E[length-biased X] = (1/λ² + 1/λ²) / (1/λ) = 2/λ = **20 minutes**

**ML connections:**
- **Survivorship bias:** Models evaluated on surviving examples (those that weren't dropped) face inspection paradox
- **Sampling bias:** Sampling user sessions by random time → you oversample long sessions
- **Training data imbalance:** Overrepresented examples are "longer intervals" — the model lands in them more

---

### 🔢 Problem 6 — Full ML Interview Scenario: Model Selection Under Uncertainty

**Problem:** You have three models trained on different data subsets:
- Model A: used 50% of time, P(correct) = 0.92
- Model B: used 30% of time, P(correct) = 0.85
- Model C: used 20% of time, P(correct) = 0.78

A prediction was made and it was **wrong**. Answer all of the following:

**(a)** What is the overall accuracy of the ensemble?

**(b)** Given the prediction was wrong, what is the probability each model made it?

**(c)** You want to disable the model most responsible for errors. Which do you disable?

**(d)** After disabling that model, its traffic is split equally between A and B. What is the new accuracy?

**Solution:**

**(a) Overall accuracy (LOTP):**
P(correct) = 0.92×0.50 + 0.85×0.30 + 0.78×0.20
= 0.460 + 0.255 + 0.156
= **0.871**

**(b) P(model | wrong) — apply Bayes:**

P(wrong) = 1 − 0.871 = 0.129

P(wrong ∩ A) = P(wrong|A)·P(A) = 0.08 × 0.50 = 0.040
P(wrong ∩ B) = 0.15 × 0.30 = 0.045
P(wrong ∩ C) = 0.22 × 0.20 = 0.044

Verify: 0.040 + 0.045 + 0.044 = 0.129 ✓

P(A|wrong) = 0.040/0.129 = **31.0%**
P(B|wrong) = 0.045/0.129 = **34.9%**
P(C|wrong) = 0.044/0.129 = **34.1%**

**(c)** Model B contributes 34.9% of errors — disable Model B.

*(Note: Even though C has the worst accuracy, B causes more total errors because it's used more often. This is a classic base-rate trap in model debugging.)*

**(d) New traffic after disabling B:**
- Model A: 50% + 15% = 65%
- Model C: 20% + 15% = 35%

New accuracy = 0.92×0.65 + 0.78×0.35
= 0.598 + 0.273
= **0.871**

Interesting — same accuracy! Because B was exactly average. In practice, you'd see improvement if the disabled model was below the weighted average, degradation if above.

---

### 🔢 Problem 7 — The Hat Check Problem (Derangements)

**Problem:** n people check their hats. The hats are returned randomly. What is the probability that NO person gets their own hat back?

**Why it matters in ML:** Permutation testing, shuffle validation, random assignment quality.

**Solution:**

A **derangement** is a permutation where no element appears in its original position.

Let Aᵢ = "person i gets their own hat."

P(at least one person gets own hat) = P(A₁ ∪ A₂ ∪ ... ∪ Aₙ)

By inclusion-exclusion:

```
P(A₁∪...∪Aₙ) = Σ P(Aᵢ) − Σ P(Aᵢ∩Aⱼ) + Σ P(Aᵢ∩Aⱼ∩Aₖ) − ...

= C(n,1)·(1/n) − C(n,2)·(1/n(n-1)) + C(n,3)·(1/n(n-1)(n-2)) − ...

= 1 − 1/2! + 1/3! − 1/4! + ... + (−1)ⁿ⁺¹/n!
```

Therefore:
```
P(no one gets own hat) = 1 − P(at least one gets own hat)
= 1 − 1 + 1/2! − 1/3! + 1/4! − ...
= Σₖ₌₀ⁿ (−1)ᵏ/k!
```

As n → ∞, this converges to **1/e ≈ 0.3679**

| n | P(derangement) |
|---|---|
| 1 | 0 |
| 2 | 0.500 |
| 3 | 0.333 |
| 4 | 0.375 |
| 5 | 0.367 |
| ∞ | **1/e ≈ 0.368** |

The probability stabilizes at 1/e remarkably quickly.

**ML connection:** In permutation testing, you shuffle labels n times and check how often the shuffled result beats the real one. The derangement result tells you the probability structure of random permutations — for large n, about 36.8% of permutations are derangements.

---

## 3. Rapid-Fire Review: 15 True/False Interview Questions

Answer each before reading the verdict.

| # | Statement | True/False |
|---|---|---|
| 1 | P(A\|B) + P(Aᶜ\|B) = 1 | **True** — complements under conditioning |
| 2 | If A and B are mutually exclusive, they are independent | **False** — ME with P>0 → dependent |
| 3 | P(A∪B) ≤ P(A) + P(B) always | **True** — Boole's inequality |
| 4 | Pairwise independence implies mutual independence | **False** — counterexample: 3 coins |
| 5 | P(A\|B) = P(B\|A) | **False** — prosecutor's fallacy |
| 6 | If P(A\|B) = P(A), then P(B\|A) = P(B) | **True** — independence is symmetric |
| 7 | Conditional independence implies marginal independence | **False** — see Naive Bayes features |
| 8 | Marginal independence implies conditional independence | **False** — Berkson's paradox |
| 9 | C(n,r) = C(n, n−r) | **True** — symmetry of combinations |
| 10 | Zero correlation implies independence | **False** — only for Gaussians |
| 11 | In a Poisson process, waiting time is memoryless | **True** — Exponential distribution property |
| 12 | P(A∩B) = P(A)·P(B\|A) always (not just when independent) | **True** — multiplication rule |
| 13 | With 23 people, P(shared birthday) > 0.5 | **True** — birthday paradox |
| 14 | Bootstrap samples on average exclude 1/e ≈ 36.8% of data | **True** — OOB samples |
| 15 | P(A\|B) is undefined if P(B) = 0 | **True** — division by zero |

---

## 4. Five Real ML Interview Problems (Mixed Topics)

---

**Interview Q1** *(Google/Meta style)*: "You're building a spam classifier. The word 'Rolex' appears in 40% of spam and 2% of ham. Spam is 20% of all email. If an email contains 'Rolex', what's the probability it's spam?"

**Answer:**
P(spam|Rolex) = P(Rolex|spam)·P(spam) / P(Rolex)
P(Rolex) = 0.40×0.20 + 0.02×0.80 = 0.08 + 0.016 = 0.096
P(spam|Rolex) = 0.08/0.096 = **83.3%**

---

**Interview Q2** *(Amazon style)*: "Your A/B test shows variant B has 5% higher CTR than A. The p-value is 0.03. Can you ship variant B?"

**Answer:** A p-value of 0.03 means if there were truly no difference, we'd see this result 3% of the time by chance. It does NOT mean P(B is better) = 97%. Before shipping:
- Check sample size and statistical power
- Check for novelty effect (did test run long enough?)
- Consider practical significance, not just statistical significance
- Use Bayesian A/B testing for direct P(B > A) calculation
- Check for Simpson's paradox across user segments

---

**Interview Q3** *(ML system design)*: "Your model has 99% accuracy on balanced test data. In production, the positive class is 0.1% of traffic. Is your model useful?"

**Answer:** Probably not without more info. A model that always predicts negative gets 99.9% accuracy. With 99% accuracy and 0.1% prevalence:
P(true positive | model says positive) = precision — need to compute via Bayes.
Assuming P(positive|actual positive) = 0.99, P(positive|actual negative) = 0.01:
P(pos) = 0.99×0.001 + 0.01×0.999 = 0.00099 + 0.00999 = 0.01098
Precision = 0.00099/0.01098 ≈ **9%** — only 9 in 100 flagged items are actually positive.
Conclusion: The model generates 10× more false positives than true positives — nearly useless for rare event detection.

---

**Interview Q4** *(Reasoning)*: "You have two children. One is a boy born on a Tuesday. What is P(both boys)?"

**Answer (careful!):**
This is a famous conditional probability trap. The answer is NOT 1/2 or 1/3.

Sample space: (day, sex) combinations. With "at least one boy born on Tuesday" as condition:
P(both boys | at least one boy born on Tuesday) = 13/27

*(Full derivation involves careful enumeration — the Tuesday detail surprisingly changes the answer from 1/3 to 13/27.)*

Key lesson: Carefully define your sample space before computing.

---

**Interview Q5** *(Deep)*: "Explain how the chain rule of probability relates to the architecture of autoregressive language models like GPT."

**Answer:**
GPT models the joint probability of a sequence of tokens using the chain rule:
```
P(x₁, x₂, ..., xₙ) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(xₙ|x₁,...,xₙ₋₁)
```
Each transformer layer computes P(xₜ | x₁,...,xₜ₋₁) — the conditional probability of the next token given all previous tokens. The attention mechanism is how the model captures the conditioning on all previous context. Training minimizes cross-entropy loss = negative log probability of the true token = −log P(xₜ|x₁,...,xₜ₋₁), summed over all positions. This is exactly maximizing the log chain rule expansion — MLE on the chain rule factorization.

---

## 5. Looking Ahead — Unit 2 Preview

Starting **Day 7**, we enter the heart of probability theory:

**Unit 2: Random Variables & Distributions (Days 7–14)**

| Day | Topic |
|---|---|
| Day 7 | Random Variables — PMF, CDF (Discrete) |
| Day 8 | Expectation, Variance, Standard Deviation |
| Day 9 | Bernoulli, Binomial, Geometric distributions |
| Day 10 | Poisson Distribution |
| Day 11 | Continuous RVs — PDF, CDF, Uniform, Exponential |
| Day 12 | Normal / Gaussian Distribution |
| Day 13 | Joint Distributions, Covariance, Correlation |
| Day 14 | Unit 2 Review + Hard Problems |

Random variables are how we go from abstract events to **numbers we can compute with** — the bridge to all of statistics and ML.

---
*End of Day 6 — Unit 1 Complete | Next: Day 7 — Random Variables, PMF & CDF*
