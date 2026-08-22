# Day 5 — Independence & Conditional Independence
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 2 (Section 2.5)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Independence is Central to ML

Independence is the assumption that makes ML tractable. Without it, the math explodes.

| ML Concept | Independence Assumption |
|---|---|
| Naive Bayes | Features conditionally independent given class |
| Logistic Regression | Samples i.i.d. (independent, identically distributed) |
| Neural network training | Mini-batch samples assumed independent |
| Bootstrapping | Draws with replacement ≈ independent |
| Train/test split | Test set assumed independent of training |
| Dropout | Each neuron dropped independently (Bernoulli) |
| Gaussian Naive Bayes | Features independent Gaussians given class |
| Hidden Markov Models | Observations independent given hidden state |

When someone says "i.i.d. assumption" in an interview — they mean independence. Understanding it deeply separates strong candidates.

---

## 2. Independence of Two Events

> **Definition:** Events A and B are **independent** if:
> ```
> P(A ∩ B) = P(A) · P(B)
> ```

**Equivalent forms** (all say the same thing):
```
P(A ∩ B) = P(A) · P(B)        [definition]
P(A|B)   = P(A)               [B gives no info about A]
P(B|A)   = P(B)               [A gives no info about B]
```

**Intuition:** Knowing B happened tells you nothing new about A. The occurrence of B doesn't update your belief about A.

### Important: Independence ≠ Mutually Exclusive

This is the single most common confusion in interviews:

| | Mutually Exclusive | Independent |
|---|---|---|
| **Definition** | A ∩ B = ∅ | P(A∩B) = P(A)·P(B) |
| **Can both occur?** | No | Yes (usually) |
| **If P(A),P(B) > 0** | P(A∩B) = 0 | P(A∩B) = P(A)·P(B) > 0 |
| **Relationship** | Knowing B occurred means A didn't | Knowing B tells you nothing about A |
| **Are they compatible?** | Only if P(A)=0 or P(B)=0 | Always possible |

**Key insight:** If A and B are mutually exclusive with P(A) > 0 and P(B) > 0, they are **dependent** — knowing A occurred tells you B did NOT occur.

---

## 3. Independence of Multiple Events

> **Definition:** Events A₁, A₂, ..., Aₙ are **mutually independent** if for every subset {Aᵢ₁, ..., Aᵢₖ}:
> ```
> P(Aᵢ₁ ∩ Aᵢ₂ ∩ ... ∩ Aᵢₖ) = P(Aᵢ₁) · P(Aᵢ₂) · ... · P(Aᵢₖ)
> ```

This must hold for **all subsets**, not just pairs.

### Pairwise Independence ≠ Mutual Independence

Events can be pairwise independent but NOT mutually independent. Classic counterexample:

**Example:** Toss two fair coins. Define:
- A = "Coin 1 is Heads"
- B = "Coin 2 is Heads"
- C = "Both coins show the same face"

Check pairwise:
- P(A) = 1/2, P(B) = 1/2, P(C) = 1/2
- P(A∩B) = P(HH) = 1/4 = P(A)·P(B) ✓ independent
- P(A∩C) = P(HH) = 1/4 = P(A)·P(C) ✓ independent
- P(B∩C) = P(HH) = 1/4 = P(B)·P(C) ✓ independent

Check mutual independence:
- P(A∩B∩C) = P(HH) = 1/4
- P(A)·P(B)·P(C) = (1/2)³ = 1/8

1/4 ≠ 1/8 → **NOT mutually independent!**

Knowing A and B both occurred (both heads) tells you C definitely occurred — so C is dependent on {A,B} jointly.

---

## 4. Conditional Independence — The Most Important Concept

> **Definition:** A and B are **conditionally independent given C** if:
> ```
> P(A ∩ B | C) = P(A|C) · P(B|C)
> ```

Equivalently:
```
P(A | B, C) = P(A | C)
```

**Intuition:** Once you know C, knowing B gives no additional information about A.

### The Critical Asymmetry

**Independence does NOT imply conditional independence, and conditional independence does NOT imply independence.**

These are completely separate concepts. This is a top interview question.

---

## 5. Four Scenarios — Every Interview Covers These

### Scenario 1: Independent, but NOT Conditionally Independent

**Example:** Let:
- A = "it rained today"
- B = "ground is wet"
- C = "sprinkler was on"

A and B might be marginally independent (in some contrived setup). But given C (sprinkler on), knowing A (rain) changes P(B) — rain makes it MORE wet even with sprinkler. So A and B are not conditionally independent given C.

Better classical example — **Berkson's Paradox:**
- A = "talented at math"
- B = "talented at writing"

In the general population, A and B may be independent (or even slightly positively correlated).

But in a group **conditional on** C = "admitted to a top university":
- Students admitted are talented in at least one area
- Knowing someone is NOT a math genius makes it more likely they're a writing genius
- So A and B become **negatively correlated** given C

Conditioning on a common effect creates spurious dependence. This is **selection bias** in ML.

---

### Scenario 2: Conditionally Independent, but NOT Marginally Independent

**Example — Naive Bayes justification:**
- A = "email contains 'free'"
- B = "email contains 'winner'"
- C = "email is spam"

Marginally: A and B are correlated — emails with "free" tend to also have "winner" (both spam words).

Conditionally on C = spam: A and B become approximately independent — knowing C=spam already explains why both words appear. Within the spam class, whether the word "free" appears doesn't tell you much more about whether "winner" appears.

**This is exactly why Naive Bayes works** — features are marginally correlated but approximately conditionally independent given the class.

---

### Scenario 3: Causal Chain (d-separation)

```
A → C → B
```

A and B are marginally dependent (A influences C which influences B).
A and B are **conditionally independent given C** — once you know C, A gives no extra info about B.

**ML example:** Word → Topic → Document. Given the topic, the specific word used doesn't further determine document class.

---

### Scenario 4: Common Cause (Confounding)

```
A ← C → B
```

A and B are marginally dependent (both caused by C).
A and B are **conditionally independent given C** — once you know the common cause, A and B are unrelated.

**ML example:** Ice cream sales and drowning rates are correlated. Both caused by C = summer/hot weather. Given C, they're conditionally independent. This is **confounding** — a huge problem in causal ML.

---

## 6. i.i.d. — Independent and Identically Distributed

> A sequence X₁, X₂, ..., Xₙ is **i.i.d.** if:
> 1. Each Xᵢ has the same distribution F (identically distributed)
> 2. X₁, X₂, ..., Xₙ are mutually independent

**Why it matters in ML:**
- Most ML theory assumes training data is i.i.d. samples from some distribution P(X, Y)
- The Law of Large Numbers requires i.i.d.
- The Central Limit Theorem requires i.i.d.
- When i.i.d. fails → distribution shift, data leakage, autocorrelation

**When i.i.d. is violated:**
| Situation | Violation |
|---|---|
| Time series data | Not independent (autocorrelated) |
| Same user appearing multiple times | Not independent (correlated within user) |
| Train/test from different time periods | Not identically distributed (distribution shift) |
| Data leakage | Train and test not independent |

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — Checking Independence

**Problem:** Roll a fair die. Define:
- A = "roll is even" = {2, 4, 6}
- B = "roll is ≤ 3" = {1, 2, 3}
- C = "roll is 2" = {2}

Check which pairs are independent.

**Solution:**

P(A) = 3/6 = 1/2
P(B) = 3/6 = 1/2
P(C) = 1/6

**A and B:**
P(A ∩ B) = P({2}) = 1/6
P(A)·P(B) = 1/2 × 1/2 = 1/4
1/6 ≠ 1/4 → **Dependent**

Intuition: knowing the roll is ≤ 3 changes the probability it's even (only 2 qualifies among {1,2,3}).

**A and C:**
P(A ∩ C) = P({2}) = 1/6
P(A)·P(C) = 1/2 × 1/6 = 1/12
1/6 ≠ 1/12 → **Dependent**

Knowing C = {2} tells you it's definitely even.

**B and C:**
P(B ∩ C) = P({2}) = 1/6
P(B)·P(C) = 1/2 × 1/6 = 1/12
1/6 ≠ 1/12 → **Dependent**

---

### 🔢 Numerical 2 — Independence in Series vs Parallel Systems

**Problem:** A data pipeline has components that can fail independently.

**Series system** (all must work): Components A, B, C with reliability (P(works)) = 0.9, 0.95, 0.98

**Parallel system** (at least one must work): Same components.

Find P(system works) for each.

**Solution:**

**Series** (all must work — intersection):
P(system works) = P(A) · P(B) · P(C)  [by independence]
= 0.9 × 0.95 × 0.98
= **0.8379**

Even reliable components degrade badly in series. A pipeline with 100 steps each 99% reliable:
P(all work) = 0.99¹⁰⁰ = **0.366** — only 36.6% reliable!

**Parallel** (at least one works — use complement):
P(all fail) = P(Aᶜ)·P(Bᶜ)·P(Cᶜ) = 0.10 × 0.05 × 0.02 = 0.0001
P(system works) = 1 - 0.0001 = **0.9999**

**ML lesson:** Ensemble methods (Random Forests, Boosting) work like parallel systems — even weak learners combine to high reliability. Model checkpointing, redundant data storage = parallel systems.

---

### 🔢 Numerical 3 — Conditional Independence: Naive Bayes from Scratch

**Problem:** Classify documents as "Tech" or "Sports". Two binary features:
- X₁ = "contains 'algorithm'" (1 = yes, 0 = no)
- X₂ = "contains 'player'" (1 = yes, 0 = no)

From training data:
- P(Tech) = 0.5, P(Sports) = 0.5
- P(X₁=1 | Tech) = 0.7,   P(X₁=1 | Sports) = 0.1
- P(X₂=1 | Tech) = 0.1,   P(X₂=1 | Sports) = 0.8

**Part A:** Are X₁ and X₂ marginally independent?

**Part B:** Are X₁ and X₂ conditionally independent given class?

**Part C:** Classify document with X₁=1, X₂=0.

**Solution:**

**Part A — Marginal independence:**

P(X₁=1) = P(X₁=1|Tech)P(Tech) + P(X₁=1|Sports)P(Sports)
= 0.7×0.5 + 0.1×0.5 = 0.35 + 0.05 = 0.40

P(X₂=1) = 0.1×0.5 + 0.8×0.5 = 0.05 + 0.40 = 0.45

P(X₁=1, X₂=1) = P(X₁=1,X₂=1|Tech)P(Tech) + P(X₁=1,X₂=1|Sports)P(Sports)

Under conditional independence assumption:
= P(X₁=1|Tech)P(X₂=1|Tech)×0.5 + P(X₁=1|Sports)P(X₂=1|Sports)×0.5
= (0.7×0.1)×0.5 + (0.1×0.8)×0.5
= 0.035 + 0.040 = 0.075

P(X₁=1)·P(X₂=1) = 0.40 × 0.45 = 0.18

0.075 ≠ 0.18 → **X₁ and X₂ are NOT marginally independent**

They're correlated — both driven by the latent class variable.

**Part B:** By the Naive Bayes assumption, we treat them as conditionally independent given class. The model assumes this — whether it's exactly true is another matter.

**Part C — Classification:**

P(Tech | X₁=1, X₂=0) ∝ P(Tech)·P(X₁=1|Tech)·P(X₂=0|Tech)
= 0.5 × 0.7 × (1−0.1)
= 0.5 × 0.7 × 0.9 = **0.315**

P(Sports | X₁=1, X₂=0) ∝ P(Sports)·P(X₁=1|Sports)·P(X₂=0|Sports)
= 0.5 × 0.1 × (1−0.8)
= 0.5 × 0.1 × 0.2 = **0.010**

Normalize: Total = 0.315 + 0.010 = 0.325

P(Tech | X₁=1, X₂=0) = 0.315/0.325 = **96.9% → Classify as TECH** ✓

---

### 🔢 Numerical 4 — The i.i.d. Assumption in Practice

**Problem:** A model is trained on customer transactions. You sample 5 transactions.

**(a)** If transactions are i.i.d. with P(fraud) = 0.02, what is P(at least one fraud in 5 transactions)?

**(b)** If transactions are NOT independent (fraud comes in clusters — one fraud triggers more), and instead P(fraud on transaction k | fraud on transaction k-1) = 0.4, while P(fraud on first) = 0.02, what is P(fraud on transaction 2)?

**Solution:**

**(a) i.i.d. case:**
P(no fraud in 5) = (1 − 0.02)⁵ = 0.98⁵ = 0.9039
P(at least one fraud) = 1 − 0.9039 = **0.0961 ≈ 9.6%**

**(b) Dependent case (using chain rule):**
P(fraud on tx 2) = P(fraud on 2 | fraud on 1)·P(fraud on 1) + P(fraud on 2 | no fraud on 1)·P(no fraud on 1)

We need P(fraud on 2 | no fraud on 1). Assume:
P(fraud on k | no fraud on k-1) = 0.02 (base rate)

P(fraud on tx 2) = 0.4 × 0.02 + 0.02 × 0.98
= 0.008 + 0.0196
= **0.0276**

Under independence: P(fraud on tx 2) = 0.02.
Under dependence: 0.0276 — higher because fraud clusters.

**ML lesson:** When i.i.d. fails, standard error estimates are wrong, p-values are invalid, and your model may silently overfit. Always check for autocorrelation in time series, user-level clustering in recommendation systems, etc.

---

### 🔢 Numerical 5 — Berkson's Paradox (Selection Bias in ML)

**Problem:** In the general population:
- P(good coder) = 0.3
- P(good communicator) = 0.4
- These skills are independent: P(both) = 0.12

Your company only interviews candidates who are good at **at least one** skill.

Among interviewed candidates, are coding and communication skills still independent?

**Solution:**

P(interviewed) = P(good coder OR good communicator)
= 0.3 + 0.4 − 0.12 = **0.58**

Among interviewed candidates:
P(good coder | interviewed) = P(good coder ∩ interviewed) / P(interviewed)
= 0.3 / 0.58 = 0.517

P(good communicator | interviewed) = 0.4 / 0.58 = 0.690

P(both | interviewed) = P(both) / P(interviewed) = 0.12 / 0.58 = 0.207

Check independence:
P(coder|int) × P(comm|int) = 0.517 × 0.690 = **0.357**

0.207 ≠ 0.357 → **Dependent among interviewed candidates!**

In fact, within your interviewed pool, knowing someone is a great coder makes it slightly LESS likely they're a great communicator (because if they're already in the pool, they don't need another skill as much).

**ML lesson:** This is Berkson's paradox / collider bias. Conditioning on a collider (the "interviewed" variable, which is caused by both skills) induces spurious negative correlation between its causes. This happens constantly in biased datasets — models trained on such data learn spurious correlations.

---

### 🔢 Numerical 6 — Independence and Variance (Preview of Day 8)

**Problem:** Two features X and Y are independent with:
- E[X] = 2, Var(X) = 4
- E[Y] = 3, Var(Y) = 9

Find Var(X + Y) and Var(X − Y).

**Solution:**

**Key theorem:** If X and Y are independent:
```
Var(X + Y) = Var(X) + Var(Y)
Var(X − Y) = Var(X) + Var(Y)    [subtraction adds variances!]
```

Var(X + Y) = 4 + 9 = **13**
Var(X − Y) = 4 + 9 = **13**

**(If they were NOT independent, you'd need the covariance term — Day 13.)**

**ML lesson:** If features are independent, the variance of a linear combination is just the sum of individual variances. This simplification underpins PCA (when components are orthogonal/independent) and explains why decorrelating features helps many models.

---

## 8. Testing for Independence in Practice

In real ML work, you don't know if features are independent — you test:

| Test | What It Checks |
|---|---|
| Pearson correlation | Linear dependence |
| Spearman correlation | Monotonic dependence |
| Chi-squared test | Independence of categorical variables |
| Mutual information | Any form of dependence (Day 28) |
| VIF (Variance Inflation Factor) | Multicollinearity in regression features |

**Important:** Zero correlation does NOT imply independence (only for Gaussians does it).

---

## 9. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the difference between independent and mutually exclusive?" | ME: can't co-occur. Ind: knowing one doesn't change probability of other. ME with P>0 implies dependent. |
| "What does i.i.d. mean and when does it fail?" | Same distribution + independent. Fails with time series, user clustering, distribution shift. |
| "Can two events be both independent and mutually exclusive?" | Only if at least one has probability 0. |
| "What is conditional independence and why does Naive Bayes use it?" | P(A\|B,C)=P(A\|C). Makes joint likelihood factorizable. |
| "What is Berkson's paradox?" | Conditioning on a collider induces spurious dependence between its causes. |
| "Does zero covariance imply independence?" | No — only for jointly Gaussian random variables. |
| "Why does pairwise independence not imply mutual independence?" | Pairwise checks only pairs; mutual requires all subsets. |

---

## 10. Key Formulas — Cheat Sheet for Day 5

```
Independence (2 events):
    P(A ∩ B) = P(A) · P(B)
    ⟺ P(A|B) = P(A)
    ⟺ P(B|A) = P(B)

Mutual Independence (n events):
    P(Aᵢ₁ ∩ ... ∩ Aᵢₖ) = P(Aᵢ₁)·...·P(Aᵢₖ)  for ALL subsets

Conditional Independence:
    P(A ∩ B | C) = P(A|C) · P(B|C)
    ⟺ P(A|B,C) = P(A|C)

Independence + variance:
    Var(X + Y) = Var(X) + Var(Y)   [if independent]

Series reliability:
    P(system works) = Π P(componentᵢ works)

Parallel reliability:
    P(system works) = 1 - Π P(componentᵢ fails)

i.i.d.:
    X₁,...,Xₙ same distribution AND mutually independent
```

---

## 11. Practice Problems (Solve Before Day 6)

1. Flip two fair coins. Define A = "first coin is H", B = "second coin is H", C = "coins differ". Show that any two of {A,B,C} are pairwise independent, but all three are NOT mutually independent.

2. A neural network has 10 layers, each passing information correctly with probability 0.95, independently. What is the probability the signal survives all 10 layers? What does this tell you about deep network training?

3. From the joint distribution below, determine if X and Y are independent:

   | | Y=0 | Y=1 |
   |---|---|---|
   | X=0 | 0.20 | 0.30 |
   | X=1 | 0.12 | 0.38 |

4. *(Conditional independence)* In a Hidden Markov Model, observations O₁ and O₂ are conditionally independent given the hidden state H. P(O₁=1|H=0)=0.3, P(O₁=1|H=1)=0.8, P(O₂=1|H=0)=0.4, P(O₂=1|H=1)=0.7, P(H=1)=0.6. Find P(O₁=1, O₂=1).

5. *(Interview-level)* Explain in your own words why zero correlation does not imply independence. Construct a simple example where X and Y are uncorrelated but clearly dependent.

---

## 12. Looking Ahead

**Day 6** — **Unit 1 Review + Hard Interview Problems.** We consolidate Days 1–5 with the hardest interview problems on probability foundations: birthday paradox, coupon collector, geometric probability, and 5 real ML interview scenarios that combine everything.

---
*End of Day 5 | Next: Day 6 — Unit 1 Review & Hard Interview Problems*
