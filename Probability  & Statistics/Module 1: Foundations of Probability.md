# Module 1: Foundations of Probability
### Google L5 Data Scientist Interview Prep

---

## 1. Sample Spaces & Events

### Core Definitions

**Experiment**: Any process with an uncertain outcome.
**Sample Space (Ω)**: The set of ALL possible outcomes.
**Event (A)**: Any subset of the sample space.
**Elementary outcome (ω)**: A single outcome in Ω.

### Set Operations on Events

| Operation | Notation | Meaning |
|---|---|---|
| Union | A ∪ B | A or B (or both) occur |
| Intersection | A ∩ B | Both A and B occur |
| Complement | Aᶜ | A does not occur |
| Difference | A \ B | A occurs but not B |

### De Morgan's Laws (memorize these)

```
(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
```

**Intuition**: "Not (A or B)" = "Not A AND Not B"

---

## 2. Kolmogorov Axioms

All of probability theory rests on three axioms:

1. **Non-negativity**: P(A) ≥ 0 for any event A
2. **Normalization**: P(Ω) = 1
3. **Countable additivity**: If A₁, A₂, ... are mutually exclusive (disjoint), then:
   P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...

### Derived Properties

From these three axioms, everything else follows:

- P(∅) = 0
- P(Aᶜ) = 1 − P(A)
- P(A ∪ B) = P(A) + P(B) − P(A ∩ B)   ← **Inclusion-Exclusion**
- If A ⊆ B, then P(A) ≤ P(B)
- 0 ≤ P(A) ≤ 1

---

## 3. Counting & Classical Probability

When all outcomes are equally likely: **P(A) = |A| / |Ω|**

### Counting Rules

| Rule | Formula | When to use |
|---|---|---|
| Multiplication | n₁ × n₂ × ... | Sequential choices |
| Permutation | n! / (n−k)! | Ordered selection, no replacement |
| Combination | n! / (k!(n−k)!) | Unordered selection, no replacement |
| With replacement | nᵏ | k draws, order matters |

### 📌 Example 1: Birthday Problem

**Q**: What's the probability that at least 2 people in a room of 23 share a birthday?

**Approach**: Use complement — P(at least one shared) = 1 − P(all different)

```
P(all different) = (365/365) × (364/365) × (363/365) × ... × (343/365)
                 = 365! / (342! × 365²³)
                 ≈ 0.4927
```

**P(at least one shared) = 1 − 0.4927 ≈ 0.5073**

With only 23 people, it's MORE likely than not that two share a birthday.

**Why this matters for interviews**: Google loves this as a test of whether you instinctively use the complement trick. Always check if P(complement) is easier to compute.

---

## 4. Conditional Probability

### Definition

```
P(A | B) = P(A ∩ B) / P(B),   provided P(B) > 0
```

Read as: "Probability of A given that B has occurred."

**Intuition**: You've restricted your universe from Ω to B. Now what fraction of B contains A?

### Multiplication Rule

```
P(A ∩ B) = P(A | B) × P(B) = P(B | A) × P(A)
```

### Chain Rule (generalization)

```
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) × P(A₂|A₁) × P(A₃|A₁,A₂) × ... × P(Aₙ|A₁,...,Aₙ₋₁)
```

### 📌 Example 2: Card Draw

**Q**: A card is drawn from a standard deck. Given it's a face card, what's the probability it's a King?

- Face cards: {J, Q, K} × 4 suits = 12 cards → P(face card) = 12/52
- Kings: 4 cards → P(King) = 4/52
- P(King | face card) = P(King ∩ face card) / P(face card) = (4/52) / (12/52) = 4/12 = **1/3**

### 📌 Example 3: Sequential Draws (Google-style)

**Q**: A bag has 3 red and 2 blue balls. Two drawn without replacement. P(both red)?

```
P(R₁ ∩ R₂) = P(R₁) × P(R₂ | R₁)
            = (3/5) × (2/4)
            = 6/20
            = 0.30
```

---

## 5. Independence

### Definition

Events A and B are **independent** if:
```
P(A ∩ B) = P(A) × P(B)
```

Equivalently:
```
P(A | B) = P(A)    [knowing B gives no info about A]
P(B | A) = P(B)
```

### ⚠️ Critical Distinction: Independence vs. Mutual Exclusivity

| Property | Definition | Can both hold? |
|---|---|---|
| Independent | P(A∩B) = P(A)P(B) | Only if P(A)=0 or P(B)=0 |
| Mutually exclusive | P(A∩B) = 0 | Yes, trivially if one has prob 0 |

**Mutually exclusive events with positive probability are NEVER independent.**

Why? If A occurs, B definitely cannot — so knowing A changes what we know about B.

### Pairwise vs. Mutual Independence

- **Pairwise independent**: Every pair is independent
- **Mutually independent**: Every subset is independent

Pairwise independence does NOT imply mutual independence.

**Example**: Toss two fair coins. Let A = {H on coin 1}, B = {H on coin 2}, C = {both same}.
- A,B independent ✓; A,C independent ✓; B,C independent ✓
- But P(A∩B∩C) = 1/4 ≠ P(A)P(B)P(C) = 1/8. Not mutually independent.

---

## 6. Law of Total Probability

### Setup: Partition

Events B₁, B₂, ..., Bₙ form a **partition** of Ω if:
- They are mutually exclusive: Bᵢ ∩ Bⱼ = ∅ for i ≠ j
- They are exhaustive: B₁ ∪ B₂ ∪ ... ∪ Bₙ = Ω

### The Law

```
P(A) = Σᵢ P(A | Bᵢ) × P(Bᵢ)
```

**Intuition**: Break a hard probability into cases (the Bᵢ partition), compute P(A) within each case, weight by how likely each case is.

### 📌 Example 4: Email Spam (Google-relevant)

**Q**: 30% of emails are spam. Your spam filter catches spam with 95% accuracy. It also flags 2% of legitimate emails as spam. What fraction of all emails get flagged?

Let S = spam, L = legitimate, F = flagged.

```
P(F) = P(F|S)×P(S) + P(F|L)×P(L)
     = 0.95 × 0.30 + 0.02 × 0.70
     = 0.285 + 0.014
     = 0.299
```

About **29.9% of all emails get flagged**. This is the setup for Bayes' theorem.

---

## 7. Bayes' Theorem

### The Formula

```
P(B | A) = P(A | B) × P(B) / P(A)
```

Expanded using Law of Total Probability in the denominator:

```
P(Bᵢ | A) = P(A | Bᵢ) × P(Bᵢ) / Σⱼ [P(A | Bⱼ) × P(Bⱼ)]
```

### Terminology

| Term | Symbol | Meaning |
|---|---|---|
| Prior | P(B) | Belief about B before seeing evidence |
| Likelihood | P(A\|B) | How probable is evidence A if B is true |
| Marginal | P(A) | Total probability of evidence A |
| Posterior | P(B\|A) | Updated belief after seeing A |

```
Posterior ∝ Likelihood × Prior
```

### 📌 Example 5: Medical Test (Classic)

**Q**: A disease affects 1% of the population. A test has 99% sensitivity (P(+|disease)) and 95% specificity (P(−|no disease)). You test positive. What's the probability you have the disease?

Define:
- D = has disease, P(D) = 0.01
- + = positive test

Given:
- P(+|D) = 0.99 (sensitivity)
- P(+|Dᶜ) = 1 − 0.95 = 0.05 (false positive rate)

Apply Bayes:

```
P(D|+) = P(+|D) × P(D) / P(+)

P(+) = P(+|D)×P(D) + P(+|Dᶜ)×P(Dᶜ)
     = 0.99 × 0.01 + 0.05 × 0.99
     = 0.0099 + 0.0495
     = 0.0594

P(D|+) = 0.0099 / 0.0594 ≈ 0.167
```

**Only ~16.7% chance of actually having the disease despite a positive test.**

**Why?** The disease is rare (1%). False positives (5% of 99% healthy people) vastly outnumber true positives (99% of 1% sick people).

**L5 Insight**: This is why Google cares about base rates in anomaly detection. A model that flags 1% of users as fraudulent in a system where true fraud is 0.01% will have terrible precision.

### 📌 Example 6: A/B Test Interpretation (Google-style)

**Q**: Your experiment shows a statistically significant lift (p < 0.05). You run 20 experiments per quarter. Historically, only 10% of tested features truly improve the metric. What's the probability this result reflects a real improvement?

Let T = truly effective, S = statistically significant result.

- P(T) = 0.10 (prior — only 10% of features actually work)
- P(S|T) = 0.80 (80% power — standard assumption)
- P(S|Tᶜ) = 0.05 (false positive rate = α)

```
P(S) = P(S|T)×P(T) + P(S|Tᶜ)×P(Tᶜ)
     = 0.80 × 0.10 + 0.05 × 0.90
     = 0.08 + 0.045
     = 0.125

P(T|S) = (0.80 × 0.10) / 0.125
        = 0.08 / 0.125
        = 0.64
```

**Only 64% chance the result is real** — even with a significant p-value. This is the **positive predictive value** of your testing process.

**L5 Insight**: This is why multiple testing correction matters and why you shouldn't ship every "significant" result. The prior matters enormously.

---

## 8. Inclusion-Exclusion Principle

### Two Events

```
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
```

### Three Events

```
P(A ∪ B ∪ C) = P(A) + P(B) + P(C)
              − P(A∩B) − P(A∩C) − P(B∩C)
              + P(A∩B∩C)
```

### General Pattern

Add individual probabilities, subtract pairwise, add triples, subtract quadruples...

### 📌 Example 7: User Engagement

**Q**: 60% of users click Search, 45% click Ads, 30% click both. What fraction click at least one?

```
P(S ∪ A) = P(S) + P(A) − P(S ∩ A)
          = 0.60 + 0.45 − 0.30
          = 0.75
```

75% of users click at least one feature.

---

## 9. Common Interview Traps & Pitfalls

### Trap 1: Confusing P(A|B) and P(B|A)
The **Prosecutor's Fallacy**: P(evidence|innocent) is NOT P(innocent|evidence). Always use Bayes to flip conditioning.

### Trap 2: Independence Assumption
If the problem says "independently chosen" — use multiplication. Don't assume independence without being told.

### Trap 3: Forgetting Base Rates
Rare events create counterintuitive posteriors (see medical test example). Always ask: what's the prior?

### Trap 4: Complement Shortcut
When computing P(at least one ...), use 1 − P(none). Almost always easier.

### Trap 5: Conditional ≠ Causal
P(B|A) > P(B) means A and B are associated, not that A causes B.

### Trap 6: Mutually Exclusive ≠ Independent
This is a common confusion. If A and B can't both happen, knowing A tells you B can't happen — they are maximally dependent.

---

## Q&A Section

### Q1 (Warm-up)
**Q**: A fair die is rolled twice. What is P(sum = 7)?

**A**: Sample space has 6² = 36 equally likely outcomes. Favorable: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) = 6 outcomes.
P(sum=7) = 6/36 = **1/6**.

---

### Q2 (Conditional)
**Q**: From a standard deck, two cards drawn without replacement. P(second is Ace | first is Ace)?

**A**: Given first is Ace, 51 cards remain, 3 are Aces.
P(2nd Ace | 1st Ace) = **3/51 = 1/17 ≈ 0.059**

---

### Q3 (Bayes — Google-style)
**Q**: A/B test: feature rolled to 50% of users. 5% of control users convert, 7% of treatment users convert. A randomly selected converter is found. What's the probability they were in treatment?

**A**: Let T = treatment, C = convert.

```
P(T|C) = P(C|T)×P(T) / P(C)

P(C) = P(C|T)×P(T) + P(C|control)×P(control)
     = 0.07×0.5 + 0.05×0.5
     = 0.035 + 0.025 = 0.06

P(T|C) = 0.035 / 0.06 ≈ 0.583
```

**~58.3%** of converters were in treatment.

---

### Q4 (Independence Check)
**Q**: Events A and B have P(A) = 0.4, P(B) = 0.3, P(A∩B) = 0.12. Are they independent?

**A**: Check: P(A)×P(B) = 0.4×0.3 = 0.12 = P(A∩B). **Yes, they are independent.**

---

### Q5 (Harder — L5 Level)
**Q**: You're analyzing Google Search. 70% of queries are informational (I), 30% are navigational (N). Click-through rate: 80% for I, 60% for N. A query results in a click. What's the probability it was informational?

**A**:
```
P(I|click) = P(click|I)×P(I) / P(click)

P(click) = 0.80×0.70 + 0.60×0.30
          = 0.56 + 0.18 = 0.74

P(I|click) = 0.56 / 0.74 ≈ 0.757
```

**75.7%** of clicked queries were informational.

---

### Q6 (Trap Question)
**Q**: P(A|B) = 0.9 and P(B) = 0.01. Is P(B|A) also high?

**A**: Not necessarily. Apply Bayes:

```
P(B|A) = P(A|B)×P(B) / P(A)
```

If P(A) = 0.10:
```
P(B|A) = 0.9 × 0.01 / 0.10 = 0.09
```

**P(B|A) = 9%** — very low, despite P(A|B) = 90%. This is the prosecutor's fallacy. The rarity of B (1%) dominates.

---

## Cheat Sheet: Module 1

```
┌─────────────────────────────────────────────────────────────────┐
│                  PROBABILITY FOUNDATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│  AXIOMS                                                          │
│  P(A) ≥ 0                                                        │
│  P(Ω) = 1                                                        │
│  Disjoint: P(A∪B) = P(A) + P(B)                                 │
│                                                                   │
│  KEY FORMULAS                                                     │
│  Complement:        P(Aᶜ) = 1 − P(A)                            │
│  Inclusion-Excl:    P(A∪B) = P(A)+P(B)−P(A∩B)                  │
│  Conditional:       P(A|B) = P(A∩B) / P(B)                      │
│  Multiplication:    P(A∩B) = P(A|B)·P(B)                        │
│  Independence:      P(A∩B) = P(A)·P(B)                          │
│  Total Prob:        P(A) = Σ P(A|Bᵢ)·P(Bᵢ)                     │
│  Bayes:             P(B|A) = P(A|B)·P(B) / P(A)                 │
│                                                                   │
│  BAYES COMPONENTS                                                 │
│  Prior       → P(B)      what you believed before                │
│  Likelihood  → P(A|B)    how well B explains A                   │
│  Marginal    → P(A)      total probability of evidence           │
│  Posterior   → P(B|A)    updated belief                          │
│                                                                   │
│  TRICKS                                                           │
│  "At least one" → use complement: 1 − P(none)                   │
│  Flip P(A|B)?   → use Bayes                                      │
│  Sequential?    → use chain rule / multiplication rule           │
│  Partition?     → use law of total probability                   │
│                                                                   │
│  TRAPS                                                            │
│  ✗ P(A|B) ≠ P(B|A)         Prosecutor's fallacy                 │
│  ✗ Exclusive ≠ Independent  They are opposites                   │
│  ✗ Ignore base rates        Rare events crush posteriors         │
│  ✗ Assume independence      Must be stated or proven             │
└─────────────────────────────────────────────────────────────────┘
```

---

*Next → Module 2: Random Variables & Distributions*
