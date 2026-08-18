# Day 3 — Conditional Probability & The Chain Rule
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 2 (Sections 2.1–2.4)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Conditional Probability is the Heart of ML

Almost every ML problem is a conditional probability problem.

| ML Task | Conditional Probability |
|---|---|
| Classification | P(class = cat \| image pixels) |
| Language model | P(next word \| previous words) |
| Spam filter | P(spam \| email content) |
| Medical diagnosis | P(disease \| symptoms) |
| Recommendation | P(user clicks \| user history, item features) |
| Reinforcement learning | P(reward \| state, action) |

The moment you condition on something — you've updated your belief given new information. That's machine learning.

---

## 2. Definition of Conditional Probability

> **Definition:** The conditional probability of event A given event B has occurred is:

```
         P(A ∩ B)
P(A|B) = ————————       provided P(B) > 0
           P(B)
```

**Reading:** "Probability of A given B"

### Intuition — Shrinking the Sample Space

When we learn B has occurred, we **restrict** our universe to B. Within that restricted world, what fraction belongs to A?

```
Original universe: Ω
After learning B:  new universe = B
P(A|B) = (size of A within B) / (size of B)
        = P(A ∩ B) / P(B)
```

Think of it as zooming in: B becomes the new Ω, and we renormalize.

### Visual

```
┌──────────────────────────┐
│           Ω              │
│    ┌──────────┐          │
│    │    B     │          │
│    │  ┌───┐  │          │
│    │  │A∩B│  │          │
│    │  └───┘  │          │
│    └──────────┘          │
└──────────────────────────┘

P(A|B) = area(A∩B) / area(B)
```

---

## 3. Properties of Conditional Probability

Given a fixed B with P(B) > 0, the function P(·|B) is itself a valid probability measure — it satisfies all three Kolmogorov axioms:

```
1. P(A|B) ≥ 0                          [non-negativity]
2. P(Ω|B) = 1                          [normalization]
3. If A₁ ∩ A₂ = ∅, then
   P(A₁ ∪ A₂ | B) = P(A₁|B) + P(A₂|B) [additivity]
```

Also:
```
P(Aᶜ|B) = 1 - P(A|B)
P(A|B) + P(Aᶜ|B) = 1
```

**Important:** P(A|B) ≠ P(B|A) in general. Confusing these is called the **Prosecutor's Fallacy** — a critical concept for data science interviews.

---

## 4. The Multiplication Rule (Chain Rule — 2 Events)

Rearranging the definition:

```
P(A ∩ B) = P(A|B) · P(B)
```

Also equivalently:
```
P(A ∩ B) = P(B|A) · P(A)
```

**Reading:** "The probability that both A and B occur = probability B occurs × probability A occurs given B."

This is the **two-event chain rule**. It's the product rule and it's everywhere.

---

## 5. The Chain Rule (General — n Events)

Extend the multiplication rule to n events:

```
P(A₁ ∩ A₂ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)
```

**Formula breakdown — term by term:**
- P(A₁) — probability of first event, no conditions
- P(A₂|A₁) — probability of second, given first happened
- P(A₃|A₁,A₂) — probability of third, given first two happened
- ...each term conditions on everything before it

### Chain Rule in NLP / Language Models

A language model computes:
```
P(w₁, w₂, w₃, ..., wₙ) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(wₙ|w₁,...,wₙ₋₁)
```

This is **exactly** the chain rule. GPT, LLaMA, every autoregressive language model factorizes sentence probability this way. The chain rule is literally what makes language modeling possible.

---

## 6. The Law of Total Probability (LOTP)

> **Setup:** Partition the sample space Ω into mutually exclusive, exhaustive events B₁, B₂, ..., Bₙ
> (meaning: Bᵢ ∩ Bⱼ = ∅ for i≠j, and B₁ ∪ B₂ ∪ ... ∪ Bₙ = Ω)

```
P(A) = P(A|B₁)·P(B₁) + P(A|B₂)·P(B₂) + ... + P(A|Bₙ)·P(Bₙ)

     = Σᵢ P(A|Bᵢ) · P(Bᵢ)
```

**Intuition:** To find P(A), break the world into exhaustive cases B₁,...,Bₙ. Compute P(A) within each case, weight by how likely each case is, sum up.

### Simple 2-partition version (most common in interviews):

```
P(A) = P(A|B)·P(B) + P(A|Bᶜ)·P(Bᶜ)
```

**ML use:** Computing overall model error across subgroups, marginalizing over latent variables, computing P(output) in mixture models.

---

## 7. Common Mistakes & The Prosecutor's Fallacy

### Mistake 1: P(A|B) = P(B|A)

These are NOT the same:
- P(test positive | has disease) = sensitivity of test
- P(has disease | test positive) = what the patient actually wants to know

Confusing these is **the prosecutor's fallacy**:
- "The probability of this DNA match if the suspect is innocent is 1 in a million"
- Does NOT mean "probability suspect is innocent given match is 1 in a million"

### Mistake 2: Conditioning on zero-probability events

P(A|B) requires P(B) > 0. Conditioning on impossible events is undefined.

### Mistake 3: Forgetting to renormalize

When you condition on B, all probabilities must be reweighted so they sum to 1 within B.

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — Basic Conditional Probability

**Problem:** In a dataset of 1000 emails:
- 300 are spam
- Of the spam emails, 270 contain the word "free"
- Of the non-spam emails, 70 contain the word "free"

Find:
- (a) P(spam)
- (b) P(contains "free" | spam)
- (c) P(spam ∩ contains "free")
- (d) P(contains "free")

**Solution:**

**(a)** P(spam) = 300/1000 = **0.30**

**(b)** P("free" | spam) = 270/300 = **0.90**

**(c)** Using multiplication rule:  
P(spam ∩ "free") = P("free"|spam) · P(spam) = 0.90 × 0.30 = **0.27**

Verify: 270/1000 = 0.27 ✓

**(d)** Using LOTP:  
P("free") = P("free"|spam)·P(spam) + P("free"|not spam)·P(not spam)  
= 0.90 × 0.30 + (70/700) × 0.70  
= 0.27 + 0.10 × 0.70  
= 0.27 + 0.07  
= **0.34**

---

### 🔢 Numerical 2 — Chain Rule for Joint Probability

**Problem:** In a dataset, the following probabilities are known:
- P(feature A is high) = 0.4
- P(feature B is high | feature A is high) = 0.6
- P(label = 1 | feature A high, feature B high) = 0.8

Find P(feature A high ∩ feature B high ∩ label = 1).

**Solution:**

Using the chain rule:
P(A ∩ B ∩ L) = P(A) · P(B|A) · P(L|A,B)  
= 0.4 × 0.6 × 0.8  
= **0.192**

About 19.2% of data points have both features high and label = 1.

---

### 🔢 Numerical 3 — Law of Total Probability

**Problem:** A company uses two ML models:
- Model 1 (legacy): used 60% of the time, error rate 15%
- Model 2 (new): used 40% of the time, error rate 5%

What is the overall error rate of the system?

**Solution:**

Let E = "error occurs", M1 = "Model 1 is used", M2 = "Model 2 is used"

{M1, M2} partition the space (mutually exclusive, exhaustive).

P(E) = P(E|M1)·P(M1) + P(E|M2)·P(M2)  
= 0.15 × 0.60 + 0.05 × 0.40  
= 0.090 + 0.020  
= **0.11**

Overall error rate = **11%**

**ML insight:** This is how you compute weighted average error across model ensembles or A/B test groups.

---

### 🔢 Numerical 4 — The Prosecutor's Fallacy (Interview Classic)

**Problem:** A disease affects 1% of the population. A diagnostic test has:
- Sensitivity: P(test+ | disease) = 0.95
- Specificity: P(test− | no disease) = 0.90

A patient tests positive. What is the probability they actually have the disease?

*Note: This is Bayes' Theorem — but let's solve it using conditional probability + LOTP directly.*

**Solution:**

Let D = "has disease", T = "tests positive"

Known:
- P(D) = 0.01
- P(Dᶜ) = 0.99
- P(T|D) = 0.95
- P(T|Dᶜ) = 1 - 0.90 = 0.10 (false positive rate)

Step 1 — Find P(T) using LOTP:  
P(T) = P(T|D)·P(D) + P(T|Dᶜ)·P(Dᶜ)  
= 0.95 × 0.01 + 0.10 × 0.99  
= 0.0095 + 0.099  
= **0.1085**

Step 2 — Find P(D|T):  
P(D|T) = P(D ∩ T) / P(T)  
= [P(T|D) · P(D)] / P(T)  
= 0.0095 / 0.1085  
= **≈ 0.0876 = 8.76%**

**Shocking result:** Even with a 95% accurate test, a positive result only means ~8.76% chance of disease!

**Why?** Because the disease is rare (1%). Most positive tests come from the large pool of healthy people (99%) who have a 10% false positive rate. This is the **base rate fallacy** — ignoring the prior probability.

**ML interview lesson:** Precision matters more than accuracy for rare-event classification. A model that always predicts "no disease" gets 99% accuracy but is useless.

---

### 🔢 Numerical 5 — Chain Rule in Language Modeling

**Problem:** A bigram language model estimates:
- P(The) = 0.08
- P(cat | The) = 0.05
- P(sat | The, cat) ≈ P(sat | cat) = 0.12 *(bigram approximation)*
- P(on | cat, sat) ≈ P(on | sat) = 0.09

Estimate P("The cat sat on") using the chain rule.

**Solution:**

P("The cat sat on") = P(The) · P(cat|The) · P(sat|cat) · P(on|sat)  
= 0.08 × 0.05 × 0.12 × 0.09  
= **0.0000432**

**Key insight:** Sentence probabilities are tiny — that's why language models work in **log space**:

log P = log(0.08) + log(0.05) + log(0.12) + log(0.09)  
= −2.526 + (−2.996) + (−2.120) + (−2.408)  
= **−10.05**

This is where **log-likelihood** and **cross-entropy loss** come from — they're the chain rule applied in log space.

---

### 🔢 Numerical 6 — Conditional Probability Table (ML Interview Format)

**Problem:** A fraud detection model produces the following joint distribution over prediction (P̂) and true label (Y):

|  | Y=0 (legit) | Y=1 (fraud) |
|---|---|---|
| **P̂=0 (predict legit)** | 0.70 | 0.05 |
| **P̂=1 (predict fraud)** | 0.10 | 0.15 |

Find:
- (a) P(Y=1) — prevalence of fraud
- (b) P(P̂=1 | Y=1) — recall / sensitivity
- (c) P(Y=1 | P̂=1) — precision
- (d) P(P̂=1 | Y=0) — false positive rate

**Solution:**

**(a)** P(Y=1) = P(P̂=0, Y=1) + P(P̂=1, Y=1) = 0.05 + 0.15 = **0.20**

**(b)** P(P̂=1 | Y=1) = P(P̂=1 ∩ Y=1) / P(Y=1) = 0.15 / 0.20 = **0.75** (75% recall)

**(c)** P(P̂=1) = 0.10 + 0.15 = 0.25  
P(Y=1 | P̂=1) = P(P̂=1 ∩ Y=1) / P(P̂=1) = 0.15 / 0.25 = **0.60** (60% precision)

**(d)** P(Y=0) = 0.70 + 0.10 = 0.80  
P(P̂=1 | Y=0) = P(P̂=1 ∩ Y=0) / P(Y=0) = 0.10 / 0.80 = **0.125** (12.5% FPR)

**ML punchline:** Precision and recall are conditional probabilities. Every metric in a confusion matrix is a conditional probability. This is not a coincidence — it's the definition.

---

### 🔢 Numerical 7 — LOTP for Model Ensembles

**Problem:** You ensemble 3 models. For any given input:
- Model A is selected 50% of the time, accuracy 90%
- Model B is selected 30% of the time, accuracy 80%
- Model C is selected 20% of the time, accuracy 70%

(a) What is the overall accuracy?  
(b) Given a mistake was made, what is the probability it came from Model C?

**Solution:**

**(a)** Let Correct = "correct prediction"

P(Correct) = P(C|A)P(A) + P(C|B)P(B) + P(C|C_model)P(C_model)  
= 0.90×0.50 + 0.80×0.30 + 0.70×0.20  
= 0.45 + 0.24 + 0.14  
= **0.83**

**(b)** P(error) = 1 - 0.83 = 0.17

P(error ∩ Model C) = P(error|Model C) · P(Model C) = 0.30 × 0.20 = 0.06

P(Model C | error) = 0.06 / 0.17 = **0.353 ≈ 35.3%**

Model C, despite being used only 20% of the time, causes 35.3% of errors — it's the weakest link.

---

## 9. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the difference between P(A\|B) and P(B\|A)?" | Not equal in general; confusing them = prosecutor's fallacy |
| "Derive the chain rule for 3 events" | P(A∩B∩C) = P(A)·P(B\|A)·P(C\|A,B) |
| "What is the Law of Total Probability?" | Marginalize over exhaustive partition |
| "How does a language model compute sentence probability?" | Chain rule: product of conditional word probabilities |
| "What is precision in terms of conditional probability?" | P(Y=1 \| Ŷ=1) |
| "What is recall in terms of conditional probability?" | P(Ŷ=1 \| Y=1) |
| "Why does a 99% accurate test have low PPV for rare diseases?" | Base rate / prior dominates — LOTP shows it |

---

## 10. Key Formulas — Cheat Sheet for Day 3

```
Conditional Probability:
    P(A|B) = P(A ∩ B) / P(B)        [P(B) > 0]

Multiplication Rule (2 events):
    P(A ∩ B) = P(A|B) · P(B)
             = P(B|A) · P(A)

Chain Rule (n events):
    P(A₁ ∩ ... ∩ Aₙ) = P(A₁) · P(A₂|A₁) · P(A₃|A₁,A₂) · ... · P(Aₙ|A₁,...,Aₙ₋₁)

Law of Total Probability:
    P(A) = Σᵢ P(A|Bᵢ) · P(Bᵢ)      [B₁,...,Bₙ partition Ω]

Two-partition LOTP:
    P(A) = P(A|B)·P(B) + P(A|Bᶜ)·P(Bᶜ)

Complement under conditioning:
    P(Aᶜ|B) = 1 - P(A|B)

ML metrics as conditional probabilities:
    Precision  = P(Y=1 | Ŷ=1)
    Recall     = P(Ŷ=1 | Y=1)
    FPR        = P(Ŷ=1 | Y=0)
    Specificity = P(Ŷ=0 | Y=0)
```

---

## 11. Practice Problems (Solve Before Day 4)

1. In a dataset: 40% of samples are class A, 60% are class B. Your model has P(predict A | true A) = 0.85 and P(predict A | true B) = 0.20. What is P(model predicts A)?

2. **Prove** using the definition of conditional probability that:  
P(A ∩ B ∩ C) = P(A) · P(B|A) · P(C|A,B)

3. A data pipeline has 3 stages. Each stage fails independently with probability 0.05. Given the pipeline failed, what is the probability it failed at stage 1?  
*(Hint: use LOTP + multiplication rule)*

4. *(Interview-level)* A classifier achieves 95% accuracy on class 0 (90% of data) and 60% accuracy on class 1 (10% of data). What is the overall accuracy? What does this tell you about accuracy as a metric for imbalanced datasets?

5. *(Hard)* A trigram language model estimates P("I love coffee") using:  
P(I) = 0.10, P(love|I) = 0.03, P(coffee|I, love) = 0.15  
What is P("I love coffee")? What is the log probability? Why do language models use log probabilities?

---

## 12. Looking Ahead

**Day 4** — **Bayes' Theorem**: The most important formula in ML. We already used it in Numerical 4 today. Tomorrow we formalize it, prove it, and apply it to Naive Bayes classifiers, posterior inference, and the famous Monty Hall problem.

---
*End of Day 3 | Next: Day 4 — Bayes' Theorem, The Engine of ML*
