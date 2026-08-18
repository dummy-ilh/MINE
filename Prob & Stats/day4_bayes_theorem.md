# Day 4 — Bayes' Theorem: The Engine of ML
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 2 (Section 2.3)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Bayes' Theorem is the Most Important Formula in ML

If you had to pick **one** formula that underlies modern machine learning, it's Bayes' theorem.

| ML Application | How Bayes' Theorem Appears |
|---|---|
| Naive Bayes classifier | Directly — P(class \| features) ∝ P(features \| class) · P(class) |
| Bayesian neural networks | Posterior over weights: P(w \| data) |
| Spam filtering | P(spam \| words) |
| Medical AI | P(disease \| symptoms, tests) |
| Bayesian optimization | Posterior over objective function |
| Probabilistic graphical models | Entire framework is Bayesian |
| A/B testing (Bayesian) | Update belief about conversion rate |

Even frequentist ML methods have Bayesian interpretations. Regularization (L2) = Gaussian prior. MAP estimation = regularized MLE. You cannot escape Bayes.

---

## 2. Derivation of Bayes' Theorem

Starting from the definition of conditional probability (Day 3):

```
P(A|B) = P(A ∩ B) / P(B)      ...(1)
P(B|A) = P(A ∩ B) / P(A)      ...(2)
```

From (2): P(A ∩ B) = P(B|A) · P(A)

Substitute into (1):

```
┌─────────────────────────────────────┐
│                                     │
│         P(B|A) · P(A)              │
│  P(A|B) = ————————————             │
│               P(B)                  │
│                                     │
└─────────────────────────────────────┘
```

That's it. That's Bayes' theorem. Four lines of algebra. But the implications are profound.

---

## 3. The Full Form — With Law of Total Probability

Usually we don't know P(B) directly. We expand it using LOTP:

```
                P(B|A) · P(A)
P(A|B) = ————————————————————————————————
          P(B|A)·P(A) + P(B|Aᶜ)·P(Aᶜ)
```

**General form** with partition {A₁, A₂, ..., Aₙ}:

```
              P(B|Aᵢ) · P(Aᵢ)
P(Aᵢ|B) = ————————————————————————
            Σⱼ P(B|Aⱼ) · P(Aⱼ)
```

---

## 4. The Bayesian Vocabulary — Critical for Interviews

Every term has a name. Interviewers use these terms — know them cold.

```
         P(B|A) · P(A)
P(A|B) = ———————————————
               P(B)
```

| Term | Formula | Meaning |
|---|---|---|
| **Prior** | P(A) | Belief about A *before* seeing evidence B |
| **Likelihood** | P(B\|A) | How probable is evidence B if A is true |
| **Marginal / Evidence** | P(B) | Total probability of evidence (normalizing constant) |
| **Posterior** | P(A\|B) | Updated belief about A *after* seeing evidence B |

**The Bayesian update cycle:**
```
Prior → [observe evidence B] → Posterior
```

In Bayesian ML, today's posterior becomes tomorrow's prior as new data arrives. This is **online/sequential learning**.

### The Proportionality Form (Used Constantly in ML)

Since P(B) is just a normalizing constant (doesn't depend on A):

```
P(A|B) ∝ P(B|A) · P(A)

Posterior ∝ Likelihood × Prior
```

This is how Naive Bayes, MAP estimation, and most Bayesian inference works — compute the unnormalized posterior, then normalize.

---

## 5. Bayes' Theorem for Multiple Hypotheses

In classification with K classes:

```
P(Cₖ | x) = P(x | Cₖ) · P(Cₖ) / P(x)

where P(x) = Σₖ P(x|Cₖ) · P(Cₖ)
```

The predicted class is:

```
ĉ = argmax_k P(Cₖ|x) = argmax_k P(x|Cₖ) · P(Cₖ)
```

*(The argmax drops P(x) since it's the same for all classes.)*

This is the **Maximum A Posteriori (MAP)** decision rule — the backbone of Naive Bayes.

---

## 6. Naive Bayes Classifier — Bayes in Action

**Setup:** Features x = (x₁, x₂, ..., xₙ), classes C₁,...,Cₖ

**Naive assumption:** Features are conditionally independent given the class:

```
P(x₁, x₂, ..., xₙ | Cₖ) = P(x₁|Cₖ) · P(x₂|Cₖ) · ... · P(xₙ|Cₖ)
```

**Classifier:**
```
ĉ = argmax_k P(Cₖ) · Π_{i=1}^{n} P(xᵢ|Cₖ)
```

In log space (to avoid underflow):
```
ĉ = argmax_k [log P(Cₖ) + Σᵢ log P(xᵢ|Cₖ)]
```

**Why "naive"?** The independence assumption is almost never true in reality — but the classifier works surprisingly well anyway. It's naive, not wrong.

---

## 7. Worked Numericals

---

### 🔢 Numerical 1 — Classic Disease Test (Full Bayes)

**Problem:** (Building on Day 3, Numerical 4 — now with full Bayes framework)

- Disease prevalence: P(D) = 0.01
- Test sensitivity: P(T⁺|D) = 0.95
- Test specificity: P(T⁻|Dᶜ) = 0.90, so P(T⁺|Dᶜ) = 0.10

Find P(D|T⁺) — probability of disease given positive test.

**Solution:**

Identify the Bayesian terms:
- Prior: P(D) = 0.01
- Likelihood: P(T⁺|D) = 0.95
- Evidence: P(T⁺) = ? (compute via LOTP)

P(T⁺) = P(T⁺|D)·P(D) + P(T⁺|Dᶜ)·P(Dᶜ)
= 0.95 × 0.01 + 0.10 × 0.99
= 0.0095 + 0.0990
= 0.1085

Apply Bayes:
```
P(D|T⁺) = P(T⁺|D) · P(D) / P(T⁺)
         = 0.0095 / 0.1085
         = 0.0876 ≈ 8.76%
```

**Now: what if we test again and get a second positive?**

The posterior from test 1 becomes the prior for test 2:
- New prior: P(D) = 0.0876
- P(T⁺) = 0.95 × 0.0876 + 0.10 × 0.9124
         = 0.0832 + 0.0912 = 0.1744

```
P(D | T⁺, T⁺) = 0.0832 / 0.1744 ≈ 0.477 ≈ 47.7%
```

Two positive tests → probability jumps from 8.76% to 47.7%. A third positive test would push it above 90%. This is **sequential Bayesian updating** — the foundation of Bayesian online learning.

---

### 🔢 Numerical 2 — Naive Bayes Spam Classifier

**Problem:** Train a Naive Bayes spam classifier on this dataset:

| Email | "free" | "meeting" | Label |
|---|---|---|---|
| 1 | Yes | No | Spam |
| 2 | Yes | Yes | Spam |
| 3 | No | No | Spam |
| 4 | No | Yes | Not spam |
| 5 | No | Yes | Not spam |
| 6 | Yes | No | Not spam |

**Classify a new email:** contains "free" = Yes, "meeting" = No

**Solution:**

**Step 1 — Compute priors:**
P(spam) = 3/6 = 0.5
P(not spam) = 3/6 = 0.5

**Step 2 — Compute likelihoods (with Laplace smoothing +1 to avoid zero):**

For "free" = Yes:
- P("free"=Yes | spam) = (2+1)/(3+2) = 3/5 = 0.60
- P("free"=Yes | not spam) = (1+1)/(3+2) = 2/5 = 0.40

For "meeting" = No:
- P("meeting"=No | spam) = (2+1)/(3+2) = 3/5 = 0.60
- P("meeting"=No | not spam) = (1+1)/(3+2) = 2/5 = 0.40

**Step 3 — Compute unnormalized posteriors:**

P(spam | x) ∝ P(spam) · P("free"=Y|spam) · P("meeting"=N|spam)
= 0.5 × 0.60 × 0.60 = **0.180**

P(not spam | x) ∝ P(not spam) · P("free"=Y|¬spam) · P("meeting"=N|¬spam)
= 0.5 × 0.40 × 0.40 = **0.080**

**Step 4 — Normalize:**

Total = 0.180 + 0.080 = 0.260

P(spam | x) = 0.180 / 0.260 = **0.692 → Classified as SPAM** ✓

---

### 🔢 Numerical 3 — The Monty Hall Problem

**Problem:** Three doors. Behind one is a car, behind two are goats. You pick Door 1. The host (who knows what's behind each door) opens Door 3, revealing a goat. Should you switch to Door 2?

**Solution using Bayes:**

Let Cᵢ = "car is behind door i"
Let H₃ = "host opens door 3"

Prior: P(C₁) = P(C₂) = P(C₃) = 1/3

Likelihoods — P(H₃ | Cᵢ):
- P(H₃|C₁) = 1/2 (car behind door 1, host can open door 2 or 3 — picks randomly)
- P(H₃|C₂) = 1 (car behind door 2, host MUST open door 3)
- P(H₃|C₃) = 0 (car behind door 3, host can't open door 3)

Evidence: P(H₃) = P(H₃|C₁)·P(C₁) + P(H₃|C₂)·P(C₂) + P(H₃|C₃)·P(C₃)
= (1/2)(1/3) + (1)(1/3) + (0)(1/3)
= 1/6 + 1/3 + 0 = **1/2**

Posteriors:

```
P(C₁|H₃) = P(H₃|C₁)·P(C₁) / P(H₃) = (1/2)(1/3) / (1/2) = 1/3

P(C₂|H₃) = P(H₃|C₂)·P(C₂) / P(H₃) = (1)(1/3)  / (1/2) = 2/3

P(C₃|H₃) = 0
```

**Conclusion:** After the host opens Door 3:
- P(car behind Door 1) = **1/3** — your original door
- P(car behind Door 2) = **2/3** — the other door

**ALWAYS SWITCH.** Switching doubles your probability of winning.

**Why?** The host's action provides information. The host is forced to open a goat door — this asymmetrically updates our belief about Door 2. Bayes captures this exactly.

---

### 🔢 Numerical 4 — Bayesian A/B Testing

**Problem:** You're testing two versions of a recommendation model:
- Model A: historically 30% click-through rate
- Model B: new model, unknown CTR θ

After testing Model B on 10 users, 7 click.

Using a uniform prior on θ (no prior belief about Model B's CTR), what is the posterior probability that θ > 0.5?

**Solution:**

With uniform prior P(θ) = 1 on [0,1] and 7 successes in 10 trials:

The posterior is a **Beta distribution**: θ|data ~ Beta(7+1, 3+1) = Beta(8, 4)

*(Full derivation on Day 24 — for now, trust the formula)*

P(θ > 0.5) = P(Beta(8,4) > 0.5)

The mean of Beta(8,4) = 8/(8+4) = 8/12 = **0.667**

Using the CDF of Beta(8,4):
P(θ > 0.5) ≈ **0.855**

There's an 85.5% posterior probability that Model B's true CTR exceeds 50% — much better than Model A's 30%. Ship it (with more testing).

**ML lesson:** Bayesian A/B testing gives you a probability that one model is better, not just a p-value. It's more interpretable for business decisions.

---

### 🔢 Numerical 5 — Bayesian Document Classification (NLP)

**Problem:** A text classifier has 2 classes: Tech and Sports.

From training data:
- P(Tech) = 0.6, P(Sports) = 0.4
- P("python" | Tech) = 0.3,   P("python" | Sports) = 0.01
- P("training" | Tech) = 0.2, P("training" | Sports) = 0.25
- P("model" | Tech) = 0.25,   P("model" | Sports) = 0.05

Classify the document: *"python training model"*

**Solution:**

Using Naive Bayes (log form to avoid underflow):

**log P(Tech | doc) ∝** log P(Tech) + log P("python"|Tech) + log P("training"|Tech) + log P("model"|Tech)
= log(0.6) + log(0.3) + log(0.2) + log(0.25)
= −0.511 + (−1.204) + (−1.609) + (−1.386)
= **−4.710**

**log P(Sports | doc) ∝** log P(Sports) + log P("python"|Sports) + log P("training"|Sports) + log P("model"|Sports)
= log(0.4) + log(0.01) + log(0.25) + log(0.05)
= −0.916 + (−4.605) + (−1.386) + (−2.996)
= **−9.903**

Since −4.710 > −9.903:

**Classification: TECH** ✓

Unnormalized probabilities:
- e^(−4.710) = 0.00899 → after normalization: **99.7% Tech**
- e^(−9.903) = 0.0000500 → **0.3% Sports**

The word "python" is the decisive signal — highly specific to Tech.

---

### 🔢 Numerical 6 — Prior Sensitivity Analysis

**Problem:** Same disease test as Numerical 1 (sensitivity 95%, specificity 90%). Compute P(D|T⁺) for different disease prevalences:

| Prevalence P(D) | P(D\|T⁺) |
|---|---|
| 0.001 (1 in 1000) | ? |
| 0.01 (1 in 100) | 8.76% (computed) |
| 0.10 (1 in 10) | ? |
| 0.50 (1 in 2) | ? |

**Solution using Bayes:**

Formula: P(D|T⁺) = 0.95·P(D) / [0.95·P(D) + 0.10·(1-P(D))]

**P(D) = 0.001:**
= 0.00095 / (0.00095 + 0.0999) = 0.00095 / 0.10085 = **0.94%**

**P(D) = 0.10:**
= 0.095 / (0.095 + 0.090) = 0.095 / 0.185 = **51.4%**

**P(D) = 0.50:**
= 0.475 / (0.475 + 0.050) = 0.475 / 0.525 = **90.5%**

**Summary table:**

| Prevalence P(D) | P(D\|T⁺) |
|---|---|
| 0.1% | 0.94% |
| 1% | 8.76% |
| 10% | 51.4% |
| 50% | 90.5% |

**ML lesson:** The prior (prevalence) dominates when it's extreme. This is why:
- Class imbalance destroys model reliability
- Accuracy is a misleading metric for rare events
- You should always ask "what's the base rate?" before trusting a model's output

---

## 8. The Bayesian vs Frequentist View in ML

| | Frequentist | Bayesian |
|---|---|---|
| Parameters | Fixed, unknown constants | Random variables with distributions |
| Probability | Long-run frequency | Degree of belief |
| Inference | MLE — find parameters that maximize likelihood | MAP/posterior — update prior with data |
| Regularization | Ad-hoc penalty | L2 = Gaussian prior; L1 = Laplace prior |
| Confidence interval | "95% of such intervals contain true param" | "95% probability param is in this interval" (credible interval) |
| Prediction | Point estimate | Full posterior predictive distribution |

In ML interviews, knowing both views and their connection is a strong signal.

---

## 9. Common Interview Questions

| Question | Key Idea |
|---|---|
| "State Bayes' theorem and name each term" | Prior, likelihood, evidence, posterior |
| "What is the Naive Bayes assumption?" | Conditional independence of features given class |
| "Why is the Monty Hall answer 2/3 not 1/2?" | Host's action is informative — Bayes updates correctly |
| "What is MAP estimation?" | argmax of posterior = argmax of likelihood × prior |
| "How does L2 regularization relate to Bayes?" | L2 penalty = Gaussian prior on weights |
| "Why does Naive Bayes work well despite wrong independence assumption?" | Decision boundary only needs correct ranking of posteriors, not calibrated values |
| "What is a conjugate prior?" | Prior and posterior have the same distributional form (e.g., Beta-Binomial) |

---

## 10. Key Formulas — Cheat Sheet for Day 4

```
Bayes' Theorem:
    P(A|B) = P(B|A) · P(A) / P(B)

Full form:
    P(A|B) = P(B|A)·P(A) / [P(B|A)·P(A) + P(B|Aᶜ)·P(Aᶜ)]

Proportionality:
    P(A|B) ∝ P(B|A) · P(A)
    Posterior ∝ Likelihood × Prior

Bayesian vocabulary:
    Prior      = P(A)        [before evidence]
    Likelihood = P(B|A)      [how well A explains B]
    Evidence   = P(B)        [normalizing constant]
    Posterior  = P(A|B)      [after evidence]

Naive Bayes classifier:
    ĉ = argmax_k P(Cₖ) · Πᵢ P(xᵢ|Cₖ)
      = argmax_k [log P(Cₖ) + Σᵢ log P(xᵢ|Cₖ)]

MAP estimation:
    θ_MAP = argmax_θ P(θ|data) = argmax_θ [P(data|θ) · P(θ)]
```

---

## 11. Practice Problems (Solve Before Day 5)

1. A factory has 3 machines: A (50% of production, 2% defect rate), B (30%, 3% defect), C (20%, 5% defect). A randomly selected item is defective. What is the probability it came from machine C?

2. **Derive** Bayes' theorem from the definition of conditional probability in 3 lines.

3. A Naive Bayes classifier is trained on emails. P(spam) = 0.4. For a new email containing words w₁ and w₂:
   - P(w₁|spam)=0.6, P(w₁|ham)=0.1
   - P(w₂|spam)=0.3, P(w₂|ham)=0.8
   Classify the email. Compute the posterior probability of spam.

4. *(Monty Hall variant)* Now there are 4 doors (1 car, 3 goats). You pick Door 1. The host opens Door 4 (goat). Should you switch? What is P(car behind Door 2)?  
*(Hint: Enumerate likelihoods carefully.)*

5. *(Interview-level)* Explain why L2 regularization (adding λ||w||² to loss) is equivalent to placing a Gaussian prior N(0, 1/λ) on the weights. Show the connection mathematically using MAP estimation.

---

## 12. Looking Ahead

**Day 5** — **Independence & Conditional Independence.** The most subtle concept in probability — and the one most often misunderstood in interviews. We'll see how P(A∩B) = P(A)·P(B) leads to the entire architecture of graphical models, and why "correlated but conditionally independent" features are the secret behind Naive Bayes.

---
*End of Day 4 | Next: Day 5 — Independence & Conditional Independence*
