# MIT 18.05 — Introduction to Probability and Statistics
## Class 12: Bayesian Updating — Probabilistic Prediction & Odds
### Complete Study Notes | Jeremy Orloff & Jonathan Bloom | Spring 2022

---

> **How to use these notes:** This is a complete, self-contained reference for Class 12. It covers two closely related topics: (1) Prior and Posterior Predictive Probabilities, and (2) Odds, Bayes Factors, and the Strength of Evidence. Every example from all three PDFs is reproduced in full with expanded explanations and step-by-step reasoning.

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [From Updating Hypotheses to Predicting Outcomes](#2-from-updating-hypotheses-to-predicting-outcomes)
3. [Prior Predictive Probabilities](#3-prior-predictive-probabilities)
4. [Posterior Predictive Probabilities](#4-posterior-predictive-probabilities)
5. [The Four Probability Types — How They Relate](#5-the-four-probability-types--how-they-relate)
6. [Odds — Definition and Conversion](#6-odds--definition-and-conversion)
7. [Updating Odds: The Bayes Factor](#7-updating-odds-the-bayes-factor)
8. [The Bayes Factor as a Measure of Evidence](#8-the-bayes-factor-as-a-measure-of-evidence)
9. [Iterated Updating with Bayes Factors](#9-iterated-updating-with-bayes-factors)
10. [Log Odds](#10-log-odds)
11. [Conditional Independence](#11-conditional-independence)
12. [In-Class Problems with Full Solutions](#12-in-class-problems-with-full-solutions)
13. [Common Mistakes](#13-common-mistakes)
14. [Connections to Machine Learning & AI](#14-connections-to-machine-learning--ai)
15. [Quick Summary](#15-quick-summary)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Compute prior predictive probabilities** using the law of total probability before data is collected.
2. **Compute posterior predictive probabilities** using the law of total probability after updating with data.
3. **Convert between probability and odds** fluently.
4. **Update prior odds to posterior odds** using Bayes factors.
5. **Interpret Bayes factors** as a quantitative measure of the strength of evidence.
6. **Combine independent pieces of evidence** by multiplying Bayes factors (or adding log odds).
7. **Distinguish** between prior/posterior (for hypotheses) and prior predictive/posterior predictive (for outcomes).

---

## 2. From Updating Hypotheses to Predicting Outcomes

### 2.1 The Big Picture

In Class 11 we learned to update the probability of **hypotheses** given data. Now we go one step further: we use the updated (posterior) probabilities of hypotheses to **predict the probability of future observations**.

This is the full Bayesian pipeline:

```
Prior beliefs about     Observe     Posterior beliefs about     Predict
hypotheses P(H)    →    data D   →  hypotheses P(H|D)       →  outcomes P(D_new | D)
```

### 2.2 Why Predict?

The ultimate goal of a statistical model is not just to say "hypothesis $A$ is most likely" — it's to make useful predictions. Examples:

- **Medicine:** Given a patient's test results, what is the probability they respond to treatment $X$?
- **Weather:** Given today's atmospheric data, what is the probability of rain tomorrow?
- **Finance:** Given a company's historical returns, what is the probability of a positive return next quarter?
- **Sports:** Given past performance data, what is the probability this team wins the next game?
- **NLP:** Given the preceding tokens, what is the probability of each next token?

All of these require **predictive probabilities**, not just posterior probabilities of hypotheses.

### 2.3 Words of Estimative Probability (WEP)

There are three levels of precision in predictions:

| Style | Example |
|---|---|
| Simple prediction | "It will rain tomorrow." |
| Words of estimative probability (WEP) | "It is *likely* to rain tomorrow." |
| Probabilistic prediction | "Tomorrow it will rain with probability 60%." |

Probabilistic prediction is the most precise and the most useful. Words like "likely," "possible," "might," "cannot rule out," and "it's conceivable" are called **weasel words** — they sound informative but are actually vague and likely to cause misunderstanding.

> **For AI/ML:** All modern models produce probabilistic predictions. A classifier outputs $P(\text{class} = k \mid \mathbf{x})$, not just a class label. A language model outputs $P(\text{next token} \mid \text{context})$. The ability to reason with these probabilities is essential.

---

## 3. Prior Predictive Probabilities

### 3.1 Concept Overview

The **prior predictive probability** of an outcome is the probability of observing that outcome **before** any data has been collected. It is computed by averaging the outcome probability across all hypotheses, weighted by the prior probabilities of the hypotheses.

### 3.2 Formal Definition

Let $\mathcal{H}_1, \mathcal{H}_2, \ldots, \mathcal{H}_n$ be mutually exclusive and exhaustive hypotheses with prior probabilities $P(\mathcal{H}_i)$. Let $\mathcal{D}_{\text{new}}$ be a future observation. The prior predictive probability of $\mathcal{D}_{\text{new}}$ is:

$$\boxed{P(\mathcal{D}_{\text{new}}) = \sum_{i=1}^{n} P(\mathcal{D}_{\text{new}} \mid \mathcal{H}_i)\, P(\mathcal{H}_i)}$$

This is simply the **law of total probability** applied to outcomes, with hypotheses playing the role of the partition.

### 3.3 Intuition

Think of the prior predictive probability as a **weighted average** of the outcome probabilities under each hypothesis, where the weights are the prior probabilities of the hypotheses.

- If you're equally unsure among all hypotheses, you weight them equally.
- If you strongly believe in one hypothesis, its outcome probability dominates the average.

Before any data, you don't know which hypothesis is true, so you hedge across all of them.

---

### Example 1 — Prior Predictive Probability (Coins, Drawer Setup)

**Problem:** A drawer contains 4 coins: 2 of type $A$ ($P(H) = 0.5$), 1 of type $B$ ($P(H) = 0.6$), 1 of type $C$ ($P(H) = 0.9$). A coin is chosen at random. Before flipping, what is the probability the coin will land heads? Tails?

**Prior probabilities of hypotheses:**

$$P(A) = \frac{2}{4} = 0.5, \quad P(B) = \frac{1}{4} = 0.25, \quad P(C) = \frac{1}{4} = 0.25$$

**Likelihoods (probability of heads given coin type):**

$$P(H \mid A) = 0.5, \quad P(H \mid B) = 0.6, \quad P(H \mid C) = 0.9$$

**Step 1 — Apply the law of total probability for heads:**

$$P(D_H) = P(D_H \mid A)\,P(A) + P(D_H \mid B)\,P(B) + P(D_H \mid C)\,P(C)$$

$$= 0.5 \times 0.5 + 0.6 \times 0.25 + 0.9 \times 0.25$$

$$= 0.25 + 0.15 + 0.225 = \mathbf{0.625}$$

**Step 2 — Tails is the complement:**

$$P(D_T) = 1 - P(D_H) = 1 - 0.625 = \mathbf{0.375}$$

**Interpretation:** Before seeing any flip, there is a 62.5% chance of heads. This exceeds 50% because two of the three coin types are biased toward heads.

---

## 4. Posterior Predictive Probabilities

### 4.1 Concept Overview

The **posterior predictive probability** is the probability of a future outcome **after** updating on observed data. It uses the posterior probabilities of hypotheses (rather than the priors) as the weights in the total probability formula.

### 4.2 Formal Definition

After observing data $\mathcal{D}$ and computing posterior probabilities $P(\mathcal{H}_i \mid \mathcal{D})$, the posterior predictive probability of a new outcome $\mathcal{D}_{\text{new}}$ is:

$$\boxed{P(\mathcal{D}_{\text{new}} \mid \mathcal{D}) = \sum_{i=1}^{n} P(\mathcal{D}_{\text{new}} \mid \mathcal{H}_i)\, P(\mathcal{H}_i \mid \mathcal{D})}$$

The structure is identical to the prior predictive formula, but the weights are **posteriors** instead of priors.

### 4.3 Intuition

After observing data, you know more about which hypothesis is likely. So you weight the outcome probabilities by the updated (posterior) beliefs. If the data made hypothesis $C$ much more likely, then $C$'s outcome probability gets a bigger weight.

> **Key insight:** The posterior predictive is a more informed prediction than the prior predictive. Data makes your prediction sharper by concentrating weight on the hypotheses consistent with the observed evidence.

---

### Example 2 — Posterior Predictive Probability (Coins, First Flip = Heads)

**Problem:** Same drawer as Example 1. The coin is flipped once and lands **heads**. Given this data, what is the probability the next flip lands heads? Tails?

**Step 1 — Bayesian update table (first flip = heads):**

Data $\mathcal{D}$: first flip is heads.

| Hypothesis | Prior $P(H)$ | Likelihood $P(\mathcal{D} \mid H)$ | Bayes Numerator | Posterior $P(H \mid \mathcal{D})$ |
|---|---|---|---|---|
| $A$ | 0.50 | 0.5 | 0.250 | $0.250/0.625 = 0.400$ |
| $B$ | 0.25 | 0.6 | 0.150 | $0.150/0.625 = 0.240$ |
| $C$ | 0.25 | 0.9 | 0.225 | $0.225/0.625 = 0.360$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{D}) = 0.625$ | **1** |

**Step 2 — Posterior predictive probability of heads on the next flip:**

$$P(D_H \mid \mathcal{D}) = P(D_H \mid A)\,P(A \mid \mathcal{D}) + P(D_H \mid B)\,P(B \mid \mathcal{D}) + P(D_H \mid C)\,P(C \mid \mathcal{D})$$

$$= 0.5 \times 0.400 + 0.6 \times 0.240 + 0.9 \times 0.360$$

$$= 0.200 + 0.144 + 0.324 = \mathbf{0.668}$$

**Step 3 — Tails:**

$$P(D_T \mid \mathcal{D}) = 1 - 0.668 = \mathbf{0.332}$$

**Comparison with prior predictive:**

| | Prior Predictive | Posterior Predictive |
|---|---|---|
| $P(\text{heads})$ | 0.625 | 0.668 |
| $P(\text{tails})$ | 0.375 | 0.332 |

**Interpretation:** Observing heads on the first flip **increases** the probability of heads on the second flip (from 62.5% to 66.8%). This is not a coincidence — observing heads provides evidence in favour of the more biased coins ($B$ and $C$), which are more likely to produce heads on the next flip.

> **This is not a violation of independence!** The two flips of the same coin are independent *given the coin type*. But we don't know the coin type. Learning something about the likely coin type from flip 1 is informative about flip 2. This is precisely the Bayesian mechanism.

---

## 5. The Four Probability Types — How They Relate

This is a critical conceptual distinction. Students frequently confuse these four quantities.

### 5.1 Summary Table

| Term | Symbol | What it's about | When computed |
|---|---|---|---|
| **Prior probability** | $P(\mathcal{H})$ | Probability of a *hypothesis* | Before data |
| **Posterior probability** | $P(\mathcal{H} \mid \mathcal{D})$ | Probability of a *hypothesis* | After data |
| **Prior predictive** | $P(\mathcal{D}_{\text{new}})$ | Probability of an *outcome* | Before data |
| **Posterior predictive** | $P(\mathcal{D}_{\text{new}} \mid \mathcal{D})$ | Probability of an *outcome* | After data |

### 5.2 The Conceptual Chain

```
Prior P(H)  ────────────────────────────► Prior predictive P(D_new)
    │                                              │
    │  observe data D                              │  observe data D
    ▼                                              ▼
Posterior P(H|D) ────────────────────────► Posterior predictive P(D_new|D)
```

Both arrows going rightward use the **law of total probability**:
$$P(\text{outcome}) = \sum_i P(\text{outcome} \mid \mathcal{H}_i)\, P(\mathcal{H}_i) \quad \text{(using whichever weights are appropriate)}$$

### 5.3 Memory Aid

> **Prior/Posterior:** for **hypotheses** (what caused the data).
>
> **Predictive:** for **outcomes** (what data will look like).
>
> The word "predictive" tells you: we're predicting future data.

---

## 6. Odds — Definition and Conversion

### 6.1 Concept Overview

Odds are an alternative way to express the probability of an event. They are particularly common in gambling, medicine, and law. More importantly for us, odds have a beautiful mathematical property: **Bayes' theorem takes a multiplicative form in odds**, making calculations much simpler.

### 6.2 Formal Definition

> **Definition:** The **odds** of event $E$ are:
> $$O(E) = \frac{P(E)}{P(E^c)} = \frac{P(E)}{1 - P(E)}$$

If unspecified, the denominator is always $P(E^c)$, the complement of $E$.

### 6.3 Conversion Formulas

Going from probability to odds and back:

$$\boxed{O(E) = \frac{p}{1-p}} \qquad \text{and} \qquad \boxed{P(E) = \frac{q}{1+q}}$$

where $p = P(E)$ and $q = O(E)$.

**Derivation of the second formula:** If $O(E) = q$, then $p/(1-p) = q$, so $p = q(1-p) = q - qp$, giving $p(1+q) = q$, hence $p = q/(1+q)$.

### 6.4 Key Properties of Odds

1. **Probabilities** are between 0 and 1; **odds** are between 0 and $\infty$.
2. The complement: $O(E^c) = 1/O(E)$.
3. When $P(E)$ is small (rare events): $O(E) \approx P(E)$. (Because $1-P(E) \approx 1$.)
4. Odds of 1 ("evens" or "fifty-fifty") corresponds to probability 1/2.
5. Odds of 2 means the event is twice as likely to happen as not: $P(E) = 2/3$.

### 6.5 Worked Conversion Examples

---

**Example A — Fair Coin:**

$P(\text{heads}) = 1/2$, so:
$$O(\text{heads}) = \frac{1/2}{1/2} = 1$$

The odds are 1 to 1. We say "fifty-fifty" or "even odds."

---

**Example B — Standard Die (Rolling a 4):**

$P(\text{roll 4}) = 1/6$, so:
$$O(\text{roll 4}) = \frac{1/6}{5/6} = \frac{1}{5}$$

The odds are 1 to 5 *for*, or equivalently 5 to 1 *against* rolling a 4.

---

**Example C — Poker: Pair in a 5-Card Hand:**

$P(\text{pair}) = 0.42257$, so:
$$O(\text{pair}) = \frac{0.42257}{1 - 0.42257} = \frac{0.42257}{0.57743} = 0.73181$$

---

**Example D — Poker: Full House:**

$P(F) = 0.00145214$, so:
$$O(F) = \frac{0.00145214}{1 - 0.00145214} = \frac{0.00145214}{0.99855} \approx 0.001454$$

Note: $P(F) = 0.001452$ and $O(F) = 0.001454$ — they differ only in the fourth significant digit, illustrating property 3 above (for rare events, odds ≈ probability).

The odds of NOT having a full house:
$$O(F^c) = \frac{1}{O(F)} = \frac{1}{0.001454} \approx 687$$

---

## 7. Updating Odds: The Bayes Factor

### 7.1 Derivation of the Odds Form of Bayes' Theorem

This is one of the most elegant results in Bayesian statistics. Starting from Bayes' theorem for two competing hypotheses $H$ and $H^c$:

$$P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)} \qquad \text{and} \qquad P(H^c \mid D) = \frac{P(D \mid H^c)\, P(H^c)}{P(D)}$$

Divide the first by the second. The $P(D)$ in the denominators cancels:

$$\frac{P(H \mid D)}{P(H^c \mid D)} = \frac{P(D \mid H)}{P(D \mid H^c)} \cdot \frac{P(H)}{P(H^c)}$$

In the language of odds:

$$\boxed{O(H \mid D) = \underbrace{\frac{P(D \mid H)}{P(D \mid H^c)}}_{\text{Bayes factor}} \times \underbrace{O(H)}_{\text{prior odds}}}$$

Or equivalently:

$$\boxed{\text{posterior odds} = \text{Bayes factor} \times \text{prior odds}}$$

> **Why is this beautiful?** The total probability of data $P(D)$ completely drops out. You never need to compute the normalising constant.

### 7.2 Definition of the Bayes Factor

> **Definition:** For a hypothesis $H$ and data $D$, the **Bayes factor** (also called the **likelihood ratio**) is:
> $$BF = \frac{P(D \mid H)}{P(D \mid H^c)}$$

The Bayes factor measures how much more (or less) likely the observed data is under hypothesis $H$ than under $H^c$.

### 7.3 Practical Computation

To find posterior odds:

1. Compute prior odds: $O(H) = P(H)/P(H^c)$
2. Compute Bayes factor: $BF = P(D \mid H) / P(D \mid H^c)$
3. Multiply: $O(H \mid D) = BF \times O(H)$
4. (Optional) Convert back to probability: $P(H \mid D) = O(H \mid D) / (1 + O(H \mid D))$

---

### Example 3 — Disease Screening (Odds Form)

**Problem:** True positive rate 99%, false positive rate 2%, disease prevalence 0.5%. A random person tests positive. Find the prior and posterior odds.

**Step 1 — Prior odds:**

$$O(\mathcal{H}_+) = \frac{P(\mathcal{H}_+)}{P(\mathcal{H}_-)} = \frac{0.005}{0.995} \approx 0.00503$$

**Step 2 — Bayes factor:**

$$BF = \frac{P(\mathcal{T}_+ \mid \mathcal{H}_+)}{P(\mathcal{T}_+ \mid \mathcal{H}_-)} = \frac{0.99}{0.02} = 49.5 \approx 50$$

**Step 3 — Posterior odds:**

$$O(\mathcal{H}_+ \mid \mathcal{T}_+) = BF \times O(\mathcal{H}_+) = 49.5 \times \frac{0.005}{0.995} = \frac{0.99 \times 0.005}{0.02 \times 0.995} \approx \frac{1}{4}$$

**Step 4 — Convert to posterior probability:**

$$P(\mathcal{H}_+ \mid \mathcal{T}_+) = \frac{O(\mathcal{H}_+ \mid \mathcal{T}_+)}{1 + O(\mathcal{H}_+ \mid \mathcal{T}_+)} = \frac{0.25}{1.25} = 0.20 = 20\%$$

**Interpretation:**

- Prior odds: approximately 1 in 200 (disease is rare).
- Bayes factor: approximately 50 (the test provides strong evidence).
- Posterior odds: approximately 1 in 4 (after a positive test, disease is still less likely than no disease!).
- Posterior probability: ≈ 20%.

The Bayes factor of 50 tells us the positive test is 50 times more likely to occur if the person has the disease than if they don't — this is strong evidence. But the prior is so extreme (1 in 200) that even multiplying by 50 leaves the posterior odds at only 1 in 4.

---

### Example 4 — Marfan Syndrome (Ocular Features)

**Problem:** Marfan syndrome occurs in 1 in every 15,000 people. About 70% of Marfan patients have at least one ocular feature; only 7% of people without Marfan do. A person has at least one ocular feature. What are the odds they have Marfan syndrome?

**Step 1 — Prior odds:**

$$O(M) = \frac{P(M)}{P(M^c)} = \frac{1/15000}{14999/15000} = \frac{1}{14999} \approx 0.0000667$$

**Step 2 — Bayes factor:**

$$BF_F = \frac{P(F \mid M)}{P(F \mid M^c)} = \frac{0.7}{0.07} = 10$$

**Step 3 — Posterior odds:**

$$O(M \mid F) = BF_F \times O(M) = 10 \times \frac{1}{14999} = \frac{10}{14999} \approx 0.000667$$

**Step 4 — Convert to probability:**

$$P(M \mid F) = \frac{0.000667}{1.000667} \approx 0.0667\% \approx 1 \text{ in } 1499$$

**Verification using Bayesian update table:**

| Hypothesis | Prior $P(H)$ | Likelihood $P(F \mid H)$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $M$ | 0.0000667 | 0.7 | 0.0000467 | 0.000667 |
| $M^c$ | 0.9999333 | 0.07 | 0.069995 | 0.999333 |
| **Total** | **1** | **NO SUM** | 0.070042 | **1** |

**Interpretation:** The ocular feature is strong evidence (Bayes factor = 10) and increases the probability of Marfan syndrome tenfold. But the prior is so small that the posterior is still very small. The person would be wise to undergo further testing.

---

## 8. The Bayes Factor as a Measure of Evidence

### 8.1 The Core Insight

The Bayes factor tells you whether data is **evidence for or against** a hypothesis, and **how strong** that evidence is — independently of the prior.

$$\boxed{BF = \frac{P(D \mid H)}{P(D \mid H^c)}}$$

| Bayes Factor | Interpretation |
|---|---|
| $BF > 1$ | Data is evidence **for** $H$: posterior odds > prior odds |
| $BF = 1$ | Data provides **no evidence** either way |
| $BF < 1$ | Data is evidence **against** $H$: posterior odds < prior odds |
| $BF \gg 1$ | Very strong evidence for $H$ |
| $BF \approx 0$ | Very strong evidence against $H$ |

### 8.2 Breaking Even

The data is "neutral" ($BF = 1$) when $P(D \mid H) = P(D \mid H^c)$ — the data is equally likely under both hypotheses. In this case, posterior odds = prior odds, and you've learned nothing from the data.

---

### Example 5 — CSI: Blood Types at a Crime Scene

This example (from MacKay's *Information Theory, Inference, and Learning Algorithms*) illustrates how the same data can be evidence for one suspect and against another.

**Setup:**

- Two perpetrators left blood at a crime scene: one type O, one type AB.
- In the population: 60% are type O, 1% are type AB.
- Suspect **Oliver** has type O blood.
- Suspect **Alberto** has type AB blood.

**Data:** $D$ = "type O and type AB blood were found at the scene."

---

#### Part (a) — Oliver (Type O)

**Hypotheses:**
- $S$: Oliver and one unknown person were at the scene.
- $S^c$: Two unknown people were at the scene (Oliver was not there).

**Computing $P(D \mid S)$:**

If Oliver was at the scene, his type O blood is already accounted for. The data requires the second person to have type AB blood. Given that a random person has type AB with probability 0.01:

$$P(D \mid S) = 0.01$$

**Computing $P(D \mid S^c)$:**

If two random, unknown people were at the scene, we need one of them to be type O and the other type AB. With $N$ people in the population, $N_O = 0.6N$ type O and $N_{AB} = 0.01N$ type AB:

$$P(D \mid S^c) = \frac{\binom{N_O}{1}\binom{N_{AB}}{1}}{\binom{N}{2}} = \frac{N_O \cdot N_{AB}}{N(N-1)/2} = 2 \cdot \frac{N_O}{N} \cdot \frac{N_{AB}}{N-1} \approx 2 \times 0.6 \times 0.01 = 0.012$$

The factor of 2 arises because choosing "person 1 is O, person 2 is AB" and "person 1 is AB, person 2 is O" are both counted.

**Bayes factor for Oliver:**

$$BF_{\text{Oliver}} = \frac{P(D \mid S)}{P(D \mid S^c)} = \frac{0.01}{2 \times 0.6 \times 0.01} = \frac{0.01}{0.012} \approx 0.83$$

**Posterior odds:**

$$O(S \mid D) = 0.83 \times O(S)$$

Since $BF < 1$, the data is **(weak) evidence against Oliver** being at the scene! The data actually slightly decreases the probability that Oliver was there.

**Intuition:** Type O blood is very common (60% of the population). If Oliver is innocent, it would not be surprising at all to find type O blood at a two-person crime scene — there's a $2 \times 0.6 \times 0.01 = 1.2\%$ chance of that happening by coincidence. Oliver's presence only increases this to 1%. So his presence barely adds any explanatory power.

**Break-even point:** If the frequency of type O in the population were exactly 50% (matching Oliver's presence exactly), the Bayes factor would be 1 and the evidence would be neutral.

---

#### Part (b) — Alberto (Type AB)

**Hypotheses:**
- $A$: Alberto and one unknown person were at the scene.
- $A^c$: Two unknown people were at the scene.

**Computing $P(D \mid A)$:**

If Alberto (type AB) was at the scene, his blood is accounted for. The second person needs to be type O:

$$P(D \mid A) = 0.6$$

**Bayes factor for Alberto:**

$$BF_{\text{Alberto}} = \frac{P(D \mid A)}{P(D \mid A^c)} = \frac{0.6}{2 \times 0.6 \times 0.01} = \frac{0.6}{0.012} = 50$$

**Posterior odds:**

$$O(A \mid D) = 50 \times O(A)$$

Since $BF \gg 1$, the data is **(strong) evidence for Alberto** being at the scene.

**Intuition:** Type AB blood is extremely rare (1% of the population). If two random people were at the scene, the probability that one of them happens to be type AB is only about 2%. But if Alberto (who IS type AB) was there, his blood would certainly be found. The data is 50 times more likely given Alberto's presence than his absence — overwhelming evidence.

---

**Comparison:**

| Suspect | Blood Type | Blood Type Frequency | Bayes Factor | Evidence |
|---|---|---|---|---|
| Oliver | O | 60% (common) | 0.83 | Weak evidence *against* |
| Alberto | AB | 1% (rare) | 50 | Strong evidence *for* |

**Key lesson:** Finding a common blood type at a crime scene barely implicates a suspect, but finding a rare blood type is strong evidence.

---

## 9. Iterated Updating with Bayes Factors

### 9.1 Multiple Independent Pieces of Evidence

If data arrives in two conditionally independent stages $D_1$ then $D_2$, the Bayes factors multiply:

$$\boxed{O(H \mid D_1, D_2) = BF_2 \times BF_1 \times O(H)}$$

More generally, for $k$ conditionally independent pieces of evidence:

$$O(H \mid D_1, \ldots, D_k) = \left(\prod_{i=1}^{k} BF_i\right) \times O(H)$$

### 9.2 Why This Works

Conditional independence means $P(D_1, D_2 \mid H) = P(D_1 \mid H)\,P(D_2 \mid H)$. Substituting into the odds formula:

$$O(H \mid D_1, D_2) = \frac{P(D_1, D_2 \mid H)}{P(D_1, D_2 \mid H^c)} \cdot O(H) = \frac{P(D_1 \mid H)P(D_2 \mid H)}{P(D_1 \mid H^c)P(D_2 \mid H^c)} \cdot O(H) = BF_1 \times BF_2 \times O(H)$$

---

### Example 6 — Marfan Syndrome: Two Symptoms

**Problem:** Recall from Example 4 that Marfan syndrome occurs in 1 in 15,000 people. We established:

- $BF_F = 10$ for ocular features.
- New symptom: the **wrist sign** — the ability to wrap one hand around the other wrist to cover the pinky nail with the thumb.
  - $P(W \mid M) = 0.9$ (90% of Marfan patients have it)
  - $P(W \mid M^c) = 0.1$ (10% of the general population have it)

Compute the posterior odds and probability for a person who has BOTH symptoms.

**Step 1 — Bayes factor for wrist sign:**

$$BF_W = \frac{P(W \mid M)}{P(W \mid M^c)} = \frac{0.9}{0.1} = 9$$

**Step 2 — Combined posterior odds (assuming conditional independence):**

$$O(M \mid F, W) = BF_W \times BF_F \times O(M) = 9 \times 10 \times \frac{1}{14999} = \frac{90}{14999} \approx \frac{6}{1000}$$

**Step 3 — Convert to probability:**

$$P(M \mid F, W) = \frac{6/1000}{1 + 6/1000} = \frac{6}{1006} \approx 0.596\%$$

**Summary of progression:**

| Information | Odds of Marfan | Probability of Marfan |
|---|---|---|
| Prior (no symptoms) | 1 : 14999 | 0.0067% |
| After ocular features | 10 : 14999 | 0.067% |
| After both symptoms | 90 : 14999 | 0.60% |

Both symptoms multiply the prior odds by 90 (= 9 × 10). Still less than 1%, but 90× the original probability — serious enough to warrant further testing.

**Cancellation effect:** If a person has exactly ONE of the two symptoms, the product of Bayes factors is either $10/1 = 10$ (ocular only) or $9/1 = 9$ (wrist only). These two together would give $10 \times (1/9) \approx 1.1$ or $(1/10) \times 9 \approx 0.9$ — nearly 1. So the two symptoms nearly cancel each other out as evidence if one is present and one is absent.

> **Practical note:** This framework is used in real clinical diagnosis. Doctors combine symptoms as independent (or approximately independent) pieces of evidence. Each symptom is a Bayes factor that multiplies the odds.

---

## 10. Log Odds

### 10.1 Motivation

Working with products is less intuitive than working with sums. Taking the natural logarithm of the odds update formula converts products to sums:

$$O(H \mid D_1, D_2) = BF_2 \times BF_1 \times O(H)$$

becomes:

$$\ln O(H \mid D_1, D_2) = \ln(BF_2) + \ln(BF_1) + \ln O(H)$$

### 10.2 Formal Definition

$$\text{log odds of } H = \ln\left(\frac{P(H)}{P(H^c)}\right) = \ln P(H) - \ln P(H^c)$$

### 10.3 The Evidence Framework

With log odds, each piece of data contributes an additive **log Bayes factor**:

$$\underbrace{\ln O(H \mid D_1, \ldots, D_k)}_{\text{posterior log odds}} = \underbrace{\ln O(H)}_{\text{prior log odds}} + \sum_{i=1}^{k} \underbrace{\ln(BF_i)}_{\text{evidence from } D_i}$$

| Quantity | Interpretation |
|---|---|
| $\ln(BF_i) > 0$ ($BF_i > 1$) | Positive evidence for $H$ |
| $\ln(BF_i) = 0$ ($BF_i = 1$) | No evidence |
| $\ln(BF_i) < 0$ ($BF_i < 1$) | Negative evidence against $H$ |

### 10.4 Connection to Machine Learning

Log odds are the foundation of **logistic regression**, one of the most widely used classification algorithms. The logistic regression model is:

$$\ln\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \mathbf{w} \cdot \mathbf{x} + b$$

Each feature $x_i$ contributes additively to the log odds, exactly like independent Bayes factors. The sigmoid function converts log odds back to probability:

$$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

This is Bayesian reasoning in disguise: each feature is a piece of evidence that updates the prior log odds.

---

## 11. Conditional Independence

### 11.1 Definition

Events $D_1$ and $D_2$ are **conditionally independent given hypothesis $H$** if:

$$P(D_1, D_2 \mid H) = P(D_1 \mid H)\, P(D_2 \mid H) \quad \text{for every hypothesis } H$$

### 11.2 Why It Matters for Bayes Factors

Conditional independence is the condition that allows Bayes factors to multiply:

$$BF_{12} = \frac{P(D_1, D_2 \mid H)}{P(D_1, D_2 \mid H^c)} = \frac{P(D_1 \mid H)P(D_2 \mid H)}{P(D_1 \mid H^c)P(D_2 \mid H^c)} = BF_1 \times BF_2$$

Without conditional independence, you cannot simply multiply Bayes factors.

### 11.3 Conditional Independence ≠ Independence

This is a subtle but critical distinction. Dice rolls from an unknown die are a perfect illustration.

---

### Example 7 — Dice Rolls: Conditionally Independent but NOT Independent

**Setup:** Five dice (4, 6, 8, 12, 20 sides), one chosen at random. Roll twice. First roll: **7**. Second roll: **11**.

**Are the rolls conditionally independent?**

For each fixed die, the rolls are independent. Check for the 8-sided die:

$$P(\text{7 on roll 1} \mid d8) = 1/8$$
$$P(\text{11 on roll 2} \mid d8) = 0 \quad (\text{impossible on 8-sided die})$$
$$P(\text{7 on roll 1, 11 on roll 2} \mid d8) = 0 = (1/8)(0) \checkmark$$

For the 20-sided die:

$$P(\text{7 on roll 1} \mid d20) = 1/20$$
$$P(\text{11 on roll 2} \mid d20) = 1/20$$
$$P(\text{7 on roll 1, 11 on roll 2} \mid d20) = (1/20)^2 = 1/400 = (1/20)(1/20) \checkmark$$

This works for every die hypothesis, so **the rolls ARE conditionally independent**.

**Are the rolls (unconditionally) independent?**

We need to check whether $P(\text{7, 11}) = P(\text{7}) \times P(\text{11})$.

$$P(\text{7 on roll 1}) = \frac{1}{5}\left(0 + 0 + \frac{1}{8} + \frac{1}{12} + \frac{1}{20}\right) = \frac{1}{5} \cdot \frac{31}{120} = \frac{31}{600}$$

$$P(\text{11 on roll 2}) = \frac{1}{5}\left(0 + 0 + 0 + \frac{1}{12} + \frac{1}{20}\right) = \frac{1}{5} \cdot \frac{8}{60} = \frac{2}{75}$$

$$P(\text{7}) \times P(\text{11}) = \frac{31}{600} \times \frac{2}{75} = \frac{62}{45000} = \frac{31}{22500}$$

For the left side, use the law of total probability:

$$P(\text{7, 11}) = P(\text{11} \mid \text{7})\, P(\text{7})$$

After observing 7, the posterior probabilities of the dice are (from Class 11):
- $d8$: $P(d8 \mid \text{7}) = 30/93$
- $d12$: $P(d12 \mid \text{7}) = 31 \cdot (1/12) / \text{norm}$ ... 

Using the PDF's results: after observing 7, posteriors for $d12$ and $d20$ are $30/93$ and $6/93$ respectively (these come from Bayes numerators proportional to $1/12$ and $1/20$ among the remaining feasible dice with equal priors).

$$P(\text{11} \mid \text{7}) = \frac{30}{93} \cdot \frac{1}{12} + \frac{6}{93} \cdot \frac{1}{20} \cdot \ldots$$

The detailed calculation from the PDF shows $P(\text{7, 11}) = 17/9000 \neq 31/22500$.

**Why are they NOT independent?** Knowing roll 1 is 7 **rules out the 4-sided and 6-sided dice**, making an 11 on roll 2 more likely (only 12-sided and 20-sided dice remain as possibilities). The information from roll 1 propagates through the hypotheses to affect predictions for roll 2.

> **Key insight:** When you don't know the underlying parameter (which die, which coin, etc.), repeated observations from the same unknown source are **not** independent — they are correlated through the unknown parameter. Bayesian inference handles this naturally.

---

## 12. In-Class Problems with Full Solutions

### Problem 1 — Three Coins, 5 Tosses

**Setup:**
- Three coins: $C_{0.5}$ (fair), $C_{0.6}$, $C_{0.9}$.
- One chosen at random (equal priors: $1/3$ each).
- Toss 5 times; observe **1 head and 4 tails**.

#### Part (a) — Hypotheses and Priors

Hypotheses: $C_{0.5}$, $C_{0.6}$, $C_{0.9}$ (the chosen coin has that probability of heads).

Prior: $P(C_{0.5}) = P(C_{0.6}) = P(C_{0.9}) = 1/3$ (uniform, since we pick at random).

#### Part (b) — Prior Predictive Distribution

Before seeing any data, compute the probability of heads on a single toss:

$$P(H) = P(H \mid C_{0.5}) \cdot \frac{1}{3} + P(H \mid C_{0.6}) \cdot \frac{1}{3} + P(H \mid C_{0.9}) \cdot \frac{1}{3}$$

$$= 0.5 \cdot \frac{1}{3} + 0.6 \cdot \frac{1}{3} + 0.9 \cdot \frac{1}{3} = \frac{0.5 + 0.6 + 0.9}{3} = \frac{2.0}{3} = \frac{2}{3}$$

$$P(T) = 1 - P(H) = \frac{1}{3}$$

**Interpretation:** Without any data, the best guess for the next toss is 2/3 heads. This is higher than 50% because two of the three coins are biased toward heads.

#### Part (c) — Bayesian Update for 1 Head in 5 Tosses

**Likelihoods:** The likelihood of observing exactly 1 head in 5 tosses uses the binomial probability:

$$P(1 \text{ head in 5} \mid C_\theta) = \binom{5}{1} \theta^1 (1-\theta)^4 = 5\theta(1-\theta)^4$$

Computing each:

$$P(\mathcal{D} \mid C_{0.5}) = 5 \times 0.5 \times (0.5)^4 = 5 \times (0.5)^5 = 5/32 \approx 0.15625$$

$$P(\mathcal{D} \mid C_{0.6}) = 5 \times 0.6 \times (0.4)^4 = 5 \times 0.6 \times 0.0256 = 0.0768$$

$$P(\mathcal{D} \mid C_{0.9}) = 5 \times 0.9 \times (0.1)^4 = 5 \times 0.9 \times 0.0001 = 0.00045$$

**Bayesian update table:**

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $C_{0.5}$ | 1/3 | 0.15625 | $0.15625/3 = 0.05208$ | $0.05208/0.0778 \approx 0.669$ |
| $C_{0.6}$ | 1/3 | 0.07680 | $0.07680/3 = 0.02560$ | $0.02560/0.0778 \approx 0.329$ |
| $C_{0.9}$ | 1/3 | 0.00045 | $0.00045/3 = 0.00015$ | $0.00015/0.0778 \approx 0.002$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{D}) = 0.0778$ | **1** |

**Note on the $\binom{5}{1}$ factor:** The factor of 5 appears in every likelihood, so it cancels when computing the posterior. The notes point out that whether you know the order of tosses or just the count (1 head in 5 tosses vs. THTTT specifically) does not affect the posterior, because the $\binom{5}{1}$ factor is common to all hypotheses and cancels in the normalisation.

**Interpretation of posteriors:**

- Before data: all three coins equally likely (33.3% each).
- After observing only 1 head in 5 tosses: fair coin ($C_{0.5}$) is overwhelmingly likely (66.9%), the 0.6 coin is plausible (32.9%), and the 0.9 coin is nearly eliminated (0.2%). Getting only 1 head in 5 tosses is very unlikely for a coin biased toward heads.

#### Part (d) — Posterior Predictive Distribution

Using the posterior probabilities as weights:

$$P(H \mid \mathcal{D}) = P(H \mid C_{0.5})\,P(C_{0.5} \mid \mathcal{D}) + P(H \mid C_{0.6})\,P(C_{0.6} \mid \mathcal{D}) + P(H \mid C_{0.9})\,P(C_{0.9} \mid \mathcal{D})$$

$$= 0.5 \times 0.669 + 0.6 \times 0.329 + 0.9 \times 0.002$$

$$= 0.3345 + 0.1974 + 0.0018 = \mathbf{0.5337}$$

$$P(T \mid \mathcal{D}) = 1 - 0.5337 = \mathbf{0.4663}$$

**Comparison:**

| | Prior Predictive | Posterior Predictive |
|---|---|---|
| $P(\text{heads})$ | 0.667 | 0.534 |
| $P(\text{tails})$ | 0.333 | 0.466 |

**Interpretation:**

Seeing only 1 head in 5 tosses pushes the prediction toward the fair coin (which is less biased toward heads). The posterior predictive probability of heads drops from 66.7% to 53.4%. The data has updated our belief that this coin is close to fair, so our next-toss prediction moves toward 50%.

**Answer to the concept question at the top of Class 12:** The best guess for $P(\text{heads on next toss})$ is approximately **0.5** (answer (e)), which is closest to the posterior predictive of 0.534.

---

### Problem 2 — Screening Test: Odds Form

**Setup:** Disease prevalence 0.005, false positive rate 0.05, false negative rate 0.02.

So: $P(\mathcal{T}_+ \mid \mathcal{H}_+) = 1 - 0.02 = 0.98$ and $P(\mathcal{T}_+ \mid \mathcal{H}_-) = 0.05$.

#### Part (a) — Prior Odds

$$O(\mathcal{H}_+) = \frac{P(\mathcal{H}_+)}{P(\mathcal{H}_-)} = \frac{0.005}{0.995} \approx 0.00503$$

#### Part (b) — Bayes Factor

$$BF = \frac{P(\mathcal{T}_+ \mid \mathcal{H}_+)}{P(\mathcal{T}_+ \mid \mathcal{H}_-)} = \frac{0.98}{0.05} = 19.6$$

#### Part (c) — Posterior Odds

$$O(\mathcal{H}_+ \mid \mathcal{T}_+) = BF \times O(\mathcal{H}_+) = 19.6 \times 0.00503 \approx 0.0985$$

Convert to probability: $P(\mathcal{H}_+ \mid \mathcal{T}_+) = 0.0985 / (1 + 0.0985) \approx 9.0\%$

#### Part (d) — Strength of Evidence

A Bayes factor of 19.6 is strong evidence for the disease given a positive test. The positive test is nearly 20 times more likely to occur if the patient has the disease than if they don't. However, the prior odds are so small (about 1 in 200) that the posterior odds remain below 1 (about 1 in 10). The posterior probability is about 9%.

**Comparison note:** This problem uses a 5% false positive rate (vs. 2% in Examples 3 and 4 of the lecture notes), yielding a smaller Bayes factor (19.6 vs. 49.5) and lower posterior probability (9% vs. 20%).

---

### Problem 3 — CSI Blood Types (Full Solution)

This is the same as Examples 5a and 5b above. See Section 8 for full solutions.

**Summary:**

| Suspect | Blood Type | $P(D \mid \text{at scene})$ | $P(D \mid \text{not at scene})$ | Bayes Factor | Evidence |
|---|---|---|---|---|---|
| Oliver | O (60%) | 0.01 | $2(0.6)(0.01) = 0.012$ | $0.83 < 1$ | Weak against |
| Alberto | AB (1%) | 0.60 | $2(0.6)(0.01) = 0.012$ | $50 > 1$ | Strong for |

---

### Concept Question 2 — Does Order of Tosses Matter?

**Question:** Does knowing the tosses were "THTTT" (vs. just "1 head in 4 tails") affect the posterior distribution?

**Answer: No.**

When order is known (specific sequence THTTT), the likelihood is:
$$P(\text{THTTT} \mid \theta) = \theta^1(1-\theta)^4$$

When order is unknown (1 head in 5 tosses), the likelihood is:
$$P(\text{1 head in 5} \mid \theta) = \binom{5}{1}\theta^1(1-\theta)^4 = 5 \cdot \theta(1-\theta)^4$$

The factor $\binom{5}{1} = 5$ is a constant that appears for ALL hypotheses $\theta \in \{0.5, 0.6, 0.9\}$. When you normalise to compute posteriors (divide by the sum of Bayes numerators), this constant cancels. Therefore:

$$P(C_\theta \mid \text{1 head in 5}) = P(C_\theta \mid \text{THTTT})$$

**Proof sketch:**

$$\frac{P(\text{1H in 5} \mid C_{0.5}) \cdot P(C_{0.5})}{\sum_\theta P(\text{1H in 5} \mid C_\theta) P(C_\theta)} = \frac{5 \cdot P(\text{THTTT} \mid C_{0.5}) \cdot P(C_{0.5})}{5 \cdot \sum_\theta P(\text{THTTT} \mid C_\theta) P(C_\theta)} = \frac{P(\text{THTTT} \mid C_{0.5}) \cdot P(C_{0.5})}{\sum_\theta P(\text{THTTT} \mid C_\theta) P(C_\theta)}$$

The 5's cancel.

**General principle:** Any factor in the likelihood that is common across all hypotheses (i.e., depends only on the data, not the hypothesis) does not affect the posterior. The posterior only depends on how the likelihoods *differ* across hypotheses.

---

## 13. Common Mistakes

### Mistake 1: Confusing predictive probabilities with hypothesis probabilities

**Wrong:** "The posterior probability of heads is 0.669."

**Right:** "The posterior **predictive** probability of heads is 0.534." (The posterior probability of hypothesis $C_{0.5}$ is 0.669.)

The posteriors are for hypotheses. The predictive probabilities are for outcomes.

---

### Mistake 2: Using the wrong weights in the predictive formula

**Wrong:** Computing the posterior predictive probability using prior weights:

$$P(H \mid D) \neq P(H \mid C_{0.5}) P(C_{0.5}) + P(H \mid C_{0.6}) P(C_{0.6}) + \ldots$$

**Right:** Use posterior weights:

$$P(H \mid D) = P(H \mid C_{0.5}) P(C_{0.5} \mid D) + P(H \mid C_{0.6}) P(C_{0.6} \mid D) + \ldots$$

---

### Mistake 3: Thinking Bayes factor = posterior probability

**Wrong:** "The Bayes factor is 50, so the probability of the hypothesis is very high."

**Right:** The Bayes factor multiplies the **prior odds**, not the prior probability. If the prior odds are 1 in 1,000,000, even a Bayes factor of 50 gives posterior odds of only 1 in 20,000.

---

### Mistake 4: Multiplying Bayes factors without checking conditional independence

**Wrong:** Combining Bayes factors when symptoms are correlated.

**Right:** The product rule $O(H \mid D_1, D_2) = BF_1 \times BF_2 \times O(H)$ requires that $D_1$ and $D_2$ are conditionally independent given $H$ and $H^c$.

---

### Mistake 5: Forgetting the factor of 2 in "two random perpetrators" calculations

**Wrong:** $P(D \mid S^c) = 0.6 \times 0.01$ (choosing one O and one AB in a specific order).

**Right:** $P(D \mid S^c) = 2 \times 0.6 \times 0.01$ (because either of the two people could be type O, and the other type AB). The factor of 2 comes from the combinatorics of choosing an unordered pair.

---

### Mistake 6: Treating correlated observations as independent

**Wrong:** "Since the dice rolls are independent given the die, the rolls are independent."

**Right:** The rolls are conditionally independent (given the die) but NOT unconditionally independent (marginally). Observing roll 1 changes your beliefs about which die it is, which changes your prediction for roll 2.

---

## 14. Connections to Machine Learning & AI

### 14.1 Probabilistic Classifiers

Every probabilistic classifier (logistic regression, Naïve Bayes, neural networks with softmax) outputs posterior predictive probabilities:

$$P(Y = k \mid \mathbf{x}) \quad \text{for each class } k$$

These are exactly posterior predictive probabilities: they account for both the model (hypothesis) and the evidence (features).

### 14.2 Naïve Bayes = Iterated Bayes Factors

The Naïve Bayes classifier assumes conditional independence of features given the class label. This means:

$$O(Y=1 \mid x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} BF_i \times O(Y=1)$$

Each feature contributes a Bayes factor. This is exactly the structure of Section 9 (iterated updating with Bayes factors).

### 14.3 Logistic Regression = Log Odds with Linear Features

Logistic regression models the log odds as a linear function of features:

$$\ln O(Y=1 \mid \mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

Each weight $w_i$ is (approximately) the $\ln(BF_i)$ for feature $x_i$. The bias $b$ encodes the prior log odds.

### 14.4 Calibration

A model is **calibrated** if its posterior predictive probabilities are accurate: when it says 70%, it's right about 70% of the time. Bayesian models that use correct priors and likelihoods are inherently calibrated. Many neural networks are not calibrated without additional post-processing (temperature scaling, Platt scaling).

### 14.5 Uncertainty Quantification

The posterior predictive distribution is the right tool for expressing uncertainty in predictions. A model that says "70% ± wide uncertainty" is more honest and useful than one that always gives confident single-probability outputs. Bayesian methods naturally quantify this uncertainty through the distribution over hypotheses.

---

## 15. Quick Summary

### Part A: Predictive Probabilities

| | Formula | Weights |
|---|---|---|
| **Prior predictive** | $P(D_{\text{new}}) = \sum_i P(D_{\text{new}} \mid H_i) P(H_i)$ | Prior probs |
| **Posterior predictive** | $P(D_{\text{new}} \mid D) = \sum_i P(D_{\text{new}} \mid H_i) P(H_i \mid D)$ | Posterior probs |

**The difference:** Same formula structure, different weights (prior vs. posterior probabilities of hypotheses).

### Part B: Odds and Bayes Factors

| Concept | Formula |
|---|---|
| Odds from probability | $O(E) = p/(1-p)$ |
| Probability from odds | $P(E) = q/(1+q)$ |
| Posterior odds | $O(H \mid D) = BF \times O(H)$ |
| Bayes factor | $BF = P(D \mid H) / P(D \mid H^c)$ |
| Iterated updating | $O(H \mid D_1, D_2) = BF_1 \times BF_2 \times O(H)$ |
| Log odds update | $\ln O(H \mid D_1, D_2) = \ln BF_1 + \ln BF_2 + \ln O(H)$ |

### Key Conceptual Distinctions

| Term | About | When |
|---|---|---|
| Prior $P(H)$ | Hypothesis | Before data |
| Posterior $P(H \mid D)$ | Hypothesis | After data |
| Prior predictive $P(D_{\text{new}})$ | Outcome | Before data |
| Posterior predictive $P(D_{\text{new}} \mid D)$ | Outcome | After data |

### Evidence Interpretation

| Bayes Factor | Evidence |
|---|---|
| $BF > 1$ | For hypothesis $H$ |
| $BF = 1$ | No evidence either way |
| $BF < 1$ | Against hypothesis $H$ |

### Warning Signs

- ⚠️ Using prior weights instead of posterior weights in the predictive formula → gives prior predictive, not posterior predictive
- ⚠️ Forgetting that Bayes factors need conditional independence to multiply
- ⚠️ Forgetting the factor of 2 when computing probability of two random people having two specific blood types
- ⚠️ Confusing BF with posterior probability (BF is a multiplier on odds, not a probability itself)
- ⚠️ Treating marginally correlated observations as independent (the rolls of an unknown die are conditionally independent but NOT marginally independent)

---

*These notes cover all material from MIT 18.05 Class 12 (Bayesian Updating: Probabilistic Prediction & Odds), Spring 2022. All examples from the prep documents (both parts) and the in-class problem set are reproduced with expanded step-by-step reasoning.*

*Source: MIT OpenCourseWare — https://ocw.mit.edu | 18.05 Introduction to Probability and Statistics, Spring 2022*
