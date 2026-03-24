# MIT 18.05 — Introduction to Probability and Statistics
## Class 11: Bayesian Updating with Discrete Priors
### Complete Study Notes | Jeremy Orloff & Jonathan Bloom | Spring 2022

---

> **How to use these notes:** These notes are a complete, self-contained reference for Class 11. Every example from both the lecture prep document and the in-class problem set is reproduced here in full, with expanded explanations, step-by-step reasoning, and intuition for machine learning and data science applications.

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Why Bayesian Updating Matters](#2-why-bayesian-updating-matters)
3. [Review of Bayes' Theorem](#3-review-of-bayes-theorem)
4. [Core Terminology](#4-core-terminology)
5. [The Bayesian Update Table](#5-the-bayesian-update-table)
6. [Worked Examples — Coins](#6-worked-examples--coins)
7. [Prior and Posterior PMFs](#7-prior-and-posterior-pmfs)
8. [The Full Likelihood Table](#8-the-full-likelihood-table)
9. [Updating Again and Again (Iterated Updates)](#9-updating-again-and-again-iterated-updates)
10. [The Base Rate Fallacy](#10-the-base-rate-fallacy)
11. [In-Class Problems with Full Solutions](#11-in-class-problems-with-full-solutions)
12. [Probabilistic Prediction](#12-probabilistic-prediction)
13. [Common Mistakes](#13-common-mistakes)
14. [Connections to Machine Learning & AI](#14-connections-to-machine-learning--ai)
15. [Quick Summary](#15-quick-summary)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Apply Bayes' theorem** to compute updated (posterior) probabilities from observed data.
2. **Define and identify** the five key roles in Bayesian inference:
   - Prior probability
   - Data
   - Hypothesis
   - Likelihood (Bayes' term)
   - Posterior probability
3. **Build and use a Bayesian update table** to systematically compute posterior probabilities.
4. **Iterate** the update: use a posterior as a new prior when additional data arrives.
5. **Recognize and avoid** the base rate fallacy.

---

## 2. Why Bayesian Updating Matters

### 2.1 Concept Overview

Bayesian updating is the formal, mathematically rigorous way of **changing your beliefs in response to evidence**. Before you see any data you have some belief (the *prior*). After observing data, you revise that belief to get a new, more informed belief (the *posterior*).

This is precisely what learning means — you start with some assumption about the world, observe evidence, and rationally update your model.

### 2.2 Intuition

Think of a detective story. Before examining a crime scene a detective might suspect any of five suspects equally. After finding a fingerprint belonging to Suspect B, the detective now assigns Suspect B a much higher probability. The detective did not *prove* B is guilty — they simply updated their belief rationally in response to evidence.

Bayesian inference formalises exactly this reasoning:

$$\text{prior belief} \xrightarrow{\text{observe data}} \text{posterior belief}$$

### 2.3 Why It Underpins Statistics (and ML)

- **Spam filters** compute: given this email's word frequencies, what is the probability it is spam?
- **Medical diagnosis** computes: given a positive test, what is the probability the patient is actually sick?
- **Recommendation systems** compute: given this user's past behaviour, what is the probability they will like this item?
- **Language models** compute: given the preceding tokens, what is the probability of the next token?

All of these are Bayesian updates.

---

## 3. Review of Bayes' Theorem

### 3.1 Formal Statement

Let $\mathcal{H}$ be a hypothesis (event) and $\mathcal{D}$ be data (another event). Then:

$$\boxed{P(\mathcal{H} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H})\, P(\mathcal{H})}{P(\mathcal{D})}}$$

### 3.2 The Law of Total Probability (denominator expansion)

When the hypotheses $\mathcal{H}_1, \mathcal{H}_2, \ldots, \mathcal{H}_n$ partition the sample space (they are mutually exclusive and exhaustive):

$$P(\mathcal{D}) = \sum_{i=1}^{n} P(\mathcal{D} \mid \mathcal{H}_i)\, P(\mathcal{H}_i)$$

This is the denominator in Bayes' theorem and is called the **total probability of the data**.

### 3.3 The Most Elegant Form

With the data $\mathcal{D}$ fixed, $P(\mathcal{D})$ is just a normalising constant. Bayes' theorem becomes a *proportionality* statement as a function of $\mathcal{H}$:

$$\boxed{P(\mathcal{H} \mid \mathcal{D}) \;\propto\; P(\mathcal{D} \mid \mathcal{H})\cdot P(\mathcal{H})}$$

In words:

$$\text{posterior} \;\propto\; \text{likelihood} \times \text{prior}$$

This is the most important single equation in Bayesian statistics.

### 3.4 Inversion Insight

Notice that Bayes' theorem **inverts** the conditioning:

- You know $P(\mathcal{D} \mid \mathcal{H})$ — how likely the data is under each hypothesis.
- You want $P(\mathcal{H} \mid \mathcal{D})$ — how likely each hypothesis is, given the data.

These two quantities can be radically different. A test that is 99% accurate (high $P(\mathcal{D} \mid \mathcal{H})$) can still mean that a positive result is more likely a false positive than a true positive (low $P(\mathcal{H} \mid \mathcal{D})$), if the disease is rare. This is the **base rate fallacy**, covered in Section 10.

---

## 4. Core Terminology

Before doing any calculation it is essential to be precise about what each term means. The following definitions are illustrated with the coin example that runs throughout Class 11.

---

### 4.1 Experiment

The procedure that produces the data. For example: *pick a coin from a drawer at random, flip it once, record the result.*

---

### 4.2 Data $\mathcal{D}$

The observed outcome of the experiment. The data is the **evidence** that updates our beliefs.

> **Key insight:** The data is *fixed* — it is what we actually observed. The likelihood and posterior are computed *given* this fixed data.

Example: $\mathcal{D} = \{\text{the coin landed heads}\}$.

---

### 4.3 Hypotheses $\mathcal{H}$

The set of mutually exclusive and exhaustive candidate explanations for what is true. They are the "possible answers" to the question you are asking.

Example: $\mathcal{H} \in \{A, B, C\}$ where $A$, $B$, $C$ denote the possible coin types.

> **Important:** Hypotheses must cover all possibilities (exhaustive) and cannot overlap (mutually exclusive), so their prior probabilities sum to 1.

---

### 4.4 Prior Probability $P(\mathcal{H})$

The probability assigned to each hypothesis **before** observing the data. This encodes your existing knowledge or beliefs.

$$\text{prior} = P(\mathcal{H}) \quad \text{(before data)}$$

The prior is subjective in the sense that it comes from context (e.g., how many coins of each type are in the drawer). In the absence of any prior knowledge, a **uniform prior** assigns equal probability to all hypotheses.

---

### 4.5 Likelihood $P(\mathcal{D} \mid \mathcal{H})$

The probability of observing the data **assuming the hypothesis is true**.

$$\text{likelihood} = P(\mathcal{D} \mid \mathcal{H})$$

> ⚠️ **Critical warning:** The likelihood is a *probability of the data given the hypothesis*. It is **NOT** a probability of the hypothesis. The likelihood column in a Bayesian update table does NOT sum to 1.

**The likelihood is the bridge** between the data and the hypotheses. It tells you how consistent each hypothesis is with the observed data.

---

### 4.6 Posterior Probability $P(\mathcal{H} \mid \mathcal{D})$

The probability assigned to each hypothesis **after** observing the data. This is what you actually want to compute.

$$\text{posterior} = P(\mathcal{H} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H})\, P(\mathcal{H})}{P(\mathcal{D})}$$

The posterior is the updated prior: it incorporates both your prior belief and the evidence from the data.

---

### 4.7 Summary Table of Terminology

| Term | Symbol | Meaning | Sums to 1? |
|---|---|---|---|
| Prior | $P(\mathcal{H})$ | Probability of hypothesis *before* data | Yes |
| Likelihood | $P(\mathcal{D} \mid \mathcal{H})$ | Probability of data *given* hypothesis | **No** |
| Bayes numerator | $P(\mathcal{D} \mid \mathcal{H})\,P(\mathcal{H})$ | Product of likelihood and prior | No |
| Total prob. of data | $P(\mathcal{D})$ | Sum of Bayes numerators | — |
| Posterior | $P(\mathcal{H} \mid \mathcal{D})$ | Probability of hypothesis *after* data | Yes |

---

## 5. The Bayesian Update Table

### 5.1 Why Use a Table?

The tree diagram approach works for two or three hypotheses, but quickly becomes unwieldy. The Bayesian update table is a systematic, scalable format that:

- Organises all quantities in one place.
- Makes the structure of Bayes' theorem visually obvious.
- Is easy to extend to many hypotheses.
- Naturally handles multiple rounds of updating.

### 5.2 Structure of the Table

| Hypothesis $\mathcal{H}$ | Prior $P(\mathcal{H})$ | Likelihood $P(\mathcal{D}\mid\mathcal{H})$ | Bayes Numerator $P(\mathcal{D}\mid\mathcal{H})\,P(\mathcal{H})$ | Posterior $P(\mathcal{H}\mid\mathcal{D})$ |
|---|---|---|---|---|
| $\mathcal{H}_1$ | $p_1$ | $\ell_1$ | $\ell_1 p_1$ | $\ell_1 p_1 / P(\mathcal{D})$ |
| $\mathcal{H}_2$ | $p_2$ | $\ell_2$ | $\ell_2 p_2$ | $\ell_2 p_2 / P(\mathcal{D})$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{D})$ | **1** |

### 5.3 How to Fill in the Table — Step by Step

**Step 1:** List all hypotheses in the first column.

**Step 2:** Write the prior probability $P(\mathcal{H})$ for each hypothesis. These must sum to 1.

**Step 3:** Write the likelihood $P(\mathcal{D} \mid \mathcal{H})$ for each hypothesis given the observed data. These do **not** need to sum to 1.

**Step 4:** Compute each Bayes numerator: multiply prior × likelihood.

**Step 5:** Sum the Bayes numerators to get $P(\mathcal{D})$ (the total probability of the data).

**Step 6:** Divide each Bayes numerator by $P(\mathcal{D})$ to get the posterior.

> **Note:** Do NOT sum the likelihood column. This is a common error. The likelihood is not a probability distribution over hypotheses.

### 5.4 Key Insight: Only the Ratios Matter

Because $P(\mathcal{D})$ is the same for all hypotheses, the posterior probabilities are proportional to the Bayes numerators:

$$P(\mathcal{H}_i \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H}_i)\, P(\mathcal{H}_i)}{P(\mathcal{D})}$$

To find the **most probable hypothesis**, you only need to identify the largest Bayes numerator — no need to normalise.

---

## 6. Worked Examples — Coins

### Example 1 — Three Types of Coins (Heads Observed)

**Problem:** A drawer contains 5 coins: 2 of type $A$ (fair, $P(H) = 0.5$), 2 of type $B$ (bent, $P(H) = 0.6$), 1 of type $C$ (bent, $P(H) = 0.9$). A coin is chosen at random and flipped. It lands **heads**. What is the probability the coin is type $A$? Type $B$? Type $C$?

**Setup:**

- Hypotheses: $\mathcal{H} \in \{A, B, C\}$
- Data: $\mathcal{D} = \{\text{heads}\}$
- Priors: $P(A) = 2/5 = 0.4$, $P(B) = 2/5 = 0.4$, $P(C) = 1/5 = 0.2$
- Likelihoods: $P(\mathcal{D} \mid A) = 0.5$, $P(\mathcal{D} \mid B) = 0.6$, $P(\mathcal{D} \mid C) = 0.9$

**Tree diagram:**

```
           A (0.4)       B (0.4)       C (0.2)
          /      \      /      \      /      \
       H(0.5)  T(0.5) H(0.6) T(0.4) H(0.9) T(0.1)
```

**Step 1 — Total probability of heads:**

$$P(\mathcal{D}) = P(\mathcal{D} \mid A) P(A) + P(\mathcal{D} \mid B) P(B) + P(\mathcal{D} \mid C) P(C)$$

$$= 0.5 \times 0.4 + 0.6 \times 0.4 + 0.9 \times 0.2$$

$$= 0.20 + 0.24 + 0.18 = 0.62$$

**Step 2 — Bayesian update table:**

| Hypothesis | Prior $P(\mathcal{H})$ | Likelihood $P(\mathcal{D}\mid\mathcal{H})$ | Bayes Numerator | Posterior $P(\mathcal{H}\mid\mathcal{D})$ |
|---|---|---|---|---|
| $A$ | 0.4 | 0.5 | 0.20 | $0.20 / 0.62 = 0.3226$ |
| $B$ | 0.4 | 0.6 | 0.24 | $0.24 / 0.62 = 0.3871$ |
| $C$ | 0.2 | 0.9 | 0.18 | $0.18 / 0.62 = 0.2903$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{D}) = 0.62$ | **1** |

**Final Answers:**

$$P(A \mid \mathcal{D}) \approx 0.3226, \quad P(B \mid \mathcal{D}) \approx 0.3871, \quad P(C \mid \mathcal{D}) \approx 0.2903$$

**Interpretation:**

- Before the flip: $A$ and $B$ were equally likely (both 40%), and $C$ was least likely (20%).
- After observing heads: $B$ is now most likely (38.7%), $A$ has slightly decreased (32.3%), and $C$ has jumped from 20% to 29%.
- Why did $C$ increase the most? Because type $C$ coins land heads 90% of the time — observing heads is *most consistent* with hypothesis $C$. The data "pulls" toward the hypothesis with the highest likelihood.
- Why is $C$ not the winner? Because $C$ had only a 20% prior (just 1 out of 5 coins). The posterior is a tug-of-war between likelihood and prior. $B$ wins because it balances a high prior (40%) with a reasonably high likelihood (60%).

> **Key insight:** The MLE (Maximum Likelihood Estimate) would pick hypothesis $C$ because $P(\mathcal{D} \mid C) = 0.9$ is highest. But the posterior picks $B$ because it accounts for the prior. MLE ignores prior information; Bayesian inference incorporates it.

---

### Example 2 — Three Types of Coins (Tails Observed)

**Problem:** Same drawer as Example 1. The coin is flipped and lands **tails**. Find the posterior probabilities.

**Setup:** Same hypotheses and priors. New data: $\mathcal{D} = \{\text{tails}\}$.

New likelihoods:

$$P(\mathcal{D} \mid A) = 0.5, \quad P(\mathcal{D} \mid B) = 0.4, \quad P(\mathcal{D} \mid C) = 0.1$$

**Bayesian update table:**

| Hypothesis $\theta$ | Prior $p(\theta)$ | Likelihood $p(x=0 \mid \theta)$ | Bayes Numerator | Posterior $p(\theta \mid x=0)$ |
|---|---|---|---|---|
| 0.5 ($A$) | 0.4 | 0.5 | 0.20 | $0.20 / 0.38 = 0.5263$ |
| 0.6 ($B$) | 0.4 | 0.4 | 0.16 | $0.16 / 0.38 = 0.4211$ |
| 0.9 ($C$) | 0.2 | 0.1 | 0.02 | $0.02 / 0.38 = 0.0526$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{D}) = 0.38$ | **1** |

**Interpretation:**

Observing tails is *very inconsistent* with a biased-toward-heads coin (type $C$). The data is most consistent with the fair coin $A$ (50% chance of tails). Therefore $A$ rises from 40% to 52.6%, while $C$ collapses from 20% to just 5.3%. Tails is strong evidence against a coin with 90% heads probability.

---

### Example 3 — Two Flips: Both Heads

**Problem:** Same drawer. The coin is flipped **twice** and both times it lands **heads**. What are the posterior probabilities?

**Two approaches — both give the same answer:**

#### Approach A: One Big Update

Data: $\mathcal{D} = \{x_1 = 1, x_2 = 1\}$ (two heads). Assuming flips are independent given the coin type:

$$P(x_1=1, x_2=1 \mid \theta) = \theta \times \theta = \theta^2$$

| $\theta$ | Prior $p(\theta)$ | Likelihood $\theta^2$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| 0.5 | 0.4 | 0.25 | 0.10 | $0.10/0.406 = 0.2463$ |
| 0.6 | 0.4 | 0.36 | 0.144 | $0.144/0.406 = 0.3547$ |
| 0.9 | 0.2 | 0.81 | 0.162 | $0.162/0.406 = 0.3990$ |
| **Total** | **1** | | $P(\mathcal{D}) = 0.406$ | **1** |

#### Approach B: Two Sequential Updates

**Update 1:** Start with original priors, observe first heads. (This is just Example 1.)

After flip 1: posteriors are $(0.3226,\; 0.3871,\; 0.2903)$.

**Update 2:** Use these posteriors as the new priors, observe second heads.

| $\theta$ | New Prior (= post. 1) | Likelihood $\theta$ | Bayes Numerator | Final Posterior |
|---|---|---|---|---|
| 0.5 | 0.3226 | 0.5 | 0.1613 | $0.1613/0.6546 = 0.2463$ |
| 0.6 | 0.3871 | 0.6 | 0.2323 | $0.2323/0.6546 = 0.3547$ |
| 0.9 | 0.2903 | 0.9 | 0.2613 | $0.2613/0.6546 = 0.3990$ |
| **Total** | **1** | | 0.6546 | **1** |

Both approaches yield the same final posteriors: approximately $(0.246, 0.355, 0.399)$.

> **Fundamental property of Bayesian updating:** The order in which data arrives doesn't matter. You can process all data at once or sequentially; the final posterior is the same. This is because multiplication is commutative and associative.

**Interpretation:** After two heads, type $C$ (the strongly biased coin) has overtaken type $B$ and is now the most probable hypothesis (39.9%). Two heads in a row is increasingly strong evidence for a biased coin.

---

## 7. Prior and Posterior PMFs

### 7.1 Representing Hypotheses as Random Variables

When the hypotheses have numerical values we can represent them as a random variable $\theta$ with a probability mass function (PMF).

**Standard notation:**

- $\theta$: the value of the hypothesis
- $p(\theta)$: the **prior PMF** — the probability of each hypothesis before data
- $p(\theta \mid \mathcal{D})$: the **posterior PMF** — the probability of each hypothesis after data
- $p(\mathcal{D} \mid \theta)$: the **likelihood function** — note this is NOT a PMF in $\theta$

**Example from coins:** Map the coin types to their bias values: $A \to 0.5$, $B \to 0.6$, $C \to 0.9$.

| Hypothesis | $\theta$ | Prior $p(\theta)$ | Posterior $p(\theta \mid x=1)$ |
|---|---|---|---|
| $A$ | 0.5 | 0.4 | 0.3226 |
| $B$ | 0.6 | 0.4 | 0.3871 |
| $C$ | 0.9 | 0.2 | 0.2903 |

### 7.2 Visualising the Update

The prior and posterior PMFs can be plotted as bar charts (spike plots) on the same axis. After observing heads:

- Spike at $\theta = 0.9$ grows (from 0.2 to 0.29)
- Spikes at $\theta = 0.5$ and $\theta = 0.6$ shrink/shift

After observing tails:
- Spike at $\theta = 0.5$ grows (from 0.4 to 0.53)
- Spike at $\theta = 0.9$ collapses (from 0.2 to 0.05)

**Visualising the two-flip update shows three stages:**

```
Prior p(θ)        After flip 1 (H)      After flip 2 (H)
  |                   |                     |
.4| ██ ██             .4| ██                 .4|    ██
  |         ■         .3|    ██              .3|       ██
.2|                   .2|       ██          .2| ██
  |___________        |___________          |___________
  .5 .6     .9        .5 .6     .9          .5 .6     .9
```

Type $C$ steadily gains probability with each heads observation.

---

## 8. The Full Likelihood Table

### 8.1 What Is a Full Likelihood Table?

If you want to be prepared for **any possible data outcome**, you should compute the likelihood $P(\mathcal{D} \mid \mathcal{H})$ for every combination of hypothesis and data *before* seeing the actual data.

This is the **full likelihood table**: rows are hypotheses, columns are data outcomes.

### 8.2 Example — Coins (Full Likelihood Table)

For the coin example with data $x \in \{0 \text{ (tails)}, 1 \text{ (heads)}\}$:

| Hypothesis $\theta$ | $p(x = 0 \mid \theta)$ | $p(x = 1 \mid \theta)$ |
|---|---|---|
| 0.5 | 0.5 | 0.5 |
| 0.6 | 0.4 | 0.6 |
| 0.9 | 0.1 | 0.9 |

> **Usage:** When data comes in, you look up the appropriate column and plug it into your Bayesian update table as the likelihood column.

### 8.3 Example — Disease Screening (Full Likelihood Table)

For a disease screening test with hypotheses $\{\mathcal{H}_+, \mathcal{H}_-\}$ and data $\{\mathcal{T}_+, \mathcal{T}_-\}$:

| Hypothesis | $P(\mathcal{T}_+ \mid \mathcal{H})$ | $P(\mathcal{T}_- \mid \mathcal{H})$ |
|---|---|---|
| $\mathcal{H}_+$ (has disease) | 0.99 | 0.01 |
| $\mathcal{H}_-$ (no disease) | 0.02 | 0.98 |

The row for $\mathcal{H}_+$ says: if the patient has the disease, there is a 99% chance of testing positive and 1% chance of testing negative (false negative). Each row sums to 1 (since the test result is one of two outcomes). Columns do NOT need to sum to 1.

---

## 9. Updating Again and Again (Iterated Updates)

### 9.1 Concept

Life gives us data one observation at a time. Bayesian updating is designed for exactly this: after each new piece of data, you update your posterior. The posterior from round $k$ becomes the prior for round $k+1$.

$$\underbrace{p(\theta)}_{\text{prior}} \xrightarrow{\mathcal{D}_1} \underbrace{p(\theta \mid \mathcal{D}_1)}_{\text{posterior 1 = new prior}} \xrightarrow{\mathcal{D}_2} \underbrace{p(\theta \mid \mathcal{D}_1, \mathcal{D}_2)}_{\text{posterior 2}}$$

### 9.2 The Critical Theorem: Order Doesn't Matter

If the data points are **conditionally independent given the hypothesis** (i.e., knowing the hypothesis, the observations don't influence each other), then:

$$p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_1 \mid \theta)\, p(\mathcal{D}_2 \mid \theta)\, p(\theta)$$

This means:
- **Sequential update** (update for $\mathcal{D}_1$, then update the result for $\mathcal{D}_2$) gives the same answer as
- **Batch update** (update for $(\mathcal{D}_1, \mathcal{D}_2)$ all at once)

**Why?** Because $p(\mathcal{D}_1, \mathcal{D}_2 \mid \theta) = p(\mathcal{D}_1 \mid \theta)\, p(\mathcal{D}_2 \mid \theta)$ under conditional independence, which is just the product of the individual likelihoods.

---

## 10. The Base Rate Fallacy

### 10.1 Concept

The **base rate fallacy** occurs when people ignore the prior probability (base rate) of a hypothesis and over-weight the likelihood. This leads to dramatically wrong conclusions.

The classic context: medical screening tests. Many people intuitively believe that a 99%-accurate test means a positive result is 99% likely to indicate disease. This can be spectacularly wrong.

### 10.2 Formal Setup

| Quantity | Value |
|---|---|
| Disease prevalence (prior) | $P(\mathcal{H}_+) = 0.005$ (0.5%) |
| True positive rate (sensitivity) | $P(\mathcal{T}_+ \mid \mathcal{H}_+) = 0.99$ |
| False positive rate | $P(\mathcal{T}_+ \mid \mathcal{H}_-) = 0.02$ |
| True negative rate | $P(\mathcal{T}_- \mid \mathcal{H}_-) = 0.98$ |
| False negative rate | $P(\mathcal{T}_- \mid \mathcal{H}_+) = 0.01$ |

**Question:** A random person tests positive. What is the probability they actually have the disease?

### 10.3 Tree Diagram

```
                  0.005                    0.995
               H+ (disease)            H- (no disease)
              /             \          /              \
           T+               T-       T+               T-
          (0.99)           (0.01)   (0.02)           (0.98)
```

Paths:
- $P(\mathcal{H}_+, \mathcal{T}_+) = 0.005 \times 0.99 = 0.00495$
- $P(\mathcal{H}_-, \mathcal{T}_+) = 0.995 \times 0.02 = 0.0199$

### 10.4 Bayesian Update Table

| Hypothesis | Prior $P(\mathcal{H})$ | Likelihood $P(\mathcal{T}_+\mid\mathcal{H})$ | Bayes Numerator | Posterior $P(\mathcal{H}\mid\mathcal{T}_+)$ |
|---|---|---|---|---|
| $\mathcal{H}_+$ (disease) | 0.005 | 0.99 | 0.00495 | $0.00495 / 0.02485 \approx 0.199$ |
| $\mathcal{H}_-$ (no disease) | 0.995 | 0.02 | 0.01990 | $0.01990 / 0.02485 \approx 0.801$ |
| **Total** | **1** | **NO SUM** | $P(\mathcal{T}_+) = 0.02485$ | **1** |

### 10.5 Counterintuitive Result

$$P(\text{has disease} \mid \text{positive test}) \approx 20\%$$

Despite the test being 99% sensitive and 98% specific, a positive result is **only about 20% likely to be a true positive**!

**Why?** 

Think about 100,000 people tested:

- $100,000 \times 0.005 = 500$ have the disease. Of these, $500 \times 0.99 = 495$ test positive (true positives).
- $100,000 \times 0.995 = 99,500$ do not have the disease. Of these, $99,500 \times 0.02 = 1,990$ test positive (false positives).
- Total positive tests: $495 + 1,990 = 2,485$
- Fraction that are true positives: $495 / 2,485 \approx 19.9\%$

There are roughly 4 false positives for every true positive, because the disease is so rare.

### 10.6 Key Lesson

$$P(\mathcal{T}_+ \mid \mathcal{H}_+) = 0.99 \quad \text{is very different from} \quad P(\mathcal{H}_+ \mid \mathcal{T}_+) \approx 0.20$$

The positive test **greatly increases** the probability of disease (from 0.5% to 20% — a 40× increase!), but a positive result is still more likely to be wrong than right, because of the very low base rate.

---

## 11. In-Class Problems with Full Solutions

### Problem 1 — Disease Screening (Full Walkthrough)

**Setup:**
- Disease prevalence: $P(\mathcal{H}_+) = 0.005$
- False positive rate: $P(\mathcal{T}_+ \mid \mathcal{H}_-) = 0.02$
- False negative rate: $P(\mathcal{T}_- \mid \mathcal{H}_+) = 0.01$, so true positive rate is $P(\mathcal{T}_+ \mid \mathcal{H}_+) = 0.99$

A random patient is screened and tests **positive**.

#### Part (a) — Tree Diagram + Bayes' Theorem

Derived values:
$$P(\mathcal{H}_-) = 1 - 0.005 = 0.995$$
$$P(\mathcal{T}_+ \mid \mathcal{H}_+) = 1 - 0.01 = 0.99$$
$$P(\mathcal{T}_+ \mid \mathcal{H}_-) = 0.02$$

Total probability of a positive test:
$$P(\mathcal{T}_+) = (0.99)(0.005) + (0.02)(0.995) = 0.00495 + 0.01990 = 0.02485$$

Posteriors:
$$P(\mathcal{H}_+ \mid \mathcal{T}_+) = \frac{0.99 \times 0.005}{0.02485} = \frac{0.00495}{0.02485} \approx 0.199$$

$$P(\mathcal{H}_- \mid \mathcal{T}_+) = \frac{0.02 \times 0.995}{0.02485} = \frac{0.01990}{0.02485} \approx 0.801$$

#### Part (b) — Identifying the Terms

| Term | Value | Meaning |
|---|---|---|
| Data | $\mathcal{T}_+$ | Positive test result |
| Hypotheses | $\mathcal{H}_+$, $\mathcal{H}_-$ | Has disease; does not have disease |
| Likelihoods | $P(\mathcal{T}_+ \mid \mathcal{H}_+) = 0.99$; $P(\mathcal{T}_+ \mid \mathcal{H}_-) = 0.02$ | Prob. of positive test under each hypothesis |
| Priors | $P(\mathcal{H}_+) = 0.005$; $P(\mathcal{H}_-) = 0.995$ | Disease prevalence |
| Posteriors | $P(\mathcal{H}_+ \mid \mathcal{T}_+) \approx 0.199$; $P(\mathcal{H}_- \mid \mathcal{T}_+) \approx 0.801$ | Updated probabilities |

The structural relationship:

$$\underbrace{P(\mathcal{H}_+ \mid \mathcal{T}_+)}_{\text{Posterior}} = \frac{\overbrace{P(\mathcal{T}_+ \mid \mathcal{H}_+)}^{\text{Likelihood}} \cdot \overbrace{P(\mathcal{H}_+)}^{\text{Prior}}}{\underbrace{P(\mathcal{T}_+)}_{\text{Total prob. of data}}}$$

#### Part (c) — Full Likelihood Table

| Hypothesis | $P(\mathcal{T}_+ \mid \mathcal{H})$ | $P(\mathcal{T}_- \mid \mathcal{H})$ |
|---|---|---|
| $\mathcal{H}_+$ | 0.99 | 0.01 |
| $\mathcal{H}_-$ | 0.02 | 0.98 |

Each row sums to 1 (the two test outcomes are exhaustive). Columns do not sum to 1.

#### Part (d) — Full Bayesian Update Table

| Hypothesis $\mathcal{H}$ | Prior $P(\mathcal{H})$ | Likelihood $P(\mathcal{T}_+\mid\mathcal{H})$ | Bayes Numerator | Posterior $P(\mathcal{H}\mid\mathcal{T}_+)$ |
|---|---|---|---|---|
| $\mathcal{H}_+$ | 0.005 | 0.99 | 0.00495 | 0.199 |
| $\mathcal{H}_-$ | 0.995 | 0.02 | 0.01990 | 0.801 |
| **Total** | **1** | **NO SUM** | $P(\mathcal{T}_+) = 0.02485$ | **1** |

---

### Problem 2 — Five Dice

**Setup:** You have five dice: 4-sided ($d4$), 6-sided ($d6$), 8-sided ($d8$), 12-sided ($d12$), 20-sided ($d20$). One is chosen at random and rolled.

**Prior:** Uniform — $P(\mathcal{H}_{n}) = 1/5$ for each $n \in \{4, 6, 8, 12, 20\}$.

**Likelihood:** If a $k$-sided die is rolled, each face has probability $1/k$. A roll of value $v$ is possible only if $v \leq k$:

$$P(\text{roll} = v \mid \mathcal{H}_k) = \begin{cases} 1/k & \text{if } v \leq k \\ 0 & \text{if } v > k \end{cases}$$

#### Part (a) — Hypotheses

$\mathcal{H}_4$: die is 4-sided; $\mathcal{H}_6$: 6-sided; $\mathcal{H}_8$: 8-sided; $\mathcal{H}_{12}$: 12-sided; $\mathcal{H}_{20}$: 20-sided.

#### Part (b) — Full Likelihood Table

| Hypothesis $\mathcal{H}$ | $P(5 \mid \mathcal{H})$ | $P(9 \mid \mathcal{H})$ | $P(13 \mid \mathcal{H})$ |
|---|---|---|---|
| $\mathcal{H}_4$ | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 1/6 | 0 | 0 |
| $\mathcal{H}_8$ | 1/8 | 1/8 | 0 |
| $\mathcal{H}_{12}$ | 1/12 | 1/12 | 0 |
| $\mathcal{H}_{20}$ | 1/20 | 1/20 | 1/20 |

**Reading the table:** A roll of 13 is impossible on any die with fewer than 13 sides, so likelihoods are 0 for $d4$, $d6$, $d8$, $d12$. Only $d20$ can produce a 13.

#### Part (c) — Update for Roll = 13

Data $\mathcal{D}$: rolled a **13**.

| Hypothesis | Prior | Likelihood $P(13\mid\mathcal{H})$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_8$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_{12}$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_{20}$ | 1/5 | 1/20 | $1/100$ | **1** |
| **Total** | **1** | | $1/100$ | **1** |

**Conclusion:** Rolling a 13 proves the die is the 20-sided die with **certainty** ($P(\mathcal{H}_{20} \mid 13) = 1$). No other die can produce a 13.

**Intuition:** The data is **impossible** under four of the five hypotheses. Bayes' theorem correctly assigns them zero posterior probability.

#### Part (d) — Update for Roll = 5

Data $\mathcal{D}$: rolled a **5**.

A 5 is impossible on $d4$, possible on $d6$, $d8$, $d12$, $d20$.

Computing Bayes numerators (prior is 1/5 for all):

| Hypothesis | Prior | Likelihood $P(5\mid\mathcal{H})$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 1/5 | 1/6 | $1/30$ | |
| $\mathcal{H}_8$ | 1/5 | 1/8 | $1/40$ | |
| $\mathcal{H}_{12}$ | 1/5 | 1/12 | $1/60$ | |
| $\mathcal{H}_{20}$ | 1/5 | 1/20 | $1/100$ | |
| **Total** | **1** | | $P(\mathcal{D})$ | **1** |

Computing $P(\mathcal{D})$: find common denominator (LCM of 30, 40, 60, 100 = 600):

$$\frac{1}{30} + \frac{1}{40} + \frac{1}{60} + \frac{1}{100} = \frac{20}{600} + \frac{15}{600} + \frac{10}{600} + \frac{6}{600} = \frac{51}{600} = 0.085$$

Posteriors:

| Hypothesis | Bayes Numerator | Posterior |
|---|---|---|
| $\mathcal{H}_4$ | 0 | 0 |
| $\mathcal{H}_6$ | $1/30 \approx 0.0333$ | $0.0333/0.085 \approx 0.392$ |
| $\mathcal{H}_8$ | $1/40 = 0.025$ | $0.025/0.085 \approx 0.294$ |
| $\mathcal{H}_{12}$ | $1/60 \approx 0.0167$ | $0.0167/0.085 \approx 0.196$ |
| $\mathcal{H}_{20}$ | $1/100 = 0.01$ | $0.01/0.085 \approx 0.118$ |
| **Total** | 0.085 | **1** |

**Conclusion:** $\mathcal{H}_4$ is eliminated. $\mathcal{H}_6$ is most likely (39.2%). Smaller dice become more probable because a 5 is a larger fraction of their possible outcomes: $P(5 \mid d6) = 1/6 > P(5 \mid d20) = 1/20$.

**Intuition:** A 5 is a "small" number. If the die produced a small number, it is more likely to be a small die than a large one. Think of it this way: if you rolled a fair 20-sided die, getting exactly a 5 (out of 20 equally likely outcomes) is less informative than getting a 5 on a 6-sided die (where 5 is near the maximum).

#### Part (e) — Update for Roll = 9

Data $\mathcal{D}$: rolled a **9**.

A 9 is impossible on $d4$, $d6$, $d8$. Possible on $d12$ and $d20$.

| Hypothesis | Prior | Likelihood $P(9\mid\mathcal{H})$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_8$ | 1/5 | 0 | 0 | 0 |
| $\mathcal{H}_{12}$ | 1/5 | 1/12 | $1/60$ | |
| $\mathcal{H}_{20}$ | 1/5 | 1/20 | $1/100$ | |
| **Total** | **1** | | $P(\mathcal{D})$ | **1** |

$$P(\mathcal{D}) = \frac{1}{60} + \frac{1}{100} = \frac{5}{300} + \frac{3}{300} = \frac{8}{300} = \frac{2}{75} \approx 0.0267$$

$$P(\mathcal{H}_{12} \mid 9) = \frac{1/60}{2/75} = \frac{1}{60} \times \frac{75}{2} = \frac{75}{120} = 0.625$$

$$P(\mathcal{H}_{20} \mid 9) = \frac{1/100}{2/75} = \frac{1}{100} \times \frac{75}{2} = \frac{75}{200} = 0.375$$

| Hypothesis | Posterior |
|---|---|
| $\mathcal{H}_4$ | 0 |
| $\mathcal{H}_6$ | 0 |
| $\mathcal{H}_8$ | 0 |
| $\mathcal{H}_{12}$ | **0.625** |
| $\mathcal{H}_{20}$ | 0.375 |

**Conclusion:** $\mathcal{H}_{12}$ is most likely (62.5%). A 9 eliminates all dice with fewer than 9 sides, and among the survivors, $d12$ is more likely because $P(9 \mid d12) = 1/12 > P(9 \mid d20) = 1/20$.

---

### Problem 3 — Iterated Updates: Two Dice Rolls

**Setup:** Same five dice, uniform priors. Rolled a **9** followed by a **5** (from the same die).

Goal: Find the posterior probabilities after both rolls.

#### Approach (a) — Two Sequential Updates

**Step 1:** Update for roll = 9. From Part (e) above, posterior after observing 9:

| Hypothesis | Posterior after 9 |
|---|---|
| $\mathcal{H}_4$ | 0 |
| $\mathcal{H}_6$ | 0 |
| $\mathcal{H}_8$ | 0 |
| $\mathcal{H}_{12}$ | 0.625 |
| $\mathcal{H}_{20}$ | 0.375 |

**Step 2:** Update for roll = 5, using posteriors from Step 1 as new priors.

Note: A 5 is impossible on $d4$, $d6$, $d8$ — but those already have probability 0, so no change.

| Hypothesis | New Prior (= post. after 9) | Likelihood $P(5\mid\mathcal{H})$ | Bayes Numerator | Final Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 0 | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 0 | 1/6 | 0 | 0 |
| $\mathcal{H}_8$ | 0 | 1/8 | 0 | 0 |
| $\mathcal{H}_{12}$ | 0.625 | 1/12 | $0.625/12 \approx 0.05208$ | |
| $\mathcal{H}_{20}$ | 0.375 | 1/20 | $0.375/20 = 0.01875$ | |
| **Total** | **1** | | $\approx 0.0708$ | **1** |

**Alternatively**, work with exact fractions. After roll of 9:
- $P(\mathcal{H}_{12}) = 1/60 \div (2/75) = 5/8$
- $P(\mathcal{H}_{20}) = 1/100 \div (2/75) = 3/8$

Bayes numerator for roll of 5:
- $\mathcal{H}_{12}$: $(5/8)(1/12) = 5/96$... wait, let me use the table from the PDF directly.

From the PDF's two-step table:

| Hyp. | Prior | Likel. 1 (roll 9) | Bayes Num. 1 | Likel. 2 (roll 5) | Bayes Num. 2 | Posterior |
|---|---|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/5 | 0 | 0 | 0 | 0 | 0 |
| $\mathcal{H}_6$ | 1/5 | 0 | 0 | 1/6 | 0 | 0 |
| $\mathcal{H}_8$ | 1/5 | 0 | 0 | 1/8 | 0 | 0 |
| $\mathcal{H}_{12}$ | 1/5 | 1/12 | 1/60 | 1/12 | **1/720** | |
| $\mathcal{H}_{20}$ | 1/5 | 1/20 | 1/100 | 1/20 | **1/2000** | |
| **Total** | **1** | | | | $\approx 0.0019$ | **1** |

**Key formula for two-step Bayes numerator:**

$$\text{Bayes Num. 2} = \text{likelihood 2} \times \text{Bayes Num. 1} = P(\mathcal{D}_2 \mid \mathcal{H}) \times P(\mathcal{D}_1 \mid \mathcal{H}) \times P(\mathcal{H})$$

Computing the total:
$$P(\mathcal{D}_1, \mathcal{D}_2) = \frac{1}{720} + \frac{1}{2000}$$

LCM of 720 and 2000: factor as $720 = 2^4 \times 3^2 \times 5$ and $2000 = 2^4 \times 5^3$, so LCM $= 2^4 \times 3^2 \times 5^3 = 18000$.

$$= \frac{25}{18000} + \frac{9}{18000} = \frac{34}{18000} = \frac{17}{9000} \approx 0.001889$$

$$P(\mathcal{H}_{12} \mid \mathcal{D}_1, \mathcal{D}_2) = \frac{1/720}{17/9000} = \frac{9000}{720 \times 17} = \frac{9000}{12240} = \frac{25}{34} \approx 0.735$$

$$P(\mathcal{H}_{20} \mid \mathcal{D}_1, \mathcal{D}_2) = \frac{1/2000}{17/9000} = \frac{9000}{2000 \times 17} = \frac{9000}{34000} = \frac{9}{34} \approx 0.265$$

#### Approach (b) — Single Batch Update

Data: "rolled a 9, then a 5" from the same die.

Assuming conditional independence:
$$P(\text{9 then 5} \mid \mathcal{H}_k) = P(9 \mid \mathcal{H}_k) \times P(5 \mid \mathcal{H}_k)$$

| Hypothesis | Prior | Likelihood $P(9,5\mid\mathcal{H})$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/5 | $0 \times 0 = 0$ | 0 | 0 |
| $\mathcal{H}_6$ | 1/5 | $0 \times 1/6 = 0$ | 0 | 0 |
| $\mathcal{H}_8$ | 1/5 | $0 \times 1/8 = 0$ | 0 | 0 |
| $\mathcal{H}_{12}$ | 1/5 | $(1/12)(1/12) = 1/144$ | $1/720$ | $0.735$ |
| $\mathcal{H}_{20}$ | 1/5 | $(1/20)(1/20) = 1/400$ | $1/2000$ | $0.265$ |
| **Total** | **1** | | $\approx 0.00189$ | **1** |

Both approaches yield identical results: $P(\mathcal{H}_{12} \mid 9,5) \approx 0.735$ and $P(\mathcal{H}_{20} \mid 9,5) \approx 0.265$.

**Conclusion:** Observing both a 9 and a 5 strongly favours the 12-sided die. A 9 and a 5 are both in the range $[1, 12]$, but the 12-sided die produces each face with probability 1/12 while the 20-sided die does so with only 1/20. The smaller die is more "efficient" at producing these values.

---

### Extra Problem 1 — Bag of Dice (900 vs 1000)

**Setup:** A bag contains 1 four-sided die and 999 six-sided dice. One is chosen at random and rolled. The result is a **3**.

Find the probability the chosen die is 4-sided.

**Priors:** $P(\mathcal{H}_4) = 1/1000$, $P(\mathcal{H}_6) = 999/1000$

**Likelihoods:** $P(3 \mid \mathcal{H}_4) = 1/4$ (4-sided die), $P(3 \mid \mathcal{H}_6) = 1/6$ (6-sided die)

**Total probability of roll = 3:**

$$P(R=3) = \frac{1}{4} \cdot \frac{1}{1000} + \frac{1}{6} \cdot \frac{999}{1000} = \frac{1}{4000} + \frac{999}{6000}$$

$$= \frac{3}{12000} + \frac{1998}{12000} = \frac{2001}{12000} \approx 0.1668$$

**Bayesian update table:**

| Hypothesis | Prior $P(\mathcal{H})$ | Likelihood $P(3\mid\mathcal{H})$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\mathcal{H}_4$ | 1/1000 | 1/4 | $1/4000$ | $\approx 0.0015$ |
| $\mathcal{H}_6$ | 999/1000 | 1/6 | $999/6000$ | $\approx 0.9985$ |
| **Total** | **1** | **NO SUM** | $P(R=3) \approx 0.167$ | **1** |

Computing posteriors:

$$P(\mathcal{H}_4 \mid R=3) = \frac{1/4000}{2001/12000} = \frac{12000}{4000 \times 2001} = \frac{3}{2001} \approx 0.0015$$

$$P(\mathcal{H}_6 \mid R=3) = \frac{999/6000}{2001/12000} = \frac{999 \times 12000}{6000 \times 2001} = \frac{1998}{2001} \approx 0.9985$$

**Interpretation:** 

Even though rolling a 3 is *more likely* under the 4-sided hypothesis ($P(3 \mid \mathcal{H}_4) = 0.25$) than under the 6-sided hypothesis ($P(3 \mid \mathcal{H}_6) = 0.167$), the posterior probability of $\mathcal{H}_4$ is still tiny (0.15%).

**Why?** The prior of $\mathcal{H}_4$ is 1/1000 — overwhelmingly unlikely before seeing any data. Rolling a 3 increases the probability of $\mathcal{H}_4$ by a ratio of $0.25/0.167 = 1.5$, but starting from 0.1% and multiplying by 1.5 still gives only 0.15%.

The prior dominates because it is so extreme. This is a perfect illustration of why **base rates matter**.

---

## 12. Probabilistic Prediction

### 12.1 Setup

Sometimes we want to predict the outcome of a future observation given past data, without knowing the true hypothesis. This requires averaging over hypotheses.

**Setup:** Same five dice. Roll once and observe $\mathcal{D}_1$. What is the probability of a particular value on the second roll?

### 12.2 Law of Total Probability for Prediction

$$P(\mathcal{D}_2 \mid \mathcal{D}_1) = \sum_{\mathcal{H}} P(\mathcal{D}_2 \mid \mathcal{H})\, P(\mathcal{H} \mid \mathcal{D}_1)$$

This is a weighted average of the likelihood of $\mathcal{D}_2$ under each hypothesis, weighted by the posterior probabilities (how likely each hypothesis is, given $\mathcal{D}_1$).

### 12.3 Worked Example

**Find $P(\mathcal{D}_1 = 5)$ and $P(\mathcal{D}_2 = 4 \mid \mathcal{D}_1 = 5)$.**

**Step 1:** $P(\mathcal{D}_1 = 5)$ is just the sum of Bayes numerators from the "roll = 5" update:

$$P(\mathcal{D}_1 = 5) = \frac{1}{30} + \frac{1}{40} + \frac{1}{60} + \frac{1}{100} = \frac{51}{600} \approx 0.085$$

**Step 2:** After observing $\mathcal{D}_1 = 5$, the posterior distribution is (from Problem 2d):

| Hypothesis | $P(\mathcal{H} \mid \mathcal{D}_1 = 5)$ | $P(\mathcal{D}_2 = 4 \mid \mathcal{H})$ | Product |
|---|---|---|---|
| $\mathcal{H}_4$ | 0 | — | 0 |
| $\mathcal{H}_6$ | 0.392 | 1/6 | $0.392/6 \approx 0.0653$ |
| $\mathcal{H}_8$ | 0.294 | 1/8 | $0.294/8 \approx 0.0368$ |
| $\mathcal{H}_{12}$ | 0.196 | 1/12 | $0.196/12 \approx 0.0163$ |
| $\mathcal{H}_{20}$ | 0.118 | 1/20 | $0.118/20 \approx 0.0059$ |
| **Total** | **1** | | **0.124** |

$$P(\mathcal{D}_2 = 4 \mid \mathcal{D}_1 = 5) \approx 0.124$$

**Interpretation:** Given that the first roll was a 5, the probability of rolling a 4 next is about 12.4%. This accounts for uncertainty about which die we're using: if the die is $d6$ (most likely after rolling a 5), then $P(4 \mid d6) = 1/6 \approx 16.7\%$; if it's a $d20$ (least likely), then $P(4 \mid d20) = 5\%$.

---

## 13. Common Mistakes

### Mistake 1: Summing the likelihood column

**Wrong:** Adding up the likelihood column and setting it equal to 1.

**Right:** The likelihood $P(\mathcal{D} \mid \mathcal{H})$ is not a probability distribution over hypotheses. It can sum to anything. Only the prior and posterior columns sum to 1.

---

### Mistake 2: Confusing $P(\mathcal{D} \mid \mathcal{H})$ with $P(\mathcal{H} \mid \mathcal{D})$

**Wrong:** Thinking "the test is 99% accurate, so a positive test means I'm 99% likely to be sick."

**Right:** $P(\text{positive} \mid \text{sick}) = 0.99$ (sensitivity) is completely different from $P(\text{sick} \mid \text{positive})$, which also depends on the prevalence (prior).

---

### Mistake 3: Ignoring the prior (base rate fallacy)

**Wrong:** Using only the likelihood to draw conclusions.

**Right:** The posterior depends on BOTH the likelihood AND the prior. A very strong prior can dominate even a strong likelihood.

---

### Mistake 4: Treating zero-likelihood hypotheses as impossible before the data

**Wrong:** Eliminating hypotheses before computing the update.

**Right:** Hypotheses with zero likelihood for the observed data will naturally get posterior probability zero. You don't need to remove them manually; the math handles it.

---

### Mistake 5: Forgetting to normalise in the final step

**Wrong:** Reporting Bayes numerators as posterior probabilities.

**Right:** Divide each Bayes numerator by $P(\mathcal{D})$ (the sum of all numerators) to get proper posteriors that sum to 1.

---

### Mistake 6: Confusing prior of $\mathcal{H}$ with likelihood

**Wrong:** Using the probability of a hypothesis as the likelihood.

**Right:** The likelihood $P(\mathcal{D} \mid \mathcal{H})$ is always conditioned on the hypothesis — it asks "how probable is this data IF this hypothesis were true?"

---

## 14. Connections to Machine Learning & AI

### 14.1 Naïve Bayes Classifier

The naïve Bayes classifier is a direct application of iterated Bayesian updating. Given a document with words $w_1, w_2, \ldots, w_n$, it computes:

$$P(\text{class} \mid w_1, \ldots, w_n) \propto P(\text{class}) \prod_{i=1}^{n} P(w_i \mid \text{class})$$

This is exactly iterated Bayesian updating: multiply the prior by each word's likelihood (assuming conditional independence of words given the class).

### 14.2 Bayesian Inference vs. MLE

| Method | Uses prior? | Gives distribution? | Best when |
|---|---|---|---|
| MLE | No | No (point estimate) | Large data, uninformative prior |
| MAP (Maximum A Posteriori) | Yes | No (point estimate) | Moderate data |
| Full Bayes | Yes | Yes (full posterior) | Small data, informative prior |

MLE is the special case of Bayesian updating with a uniform prior: when all hypotheses start equally likely, the posterior is proportional to the likelihood alone.

### 14.3 Bayesian Updating = Online Learning

Every time you observe a new data point, you can update your posterior. The previous posterior becomes the new prior. This is exactly **online (sequential) learning**: the model updates its beliefs incrementally as new data arrives, without needing to reprocess all past data.

### 14.4 Connection to the Posterior ∝ Likelihood × Prior

In deep learning, regularisation can be given a Bayesian interpretation:

- L2 regularisation (weight decay) ↔ Gaussian prior on weights
- L1 regularisation (LASSO) ↔ Laplace prior on weights

The regularised loss is the negative log of (likelihood × prior), and minimising it corresponds to finding the MAP estimate.

---

## 15. Quick Summary

### The Five Key Terms

| Term | Symbol | Role |
|---|---|---|
| Prior | $P(\mathcal{H})$ | What you believe before seeing data |
| Likelihood | $P(\mathcal{D} \mid \mathcal{H})$ | How consistent the data is with each hypothesis |
| Bayes numerator | $P(\mathcal{D} \mid \mathcal{H}) P(\mathcal{H})$ | Un-normalised update |
| Total prob. of data | $P(\mathcal{D})$ | Normalisation constant; sum of Bayes numerators |
| Posterior | $P(\mathcal{H} \mid \mathcal{D})$ | What you believe after seeing data |

### The Master Formula

$$\boxed{P(\mathcal{H} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H}) \cdot P(\mathcal{H})}{P(\mathcal{D})}}$$

### The Elegant Form

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

### Bayesian Update Table Recipe

1. List all hypotheses
2. Write priors (sum = 1)
3. Write likelihoods (do NOT sum)
4. Compute Bayes numerators = prior × likelihood
5. Sum numerators to get $P(\mathcal{D})$
6. Divide each numerator by $P(\mathcal{D})$ → posteriors (sum = 1)

### Key Properties

- **Iterative:** Posterior from update $k$ becomes prior for update $k+1$
- **Order-independent:** Sequential and batch updates give the same final posterior (under conditional independence)
- **Zero likelihood → zero posterior:** Data impossible under $\mathcal{H}$ eliminates $\mathcal{H}$
- **Extreme priors dominate:** A very strong prior requires very strong data to be overcome
- **Likelihood ≠ posterior:** $P(\mathcal{D} \mid \mathcal{H}) \neq P(\mathcal{H} \mid \mathcal{D})$ — confusing these is the base rate fallacy

### Warning Signs

- ⚠️ If your likelihood column sums to 1, you've done something wrong
- ⚠️ If your posterior > 1 for any hypothesis, you forgot to normalise
- ⚠️ If a 99% accurate test gives you 99% posterior — check your prior

---

*These notes cover all material from MIT 18.05 Class 11 (Bayesian Updating with Discrete Priors), Spring 2022, including both the lecture preparation document and the full in-class problem set with solutions. All examples are reproduced with expanded step-by-step reasoning.*

*Source: MIT OpenCourseWare — https://ocw.mit.edu | 18.05 Introduction to Probability and Statistics, Spring 2022*
