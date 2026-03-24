# MIT 18.05 — Introduction to Probability and Statistics
## Class 16: Choosing Priors & Probability Intervals
### Complete Study Notes | Spring 2022

> **Authors:** Jeremy Orloff and Jonathan Bloom  
> **Source:** MIT OpenCourseWare — 18.05, Spring 2022  
> **Topics Covered:** Choosing priors, flat vs. informed priors, rigid priors, robustness, probability (credible) intervals, constructing priors from subjective probability intervals, two-parameter Bayesian inference

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Introduction: The Art of Choosing a Prior](#2-introduction-the-art-of-choosing-a-prior)
3. [Bayesian vs. Frequentist — A First Look](#3-bayesian-vs-frequentist--a-first-look)
4. [Flat (Uniform) Priors](#4-flat-uniform-priors)
5. [Informed Priors](#5-informed-priors)
6. [Rigid Priors and Their Dangers](#6-rigid-priors-and-their-dangers)
7. [Robustness: More Data Lessens Prior Dependence](#7-robustness-more-data-lessens-prior-dependence)
8. [Two-Parameter Bayesian Inference (2D Tables)](#8-two-parameter-bayesian-inference-2d-tables)
9. [Probability (Credible) Intervals — Full Theory](#9-probability-credible-intervals--full-theory)
10. [Constructing Priors from Subjective Probability Intervals](#10-constructing-priors-from-subjective-probability-intervals)
11. [Worked Examples — Complete Solutions](#11-worked-examples--complete-solutions)
12. [Board Problem — ECMO vs. CVT Full Solution](#12-board-problem--ecmo-vs-cvt-full-solution)
13. [Common Mistakes](#13-common-mistakes)
14. [Quick Reference Summary](#14-quick-reference-summary)

---

## 1. Learning Goals

By the end of Class 16, you should be able to:

1. **Understand** that the choice of prior affects the posterior, and know how to justify your prior choice.
2. **Recognize** that too rigid a prior can prevent learning from data — even an infinite amount of data cannot overcome a prior of zero.
3. **Demonstrate** that more (and better) data reduces dependence on the prior.
4. **Construct** a reasonable prior from prior knowledge of the system.
5. **Define** probability intervals (credible intervals) and find them from a pmf or pdf.
6. **Use** subjective probability intervals to build priors "from scratch."
7. **Perform** two-parameter Bayesian inference using 2D hypothesis tables.

---

## 2. Introduction: The Art of Choosing a Prior

### 2.1 Concept Overview

Up to this point we have always been given a prior distribution. When the prior is known with certainty, Bayesian inference is simply an application of Bayes' theorem — there is no controversy about how to proceed.

The **art of statistics** begins when the prior is not known with certainty. In practice, this is almost always the case: we rarely have a perfectly calibrated prior. This raises the fundamental question:

> **How do we choose a prior when we don't know it for sure?**

There are two major schools of thought: **Bayesian** and **Frequentist**. This class focuses on the Bayesian approach to prior choice.

### 2.2 Bayes' Theorem Recap

$$\underbrace{P(H \mid D)}_{\text{posterior}} = \frac{\underbrace{P(D \mid H)}_{\text{likelihood}} \cdot \underbrace{P(H)}_{\text{prior}}}{\underbrace{P(D)}_{\text{normalizing constant}}}$$

Equivalently:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

### 2.3 Key Principle: Good Data Matters Most

> **Key Principle:** It is always the case that more and better data allows for stronger conclusions and lessens the influence of the prior. The emphasis should be as much on **better data (quality)** as on more data (quantity).

This is arguably the most practically important insight of the class: choose your prior carefully, but don't obsess over it — a well-designed experiment with sufficient high-quality data will eventually wash out any reasonable prior.

---

## 3. Bayesian vs. Frequentist — A First Look

### 3.1 The Core Difference

| Aspect | Bayesian | Frequentist |
|---|---|---|
| Uses a prior? | Yes — always requires $P(H)$ | No — avoids priors entirely |
| What it computes | Posterior $P(H \mid D)$ | Likelihood $P(D \mid H)$ |
| Statement of result | "Parameter has 95% probability of being in $[a,b]$" | "If we repeated this experiment many times, 95% of such intervals would contain the true parameter" |
| Prior dependence | Yes — posterior depends on prior | No prior involved |
| Key challenge | Choosing the prior | Interpreting confidence intervals |

### 3.2 Two Benefits of the Bayesian Approach

**Benefit 1: Direct answer to the right question.**

The posterior probability $P(H \mid D)$ is usually exactly what we want to know. A Bayesian can say:

> "The parameter of interest has probability 0.95 of being between 0.49 and 0.51."

A frequentist confidence interval cannot be interpreted this way — it's a statement about the procedure, not about the specific interval computed.

**Benefit 2: Transparent assumptions.**

The assumptions that go into choosing the prior can be clearly spelled out and critiqued. This makes the inference process auditable and reproducible.

---

## 4. Flat (Uniform) Priors

### 4.1 Concept Overview

A **flat prior** (also called a uniform prior or non-informative prior) assigns equal probability to all hypotheses. It expresses complete ignorance — we have no prior reason to favor one hypothesis over another.

### 4.2 When to Use a Flat Prior

Use a flat prior when:
- You genuinely have no prior information about $\theta$.
- You want a "neutral" starting point that lets the data speak for itself.
- You want to establish a baseline to compare against informed priors.

### 4.3 Intuition

> **Intuition:** A flat prior says "I have no idea." After observing data, the posterior is proportional to the likelihood alone. The posterior therefore peaks where the likelihood is highest — at the maximum likelihood estimate (MLE). In this sense, flat prior Bayesian inference and maximum likelihood estimation give consistent answers.

### 4.4 Worked Example — Dice with Flat Prior

**Setup:** A drawer contains dice with 4, 6, 8, 12, or 20 sides. A die is picked at random and rolled 5 times. Results in order: **4, 2, 4, 7, 5**.

**Flat prior:** No information about the distribution of dice, so $P(H_k) = 1/5$ for each hypothesis $H_k$ (the $k$-sided die was picked).

**Complete sequential update table:**

| Hypothesis | Prior | Lik₁ (roll=4) | Post₁ | Lik₂ (roll=2) | Post₂ | Lik₃ (roll=4) | Post₃ | Lik₄ (roll=7) | Post₄ | Lik₅ (roll=5) | Post₅ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $H_4$ | 1/5 | 1/4 | 0.370 | 1/4 | 0.542 | 1/4 | 0.682 | 0 | **0.000** | 0 | **0.000** |
| $H_6$ | 1/5 | 1/6 | 0.247 | 1/6 | 0.241 | 1/6 | 0.202 | 0 | **0.000** | 1/6 | **0.000** |
| $H_8$ | 1/5 | 1/8 | 0.185 | 1/8 | 0.135 | 1/8 | 0.085 | 1/8 | **0.818** | 1/8 | **0.876** |
| $H_{12}$ | 1/5 | 1/12 | 0.123 | 1/12 | 0.060 | 1/12 | 0.025 | 1/12 | **0.161** | 1/12 | **0.115** |
| $H_{20}$ | 1/5 | 1/20 | 0.074 | 1/20 | 0.022 | 1/20 | 0.005 | 1/20 | **0.021** | 1/20 | **0.009** |

**Step-by-step reasoning:**

- **Rolls 1–3 (values 4, 2, 4):** All dice could produce these values. $H_4$ gains probability (it's most "surprising" that a 4-sided die keeps hitting these low values, so actually $H_4$ gets more weight since 4/4 probability vs 4/20). Wait — we need to think carefully. The likelihood for rolling a 4 on an $n$-sided die is $1/n$. So the 4-sided die gives the highest likelihood. After three rolls, $H_4$ dominates.

- **Roll 4 (value = 7):** This is impossible on a 4-sided die ($P = 0$) and on a 6-sided die ($P = 0$). Both hypotheses are instantly eliminated. This single data point rules out $H_4$ and $H_6$ permanently.

- **Roll 5 (value = 5):** 5 is possible on $H_8$, $H_{12}$, and $H_{20}$. The 8-sided die dominates.

**Final posterior:** $P(H_8 \mid \text{data}) = 0.876$. The evidence strongly favors the 8-sided die.

**Why is $H_8$ not even higher?** Because $H_{12}$ and $H_{20}$ also had non-zero likelihood for all five rolls. But their likelihoods were systematically lower, so they retain only small probability.

---

## 5. Informed Priors

### 5.1 Concept Overview

An **informed prior** incorporates genuine prior knowledge about the system into the prior distribution. This prior knowledge might come from previous experiments, domain expertise, theoretical considerations, or historical data.

### 5.2 How to Build an Informed Prior

**Step 1:** Identify what you already know about the system.

**Step 2:** Translate that knowledge into probability weights over hypotheses.

**Step 3:** Make sure no reasonable hypothesis gets probability exactly 0 (unless it is truly impossible).

**Step 4:** Normalize if necessary.

**Step 5:** Justify your choices clearly so others can critique them.

### 5.3 Worked Example — Dice with Biased Prior

**Same setup as above, but now:** We have reason to believe the drawer contains ten times as many 20-sided dice as any other type.

**Prior:** $P(H_{20}) = 10 \times P(H_k)$ for $k \in \{4,6,8,12\}$. Normalizing:
$4 \times p + 10p = 1 \Rightarrow p = 1/14 \approx 0.071$.

So: $P(H_4) = P(H_6) = P(H_8) = P(H_{12}) = 0.071$, $P(H_{20}) = 0.714$.

**Final posterior (after all 5 rolls):**

| Hypothesis | Prior | Post₅ |
|---|---|---|
| $H_4$ | 0.071 | 0.000 |
| $H_6$ | 0.071 | 0.000 |
| $H_8$ | 0.071 | 0.810 |
| $H_{12}$ | 0.071 | 0.107 |
| $H_{20}$ | 0.714 | 0.083 |

**Observation:** Even with a prior 10× biased toward $H_{20}$, the data still pushes the posterior to favor $H_8$ (0.810). The data overpowers the prior.

### 5.4 What if the Prior Is 100× Biased Toward $H_{20}$?

**Prior:** $P(H_{20}) \approx 0.9615$, all others $\approx 0.0096$.

**Final posterior:**

| Hypothesis | Prior | Post₅ |
|---|---|---|
| $H_4$ | 0.0096 | 0.000 |
| $H_6$ | 0.0096 | 0.000 |
| $H_8$ | 0.0096 | 0.464 |
| $H_{12}$ | 0.0096 | 0.061 |
| $H_{20}$ | 0.9615 | 0.475 |

**Observation:** Now $H_8$ and $H_{20}$ are roughly tied (0.464 vs. 0.475). The extremely strong prior in favor of $H_{20}$ is barely overcome by 5 data points. The posterior gives roughly even odds to the two remaining hypotheses.

> **Lesson:** A very strong (but not rigid) prior requires more data to overcome. With only 5 data points, we cannot fully overcome a 100× prior bias. But notice — the data did shift us substantially from the prior (from 96% $H_{20}$ to 47.5% $H_{20}$). More data would eventually dominate.

---

## 6. Rigid Priors and Their Dangers

### 6.1 Concept Overview

A **rigid prior** assigns probability 1 to one hypothesis and probability 0 to all others. This is the most extreme form of prior and it has severe consequences.

> **Critical Fact:** If $P(H_0) = 0$, then no matter what the data says, $P(H_0 \mid D) = 0$ forever. A zero-probability hypothesis can never be updated away from zero.

This follows directly from Bayes' theorem:

$$P(H_0 \mid D) = \frac{P(D \mid H_0) \cdot P(H_0)}{P(D)} = \frac{P(D \mid H_0) \cdot 0}{P(D)} = 0$$

### 6.2 Case 1: Mild Cognitive Dissonance

**Setup:** Prior $P(H_{20}) = 1$, all other hypotheses have prior 0.

**Update table (all 5 rolls):**

| Hypothesis | Prior | Post₁ | Post₂ | Post₃ | Post₄ | Post₅ |
|---|---|---|---|---|---|---|
| $H_4$ | 0 | 0 | 0 | 0 | 0 | 0 |
| $H_6$ | 0 | 0 | 0 | 0 | 0 | 0 |
| $H_8$ | 0 | 0 | 0 | 0 | 0 | 0 |
| $H_{12}$ | 0 | 0 | 0 | 0 | 0 | 0 |
| $H_{20}$ | 1 | **1** | **1** | **1** | **1** | **1** |

**Result:** No matter what the data is, the posterior is permanently stuck at $H_{20}$. Even rolling a 7 (which is extremely unlikely on a 20-sided die) doesn't change anything.

**Cognitive dissonance:** The belief is internally consistent (7 is possible on a 20-sided die, just unlikely), but the stubborn refusal to update is irrational given the data. This is "mild" dissonance — the model isn't technically broken, it's just completely inflexible.

### 6.3 Case 2: Severe Cognitive Dissonance (Mathematical Breakdown)

**Setup:** Prior $P(H_4) = 1$, all others have prior 0.

Everything goes fine until **Roll 4 (value = 7)**: a 7 is **impossible** on a 4-sided die. So $P(\text{roll}=7 \mid H_4) = 0$.

**The Bayes numerator column becomes all zeros:**

| Hypothesis | Prior | Lik₄ | Bayes Numer₄ | Post₄ |
|---|---|---|---|---|
| $H_4$ | 1 | 0 | **0** | ??? |
| $H_6$ | 0 | 0 | 0 | ??? |
| $H_8$ | 0 | 1/8 | 0 | ??? |
| $H_{12}$ | 0 | 1/12 | 0 | ??? |
| $H_{20}$ | 0 | 1/20 | 0 | ??? |

**Problem:** The entire Bayes numerator column is zero. We cannot normalize — division by zero. The posterior is undefined.

**Practical interpretation:** Your model has broken down. The data is telling you that something you considered impossible has happened. You must either:
1. Admit your model is wrong and revise the prior to allow other hypotheses.
2. Suspect data error or corruption.

> **Lesson:** Never set a prior to exactly 0 for a hypothesis unless it is truly logically or physically impossible. Even a very small prior (like 0.001) is better than zero, because it allows the posterior to eventually recover if that hypothesis turns out to be correct.

---

## 7. Robustness: More Data Lessens Prior Dependence

### 7.1 The Key Theorem (Informal)

> **Informal Theorem (Bayesian Consistency):** Under mild regularity conditions, as the number of data points $n \to \infty$, the posterior converges to a point mass at the true value of $\theta$, regardless of the prior (as long as the prior assigns positive probability to a neighborhood of the true value).

This means: if you use a reasonable prior (no zeros in relevant regions) and collect enough data, **all reasonable priors will give essentially the same posterior.**

### 7.2 Practical Implication

When you report Bayesian results, it is good practice to:

1. **Justify** your prior choice.
2. **Explore** a range of reasonable priors (sensitivity analysis).
3. **Report** if all priors give consistent conclusions — this means the conclusions are **robust** to prior choice.
4. **Flag** if conclusions are sensitive to prior choice — this means you need more data.

### 7.3 Evidence from the Dice Example

| Prior assumption | $P(H_8 \mid \text{5 data points})$ |
|---|---|
| Flat (uniform) | 0.876 |
| $H_{20}$ gets 10× weight | 0.810 |
| $H_{20}$ gets 100× weight | 0.464 |

The flat and mildly biased priors agree very closely. Only the extremely biased (100×) prior gives a noticeably different result, and even then, 5 data points are enough to move the posterior substantially.

> **Conclusion from dice example:** With only 5 data points, flat and "10× biased" priors give very similar posteriors, showing the results are robust to this level of prior uncertainty.

---

## 8. Two-Parameter Bayesian Inference (2D Tables)

### 8.1 Concept Overview

When there are **two unknown parameters** (e.g., two unknown probabilities $\theta_1$ and $\theta_2$), we need a **joint prior** over both parameters simultaneously. The hypothesis space becomes two-dimensional, represented as a grid/table.

### 8.2 The Setup

- **Joint hypothesis:** $(\theta_1, \theta_2)$ — a pair of values
- **Joint prior:** $p(\theta_1, \theta_2)$ — a 2D table of prior probabilities
- **Likelihood:** $p(\text{data} \mid \theta_1, \theta_2)$ — a 2D table
- **Joint posterior:** $p(\theta_1, \theta_2 \mid \text{data}) \propto p(\text{data} \mid \theta_1, \theta_2) \cdot p(\theta_1, \theta_2)$

**The update rule is identical to the 1D case** — multiply entry by entry, then normalize the entire table to sum to 1.

### 8.3 Why 2D?

In many real problems, two separate treatments, populations, or processes each have their own unknown parameter. Examples:
- $\theta_E$ = ECMO survival probability, $\theta_C$ = CVT survival probability (see Board Problem)
- $\theta_S$ = malaria probability for sickle-cell carriers, $\theta_N$ = malaria probability for non-carriers
- $p_1$ = click rate for ad version A, $p_2$ = click rate for ad version B (A/B testing)

### 8.4 Computing Event Probabilities from 2D Tables

After computing the joint posterior, we can compute probabilities of events involving both parameters by **summing the relevant cells**:

$$P(\theta_1 > \theta_2 \mid \text{data}) = \sum_{\{(i,j):\, \theta_1^{(i)} > \theta_2^{(j)}\}} p(\theta_1^{(i)}, \theta_2^{(j)} \mid \text{data})$$

$$P(\theta_1 - \theta_2 \geq c \mid \text{data}) = \sum_{\{(i,j):\, \theta_1^{(i)} - \theta_2^{(j)} \geq c\}} p(\theta_1^{(i)}, \theta_2^{(j)} \mid \text{data})$$

This is one of the great powers of Bayesian inference — you can directly compute the posterior probability of any event or comparison you care about.

---

### 8.5 Full Example — Malaria and the Sickle-Cell Gene

**Background:**

By the 1950s, scientists hypothesized that carriers of the sickle-cell gene were more resistant to malaria. In one experiment, 30 African volunteers were injected with malaria: 15 carriers (S) and 15 non-carriers (N).

**Data:**

|  | $D^+$ (developed malaria) | $D^-$ (did not) | Total |
|---|---|---|---|
| S (sickle-cell carrier) | 2 | 13 | 15 |
| N (non-carrier) | 14 | 1 | 15 |
| Total | 16 | 14 | 30 |

**Parameters:**
- $\theta_S$ = probability that an injected carrier develops malaria
- $\theta_N$ = probability that an injected non-carrier develops malaria

**Likelihood:**

$$P(\text{data} \mid \theta_S, \theta_N) = c\,\theta_S^2(1-\theta_S)^{13}\,\theta_N^{14}(1-\theta_N)^1$$

where $c = \binom{15}{2}\binom{15}{14}$ is a binomial coefficient constant that does not affect relative posteriors.

**Hypothesis space:** Each of $\theta_S$ and $\theta_N$ takes values in $\{0, 0.2, 0.4, 0.6, 0.8, 1\}$, giving $6 \times 6 = 36$ joint hypotheses.

**Color coding for hypothesis table:**

| Region | Condition | Meaning |
|---|---|---|
| Light blue (diagonal) | $\theta_S = \theta_N$ | No protection |
| Darker blue (above diagonal) | $\theta_N > \theta_S$ | Some protection |
| Orange (far above diagonal) | $\theta_N - \theta_S \geq 0.6$ | Strong protection |
| White (below diagonal) | $\theta_S > \theta_N$ | Sickle-cell increases risk |

**Key Likelihood Table** (scaled by $100000/c$ for readability):

| $\theta_N \backslash \theta_S$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|
| **1** | 0 | 0 | 0 | 0 | 0 | 0 |
| **0.8** | 0 | **1.93428** | **0.18381** | 0.00213 | 0 | 0 |
| **0.6** | 0 | 0.06893 | 0.00655 | 0.00008 | 0 | 0 |
| **0.4** | 0 | 0.00035 | 0.00003 | 0 | 0 | 0 |
| **0.2** | 0 | 0 | 0 | 0 | 0 | 0 |
| **0** | 0 | 0 | 0 | 0 | 0 | 0 |

**Why are most cells zero?** Because $\theta_S^2(1-\theta_S)^{13}$ is extremely small unless $\theta_S$ is near 0 (2 out of 15 carriers developed malaria). And $\theta_N^{14}(1-\theta_N)$ is extremely small unless $\theta_N$ is near 1 (14 out of 15 non-carriers developed malaria). The combination peaks at $(\theta_S, \theta_N) = (0.2, 0.8)$.

#### 8.5.1 Flat Prior Analysis

**Flat prior:** Each of 36 cells gets $1/36$.

**Posterior to flat prior:**

| $\theta_N \backslash \theta_S$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 | $p(\theta_N \mid \text{data})$ |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| **0.8** | 0 | **0.88075** | **0.08370** | 0.00097 | 0 | 0 | **0.96542** |
| 0.6 | 0 | 0.03139 | 0.00298 | 0.00003 | 0 | 0 | 0.03440 |
| 0.4 | 0 | 0.00016 | 0.00002 | 0 | 0 | 0 | 0.00018 |
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| $p(\theta_S \mid \text{data})$ | 0 | 0.91230 | 0.08670 | 0.00100 | 0 | 0 | 1.00000 |

**Key results from flat prior:**

$$P(\theta_N > \theta_S \mid \text{data}) = \sum_{\text{orange + darker blue}} = 0.99995$$

$$P(\theta_N - \theta_S > 0.6 \mid \text{data}) = \sum_{\text{orange}} = 0.88075$$

**Interpretation:** With the flat prior, it is **essentially certain** (99.995%) that sickle-cell provides some protection, and **highly probable** (88%) that it provides strong protection ($\geq 0.6$ difference).

#### 8.5.2 Informed Prior Analysis

**Motivation:** The experiment was not run without prior context. There was substantial circumstantial evidence that sickle-cell offered some protection.

**Informed prior construction:**
- 24% probability split evenly among 6 diagonal cells (no protection)
- 6% probability split evenly among 15 below-diagonal cells (negative effect — possible but small)
- 70% probability split evenly among 15 above-diagonal cells (some or strong protection)

Each "no protection" cell: $0.24/6 = 0.04000$  
Each "negative effect" cell: $0.06/15 = 0.00400$  
Each "protective" cell: $0.70/15 = 0.04667$

**Informed prior table:**

| $\theta_N \backslash \theta_S$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 | $p(\theta_N)$ |
|---|---|---|---|---|---|---|---|
| 1 | 0.04667 | 0.04667 | 0.04667 | 0.04667 | 0.04667 | **0.04000** | 0.27333 |
| 0.8 | 0.04667 | 0.04667 | 0.04667 | 0.04667 | **0.04000** | 0.00400 | 0.23067 |
| 0.6 | 0.04667 | 0.04667 | 0.04667 | **0.04000** | 0.00400 | 0.00400 | 0.18800 |
| 0.4 | 0.04667 | 0.04667 | **0.04000** | 0.00400 | 0.00400 | 0.00400 | 0.14533 |
| 0.2 | 0.04667 | **0.04000** | 0.00400 | 0.00400 | 0.00400 | 0.00400 | 0.10267 |
| 0 | **0.04000** | 0.00400 | 0.00400 | 0.00400 | 0.00400 | 0.00400 | 0.06000 |
| $p(\theta_S)$ | 0.27333 | 0.23067 | 0.18800 | 0.14533 | 0.10267 | 0.06000 | 1.0 |

*(Bold = diagonal cells where $\theta_N = \theta_S$.)*

**Posterior to informed prior:**

| $\theta_N \backslash \theta_S$ | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 1 | $p(\theta_N \mid \text{data})$ |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| **0.8** | 0 | **0.88076** | **0.08370** | 0.00097 | 0 | 0 | **0.96543** |
| 0.6 | 0 | 0.03139 | 0.00298 | 0.00003 | 0 | 0 | 0.03440 |
| 0.4 | 0 | 0.00016 | 0.00001 | 0 | 0 | 0 | 0.00017 |
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00000 |
| $p(\theta_S \mid \text{data})$ | 0 | 0.91231 | 0.08669 | 0.00100 | 0 | 0 | 1.00000 |

**Key results from informed prior:**

$$P(\theta_N > \theta_S \mid \text{data}) = 0.99996$$

$$P(\theta_N - \theta_S > 0.6 \mid \text{data}) = 0.88076$$

#### 8.5.3 Robustness Conclusion

| Prior | $P(\theta_N > \theta_S)$ | $P(\theta_N - \theta_S > 0.6)$ |
|---|---|---|
| Flat | 0.99995 | 0.88075 |
| Informed | 0.99996 | 0.88076 |

The two posteriors are **nearly identical** despite very different priors. This demonstrates robustness: the data is strong enough that the prior choice doesn't matter.

> **Scientific Conclusion:** The data from this small study (30 subjects) provides very strong evidence that the sickle-cell gene protects against malaria, and strong evidence (88%) that this protection is substantial (reducing malaria probability by at least 0.6).

#### 8.5.4 PDALX — Probability the Difference is At Least X

A powerful visualization tool is the **PDALX curve**: for each value $x$, plot:

$$\text{PDALX}(x) = P(\theta_N - \theta_S \geq x \mid \text{data})$$

This gives a complete picture of how certain we are about the magnitude of the effect.

Key observations from the PDALX curve:
- $P(\theta_N - \theta_S \geq 0) \approx 1.0$ — almost certain some protection exists
- $P(\theta_N - \theta_S \geq 0.4) \approx 1.0$ — almost certain the protection is substantial
- $P(\theta_N - \theta_S \geq 0.6) \approx 0.88$ — high probability of strong protection
- $P(\theta_N - \theta_S \geq 0.8) \approx 0$ — very unlikely the protection is near maximal

---

## 9. Probability (Credible) Intervals — Full Theory

### 9.1 Concept Overview

Once we have a posterior distribution (whether from a pmf or pdf), we need to **summarize** it. One of the most useful summaries is the **probability interval** (also called a **credible interval** in the Bayesian context).

### 9.2 Formal Definition

> **Definition.** A **$p$-probability interval** for parameter $\theta$ is any interval $[a, b]$ such that:
>
> $$P(a \leq \theta \leq b) = p$$
>
> In the **discrete case** (pmf $p(\theta)$):
> $$\sum_{\theta_i \in [a,b]} p(\theta_i) = p$$
>
> In the **continuous case** (pdf $f(\theta)$):
> $$\int_a^b f(\theta)\,d\theta = p$$

**Terminology:** A $0.9$-probability interval is also called a **90% probability interval** or a **90% credible interval**.

> **Contrast with Frequentist:** A Bayesian credible interval has a direct probability interpretation: "There is a 90% probability that $\theta$ lies in $[a,b]$." A frequentist confidence interval does NOT have this interpretation — it means "90% of such intervals constructed by this procedure would contain the true $\theta$."

### 9.3 Non-Uniqueness

> **Important:** The $p$-probability interval for $\theta$ is **not unique.** There are infinitely many intervals $[a,b]$ satisfying $P(a \leq \theta \leq b) = p$.

**Example (Concept Question):** Between the 0.05-quantile and the 0.55-quantile is a 0.5-probability interval. But so is the interval from the 0.25-quantile to the 0.75-quantile.

$$[q_{0.05}, q_{0.55}]: \quad P = 0.55 - 0.05 = 0.50 \checkmark$$
$$[q_{0.25}, q_{0.75}]: \quad P = 0.75 - 0.25 = 0.50 \checkmark$$

### 9.4 Q-Notation for Probability Intervals

Using quantile notation: the $s$-quantile $q_s$ satisfies $P(\theta \leq q_s) = s$.

A $p$-probability interval is any $[q_s, q_t]$ with $t - s = p$:

$$P(q_s \leq \theta \leq q_t) = t - s = p$$

### 9.5 Symmetric Probability Intervals

> **Definition.** A **symmetric $p$-probability interval** is $[q_{(1-p)/2},\; q_{(1+p)/2}]$.
>
> It leaves equal probability $(1-p)/2$ on each side.

For $p = 0.5$: symmetric interval is $[q_{0.25}, q_{0.75}]$ — the interquartile range.  
For $p = 0.9$: symmetric interval is $[q_{0.05}, q_{0.95}]$.  
For $p = 0.68$: symmetric interval is $[q_{0.16}, q_{0.84}]$ — for $\text{N}(0,1)$, this is $[-1, 1]$.

> **Why symmetric?** For unimodal symmetric distributions, the symmetric interval is the **shortest** interval with the given probability. Shorter = more informative = tighter constraint on $\theta$.

### 9.6 Shortest Probability Intervals

For any fixed $p$, the **shortest** $p$-probability interval is the one centered under the **peak (mode)** of the distribution. This is sometimes called the **Highest Posterior Density (HPD) interval**.

**Reasoning:** By centering the interval where the density is highest, we capture the most probability per unit length of interval.

### 9.7 Key Property: Containment Implies Ordering

> **Key Property:** If a $p_1$-probability interval is **fully contained in** a $p_2$-probability interval, then $p_1 < p_2$.
>
> In other words: a contained interval has less probability than the containing interval.

**Note:** A $p_1$-probability interval can be **longer** than a $p_2$-probability interval even when $p_1 < p_2$, if they are in different parts of the distribution. However, if one is contained in the other, the containing one must have more probability.

### 9.8 Concept Question: Shrink or Stretch?

**Question:** To convert an 80% probability interval to a 90% interval, should you shrink it or stretch it?

**Answer: Stretch (make it bigger).**

**Reasoning:** A larger probability requires capturing more of the distribution. You must extend the interval to include more area under the curve. Going from 80% to 90% means including 10 more percentage points of probability, which requires a wider interval.

> **Intuition:** Think of a river: to capture 90% of the fish, you need a wider net than to capture 80% of them.

### 9.9 Probability Intervals for the Normal Distribution

For $\text{N}(\mu, \sigma^2)$, symmetric probability intervals have nice closed forms:

| Probability $p$ | Interval | For $\text{N}(0,1)$ |
|---|---|---|
| 0.50 | $[\mu - 0.674\sigma,\; \mu + 0.674\sigma]$ | $[-0.674, 0.674]$ |
| 0.68 | $[\mu - \sigma,\; \mu + \sigma]$ | $[-1, 1]$ |
| 0.90 | $[\mu - 1.645\sigma,\; \mu + 1.645\sigma]$ | $[-1.645, 1.645]$ |
| 0.95 | $[\mu - 1.960\sigma,\; \mu + 1.960\sigma]$ | $[-1.96, 1.96]$ |
| 0.99 | $[\mu - 2.576\sigma,\; \mu + 2.576\sigma]$ | $[-2.576, 2.576]$ |

**Key observations from the Normal distribution figure:**
1. All blue bars span a 0.68-probability interval; the shortest blue bar runs from $-1$ to $1$ (symmetric).
2. All green bars span a 0.90-probability interval; they are longer than blue because they contain more probability.
3. The shortest bar of each color is the symmetric one.

### 9.10 Probability Intervals for the Beta Distribution

For Beta$(a,b)$, there is no simple closed form; use software (e.g., `qbeta` in R).

**Example:** Beta$(10, 4)$ has mean $10/14 \approx 0.714$.

- The symmetric 0.68-probability interval is shorter than the non-symmetric 0.68-interval located far from the peak.
- Two different $p = 0.68$ intervals can have **very different lengths** depending on where they are placed relative to the mode.

**Key observation:** The interval positioned under the peak of the Beta$(10,4)$ distribution is shorter than one positioned away from the peak, even though both contain 68% of the probability.

### 9.11 How Bayesian Updating Shrinks Probability Intervals

> **Key Result:** After updating from prior $f(\theta)$ to posterior $f(\theta \mid x)$, the $p$-probability interval for the posterior tends to be **shorter** than the $p$-probability interval for the prior.

This formalizes the intuition that data makes us more certain: the posterior distribution is more concentrated than the prior, so we need a smaller interval to capture the same fraction of probability.

---

## 10. Constructing Priors from Subjective Probability Intervals

### 10.1 The Problem

Often we don't have a clean formula for our prior. We have qualitative knowledge: "I think $\theta$ is probably between 0.4 and 0.6" or "I'd be surprised if $\theta$ were above 0.7."

How do we turn such qualitative beliefs into a formal prior distribution?

### 10.2 The Strategy: Match a Distribution to Your Beliefs

**General approach:**
1. Estimate the **median** of $\theta$ (the value you think is equally likely to be above or below).
2. Estimate the **0.5-probability interval** (where you'd bet even odds $\theta$ is inside vs. outside).
3. Estimate the **0.9-probability interval** (the range you're 90% sure contains $\theta$).
4. Find a parametric distribution (e.g., Beta, Normal) whose quantiles match your estimates.

### 10.3 Method 1: Estimating Intervals Directly

**Step 1:** Use your prior evidence to estimate a 90% interval and a 50% interval.

**Step 2:** Find Beta$(a, b)$ parameters (using R's `pbeta` function) such that:
- `pbeta(upper_90, a, b) - pbeta(lower_90, a, b)` $\approx 0.9$
- `pbeta(upper_50, a, b) - pbeta(lower_50, a, b)` $\approx 0.5$

### 10.4 Method 2: Estimating Quantiles Directly

**Step 1:** Estimate the **median** $q_{0.5}$ (the 50th percentile).

**Step 2:** Use "divide and conquer":
- Estimate $q_{0.25}$: the value such that you think $\theta$ is equally likely to be below $q_{0.25}$ or between $q_{0.25}$ and $q_{0.5}$.
- Estimate $q_{0.75}$: similarly, above $q_{0.5}$.

**Step 3:** Find Beta$(a, b)$ with these quartiles using R's `qbeta` function or trial and error.

### 10.5 Worked Example 3 — Building a Prior for the 2013 SC Election

**Setup:** The 2013 special election in South Carolina, pitting Republican Mark Sanford against Democrat Elizabeth Colbert Busch. Let $\theta$ = fraction of voters favoring Sanford.

**Prior evidence:**
- Sanford is a former SC Congressman and Governor.
- In 2009, he resigned after a personal scandal (affair in Argentina).
- In 2013, he won a Republican primary over 15 opponents.
- In the 2012 presidential election, Romney beat Obama 58% to 40% in this district.
- Colbert bump: Busch is Stephen Colbert's sister — a minor boost from Colbert's fanbase.

**Step 1: Construct 0.9 and 0.5 probability intervals.**

*90% interval reasoning:*
- Upper bound: Romney got 58% in this district. Sanford isn't likely to exceed this given his scandals. Set upper bound at 0.65.
- Lower bound: With all the negatives, he could lose badly. Set lower bound at 0.30.
- **0.9 interval: $[0.30, 0.65]$**

*50% interval reasoning:*
- Unlikely Sanford gets more than Romney's 57% → leave 25% probability above 0.57.
- Lower end harder to predict → leave 25% probability below 0.42.
- **0.5 interval: $[0.42, 0.57]$**

**Step 2: Find a Beta distribution matching these intervals.**

Using R:
```r
a <- 11
b <- 12
pbeta(0.65, a, b) - pbeta(0.30, a, b)  # gives 0.91
pbeta(0.57, a, b) - pbeta(0.42, a, b)  # gives 0.52
```

Result: **Beta$(11, 12)$** matches our desired intervals well.
- $P([0.30, 0.65]) = 0.91$ (target: 0.90 ✓)
- $P([0.42, 0.57]) = 0.52$ (target: 0.50 ✓)

**Properties of Beta$(11, 12)$:**
- Mean: $11/23 \approx 0.478$ (reflecting a slight lean toward Sanford losing)
- The distribution is slightly left-skewed around 0.5, reflecting uncertainty but with a slight lean toward Busch.

**Historic outcome:** Sanford won with 54%, Busch received 45.2%. The prior with 50% interval $[0.42, 0.57]$ did capture this outcome (54% is barely outside the 50% interval, as expected).

---

### 10.6 Worked Example 4 — Prior Construction via Quantile Estimation

**Same setup:** Sanford vs. Colbert-Busch, 2013.

**Step 1: Estimate the median.**

The strongest evidence is the 58%–40% Romney victory. But given Sanford's negatives and Busch's Colbert bump, estimate the median at **0.47** (slightly below 50%, meaning we lean toward Busch winning).

**Step 2: Estimate quartiles (divide and conquer).**

- **25th percentile ($q_{0.25}$):** In a district that went 58% for Romney, it's hard to imagine Sanford falling much below 40%. Estimate $q_{0.25} = 0.40$.
- **75th percentile ($q_{0.75}$):** Given his negatives, hard to see him exceeding 58%. Estimate $q_{0.75} = 0.55$.

**Step 3: Find Beta$(a, b)$ matching these quartiles.**

Using R to search (allowing non-integer $a$, $b$):

```r
# Search for a, b such that:
# qbeta(0.25, a, b) ≈ 0.40
# qbeta(0.50, a, b) ≈ 0.47
# qbeta(0.75, a, b) ≈ 0.55
```

Result: **Beta$(9.9, 11.0)$** with actual quantiles:
- $q_{0.25} = 0.399$ (target: 0.40 ✓)
- $q_{0.50} = 0.472$ (target: 0.47 ✓)
- $q_{0.75} = 0.547$ (target: 0.55 ✓)

**Comparison with Method 1:**
Both methods give similar distributions: Beta$(11,12)$ vs. Beta$(9.9, 11.0)$. These are very close — the two elicitation methods are consistent. This gives us confidence in our prior.

**Using this prior for inference:** If we observed poll data, we would update this prior via Bayes' theorem to get a posterior over $\theta$. The posterior probability interval would be narrower than the prior interval, reflecting the additional information from the poll.

---

## 11. Worked Examples — Complete Solutions

### 11.1 Example: Probability Interval Concept (Q-Notation)

**Example 1 (from PDF):** The interval from the 0.05-quantile to the 0.55-quantile is a 0.5-probability interval.

**Solution:** $P(q_{0.05} \leq \theta \leq q_{0.55}) = 0.55 - 0.05 = 0.50$. ✓

**Example 2 (from PDF):** The 0.5-probability intervals $[q_{0.25}, q_{0.75}]$ and $[q_{0.05}, q_{0.55}]$.

**Verification:**
- $[q_{0.25}, q_{0.75}]$: $P = 0.75 - 0.25 = 0.50$ ✓
- $[q_{0.05}, q_{0.55}]$: $P = 0.55 - 0.05 = 0.50$ ✓

Both are valid 50% probability intervals. The first is **symmetric** (0.25 probability left on each side); the second is **asymmetric** (0.05 left on the left, 0.45 left on the right).

**Which is shorter?** For a unimodal distribution, the interval under the peak is shorter. For a symmetric unimodal distribution, the symmetric interval $[q_{0.25}, q_{0.75}]$ is shorter.

---

## 12. Board Problem — ECMO vs. CVT Full Solution

### 12.1 Problem Setup

**Context:** Comparison of two treatments for newborns with severe respiratory failure:
1. **CVT:** Conventional therapy (hyperventilation and drugs)
2. **ECMO:** Extracorporeal membrane oxygenation (invasive procedure)

**Historical data (Michigan, 1983):**
- ECMO: 19/19 survived
- CVT: 0/3 survived

**New data (Harvard randomized study):**
- ECMO: 28/29 survived
- CVT: 6/10 survived

**Parameters:**
- $\theta_E$ = probability that an ECMO baby survives
- $\theta_C$ = probability that a CVT baby survives

**Discrete hypothesis values for both:** $\{0.125, 0.375, 0.625, 0.875\}$

This gives a $4 \times 4 = 16$ joint hypothesis space.

---

### 12.2 Part (a) — Flat Prior Table

With 16 equally likely hypotheses, each gets prior $1/16 = 0.0625$:

| $\theta_C \backslash \theta_E$ | 0.125 | 0.375 | 0.625 | 0.875 |
|---|---|---|---|---|
| **0.125** | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| **0.375** | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| **0.625** | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| **0.875** | 0.0625 | 0.0625 | 0.0625 | 0.0625 |

---

### 12.3 Part (b) — Informed Prior Based on Michigan Data

**Rationale for informed prior:**

*For $\theta_E$:* Michigan data was 19/19 ECMO survivors. This strongly suggests $\theta_E$ is near 1.0. Reasonable to weight the upper two values more heavily. Let's assign 64% of probability weight to $\theta_E \in \{0.625, 0.875\}$ and 36% to $\theta_E \in \{0.125, 0.375\}$.

*For $\theta_C$:* Michigan data was 0/3 CVT survivors. Three observations is not enough to strongly update from uniform. Keep the distribution over $\theta_C$ approximately uniform.

**Unnormalized informed prior:**

| $\theta_C \backslash \theta_E$ | 0.125 | 0.375 | 0.625 | 0.875 |
|---|---|---|---|---|
| **0.125** | 18 | 18 | 32 | 32 |
| **0.375** | 18 | 18 | 32 | 32 |
| **0.625** | 18 | 18 | 32 | 32 |
| **0.875** | 18 | 18 | 32 | 32 |

*(Sum = 4×(18+18+32+32) = 4×100 = 400. Normalized weights: 18/400 = 0.045, 32/400 = 0.08.)*

**Why these numbers?** The ratio 18:32 = 9:16 means $\theta_E \in \{0.625, 0.875\}$ gets about 1.78× more weight than $\theta_E \in \{0.125, 0.375\}$. This reflects the Michigan evidence without being too rigid.

---

### 12.4 Part (c) — Likelihood Table for Harvard Data

**Harvard data:** ECMO: 28 survived, 1 died (out of 29). CVT: 6 survived, 4 died (out of 10).

**Likelihood formula:**

$$\phi(\text{data} \mid \theta_E, \theta_C) = \binom{29}{28}\theta_E^{28}(1-\theta_E)^1 \cdot \binom{10}{6}\theta_C^6(1-\theta_C)^4 = c\,\theta_E^{28}(1-\theta_E)\,\theta_C^6(1-\theta_C)^4$$

where $c = \binom{29}{28}\binom{10}{6} = 29 \times 210 = 6090$.

**Computing each cell** (dropping constant $c$):

For $\theta_E = 0.125$: $(0.125)^{28}(0.875)^1 \approx 6.16 \times 10^{-28} \times 0.875 \approx 5.39 \times 10^{-28}$ — effectively 0.  
For $\theta_E = 0.875$: $(0.875)^{28}(0.125)^1 \approx 0.02099 \times 0.125 \approx 2.62 \times 10^{-3}$ — much larger.

**Likelihood table** (values computed by R):

| $\theta_C \backslash \theta_E$ | 0.125 | 0.375 | 0.625 | 0.875 |
|---|---|---|---|---|
| **0.125** | 6.160e-28 | 1.007e-14 | 9.835e-09 | 4.048e-05 |
| **0.375** | 1.169e-25 | 1.910e-12 | 1.866e-06 | 7.682e-03 |
| **0.625** | 3.247e-25 | 5.306e-12 | 5.184e-06 | 2.134e-02 |
| **0.875** | 3.019e-26 | 4.932e-13 | 4.819e-07 | 1.984e-03 |

**Why do the values concentrate in the upper-right?**

- High $\theta_E$ (near 0.875): 28 out of 29 survived → likelihood peaks at $\theta_E \approx 28/29 \approx 0.966$, but 0.875 is closest in our discrete grid.
- $\theta_C = 0.625$: 6 out of 10 survived → likelihood peaks at $\theta_C = 6/10 = 0.60$, so 0.625 is the closest.

---

### 12.5 Part (d) — Informed Posterior Table

**Computation:** Multiply each entry of the informed prior table by the corresponding entry in the likelihood table. Then normalize the entire table to sum to 1.

**Informed posterior:**

| $\theta_C \backslash \theta_E$ | 0.125 | 0.375 | 0.625 | 0.875 |
|---|---|---|---|---|
| **0.125** | 1.116e-26 | 1.823e-13 | 3.167e-07 | 0.001 |
| **0.375** | 2.117e-24 | 3.460e-11 | 6.010e-05 | 0.2473 |
| **0.625** | 5.882e-24 | 9.612e-11 | 1.669e-04 | 0.6871 |
| **0.875** | 5.468e-25 | 8.935e-12 | 1.552e-05 | 0.0638 |

**The posterior concentrates almost entirely in the column $\theta_E = 0.875$** — reflecting the strong Harvard evidence that ECMO has very high survival rate.

---

### 12.6 Part (e) — Probability ECMO is Better Than CVT

**Compute:** $P(\theta_E > \theta_C \mid \text{Harvard data})$

Sum all cells where $\theta_E > \theta_C$:

The cells where $\theta_E > \theta_C$ are:

| $\theta_C$ | $\theta_E$ values where $\theta_E > \theta_C$ |
|---|---|
| 0.125 | 0.375, 0.625, 0.875 |
| 0.375 | 0.625, 0.875 |
| 0.625 | 0.875 |
| 0.875 | (none) |

Summing from the informed posterior:

$$P(\theta_E > \theta_C) \approx 0 + 0 + 3.167\text{e-}07 + 0.001 + 0 + 6.010\text{e-}05 + 0.2473 + 1.669\text{e-}04 + 0.6871 = \boxed{0.936}$$

(Note: many terms are essentially zero.)

---

### 12.7 Part (f) — Probability Difference is at Least 0.6

**Compute:** $P(\theta_E - \theta_C \geq 0.6 \mid \text{Harvard data})$

Cells where $\theta_E - \theta_C \geq 0.6$:

| $\theta_C$ | $\theta_E$ values satisfying $\theta_E - \theta_C \geq 0.6$ |
|---|---|
| 0.125 | 0.875 (0.875 − 0.125 = 0.75 ≥ 0.6 ✓) |
| 0.375 | 0.875 only (0.875 − 0.375 = 0.5 < 0.6 ✗); wait: 0.625 − 0.375 = 0.25; 0.875 − 0.375 = 0.5. None ≥ 0.6. |

Actually, checking all pairs:
- $0.875 - 0.125 = 0.75 \geq 0.6$ ✓
- All others: $0.875 - 0.375 = 0.5$, $0.875 - 0.625 = 0.25$, $0.625 - 0.125 = 0.5$. None $\geq 0.6$.

So only the cell $(\theta_C = 0.125, \theta_E = 0.875)$ qualifies. Its posterior value is approximately 0.001.

$$P(\theta_E - \theta_C \geq 0.6 \mid \text{Harvard data}) \approx 0.001$$

This is very small! The data says ECMO is probably better, but the specific hypothesis that the difference is at least 0.6 (very large) is not well supported.

---

### 12.8 Comparison: Flat vs. Informed Prior

| Quantity | Flat Prior | Informed Prior |
|---|---|---|
| $P(\theta_E > \theta_C)$ | 0.936 | 0.936 |
| $P(\theta_E - \theta_C \geq 0.6)$ | 0.001 | 0.001 |

**Key finding:** Both priors give **exactly the same answers**. This demonstrates that the Harvard data (28/29 ECMO, 6/10 CVT) is powerful enough to overwhelm the prior — the posterior is robust to prior choice.

> **Conclusion:** The Harvard data provides strong evidence ($p \approx 0.936$) that ECMO is superior to CVT for treating respiratory failure in newborns. The conclusions are robust: flat and informed priors agree precisely.

---

## 13. Common Mistakes

### 13.1 Prior Choice Mistakes

| Mistake | Correction |
|---|---|
| Setting a prior to exactly 0 for a hypothesis you think is "very unlikely" | Never set a prior to 0 unless logically impossible. Use a very small value like 0.001 instead. |
| Using the same flat prior no matter what you know | If you have genuine prior knowledge, use it. An informed prior that correctly reflects knowledge will give better (more accurate) posterior estimates. |
| Making the prior too rigid (too narrow) | A narrow prior requires more data to overcome. If you're wrong about the prior, you'll need far more data to correct your beliefs. |
| Not checking robustness | Always try multiple reasonable priors and check if conclusions change. If they do, report the sensitivity; if they don't, you have evidence for robustness. |

### 13.2 Probability Interval Mistakes

| Mistake | Correction |
|---|---|
| Thinking the $p$-probability interval is unique | There are infinitely many $p$-probability intervals. The symmetric one and the HPD interval are two natural choices. |
| Confusing a 90% credible interval with a 90% confidence interval | A credible interval says "90% probability $\theta$ is here." A confidence interval says "90% of intervals constructed this way contain the true $\theta$" — very different! |
| Thinking a larger $p$ always means a wider interval | Almost always true, but not if the intervals are in different parts of the distribution. What IS true: if interval $A \subset$ interval $B$, then $p_A < p_B$. |
| Computing a 90% interval as $[q_{0.1}, q_{0.9}]$ | That's the symmetric 80% interval. The symmetric 90% interval is $[q_{0.05}, q_{0.95}]$. |

### 13.3 Two-Parameter Bayesian Mistakes

| Mistake | Correction |
|---|---|
| Forgetting to normalize the 2D posterior table | After multiplying prior × likelihood cell by cell, divide every cell by the total sum. |
| Computing $P(\theta_E > \theta_C)$ incorrectly | Sum ALL cells where $\theta_E > \theta_C$, not just the diagonal. |
| Thinking independent parameters means independent columns | The prior can be non-independent (e.g., if you believe $\theta_E > \theta_C$ a priori, the prior should reflect this dependency). |

---

## 14. Quick Reference Summary

### 14.1 Prior Choice Framework

| Situation | Recommended Prior |
|---|---|
| Completely ignorant about $\theta$ | Flat (uniform) prior |
| Have genuine prior information | Informed prior reflecting that knowledge |
| Strong prior belief, want to verify | Use multiple priors and check robustness |
| Hypothesis is logically impossible | Set to 0 (otherwise never zero!) |
| Want to encode uncertainty ranges | Use probability intervals to construct Beta prior |

### 14.2 Probability Intervals

$$p\text{-probability interval:} \quad [a, b] \text{ such that } P(a \leq \theta \leq b) = p$$

$$\text{Symmetric } p\text{-interval:} \quad [q_{(1-p)/2},\; q_{(1+p)/2}]$$

- **Not unique** — many intervals have the same probability.
- **Shortest** interval: center under the mode of the distribution (HPD interval).
- **Bayesian interpretation:** "There is $p$ probability that $\theta \in [a,b]$."
- **After updating:** Posterior interval is shorter than prior interval — data reduces uncertainty.

### 14.3 Building a Prior from Subjective Intervals

**Method 1 (Direct intervals):**
1. Estimate 0.9 interval: $[L_{0.9}, U_{0.9}]$
2. Estimate 0.5 interval: $[L_{0.5}, U_{0.5}]$
3. Find Beta$(a,b)$ using R: `pbeta(U, a, b) - pbeta(L, a, b)`

**Method 2 (Quantiles):**
1. Estimate median $q_{0.5}$
2. Estimate $q_{0.25}$, $q_{0.75}$ (divide and conquer)
3. Find Beta$(a,b)$ using R: `qbeta(0.25, a, b)`, `qbeta(0.75, a, b)`

### 14.4 Two-Parameter Bayesian Inference

$$p(\theta_1, \theta_2 \mid \text{data}) \propto p(\text{data} \mid \theta_1, \theta_2) \cdot p(\theta_1, \theta_2)$$

**Algorithm:**
1. Set up 2D prior table $p(\theta_1, \theta_2)$
2. Compute 2D likelihood table $p(\text{data} \mid \theta_1, \theta_2)$
3. Multiply element-wise
4. Normalize (divide each cell by sum of all cells)
5. Compute event probabilities by summing relevant cells

### 14.5 Key Principles

| Principle | Statement |
|---|---|
| Zero prior = zero posterior | $P(H) = 0 \Rightarrow P(H \mid D) = 0$ always |
| Robustness | More data → less dependence on prior choice |
| Credible interval | Direct probability statement about $\theta$ |
| Posterior shrinks intervals | Data makes us more certain → shorter intervals |
| Prior elicitation | Use intervals/quantiles to construct prior from knowledge |

---

*End of MIT 18.05 Class 16 Study Notes.*  
*Source: MIT OpenCourseWare, https://ocw.mit.edu — 18.05 Introduction to Probability and Statistics, Spring 2022.*
