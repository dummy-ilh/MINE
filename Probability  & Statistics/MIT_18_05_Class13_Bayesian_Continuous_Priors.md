# MIT 18.05 — Introduction to Probability and Statistics
## Class 13: Bayesian Updating with Continuous Priors
### Complete Study Notes | Jeremy Orloff & Jonathan Bloom | Spring 2022

---

> **How to use these notes:** This is a complete, self-contained reference for Class 13. The central leap here is moving from a *finite* set of hypotheses (as in Classes 11–12) to a *continuous* range of hypotheses represented by a probability density function. Every example from all three PDFs is reproduced with expanded step-by-step reasoning.

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [The Big Leap: From Discrete to Continuous Hypotheses](#2-the-big-leap-from-discrete-to-continuous-hypotheses)
3. [Examples with Continuous Ranges of Hypotheses](#3-examples-with-continuous-ranges-of-hypotheses)
4. [Notational Conventions — The Complete Reference](#4-notational-conventions--the-complete-reference)
5. [Quick Review: PDF and Infinitesimal Probability](#5-quick-review-pdf-and-infinitesimal-probability)
6. [The Law of Total Probability — Continuous Version](#6-the-law-of-total-probability--continuous-version)
7. [Bayes' Theorem — Continuous Version](#7-bayes-theorem--continuous-version)
8. [The Bayesian Update Table for Continuous Priors](#8-the-bayesian-update-table-for-continuous-priors)
9. [Flat (Uniform) Priors](#9-flat-uniform-priors)
10. [Using the Posterior PDF](#10-using-the-posterior-pdf)
11. [Predictive Probabilities with Continuous Priors](#11-predictive-probabilities-with-continuous-priors)
12. [From Discrete to Continuous — The Limiting Argument](#12-from-discrete-to-continuous--the-limiting-argument)
13. [In-Class Problems with Full Solutions](#13-in-class-problems-with-full-solutions)
14. [Common Mistakes](#14-common-mistakes)
15. [Connections to Machine Learning & AI](#15-connections-to-machine-learning--ai)
16. [Quick Summary](#16-quick-summary)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Recognise** when a problem has a continuous range of hypotheses and represent those hypotheses with a prior pdf.
2. **State and apply** the law of total probability for continuous priors and discrete likelihoods.
3. **State and apply** Bayes' theorem to update a prior pdf to a posterior pdf given data.
4. **Compute and interpret** posterior predictive probabilities in the continuous setting.
5. **Navigate all notational conventions** (big/small letters, pmf/pdf, discrete/continuous θ).
6. **Understand the relationship** between the discrete and continuous cases as a limiting process.

---

## 2. The Big Leap: From Discrete to Continuous Hypotheses

### 2.1 What Changes

In Classes 11 and 12, every problem had a **finite list** of hypotheses. For example:
- Five dice: hypotheses $\mathcal{H}_4, \mathcal{H}_6, \mathcal{H}_8, \mathcal{H}_{12}, \mathcal{H}_{20}$
- Three coin types: hypotheses $C_{0.5}, C_{0.6}, C_{0.9}$

Now we consider situations where the hypothesis is a **parameter** $\theta$ that can take **any value in a continuous range**. For example, a coin whose bias $\theta \in [0,1]$ is unknown and could literally be any real number between 0 and 1.

### 2.2 What Stays the Same

The **logic** of Bayesian updating is identical:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

The only mathematical change is: **replace pmfs with pdfs, and sums with integrals**.

| Discrete hypotheses | Continuous hypotheses |
|---|---|
| Prior probability $p(\theta_i)$ | Prior density $f(\theta)\,d\theta$ |
| Likelihood $p(x \mid \theta_i)$ | Likelihood $p(x \mid \theta)$ (same form) |
| Sum: $\sum_i$ | Integral: $\int$ |
| Posterior probability $p(\theta_i \mid x)$ | Posterior density $f(\theta \mid x)\,d\theta$ |

### 2.3 Intuition for the Transition

Think of continuous Bayesian updating as having **infinitely many** hypotheses packed densely along a number line. Just as a histogram with infinitely thin bars becomes a smooth curve (a pdf), the discrete Bayesian update table with infinitely many rows becomes an integral.

The parameter $\theta$ is itself a **random variable** — before seeing data, it has some probability distribution (the prior). After seeing data, it has an updated distribution (the posterior). Both are described by probability density functions.

---

## 3. Examples with Continuous Ranges of Hypotheses

These three standard examples illustrate when continuous priors arise naturally.

### Example 1 — Unknown Success Probability

A system can succeed or fail with unknown probability $p$. We hypothesise that $p$ can be **any value in $[0,1]$**. This is typically modelled as a "bent coin" with unknown probability $\theta$ of heads.

**Why continuous?** Any rational number and any irrational number in $[0,1]$ is a valid probability. There are uncountably many hypotheses.

### Example 2 — Unknown Decay Rate

The lifetime of an isotope follows an exponential distribution $\text{Exp}(\lambda)$ with unknown rate $\lambda$. The mean lifetime $1/\lambda$ can be any positive real number: $\lambda \in (0, \infty)$.

### Example 3 — Unknown Normal Parameters

Gestational length for single births is approximately $\text{Normal}(\mu, \sigma^2)$ with unknown $\mu$ and $\sigma$. With millions of data points we know $\mu \approx 40$ weeks, $\sigma \approx 1$ week, but both parameters are uncertain and can (in principle) take any values in $(-\infty,\infty)$ and $(0,\infty)$ respectively.

> **Key insight:** In all of these, the **parametrised distribution** is the model. Each possible choice of the parameter(s) is a hypothesis. Having a continuous parameter means having uncountably many hypotheses.

---

## 4. Notational Conventions — The Complete Reference

Class 13 introduces a new layer of notation. This section collects everything in one place. Mastering notation is non-negotiable for the rest of the course.

### 4.1 General Notation

| Symbol | Meaning |
|---|---|
| Capital $A$, $B$, $\mathcal{H}$, $\mathcal{D}$ | Events |
| Capital $P(A)$ | Probability of an event |
| Capital $X$, $\Theta$ | Random variables |
| Small $x$ | A value taken by random variable $X$ |
| Small $p(x)$ | Probability mass function (discrete RV) |
| Small $f(x)$ | Probability density function (continuous RV) |
| Connection: | $P(X = x) = p(x)$ for discrete; $P(a \leq X \leq b) = \int_a^b f(x)\,dx$ for continuous |

### 4.2 Bayesian Notation

In the context of Bayesian updating:

| Symbol | Meaning |
|---|---|
| $\mathcal{H}$ or $\mathcal{H}_\theta$ | A hypothesis (event) |
| $\theta$ | The value of a model parameter (hypothesis value); Greek letters for parameters |
| $\mathcal{D}$ | Data as an event |
| $x$ | Data as a value; English letters for data |
| $p(\theta)$ | PMF for a discrete hypothesis parameter |
| $f(\theta)$ | PDF for a continuous hypothesis parameter |
| $f(\theta)\,d\theta$ | Probability that $\theta$ is in an infinitesimal interval around $\theta$ |
| $p(x \mid \theta)$ | Likelihood: probability of data $x$ given parameter $\theta$ |
| $f(\theta \mid x)$ | Posterior density of $\theta$ given data $x$ |

### 4.3 The Master Comparison Table

This table shows the parallel between discrete and continuous Bayesian updating. Memorise this.

| | **Discrete $\theta$** | **Continuous $\theta$** |
|---|---|---|
| Prior | $p(\theta)$ | $f(\theta)\,d\theta$ |
| Likelihood | $p(x \mid \theta)$ | $p(x \mid \theta)$ (same!) |
| Bayes numerator | $p(x \mid \theta)\,p(\theta)$ | $p(x \mid \theta)\,f(\theta)\,d\theta$ |
| Total prob. of data | $\displaystyle\sum_\theta p(x \mid \theta)\,p(\theta)$ | $\displaystyle\int p(x \mid \theta)\,f(\theta)\,d\theta$ |
| Posterior | $p(\theta \mid x)$ | $f(\theta \mid x)\,d\theta$ |

### 4.4 Why Keep the $d\theta$?

The notes emphasise including $d\theta$ in the Bayesian update table. Here is why:

- $f(\theta)$ is a **density** (units: probability per unit $\theta$), not a probability.
- $f(\theta)\,d\theta$ is a **probability** (dimensionless): the probability that $\theta$ is in an infinitesimal interval of width $d\theta$ around $\theta$.
- By keeping $d\theta$ in every entry of the table, every entry is a genuine probability, and our usual probability rules apply cleanly.
- Many textbooks drop $d\theta$ and work directly with densities — this is valid but requires more care.

> **Reminder:** When you write "hypothesis $\theta$" in a continuous update table, you actually mean "the hypothesis that the parameter is in an interval of width $d\theta$ around the value $\theta$."

---

## 5. Quick Review: PDF and Infinitesimal Probability

### 5.1 The Core Idea

For a continuous random variable $X$ with pdf $f(x)$:

$$P(c \leq X \leq d) = \int_c^d f(x)\,dx$$

For an infinitesimally thin slice of width $dx$ around $x$:

$$P(X \in [x, x+dx]) = f(x)\,dx$$

This is the probability $f(x)\,dx$ that appears in the Bayesian update table. The integral over all such slices gives the total probability:

$$\int_{-\infty}^{\infty} f(x)\,dx = 1$$

### 5.2 Density Has Units

If $x$ is measured in centimetres, then $f(x)$ has units of probability/cm. The product $f(x)\,dx$ (probability/cm × cm) is dimensionless — a pure probability.

This is why a pdf can be greater than 1: it is a density, not a probability. Only the area under the curve must equal 1.

### 5.3 The Infinitesimal as a Limiting Sum

The integral $\int_c^d f(x)\,dx$ is the limit of a Riemann sum:

$$\int_c^d f(x)\,dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i)\,\Delta x$$

This is exactly the connection between discrete and continuous Bayesian updating — see Section 12.

---

## 6. The Law of Total Probability — Continuous Version

### 6.1 From Discrete to Continuous

**Discrete version** (finite hypotheses $\theta_1, \ldots, \theta_n$):

$$p(x) = \sum_{i=1}^{n} p(x \mid \theta_i)\,p(\theta_i)$$

**Continuous version** (parameter $\theta$ ranges over $[a,b]$ with pdf $f(\theta)$):

$$\boxed{p(x) = \int_a^b p(x \mid \theta)\,f(\theta)\,d\theta}$$

**Proof by analogy:** Each term $p(x \mid \theta_i)\,p(\theta_i)$ in the discrete sum becomes $p(x \mid \theta)\,f(\theta)\,d\theta$ in the continuous integral. The analogy is exact: $p(\theta_i)$ becomes $f(\theta)\,d\theta$, and the sum becomes an integral.

### 6.2 Interpretation

$p(x)$ is called the **prior predictive probability** of the data $x$. It answers: "Before seeing any data, what is the probability of observing outcome $x$?" It averages the likelihood $p(x \mid \theta)$ over all hypotheses, weighted by the prior $f(\theta)$.

---

### Example 5 — Law of Total Probability with Continuous Prior

**Problem:** A bent coin has unknown probability $\theta$ of heads, where $\theta$ follows the prior pdf $f(\theta) = 2\theta$ on $[0,1]$. What is the prior predictive probability of heads on a single flip?

**Step 1 — Identify the likelihood:**

$$p(x = 1 \mid \theta) = \theta \quad \text{(heads given bias } \theta\text{)}$$

**Step 2 — Apply the law of total probability:**

$$p(x=1) = \int_0^1 p(x=1 \mid \theta)\,f(\theta)\,d\theta = \int_0^1 \theta \cdot 2\theta\,d\theta = \int_0^1 2\theta^2\,d\theta$$

**Step 3 — Evaluate the integral:**

$$= 2 \cdot \frac{\theta^3}{3}\Big|_0^1 = \frac{2}{3}$$

**Final Answer:** $p(\text{heads}) = 2/3$.

**Interpretation:** The prior $f(\theta) = 2\theta$ puts more weight on larger values of $\theta$ (biased toward $\theta$ close to 1). Consequently, the prior predictive probability of heads (2/3) exceeds 1/2. This makes intuitive sense: if you believe the coin is probably biased toward heads, you'd predict heads more than half the time before any flips.

**Verification:** Tails probability = $\int_0^1 (1-\theta) \cdot 2\theta\,d\theta = \int_0^1 (2\theta - 2\theta^2)\,d\theta = 1 - 2/3 = 1/3$. And $2/3 + 1/3 = 1$. ✓

---

## 7. Bayes' Theorem — Continuous Version

### 7.1 The Theorem

> **Theorem (Bayes' Theorem for Continuous Priors):** Let $\theta \in [a,b]$ be a continuous parameter with prior pdf $f(\theta)$, and let $x$ be discrete data with likelihood $p(x \mid \theta)$. Then the posterior probability that $\theta$ is in an infinitesimal interval around $\theta$ given data $x$ is:
>
> $$\boxed{f(\theta \mid x)\,d\theta = \frac{p(x \mid \theta)\,f(\theta)\,d\theta}{\displaystyle\int_a^b p(x \mid \theta)\,f(\theta)\,d\theta}}$$

### 7.2 Derivation

This is not a new theorem — it is the same Bayes' theorem applied to infinitesimal events.

Let $\mathcal{H}$ be the event "parameter $\Theta$ is in an interval of width $d\theta$ around $\theta$", and $\mathcal{D}$ be the event "data equals $x$". Then:

$$P(\mathcal{H}) = f(\theta)\,d\theta, \quad P(\mathcal{D}) = p(x), \quad P(\mathcal{D} \mid \mathcal{H}) = p(x \mid \theta)$$

Standard Bayes' theorem gives:

$$P(\mathcal{H} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H})\,P(\mathcal{H})}{P(\mathcal{D})} = \frac{p(x \mid \theta)\,f(\theta)\,d\theta}{p(x)}$$

The left side is $f(\theta \mid x)\,d\theta$. Dividing both sides by $d\theta$:

$$f(\theta \mid x) = \frac{p(x \mid \theta)\,f(\theta)}{p(x)} = \frac{p(x \mid \theta)\,f(\theta)}{\displaystyle\int_a^b p(x \mid \theta)\,f(\theta)\,d\theta}$$

### 7.3 The Proportionality Form

Since $p(x)$ (the denominator) does not depend on $\theta$, we can write:

$$\boxed{f(\theta \mid x) \propto p(x \mid \theta) \cdot f(\theta)}$$

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

This is the same elegant form as in the discrete case. The normalising constant $p(x)$ ensures the posterior integrates to 1.

### 7.4 Practical Computation

In practice, to find the posterior pdf:

1. Form the **unnormalised posterior**: $g(\theta) = p(x \mid \theta) \cdot f(\theta)$
2. Find the **normalising constant**: $C = \int_a^b g(\theta)\,d\theta$
3. The **posterior pdf** is: $f(\theta \mid x) = g(\theta) / C$

Equivalently: recognise the form of $g(\theta)$ as a known distribution and use the fact that its normalising constant is already known.

---

## 8. The Bayesian Update Table for Continuous Priors

### 8.1 Structure

The continuous update table has a single row for the generic hypothesis $\theta$ (representing all values in the range):

| Hypothesis | Range | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[a,b]$ | $f(\theta)\,d\theta$ | $p(x \mid \theta)$ | $p(x \mid \theta)\,f(\theta)\,d\theta$ | $f(\theta \mid x)\,d\theta$ |
| **Total** | $[a,b]$ | $\int_a^b f(\theta)\,d\theta = 1$ | NO SUM | $p(x) = \int_a^b p(x\mid\theta)f(\theta)\,d\theta$ | $1$ |

### 8.2 Key Differences from the Discrete Table

1. **One row** instead of $n$ rows (the single row represents the entire continuum).
2. **Prior entry** is $f(\theta)\,d\theta$ (probability), not just $f(\theta)$ (density).
3. **Total probability** is computed by integration, not summation.
4. **Posterior entry** is $f(\theta \mid x)\,d\theta$; extract the pdf by dividing by $d\theta$.

---

### Example 6 — Full Bayesian Update (Prior $f(\theta) = 2\theta$, Data = HTT)

**Problem:** Bent coin with unknown bias $\theta$. Prior pdf: $f(\theta) = 2\theta$ on $[0,1]$. Observe sequence: $x = (H, T, T) = (1, 0, 0)$. Find the posterior pdf.

**Step 1 — Likelihood:** Assuming independent flips:

$$p(x = 1,0,0 \mid \theta) = \theta^1(1-\theta)^2 = \theta(1-\theta)^2$$

**Step 2 — Bayesian update table:**

| Hypothesis | Range | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $2\theta\,d\theta$ | $\theta(1-\theta)^2$ | $2\theta^2(1-\theta)^2\,d\theta$ | $f(\theta \mid x)\,d\theta$ |
| **Total** | $[0,1]$ | $1$ | NO SUM | $p(x) = \int_0^1 2\theta^2(1-\theta)^2\,d\theta$ | $1$ |

**Step 3 — Compute total probability of data:**

$$p(x) = \int_0^1 2\theta^2(1-\theta)^2\,d\theta = 2\int_0^1 (\theta^2 - 2\theta^3 + \theta^4)\,d\theta$$

$$= 2\left[\frac{\theta^3}{3} - \frac{\theta^4}{2} + \frac{\theta^5}{5}\right]_0^1 = 2\left(\frac{1}{3} - \frac{1}{2} + \frac{1}{5}\right) = 2 \cdot \frac{10 - 15 + 6}{30} = 2 \cdot \frac{1}{30} = \frac{1}{15}$$

**Step 4 — Posterior pdf:**

$$f(\theta \mid x) = \frac{2\theta^2(1-\theta)^2}{1/15} = 30\theta^2(1-\theta)^2$$

**Verification:** $\int_0^1 30\theta^2(1-\theta)^2\,d\theta = 30 \cdot \frac{1}{30} = 1$ ✓

**Shortcut:** Once you recognise that $f(\theta \mid x) \propto \theta^2(1-\theta)^2$, you only need to find the constant $c$ such that $\int_0^1 c\,\theta^2(1-\theta)^2\,d\theta = 1$. No need to keep track of all constants during intermediate steps.

**Interpretation:**

Before data, the prior $f(\theta) = 2\theta$ was weighted toward $\theta$ near 1 (biased toward heads). After observing 1 head and 2 tails, the posterior shifts: $30\theta^2(1-\theta)^2$ is a symmetric-ish distribution peaked around $\theta = 0.5$, reflecting that 1H + 2T suggests a roughly fair-to-slightly-biased coin.

---

### Example 7 — Flat Prior, Data = Heads (1 flip)

**Problem:** Bent coin, flat prior $f(\theta) = 1$ on $[0,1]$, observe $x = 1$ (heads). Find the posterior.

**Step 1 — Bayesian update table:**

| Hypothesis | Range | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $1 \cdot d\theta$ | $\theta$ | $\theta\,d\theta$ | $f(\theta \mid x)\,d\theta$ |
| **Total** | $[0,1]$ | $1$ | NO SUM | $\int_0^1 \theta\,d\theta = 1/2$ | $1$ |

**Step 2 — Posterior pdf:**

$$f(\theta \mid x=1) = \frac{\theta}{1/2} = 2\theta$$

**Final Answer:** $f(\theta \mid x=1) = 2\theta$ on $[0,1]$.

**Interpretation:**

The flat prior says "all values of $\theta$ are equally plausible." After observing one head, larger values of $\theta$ become more likely (intuitively: if the coin landed heads, it's probably somewhat biased toward heads). The posterior $2\theta$ puts twice as much density at $\theta = 1$ as at $\theta = 0$.

Notice: this posterior ($2\theta$) is exactly the prior used in Example 4 and Example 5! The course deliberately chains these examples to show iterative updating.

---

## 9. Flat (Uniform) Priors

### 9.1 Definition

A **flat** or **uniform prior** assigns equal density to all hypotheses:

$$f(\theta) = \frac{1}{b-a}, \quad \theta \in [a,b]$$

For $\theta \in [0,1]$ this simplifies to $f(\theta) = 1$.

### 9.2 Interpretation

A flat prior expresses **complete ignorance** about the parameter — no value of $\theta$ is considered more or less likely a priori. The posterior is then entirely determined by the likelihood:

$$f(\theta \mid x) \propto p(x \mid \theta) \cdot 1 = p(x \mid \theta)$$

This means the posterior is proportional to the likelihood. The Maximum A Posteriori (MAP) estimate with a flat prior equals the Maximum Likelihood Estimate (MLE).

### 9.3 Warning: Flat Priors Are Not Truly "Uninformative"

Flat priors depend on the parameterisation. A prior that is flat in $\theta$ is not flat in $\theta^2$ or $\log \theta$. This is the motivation for more sophisticated non-informative priors (Jeffreys prior), but that is beyond the scope of this course.

---

## 10. Using the Posterior PDF

### 10.1 What Can You Do with a Posterior PDF?

The posterior pdf $f(\theta \mid x)$ is a complete description of your updated beliefs about $\theta$ given the data. From it you can compute:

- **Posterior probability** that $\theta$ is in any range $[c,d]$:
$$P(c \leq \theta \leq d \mid x) = \int_c^d f(\theta \mid x)\,d\theta$$

- **Posterior mean** (expected value of $\theta$):
$$E[\theta \mid x] = \int_a^b \theta\,f(\theta \mid x)\,d\theta$$

- **Posterior mode** (MAP estimate): the value of $\theta$ that maximises $f(\theta \mid x)$
- **Credible intervals** (Bayesian equivalent of confidence intervals)

---

### Example 8 — Using the Posterior PDF

**Problem:** Same as Example 7: flat prior, data = 1 head. Posterior: $f(\theta \mid x=1) = 2\theta$.

**(a)** Show that with a flat prior, the coin is a priori equally likely to be biased toward heads or tails.

**(b)** After observing one head, what is the posterior probability that the coin is biased toward heads ($\theta > 0.5$)?

**Part (a) — Prior probability:**

$$P(\theta > 0.5) = \int_{0.5}^{1} f(\theta)\,d\theta = \int_{0.5}^{1} 1\,d\theta = \theta\Big|_{0.5}^{1} = 1 - 0.5 = \frac{1}{2}$$

The flat prior assigns equal probability (1/2) to "biased toward heads" and "biased toward tails." This confirms the flat prior is symmetric/uninformative about the direction of bias.

**Part (b) — Posterior probability:**

$$P(\theta > 0.5 \mid x=1) = \int_{0.5}^{1} f(\theta \mid x=1)\,d\theta = \int_{0.5}^{1} 2\theta\,d\theta = \theta^2\Big|_{0.5}^{1} = 1 - 0.25 = \frac{3}{4}$$

**Interpretation:** Observing one head increases the probability that the coin is head-biased from 50% to 75%. One data point is sufficient to meaningfully update our beliefs.

> **Conceptual note:** The probability $P(\theta > 0.5 \mid x)$ is a **posterior probability** — it's a probability about the parameter (hypothesis), computed from the posterior pdf. This is different from the posterior predictive probability of observing heads on the next flip (which is computed separately, as in Section 11).

---

## 11. Predictive Probabilities with Continuous Priors

### 11.1 Prior and Posterior Predictive

The formulas are the continuous analogues of what we did in Class 12:

**Prior predictive probability of outcome $x_{\text{new}}$:**

$$\boxed{p(x_{\text{new}}) = \int_a^b p(x_{\text{new}} \mid \theta)\,f(\theta)\,d\theta}$$

**Posterior predictive probability of outcome $x_{\text{new}}$ given past data $x$:**

$$\boxed{p(x_{\text{new}} \mid x) = \int_a^b p(x_{\text{new}} \mid \theta)\,f(\theta \mid x)\,d\theta}$$

The structure is identical: weighted average of $p(x_{\text{new}} \mid \theta)$ over all hypotheses, but using the prior pdf for the prior predictive and the posterior pdf for the posterior predictive.

---

### Example 9 — Prior and Posterior Predictive (Prior $f(\theta) = 2\theta$, First Flip = H)

**Problem:** Bent coin, prior $f(\theta) = 2\theta$ on $[0,1]$. 

**(a)** Compute the prior predictive probability of heads.

**(b)** Given the first flip was heads, find the posterior predictive probability of heads and tails on the second flip.

**Part (a) — Prior predictive (identical to Example 5):**

$$p(x_1 = 1) = \int_0^1 \theta \cdot 2\theta\,d\theta = \int_0^1 2\theta^2\,d\theta = \frac{2}{3}$$

**Part (b) — Posterior predictive after observing first flip = heads:**

**Step 1:** Find the posterior pdf after observing $x_1 = 1$ (heads).

From the Bayesian update table with prior $f(\theta) = 2\theta$ and likelihood $p(x_1 = 1 \mid \theta) = \theta$:

Bayes numerator: $\theta \cdot 2\theta\,d\theta = 2\theta^2\,d\theta$

Total probability of data: $p(x_1 = 1) = 2/3$ (from Part a)

Posterior pdf: $f(\theta \mid x_1 = 1) = \frac{2\theta^2}{2/3} = 3\theta^2$

**Step 2:** Compute posterior predictive probability of heads on flip 2:

$$p(x_2 = 1 \mid x_1 = 1) = \int_0^1 p(x_2 = 1 \mid \theta)\,f(\theta \mid x_1 = 1)\,d\theta = \int_0^1 \theta \cdot 3\theta^2\,d\theta = \int_0^1 3\theta^3\,d\theta = \frac{3}{4}$$

**Step 3:** Posterior predictive probability of tails:

$$p(x_2 = 0 \mid x_1 = 1) = 1 - \frac{3}{4} = \frac{1}{4}$$

(Alternatively: $\int_0^1 (1-\theta) \cdot 3\theta^2\,d\theta = \int_0^1(3\theta^2 - 3\theta^3)\,d\theta = 1 - 3/4 = 1/4$.)

**Summary:**

| | Prior Predictive | Posterior Predictive (after 1 head) |
|---|---|---|
| $p(\text{heads})$ | 2/3 ≈ 0.667 | 3/4 = 0.750 |
| $p(\text{tails})$ | 1/3 ≈ 0.333 | 1/4 = 0.250 |

**Interpretation:** Observing heads on the first flip increased our belief in a head-biased coin. Consequently, the posterior predictive probability of heads on the next flip rises from 2/3 to 3/4. The data makes our prediction sharper in the direction of the observed outcome.

---

## 12. From Discrete to Continuous — The Limiting Argument

This optional section builds the intuition that the continuous update is literally the limit of a discrete update as the number of hypotheses goes to infinity.

### 12.1 The Construction

**Setup:** Flat prior $f(\theta) = 1$, data = 1 head. We discretise $[0,1]$ into $n$ equal intervals of width $\Delta\theta = 1/n$. Place the hypothesis $\theta_i$ at the centre of the $i$-th interval. The flat prior assigns probability $\Delta\theta = 1/n$ to each hypothesis.

As $n \to \infty$, the discrete update converges to the continuous update.

### 12.2 The 4-Hypothesis Case

Divide $[0,1]$ into 4 intervals, $\Delta\theta = 1/4$. Hypotheses at centres: $\theta = 1/8, 3/8, 5/8, 7/8$.

Prior probability of each: $\Delta\theta = 1/4$.

| Hypothesis $\theta_i$ | Prior $1/4$ | Likelihood $\theta_i$ | Bayes Numerator $(1/4)\theta_i$ | Posterior |
|---|---|---|---|---|
| $1/8$ | $1/4$ | $1/8$ | $(1/4)(1/8) = 1/32$ | $0.0625$ |
| $3/8$ | $1/4$ | $3/8$ | $(1/4)(3/8) = 3/32$ | $0.1875$ |
| $5/8$ | $1/4$ | $5/8$ | $(1/4)(5/8) = 5/32$ | $0.3125$ |
| $7/8$ | $1/4$ | $7/8$ | $(1/4)(7/8) = 7/32$ | $0.4375$ |
| **Total** | **1** | | $\sum = 16/32 = 1/2$ | **1** |

The posterior probability of $\theta_i$ is proportional to $\theta_i$. As a density histogram (dividing probability by interval width $\Delta\theta = 1/4$):

- Posterior density at $\theta = 1/8$: $0.0625 / 0.25 = 0.25$
- Posterior density at $\theta = 3/8$: $0.1875 / 0.25 = 0.75$
- Posterior density at $\theta = 5/8$: $0.3125 / 0.25 = 1.25$
- Posterior density at $\theta = 7/8$: $0.4375 / 0.25 = 1.75$

These are approximately $2\theta_i$ — matching the continuous posterior $f(\theta \mid x=1) = 2\theta$ from Example 7!

### 12.3 The Convergence

As the number of hypotheses grows (4 → 8 → 20 → ∞):

- The density histograms approximate the continuous pdf more closely.
- The discrete sum $\sum_i p(x \mid \theta_i)\,p(\theta_i)$ converges to the integral $\int_0^1 p(x \mid \theta)\,f(\theta)\,d\theta$.
- The discrete posterior pmf converges to the continuous posterior pdf.

**Mathematical statement:** With $n$ hypotheses and $\Delta\theta = 1/n$:

$$\sum_{i=1}^{n} p(x \mid \theta_i) \cdot (1 \cdot \Delta\theta) \;\to\; \int_0^1 p(x \mid \theta) \cdot 1\,d\theta \quad \text{as } n \to \infty$$

This is just a Riemann sum converging to a Riemann integral.

### 12.4 Key Insight

The continuous Bayesian update is not a new formula — it is the natural limiting case of the discrete Bayesian update as the number of equally spaced hypotheses tends to infinity. This should make the continuous formula feel less mysterious.

---

## 13. In-Class Problems with Full Solutions

### Class Example 1 — Three Coin Types, Data = TT

**Problem:** Three coins with $P(\text{heads}) \in \{0.25, 0.5, 0.75\}$, in ratio 1:2:1 (so priors 1/4, 1/2, 1/4). Toss twice and get TT. Find $P(\theta = 0.25 \mid TT)$.

**Step 1 — Likelihoods:**

$$p(TT \mid \theta = 0.25) = (0.75)^2 = 0.5625$$
$$p(TT \mid \theta = 0.5) = (0.5)^2 = 0.25$$
$$p(TT \mid \theta = 0.75) = (0.25)^2 = 0.0625$$

**Step 2 — Bayesian update table:**

| Hypothesis $\theta$ | Prior $p(\theta)$ | Likelihood $p(TT \mid \theta)$ | Bayes Numerator | Posterior |
|---|---|---|---|---|
| 0.25 | 1/4 | $(0.75)^2 = 0.5625$ | $0.5625 \times 0.25 = 0.1406$ | $0.1406/0.281 = 0.500$ |
| 0.50 | 1/2 | $(0.5)^2 = 0.25$ | $0.25 \times 0.5 = 0.125$ | $0.125/0.281 = 0.444$ |
| 0.75 | 1/4 | $(0.25)^2 = 0.0625$ | $0.0625 \times 0.25 = 0.0156$ | $0.0156/0.281 = 0.056$ |
| **Total** | **1** | **NO SUM** | $p(TT) = 0.281$ | **1** |

**Final Answer:** $P(\theta = 0.25 \mid TT) = 0.500$

**Written out in full (as shown in the PDF):**

$$P(\theta_{0.25} \mid TT) = \frac{(0.75)^2 \cdot (1/4)}{(0.75)^2(1/4) + (0.5)^2(1/2) + (0.25)^2(1/4)} = \frac{0.1406}{0.281} = 0.5$$

**Interpretation:** Two tails in a row is strong evidence for the less head-biased coin. The posterior probability of $\theta = 0.25$ jumps from 25% (prior) to 50% (posterior), while $\theta = 0.75$ collapses from 25% to only 5.6%. Two tails is four times more likely under $\theta = 0.25$ than under $\theta = 0.5$: $(0.75)^2 / (0.5)^2 = 0.5625/0.25 = 2.25$, and this ratio times the prior ratio $2:1$ gives the posterior ratio of approximately $2.25 \times 2 / 1 \approx 4.5:1$ after accounting for the third hypothesis.

**Note:** $p(TT)$ is also called the **prior predictive probability of the data** TT.

---

### Problem 1 — Law of Total Probability with Prior PDF

**Problem:** A coin has unknown $\theta$ with prior pdf $f(\theta) = 3\theta^2$ on $[0,1]$.

**(a)** Find the probability of tails on the first toss.

**(b)** Describe a real-world experiment this models.

**Part (a):**

$$p(x=0) = \int_0^1 p(x=0 \mid \theta)\,f(\theta)\,d\theta = \int_0^1 (1-\theta) \cdot 3\theta^2\,d\theta$$

$$= \int_0^1 (3\theta^2 - 3\theta^3)\,d\theta = \left[\theta^3 - \frac{3\theta^4}{4}\right]_0^1 = 1 - \frac{3}{4} = \frac{1}{4}$$

**Final Answer:** $p(\text{tails}) = 1/4$.

**Why is this less than 1/2?** The prior $f(\theta) = 3\theta^2$ is heavily weighted toward large $\theta$ (near 1). Since large $\theta$ means high probability of heads, the prior tilts strongly toward heads. Consequently, tails is expected less than half the time.

**Part (b) — Real-world scenario:**

A medical treatment has unknown probability $\theta$ of success. We believe a priori that it's likely a good treatment (hence the prior $f(\theta) = 3\theta^2$ which is biased toward high $\theta$, i.e., high success rates). The "tails" event corresponds to "treatment fails." If the first patient treated is not cured (tails), we update downward our belief in the treatment's efficacy.

---

### Problem 2 — Bent Coin 1 (Two Sequential Updates)

**Problem:** Bent coin, prior $f(\theta) = 2\theta$ on $[0,1]$.

**(a)** Flip once, get **heads**. Find posterior pdf.

**(b)** Flip again, get **tails**. Update the posterior from (a).

**(c)** Graph prior and both posteriors.

#### Part (a) — After Heads

**Update table:**

| Hyp. | Range | Prior | Likelihood $p(H\mid\theta)$ | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $2\theta\,d\theta$ | $\theta$ | $2\theta^2\,d\theta$ | $f(\theta \mid H)\,d\theta$ |
| **Total** | | 1 | | $\int_0^1 2\theta^2\,d\theta = 2/3$ | 1 |

Posterior pdf: $f(\theta \mid H) = \frac{2\theta^2}{2/3} = 3\theta^2$

**Verification:** $\int_0^1 3\theta^2\,d\theta = \theta^3\big|_0^1 = 1$ ✓

**Interpretation:** After one heads, the distribution shifts even further toward large $\theta$. The prior was $2\theta$ (linear, favoring large $\theta$); the posterior is $3\theta^2$ (quadratic, favoring large $\theta$ even more strongly).

#### Part (b) — After Tails (using posterior from (a) as new prior)

The new prior is $f(\theta) = 3\theta^2$ (the posterior from part a).

**Update table:**

| Hyp. | Range | Prior | Likelihood $p(T\mid\theta)$ | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $3\theta^2\,d\theta$ | $1-\theta$ | $3\theta^2(1-\theta)\,d\theta$ | $f(\theta \mid H, T)\,d\theta$ |
| **Total** | | 1 | | $\int_0^1 3\theta^2(1-\theta)\,d\theta$ | 1 |

Computing the normalising constant:

$$\int_0^1 3\theta^2(1-\theta)\,d\theta = 3\int_0^1(\theta^2 - \theta^3)\,d\theta = 3\left[\frac{1}{3} - \frac{1}{4}\right] = 3 \cdot \frac{1}{12} = \frac{1}{4}$$

Posterior pdf: $f(\theta \mid H, T) = \frac{3\theta^2(1-\theta)}{1/4} = 12\theta^2(1-\theta)$

**Verification:** $\int_0^1 12\theta^2(1-\theta)\,d\theta = 4 \cdot \int_0^1 3\theta^2(1-\theta)\,d\theta = 4 \cdot \frac{1}{4} = 1$ ✓

**Interpretation:**

- Prior $f(\theta) = 2\theta$: gently biased toward large $\theta$ (believed coin is likely head-biased).
- Posterior (a) $f(\theta) = 3\theta^2$: more strongly biased toward large $\theta$ after observing heads.
- Posterior (b) $f(\theta) = 12\theta^2(1-\theta)$: the mode is at $\theta = 2/3$ (solve $d/d\theta[\theta^2(1-\theta)] = 0 \Rightarrow 2\theta - 3\theta^2 = 0 \Rightarrow \theta = 2/3$). This makes sense: we observed 1 head and 1 tail (a 50/50 split), but the prior believed in a head-biased coin, so the posterior peaks above 0.5.

**Critical feature:** Posterior (b) satisfies $f(\theta \mid H,T) = 0$ at $\theta = 1$. This is because observing tails is impossible if $\theta = 1$. Bayes' theorem correctly assigns zero posterior density to values of $\theta$ for which the observed data is impossible.

**Shortcut:** Notice we didn't need to re-normalise after each step from scratch. We updated sequentially: used the posterior from step (a) as the prior for step (b). This is the iterative nature of Bayesian updating.

---

### Problem 3 — Bent Coin 2 (27 Tosses, 15 Heads, 12 Tails)

**Problem:** Bent coin, flat prior $f(\theta) = 1$ on $[0,1]$. Toss 27 times: 15 heads, 12 tails. Find the posterior pdf. Write the normalising constant as an integral but don't evaluate it; call it $T$.

**Step 1 — Likelihood:**

The likelihood of 15 heads and 12 tails in 27 tosses (with known count, not specified order):

$$p(\text{15H, 12T} \mid \theta) = \binom{27}{15}\theta^{15}(1-\theta)^{12}$$

**Step 2 — Bayesian update table:**

| Hyp. | Range | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $1 \cdot d\theta$ | $\binom{27}{15}\theta^{15}(1-\theta)^{12}$ | $\binom{27}{15}\theta^{15}(1-\theta)^{12}\,d\theta$ | $c\,\theta^{15}(1-\theta)^{12}\,d\theta$ |
| **Total** | | 1 | | $T = \int_0^1 \binom{27}{15}\theta^{15}(1-\theta)^{12}\,d\theta$ | 1 |

**Step 3 — Posterior pdf:**

$$f(\theta \mid \text{data}) = c\,\theta^{15}(1-\theta)^{12}, \quad \text{where } c = \frac{\binom{27}{15}}{T}$$

Note: The binomial coefficient $\binom{27}{15}$ is a constant that does not depend on $\theta$, so it cancels in the normalisation. Therefore:

$$f(\theta \mid \text{data}) \propto \theta^{15}(1-\theta)^{12}$$

The normalising constant can be evaluated using the **Beta function**:

$$\int_0^1 \theta^{15}(1-\theta)^{12}\,d\theta = B(16, 13) = \frac{\Gamma(16)\Gamma(13)}{\Gamma(29)} = \frac{15! \cdot 12!}{28!}$$

Therefore: $c = \frac{28!}{15!\,12!}$

**This is a Beta distribution:** $f(\theta \mid \text{data}) = \text{Beta}(16, 13)$ with parameters $\alpha = 15+1 = 16$ and $\beta = 12+1 = 13$.

**General pattern:** With a flat prior and $n$ tosses resulting in $h$ heads and $t = n-h$ tails:

$$f(\theta \mid h\text{ heads}, t\text{ tails}) = \text{Beta}(h+1, t+1) \propto \theta^h(1-\theta)^t$$

**Interpretation:**

The posterior is sharply peaked around $\theta \approx 15/27 \approx 0.556$ (the observed proportion of heads). With 27 data points, the posterior is much more concentrated than the flat prior — the data has substantially narrowed our uncertainty about $\theta$.

The prior (flat = $\text{Beta}(1,1)$) and posterior ($\text{Beta}(16,13)$) are both Beta distributions. This is not a coincidence — the Beta distribution is the **conjugate prior** for Bernoulli/Binomial likelihoods (see Section 15).

---

## 14. Common Mistakes

### Mistake 1: Confusing the pdf with a probability

**Wrong:** "The posterior pdf is $f(\theta \mid x) = 3\theta^2$, so the probability that $\theta = 0.5$ is $3(0.5)^2 = 0.75$."

**Right:** $f(\theta \mid x)$ is a *density*. The probability that $\theta$ takes any single value is zero. The probability that $\theta \in [0.4, 0.6]$ is $\int_{0.4}^{0.6} 3\theta^2\,d\theta$.

---

### Mistake 2: Forgetting to include $d\theta$ in the update table

**Wrong:** Writing prior as $f(\theta)$ and Bayes numerator as $p(x|\theta)\,f(\theta)$.

**Right:** The prior probability is $f(\theta)\,d\theta$, and the Bayes numerator is $p(x|\theta)\,f(\theta)\,d\theta$. Including $d\theta$ ensures all entries are probabilities.

---

### Mistake 3: Summing instead of integrating

**Wrong:** Computing $T = \sum_\theta p(x|\theta)f(\theta)$ when $\theta$ is continuous.

**Right:** $T = \int p(x|\theta)f(\theta)\,d\theta$.

---

### Mistake 4: Forgetting to normalise the posterior

**Wrong:** Reporting $g(\theta) = p(x|\theta)f(\theta)$ as the posterior pdf when $\int g(\theta)\,d\theta \neq 1$.

**Right:** Divide by the normalising constant: $f(\theta \mid x) = g(\theta) / \int g(\theta)\,d\theta$.

---

### Mistake 5: Dropping the binomial coefficient prematurely

**Wrong:** Setting likelihood $= \theta^h(1-\theta)^t$ when the order is not specified (forgetting the $\binom{n}{h}$ factor).

**Right:** When only the counts $h$ and $t$ are known (not the order), the likelihood is $\binom{n}{h}\theta^h(1-\theta)^t$. However, since $\binom{n}{h}$ is a constant that doesn't depend on $\theta$, it cancels when computing the posterior. You can include it or drop it — but be consistent.

---

### Mistake 6: Using the posterior pdf value instead of the posterior probability

**Wrong:** Answering "the probability that the coin is fair ($\theta = 0.5$) after the data is $f(0.5 \mid x) = 3(0.5)^2 = 0.75$."

**Right:** $P(\theta = 0.5 \mid x) = 0$ for a continuous parameter. To ask "how likely is the coin to be approximately fair?" compute $P(0.45 \leq \theta \leq 0.55 \mid x) = \int_{0.45}^{0.55} f(\theta \mid x)\,d\theta$.

---

### Mistake 7: Confusing prior/posterior (for $\theta$) with prior/posterior predictive (for $x$)

**Wrong:** Using the posterior pdf $f(\theta \mid x)$ directly as the probability of future data.

**Right:** Future data probability requires the posterior predictive:
$$p(x_{\text{new}} \mid x) = \int p(x_{\text{new}} \mid \theta)\,f(\theta \mid x)\,d\theta$$

---

## 15. Connections to Machine Learning & AI

### 15.1 Conjugate Priors

The most important practical insight from Class 13 for ML is the concept of **conjugate priors**. A prior is conjugate to a likelihood if the posterior has the same functional form as the prior.

**Example:** The Beta distribution is conjugate to the Binomial likelihood.

- Prior: $f(\theta) = \text{Beta}(\alpha, \beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$
- Likelihood: $p(h \text{ heads}, t \text{ tails} \mid \theta) \propto \theta^h(1-\theta)^t$
- Posterior: $f(\theta \mid h, t) \propto \theta^{\alpha+h-1}(1-\theta)^{\beta+t-1} = \text{Beta}(\alpha+h, \beta+t)$

The update rule is simply: **add counts to parameters**.

| Prior $\text{Beta}(\alpha, \beta)$ | Data: $h$ heads, $t$ tails | Posterior $\text{Beta}(\alpha+h, \beta+t)$ |
|---|---|---|

This is enormously convenient: no integrals need to be evaluated. Conjugate prior pairs underlie much of practical Bayesian inference.

### 15.2 Beta Distribution Parameters as "Pseudo-Counts"

In $\text{Beta}(\alpha, \beta)$:
- $\alpha - 1$ can be thought of as "prior pseudo-heads"
- $\beta - 1$ can be thought of as "prior pseudo-tails"
- The flat prior $f(\theta) = 1 = \text{Beta}(1,1)$ corresponds to 0 pseudo-counts

After observing $h$ heads and $t$ tails with a $\text{Beta}(\alpha, \beta)$ prior:
$$\text{Posterior mean} = \frac{\alpha + h}{\alpha + \beta + h + t}$$

This is the **Bayesian estimate** of $\theta$. With $\alpha = \beta = 1$ (flat prior):

$$\text{Posterior mean} = \frac{h+1}{h+t+2} \approx \frac{h}{n} \text{ for large } n$$

For small $n$, the "+1" terms prevent the extreme estimates of 0 or 1 (Laplace smoothing in NLP).

### 15.3 Maximum A Posteriori (MAP) Estimation

The MAP estimate is the $\theta$ that maximises the posterior pdf:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta f(\theta \mid x) = \arg\max_\theta \left[p(x \mid \theta) \cdot f(\theta)\right]$$

With a flat prior ($f(\theta) = 1$), MAP reduces to MLE:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(x \mid \theta) = \hat{\theta}_{\text{MLE}}$$

With an informative prior, MAP incorporates prior knowledge and is equivalent to regularised likelihood maximisation.

### 15.4 The Bayesian Approach to Neural Networks

Bayesian deep learning places prior distributions over network weights $\mathbf{w}$ and updates them to posteriors given training data $\mathcal{D}$:

$$p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w}) \cdot p(\mathbf{w})$$

This is exactly the continuous Bayesian update from Class 13, but in very high dimensions. The prior $p(\mathbf{w})$ acts as regularisation; the posterior $p(\mathbf{w} \mid \mathcal{D})$ represents uncertainty about the model.

### 15.5 Online Learning and Sequential Updates

A key advantage of Bayesian updating: **it naturally handles data arriving sequentially**. After each new observation, you update the posterior using the previous posterior as the new prior. This is exactly what we did in Problem 2 (two flips) — this principle scales to streaming data, online recommendation systems, and adaptive experiments.

---

## 16. Quick Summary

### The Core Formulas

| Concept | Formula |
|---|---|
| Continuous Bayes' theorem | $f(\theta \mid x) = \dfrac{p(x \mid \theta)\,f(\theta)}{\int p(x \mid \theta)\,f(\theta)\,d\theta}$ |
| Proportionality form | $f(\theta \mid x) \propto p(x \mid \theta) \cdot f(\theta)$ |
| Law of total probability | $p(x) = \int p(x \mid \theta)\,f(\theta)\,d\theta$ |
| Prior predictive | $p(x_{\text{new}}) = \int p(x_{\text{new}} \mid \theta)\,f(\theta)\,d\theta$ |
| Posterior predictive | $p(x_{\text{new}} \mid x) = \int p(x_{\text{new}} \mid \theta)\,f(\theta \mid x)\,d\theta$ |

### Discrete vs. Continuous Parallel

| | **Discrete** | **Continuous** |
|---|---|---|
| Prior | $p(\theta)$ | $f(\theta)\,d\theta$ |
| Bayes numerator | $p(x|\theta)\,p(\theta)$ | $p(x|\theta)\,f(\theta)\,d\theta$ |
| Total prob. of data | $\sum_\theta p(x|\theta)\,p(\theta)$ | $\int p(x|\theta)\,f(\theta)\,d\theta$ |
| Posterior | $p(\theta|x)$ | $f(\theta|x)\,d\theta$ |

### Step-by-Step Update Procedure

1. Write down the prior pdf $f(\theta)$ and its range.
2. Write down the likelihood $p(x \mid \theta)$.
3. Form the unnormalised posterior: $g(\theta) = p(x \mid \theta) \cdot f(\theta)$.
4. Compute the normalising constant: $C = \int g(\theta)\,d\theta$.
5. Posterior pdf: $f(\theta \mid x) = g(\theta)/C$.
6. (Optionally) Use the posterior for predictions, probabilities, MAP estimates.

### Key Conceptual Points

- **$f(\theta)$ is a density, not a probability.** Probabilities require integration over an interval.
- **$f(\theta)\,d\theta$ is a probability.** This infinitesimal probability appears in the update table.
- **The $d\theta$ in the table ensures all entries are probabilities** and Bayes' theorem applies directly.
- **The continuous update is the limit of the discrete update** as the number of hypotheses → ∞.
- **Flat prior + binomial likelihood → Beta posterior.** The Beta family is conjugate to the Binomial.
- **Sequential updates work:** posterior from step $k$ becomes prior for step $k+1$, with the same final result as a batch update.

### Warning Signs

- ⚠️ Posterior pdf value $> 1$ at some point — this is fine! Densities can exceed 1.
- ⚠️ Posterior doesn't integrate to 1 — you forgot to normalise.
- ⚠️ You reported $f(0.5 \mid x)$ as "the probability that $\theta = 0.5$" — wrong; that probability is zero.
- ⚠️ You're summing instead of integrating — whenever $\theta$ is continuous, you must integrate.

---

*These notes cover all material from MIT 18.05 Class 13 (Bayesian Updating with Continuous Priors & Notational Conventions), Spring 2022. All examples from the three documents are reproduced with expanded reasoning.*

*Source: MIT OpenCourseWare — https://ocw.mit.edu | 18.05 Introduction to Probability and Statistics, Spring 2022*
