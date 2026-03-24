# MIT 18.05 — Introduction to Probability and Statistics
## Class 14: Continuous Data with Continuous Priors
### Complete Study Notes | Jeremy Orloff & Jonathan Bloom | Spring 2022

---

> **Note on this reading:** This is a supplementary (non-assigned) reading that completes the Bayesian updating framework by handling the case where **both** hypotheses **and** data are continuous. It also provides a unified review of all three cases: discrete/discrete, continuous/discrete, and continuous/continuous.

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [The Three Cases of Bayesian Updating — A Unified View](#2-the-three-cases-of-bayesian-updating--a-unified-view)
3. [Case 1: Discrete Hypotheses, Discrete Data](#3-case-1-discrete-hypotheses-discrete-data)
4. [Case 2: Continuous Hypotheses, Discrete Data](#4-case-2-continuous-hypotheses-discrete-data)
5. [Case 3: Continuous Hypotheses, Continuous Data](#5-case-3-continuous-hypotheses-continuous-data)
6. [Notational Conventions — The Full Picture](#6-notational-conventions--the-full-picture)
7. [The Normal–Normal Conjugate Model](#7-the-normalnormal-conjugate-model)
8. [Predictive Probabilities with Continuous Data](#8-predictive-probabilities-with-continuous-data)
9. [Common Mistakes](#9-common-mistakes)
10. [Connections to Machine Learning & AI](#10-connections-to-machine-learning--ai)
11. [Quick Summary](#11-quick-summary)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Construct a Bayesian update table** for continuous hypotheses and continuous data.
2. **Recognise the pdf of a normal distribution** and identify its mean and variance from the form of the exponent.
3. **Complete the square** in a quadratic exponent to identify a normal distribution.
4. **Apply the Normal–Normal conjugate update** to find the posterior mean and variance.
5. **Compute prior and posterior predictive pdfs** when both data and hypotheses are continuous.
6. **Navigate all three cases** of Bayesian updating in a unified framework.

---

## 2. The Three Cases of Bayesian Updating — A Unified View

### 2.1 The Full Landscape

Throughout Classes 11–14 we have built up Bayesian updating in three stages:

| Case | Hypotheses | Data | Prior | Likelihood | Posterior |
|---|---|---|---|---|---|
| **1** | Discrete | Discrete | $P(\mathcal{H})$ or $p(\theta)$ | $p(x \mid \theta)$ | $P(\mathcal{H}\mid x)$ or $p(\theta\mid x)$ |
| **2** | Continuous | Discrete | $f(\theta)\,d\theta$ | $p(x \mid \theta)$ | $f(\theta\mid x)\,d\theta$ |
| **3** | Continuous | Continuous | $f(\theta)\,d\theta$ | $\phi(x \mid \theta)\,dx$ | $f(\theta\mid x)\,d\theta$ |

### 2.2 What Changes in Each Step

**Step 1 → 2:** Hypotheses become continuous. Replace the discrete prior pmf $p(\theta)$ with a continuous prior pdf $f(\theta)$. Replace sums over hypotheses with integrals. The likelihood stays as a pmf $p(x \mid \theta)$ (since data is still discrete).

**Step 2 → 3:** Data also becomes continuous. The likelihood is now a pdf $\phi(x \mid \theta)$ rather than a pmf $p(x \mid \theta)$. The likelihood picks up a $dx$ factor. However, since $dx$ is a constant with respect to $\theta$ (the data $x$ is fixed), it cancels in the normalisation and we can often drop it from our tables.

### 2.3 What Never Changes

The logical core of Bayes' theorem remains identical across all three cases:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

The normalising constant (total probability of the data) changes form — sum in Case 1, integral in Cases 2 and 3 — but the structure is the same.

### 2.4 Notation for Data PDFs

When data is continuous, we use $\phi(x \mid \theta)$ for the likelihood (probability density of data $x$ given hypothesis $\theta$). The choice of $\phi$ (Greek $f$) instead of $f$ distinguishes:

- $f(\theta)$: density for the **hypothesis parameter** $\theta$
- $\phi(x \mid \theta)$: density for the **data** $x$ given parameter $\theta$

Many textbooks use $f$ for both, relying on context. We will use $\phi$ for data densities to keep the notation clear.

---

## 3. Case 1: Discrete Hypotheses, Discrete Data

### 3.1 Setup

- Hypotheses: a finite list $\mathcal{H}_1, \ldots, \mathcal{H}_n$ (or equivalently, discrete values $\theta_1, \ldots, \theta_n$)
- Data: takes discrete values $x \in \{x_1, x_2, \ldots\}$
- Prior: $P(\mathcal{H}_i)$ or $p(\theta_i)$ — a probability mass function
- Likelihood: $p(x \mid \theta_i)$ — a pmf for data given each hypothesis
- Posterior: $p(\theta_i \mid x) = \frac{p(x \mid \theta_i)\,p(\theta_i)}{p(x)}$

**Total probability of data:**
$$p(x) = \sum_{i} p(x \mid \theta_i)\,p(\theta_i)$$

---

### Example 1 — Discrete Hypotheses, Discrete Data

**Problem:** Data $x$ can take values $-1$ or $1$. There are three hypotheses $A$, $B$, $C$. Priors and likelihoods are given. Data observed: $x_1 = 1$. Find posterior probabilities.

**Prior probabilities:**

| Hypothesis | Prior $P(\mathcal{H})$ |
|---|---|
| $A$ | 0.1 |
| $B$ | 0.3 |
| $C$ | 0.6 |

**Full likelihood table:**

| Hypothesis | $p(x=-1 \mid \mathcal{H})$ | $p(x=1 \mid \mathcal{H})$ |
|---|---|---|
| $A$ | 0.2 | 0.8 |
| $B$ | 0.5 | 0.5 |
| $C$ | 0.7 | 0.3 |

**Step 1 — Extract likelihood column for $x = 1$:**

The observed data $x = 1$ selects the $p(x=1 \mid \mathcal{H})$ column.

**Step 2 — Bayesian update table:**

| Hypothesis | Prior $P(\mathcal{H})$ | Likelihood $p(x=1\mid\mathcal{H})$ | Bayes Numerator | Posterior $P(\mathcal{H}\mid x=1)$ |
|---|---|---|---|---|
| $A$ | 0.1 | 0.8 | $0.1 \times 0.8 = 0.08$ | $0.08/0.41 = 0.195$ |
| $B$ | 0.3 | 0.5 | $0.3 \times 0.5 = 0.15$ | $0.15/0.41 = 0.366$ |
| $C$ | 0.6 | 0.3 | $0.6 \times 0.3 = 0.18$ | $0.18/0.41 = 0.439$ |
| **Total** | **1** | **NO SUM** | $p(x=1) = 0.41$ | **1** |

**Final Answers:**

$$P(A \mid x=1) \approx 0.195, \quad P(B \mid x=1) \approx 0.366, \quad P(C \mid x=1) \approx 0.439$$

**Interpretation:**

Before data: $C$ was most likely (60%). After observing $x=1$, $C$ remains most likely (44%) but has decreased because hypothesis $C$ has the *lowest* likelihood of producing $x=1$ (30%). Hypothesis $A$ has the highest likelihood of producing $x=1$ (80%), which increases its posterior despite its low prior.

This is a perfect illustration of the tug-of-war between prior and likelihood:
- $A$: low prior (10%) but high likelihood (80%) → moderate posterior (19.5%)
- $C$: high prior (60%) but low likelihood (30%) → posterior (43.9%) lower than prior

> **Note on "NO SUM":** The likelihood column does not sum to 1 (0.8 + 0.5 + 0.3 = 1.6 ≠ 1). This is expected — likelihoods are not probabilities over hypotheses. We write "NO SUM" to prevent the mistake of summing this column.

---

## 4. Case 2: Continuous Hypotheses, Discrete Data

### 4.1 Setup

- Hypotheses: continuous parameter $\theta \in [a,b]$
- Data: discrete values $x \in \{x_0, x_1, \ldots\}$
- Prior: $f(\theta)\,d\theta$ — continuous pdf
- Likelihood: $p(x \mid \theta)$ — still a pmf (data is discrete)
- Posterior: $f(\theta \mid x)\,d\theta = \frac{p(x\mid\theta)\,f(\theta)\,d\theta}{p(x)}$

**Total probability of data:**
$$p(x) = \int_a^b p(x \mid \theta)\,f(\theta)\,d\theta$$

This case was covered in depth in Class 13. Example 2 below illustrates it with a Binomial likelihood.

---

### Example 2 — Binomial Likelihood, Continuous Prior

**Problem:** Data $x \sim \text{Binomial}(5, \theta)$ with continuous parameter $\theta \in [0,1]$. Prior: $f(\theta) = 2\theta$. Observe $x = 2$. Find the posterior pdf for $\theta$.

**Full likelihood table (one row for all $\theta$):**

| Hypothesis $\theta$ | $p(x=0\mid\theta)$ | $p(x=1\mid\theta)$ | $p(x=2\mid\theta)$ | $p(x=3\mid\theta)$ | $p(x=4\mid\theta)$ | $p(x=5\mid\theta)$ |
|---|---|---|---|---|---|---|
| $\theta$ | $(1-\theta)^5$ | $5\theta(1-\theta)^4$ | $10\theta^2(1-\theta)^3$ | $10\theta^3(1-\theta)^2$ | $5\theta^4(1-\theta)$ | $\theta^5$ |

**Step 1 — Extract likelihood for $x = 2$:**

$$p(x=2 \mid \theta) = \binom{5}{2}\theta^2(1-\theta)^3 = 10\theta^2(1-\theta)^3$$

**Step 2 — Bayesian update table:**

| Hypothesis | Range | Prior | Likelihood ($x=2$) | Bayes Numerator | Posterior |
|---|---|---|---|---|---|
| $\theta$ | $[0,1]$ | $2\theta\,d\theta$ | $10\theta^2(1-\theta)^3$ | $20\theta^3(1-\theta)^3\,d\theta$ | $f(\theta\mid x=2)\,d\theta$ |
| **Total** | | 1 | NO SUM | $\int_0^1 20\theta^3(1-\theta)^3\,d\theta$ | 1 |

**Step 3 — Compute normalising constant using the Beta integral:**

$$\int_0^1 \theta^3(1-\theta)^3\,d\theta = B(4,4) = \frac{3!\cdot 3!}{7!} = \frac{36}{5040} = \frac{1}{140}$$

So $\int_0^1 20\theta^3(1-\theta)^3\,d\theta = 20 \cdot \frac{1}{140} = \frac{1}{7}$.

**Step 4 — Posterior pdf:**

$$f(\theta \mid x=2) = \frac{20\theta^3(1-\theta)^3}{1/7} = 140\theta^3(1-\theta)^3$$

This is a $\text{Beta}(4, 4)$ distribution (verifiable: $\frac{7!}{3!\,3!} = 140$).

**Verification:** $\int_0^1 140\theta^3(1-\theta)^3\,d\theta = 140 \cdot \frac{1}{140} = 1$ ✓

**Note from the PDF:** The normalising constant in the table equals:
$$\frac{2\binom{5}{2} \cdot 3!\,3!}{7!} = \frac{2 \times 10 \times 6 \times 6}{5040} = \frac{2}{7}$$

Wait — the table in the PDF states the total is $\frac{2\binom{5}{2}\cdot 3!\,3!}{7!}$ which simplifies correctly. The posterior is confirmed as $\frac{7!}{3!\,3!}\theta^3(1-\theta)^3 = 140\theta^3(1-\theta)^3$.

**Interpretation:** Observing 2 successes out of 5 with a prior that favours large $\theta$ produces a posterior $\text{Beta}(4,4)$, which is symmetric around $\theta = 0.5$ (the observed frequency). The prior's upward bias has been partially corrected by data showing a 2/5 success rate.

---

## 5. Case 3: Continuous Hypotheses, Continuous Data

### 5.1 The New Element: Continuous Likelihood

When data $x$ is also continuous, the likelihood is no longer a pmf but a **pdf** $\phi(x \mid \theta)$. The probability that the data falls in a small interval $dx$ around $x$ is $\phi(x \mid \theta)\,dx$.

### 5.2 The Update Table with Both $d\theta$ and $dx$

With both infinitesimals included, the Bayesian update table looks like:

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $f(\theta)\,d\theta$ | $\phi(x\mid\theta)\,dx$ | $\phi(x\mid\theta)\,f(\theta)\,d\theta\,dx$ | $f(\theta\mid x)\,d\theta$ |
| **Total** | 1 | NO SUM | $\phi(x)\,dx = \left(\int\phi(x\mid\theta)f(\theta)\,d\theta\right)dx$ | 1 |

### 5.3 Why $dx$ Can Be Dropped

When we compute the posterior $f(\theta \mid x)$, we divide the Bayes numerator by the total probability:

$$f(\theta \mid x)\,d\theta = \frac{\phi(x\mid\theta)\,f(\theta)\,d\theta\,dx}{\phi(x)\,dx}$$

The $dx$ appears in both numerator and denominator and **cancels**:

$$f(\theta \mid x) = \frac{\phi(x\mid\theta)\,f(\theta)}{\phi(x)}$$

Since $x$ is the **fixed, observed** data value, $dx$ is just a constant multiplier throughout the table. Dropping it simplifies notation without any loss of correctness.

**Simplified table (without $dx$):**

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $f(\theta)\,d\theta$ | $\phi(x\mid\theta)$ | $\phi(x\mid\theta)\,f(\theta)\,d\theta$ | $f(\theta\mid x)\,d\theta$ |
| **Total** | 1 | NO SUM | $\phi(x) = \int\phi(x\mid\theta)f(\theta)\,d\theta$ | 1 |

### 5.4 Bayes' Theorem in This Case

$$\boxed{f(\theta \mid x) = \frac{\phi(x\mid\theta)\,f(\theta)}{\phi(x)} = \frac{\phi(x\mid\theta)\,f(\theta)}{\int \phi(x\mid\theta)\,f(\theta)\,d\theta}}$$

And in proportionality form:

$$\boxed{f(\theta \mid x) \propto \phi(x \mid \theta) \cdot f(\theta)}$$

This is structurally identical to all previous cases. Only the interpretation of the likelihood changes: it is now a **density** (not a probability) of the data.

---

## 6. Notational Conventions — The Full Picture

### 6.1 Symbol Choices

The course uses two distinct symbols for pdfs to avoid confusion:

| Symbol | Used for | Example |
|---|---|---|
| $f(\theta)$ | PDF of the **hypothesis parameter** $\theta$ | Prior $f(\theta) = 2\theta$; posterior $f(\theta \mid x)$ |
| $\phi(x \mid \theta)$ | PDF of the **data** $x$ given parameter $\theta$ | Likelihood $\phi(x \mid \theta) = \frac{1}{\sqrt{2\pi}}e^{-(x-\theta)^2/2}$ |

### 6.2 The Philosophical Alternative (One $f$ for Everything)

Many textbooks (and much ML literature) use a single letter $f$ for all densities, relying on the argument letters to disambiguate:

- $f(x \mid \theta)$: density of data $x$ given parameter $\theta$ → the likelihood
- $f(\theta \mid x)$: density of parameter $\theta$ given data $x$ → the posterior

The philosophical justification: think of $f$ as a universal joint density. Then $f(\theta, x)$ is the joint density, $f(x \mid \theta) = f(\theta,x)/f(\theta)$ is the conditional density of $x$ given $\theta$, and so on.

**The danger:** $f(x \mid \theta)$ and $f(\theta \mid x)$ look nearly identical but are completely different. Confusion between them is the base rate fallacy in notation form.

> **Course convention:** Use $\phi$ for data densities and $f$ for parameter densities. This makes the type of each quantity immediately visible.

### 6.3 Complete Notation Reference Table

| Quantity | Discrete $\theta$, Discrete $x$ | Continuous $\theta$, Discrete $x$ | Continuous $\theta$, Continuous $x$ |
|---|---|---|---|
| Prior | $p(\theta)$ | $f(\theta)\,d\theta$ | $f(\theta)\,d\theta$ |
| Likelihood | $p(x\mid\theta)$ | $p(x\mid\theta)$ | $\phi(x\mid\theta)$ [or $\phi(x\mid\theta)dx$] |
| Bayes numerator | $p(x\mid\theta)p(\theta)$ | $p(x\mid\theta)f(\theta)d\theta$ | $\phi(x\mid\theta)f(\theta)d\theta$ |
| Total prob. of data | $\sum_\theta p(x\mid\theta)p(\theta)$ | $\int p(x\mid\theta)f(\theta)d\theta$ | $\int\phi(x\mid\theta)f(\theta)d\theta$ |
| Posterior | $p(\theta\mid x)$ | $f(\theta\mid x)d\theta$ | $f(\theta\mid x)d\theta$ |

---

## 7. The Normal–Normal Conjugate Model

### 7.1 Why Normal–Normal?

The most important example of continuous data with continuous priors is when:
- Data $x \sim \mathcal{N}(\theta, \sigma^2)$ — normal with unknown mean $\theta$, known variance $\sigma^2$
- Prior $\theta \sim \mathcal{N}(\mu_0, \tau^2)$ — normal prior on the unknown mean

The remarkable fact: **the posterior is also normal**. The normal distribution is its own conjugate prior for the normal likelihood (with known variance). This makes the Normal–Normal model analytically tractable and extremely important in practice.

### 7.2 The Completing the Square Technique

The key algebraic tool is **completing the square** in the exponent. Recall:

$$ax^2 + bx + c = a\left(x + \frac{b}{2a}\right)^2 + \left(c - \frac{b^2}{4a}\right)$$

For a normal distribution $\mathcal{N}(\mu, \sigma^2)$, the exponent has the form:

$$-\frac{(x - \mu)^2}{2\sigma^2} = -\frac{1}{2\sigma^2}\left(x^2 - 2\mu x + \mu^2\right)$$

When we multiply two Gaussians (prior × likelihood), the exponents add. The result is a quadratic in $\theta$. Completing the square reveals the mean and variance of the resulting normal distribution.

---

### Example 3 — Normal Prior, Normal Likelihood, Specific Data ($x = 5$)

**Problem:**
- Data: $x = 5$ drawn from $x \sim \mathcal{N}(\theta, 1)$ (known variance $\sigma^2 = 1$)
- Prior: $\theta \sim \mathcal{N}(2, 1)$ (prior mean $\mu_0 = 2$, prior variance $\tau^2 = 1$)

Find the posterior distribution of $\theta$.

**Step 1 — Write down the prior and likelihood pdfs:**

$$f(\theta) = \frac{1}{\sqrt{2\pi}} e^{-(\theta-2)^2/2}$$

$$\phi(x=5 \mid \theta) = \frac{1}{\sqrt{2\pi}} e^{-(5-\theta)^2/2}$$

**Step 2 — Compute the Bayes numerator (prior × likelihood):**

We need to multiply these two Gaussians and simplify:

$$\text{prior} \times \text{likelihood} = \frac{1}{\sqrt{2\pi}} e^{-(\theta-2)^2/2} \cdot \frac{1}{\sqrt{2\pi}} e^{-(5-\theta)^2/2}$$

$$= \frac{1}{2\pi} \exp\!\left(-\frac{(\theta-2)^2 + (5-\theta)^2}{2}\right)$$

**Step 3 — Expand the combined exponent:**

$$(\theta-2)^2 + (5-\theta)^2 = \theta^2 - 4\theta + 4 + 25 - 10\theta + \theta^2 = 2\theta^2 - 14\theta + 29$$

So the exponent argument is:

$$-\frac{2\theta^2 - 14\theta + 29}{2} = -(\theta^2 - 7\theta + 29/2)$$

**Step 4 — Complete the square in $\theta$:**

$$\theta^2 - 7\theta + \frac{29}{2} = \left(\theta - \frac{7}{2}\right)^2 - \frac{49}{4} + \frac{29}{2} = \left(\theta - \frac{7}{2}\right)^2 + \frac{9}{4}$$

Therefore:

$$-(\theta^2 - 7\theta + 29/2) = -\left(\theta - \frac{7}{2}\right)^2 - \frac{9}{4}$$

**Step 5 — Factor out the constant:**

$$\text{prior} \times \text{likelihood} = \frac{1}{2\pi}\exp\!\left(-\frac{9}{4} - \left(\theta - \frac{7}{2}\right)^2\right) = \underbrace{\frac{e^{-9/4}}{2\pi}}_{c_1} \cdot e^{-(\theta - 7/2)^2}$$

**Step 6 — Identify the posterior distribution:**

The Bayes numerator is proportional to $e^{-(\theta - 7/2)^2}$ as a function of $\theta$. Comparing with the general normal form $e^{-(\theta-\mu)^2/(2\sigma^2)}$:

$$e^{-(\theta - 7/2)^2} = e^{-(\theta - 7/2)^2 / (2 \cdot 1/2)}$$

This is the kernel of $\mathcal{N}(\mu, \sigma^2)$ with:
$$\mu = \frac{7}{2} = 3.5, \quad 2\sigma^2 = 1 \implies \sigma^2 = \frac{1}{2}$$

**Bayesian update table:**

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $\frac{1}{\sqrt{2\pi}}e^{-(\theta-2)^2/2}\,d\theta$ | $\frac{1}{\sqrt{2\pi}}e^{-(5-\theta)^2/2}$ | $c_1 e^{-(\theta-7/2)^2}\,d\theta$ | $c_2 e^{-(\theta-7/2)^2}\,d\theta$ |
| Distribution | $\theta \sim \mathcal{N}(2, 1)$ | $x \sim \mathcal{N}(\theta, 1)$ | | $\theta \mid x=5 \sim \mathcal{N}(7/2, 1/2)$ |
| **Total** | 1 | NO SUM | $\phi(x=5)$ | 1 |

**Final Answer:** The posterior is $\theta \mid x=5 \sim \mathcal{N}\!\left(\frac{7}{2},\, \frac{1}{2}\right)$.

**No need to compute $\phi(x=5)$:** Once we recognise the posterior is $\mathcal{N}(7/2, 1/2)$, we know the normalising constant is $\frac{1}{\sqrt{2\pi \cdot 1/2}} = \frac{1}{\sqrt{\pi}}$. We don't need to evaluate the integral.

**Interpretation — The Data Pulls the Prior:**

| | Value |
|---|---|
| Prior mean | 2 |
| Observed data | 5 |
| Posterior mean | 3.5 = (2+5)/2 |
| Prior variance | 1 |
| Posterior variance | 1/2 |

The posterior mean (3.5) is the **average** of the prior mean (2) and the data value (5). This happens because both the prior and the likelihood have the same variance (1). When they are equal, the posterior mean is exactly halfway between the prior mean and the data.

The posterior variance (1/2) is **smaller** than both the prior variance (1) and the data variance (1). Data has reduced our uncertainty about $\theta$.

---

### Example 4 — Normal Prior, Normal Likelihood, General Data $x$

**Problem:** Same setup as Example 3, but with general data $x$ instead of $x=5$.

- Data: $x \sim \mathcal{N}(\theta, 1)$
- Prior: $\theta \sim \mathcal{N}(2, 1)$

**Step 1 — Prior and likelihood:**

$$f(\theta) = \frac{1}{\sqrt{2\pi}} e^{-(\theta-2)^2/2}, \qquad \phi(x \mid \theta) = \frac{1}{\sqrt{2\pi}} e^{-(x-\theta)^2/2}$$

**Step 2 — Combine exponents:**

$$(\theta-2)^2 + (x-\theta)^2 = \theta^2 - 4\theta + 4 + x^2 - 2x\theta + \theta^2 = 2\theta^2 - (4+2x)\theta + (4+x^2)$$

Dividing by 2:

$$\frac{2\theta^2 - (4+2x)\theta + (4+x^2)}{2} = \theta^2 - (2+x)\theta + \frac{4+x^2}{2}$$

**Step 3 — Complete the square in $\theta$:**

$$\theta^2 - (2+x)\theta + \frac{4+x^2}{2} = \left(\theta - \frac{2+x}{2}\right)^2 - \left(\frac{2+x}{2}\right)^2 + \frac{4+x^2}{2}$$

$$= \left(\theta - \left(1 + \frac{x}{2}\right)\right)^2 + \text{terms involving only } x$$

**Step 4 — Bayes numerator:**

$$\text{prior} \times \text{likelihood} = \frac{1}{2\pi}\exp\!\left(-\theta^2 + (2+x)\theta - \frac{4+x^2}{2}\right) = c_1(x) \cdot e^{-(\theta - (1+x/2))^2}$$

where $c_1(x)$ collects all terms that depend only on $x$ (not on $\theta$).

**Step 5 — Identify posterior:**

The Bayes numerator has the form $e^{-(\theta - \mu_1)^2/(2\sigma_1^2)}$ with:

$$\mu_1 = 1 + \frac{x}{2}, \qquad \sigma_1^2 = \frac{1}{2}$$

**Bayesian update table:**

| | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| **Distribution** | $\theta \sim \mathcal{N}(2, 1)$ | $x \sim \mathcal{N}(\theta, 1)$ | $\propto e^{-(\theta-(1+x/2))^2}$ | $\theta\mid x \sim \mathcal{N}(1+x/2,\; 1/2)$ |

**Final Answer:** For general data $x$:

$$\theta \mid x \sim \mathcal{N}\!\left(1 + \frac{x}{2},\; \frac{1}{2}\right)$$

**Verification with $x=5$:** Posterior mean = $1 + 5/2 = 7/2$ ✓. This matches Example 3.

**Interpretation:**

The posterior mean $\mu_1 = 1 + x/2$ is a **weighted combination** of the prior mean and the data. With prior $\mathcal{N}(2,1)$ and likelihood $\mathcal{N}(\theta, 1)$ (equal variances):

$$\mu_1 = 1 + \frac{x}{2} = \frac{1}{2} \cdot 2 + \frac{1}{2} \cdot x = \frac{\mu_0 + x}{2}$$

The posterior mean is exactly the average of the prior mean (2) and the data ($x$), weighted equally because both have variance 1. See Section 7.3 for the general formula.

---

### 7.3 The General Normal–Normal Update Formula

The examples above illustrate a general pattern. For the model:
- Likelihood: $x \sim \mathcal{N}(\theta, \sigma^2)$ with **known** data variance $\sigma^2$
- Prior: $\theta \sim \mathcal{N}(\mu_0, \tau^2)$ with prior mean $\mu_0$ and variance $\tau^2$

The posterior after observing data $x$ is:

$$\boxed{\theta \mid x \sim \mathcal{N}(\mu_1, \tau_1^2)}$$

where:

$$\boxed{\mu_1 = \frac{\frac{1}{\tau^2}\mu_0 + \frac{1}{\sigma^2}x}{\frac{1}{\tau^2} + \frac{1}{\sigma^2}} = \frac{\sigma^2 \mu_0 + \tau^2 x}{\sigma^2 + \tau^2}}$$

$$\boxed{\frac{1}{\tau_1^2} = \frac{1}{\tau^2} + \frac{1}{\sigma^2} \quad \Leftrightarrow \quad \tau_1^2 = \frac{\sigma^2 \tau^2}{\sigma^2 + \tau^2}}$$

**Verification with Examples 3 and 4:** $\mu_0 = 2$, $\tau^2 = 1$, $\sigma^2 = 1$.

$$\mu_1 = \frac{1 \cdot 2 + 1 \cdot x}{1 + 1} = \frac{2 + x}{2} = 1 + \frac{x}{2} \checkmark$$

$$\tau_1^2 = \frac{1 \cdot 1}{1 + 1} = \frac{1}{2} \checkmark$$

### 7.4 Key Insights from the Normal–Normal Formula

**1. Posterior mean as weighted average:**

$$\mu_1 = w_0 \mu_0 + w_x x, \quad \text{where } w_0 = \frac{1/\tau^2}{1/\tau^2 + 1/\sigma^2}, \quad w_x = \frac{1/\sigma^2}{1/\tau^2 + 1/\sigma^2}$$

The weights are proportional to the **precisions** (reciprocals of variances). High-precision information gets more weight.

**2. Data always reduces uncertainty:**

$$\frac{1}{\tau_1^2} = \frac{1}{\tau^2} + \frac{1}{\sigma^2} > \frac{1}{\tau^2}$$

The posterior precision is always greater than the prior precision. Equivalently, $\tau_1^2 < \tau^2$: the posterior variance is always strictly less than the prior variance. Data always reduces uncertainty.

**3. When prior is very uncertain ($\tau^2 \to \infty$, "flat prior"):**

$$\mu_1 \to x, \quad \tau_1^2 \to \sigma^2$$

With an uninformative prior, the posterior is concentrated at the data value — Bayesian and frequentist inference agree.

**4. When prior is very informative ($\tau^2 \to 0$):**

$$\mu_1 \to \mu_0, \quad \tau_1^2 \to 0$$

A very precise prior dominates the data — you need a lot of data to overcome a strong prior.

**5. Data pulls the posterior toward itself:**

If $x > \mu_0$, then $\mu_1 > \mu_0$ — the data pulls the posterior mean toward the data value. The amount of pulling depends on the relative precisions.

### 7.5 Multiple Data Points

For $n$ i.i.d. observations $x_1, x_2, \ldots, x_n$ from $\mathcal{N}(\theta, \sigma^2)$ with prior $\theta \sim \mathcal{N}(\mu_0, \tau^2)$:

$$\theta \mid x_1, \ldots, x_n \sim \mathcal{N}(\mu_n, \tau_n^2)$$

where:

$$\frac{1}{\tau_n^2} = \frac{1}{\tau^2} + \frac{n}{\sigma^2}, \qquad \mu_n = \frac{\frac{1}{\tau^2}\mu_0 + \frac{n}{\sigma^2}\bar{x}}{\frac{1}{\tau^2} + \frac{n}{\sigma^2}}$$

and $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is the sample mean.

As $n \to \infty$: $\mu_n \to \bar{x}$ and $\tau_n^2 \to 0$ — with enough data the posterior concentrates at the sample mean regardless of the prior.

---

## 8. Predictive Probabilities with Continuous Data

### 8.1 Prior Predictive PDF

When the data $x$ is continuous, the "total probability of data" becomes a **probability density** for $x$:

$$\phi(x) = \int \phi(x \mid \theta)\,f(\theta)\,d\theta$$

This is the **prior predictive pdf** for future observations. It gives the marginal density of $x$ before observing any data, averaging over all hypotheses weighted by the prior.

### 8.2 Posterior Predictive PDF

After observing past data $x_1$, the posterior predictive pdf for a new observation $x_2$ is:

$$\phi(x_2 \mid x_1) = \int \phi(x_2 \mid \theta)\,f(\theta \mid x_1)\,d\theta$$

Under the conditional independence assumption $\phi(x_2 \mid \theta, x_1) = \phi(x_2 \mid \theta)$ (i.e., once we know $\theta$, knowing $x_1$ gives no additional information about $x_2$):

$$\boxed{\phi(x_2 \mid x_1) = \int \phi(x_2 \mid \theta)\,f(\theta \mid x_1)\,d\theta}$$

### 8.3 Comparison Across All Cases

| Case | Prior Predictive | Posterior Predictive |
|---|---|---|
| Discrete $x$, Discrete $\theta$ | $p(x) = \sum_\theta p(x\mid\theta)p(\theta)$ | $p(x_2\mid x_1) = \sum_\theta p(x_2\mid\theta)p(\theta\mid x_1)$ |
| Discrete $x$, Continuous $\theta$ | $p(x) = \int p(x\mid\theta)f(\theta)\,d\theta$ | $p(x_2\mid x_1) = \int p(x_2\mid\theta)f(\theta\mid x_1)\,d\theta$ |
| Continuous $x$, Continuous $\theta$ | $\phi(x) = \int \phi(x\mid\theta)f(\theta)\,d\theta$ | $\phi(x_2\mid x_1) = \int \phi(x_2\mid\theta)f(\theta\mid x_1)\,d\theta$ |

The structure is identical across all cases. Only the type of the likelihood (pmf or pdf) and the type of the integral/sum change.

### 8.4 Normal–Normal Predictive Distribution

For the Normal–Normal model with prior $\theta \sim \mathcal{N}(\mu_0, \tau^2)$ and likelihood $x \sim \mathcal{N}(\theta, \sigma^2)$:

**Prior predictive:** $x \sim \mathcal{N}(\mu_0, \sigma^2 + \tau^2)$

The data has the prior mean, but variance is the sum of the two sources of uncertainty: the data variance $\sigma^2$ and the parameter uncertainty $\tau^2$.

**Posterior predictive** (after observing $x_1$): $x_2 \mid x_1 \sim \mathcal{N}(\mu_1, \sigma^2 + \tau_1^2)$

where $\mu_1$ and $\tau_1^2$ are the posterior mean and variance from Section 7.3.

---

## 9. Common Mistakes

### Mistake 1: Treating the continuous likelihood as a probability

**Wrong:** "The likelihood $\phi(x = 5 \mid \theta) = 0.3989$ is the probability that the data equals 5."

**Right:** $\phi(x \mid \theta)$ is a probability **density**. The probability that $x$ is in any specific interval $[a,b]$ is $\int_a^b \phi(x \mid \theta)\,dx$. The probability that $x$ equals any single value is zero.

---

### Mistake 2: Forgetting that $dx$ cancels in the posterior

**Wrong:** Including $dx$ in the posterior expression and being confused about what to integrate over.

**Right:** When data is continuous, include $dx$ in the likelihood and Bayes numerator, but recognise that it cancels when you form the posterior ratio $f(\theta \mid x) = \phi(x\mid\theta)f(\theta)/\phi(x)$. You can drop $dx$ from the table (keeping $d\theta$) for cleaner notation.

---

### Mistake 3: Confusing which variable to integrate over

**Wrong:** Integrating the Bayes numerator over $x$ to find the normalising constant.

**Right:** Always integrate over $\theta$ (the hypothesis parameter). The total probability is computed by summing/integrating over all hypotheses, not over data values:

$$\phi(x) = \int \phi(x \mid \theta)\,f(\theta)\,d\theta \quad \text{(integrate over } \theta\text{)}$$

---

### Mistake 4: Misidentifying the normal distribution from the exponent

**Wrong:** Looking at $e^{-2(\theta - 3)^2}$ and concluding it's a normal with mean 3 and variance 2.

**Right:** The standard normal form is $e^{-(\theta-\mu)^2/(2\sigma^2)}$. So $e^{-2(\theta-3)^2} = e^{-(\theta-3)^2/(2 \cdot 1/4)}$ gives variance $\sigma^2 = 1/4$, not 2.

**Check:** Match your exponent to $-\frac{(\theta - \mu)^2}{2\sigma^2}$ by identifying:
1. The coefficient of $\theta^2$: it should be $-\frac{1}{2\sigma^2}$, giving $\sigma^2$.
2. The coefficient of $\theta$: it should be $\frac{\mu}{\sigma^2}$, giving $\mu$.

---

### Mistake 5: Forgetting to complete the square before trying to identify the posterior

**Wrong:** Leaving the combined exponent in expanded form and being unable to identify the distribution.

**Right:** Always complete the square in $\theta$ after combining the exponent of the prior and likelihood. The squared term $(\ \theta - \text{something}\ )^2$ reveals the posterior mean, and the coefficient of that squared term reveals the posterior variance.

---

### Mistake 6: Confusing $\phi(x \mid \theta)$ with $f(\theta \mid x)$

**Wrong:** Interpreting the likelihood $\phi(5 \mid \theta)$ as "the probability of $\theta$ given data = 5."

**Right:** $\phi(5 \mid \theta)$ is the probability density of observing data $x = 5$ **assuming $\theta$ is the true parameter**. The posterior $f(\theta \mid x=5)$ is the probability density of $\theta$ given the observation. They are related by Bayes' theorem but are very different quantities.

---

## 10. Connections to Machine Learning & AI

### 10.1 Bayesian Linear Regression

Linear regression with unknown weights $\boldsymbol{\theta}$ and Gaussian noise is a direct generalisation of the Normal–Normal model:

$$y = \boldsymbol{\theta}^T\mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

With a Gaussian prior $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$, the posterior is also Gaussian:

$$\boldsymbol{\theta} \mid \mathbf{X}, \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_n, \boldsymbol{\Sigma}_n)$$

This is the multi-dimensional version of Example 4. The posterior mean $\boldsymbol{\mu}_n$ is the MAP/Bayesian estimate of the weights, and it corresponds to **ridge regression** (L2-regularised least squares). The regularisation parameter comes from the ratio $\sigma^2/\tau^2$.

### 10.2 Gaussian Processes

A Gaussian Process (GP) is an infinite-dimensional generalisation of the Normal–Normal model. Instead of a single unknown mean $\theta$, we have an unknown function $f(\mathbf{x})$, with a Gaussian prior over the entire function space. After observing data, the posterior is also a Gaussian Process — this is exact, closed-form Bayesian updating for function estimation. GPs are widely used in:
- Bayesian optimisation (hyperparameter tuning)
- Spatial statistics (kriging)
- Uncertainty quantification in ML

### 10.3 Kalman Filtering

The Kalman filter is the Normal–Normal update applied recursively in time. Given a time series $x_1, x_2, \ldots$ where each observation has Gaussian noise:

1. Prior at time $t$: $\theta_t \sim \mathcal{N}(\mu_t, \tau_t^2)$
2. Observe $x_t \sim \mathcal{N}(\theta_t, \sigma^2)$
3. Posterior = new prior for time $t+1$: $\theta_t \mid x_t \sim \mathcal{N}(\mu_{t+1}, \tau_{t+1}^2)$

The update formulas are exactly those from Section 7.3, applied at each time step. The Kalman filter is the optimal linear estimator and underlies GPS, autonomous vehicles, financial modelling, and robotics.

### 10.4 Variational Inference

In deep learning, exact Bayesian updating is computationally intractable for large models. **Variational inference** approximates the true posterior $f(\theta \mid x)$ with a simpler distribution $q(\theta)$ from a tractable family (often Gaussian). This converts the integration problem into an optimisation problem — the hallmark of modern probabilistic ML.

### 10.5 The Normal–Normal Model as the Building Block

The Normal–Normal conjugate model is the foundation of:
- Ridge regression and Tikhonov regularisation
- Gaussian Processes
- Kalman filtering and smoothing
- Bayesian neural networks (approximately)
- Empirical Bayes methods
- Factor analysis and probabilistic PCA

Understanding Examples 3 and 4 deeply is not just useful for this course — it underlies a significant fraction of all probabilistic machine learning.

---

## 11. Quick Summary

### The Three Cases at a Glance

| Case | Hypotheses | Data | Key Change from Previous |
|---|---|---|---|
| 1 | Discrete | Discrete | Baseline case; sums throughout |
| 2 | Continuous | Discrete | Replace sum with integral over $\theta$; prior/posterior are pdfs |
| 3 | Continuous | Continuous | Likelihood is also a pdf $\phi(x\mid\theta)$; $dx$ cancels in posterior |

### Key Formulas for Case 3

$$f(\theta \mid x) = \frac{\phi(x\mid\theta)\,f(\theta)}{\int\phi(x\mid\theta)\,f(\theta)\,d\theta} \propto \phi(x\mid\theta) \cdot f(\theta)$$

**Prior predictive pdf:** $\phi(x) = \int \phi(x\mid\theta)\,f(\theta)\,d\theta$

**Posterior predictive pdf:** $\phi(x_2\mid x_1) = \int \phi(x_2\mid\theta)\,f(\theta\mid x_1)\,d\theta$

### Normal–Normal Update (the most important special case)

| | Formula |
|---|---|
| Model | $x \sim \mathcal{N}(\theta, \sigma^2)$; prior $\theta \sim \mathcal{N}(\mu_0, \tau^2)$ |
| Posterior mean | $\mu_1 = \dfrac{\sigma^2\mu_0 + \tau^2 x}{\sigma^2 + \tau^2}$ |
| Posterior variance | $\tau_1^2 = \dfrac{\sigma^2\tau^2}{\sigma^2+\tau^2}$ |
| Posterior distribution | $\theta \mid x \sim \mathcal{N}(\mu_1, \tau_1^2)$ |

**Special case ($\sigma^2 = \tau^2 = 1$):** $\mu_1 = ({\mu_0 + x})/{2}$, $\tau_1^2 = 1/2$.

### The Completing the Square Recipe

To find the posterior when prior × likelihood is a product of Gaussians:

1. Write out $f(\theta) \times \phi(x\mid\theta)$ as a single exponential.
2. Collect the exponent into a polynomial in $\theta$: $a\theta^2 + b\theta + c$.
3. Complete the square: $a\theta^2 + b\theta + c = a\left(\theta + \frac{b}{2a}\right)^2 + \left(c - \frac{b^2}{4a}\right)$.
4. The posterior mean is $\mu_1 = -b/(2a)$; the posterior variance is $\sigma_1^2 = -1/(2a)$.
5. Drop the constant terms (they become part of the normalising constant).

### Key Conceptual Points

- **$dx$ cancels in the posterior** when data is continuous — drop it from the likelihood column for cleaner notation, keeping only $d\theta$.
- **Normal prior × Normal likelihood = Normal posterior** — this is the conjugate property.
- **Posterior mean = precision-weighted average** of prior mean and data.
- **Posterior precision = prior precision + data precision** — information adds.
- **Data always reduces uncertainty:** $\tau_1^2 < \tau^2$ always.
- **Predictive formulas are identical in structure** across all three cases — always a weighted average of likelihoods over hypotheses.

### Warning Signs

- ⚠️ If the posterior doesn't integrate to 1 — you forgot to normalise.
- ⚠️ If you identify variance as the coefficient of $(\theta-\mu)^2$ — it should be $-1/(2\sigma^2)$, so variance = $\frac{1}{2 \times |\text{coefficient}|}$.
- ⚠️ If you integrate over $x$ for the normalising constant — always integrate over $\theta$.
- ⚠️ If you use $\phi(x\mid\theta)$ as the posterior — that's the likelihood, not the posterior. Always apply Bayes' theorem.

---

*These notes cover all material from MIT 18.05 Class 14 (Continuous Data with Continuous Priors), Spring 2022. All examples are reproduced with expanded step-by-step reasoning, and the general Normal–Normal formula is derived and interpreted in depth.*

*Source: MIT OpenCourseWare — https://ocw.mit.edu | 18.05 Introduction to Probability and Statistics, Spring 2022*
