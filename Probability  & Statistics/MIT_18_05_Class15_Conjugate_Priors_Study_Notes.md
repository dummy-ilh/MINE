# MIT 18.05 — Introduction to Probability and Statistics
## Class 15: Conjugate Priors — Beta and Normal
### Complete Study Notes | Spring 2022

> **Authors:** Jeremy Orloff and Jonathan Bloom  
> **Source:** MIT OpenCourseWare — 18.05, Spring 2022  
> **Topics Covered:** Beta distribution, conjugate priors, Bayesian updating with Beta and Normal priors, posterior predictive probability

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Introduction to Conjugate Priors](#2-introduction-to-conjugate-priors)
3. [The Beta Distribution](#3-the-beta-distribution)
4. [Beta Priors with Binomial Likelihood](#4-beta-priors-with-binomial-likelihood)
5. [Beta Priors with Bernoulli Likelihood](#5-beta-priors-with-bernoulli-likelihood)
6. [Beta Priors with Geometric Likelihood](#6-beta-priors-with-geometric-likelihood)
7. [Bayesian Updating: Continuous Hypotheses and Continuous Data](#7-bayesian-updating-continuous-hypotheses-and-continuous-data)
8. [Normal-Normal Updating](#8-normal-normal-updating)
9. [Multiple Data Points: Normal-Normal](#9-multiple-data-points-normal-normal)
10. [The Likelihood Principle](#10-the-likelihood-principle)
11. [Concept Questions with Deep Explanations](#11-concept-questions-with-deep-explanations)
12. [Board Problems — Full Worked Solutions](#12-board-problems--full-worked-solutions)
13. [Extra Problems](#13-extra-problems)
14. [Common Mistakes](#14-common-mistakes)
15. [Quick Reference Summary](#15-quick-reference-summary)

---

## 1. Learning Goals

By the end of this class, you should be able to:

1. **Know the Beta distribution** — its two-parameter family, its range, its PDF, and its normalization constant.
2. **Understand conjugate priors** — what they are, why they matter, and how they simplify Bayesian inference.
3. **Update a Beta prior** given a Bernoulli, binomial, or geometric likelihood — doing so requires only simple arithmetic on the hyperparameters.
4. **Update a Normal prior** given a Normal likelihood with known variance — using the weighted-average formulas for the posterior mean and posterior variance.

---

## 2. Introduction to Conjugate Priors

### 2.1 Concept Overview

In Bayesian inference, we start with a **prior distribution** $f(\theta)$ representing our belief about an unknown parameter $\theta$ before seeing data. After observing data $x$, we update to a **posterior distribution** $f(\theta|x)$ using Bayes' theorem:

$$f(\theta \mid x) = \frac{\phi(x \mid \theta)\, f(\theta)}{\phi(x)}$$

where $\phi(x \mid \theta)$ is the **likelihood** and $\phi(x) = \int \phi(x \mid \theta) f(\theta)\, d\theta$ is the **normalizing constant** (also called the marginal likelihood or evidence).

In general, computing $\phi(x)$ requires evaluating a potentially hard integral. **Conjugate priors** are a special class of priors that make this calculation trivial.

### 2.2 Intuition

> **Intuition:** A conjugate prior is a prior that is "compatible" with the likelihood in the sense that, after Bayesian updating, the posterior has the **same functional form** as the prior. Only the **parameters** (hyperparameters) change — no integral needs to be computed.

Think of it like this: if your prior belongs to a certain family of distributions, and the likelihood is of a compatible type, then the posterior stays in the same family. The Bayesian update reduces to: "How do I update the parameters of my distribution given the data?"

This makes Bayesian inference **analytically tractable** — a huge practical advantage.

### 2.3 Formal Definition

> **Definition.** Suppose we have data with likelihood $\phi(x \mid \theta)$ depending on a parameter $\theta$. Suppose also that the prior distribution for $\theta$ comes from a parametrized family $\mathcal{F}$. If the posterior $f(\theta \mid x)$ also belongs to $\mathcal{F}$ (only with updated parameters), then we say $\mathcal{F}$ is a family of **conjugate priors** for the likelihood $\phi(x \mid \theta)$.

### 2.4 Key Conjugate Pairs (Preview)

| Prior Family | Likelihood | Posterior Family |
|---|---|---|
| Beta$(a, b)$ | Binomial$(N, \theta)$ | Beta$(a + x,\ b + N - x)$ |
| Beta$(a, b)$ | Bernoulli$(\theta)$ | Beta$(a+1, b)$ or Beta$(a, b+1)$ |
| Beta$(a, b)$ | Geometric$(\theta)$ | Beta$(a + x,\ b + 1)$ |
| Normal$(\mu_\text{prior}, \sigma^2_\text{prior})$ | Normal$(\theta, \sigma^2)$ | Normal$(\mu_\text{post}, \sigma^2_\text{post})$ |

---

## 3. The Beta Distribution

### 3.1 Concept Overview

The **Beta distribution** is a continuous probability distribution on the interval $[0, 1]$. It is parameterized by two positive numbers $a > 0$ and $b > 0$ (called **hyperparameters** in the Bayesian context).

Because its support is $[0, 1]$, it is ideally suited to model probabilities — for example, the unknown probability of heads on a bent coin, or the unknown probability of success in a medical treatment.

### 3.2 Intuition

> **Intuition:** The Beta distribution is an extremely flexible family. By choosing $a$ and $b$:
> - $a = b = 1$: uniform (flat prior — complete ignorance)
> - $a = b > 1$: symmetric, bell-shaped around 0.5 (belief that $\theta \approx 0.5$)
> - $a > b$: skewed right (belief that $\theta$ is large)
> - $a < b$: skewed left (belief that $\theta$ is small)
> - Large $a$ and $b$: very peaked (strong belief about $\theta$)

### 3.3 Formal Definition

> **Definition.** The **Beta$(a, b)$** distribution has PDF:
>
> $$f(\theta) = \frac{(a+b-1)!}{(a-1)!\,(b-1)!}\,\theta^{a-1}(1-\theta)^{b-1}, \quad \theta \in [0,1]$$
>
> for integer values $a \geq 1$, $b \geq 1$. (The general formula uses the Gamma function for non-integer $a, b$, but 18.05 focuses on integers.)

The parameters $a$ and $b$ are called **hyperparameters** because they parameterize the distribution of $\theta$, which is itself the parameter of the data-generating process.

### 3.4 Key Formula: The Normalization Constant

$$c = \frac{(a+b-1)!}{(a-1)!\,(b-1)!}$$

This ensures $\int_0^1 f(\theta)\, d\theta = 1$.

### 3.5 The Critical Observation (Most Important Fact)

> **Key Observation:** If a PDF $f(\theta)$ on $[0,1]$ has the form
>
> $$f(\theta) = c \cdot \theta^{a-1}(1-\theta)^{b-1}$$
>
> for some constant $c$, then $f(\theta)$ **must** be a Beta$(a, b)$ distribution, and the constant is uniquely determined:
>
> $$c = \frac{(a+b-1)!}{(a-1)!\,(b-1)!}$$

**Why this matters:** In Bayesian updating, we multiply prior × likelihood and get a Bayes numerator. If that numerator has the form $c \cdot \theta^{a-1}(1-\theta)^{b-1}$, we immediately know the posterior is Beta$(a,b)$ without computing any integral. The normalization constant is already known!

### 3.6 Mean and Variance of Beta$(a,b)$

$$\text{Mean} = \frac{a}{a+b}$$

$$\text{Variance} = \frac{ab}{(a+b)^2(a+b+1)}$$

### 3.7 Special Case: Flat Prior

The Beta$(1, 1)$ distribution is the **uniform distribution** on $[0,1]$:

$$f(\theta) = \frac{1!}{0!\,0!}\,\theta^0(1-\theta)^0 = 1$$

This represents complete prior ignorance about $\theta$.

---

## 4. Beta Priors with Binomial Likelihood

### 4.1 Setup

Suppose:
- Unknown parameter: $\theta \in [0,1]$ (e.g., probability of heads)
- Prior: $\theta \sim \text{Beta}(a, b)$
- Data: $x$ successes in $N$ trials, so $x \sim \text{Binomial}(N, \theta)$

### 4.2 Update Rule

$$\boxed{\text{Beta}(a,b) \xrightarrow{\text{observe } x \text{ successes, } N-x \text{ failures}} \text{Beta}(a+x,\; b+N-x)}$$

**Mnemonic:** Each success adds 1 to $a$; each failure adds 1 to $b$.

### 4.3 Full Derivation

**Step 1:** Write the prior PDF (dropping the normalizing constant for now):

$$f(\theta) = c_1\,\theta^{a-1}(1-\theta)^{b-1}$$

**Step 2:** Write the likelihood for $x$ successes in $N$ trials:

$$p(x \mid \theta) = \binom{N}{x}\theta^x(1-\theta)^{N-x} = c_2\,\theta^x(1-\theta)^{N-x}$$

**Step 3:** Multiply prior × likelihood (Bayes numerator):

$$f(\theta)\cdot p(x\mid\theta) \propto \theta^{a-1}(1-\theta)^{b-1}\cdot\theta^x(1-\theta)^{N-x} = \theta^{(a+x)-1}(1-\theta)^{(b+N-x)-1}$$

**Step 4:** Recognize this is the kernel of Beta$(a+x,\; b+N-x)$. By our Key Observation, the posterior is:

$$f(\theta \mid x) \sim \text{Beta}(a+x,\; b+N-x)$$

No integral needed!

### 4.4 The Full Bayesian Update Table

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $\text{Beta}(a,b)$ | $\text{Binomial}(N,\theta)$ | — | $\text{Beta}(a+x, b+N-x)$ |
| $\theta$ | $c_1\theta^{a-1}(1-\theta)^{b-1}\,d\theta$ | $c_2\theta^x(1-\theta)^{N-x}$ | $c_3\theta^{a+x-1}(1-\theta)^{b+N-x-1}\,d\theta$ | $c_4\theta^{a+x-1}(1-\theta)^{b+N-x-1}$ |

### 4.5 Worked Example 1 — Bent Coin from Flat Prior

**Problem:** A bent coin has unknown probability $\theta$ of heads. We start with a flat (uniform) prior. We toss the coin 12 times and get 8 heads and 4 tails. Find the posterior PDF.

**Solution:**

**Step 1:** Flat prior = Beta$(1, 1)$, so $a = 1$, $b = 1$.

**Step 2:** Data: $x = 8$ heads, $N - x = 4$ tails, $N = 12$.

**Step 3:** Apply the update rule:

$$\text{Beta}(1, 1) \xrightarrow{8H,\, 4T} \text{Beta}(1+8,\; 1+4) = \text{Beta}(9, 5)$$

**Step 4:** The posterior PDF is:

$$f(\theta \mid x_1) = \frac{13!}{8!\,4!}\,\theta^8(1-\theta)^4, \quad \theta \in [0,1]$$

**Interpretation:** We started with no prior knowledge. After observing 8 heads in 12 flips, our posterior is peaked around $\hat\theta = 8/12 \approx 0.67$, reflecting the data. The posterior mean is $\frac{9}{9+5} = \frac{9}{14} \approx 0.643$.

---

### 4.6 Worked Example 2 — Sequential Updating

**Problem:** Using the posterior from Example 1 as the new prior, we flip the same coin again and get $n$ heads and $m$ tails. Show that the new posterior is Beta$(9+n, 5+m)$.

**Solution:**

**Step 1:** New prior = Beta$(9, 5)$, so $f(\theta) = c_2\,\theta^8(1-\theta)^4$.

**Step 2:** New likelihood: $c_3\,\theta^n(1-\theta)^m$.

**Step 3:** Multiply:

$$\theta^8(1-\theta)^4 \cdot \theta^n(1-\theta)^m = \theta^{n+8}(1-\theta)^{m+4}$$

**Step 4:** This is the kernel of Beta$(n+9, m+5)$. So:

$$f(\theta \mid x_1, x_2) \sim \text{Beta}(n+9,\; m+5) = \text{Beta}(9+n,\; 5+m)$$

**Key Insight:** Sequential updating gives the same result as batch updating. Whether you update once with all the data, or update step by step, you get the same posterior. This is a consequence of Bayes' theorem and is extremely useful in practice (e.g., online learning).

---

## 5. Beta Priors with Bernoulli Likelihood

### 5.1 Setup

The Bernoulli$(\theta)$ distribution is just Binomial$(1, \theta)$. It models a single trial with:
- $P(\text{success}) = \theta$
- $P(\text{failure}) = 1 - \theta$

### 5.2 Update Rule

$$\boxed{\text{Beta}(a,b) \xrightarrow{\text{success}} \text{Beta}(a+1,\; b)}$$

$$\boxed{\text{Beta}(a,b) \xrightarrow{\text{failure}} \text{Beta}(a,\; b+1)}$$

Each observation updates exactly one hyperparameter by 1.

### 5.3 The Full Bayesian Update Table

| Hypothesis | Data | Prior | Likelihood | Posterior |
|---|---|---|---|---|
| $\theta$ | $x=1$ (success) | $\text{Beta}(a,b)$ | $p(x\|\theta)=\theta$ | $\text{Beta}(a+1,b)$ |
| $\theta$ | $x=0$ (failure) | $\text{Beta}(a,b)$ | $p(x\|\theta)=1-\theta$ | $\text{Beta}(a, b+1)$ |

---

## 6. Beta Priors with Geometric Likelihood

### 6.1 Setup

The **Geometric$(\theta)$** distribution models the number $x$ of successes before the **first failure**, where the probability of success on each trial is $\theta$.

$$p(x \mid \theta) = \theta^x(1-\theta), \quad x = 0, 1, 2, \ldots$$

### 6.2 Update Rule

$$\boxed{\text{Beta}(a,b) \xrightarrow{\text{observe } x \text{ successes before first failure}} \text{Beta}(a+x,\; b+1)}$$

### 6.3 The Full Update Table

| Hypothesis | Data | Prior | Likelihood | Posterior |
|---|---|---|---|---|
| $\theta$ | $x$ | $\text{Beta}(a,b)$ | $\theta^x(1-\theta)$ | $\text{Beta}(a+x,\; b+1)$ |
| $\theta$ | $x$ | $c_1\theta^{a-1}(1-\theta)^{b-1}$ | $\theta^x(1-\theta)$ | $c_3\theta^{a+x-1}(1-\theta)^b$ |

**Why Beta is conjugate for Geometric too:** The Geometric likelihood $\theta^x(1-\theta)$ is **proportional** (as a function of $\theta$) to the Binomial likelihood $\binom{N}{x}\theta^x(1-\theta)^{N-x}$ when $N = x+1$ (one failure at the end). The binomial coefficient is just a constant with respect to $\theta$, so it doesn't affect the posterior.

### 6.4 Worked Example 3 — Mario and Luigi (Likelihood Principle)

**Problem:** Mario and Luigi both use prior $f(\theta) \sim \text{Beta}(5,5)$ for the probability of heads on unusual coins.

- **Mario** flips a coin 5 times and gets **4 heads and 1 tail** (Binomial experiment).
- **Luigi** flips a coin until the first tail and gets **4 heads before the first tail** (Geometric experiment).

Show they reach the same posterior.

**Solution:**

**Mario's Update:**

- Prior: Beta$(5,5)$, so $f(\theta) = c_1\theta^4(1-\theta)^4$
- Likelihood (Binomial): $\binom{5}{4}\theta^4(1-\theta)^1 = c_2\,\theta^4(1-\theta)$
- Bayes numerator: $c_1 c_2\,\theta^4(1-\theta)^4 \cdot \theta^4(1-\theta) = c_3\,\theta^8(1-\theta)^5$
- Posterior: **Beta$(9, 6)$**

**Luigi's Update:**

- Prior: Beta$(5,5)$, so $f(\theta) = c_1\theta^4(1-\theta)^4$
- Likelihood (Geometric): $\theta^4(1-\theta)$
- Bayes numerator: $c_1\,\theta^4(1-\theta)^4 \cdot \theta^4(1-\theta) = c_3\,\theta^8(1-\theta)^5$
- Posterior: **Beta$(9, 6)$**

**Both posteriors are Beta$(9,6)$. They are identical.**

**Why?** Because Mario's and Luigi's likelihoods differ only by the constant $\binom{5}{4} = 5$, which is a multiplicative constant that cancels out in the normalization. As functions of $\theta$, both likelihoods are **proportional** to $\theta^4(1-\theta)$.

> **Intuition:** What matters for updating beliefs about $\theta$ is not *how* you designed the experiment, but only the functional relationship between $\theta$ and the probability of the observed data pattern. This is the **Likelihood Principle**.

---

## 7. Bayesian Updating: Continuous Hypotheses and Continuous Data

### 7.1 The General Framework

When both the hypothesis $\theta$ and the data $x$ are continuous, Bayesian updating works exactly as in the discrete case. The key change is:

- The **prior** is a PDF $f(\theta)\,d\theta$ over continuous $\theta$
- The **likelihood** $\phi(x \mid \theta)\,dx$ is a PDF for $x$ given $\theta$
- The **normalizing constant** requires an integral instead of a sum:

$$\phi(x) = \int_{-\infty}^{\infty} \phi(x \mid \theta)\,f(\theta)\,d\theta$$

### 7.2 The Continuous Bayesian Update Table

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $f(\theta)\,d\theta$ | $\phi(x\mid\theta)$ | $\phi(x\mid\theta)f(\theta)\,d\theta$ | $\displaystyle f(\theta\mid x)\,d\theta = \frac{\phi(x\mid\theta)f(\theta)\,d\theta}{\phi(x)}$ |
| Total (integrate over $\theta$) | 1 | — | $\phi(x) = \int \phi(x\mid\theta)f(\theta)\,d\theta$ | 1 |

### 7.3 Intuition

> **Intuition:** In the continuous-continuous case, the hypothesis $\theta$ takes a value in some interval. The prior $f(\theta)\,d\theta$ gives the probability that $\theta$ lies in a tiny interval of width $d\theta$ around $\theta$. Similarly, the likelihood $\phi(x\mid\theta)\,dx$ gives the probability that the data $x$ lies in a tiny interval of width $dx$ around $x$, given hypothesis $\theta$.
>
> The Bayesian update formula is exactly the same as before. The only computational difference is that integrals replace sums in the normalization step.

---

## 8. Normal-Normal Updating

### 8.1 Concept Overview

The **Normal distribution is its own conjugate prior**: if the likelihood is Normal with known variance, and the prior is Normal, then the posterior is also Normal.

This is one of the most important conjugate pairs in all of Bayesian statistics, with applications in signal processing, Kalman filtering, machine learning, and much more.

### 8.2 Setup

- **Unknown parameter:** $\theta$ (e.g., the true mean of some process)
- **Data:** A single measurement $x \sim \text{N}(\theta, \sigma^2)$, where $\sigma^2$ is known
- **Prior:** $\theta \sim \text{N}(\mu_\text{prior}, \sigma^2_\text{prior})$

### 8.3 The Posterior (Result)

The posterior is Normal:

$$\theta \mid x \sim \text{N}(\mu_\text{post},\; \sigma^2_\text{post})$$

### 8.4 Key Formulas — The Update Equations

**Form 1 (Precision-weighted):**

$$\frac{1}{\sigma^2_\text{post}} = \frac{1}{\sigma^2_\text{prior}} + \frac{1}{\sigma^2}, \qquad \frac{\mu_\text{post}}{\sigma^2_\text{post}} = \frac{\mu_\text{prior}}{\sigma^2_\text{prior}} + \frac{x}{\sigma^2}$$

**Form 2 (Weighted Average — easier to read):**

$$\boxed{a = \frac{1}{\sigma^2_\text{prior}}, \quad b = \frac{1}{\sigma^2}, \quad \mu_\text{post} = \frac{a\,\mu_\text{prior} + b\,x}{a+b}, \quad \sigma^2_\text{post} = \frac{1}{a+b}}$$

> **Key Insight:** The posterior mean $\mu_\text{post}$ is a **weighted average** of the prior mean $\mu_\text{prior}$ and the data $x$. The weights are the **precisions** (reciprocals of variances):
> - High precision in prior (small $\sigma^2_\text{prior}$) → large weight on prior → posterior stays near prior
> - High precision in data (small $\sigma^2$) → large weight on data → posterior moves toward data

### 8.5 Update Table

| Hypothesis | Data | Prior | Likelihood | Posterior |
|---|---|---|---|---|
| $\theta$ | $x$ | $\text{N}(\mu_\text{prior}, \sigma^2_\text{prior})$ | $\text{N}(\theta, \sigma^2)$ | $\text{N}(\mu_\text{post}, \sigma^2_\text{post})$ |
| $\theta$ | $x$ | $c_1\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}}\right)$ | $c_2\exp\!\left(-\frac{(x-\theta)^2}{2\sigma^2}\right)$ | $c_3\exp\!\left(-\frac{(\theta-\mu_\text{post})^2}{2\sigma^2_\text{post}}\right)$ |

### 8.6 Derivation by Completing the Square

**Problem:** Prior $\theta \sim \text{N}(4, 8)$, likelihood $x \sim \text{N}(\theta, 5)$, single data point $x_1 = 3$.

**Step 1:** Write prior and likelihood PDFs (dropping normalizing constants):

$$f(\theta) = c_1\,e^{-(\theta-4)^2/16}, \qquad \phi(x_1\mid\theta) = c_2\,e^{-(3-\theta)^2/10}$$

Note: Prior has $2\sigma^2_\text{prior} = 2\times 8 = 16$; likelihood has $2\sigma^2 = 2\times 5 = 10$.

**Step 2:** Multiply prior × likelihood:

$$f(\theta\mid x_1) \propto e^{-(\theta-4)^2/16}\cdot e^{-(3-\theta)^2/10} = \exp\!\left(-\frac{(\theta-4)^2}{16} - \frac{(3-\theta)^2}{10}\right)$$

**Step 3:** Combine the exponent over a common denominator of 80:

$$-\frac{(\theta-4)^2}{16} - \frac{(3-\theta)^2}{10} = \frac{-5(\theta-4)^2 - 8(3-\theta)^2}{80}$$

**Step 4:** Expand the numerator:

$$-5(\theta^2 - 8\theta + 16) - 8(9 - 6\theta + \theta^2)$$
$$= -5\theta^2 + 40\theta - 80 - 72 + 48\theta - 8\theta^2$$
$$= -13\theta^2 + 88\theta - 152$$

**Step 5:** Factor out $-13$ and complete the square:

$$= -13\left(\theta^2 - \frac{88}{13}\theta\right) - 152 = -13\left(\theta - \frac{44}{13}\right)^2 + 13\left(\frac{44}{13}\right)^2 - 152$$

The constant terms $\left[13(44/13)^2 - 152\right]$ are absorbed into the normalizing constant $c_4$.

**Step 6:** So:

$$f(\theta\mid x_1) \propto \exp\!\left(-\frac{13(\theta - 44/13)^2}{80}\right) = \exp\!\left(-\frac{(\theta - 44/13)^2}{80/13}\right)$$

This is a Normal PDF with mean $44/13$ and variance $80/26 = 40/13$.

$$\theta \mid x_1 \sim \text{N}\!\left(\frac{44}{13},\; \frac{40}{13}\right) \approx \text{N}(3.385,\; 3.077)$$

**Verification using the update formulas:**

$$a = \frac{1}{\sigma^2_\text{prior}} = \frac{1}{8}, \quad b = \frac{1}{\sigma^2} = \frac{1}{5}$$

$$\mu_\text{post} = \frac{a\,\mu_\text{prior} + b\,x}{a+b} = \frac{\frac{4}{8} + \frac{3}{5}}{\frac{1}{8}+\frac{1}{5}} = \frac{\frac{5}{10} + \frac{6}{10}}{\frac{13}{40}} = \frac{11/10}{13/40} = \frac{11}{10}\cdot\frac{40}{13} = \frac{44}{13} \approx 3.385\checkmark$$

$$\sigma^2_\text{post} = \frac{1}{a+b} = \frac{1}{\frac{1}{8}+\frac{1}{5}} = \frac{1}{\frac{13}{40}} = \frac{40}{13} \approx 3.077\checkmark$$

---

### 8.7 Worked Example 4 — Normal Updating (Numerical)

**Problem:** We have one data point $x = 2$ drawn from $\text{N}(\theta, 3^2)$. The prior is $\theta \sim \text{N}(4, 2^2)$.

**(a)** Identify $\mu_\text{prior}$, $\sigma_\text{prior}$, $\sigma$, $n$, and $\bar{x}$.

**(b)** Make a Bayesian update table (leave posterior as an unsimplified product).

**(c)** Use the update formulas to find the posterior.

**Solution:**

**(a)** Reading off the problem directly:

$$\mu_\text{prior} = 4, \quad \sigma_\text{prior} = 2, \quad \sigma = 3, \quad n = 1, \quad \bar{x} = 2$$

**(b)** Bayesian update table:

| Hypothesis | Prior | Likelihood | Posterior |
|---|---|---|---|
| $\theta$ | $f(\theta)\sim\text{N}(4, 2^2)$ | $\phi(x\|\theta)\sim\text{N}(\theta, 3^2)$ | $f(\theta\|x)\sim\text{N}(\mu_\text{post}, \sigma^2_\text{post})$ |
| $\theta$ | $c_1\exp\!\left(-\frac{(\theta-4)^2}{8}\right)$ | $c_2\exp\!\left(-\frac{(2-\theta)^2}{18}\right)$ | $c_3\exp\!\left(-\frac{(\theta-4)^2}{8}\right)\exp\!\left(-\frac{(2-\theta)^2}{18}\right)$ |

**(c)** Compute the update weights:

$$a = \frac{1}{\sigma^2_\text{prior}} = \frac{1}{4}, \quad b = \frac{1}{\sigma^2} = \frac{1}{9}$$

$$a + b = \frac{1}{4} + \frac{1}{9} = \frac{9}{36} + \frac{4}{36} = \frac{13}{36}$$

$$\mu_\text{post} = \frac{a\,\mu_\text{prior} + b\,x}{a+b} = \frac{\frac{1}{4}\cdot 4 + \frac{1}{9}\cdot 2}{\frac{13}{36}} = \frac{1 + \frac{2}{9}}{\frac{13}{36}} = \frac{\frac{11}{9}}{\frac{13}{36}} = \frac{11}{9}\cdot\frac{36}{13} = \frac{44}{13} \approx 3.385$$

$$\sigma^2_\text{post} = \frac{1}{a+b} = \frac{1}{\frac{13}{36}} = \frac{36}{13} \approx 2.769$$

**Posterior:** $f(\theta \mid x=2) \sim \text{N}(3.385,\; 2.769)$

**Interpretation:** The data point $x = 2$ pulled the posterior mean down from 4 toward 2. The posterior mean 3.385 lies between the prior mean (4) and the data (2), closer to the prior because the prior variance ($\sigma^2_\text{prior}=4$) is smaller than the data variance ($\sigma^2=9$), so we trust the prior somewhat more.

---

### 8.8 Worked Example 5 — Basketball Free Throws

**Problem:** On a basketball team, career free throw percentage $\theta$ follows $\text{N}(75, 6^2)$. In a given year, a player's percentage is $\text{N}(\theta, 4^2)$.

This season, Sophie Lie made 85% of her free throws. What is the posterior expected value of her career percentage $\theta$?

**Solution:**

**Step 1:** Identify the setup.
- Parameter of interest: $\theta$ = career average
- Data: $x = 85$ (this season's percentage)
- Prior: $\theta \sim \text{N}(75, 36)$
- Likelihood: $x \sim \text{N}(\theta, 16)$

**Step 2:** Compute update weights (precisions):

$$a = \frac{1}{\sigma^2_\text{prior}} = \frac{1}{36}, \quad b = \frac{1}{\sigma^2} = \frac{1}{16}$$

$$a + b = \frac{1}{36} + \frac{1}{16} = \frac{16}{576} + \frac{36}{576} = \frac{52}{576} = \frac{13}{144}$$

**Step 3:** Compute posterior mean and variance:

$$\mu_\text{post} = \frac{\frac{75}{36} + \frac{85}{16}}{\frac{13}{144}} = \frac{\frac{75\cdot4}{144} + \frac{85\cdot9}{144}}{\frac{13}{144}} = \frac{\frac{300 + 765}{144}}{\frac{13}{144}} = \frac{1065}{13} \approx 81.9$$

$$\sigma^2_\text{post} = \frac{1}{\frac{13}{144}} = \frac{144}{13} \approx 11.1$$

**Posterior:** $f(\theta \mid x=85) \sim \text{N}(81.9,\; 11.1)$

**Interpretation:** Sophie's 85% this season was well above the team average of 75%. The Bayesian estimate of her career average is 81.9% — significantly higher than the team average, but below her single-season performance (because one season's data is noisy). The posterior "hedges" between the prior (75) and the data (85).

---

### 8.9 Two Key Properties of Normal-Normal Updating

> **Property 1:** The posterior mean is always **between** the prior mean and the data.
>
> **Proof:** $\mu_\text{post} = \frac{a\mu_\text{prior} + bx}{a+b}$ is a convex combination (weighted average) of $\mu_\text{prior}$ and $x$ with positive weights $a/(a+b)$ and $b/(a+b)$ summing to 1. So $\mu_\text{post}$ lies between $\mu_\text{prior}$ and $x$. $\square$

> **Property 2:** The posterior variance is **smaller** than both the prior variance and the data variance.
>
> **Proof:**
> $$\sigma^2_\text{post} = \frac{1}{a+b} < \frac{1}{a} = \sigma^2_\text{prior} \quad\text{since } b > 0$$
> $$\sigma^2_\text{post} = \frac{1}{a+b} < \frac{1}{b} = \sigma^2 \quad\text{since } a > 0$$
>
> **Interpretation:** Every data point reduces our uncertainty. The more data we observe, the smaller the posterior variance becomes (i.e., the more confident we are about $\theta$). $\square$

### 8.10 Worked Example 6 — Visualizing the Update

**Problem:** $x \sim \text{N}(\theta, 4/9)$, prior $\theta \sim \text{N}(0, 1)$, observe $x = 6.5$.

**Solution:**

$$\mu_\text{prior} = 0,\quad \sigma^2_\text{prior} = 1,\quad \sigma^2 = 4/9$$

$$a = \frac{1}{1} = 1, \quad b = \frac{1}{4/9} = \frac{9}{4} = 2.25$$

$$a + b = 1 + 2.25 = 3.25 = \frac{13}{4}$$

$$\mu_\text{post} = \frac{1\cdot 0 + 2.25\cdot 6.5}{3.25} = \frac{14.625}{3.25} = 4.5$$

$$\sigma^2_\text{post} = \frac{1}{3.25} = \frac{4}{13} \approx 0.308$$

**Observations from this example:**
1. The posterior mean (4.5) is between the prior mean (0) and the data (6.5).
2. Since $b > a$ (data is more precise than prior), the posterior is pulled **closer to the data** than to the prior.
3. The posterior variance (0.308) is much smaller than both the prior variance (1) and the data variance (0.44).

---

## 9. Multiple Data Points: Normal-Normal

### 9.1 Extension to $n$ Data Points

If we observe $n$ data points $x_1, x_2, \ldots, x_n$, each drawn from $\text{N}(\theta, \sigma^2)$, the update formula generalizes naturally.

**Sequential updating** yields (after $n$ steps):

$$\frac{1}{\sigma^2_\text{post}} = \frac{1}{\sigma^2_\text{prior}} + \frac{n}{\sigma^2}$$

$$\frac{\mu_\text{post}}{\sigma^2_\text{post}} = \frac{\mu_\text{prior}}{\sigma^2_\text{prior}} + \frac{x_1 + x_2 + \cdots + x_n}{\sigma^2}$$

### 9.2 Compact Update Formulas for $n$ Data Points

Let $\bar{x} = \frac{x_1 + \cdots + x_n}{n}$ be the sample mean. Then:

$$\boxed{a = \frac{1}{\sigma^2_\text{prior}}, \quad b = \frac{n}{\sigma^2}, \quad \mu_\text{post} = \frac{a\,\mu_\text{prior} + b\,\bar{x}}{a+b}, \quad \sigma^2_\text{post} = \frac{1}{a+b}}$$

### 9.3 Interpretation

- The posterior mean is a weighted average of $\mu_\text{prior}$ and the **sample mean** $\bar{x}$.
- Weight on prior: $\frac{a}{a+b} = \frac{1/\sigma^2_\text{prior}}{1/\sigma^2_\text{prior} + n/\sigma^2}$
- Weight on data: $\frac{b}{a+b} = \frac{n/\sigma^2}{1/\sigma^2_\text{prior} + n/\sigma^2}$

Two key takeaways:

1. **More data** (large $n$) → large $b$ → posterior dominated by sample mean $\bar{x}$
2. **Strong prior** (small $\sigma^2_\text{prior}$) → large $a$ → posterior dominated by prior mean

### 9.4 Sequential Derivation (Worked Example 7)

**Problem:** Data $x_1, x_2, x_3$ each from $\text{N}(\theta, \sigma^2)$. Prior $\theta \sim \text{N}(\mu_0, \sigma_0^2)$. Show sequential and batch updating give the same result.

**Solution (Sequential):**

**After $x_1$:**

$$\frac{1}{\sigma_1^2} = \frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}, \qquad \frac{\mu_1}{\sigma_1^2} = \frac{\mu_0}{\sigma_0^2} + \frac{x_1}{\sigma^2}$$

**After $x_2$ (using $\sigma_1^2$ as new prior variance):**

$$\frac{1}{\sigma_2^2} = \frac{1}{\sigma_1^2} + \frac{1}{\sigma^2} = \frac{1}{\sigma_0^2} + \frac{2}{\sigma^2}, \qquad \frac{\mu_2}{\sigma_2^2} = \frac{\mu_0}{\sigma_0^2} + \frac{x_1+x_2}{\sigma^2}$$

**After $x_3$:**

$$\frac{1}{\sigma_3^2} = \frac{1}{\sigma_0^2} + \frac{3}{\sigma^2}, \qquad \frac{\mu_3}{\sigma_3^2} = \frac{\mu_0}{\sigma_0^2} + \frac{x_1+x_2+x_3}{\sigma^2}$$

This matches the $n=3$ batch formula with $\bar{x} = (x_1+x_2+x_3)/3$. $\checkmark$

---

## 10. The Likelihood Principle

### 10.1 Concept

> **The Likelihood Principle:** In Bayesian inference, two datasets $x_1$ and $x_2$ lead to the **same posterior** if and only if their likelihoods $\phi(x_1\mid\theta)$ and $\phi(x_2\mid\theta)$ are **proportional as functions of $\theta$**.

### 10.2 Three Statements (from Class)

Given a fixed prior, let $x_1$ and $x_2$ be two datasets.

**(a) True:** If $\phi(x_1\mid\theta) = \phi(x_2\mid\theta)$ (the likelihoods are equal), they produce the same posterior.

**(b) True:** If $\phi(x_1\mid\theta) = k\cdot\phi(x_2\mid\theta)$ for some constant $k > 0$ (proportional), they produce the same posterior. This is because the constant $k$ cancels out in the normalization step.

**(c) False:** If two likelihoods are proportional, they need not be equal. Proportionality is weaker than equality.

### 10.3 Intuition

> **Intuition:** The constant factors in the likelihood (like binomial coefficients) don't affect Bayesian inference about $\theta$ at all. They are irrelevant to the "shape" of the posterior. What matters is only the dependence of the likelihood on $\theta$.
>
> This explains the Mario-Luigi example: even though they ran different experiments, their likelihoods as functions of $\theta$ are proportional, so they reach the same posterior beliefs about $\theta$.

---

## 11. Concept Questions with Deep Explanations

### 11.1 Concept Question 1 — More Beta

**Question:** Prior is Beta$(6, 8)$. You flip a coin 7 times, getting 2 heads and 5 tails. What is the posterior?

**Options:** (a) Beta$(2,5)$, (b) Beta$(11,10)$, (c) Beta$(6,8)$, (d) Beta$(8,13)$

**Answer: (d) Beta$(8, 13)$**

**Reasoning:**

Using the Beta-Binomial update rule: Beta$(a,b) \to \text{Beta}(a + \text{heads},\; b + \text{tails})$:

$$\text{Beta}(6, 8) \xrightarrow{2H,\, 5T} \text{Beta}(6+2,\; 8+5) = \text{Beta}(8, 13)$$

**Why not (a)?** That ignores the prior.  
**Why not (b)?** That adds heads to both parameters.  
**Why not (c)?** That ignores the data entirely.

---

### 11.2 Concept Question 2 — Strong Priors

**Question:** Prior for $\theta$ is uniform on $[0, 0.7]$ and zero on $(0.7, 1]$. We flip 65 times and get 60 heads. Which graph shows the posterior?

**Answer: Graph C** — the graph spiking near $\theta = 0.7$.

**Deep Explanation:**

- The data (60 heads out of 65) strongly suggests $\theta$ is near $60/65 \approx 0.92$.
- However, our prior is **zero** for $\theta > 0.7$. This is a hard constraint — no amount of data can make the posterior positive at values where the prior is zero.
- Mathematically: $f(\theta \mid x) \propto \phi(x \mid \theta) \cdot f(\theta)$. If $f(\theta) = 0$, then $f(\theta \mid x) = 0$ regardless of the likelihood.
- So the posterior is restricted to $[0, 0.7]$, but within that range, the likelihood function $\theta^{60}(1-\theta)^5$ is maximized near the upper end $\theta = 0.7$.
- The posterior therefore spikes near $\theta = 0.7$ — the maximum likelihood estimate constrained to the allowed region.

> **Warning:** Strong priors that assign zero probability to a region **foreclose** on those parameter values forever, regardless of how much data you collect. This is why prior choice matters enormously in Bayesian inference!

---

### 11.3 Concept Question 3 — Normal Priors, Normal Likelihood (a)

**Question:** Blue graph = prior (appears to be around mean 6). Data values in order: 3, 9, 12. Which plot is the posterior to just the first data value $x_1 = 3$?

**Answer: Plot 2**

**Reasoning:**

1. The first data value is $x_1 = 3$, which is to the **left** of the prior mean (~6). So the posterior mean must shift left (toward 3), but not all the way to 3.
2. This rules out plots to the right of the prior.
3. The posterior mean lies **between** 3 and the prior mean. Among plots 1 and 2 (which are to the left of the prior), only Plot 2 is in the right range.
4. Additionally, the posterior variance must be **smaller** than the prior variance (observing data always reduces uncertainty). Plot 1 has larger variance than the prior; Plot 2 has smaller variance.
5. Therefore: **Plot 2**.

---

### 11.4 Concept Question 4 — Normal Priors, Normal Likelihood (b)

**Question:** Same setup. Which graph is the posterior to all 3 data values (3, 9, 12)?

**Answer: Plot 3**

**Reasoning:**

1. The average of the 3 data values: $\bar{x} = (3+9+12)/3 = 8$.
2. The posterior mean must lie between the prior mean (~6) and $\bar{x} = 8$, so **to the right** of the prior.
3. This suggests Plots 3 or 4 (to the right of the prior).
4. Each data point reduces variance. Since we've seen 3 data points (compared to 1 for Plot 2), the variance should be smaller than Plot 2's.
5. Between Plots 3 and 4, Plot 3 has smaller variance.
6. Therefore: **Plot 3**.

---

## 12. Board Problems — Full Worked Solutions

### 12.1 Problem 1 — Beta Priors for Medical Treatment

**Setup:** Unknown probability of success $\theta$. Prior: $\theta \sim \text{Beta}(5, 5)$.

The Beta$(5,5)$ PDF is:

$$f(\theta) = \frac{9!}{4!\,4!}\theta^4(1-\theta)^4 = c_1\,\theta^4(1-\theta)^4$$

This is a symmetric bell-shaped distribution centered at $\theta = 0.5$, reflecting the belief that the treatment has approximately even odds of working.

---

#### Part (a) — 25 Patients, 20 Successes

**Find the posterior distribution and identify its type.**

**Step 1:** Set up the Bayesian update table.

| Hypothesis | Prior | Likelihood | Bayes Numerator | Posterior |
|---|---|---|---|---|
| $\theta$ | $c_1\theta^4(1-\theta)^4\,d\theta$ | $\binom{25}{20}\theta^{20}(1-\theta)^5$ | $c_3\,\theta^{24}(1-\theta)^9\,d\theta$ | Beta$(25, 10)$ |
| Summary | Beta$(5,5)$ | Binomial prob. | — | **Beta$(25, 10)$** |

**Step 2:** Identify the type.

The Bayes numerator has the form $c\,\theta^{24}(1-\theta)^9$. Comparing with the Beta kernel $\theta^{a-1}(1-\theta)^{b-1}$, we get $a-1 = 24$ and $b-1 = 9$, so $a = 25$, $b = 10$.

**Posterior: $\theta \mid x \sim \text{Beta}(25, 10)$**

Posterior mean: $\frac{25}{25+10} = \frac{25}{35} = \frac{5}{7} \approx 0.714$.

Interpretation: We started with equal prior belief at 0.5. After observing 20 successes in 25 trials, our posterior mean is 71.4% — substantially above 0.5, reflecting strong evidence of efficacy.

---

#### Part (b) — Same Data, Recorded in Order

**Data:** S S S S F S S S S S F F S S S F S F S S S S S S S (20 S and 5 F)

**Find the posterior.**

The only change from part (a) is that the likelihood is now:

$$p(\text{specific sequence}\mid\theta) = \theta^{20}(1-\theta)^5$$

(no binomial coefficient, since the order is specified)

The posterior numerator is still:

$$\theta^4(1-\theta)^4 \cdot \theta^{20}(1-\theta)^5 = \theta^{24}(1-\theta)^9$$

**Posterior: Beta$(25, 10)$** — exactly the same as part (a).

**Explanation:** The binomial coefficient $\binom{25}{20}$ in the unordered likelihood is just a constant (a number that doesn't depend on $\theta$). It cancels out in the normalization. This is a direct application of the Likelihood Principle.

---

#### Part (c) — Posterior Predictive Probability

**Question:** Give an integral for the posterior predictive probability of success with the next patient.

The posterior predictive probability uses the law of total probability:

$$P(\text{success} \mid \text{data}) = \int_0^1 P(\text{success}\mid\theta)\cdot f(\theta\mid\text{data})\,d\theta$$

The posterior is Beta$(25, 10)$, which has PDF:

$$f(\theta\mid\text{data}) = \frac{34!}{24!\,9!}\,\theta^{24}(1-\theta)^9$$

And $P(\text{success}\mid\theta) = \theta$, so:

$$P(\text{success}\mid\text{data}) = \int_0^1 \theta\cdot\frac{34!}{24!\,9!}\,\theta^{24}(1-\theta)^9\,d\theta = \frac{34!}{24!\,9!}\int_0^1\theta^{25}(1-\theta)^9\,d\theta$$

---

#### Part (d) — Evaluate Without Direct Integration

**Goal:** Compute the integral $\displaystyle\int_0^1\theta^{25}(1-\theta)^9\,d\theta$ using knowledge of Beta PDFs.

**Key idea:** The PDF of Beta$(26, 10)$ is:

$$\frac{35!}{25!\,9!}\,\theta^{25}(1-\theta)^9$$

and it integrates to 1 over $[0,1]$. Therefore:

$$\frac{35!}{25!\,9!}\int_0^1\theta^{25}(1-\theta)^9\,d\theta = 1 \implies \int_0^1\theta^{25}(1-\theta)^9\,d\theta = \frac{25!\,9!}{35!}$$

**Substituting back:**

$$P(\text{success}\mid\text{data}) = \frac{34!}{24!\,9!}\cdot\frac{25!\,9!}{35!} = \frac{34!\cdot 25!}{24!\cdot 35!} = \frac{25}{35} = \frac{5}{7} \approx 0.714$$

**The posterior predictive probability of success is $5/7 \approx 71.4\%$.**

> **Beautiful Insight:** This equals the **posterior mean** of Beta$(25, 10)$, which is $\frac{25}{25+10} = \frac{25}{35} = \frac{5}{7}$. This is **not a coincidence** — for a Beta$( a, b)$ posterior with Bernoulli likelihood, the posterior predictive probability always equals the posterior mean $\frac{a}{a+b}$.

---

### 12.2 Problem 2 — Normal-Normal Updating

#### Problem 2a — Basic Update

**Setup:** One data point $x = 2$ from $\text{N}(\theta, 3^2)$. Prior: $\theta \sim \text{N}(4, 2^2)$.

**(a)** $\mu_\text{prior} = 4$, $\sigma_\text{prior} = 2$, $\sigma = 3$, $n = 1$, $\bar{x} = 2$.

**(b)** Bayesian update table:

| Hypothesis | Prior | Likelihood | Posterior |
|---|---|---|---|
| $\theta$ | $\text{N}(4, 4)$ | $\text{N}(\theta, 9)$ | $\text{N}(\mu_\text{post}, \sigma^2_\text{post})$ |
| $\theta$ | $c_1\exp\!\left(-\frac{(\theta-4)^2}{8}\right)$ | $c_2\exp\!\left(-\frac{(2-\theta)^2}{18}\right)$ | $c_3\exp\!\left(-\frac{(\theta-4)^2}{8}\right)\exp\!\left(-\frac{(2-\theta)^2}{18}\right)$ |

**(c)** Apply update formulas:

$$a = \frac{1}{4}, \quad b = \frac{1}{9}, \quad a+b = \frac{9}{36}+\frac{4}{36} = \frac{13}{36}$$

$$\mu_\text{post} = \frac{\frac{1}{4}\cdot 4 + \frac{1}{9}\cdot 2}{\frac{13}{36}} = \frac{1 + \frac{2}{9}}{\frac{13}{36}} = \frac{\frac{11}{9}}{\frac{13}{36}} = \frac{11}{9}\cdot\frac{36}{13} = \frac{44}{13} \approx 3.385$$

$$\sigma^2_\text{post} = \frac{1}{\frac{13}{36}} = \frac{36}{13} \approx 2.769$$

**Posterior:** $f(\theta \mid x=2) \sim \text{N}(3.385,\; 2.769)$

**Interpretation:** Prior mean was 4. Data was 2. The posterior mean (3.385) is between 4 and 2, closer to the prior because the prior variance (4) is smaller than the data variance (9), i.e., we trusted the prior more.

---

#### Problem 2b — Basketball (Sophie Lie)

**(Already solved in full in Section 8.8 above.)**

**Summary:**
- Prior: $\text{N}(75, 36)$ (team career average distribution)
- Likelihood: $\text{N}(\theta, 16)$ (individual season performance)
- Data: $x = 85$
- Posterior: $\text{N}(81.9,\; 11.1)$

The posterior expected value of Sophie's career percentage is approximately **81.9%**.

---

## 13. Extra Problems

### 13.1 Conjugate Priors — Which Pairs Are Conjugate?

**Problem:** Three candidate prior-likelihood pairs. Which are conjugate?

| | Prior | Likelihood |
|---|---|---|
| (a) Exponential/Normal | $\theta \sim \text{N}(\mu_\text{prior}, \sigma^2_\text{prior})$ | $x \mid \theta \sim \text{Exp}(\theta)$ (rate $\theta$) |
| (b) Exponential/Gamma | $\theta \sim \text{Gamma}(a, b)$ | $x \mid \theta \sim \text{Exp}(\theta)$ |
| (c) Binomial/Normal | $\theta \sim \text{N}(\mu_\text{prior}, \sigma^2_\text{prior})$ | $x \mid \theta \sim \text{Binomial}(N, \theta)$ |

**Answer: Only (b) is conjugate.**

---

**Case (a): Exponential Likelihood / Normal Prior**

Prior: $f(\theta) = c_1\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}}\right)$

Exponential likelihood: $\phi(x\mid\theta) = \theta e^{-\theta x}$

Posterior (Bayes numerator):

$$f(\theta\mid x) \propto \theta\cdot\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}}\right)\cdot e^{-\theta x} = \theta\cdot\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}} - \theta x\right)$$

The extra factor of $\theta$ in front of the exponential means this is NOT a Normal PDF. **Not conjugate.**

---

**Case (b): Exponential Likelihood / Gamma Prior**

Gamma prior: $f(\theta) = c_1\,\theta^{a-1}e^{-b\theta}$

Exponential likelihood: $\phi(x\mid\theta) = \theta e^{-\theta x}$

Posterior (Bayes numerator):

$$f(\theta\mid x) \propto \theta^{a-1}e^{-b\theta}\cdot\theta e^{-\theta x} = \theta^a e^{-(b+x)\theta} = \theta^{(a+1)-1}e^{-(b+x)\theta}$$

This is the kernel of **Gamma$(a+1, b+x)$** — same family as the prior!

**Update rule:** $\text{Gamma}(a, b) \xrightarrow{\text{observe }x} \text{Gamma}(a+1,\; b+x)$. **Conjugate!** ✓

---

**Case (c): Binomial Likelihood / Normal Prior**

Binomial likelihood: $\phi(x\mid\theta) = c_2\,\theta^x(1-\theta)^{N-x}$

Normal prior: $f(\theta) = c_1\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}}\right)$

Posterior:

$$f(\theta\mid x) \propto \theta^x(1-\theta)^{N-x}\cdot\exp\!\left(-\frac{(\theta-\mu_\text{prior})^2}{2\sigma^2_\text{prior}}\right)$$

This is a product of a polynomial in $\theta$ and a Gaussian in $\theta$. This is definitely not a Gaussian PDF. **Not conjugate.**

> **Why (b) but not (a) or (c)?** The Exponential likelihood $\theta e^{-\theta x}$ has an exponential term in $\theta$ that "fits" the exponential form of the Gamma distribution. The Normal prior is quadratic in the exponent of $\theta$, which clashes with the linear-in-$\theta$ exponent of the Exponential likelihood.

---

## 14. Common Mistakes

### 14.1 Beta Distribution Mistakes

| Mistake | Correction |
|---|---|
| Using the Beta$(a,b)$ formula with $a$ and $b$ as data, not hyperparameters | $a$ and $b$ in Beta$(a,b)$ are the prior parameters. Add data counts to them. |
| Forgetting to add the counts to **both** $a$ and $b$ | Heads add to $a$; tails add to $b$. Don't confuse which is which. |
| Using Beta$(2,5)$ directly as the posterior (from data alone) | The posterior combines prior and data. Beta$(2,5)$ ignores the prior. |
| Thinking the order of data matters for Beta-Binomial updating | It doesn't — only the total heads and tails count (Likelihood Principle). |

### 14.2 Normal-Normal Mistakes

| Mistake | Correction |
|---|---|
| Using $a = \sigma^2_\text{prior}$ instead of $a = 1/\sigma^2_\text{prior}$ | Weights are **precisions** (reciprocals of variances), not variances. |
| Computing $\mu_\text{post} = (\mu_\text{prior} + x)/2$ | This is only correct when prior and data have equal variances. Use the formula! |
| Thinking the posterior mean equals the data for large $\sigma^2_\text{prior}$ | Yes — weak prior → posterior ≈ data. But use the formula to be precise. |
| Thinking the posterior variance is the average of prior and data variances | No. It's $1/(a+b) = 1/(1/\sigma^2_\text{prior} + 1/\sigma^2)$. It's always smaller than both. |
| Forgetting $\sigma$ vs $\sigma^2$ | The formula uses variances ($\sigma^2$), not standard deviations. Be careful. |

### 14.3 General Bayesian Mistakes

| Mistake | Correction |
|---|---|
| Thinking more data can change a zero-probability prior region | If $f(\theta_0) = 0$, then $f(\theta_0\mid x) = 0$ no matter what the data says. |
| Confusing prior predictive and posterior predictive | Prior predictive uses only prior. Posterior predictive integrates against posterior. |
| Dropping binomial coefficients and getting a different posterior | Binomial coefficients are constants w.r.t. $\theta$ and don't affect the posterior. |

---

## 15. Quick Reference Summary

### 15.1 The Beta Distribution

$$\text{Beta}(a,b): \quad f(\theta) = \frac{(a+b-1)!}{(a-1)!(b-1)!}\theta^{a-1}(1-\theta)^{b-1}, \quad \theta\in[0,1]$$

- **Mean:** $a/(a+b)$
- **Special case:** Beta$(1,1)$ = Uniform$[0,1]$
- **Key fact:** If $f(\theta) = c\cdot\theta^{a-1}(1-\theta)^{b-1}$, then $f\sim\text{Beta}(a,b)$ automatically.

---

### 15.2 Conjugate Update Rules

| Prior | Likelihood | Posterior |
|---|---|---|
| Beta$(a,b)$ | Binomial$(N,\theta)$: $x$ successes | Beta$(a+x,\ b+N-x)$ |
| Beta$(a,b)$ | Bernoulli$(\theta)$: success | Beta$(a+1,\ b)$ |
| Beta$(a,b)$ | Bernoulli$(\theta)$: failure | Beta$(a,\ b+1)$ |
| Beta$(a,b)$ | Geometric$(\theta)$: $x$ successes before first failure | Beta$(a+x,\ b+1)$ |
| Normal$(\mu_p, \sigma_p^2)$ | Normal$(\theta, \sigma^2)$: data $\bar{x}$ from $n$ points | Normal$(\mu_\text{post},\sigma_\text{post}^2)$ |

---

### 15.3 Normal-Normal Update Formulas

$$a = \frac{1}{\sigma^2_\text{prior}}, \quad b = \frac{n}{\sigma^2}, \quad \mu_\text{post} = \frac{a\mu_\text{prior}+b\bar{x}}{a+b}, \quad \sigma^2_\text{post} = \frac{1}{a+b}$$

- Posterior mean is a **weighted average** of prior mean and sample mean.
- Posterior variance is always **smaller** than prior variance.
- More data → posterior dominated by data.
- Stronger prior (smaller $\sigma^2_\text{prior}$) → posterior dominated by prior.

---

### 15.4 Posterior Predictive (Beta Posterior)

For Beta$(a,b)$ posterior with Bernoulli likelihood:

$$P(\text{success}\mid\text{data}) = \frac{a}{a+b} = \text{posterior mean}$$

---

### 15.5 The Likelihood Principle

Two likelihoods that are **proportional as functions of $\theta$** yield **identical posteriors** (given the same prior). Multiplicative constants in the likelihood are irrelevant to Bayesian inference.

---

*End of MIT 18.05 Class 15 Study Notes.*  
*Source: MIT OpenCourseWare, https://ocw.mit.edu — 18.05 Introduction to Probability and Statistics, Spring 2022.*
