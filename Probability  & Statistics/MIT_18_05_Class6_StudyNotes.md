# MIT 18.05 — Introduction to Probability and Statistics
## Class 6 Complete Study Notes: Continuous Random Variables, Normal Distributions & the Central Limit Theorem

> **Source:** MIT 18.05 Spring 2022 — Class 6a, Class 6b, and Appendix (Orloff & Bloom)
> **Coverage:** Medians & Quantiles · PDFs & CDFs · Transformations · Histograms · Normal Distributions · Standardization · CLT · Law of Large Numbers · Chebyshev Inequality

---

## Table of Contents

1. [Medians and Quantiles of Continuous Distributions](#1-medians-and-quantiles)
2. [Working with PDFs and CDFs](#2-working-with-pdfs-and-cdfs)
3. [Transformations of Random Variables](#3-transformations-of-random-variables)
4. [Histograms: Frequency vs. Density](#4-histograms-frequency-vs-density)
5. [Normal Distributions & the Empirical Rule](#5-normal-distributions--the-empirical-rule)
6. [Standardization](#6-standardization)
7. [The Central Limit Theorem (CLT)](#7-the-central-limit-theorem)
8. [Sampling Approximations Using the CLT](#8-sampling-approximations-using-the-clt)
9. [Appendix: Formal Proofs — LLN & Chebyshev Inequality](#9-appendix-formal-proofs)
10. [Strategy Under Uncertainty: The Desperation Problem](#10-strategy-under-uncertainty-the-desperation-problem)
11. [Quick Reference Summary](#11-quick-reference-summary)

---

## 1. Medians and Quantiles

### 1.1 Concept Overview

The **median** of a continuous random variable is the value that splits the probability mass exactly in half — 50% of the probability lies to the left, 50% to the right. More generally, **quantiles** generalize this idea to any probability level.

Understanding quantiles is fundamental in statistics, ML model evaluation, data exploration, and constructing confidence intervals.

---

### 1.2 Intuition

Think of a probability density function (PDF) like a pile of sand spread along a number line. The **median** is the point where you draw a vertical line that splits the pile into two equal-weight halves. Each half contains exactly 50% of the total mass.

The key insight: **what matters is area under the curve, not the height of the curve.** A tall, narrow peak on the left and a short, wide tail on the right can balance out to have the same median.

> **Key Idea:** The median is determined entirely by the cumulative distribution function (CDF), not by the shape of the PDF at any single point.

---

### 1.3 Formal Definitions

**Definition — $p$-th Quantile:**

The $p$-th quantile of a random variable $X$ with CDF $F(x)$ is the value $q_p$ such that:

$$F(q_p) = P(X \leq q_p) = p$$

**Definition — Median:**

The median is the 0.5-quantile, i.e., the value $q_{0.5}$ satisfying:

$$F(q_{0.5}) = 0.5 \quad \Longleftrightarrow \quad P(X \leq q_{0.5}) = 0.5$$

**Definition — Common Named Quantiles:**

| Quantile Level | Name |
|---|---|
| $q_{0.25}$ | First quartile (Q1) |
| $q_{0.50}$ | Median (Q2) |
| $q_{0.75}$ | Third quartile (Q3) |
| $q_{0.90}$ | 90th percentile |

---

### 1.4 Key Formulas

To find the median (or any quantile) of a continuous distribution:

$$\text{Step 1: Compute } F(x) = \int_{-\infty}^{x} f(t)\, dt$$

$$\text{Step 2: Solve } F(q_p) = p \text{ for } q_p$$

---

### 1.5 Concept Questions — Greatest Median

These questions test whether you understand that **median depends on cumulative area**, not on the shape of the density at any particular point.

---

#### Concept Question 1 — Greatest Median (Plot A)

**Problem:** Three density curves (black, orange, blue) all **coincide up to point $q$**. The black curve has its median at $q$. Which density has the greatest median?

Options: (1) Black  (2) Orange  (3) Blue  (4) All the same  (5) Impossible to tell

**Answer: 4 — All the same.**

**Reasoning, step by step:**

- **Step 1:** Since the three curves coincide entirely up to point $q$, the area under all three curves from $-\infty$ to $q$ is identical.
- **Step 2:** The black curve has median at $q$, which means $F_{\text{black}}(q) = 0.5$, i.e., the area to the left of $q$ is exactly 0.5.
- **Step 3:** Since all curves coincide up to $q$, every curve also accumulates exactly 0.5 area by point $q$.
- **Step 4:** Therefore, $q$ is the median for **all three densities**.

> **Intuition:** It doesn't matter what happens to the right of $q$ — the cumulative area to the left of $q$ is the same for all three curves, and that area equals 0.5.

---

#### Concept Question 2 — Greatest Median (Plot B)

**Problem:** Three curves diverge before reaching point $q$. The black curve has median at $q$. The **blue curve lies above black** before $q$; the **orange curve lies below black** before $q$. Which has the greatest median?

Options: (1) Black  (2) Orange  (3) Blue  (4) All the same  (5) Impossible to tell

**Answer: 2 — Orange has the greatest median.**

**Reasoning, step by step:**

- **Step 1:** Since black has median at $q$, area under black from $-\infty$ to $q$ = 0.5.
- **Step 2:** Blue curve lies **above** black before $q$, so the area under blue up to $q$ is **greater than 0.5**. This means blue has already accumulated more than half its probability by $q$. So blue's median is **to the left of $q$** — it needs less distance to accumulate 0.5.
- **Step 3:** Orange curve lies **below** black before $q$, so the area under orange up to $q$ is **less than 0.5**. Orange needs to go further right to accumulate 0.5 probability. So orange's median is **to the right of $q$**.
- **Conclusion:** Orange has the greatest median ($q_{\text{orange}} > q > q_{\text{blue}}$).

> **Intuition:** A curve that is "shorter" (less dense) up to a given point needs to travel further to the right to accumulate 50% of its area. Hence it has a larger median.

**Common Mistake:** Confusing the height of the density at $q$ with the cumulative area. A taller peak does not imply a larger median — it implies more area accumulates faster, which pushes the median to the **left**, not right.

---

### 1.6 Worked Example — Finding the Median (Problem 1c)

**Problem:** Let $X$ have range $[0,1]$ with PDF $f(x) = 3x^2$. Find the median.

**Step 1: Set up the definition.**

The median $q_{0.5}$ satisfies $F(q_{0.5}) = 0.5$.

**Step 2: Compute the CDF.**

$$F(x) = \int_0^x 3u^2\, du = \left[u^3\right]_0^x = x^3$$

**Step 3: Solve $F(q_{0.5}) = 0.5$.**

$$q_{0.5}^3 = 0.5 \implies q_{0.5} = (0.5)^{1/3} = \frac{1}{2^{1/3}} \approx 0.794$$

**Final Answer:** $q_{0.5} = (0.5)^{1/3} \approx 0.794$

**Interpretation:** 79.4% of the way through the $[0,1]$ interval, the distribution has accumulated half its probability. This makes sense because $f(x) = 3x^2$ is **right-skewed** (more density near 1), so the median is pulled toward the right.

---

## 2. Working with PDFs and CDFs

### 2.1 Concept Overview

A **probability density function (PDF)** $f(x)$ describes the relative likelihood of a continuous random variable taking values near $x$. The **cumulative distribution function (CDF)** $F(x)$ gives the total probability accumulated up to $x$.

---

### 2.2 Fundamental Properties

**PDF Requirements:**
1. $f(x) \geq 0$ for all $x$
2. $\displaystyle\int_{-\infty}^{\infty} f(x)\, dx = 1$ (total probability = 1)

**CDF Definition:**
$$F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t)\, dt$$

**Relationship:**
$$f(x) = F'(x) \quad \text{(PDF is the derivative of CDF)}$$

---

### 2.3 Mean (Expected Value) of a Continuous Random Variable

$$\mu = E[X] = \int_{-\infty}^{\infty} x f(x)\, dx$$

### 2.4 Variance

$$\sigma^2 = \text{Var}(X) = E[(X-\mu)^2] = \int_{-\infty}^{\infty} (x-\mu)^2 f(x)\, dx$$

**Equivalent computational formula:**
$$\sigma^2 = E[X^2] - (E[X])^2$$

### 2.5 Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

---

### 2.6 Worked Example — Full Analysis of a PDF (Problem 1, all parts)

**Problem:** $X$ has range $[0,1]$ with PDF $f(x) = cx^2$.

---

#### Part (a): Find $c$

**Goal:** Use the normalization requirement $\int_0^1 f(x)\, dx = 1$.

**Step 1:** Set up the integral.
$$\int_0^1 cx^2\, dx = 1$$

**Step 2:** Evaluate.
$$c \cdot \left[\frac{x^3}{3}\right]_0^1 = c \cdot \frac{1}{3} = 1$$

**Step 3:** Solve for $c$.
$$c = 3$$

**Final Answer:** $f(x) = 3x^2$ on $[0,1]$.

> **Why this works:** Every valid PDF must integrate to exactly 1 (total probability). Finding $c$ via this requirement is called **normalizing** the distribution.

---

#### Part (b): Find Mean, Variance, and Standard Deviation

**Mean:**

$$\mu = \int_0^1 x \cdot 3x^2\, dx = \int_0^1 3x^3\, dx = 3 \cdot \left[\frac{x^4}{4}\right]_0^1 = \frac{3}{4}$$

**Variance:**

We use $\sigma^2 = \int_0^1 (x - \mu)^2 f(x)\, dx$ with $\mu = 3/4$:

$$\sigma^2 = \int_0^1 \left(x - \frac{3}{4}\right)^2 \cdot 3x^2\, dx$$

Expand $(x - 3/4)^2 = x^2 - \frac{3x}{2} + \frac{9}{16}$:

$$\sigma^2 = 3\int_0^1 \left(x^4 - \frac{3x^3}{2} + \frac{9x^2}{16}\right) dx$$

$$= 3\left[\frac{x^5}{5} - \frac{3x^4}{8} + \frac{9x^3}{48}\right]_0^1 = 3\left(\frac{1}{5} - \frac{3}{8} + \frac{3}{16}\right)$$

$$= 3\left(\frac{16 - 30 + 15}{80}\right) = 3 \cdot \frac{1}{80} = \frac{3}{80}$$

**Alternative shortcut using $\sigma^2 = E[X^2] - \mu^2$:**

$$E[X^2] = \int_0^1 x^2 \cdot 3x^2\, dx = 3\int_0^1 x^4\, dx = \frac{3}{5}$$

$$\sigma^2 = \frac{3}{5} - \left(\frac{3}{4}\right)^2 = \frac{3}{5} - \frac{9}{16} = \frac{48 - 45}{80} = \frac{3}{80}$$

**Standard Deviation:**

$$\sigma = \sqrt{\frac{3}{80}} = \frac{1}{4}\sqrt{\frac{3}{5}} \approx 0.194$$

**Summary:**

| Quantity | Value |
|---|---|
| Mean $\mu$ | $3/4 = 0.75$ |
| Variance $\sigma^2$ | $3/80 \approx 0.0375$ |
| Std Dev $\sigma$ | $\approx 0.194$ |

> **Interpretation:** The distribution $f(x) = 3x^2$ is right-skewed (most mass near $x=1$). The mean $\mu = 0.75$ is greater than the median $q_{0.5} \approx 0.794$... wait — actually this is one of those interesting cases. Let's double-check: mean = 0.75 and median $\approx 0.794$. Since the PDF is right-skewed (heavier density toward 1), the **median is higher than the mean** here.

---

#### Part (c): Median (already solved in Section 1.6)

$$q_{0.5} = (0.5)^{1/3} \approx 0.794$$

---

#### Part (d): Standard Deviation of the Sample Mean

**Setup:** $X_1, X_2, \ldots, X_{16}$ are independent, identically distributed (i.i.d.) copies of $X$. Let $\bar{X} = \frac{1}{16}\sum_{i=1}^{16} X_i$.

**Step 1:** Variance of a sum of independent variables.

$$\text{Var}(X_1 + X_2 + \cdots + X_{16}) = \text{Var}(X_1) + \text{Var}(X_2) + \cdots + \text{Var}(X_{16}) = 16 \cdot \text{Var}(X)$$

> **Why?** Independence allows variances to add. This is a fundamental property.

**Step 2:** Variance of the average.

$$\text{Var}(\bar{X}) = \text{Var}\!\left(\frac{X_1 + \cdots + X_{16}}{16}\right) = \frac{1}{16^2} \cdot 16 \cdot \text{Var}(X) = \frac{\text{Var}(X)}{16}$$

**Step 3:** Standard deviation of $\bar{X}$.

$$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{16}} = \frac{\sigma}{4} = \frac{0.194}{4} \approx 0.0485$$

**General Formula:**
$$\boxed{\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}}$$

> **Interpretation:** Averaging $n$ independent observations reduces the standard deviation by a factor of $\sqrt{n}$. With 16 observations, the variability of the average is **4 times smaller** than the variability of a single observation. This is the mathematical reason why larger samples give more reliable estimates.

---

#### Part (e): Expected Value of a Transformation

**Problem:** Let $Y = X^4$. Find $E[Y]$.

**The Law of the Unconscious Statistician (LOTUS):**

If $Y = g(X)$, then:
$$E[g(X)] = \int_{-\infty}^{\infty} g(x) f(x)\, dx$$

You do **not** need to find the distribution of $Y$ first.

**Step 1:** Apply LOTUS.

$$E[Y] = E[X^4] = \int_0^1 x^4 \cdot 3x^2\, dx = \int_0^1 3x^6\, dx$$

**Step 2:** Evaluate.

$$= 3 \cdot \left[\frac{x^7}{7}\right]_0^1 = \frac{3}{7}$$

**Final Answer:** $E[Y] = \dfrac{3}{7}$

---

#### Part (f): PDF of the Transformed Variable $Y = X^4$

**Two methods are shown.**

---

**Method 1 — CDF Method (recommended for beginners)**

**Step 1:** Express the CDF of $Y$ in terms of $X$.

$$F_Y(y) = P(Y \leq y) = P(X^4 \leq y) = P\!\left(X \leq y^{1/4}\right) = F_X(y^{1/4})$$

> **Why?** Since $X \geq 0$ and $y = x^4$ is monotonically increasing on $[0,1]$, the inequality $X^4 \leq y$ is equivalent to $X \leq y^{1/4}$.

**Step 2:** Substitute $F_X(x) = x^3$.

$$F_Y(y) = \left(y^{1/4}\right)^3 = y^{3/4}$$

**Step 3:** Differentiate to get the PDF.

$$f_Y(y) = F_Y'(y) = \frac{d}{dy}\, y^{3/4} = \frac{3}{4} y^{-1/4} = \frac{3}{4y^{1/4}}$$

---

**Method 2 — Change of Variables Formula**

**Setup:** $y = x^4$, so $dy = 4x^3\, dx$, giving $dx = \dfrac{dy}{4x^3} = \dfrac{dy}{4y^{3/4}}$.

**Step 1:** Substitute into the original PDF expression.

$$f_X(x)\, dx = 3x^2\, dx = 3(y^{1/4})^2 \cdot \frac{dy}{4y^{3/4}} = \frac{3y^{1/2}}{4y^{3/4}}\, dy = \frac{3}{4y^{1/4}}\, dy$$

**Step 2:** Read off the PDF.

$$f_Y(y) = \frac{3}{4y^{1/4}}, \quad y \in [0,1]$$

**Both methods give the same answer:**

$$\boxed{f_Y(y) = \frac{3}{4}y^{-1/4}, \quad y \in [0,1]}$$

> **Intuition:** The transformation $Y = X^4$ "squishes" values near 0 (since small $x$ maps to even smaller $x^4$), making low values of $Y$ much more likely. This is reflected in the fact that $f_Y(y) \to \infty$ as $y \to 0^+$ — the density "piles up" near zero.

**Common Mistake:** Forgetting the Jacobian factor $|dx/dy|$ in the change-of-variables method. The PDF must account for how the transformation stretches or compresses the probability mass.

---

## 3. Transformations of Random Variables

### 3.1 General Framework

Given $Y = g(X)$ where $g$ is monotone and differentiable:

$$f_Y(y) = f_X\!\left(g^{-1}(y)\right) \cdot \left|\frac{d}{dy} g^{-1}(y)\right|$$

Or equivalently using the CDF method:

1. Find $F_Y(y) = P(Y \leq y)$ by inverting the inequality.
2. Differentiate: $f_Y(y) = F_Y'(y)$.

### 3.2 Quick Reference: CDF Method Steps

| Step | Action |
|---|---|
| 1 | Write $F_Y(y) = P(g(X) \leq y)$ |
| 2 | Invert the inequality using properties of $g$ |
| 3 | Express as $F_X(\text{something})$ |
| 4 | Differentiate with respect to $y$ |

---

## 4. Histograms: Frequency vs. Density

### 4.1 Concept Overview

Histograms are visual summaries of data. There are two types:
- **Frequency histogram:** Bar height = count of data points in the bin
- **Density histogram:** Bar height = (count / total) / bin width = **relative frequency density**

The critical difference: **density histograms remain comparable across different bin widths.** Frequency histograms do not.

---

### 4.2 Why Density Histograms Matter

The area of each bar in a density histogram equals the **fraction of data** in that bin:

$$\text{Area of bar} = \text{height} \times \text{width} = \frac{\text{count in bin}}{n \times \text{bin width}} \times \text{bin width} = \frac{\text{count in bin}}{n}$$

So: $\sum (\text{all bar areas}) = 1$, just like a PDF.

This means density histograms can be directly compared to probability density functions.

---

### 4.3 Worked Example — Problem 2

**Data:**
```
1.0  1.2  1.3  1.6  1.6
2.1  2.2  2.6  2.7  3.1
3.2  3.4  3.8  3.9  3.9
```
Total: $n = 15$ data points.

---

#### Part (a): Equal-Width Bins of Width 0.5 (right-closed)

Bins: $(0, 0.5]$, $(0.5, 1.0]$, $(1.0, 1.5]$, $(1.5, 2.0]$, $(2.0, 2.5]$, $(2.5, 3.0]$, $(3.0, 3.5]$, $(3.5, 4.0]$

**Count data in each bin:**

| Bin | Data points | Count | Freq. Height | Density Height |
|---|---|---|---|---|
| $(0, 0.5]$ | — | 0 | 0 | 0 |
| $(0.5, 1.0]$ | 1.0 | 1 | 1 | $1/(15 \times 0.5) = 0.133$ |
| $(1.0, 1.5]$ | 1.2, 1.3 | 2 | 2 | $2/(15 \times 0.5) = 0.267$ |
| $(1.5, 2.0]$ | 1.6, 1.6 | 2 | 2 | $0.267$ |
| $(2.0, 2.5]$ | 2.1, 2.2 | 2 | 2 | $0.267$ |
| $(2.5, 3.0]$ | 2.6, 2.7 | 2 | 2 | $0.267$ |
| $(3.0, 3.5]$ | 3.1, 3.2, 3.4 | 3 | 3 | $3/(15 \times 0.5) = 0.400$ |
| $(3.5, 4.0]$ | 3.8, 3.9, 3.9 | 3 | 3 | $0.400$ |

> **Verification:** Total area $= 0.5 \times (0 + 1 + 2 + 2 + 2 + 2 + 3 + 3)/15 = 0.5 \times 15/15 = 0.5 \times 1 = 1$ ✓

---

#### Part (b): Unequal-Width Bins with Edges $0, 1, 3, 4$

Bins: $(0, 1]$, $(1, 3]$, $(3, 4]$ with widths 1, 2, 1.

**Count data:**

| Bin | Width | Data points | Count | Frequency Height | Density Height |
|---|---|---|---|---|---|
| $(0, 1]$ | 1 | 1.0 | 1 | 1 | $1/(15 \times 1) = 0.067$ |
| $(1, 3]$ | 2 | 1.2,1.3,1.6,1.6,2.1,2.2,2.6,2.7 | 8 | 8 | $8/(15 \times 2) = 0.267$ |
| $(3, 4]$ | 1 | 3.1,3.2,3.4,3.8,3.9,3.9 | 6 | 6 | $6/(15 \times 1) = 0.400$ |

---

#### Part (c): Why Density is Better for Unequal Bins

With unequal bins, a **frequency histogram** is misleading. The wide bin $(1, 3]$ contains 8 data points and the narrow bin $(3, 4]$ contains 6, but the frequency bar for the wide bin is much taller — making the $[1,3]$ region appear far more concentrated than it is.

The **density histogram** corrects for this by dividing by bin width. The height represents probability *per unit* of the x-axis, making the visual area of each bar proportional to the fraction of data — just like a PDF.

> **Rule:** Always use density histograms when bin widths are unequal. Frequency histograms are only visually comparable when bin widths are identical.

---

## 5. Normal Distributions & the Empirical Rule

### 5.1 Concept Overview

The **normal distribution** (also called Gaussian distribution) is the most important continuous distribution in probability and statistics. It arises naturally as the limiting distribution of averages (via the CLT), making it central to virtually all of classical statistics and many ML methods.

---

### 5.2 Formal Definition

A random variable $X$ is **normally distributed** with mean $\mu$ and standard deviation $\sigma$, written $X \sim N(\mu, \sigma^2)$, if its PDF is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in (-\infty, \infty)$$

**Standard Normal:** $Z \sim N(0, 1)$ has PDF:

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

and CDF:

$$\Phi(z) = \int_{-\infty}^z \phi(t)\, dt$$

---

### 5.3 The Empirical Rule (68-95-99 Rule)

For any $X \sim N(\mu, \sigma^2)$:

$$P(\mu - \sigma < X < \mu + \sigma) \approx 0.68 \quad \text{(within 1 standard deviation)}$$
$$P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.95 \quad \text{(within 2 standard deviations)}$$
$$P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 0.99 \quad \text{(within 3 standard deviations)}$$

**Visual summary:**

```
          68%
     ←————————————→
          95%
     ←——————————————————→
               99%
     ←——————————————————————→
  ——|——|——|——μ——|——|——|——
  -3σ -2σ -σ  0  σ  2σ  3σ
```

---

### 5.4 Tail Probabilities (derived from the Empirical Rule)

By symmetry of the normal distribution:

| Event | Probability |
|---|---|
| $P(X > \mu + \sigma)$ | $\approx (1 - 0.68)/2 = 0.16$ |
| $P(X < \mu - \sigma)$ | $\approx 0.16$ |
| $P(X > \mu + 2\sigma)$ | $\approx (1 - 0.95)/2 = 0.025$ |
| $P(X < \mu - 2\sigma)$ | $\approx 0.025$ |
| $P(X > \mu + 3\sigma)$ | $\approx (1 - 0.99)/2 = 0.005$ |

---

### 5.5 Concept Questions — Normal Distributions

#### Concept Question 1(a)

**Question:** $P(-\sigma < X - \mu < \sigma)$ is approximately:

(i) 0.025  (ii) 0.16  (iii) 0.68  (iv) 0.84  (v) 0.95

**Answer: (iii) 0.68**

**Reasoning:** $-\sigma < X - \mu < \sigma$ is equivalent to $\mu - \sigma < X < \mu + \sigma$, which is "within 1 standard deviation of the mean." By the empirical rule, this has probability $\approx 68\%$.

---

#### Concept Question 1(b)

**Question:** $P(X > \mu + 2\sigma)$ is approximately:

(i) 0.025  (ii) 0.16  (iii) 0.68  (iv) 0.84  (v) 0.95

**Answer: (i) 0.025**

**Reasoning, step by step:**

- Step 1: By the empirical rule, $P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.95$.
- Step 2: The remaining probability is $1 - 0.95 = 0.05$, split symmetrically between two tails.
- Step 3: $P(X > \mu + 2\sigma) \approx 0.05/2 = 0.025$.

> **Intuition:** Being more than 2 standard deviations above the mean is a rare event — it occurs only about 2.5% of the time.

---

### 5.6 Standard Normal Quantiles (R functions)

In R (and equivalent Python/scipy):

- `qnorm(p)` returns $q_p$, the $p$-th quantile of $N(0,1)$.
- `pnorm(z)` returns $\Phi(z) = P(Z \leq z)$ for $Z \sim N(0,1)$.

**Key quantiles of $N(0,1)$:**

| Quantile level $p$ | Value $q_p$ |
|---|---|
| 0.25 | $\approx -0.6745$ |
| 0.50 | $0$ (by symmetry) |
| 0.75 | $\approx 0.6745$ |
| 0.025 | $\approx -1.96$ |
| 0.975 | $\approx 1.96$ |

**Verification:** $\Phi(q_p) = p$ by definition. For example:

$$\text{pnorm}(-0.6745) = 0.25 \checkmark$$
$$\text{pnorm}(0) = 0.5 \checkmark$$
$$\text{pnorm}(0.6745) = 0.75 \checkmark$$

---

## 6. Standardization

### 6.1 Concept Overview

**Standardization** transforms any normal random variable $X \sim N(\mu, \sigma^2)$ into the standard normal $Z \sim N(0,1)$. This allows us to use a single table (or a single set of R/Python functions) to compute probabilities for any normal distribution.

---

### 6.2 Formal Definition

If $X \sim N(\mu, \sigma^2)$, define:

$$Z = \frac{X - \mu}{\sigma}$$

Then $Z \sim N(0, 1)$.

> **Intuition:** Subtracting $\mu$ shifts the distribution to be centered at 0. Dividing by $\sigma$ rescales it so the spread is exactly 1. This is like converting temperatures from Fahrenheit to Celsius — a linear transformation that preserves shape.

---

### 6.3 Proof that $Z$ has Mean 0 and Variance 1

**Problem 1 (Class 6b):** Use algebraic properties of expectation and variance to verify.

**Properties used:**

$$E[aX + b] = aE[X] + b$$
$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

**Proof — Mean:**

$$E[Z] = E\!\left[\frac{X - \mu}{\sigma}\right] = \frac{1}{\sigma} E[X - \mu] = \frac{1}{\sigma}(E[X] - \mu) = \frac{1}{\sigma}(\mu - \mu) = 0 \checkmark$$

**Proof — Variance:**

$$\text{Var}(Z) = \text{Var}\!\left(\frac{X - \mu}{\sigma}\right) = \frac{1}{\sigma^2} \text{Var}(X - \mu) = \frac{1}{\sigma^2} \text{Var}(X) = \frac{\sigma^2}{\sigma^2} = 1 \checkmark$$

> **Note:** Adding a constant ($-\mu$) does not change variance: $\text{Var}(X - \mu) = \text{Var}(X)$.

---

### 6.4 Using Standardization to Compute Probabilities

**General method:**

To find $P(a < X < b)$ when $X \sim N(\mu, \sigma^2)$:

$$P(a < X < b) = P\!\left(\frac{a-\mu}{\sigma} < Z < \frac{b-\mu}{\sigma}\right) = \Phi\!\left(\frac{b-\mu}{\sigma}\right) - \Phi\!\left(\frac{a-\mu}{\sigma}\right)$$

---

## 7. The Central Limit Theorem

### 7.1 Concept Overview

The **Central Limit Theorem (CLT)** is arguably the most important theorem in statistics. It says that the average of many independent, identically distributed random variables (regardless of their underlying distribution) is approximately normally distributed.

This is why the normal distribution appears everywhere: measurements, errors, sample averages all tend to be approximately normal, even when the underlying data is not.

---

### 7.2 Formal Statement

**Theorem (Central Limit Theorem):**

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2 < \infty$. Let $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ be their sample mean.

Then, as $n \to \infty$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

Equivalently:

$$\bar{X}_n \overset{\text{approx}}{\sim} N\!\left(\mu,\; \frac{\sigma^2}{n}\right) \quad \text{for large } n$$

**Key parameters of the sample mean's distribution:**

$$E[\bar{X}_n] = \mu \qquad \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \qquad \sigma_{\bar{X}_n} = \frac{\sigma}{\sqrt{n}}$$

---

### 7.3 Intuition

Imagine rolling a die once — you might get any value from 1 to 6. But if you average 100 die rolls, the result is almost certainly very close to 3.5. The more rolls you average, the more tightly the average clusters around 3.5, and the shape of its distribution looks more and more like a bell curve.

The CLT says this happens for **any** distribution with finite variance — not just dice.

> **Deep Intuition:** Independent errors (in either direction) tend to cancel when averaged. The bell curve naturally emerges from this averaging and cancellation process.

---

### 7.4 Worked Example — Poll Problem (Problem 2b)

**Setup:** 50% of the population supports the team of Alessandre, Gabriel, Sarah, and So Hee. A poll asks 400 random people. What is the probability that **at least 55%** of those polled prefer the team?

**Step 1: Model the situation.**

Let $X_i = 1$ if the $i$th person supports the team, $X_i = 0$ otherwise. Then $X_i \sim \text{Bernoulli}(0.5)$.

The fraction of those polled who support the team is:
$$\bar{X} = \frac{1}{400}\sum_{i=1}^{400} X_i$$

**Step 2: Find the distribution of $\bar{X}$.**

For Bernoulli(0.5): $\mu = 0.5$, $\sigma^2 = p(1-p) = 0.25$.

$$E[\bar{X}] = 0.5 \qquad \sigma_{\bar{X}} = \sqrt{\frac{0.25}{400}} = \frac{0.5}{20} = 0.025$$

**Step 3: Standardize.**

By the CLT, $\bar{X} \approx N(0.5, 0.025^2)$. Standardizing:

$$P(\bar{X} > 0.55) = P\!\left(Z > \frac{0.55 - 0.5}{0.025}\right) = P(Z > 2)$$

**Step 4: Use the empirical rule.**

$$P(Z > 2) \approx 0.025$$

**Final Answer:** There is approximately a **2.5% chance** that 55% or more of those polled prefer the team.

**Interpretation:** Even though the team has exactly 50% support in the population, a sample of 400 will occasionally show 55% support just by random chance — but this is relatively unlikely (only a 1-in-40 chance).

---

### 7.5 Worked Example — Jen Poll (Problem 2c)

**Setup:** 25% of the population supports Jen. Poll of 400 people. Find $P(\bar{J} < 0.20)$.

**Step 1: Model.**

$J_i \sim \text{Bernoulli}(0.25)$, so $\mu = 0.25$, $\sigma^2 = (0.25)(0.75) = 3/16$.

$$\sigma_{\bar{J}} = \sqrt{\frac{3/16}{400}} = \sqrt{\frac{3}{6400}} = \frac{\sqrt{3}}{80}$$

**Step 2: Standardize.**

$$P(\bar{J} < 0.20) = P\!\left(Z < \frac{0.20 - 0.25}{\sqrt{3}/80}\right) = P\!\left(Z < \frac{-0.05\cdot 80}{\sqrt{3}}\right) = P\!\left(Z < \frac{-4}{\sqrt{3}}\right)$$

**Step 3: Compute numerically.**

$$\frac{-4}{\sqrt{3}} \approx \frac{-4}{1.732} \approx -2.309$$

$$P(Z < -2.309) = \Phi(-2.309) \approx 0.0105$$

**Final Answer:** There is approximately a **1.05% chance** that fewer than 20% of those polled support Jen.

---

## 8. Sampling Approximations Using the CLT

### 8.1 Worked Example — Approximating the Standard Normal with Dice (Problem 3)

**Problem:** How can you approximate a single draw from $N(0,1)$ using 9 rolls of a 10-sided die?

**Given:** For a single roll of a 10-sided die (faces 1–10):
- $\mu = 5.5$
- $\sigma^2 = 8.25$

**Step 1: Apply the CLT.**

Let $\bar{X}$ be the average of 9 rolls. By CLT:

$$\bar{X} \approx N\!\left(5.5,\; \frac{8.25}{9}\right)$$

$$\sigma_{\bar{X}} = \sqrt{\frac{8.25}{9}} \approx 0.957$$

**Step 2: Standardize the average.**

$$Z = \frac{\bar{X} - 5.5}{0.957} \approx N(0, 1)$$

**Step 3: Procedure to generate one standard normal sample.**

1. Roll a 10-sided die 9 times.
2. Compute the average $\bar{x}$ of the 9 rolls.
3. Compute $z = \dfrac{\bar{x} - 5.5}{0.957}$.

The value $z$ is one approximate draw from $N(0, 1)$.

**Why it works:** Although die rolls are **not** normally distributed (they're discrete uniform on $\{1,\ldots,10\}$), the average of 9 of them is approximately normal by the CLT. 9 is enough for a reasonable approximation.

---

### 8.2 Bonus Problem — Accountant Rounding Errors

**Problem:** An accountant rounds each entry to the nearest dollar. The rounding error is $\sim \text{Uniform}(-0.5, 0.5)$. What is the probability that the total error in **300 entries exceeds $5** in magnitude?

**Step 1: Model the individual errors.**

$X_j \sim U(-0.5, 0.5)$, so:
$$E[X_j] = 0 \qquad \text{Var}(X_j) = \frac{(0.5 - (-0.5))^2}{12} = \frac{1}{12}$$

> **Variance of Uniform$(a,b)$:** $\sigma^2 = (b-a)^2/12$.

**Step 2: Find the distribution of total error $S = \sum_{j=1}^{300} X_j$.**

$$E[S] = 0 \qquad \text{Var}(S) = 300 \cdot \frac{1}{12} = 25 \qquad \sigma_S = 5$$

**Step 3: Standardize using CLT.**

$$S/5 \approx Z \sim N(0,1)$$

**Step 4: Compute the probability.**

$$P(|S| > 5) = P(S < -5 \text{ or } S > 5) = P(|Z| > 1) \approx 1 - 0.68 = 0.32$$

**Final Answer:** There is approximately a **32% probability** that the total rounding error exceeds $5 in absolute value.

**Interpretation:** Despite each individual rounding error being tiny (at most 50 cents), 300 of them can accumulate significantly. The standard deviation of the total error is $5 — meaning there's a 32% chance the accountant is off by more than $5 on 300 entries.

---

## 9. Appendix: Formal Proofs

### 9.1 The Law of Large Numbers (Formal Statement)

**Theorem (Weak Law of Large Numbers):**

Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$ and finite variance $\sigma^2$. Then for any $\varepsilon > 0$:

$$P\!\left(|\bar{X}_n - \mu| < \varepsilon\right) \to 1 \quad \text{as } n \to \infty$$

In words: the sample mean converges in probability to the true mean.

---

### 9.2 The Chebyshev Inequality

**Theorem:** Let $Y$ be a random variable with mean $\mu$ and variance $\sigma^2$. For any $a > 0$:

$$P(|Y - \mu| \geq a) \leq \frac{\text{Var}(Y)}{a^2}$$

> **Intuition:** If the variance of $Y$ is small, then $Y$ is unlikely to be far from its mean. Chebyshev makes this quantitative. Notably, it works for **any** distribution — not just normal.

---

#### Proof of the Chebyshev Inequality

Without loss of generality, assume $\mu = 0$ (replacing $Y$ with $Y - \mu$ does not change variance).

$$P(|Y| \geq a) = \int_{|y| \geq a} f(y)\, dy = \int_{-\infty}^{-a} f(y)\, dy + \int_a^{\infty} f(y)\, dy$$

On the region $|y| \geq a$, we have $y^2 \geq a^2$, so $\frac{y^2}{a^2} \geq 1$. Therefore:

$$P(|Y| \geq a) \leq \int_{|y| \geq a} \frac{y^2}{a^2} f(y)\, dy$$

Since $\frac{y^2}{a^2} f(y) \geq 0$ everywhere, adding the integral over $|y| < a$ only increases the value:

$$P(|Y| \geq a) \leq \int_{-\infty}^{\infty} \frac{y^2}{a^2} f(y)\, dy = \frac{1}{a^2} \int_{-\infty}^{\infty} y^2 f(y)\, dy = \frac{\text{Var}(Y)}{a^2} \quad \square$$

---

### 9.3 Proof of the Law of Large Numbers Using Chebyshev

**Proof:**

Apply Chebyshev to $Y = \bar{X}_n$:

$$P(|\bar{X}_n - \mu| \geq a) \leq \frac{\text{Var}(\bar{X}_n)}{a^2} = \frac{\sigma^2/n}{a^2} = \frac{\sigma^2}{na^2}$$

As $n \to \infty$, the right side $\to 0$. Therefore:

$$P(|\bar{X}_n - \mu| \geq a) \to 0 \quad \Longleftrightarrow \quad P(|\bar{X}_n - \mu| < a) \to 1 \quad \square$$

> **Key insight:** The variance of the sample mean $\bar{X}_n$ is $\sigma^2/n$, which shrinks to 0 as $n$ grows. Chebyshev then forces the probability of being far from $\mu$ to vanish.

---

### 9.4 Formal Statement: Density Histograms Converge to the PDF

**Statement 1:** Let $\hat{p}_k$ be the fraction of $n$ i.i.d. samples in bin $k$, and $p_k$ be the true probability of falling in bin $k$. Then for any $a > 0$:

$$P(|\hat{p}_k - p_k| < a) \to 1 \quad \text{as } n \to \infty$$

**Proof sketch:** Each observation is in bin $k$ with probability $p_k$ (a Bernoulli trial). The fraction $\hat{p}_k$ is the sample mean of these Bernoulli indicators. Applying the LLN gives the result.

**Statement 2:** The same convergence holds **simultaneously** for all $m$ bins (for any finite $m$).

**Proof sketch:** For each bin, find $n$ large enough so that $P(|\hat{p}_k - p_k| < a) > 1 - \alpha/m$. By inclusion-exclusion (union bound):

$$P(\text{all bins simultaneously close}) \geq 1 - m \cdot \frac{\alpha}{m} = 1 - \alpha$$

Since $\alpha$ can be made arbitrarily small by increasing $n$, the result follows.

**Statement 3:** If $f(x)$ is a continuous PDF on $[a,b]$, then with enough data and small enough bin width, the density histogram converges uniformly to $f(x)$ with probability approaching 1.

**Proof idea:** For small bin width $\Delta x$ around $x$, the bin probability is $\approx f(x)\Delta x$. By Statement 2, the fraction of data in the bin is also $\approx f(x)\Delta x$. Hence the histogram height $\approx f(x)$.

---

### 9.5 Technical Note: Why Variance Must be Finite

The LLN and CLT both require **finite variance**. Some distributions (like the **Cauchy distribution**) have no finite mean or variance — for these, the LLN does not hold. Sample averages of Cauchy random variables do not converge; they remain as variable as individual observations.

This is rarely an issue in practice, but it matters in heavy-tailed settings (e.g., certain financial return models, network traffic).

---

## 10. Strategy Under Uncertainty: The Desperation Problem

### 10.1 Problem Setup

- You have $100 and need $1000 by tomorrow.
- Each bet: wager $k$; win $k$ with probability $p$; lose $k$ with probability $1-p$.

Two strategies:
- **Maximal:** Bet as much as possible each round (up to what you still need).
- **Minimal:** Bet a small fixed amount (e.g., $5) each round.

---

### 10.2 Analysis

**Minimal Strategy + LLN:**

If you make many small bets, the LLN says your average winnings per bet will converge to the expected value of one bet.

For a $5 bet: $E[\text{win}] = 5p - 5(1-p) = 5(2p - 1)$

- If $p = 0.45$: $E[\text{win}] = 5(2(0.45)-1) = 5(-0.10) = -\$0.50$
- If $p = 0.80$: $E[\text{win}] = 5(2(0.80)-1) = 5(0.60) = +\$3.00$

---

### 10.3 Decisions

**Case (a): $p = 0.45$ (unfavorable game)**

Expected winnings per bet are **negative**. Making many bets guarantees you lose money on average. Use the **Maximal strategy** to minimize the number of bets and maximize the chance of a lucky streak.

> You're not likely to win, but going all-in gives you the best (though small) chance.

**Answer: Maximal strategy.**

---

**Case (b): $p = 0.80$ (favorable game)**

Expected winnings per bet are **positive ($3/bet)**. Making many bets lets the LLN work in your favor — you'll almost certainly profit. Use the **Minimal strategy** to make as many bets as possible.

> With a positive edge, grinding small bets is the winning approach.

**Answer: Minimal strategy.**

---

### 10.4 Key Insight

| Condition | Strategy | Reason |
|---|---|---|
| $p < 0.5$ (unfavorable odds) | Maximal | Avoid many bets; go for the quick win |
| $p > 0.5$ (favorable odds) | Minimal | Let LLN grind out the positive expected value |
| $p = 0.5$ (fair game) | Doesn't matter | Same expected outcome either way |

> **Connection to AI/ML:** This is exactly the explore/exploit tradeoff in reinforcement learning. When the reward signal is positive, exploit it consistently (minimal strategy). When negative, explore a different approach quickly (maximal strategy, essentially "go for broke").

---

## 11. Quick Reference Summary

### Core Formulas

| Concept | Formula |
|---|---|
| Median | $F(q_{0.5}) = 0.5$ |
| $p$-th quantile | $F(q_p) = p$ |
| PDF normalization | $\int_{-\infty}^{\infty} f(x)\, dx = 1$ |
| Mean | $\mu = \int x f(x)\, dx$ |
| Variance | $\sigma^2 = \int (x-\mu)^2 f(x)\, dx = E[X^2] - \mu^2$ |
| Variance of average | $\text{Var}(\bar{X}_n) = \sigma^2/n$ |
| Std dev of average | $\sigma_{\bar{X}} = \sigma/\sqrt{n}$ |
| Standardization | $Z = (X - \mu)/\sigma$ |
| Normal 68-95-99 rule | $P(\|X-\mu\| \leq k\sigma) \approx 0.68, 0.95, 0.99$ for $k=1,2,3$ |
| Chebyshev | $P(\|Y-\mu\| \geq a) \leq \sigma^2/a^2$ |
| CLT | $\bar{X}_n \overset{\text{approx}}{\sim} N(\mu, \sigma^2/n)$ |

---

### Key Distributions in Class 6

| Distribution | Mean | Variance | Notes |
|---|---|---|---|
| $\text{Uniform}(a,b)$ | $(a+b)/2$ | $(b-a)^2/12$ | Used for rounding errors |
| $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ | Used in poll problems |
| $N(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ | Normal distribution |
| $N(0, 1)$ | $0$ | $1$ | Standard normal |

---

### CLT Application Checklist

1. **Identify** $X_i$: what are the i.i.d. random variables?
2. **Find** $\mu = E[X_i]$ and $\sigma^2 = \text{Var}(X_i)$
3. **Identify** $n$: how many are you averaging/summing?
4. **Compute** $E[\bar{X}] = \mu$ and $\sigma_{\bar{X}} = \sigma/\sqrt{n}$
5. **Standardize:** $Z = (\bar{X} - \mu)/\sigma_{\bar{X}} \approx N(0,1)$
6. **Look up** $P(Z \leq z) = \Phi(z)$ using table or `pnorm(z)` in R

---

### Common Mistakes

| Mistake | Correction |
|---|---|
| Confusing PDF height with probability | Probability = **area** under the PDF, not height |
| Forgetting the Jacobian in change-of-variables | Always multiply by $\|dx/dy\|$ |
| Using frequency histogram for unequal bins | Always use density histogram with unequal bins |
| Thinking CLT requires normal inputs | CLT works for **any** distribution with finite variance |
| Applying CLT for small $n$ | CLT is approximate; rule of thumb is $n \geq 30$ |
| Confusing $\text{Var}(\bar{X})$ with $\text{Var}(X)$ | $\text{Var}(\bar{X}) = \text{Var}(X)/n$ |
| Assuming large sample mean = large mean | Confusion: $\sigma_{\bar{X}} \to 0$ as $n \to \infty$, not $\bar{X}$ itself |

---

*End of MIT 18.05 Class 6 Study Notes*
*Topics: Medians, Quantiles, PDFs/CDFs, Transformations, Normal Distributions, Standardization, CLT, LLN, Chebyshev Inequality*
