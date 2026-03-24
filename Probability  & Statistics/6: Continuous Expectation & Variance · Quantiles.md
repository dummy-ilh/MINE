##  6: Continuous Expectation & Variance · Quantiles · Law of Large Numbers · Central Limit Theorem
---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [Expected Value — Continuous Case](#2-expected-value--continuous-case)
3. [Variance — Continuous Case](#3-variance--continuous-case)
4. [Quantiles, Percentiles, and Medians](#4-quantiles-percentiles-and-medians)
5. [The Law of Large Numbers (LoLN)](#5-the-law-of-large-numbers-loln)
6. [Histograms and the LoLN](#6-histograms-and-the-loln)
7. [The Central Limit Theorem (CLT)](#7-the-central-limit-theorem-clt)
8. [Applications of the CLT](#8-applications-of-the-clt)
9. [How Large Does $n$ Need to Be?](#9-how-large-does-n-need-to-be)
10. [Complete Distribution Reference Table](#10-complete-distribution-reference-table)
11. [Common Mistakes Reference](#11-common-mistakes-reference)
12. [Quick Summary & Formula Sheet](#12-quick-summary--formula-sheet)

---

## 1. Learning Goals

By the end of this class you should be able to:

1. **Compute** the expected value and variance of continuous random variables using integration.
2. **Prove** that $E[X] = \mu$ and $\text{Var}(X) = \sigma^2$ for $X \sim N(\mu, \sigma^2)$.
3. **Define** and **compute** quantiles (percentiles/medians) for both discrete and continuous distributions.
4. **State** the Law of Large Numbers precisely and explain what it guarantees.
5. **Construct** and interpret density histograms correctly.
6. **State** the Central Limit Theorem and identify its key parameters.
7. **Apply** the CLT to approximate probabilities for sums and averages of i.i.d. random variables.
8. **Explain** the polling margin-of-error formula as a CLT application.
9. **Recognise** when the CLT approximation is reliable (and when it requires larger $n$).

---

## 2. Expected Value — Continuous Case

### 2.1 Concept Overview

The expected value of a continuous random variable has identical meaning to the discrete case — it is the **probability-weighted average** of all possible values, and the **centre of mass** of the distribution. The only change is mechanical: we replace sums with integrals.

> **The fundamental transition:** Discrete $\sum x_i p(x_i)$ $\longrightarrow$ Continuous $\int x f(x)\, dx$

This works because $f(x)\, dx$ represents the probability that $X$ falls in an infinitesimal interval of width $dx$ around $x$, exactly analogous to the probability mass $p(x_i)$ at a discrete point.

---

### 2.2 Formal Definition

> **Definition (Expected Value — Continuous):** Let $X$ be a continuous random variable with range $[a, b]$ and pdf $f(x)$. The **expected value** of $X$ is:
>
> $$\boxed{E[X] = \int_a^b x\, f(x)\, dx}$$
>
> For an unbounded range (e.g., $[0, \infty)$ or $(-\infty, \infty)$), integrate over the full range of $X$.

**Notation:** $E[X]$, $\mu$, $\mu_X$ — all mean the same thing.

---

### 2.3 Properties of $E[X]$ — Continuous Case

Identical to the discrete case:

1. **Linearity I:** $E[X + Y] = E[X] + E[Y]$ (no independence required)
2. **Linearity II:** $E[aX + b] = aE[X] + b$ for constants $a, b$
3. **LOTUS (functions of $X$):** $E[h(X)] = \int_{-\infty}^{\infty} h(x) f_X(x)\, dx$

---

### 2.4 Worked Examples

---

#### Example 1 — Expected Value of Uniform(0, 1)

**Problem:** $X \sim \text{Uniform}(0,1)$. Find $E[X]$.

**Setup:** Range $[0,1]$, pdf $f(x) = 1$.

$$E[X] = \int_0^1 x \cdot 1\, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}$$

**Final Answer:** $E[X] = 1/2$

**Interpretation:** The mean is at the midpoint of the range — completely expected by symmetry. The uniform distribution is symmetric about $1/2$.

---

#### Example 2 — Expected Value of $f(x) = \frac{3}{8}x^2$ on $[0,2]$

**Problem:** $X$ has range $[0,2]$ and pdf $f(x) = \frac{3}{8}x^2$. Find $E[X]$.

$$E[X] = \int_0^2 x \cdot \frac{3}{8}x^2\, dx = \int_0^2 \frac{3}{8}x^3\, dx = \frac{3}{8} \cdot \left[\frac{x^4}{4}\right]_0^2 = \frac{3}{8} \cdot \frac{16}{4} = \frac{3}{8} \cdot 4 = \frac{3}{2}$$

**Final Answer:** $E[X] = 3/2$

**Does this make sense?** Yes. The range is $[0,2]$ with midpoint 1. But the pdf $f(x) = \frac{3}{8}x^2$ is increasing — it places more probability density at larger values of $x$. Therefore the mean should be **pulled to the right** of the midpoint. And indeed $3/2 > 1$.

> **Intuition:** The mean is the centre of mass. If more probability "mass" is concentrated on the right side, the balance point shifts right. This is exactly like a see-saw with heavier weights on one side.

---

#### Example 3 — Expected Value of Exponential($\lambda$) [Full Derivation]

**Problem:** $X \sim \text{Exp}(\lambda)$. Find $E[X]$.

**Setup:** Range $[0, \infty)$, pdf $f(x) = \lambda e^{-\lambda x}$.

$$E[X] = \int_0^{\infty} x \cdot \lambda e^{-\lambda x}\, dx$$

**Integration by parts:** Let $u = x$, $v' = \lambda e^{-\lambda x}$, so $u' = 1$, $v = -e^{-\lambda x}$.

$$= \left[-x e^{-\lambda x}\right]_0^{\infty} + \int_0^{\infty} e^{-\lambda x}\, dx$$

The first term evaluates to 0 at both limits: at $x=0$ it's 0; as $x \to \infty$, the exponential decays faster than $x$ grows (so $x e^{-\lambda x} \to 0$).

$$= 0 + \left[-\frac{e^{-\lambda x}}{\lambda}\right]_0^{\infty} = 0 - \left(-\frac{1}{\lambda}\right) = \frac{1}{\lambda}$$

$$\boxed{E[\text{Exp}(\lambda)] = \frac{1}{\lambda}}$$

**Interpretation:** If taxis arrive at rate $\lambda$ per minute (e.g., $\lambda = 1/10$, meaning 1 taxi per 10 minutes), then the expected wait is $1/\lambda = 10$ minutes. Higher rate $\lambda$ = shorter expected wait.

---

#### Example 4 — Expected Value of $N(0,1)$

**Problem:** $Z \sim N(0,1)$. Find $E[Z]$.

**Setup:** pdf $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ on $(-\infty, \infty)$.

$$E[Z] = \int_{-\infty}^{\infty} \frac{z}{\sqrt{2\pi}} e^{-z^2/2}\, dz$$

**The elegant argument — odd function:**

The integrand $g(z) = z \cdot \phi(z)$ is an **odd function**: $g(-z) = -g(z)$. (Because $z$ is odd and $e^{-z^2/2}$ is even.)

The integral of any odd function over a symmetric interval $(-\infty, \infty)$ is zero (positive and negative contributions cancel exactly).

$$E[Z] = 0$$

**Verification via substitution:** Let $u = z^2/2$, so $du = z\, dz$:

$$\int_0^{\infty} \frac{z}{\sqrt{2\pi}} e^{-z^2/2}\, dz = \frac{1}{\sqrt{2\pi}} \int_0^{\infty} e^{-u}\, du = \frac{1}{\sqrt{2\pi}}$$

By symmetry, the integral from $-\infty$ to 0 contributes $-\frac{1}{\sqrt{2\pi}}$. Total: 0. ✓

**Interpretation:** The standard normal is symmetric about 0, so its mean is exactly 0.

---

#### Example 5 — Proving $E[X] = \mu$ for $X \sim N(\mu, \sigma^2)$

**Problem:** Use linearity of expectation and the standardisation theorem to prove $E[X] = \mu$.

**Step 1:** The standardisation theorem (from Class 5) gives:

$$Z = \frac{X - \mu}{\sigma} \sim N(0,1)$$

**Step 2:** Invert: $X = \sigma Z + \mu$.

**Step 3:** Apply linearity of expectation:

$$E[X] = E[\sigma Z + \mu] = \sigma E[Z] + \mu = \sigma \cdot 0 + \mu = \mu \qquad \square$$

This is much more elegant than integrating the full normal pdf directly. Linearity + standardisation gives us the result in three lines.

---

#### Example 6 — $E[X^2]$ for Exponential($\lambda$) [Full Derivation]

**Problem:** $X \sim \text{Exp}(\lambda)$. Find $E[X^2]$.

$$E[X^2] = \int_0^{\infty} x^2 \lambda e^{-\lambda x}\, dx$$

**Integration by parts twice:**

**First pass:** $u = x^2$, $v' = \lambda e^{-\lambda x}$, so $u' = 2x$, $v = -e^{-\lambda x}$:

$$= \left[-x^2 e^{-\lambda x}\right]_0^{\infty} + \int_0^{\infty} 2x e^{-\lambda x}\, dx = 0 + \int_0^{\infty} 2x e^{-\lambda x}\, dx$$

**Second pass:** $u = 2x$, $v' = e^{-\lambda x}$, so $u' = 2$, $v = -e^{-\lambda x}/\lambda$:

$$= \left[-\frac{2x}{\lambda} e^{-\lambda x}\right]_0^{\infty} + \int_0^{\infty} \frac{2}{\lambda} e^{-\lambda x}\, dx = 0 + \frac{2}{\lambda} \cdot \frac{1}{\lambda} = \frac{2}{\lambda^2}$$

$$\boxed{E[X^2] = \frac{2}{\lambda^2} \quad \text{for } X \sim \text{Exp}(\lambda)}$$

---

## 3. Variance — Continuous Case

### 3.1 Formal Definition

The definition is **identical** to the discrete case — only the computation changes (integrals instead of sums).

> **Definition (Variance — Continuous):** If $X$ is a continuous random variable with mean $\mu$:
>
> $$\boxed{\text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 f(x)\, dx}$$

**Standard deviation:** $\sigma = \sqrt{\text{Var}(X)}$

---

### 3.2 Properties of Variance — Continuous Case

Same as the discrete case:

1. **Independence required:** If $X \perp Y$, then $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$
2. **Scale and shift:** $\text{Var}(aX + b) = a^2 \text{Var}(X)$
3. **Shortcut:** $\text{Var}(X) = E[X^2] - E[X]^2 = E[X^2] - \mu^2$

---

### 3.3 Worked Examples

---

#### Example 7 — Variance of Uniform(0, 1)

**Problem:** $X \sim U(0,1)$. Find $\text{Var}(X)$ and $\sigma_X$.

**From Example 1:** $\mu = 1/2$.

**Direct computation using the definition:**

$$\text{Var}(X) = \int_0^1 \left(x - \frac{1}{2}\right)^2 \cdot 1\, dx = \left[\frac{(x-1/2)^3}{3}\right]_0^1 = \frac{(1/2)^3}{3} - \frac{(-1/2)^3}{3} = \frac{1/8}{3} + \frac{1/8}{3} = \frac{1}{12}$$

$$\boxed{\text{Var}(U(0,1)) = \frac{1}{12}, \qquad \sigma = \frac{1}{\sqrt{12}} = \frac{1}{2\sqrt{3}} \approx 0.289}$$

**More generally:** For $X \sim U(a,b)$: $\text{Var}(X) = (b-a)^2/12$.

---

#### Example 8 — Variance of Exponential($\lambda$)

**Problem:** $X \sim \text{Exp}(\lambda)$. Find $\text{Var}(X)$ and $\sigma_X$.

**From Examples 3 and 6:**

$$E[X] = \frac{1}{\lambda}, \qquad E[X^2] = \frac{2}{\lambda^2}$$

**Apply the shortcut formula:**

$$\text{Var}(X) = E[X^2] - E[X]^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}$$

$$\boxed{\text{Var}(\text{Exp}(\lambda)) = \frac{1}{\lambda^2}, \qquad \sigma_X = \frac{1}{\lambda}}$$

**Key observation:** For the exponential distribution, the mean equals the standard deviation — both equal $1/\lambda$. This makes $\text{Exp}(\lambda)$ a high-variance distribution relative to its mean: the coefficient of variation $\sigma/\mu = 1$ always.

---

#### Example 9 — Proving $\text{Var}(Z) = 1$ for $Z \sim N(0,1)$

**Problem:** Show $\text{Var}(Z) = 1$.

Since $E[Z] = 0$:

$$\text{Var}(Z) = E[Z^2] = \int_{-\infty}^{\infty} \frac{z^2}{\sqrt{2\pi}} e^{-z^2/2}\, dz$$

**Integration by parts:** $u = z$, $v' = z e^{-z^2/2}$, so $u' = 1$, $v = -e^{-z^2/2}$:

$$= \frac{1}{\sqrt{2\pi}} \left[\left(-z e^{-z^2/2}\right)\Big|_{-\infty}^{\infty} + \int_{-\infty}^{\infty} e^{-z^2/2}\, dz\right]$$

- **First term:** $z e^{-z^2/2} \to 0$ as $z \to \pm\infty$ (exponential dominates polynomial). So this term equals 0.
- **Second term:** $\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-z^2/2}\, dz = 1$ — this is exactly the total integral of the standard normal pdf, which must equal 1.

$$\text{Var}(Z) = 0 + 1 = 1 \qquad \square$$

---

#### Example 10 — Proving $\text{Var}(X) = \sigma^2$ for $X \sim N(\mu, \sigma^2)$

**Problem:** Show the variance of a general normal equals $\sigma^2$.

**Approach:** Use the substitution $z = (x - \mu)/\sigma$ (i.e., $x = \sigma z + \mu$, $dx = \sigma\, dz$):

$$\text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x-\mu)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/2\sigma^2}\, dx$$

Substituting $z = (x-\mu)/\sigma$:

$$= \int_{-\infty}^{\infty} (\sigma z)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-z^2/2} \cdot \sigma\, dz = \sigma^2 \int_{-\infty}^{\infty} \frac{z^2}{\sqrt{2\pi}} e^{-z^2/2}\, dz = \sigma^2 \cdot 1 = \sigma^2$$

where the last integral is exactly $E[Z^2] = \text{Var}(Z) = 1$ from Example 9. $\square$

> **This completes the proof** that $N(\mu, \sigma^2)$ has mean $\mu$ and variance $\sigma^2$ — precisely what the notation suggests. As mathematicians, we need to verify these are not just notational conventions but actual properties.

---

## 4. Quantiles, Percentiles, and Medians

### 4.1 Concept Overview

The mean summarises a distribution's location, but it is not the only useful location summary. The **median** gives the value that splits the distribution exactly in half. More generally, **quantiles** divide the probability into specified fractions.

> **Why quantiles matter in practice:** Income and wealth are highly right-skewed — the mean income is much higher than the median because a few billionaires inflate the average. The median is a more representative summary for such distributions. In ML, percentiles of prediction errors characterise model performance beyond just the mean.

---

### 4.2 Formal Definitions

> **Definition (Median):** The **median** of $X$ is the value $m$ satisfying:
>
> $$P(X \leq m) = 0.5 \qquad \text{equivalently} \qquad F(m) = 0.5$$

> **Definition ($p$-th Quantile):** The **$p$-th quantile** (for $0 < p < 1$) is the value $q_p$ satisfying:
>
> $$\boxed{P(X \leq q_p) = p \qquad \text{equivalently} \qquad F(q_p) = p}$$

**Graphical interpretation:**
- On the pdf graph: $q_p$ is the point such that the area to the left is $p$ and area to the right is $1-p$.
- On the cdf graph: $q_p$ is the $x$-value where the cdf reaches height $p$.

**Naming conventions:**

| Name | Step size | Example |
|---|---|---|
| Percentile | 1/100 | 60th percentile = $q_{0.60}$ |
| Decile | 1/10 | 3rd decile = $q_{0.30}$ |
| Quartile | 1/4 | 3rd quartile = $q_{0.75}$ |
| Median | — | $q_{0.50}$ |

---

### 4.3 Computing Quantiles

**For distributions with a closed-form CDF:** Set $F(q_p) = p$ and solve for $q_p$.

**For the normal distribution:** No closed-form — use the R function `qnorm(p, mu, sigma)`.

**R functions:**

```r
qnorm(p, mu, sigma)   # quantile function for N(mu, sigma^2)
qnorm(0.6, 0, 1)      # = 0.25335  (60th percentile of standard normal)
```

---

### 4.4 Worked Examples

---

#### Example 11 — Median of the Standard Normal

**Problem:** Find the median of $Z \sim N(0,1)$.

By symmetry, $\phi(z)$ is symmetric about $z = 0$, so:

$$P(Z \leq 0) = P(Z \geq 0) = 0.5$$

The median is $q_{0.5} = 0$.

**Note:** For symmetric distributions, the mean and median are equal. For $N(\mu, \sigma^2)$, both equal $\mu$.

---

#### Example 12 — Median of Exponential($\lambda$)

**Problem:** Find the median of $X \sim \text{Exp}(\lambda)$.

**Step 1:** Set $F(q_{0.5}) = 1/2$.

$$1 - e^{-\lambda q} = \frac{1}{2}$$

**Step 2:** Solve:

$$e^{-\lambda q} = \frac{1}{2} \implies -\lambda q = \ln(1/2) = -\ln 2 \implies q = \frac{\ln 2}{\lambda}$$

$$\boxed{\text{Median of Exp}(\lambda) = \frac{\ln 2}{\lambda} \approx \frac{0.693}{\lambda}}$$

**Compare with mean:** The mean is $1/\lambda$ and the median is $(\ln 2)/\lambda \approx 0.693/\lambda$.

**Median $<$ mean — why?**

The exponential pdf has a long right tail (large values are possible but rare). These large values inflate the mean but don't move the median. The median is located where the distribution has "half its weight" to the left — since the distribution is right-skewed, this point is to the left of the mean. This is the classic pattern for right-skewed distributions.

---

#### Example 13 — Quantile of Uniform(0,1)

**Problem:** Find the 0.6 quantile of $X \sim U(0,1)$.

The CDF of $U(0,1)$ is $F(x) = x$ for $x \in [0,1]$.

Setting $F(q) = 0.6$: $q = 0.6$.

So the 60th percentile is $q_{0.6} = 0.6$. (For a uniform distribution, quantiles equal their probability values — the distribution is, well, uniform!)

---

#### Example 14 — 60th Percentile of the Standard Normal

**Problem:** Find $q_{0.6}$ for $Z \sim N(0,1)$.

No closed-form solution — use R:

$$q_{0.6} = \text{qnorm}(0.6, 0, 1) = 0.25335$$

**Interpretation:** 60% of the probability in a standard normal distribution lies below $z = 0.253$. Equivalently, if $Z \sim N(0,1)$, then $P(Z \leq 0.253) = 0.60$.

---

## 5. The Law of Large Numbers (LoLN)

### 5.1 Concept Overview

The Law of Large Numbers formalises the intuitive idea that repeating an experiment many times and averaging the results gives you the true underlying mean. This is the mathematical foundation for why statistics works at all.

> **Central question the LoLN answers:** If I repeat an experiment $n$ times and average the results, how close can I expect to get to the true mean $\mu$?

**Answer:** As $n$ grows, the sample average $\bar{X}_n$ gets arbitrarily close to $\mu$ with arbitrarily high probability.

---

### 5.2 Setup and Notation

Let $X_1, X_2, \ldots, X_n$ be **independent and identically distributed (i.i.d.)** random variables, each with mean $\mu$ and standard deviation $\sigma$.

**Sample average:**

$$\bar{X}_n = \frac{X_1 + X_2 + \cdots + X_n}{n} = \frac{1}{n}\sum_{i=1}^n X_i$$

Note that $\bar{X}_n$ is itself a random variable — it varies from experiment to experiment.

**Properties of the sample average** (from variance properties):

$$E[\bar{X}_n] = \mu \qquad \text{(unbiased: average of averages = true mean)}$$

$$\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \qquad \sigma_{\bar{X}_n} = \frac{\sigma}{\sqrt{n}}$$

The standard deviation shrinks by $\sqrt{n}$ — this is the $\sigma/\sqrt{n}$ law and is the quantitative expression of "more data = less uncertainty."

---

### 5.3 Formal Statement

> **Theorem (Law of Large Numbers):** Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$. Let $\bar{X}_n$ be the average of the first $n$ variables. Then for any $a > 0$:
>
> $$\boxed{\lim_{n \to \infty} P\left(|\bar{X}_n - \mu| < a\right) = 1}$$

**Reading:** For any tolerance $a > 0$ (no matter how small), the probability that the sample average is within $a$ of the true mean converges to 1 as $n \to \infty$.

---

### 5.4 The LoLN Has Two Faces

**Face 1 (Averages converge):** The sample average $\bar{X}_n \to \mu$ in probability.

**Face 2 (Histograms converge):** With high probability, the density histogram of a large number of samples from a distribution is a good approximation of the underlying pdf $f(x)$.

This second face explains why simulation works: if you simulate 1,000,000 draws from a distribution and plot the histogram, you'll see the true pdf. This is the foundation of Monte Carlo methods.

---

### 5.5 Concrete Example — Bernoulli Coin Flips

**Problem:** Each $X_i \sim \text{Bernoulli}(0.5)$, so $\mu = 0.5$. Let $\bar{X}_n$ = fraction of heads in $n$ flips. How quickly does $P(0.4 \leq \bar{X}_n \leq 0.6)$ approach 1?

**R computations:**

| $n$ | $P(0.4 \leq \bar{X}_n \leq 0.6)$ |
|---|---|
| 10 | 0.656 |
| 50 | 0.881 |
| 100 | 0.965 |
| 500 | 0.99999 |
| 1000 | ≈ 1.000 |

The LoLN is working — the probability of being within 0.1 of the true mean converges to 1.

**Tighter tolerance (within 0.01):**

| $n$ | $P(0.49 \leq \bar{X}_n \leq 0.51)$ |
|---|---|
| 10 | 0.246 |
| 100 | 0.236 |
| 1000 | 0.493 |
| 10000 | 0.956 |

Tighter tolerance requires much larger $n$ — but the probability still converges to 1.

---

### 5.6 Important Caveat: LoLN vs. Systematic Error

The LoLN guarantees convergence to the mean of the underlying distribution. But if the measuring instrument is defective or the sample is biased, the LoLN produces a highly accurate estimate of the **wrong** quantity.

> **Example:** A thermometer with a systematic +2°C bias: averaging 10,000 measurements gives a highly precise estimate that is still 2°C too high. More data makes random errors vanish; only careful experimental design eliminates systematic errors.

This is the mathematical distinction between:
- **Random error:** Controlled by the LoLN (more samples = less random error)
- **Systematic error / sampling bias:** Not controlled by the LoLN — must be addressed through experimental design

---

## 6. Histograms and the LoLN

### 6.1 Frequency vs. Density Histograms

Given samples $x_1, \ldots, x_n$:

1. **Choose bins:** Divide the data range into $m$ intervals (bins) with boundaries $b_0, b_1, \ldots, b_m$.

2. **Place data:** Each $x_i$ goes into the bin containing its value. (R default: values on boundaries go into the left/lower bin.)

3. **Frequency histogram:** Height of each bar = **count** of data points in that bin.

4. **Density histogram:** Area of each bar = **fraction** of data in that bin. So:

$$\text{height of bar} = \frac{\text{fraction of data in bin}}{\text{bin width}}$$

**Why the density histogram is preferred:**

- Its total area is always exactly 1 (regardless of bin widths).
- Its vertical scale is the same as the pdf $f(x)$.
- With equal bin widths, frequency and density histograms have the same shape but different scales.
- With **unequal** bin widths, they can look completely different and the frequency histogram misleads! Wider bins appear to have more data than they actually contain.

> **Warning:** Unequal bin widths in a frequency histogram can be manipulated to give a misleading visual impression. Always use density histograms, especially with unequal bins.

---

### 6.2 LoLN for Histograms

> **LoLN (Histogram version):** With high probability, the density histogram of a large number of samples from a distribution is a good approximation of the underlying pdf $f(x)$ over the range of the histogram.

**Illustration:** Generating 10,000 draws from $N(0,1)$ with bin width 0.1 produces a density histogram that closely tracks the standard normal pdf $\phi(z)$.

**Why this matters:**
- It is the theoretical justification for **simulation** and **Monte Carlo methods**.
- It explains why we can use observed data to estimate unknown distributions.
- It is the empirical foundation for density estimation in ML.

---

## 7. The Central Limit Theorem (CLT)

### 7.1 Concept Overview

The CLT is one of the most profound theorems in mathematics. It says that **regardless of the original distribution** of $X_i$ (as long as the mean and variance exist), the sum or average of many i.i.d. copies converges to a **normal distribution**.

> **Why this is remarkable:** Start with a highly non-normal distribution (uniform, exponential, Bernoulli — anything). Average enough copies of it. The result approaches normality. The normal distribution is the universal attractor for averages.

> **Why this matters enormously for statistics and ML:** It's why so many statistical procedures work based on normality assumptions even when the underlying data is not normal — if you're working with means or sums of many observations, normality is approximately guaranteed.

---

### 7.2 Setup

Let $X_1, X_2, \ldots, X_n$ be **i.i.d.** with mean $\mu$ and standard deviation $\sigma$.

Define:
- **Sum:** $S_n = X_1 + X_2 + \cdots + X_n$
- **Average:** $\bar{X}_n = S_n / n$
- **Standardised version:** $Z_n = \dfrac{S_n - n\mu}{\sigma\sqrt{n}} = \dfrac{\bar{X}_n - \mu}{\sigma/\sqrt{n}}$

**Properties (exact, from linearity of $E$ and variance addition):**

| Quantity | Mean | Variance | Std Dev |
|---|---|---|---|
| $S_n$ | $n\mu$ | $n\sigma^2$ | $\sigma\sqrt{n}$ |
| $\bar{X}_n$ | $\mu$ | $\sigma^2/n$ | $\sigma/\sqrt{n}$ |
| $Z_n$ | 0 | 1 | 1 |

These are exact for any $n$. The CLT tells us about the **distribution shape** as $n \to \infty$.

---

### 7.3 Formal Statement

> **Central Limit Theorem:** Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$ and standard deviation $\sigma < \infty$. Then for large $n$:
>
> $$\boxed{\bar{X}_n \approx N\!\left(\mu,\, \frac{\sigma^2}{n}\right), \qquad S_n \approx N\!\left(n\mu,\, n\sigma^2\right), \qquad Z_n \approx N(0,1)}$$
>
> Precisely: $\displaystyle\lim_{n \to \infty} F_{Z_n}(z) = \Phi(z)$ for all $z$.

**In words:**
1. The sample average $\bar{X}_n$ is approximately normally distributed with the same mean as $X$ but variance shrunk by factor $n$.
2. The sum $S_n$ is approximately normal with mean $n\mu$ and variance growing like $n$.
3. The standardised $Z_n$ converges to the standard normal.

---

### 7.4 The CLT Standardisation Formula

The key computational step in all CLT applications is **standardising**:

$$P(S_n > a) = P\left(\frac{S_n - n\mu}{\sigma\sqrt{n}} > \frac{a - n\mu}{\sigma\sqrt{n}}\right) \approx P\left(Z > \frac{a - n\mu}{\sigma\sqrt{n}}\right)$$

where $Z \sim N(0,1)$.

**Template:**

$$\boxed{P(S_n > a) \approx P\left(Z > \frac{a - n\mu}{\sigma\sqrt{n}}\right) = 1 - \Phi\left(\frac{a - n\mu}{\sigma\sqrt{n}}\right)}$$

---

### 7.5 The 68-95-99.7 Rule (Quick Reference)

For $Z \sim N(0,1)$:

$$P(|Z| < 1) \approx 0.68 \quad P(|Z| < 2) \approx 0.95 \quad P(|Z| < 3) \approx 0.997$$

More precisely: $P(|Z| < 1.96) = 0.95$.

**One-sided versions** (derived by symmetry — each tail has half the remaining probability):

| | $k = 1$ | $k = 2$ | $k = 3$ |
|---|---|---|---|
| $P(Z < k)$ | ≈ 0.84 | ≈ 0.977 | ≈ 0.999 |
| $P(Z > k)$ | ≈ 0.16 | ≈ 0.023 | ≈ 0.001 |

**Derivation of one-sided:** $P(|Z| < 1) = 0.68$ means both tails together = $0.32$. By symmetry each tail = $0.16$. So $P(Z < 1) = 1 - 0.16 = 0.84$.

---

## 8. Applications of the CLT

### 8.1 The Standard CLT Procedure

For any CLT probability calculation:

1. **Identify** $X_i$, their distribution, $\mu$, $\sigma$, and $n$.
2. **Compute** $E[S_n] = n\mu$ and $\sigma_{S_n} = \sigma\sqrt{n}$ (or use $\bar{X}_n$ with $\sigma/\sqrt{n}$).
3. **Standardise:** Convert the event to a statement about $Z = (S_n - n\mu)/(\sigma\sqrt{n})$.
4. **Apply** the standard normal table or R's `pnorm`.

---

### 8.2 Worked Examples

---

#### Example 2 (CLT) — More than 55 Heads in 100 Coin Flips

**Problem:** Flip a fair coin 100 times. Estimate $P(\text{more than 55 heads})$.

**Step 1: Set up.**

Let $X_j \sim \text{Bernoulli}(0.5)$ for each flip. Total heads $S = \sum_{j=1}^{100} X_j$.

$$E[X_j] = 0.5, \quad \text{Var}(X_j) = 0.25, \quad \sigma_{X_j} = 0.5$$

$$E[S] = 100 \times 0.5 = 50, \quad \text{Var}(S) = 100 \times 0.25 = 25, \quad \sigma_S = 5$$

**Step 2: Apply CLT** — $S \approx N(50, 25)$.

**Step 3: Standardise and compute.**

$$P(S > 55) = P\left(\frac{S - 50}{5} > \frac{55 - 50}{5}\right) \approx P(Z > 1) = 1 - P(Z < 1) \approx 1 - 0.84 = 0.16$$

**Final Answer:** $P(S > 55) \approx 16\%$

**Interpretation:** In 100 fair coin flips, about 16% of experiments will produce more than 55 heads. This corresponds to being more than 1 standard deviation above the mean.

---

#### Example 3 (CLT) — More than 220 Heads in 400 Flips

**Problem:** Estimate $P(S > 220)$ where $S$ = total heads in 400 fair coin flips.

**Step 1:**

$$E[S] = 400 \times 0.5 = 200, \quad \sigma_S = \sqrt{400 \times 0.25} = \sqrt{100} = 10$$

**Step 2:** $S \approx N(200, 100)$.

**Step 3:**

$$P(S > 220) = P\left(\frac{S-200}{10} > \frac{220-200}{10}\right) \approx P(Z > 2) \approx 0.025$$

**Final Answer:** $P(S > 220) \approx 2.5\%$

**Important comparison with Example 2:**

Although $55/100 = 220/400 = 0.55$ (same proportion above mean), the probabilities differ:
- $P(S > 55)$ in 100 flips $\approx 16\%$
- $P(S > 220)$ in 400 flips $\approx 2.5\%$

This is the LoLN in action. With 4× more data, the distribution of the proportion concentrates much more tightly around 0.5. Getting 55% heads is a 1-standard-deviation event with $n=100$ but a 2-standard-deviation event with $n=400$.

---

#### Example 4 (CLT) — Between 40 and 60 Heads in 100 Flips

**Problem:** Estimate $P(40 \leq S \leq 60)$ for $S$ = heads in 100 fair flips.

**From Example 2:** $E[S] = 50$, $\sigma_S = 5$.

**Standardise both bounds:**

$$P(40 \leq S \leq 60) = P\left(\frac{40-50}{5} \leq Z \leq \frac{60-50}{5}\right) \approx P(-2 \leq Z \leq 2) \approx 0.954$$

**Exact answer using R:**

```r
pnorm(2) - pnorm(-2) = 0.9545
```

**Exact binomial answer:** $\approx 0.9648$

**CLT approximation error:** $\approx 1\%$ — very good for $n = 100$.

---

#### Example 5 (CLT) — Polling and Margin of Error

**Problem:** Derive the $\pm 1/\sqrt{n}$ polling margin-of-error formula.

**Setup:**
- True fraction of population preferring candidate A: $p_0$ (unknown)
- Poll $n$ people, each independently: $X_i \sim \text{Bernoulli}(p_0)$
- Sample fraction: $\bar{X} = \frac{1}{n}\sum X_i$

**Step 1: Parameters.**

$$E[\bar{X}] = p_0, \quad \sigma_{\bar{X}} = \frac{\sqrt{p_0(1-p_0)}}{\sqrt{n}}$$

**Step 2: Apply CLT.**

$$\bar{X} \approx N\!\left(p_0,\, \frac{p_0(1-p_0)}{n}\right)$$

**Step 3: Apply 95% rule.**

In 95% of polls: $\bar{X}$ is within $2\sigma_{\bar{X}} = \frac{2\sqrt{p_0(1-p_0)}}{\sqrt{n}}$ of $p_0$.

**Step 4: Conservative bound.**

$p_0(1-p_0)$ is maximised at $p_0 = 1/2$, where it equals $1/4$. So:

$$2\sigma_{\bar{X}} \leq \frac{2 \cdot (1/2)}{\sqrt{n}} = \frac{1}{\sqrt{n}}$$

**Conclusion:** In 95% of polls of $n$ people, the sample proportion $\bar{X}$ is within $\pm 1/\sqrt{n}$ of the true proportion $p_0$.

$$\boxed{\text{95\% Margin of Error} = \pm \frac{1}{\sqrt{n}}}$$

**Examples:**
- $n = 1000$: margin $\approx \pm 3.2\%$
- $n = 2500$: margin $\approx \pm 2\%$
- $n = 10000$: margin $\approx \pm 1\%$

**Critical warning:** The confidence interval $\bar{X} \pm 1/\sqrt{n}$ does NOT mean "there is a 95% probability that $p_0$ is in this interval." The true $p_0$ is a fixed (unknown) number, not a random variable. The 95% refers to the probability over repeated polls. We will revisit this subtle distinction when we study confidence intervals formally.

---

## 9. How Large Does $n$ Need to Be?

### 9.1 The Short Answer

Often, not that large. But it depends on how non-normal the original distribution is.

### 9.2 Convergence Speed Depends on Distribution Shape

**Symmetric distributions** (like Uniform): Converge very quickly.
- For Uniform, $n = 12$ already gives an excellent normal approximation.
- Rule of thumb: $n \geq 12$ is often sufficient for symmetric distributions.

**Asymmetric distributions** (like Exponential): Need more terms.
- The exponential is strongly right-skewed; it takes $n \approx 64$ for a good approximation.
- Rule of thumb: For strongly skewed distributions, $n \geq 30$–$50$ is commonly advised.

**Discrete distributions** (like Bernoulli): The CLT works but the discrete nature creates a "granular" approximation.
- For Bernoulli(0.5) at $n = 12$: looks roughly normal.
- At $n = 64$: very close to normal. The discrete values become dense enough to approximate the continuous normal well.

### 9.3 Effect of $n$ on the Distribution Shape

As $n$ grows:

1. **Standardised $\bar{X}_n$ or $Z_n$** converges to $N(0,1)$ in shape — the tails fill in, the skewness disappears.

2. **Non-standardised $\bar{X}_n$** becomes more and more concentrated around $\mu$, with standard deviation shrinking as $\sigma/\sqrt{n}$. The distribution becomes taller and narrower (more "spiked") as $n$ grows.

3. For exponential: you can visually see the convergence sequence $n = 1$ (strongly right-skewed) $\to$ $n = 4$ (moderate skew) $\to$ $n = 16$ (small skew) $\to$ $n = 64$ (nearly normal).

### 9.4 Why Use the CLT When Exact Methods Exist?

The binomial probabilities in Examples 2–4 could be computed exactly. So why use the CLT?

1. **Unknown distributions:** In most real problems, you don't know the exact distribution of $X_i$. The CLT applies as long as $\mu$ and $\sigma$ exist.

2. **Complex distributions:** Even if the distribution is known, computing exact probabilities for sums may be intractable analytically or computationally expensive.

3. **Practical sufficiency:** A 1% error (as in Example 4) is often acceptable for applied purposes.

4. **Theoretical insight:** The CLT reveals the universal structure of sampling distributions, which is the foundation of all frequentist statistical inference.

---

## 10. Complete Distribution Reference Table

### Discrete Distributions

| Distribution | Range | PMF $p(k)$ | Mean $\mu$ | Variance $\sigma^2$ |
|---|---|---|---|---|
| $\text{Ber}(p)$ | $\{0,1\}$ | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| $\text{Bin}(n,p)$ | $\{0,\ldots,n\}$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| $\text{Geo}(p)$ | $\{0,1,2,\ldots\}$ | $(1-p)^k p$ | $(1-p)/p$ | $(1-p)/p^2$ |
| $\text{Unif}(n)$ | $\{1,\ldots,n\}$ | $1/n$ | $(n+1)/2$ | $(n^2-1)/12$ |

### Continuous Distributions

| Distribution | Range | PDF $f(x)$ | Mean $\mu$ | Variance $\sigma^2$ |
|---|---|---|---|---|
| $U(a,b)$ | $[a,b]$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| $\text{Exp}(\lambda)$ | $[0,\infty)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| $N(\mu,\sigma^2)$ | $(-\infty,\infty)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |

### CLT Approximations for Sums/Averages

| Quantity | Exact Mean | Exact Variance | CLT Approx Distribution |
|---|---|---|---|
| $S_n = \sum X_i$ | $n\mu$ | $n\sigma^2$ | $N(n\mu, n\sigma^2)$ |
| $\bar{X}_n = S_n/n$ | $\mu$ | $\sigma^2/n$ | $N(\mu, \sigma^2/n)$ |
| $Z_n = (S_n - n\mu)/(\sigma\sqrt{n})$ | 0 | 1 | $N(0,1)$ |

### Key Normal Probabilities

| Interval | Probability |
|---|---|
| $P(\mu - \sigma \leq X \leq \mu + \sigma)$ | ≈ 0.68 |
| $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma)$ | ≈ 0.95 |
| $P(\mu - 1.96\sigma \leq X \leq \mu + 1.96\sigma)$ | = 0.95 (precise) |
| $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma)$ | ≈ 0.997 |
| $P(Z < 1)$ | ≈ 0.84 |
| $P(Z < 2)$ | ≈ 0.977 |

---

## 11. Common Mistakes Reference

| Mistake | Why it's wrong | Correct approach |
|---|---|---|
| $E[X] = \int f(x)\, dx$ (forgetting the $x$) | Missing the weight $x$ | $E[X] = \int x f(x)\, dx$ — multiply by $x$ before integrating |
| Thinking the median always equals the mean | Only true for symmetric distributions | For right-skewed distributions, median $<$ mean |
| Using variance addition for dependent variables in CLT problems | CLT uses independent copies | The CLT requires i.i.d. — ensure you have independence |
| Applying CLT for too small $n$ without checking skewness | CLT is an approximation that can be poor | Symmetric: $n \geq 12$ often ok; strongly skewed: need $n \geq 30$–$50$ |
| "There's a 95% probability $p_0$ is in the confidence interval" | $p_0$ is fixed; CIs are random | The 95% is the coverage rate over many repeated polls/experiments |
| Confusing $P(\|Z\|<2) \approx 0.95$ with $P(Z<2) \approx 0.977$ | One is two-sided, one is one-sided | $P(-2 < Z < 2) = 0.95$; $P(Z < 2) = 0.977$ |
| Using frequency histogram with unequal bin widths | The visual is misleading | Always use density histograms, especially with unequal bins |
| Forgetting LoLN doesn't fix systematic error | LoLN only eliminates random error | Systematic bias requires better experimental design, not more data |
| Confusing $\sigma$ and $\sigma^2$ in R's `pnorm` | R's pnorm takes $\sigma$, not $\sigma^2$ | `pnorm(x, mu, sigma)` — extract $\sigma = \sqrt{\sigma^2}$ first |
| Computing $E[\bar{X}] \neq \mu$ | Linearity ensures $E[\bar{X}] = \mu$ always | The sample mean is always an unbiased estimator of the population mean |

---

## 12. Quick Summary & Formula Sheet

### Continuous Expectation and Variance

$$E[X] = \int_{-\infty}^{\infty} x f(x)\, dx, \qquad E[h(X)] = \int_{-\infty}^{\infty} h(x) f(x)\, dx$$

$$\text{Var}(X) = E[(X-\mu)^2] = E[X^2] - \mu^2$$

### Named Means and Variances

$$E[U(0,1)] = \frac{1}{2}, \quad \text{Var} = \frac{1}{12}$$

$$E[\text{Exp}(\lambda)] = \frac{1}{\lambda}, \quad \text{Var} = \frac{1}{\lambda^2}, \quad \text{Median} = \frac{\ln 2}{\lambda}$$

$$E[N(\mu,\sigma^2)] = \mu, \quad \text{Var} = \sigma^2, \quad \text{Median} = \mu$$

### Quantiles

$$q_p = F^{-1}(p) \qquad \text{i.e. the value where } F(q_p) = p$$

### Law of Large Numbers

$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| < a) = 1 \quad \text{for any } a > 0$$

### Central Limit Theorem

$$\bar{X}_n \approx N\!\left(\mu, \frac{\sigma^2}{n}\right), \qquad S_n \approx N(n\mu, n\sigma^2), \qquad Z_n = \frac{S_n - n\mu}{\sigma\sqrt{n}} \approx N(0,1)$$

### CLT Computation Template

$$P(S_n > a) \approx P\!\left(Z > \frac{a - n\mu}{\sigma\sqrt{n}}\right) = 1 - \Phi\!\left(\frac{a - n\mu}{\sigma\sqrt{n}}\right)$$

### Polling Margin of Error (CLT Application)

$$\text{95\% margin of error} = \pm \frac{1}{\sqrt{n}} \quad \text{(conservative bound)}$$

### Key Insights

- **Discrete $\to$ continuous:** Sums become integrals. All properties of $E$ and Var are preserved.
- **Exponential:** Mean = Std Dev = $1/\lambda$. Right-skewed, so median $<$ mean.
- **Normal:** Symmetric, so mean = median = $\mu$. Fully characterised by $\mu$ and $\sigma$.
- **Quantile:** The $p$-th quantile $q_p$ is where the CDF reaches $p$. Use $F^{-1}(p)$ or R's `qnorm`.
- **LoLN:** Sample average converges to $\mu$ in probability. More data = less random error. Does NOT fix systematic error.
- **Histograms:** Density histogram = better than frequency histogram, especially with unequal bins. LoLN says density histograms of large samples converge to the true pdf.
- **CLT:** Sum/average of i.i.d. variables → normal, regardless of the original distribution.
- **$\sigma/\sqrt{n}$:** The standard deviation of $\bar{X}_n$ shrinks as $1/\sqrt{n}$. To halve the uncertainty, you need 4× the data.
- **Symmetric distributions** converge to normal quickly ($n \approx 12$). Skewed distributions need more ($n \approx 30$–$64$).

---

