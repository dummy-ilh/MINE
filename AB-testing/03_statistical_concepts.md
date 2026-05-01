# 📊 03 — Core Statistical Concepts

> *"The test statistic is the voice of your data, speaking in units of standard error."*

---

## 1️⃣ Test Statistic

### Definition

A **test statistic** is a single number computed from sample data that summarizes how far the observed data is from what H₀ predicts, in standard error units.

$$T = \frac{\hat{\theta} - \theta_0}{SE(\hat{\theta})}$$

Where:
- $\hat{\theta}$ = sample estimate (e.g., sample mean $\bar{x}$)
- $\theta_0$ = null hypothesis value
- $SE(\hat{\theta})$ = standard error of the estimator

### Intuition

The test statistic answers: **"How many standard errors away from the null value is my sample result?"**

- If $T \approx 0$ → data is consistent with H₀
- If $|T|$ is large → data is surprising under H₀ → evidence against H₀

### Common Test Statistics

| Test | Statistic | Distribution Under H₀ |
|---|---|---|
| Z-test | $Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$ | $\mathcal{N}(0,1)$ |
| t-test | $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$ | $t_{n-1}$ |
| Chi-square | $\chi^2 = \sum \frac{(O-E)^2}{E}$ | $\chi^2_k$ |
| F-test (ANOVA) | $F = \frac{MS_{between}}{MS_{within}}$ | $F_{k-1, n-k}$ |

### Numeric Example

A manufacturer claims mean battery life is 500 hours. A sample of $n=36$ gives $\bar{x} = 492$ hours with known $\sigma = 24$.

$$Z = \frac{492 - 500}{24/\sqrt{36}} = \frac{-8}{4} = -2.0$$

**Interpretation:** The sample mean is 2 standard errors below the claimed mean.

---

## 2️⃣ Sampling Distribution

### Definition

The **sampling distribution** is the probability distribution of a statistic (like $\bar{x}$) computed from all possible samples of size $n$ from the population.

### Key Properties of the Sampling Distribution of $\bar{x}$

$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

- **Mean of sampling distribution** = Population mean $\mu$ (unbiased estimator)
- **Variance of sampling distribution** = $\sigma^2 / n$ (decreases with $n$)
- **Shape**: Normal (if population is normal, OR $n$ is large by CLT)

### Visual Intuition

![Sampling distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Standard_deviation_diagram.svg/800px-Standard_deviation_diagram.svg.png)

Imagine taking 10,000 samples of size $n=30$ and computing $\bar{x}$ each time. Plotting all those $\bar{x}$ values gives the sampling distribution — a bell curve centered at the true $\mu$.

### Why This Matters for Testing

Under H₀, we know the sampling distribution. The test statistic tells us where our *single observed sample* falls on this distribution. If it falls in the extreme tails → suspicious → evidence against H₀.

---

## 3️⃣ Central Limit Theorem (CLT)

### Statement

For a population with mean $\mu$ and finite variance $\sigma^2$, the sampling distribution of $\bar{X}$ approaches a normal distribution as $n \to \infty$:

$$\bar{X} \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{for large } n$$

This holds **regardless of the population distribution**.

### Rule of Thumb

- $n \geq 30$: Usually sufficient for CLT approximation
- $n \geq 50-100$: Needed for heavily skewed populations
- $n < 30$ with non-normal data: Use t-test with caution, or non-parametric methods

### Why CLT Is the Foundation of Hypothesis Testing

CLT justifies using the normal distribution for test statistics even when the underlying population is:
- Right-skewed (e.g., income)
- Bimodal
- Discrete (e.g., binary 0/1 conversions)

**A/B Testing example:** Conversions are Bernoulli (0 or 1). Individual observations aren't normal. But with $n = 10{,}000$, the sampling distribution of mean conversion rate is approximately normal by CLT.

$$\bar{p} \approx \mathcal{N}\left(p, \frac{p(1-p)}{n}\right)$$

### Demonstration

```
Population: Exponential(λ=1) — heavily right-skewed
Sample sizes: n=1, n=5, n=30, n=100

n=1:    ████▌▌▌  (exponential)
n=5:    ██▓▓▒▒░  (still skewed)
n=30:   ▒▒▓▓▓▒▒  (roughly bell-shaped)
n=100:  ░▒▓████▓▒░  (very normal)
```

---

## 4️⃣ Standard Error (SE)

### Definition

The **standard error** is the **standard deviation of the sampling distribution** of a statistic. It measures how much the statistic varies across samples.

$$SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

For estimated SE (when $\sigma$ unknown, use sample $s$):

$$\widehat{SE}(\bar{X}) = \frac{s}{\sqrt{n}}$$

### Standard Error for Different Statistics

| Statistic | Standard Error |
|---|---|
| Sample mean $\bar{X}$ | $\sigma / \sqrt{n}$ |
| Sample proportion $\hat{p}$ | $\sqrt{p(1-p)/n}$ |
| Difference of means $\bar{X}_1 - \bar{X}_2$ | $\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}$ |
| Difference of proportions $\hat{p}_1 - \hat{p}_2$ | $\sqrt{p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2}$ |

### SE vs Standard Deviation — Critical Distinction

| | Standard Deviation ($s$) | Standard Error ($SE$) |
|---|---|---|
| Measures | Spread of individual data points | Spread of the sample statistic |
| Decreases with $n$? | ❌ No | ✅ Yes ($\propto 1/\sqrt{n}$) |
| Use for | Describing the data | Inference about population parameter |

### ⚠️ Pitfall: Confusing SD and SE

Reporting SE as if it were SD makes a distribution look tighter than it is. This is a common trick to make results look more precise. Always clarify which you're reporting.

### Effect of Sample Size on SE

$$n = 100 \Rightarrow SE = \sigma/10$$
$$n = 400 \Rightarrow SE = \sigma/20 \quad \text{(half the SE, 4× the data)}$$

Doubling precision requires quadrupling the sample size — the **square root law**.

---

## 🔗 How These Concepts Connect

```
Population (unknown μ, σ)
        ↓
Take sample of size n
        ↓
Compute sample statistic (e.g., x̄)
        ↓
CLT: x̄ ~ N(μ, σ²/n) for large n
        ↓
SE = σ/√n tells us how variable x̄ is
        ↓
Test statistic = (x̄ - μ₀) / SE
    measures "how surprising is this x̄ if H₀ is true?"
        ↓
Compare to null distribution → get p-value
```

---

## 💬 Interview Questions & Answers

**Q1: Why do we divide by $\sqrt{n}$ in the standard error?**

> A: Because we're computing the mean of $n$ independent observations. The variance of a mean is $\text{Var}(\bar{X}) = \sigma^2/n$ by the independence property of variances (Var of sum = sum of variances, divided by $n^2$). Taking the square root gives $\sigma/\sqrt{n}$. Intuitively: the more data you have, the less the sample mean bounces around — and it bounces less by a factor of $\sqrt{n}$.

**Q2: When does CLT apply? When does it fail?**

> A: CLT applies when observations are i.i.d. (independent and identically distributed) with finite mean and variance. It fails when: observations have infinite variance (heavy-tailed distributions like Cauchy), observations are not independent (time series, clustered data), or when n is very small with highly skewed data. In practice, financial returns can have very fat tails where CLT convergence is slow.

**Q3: What's the difference between standard deviation and standard error? Why does it matter?**

> A: SD measures the variability of individual data points. SE measures the variability of a sample statistic (like the mean) across different samples. SD doesn't shrink with more data; SE does ($\propto 1/\sqrt{n}$). It matters because SE is what appears in confidence intervals and test statistics — not SD. Confusing the two leads to wildly incorrect inference.

**Q4: In an A/B test, conversion rates are binary. Can you use a Z-test?**

> A: Yes, by CLT. Individual conversions are Bernoulli (0/1), but with large enough samples (typically $n \geq 100$ per group, or $np \geq 5$ and $n(1-p) \geq 5$), the sampling distribution of the conversion rate is approximately normal. We use the two-proportion Z-test with SE based on the pooled proportion under H₀.

---

## 🚨 Common Pitfalls

1. **Using SE where SD is appropriate** (or vice versa) — especially in reporting results.
2. **Assuming CLT applies for tiny samples with skewed data**.
3. **Using population $\sigma$ when it's unknown** — must estimate with $s$ and use t-distribution.
4. **Ignoring dependence** — CLT requires independent observations. Violated in time series or clustered experiments.
5. **Forgetting that SE decreases with n** — this means with huge $n$, you'll always detect tiny, meaningless effects.

---

*← [02 — Errors](02_errors.md) | [04 — p-value →](04_pvalue.md)*
