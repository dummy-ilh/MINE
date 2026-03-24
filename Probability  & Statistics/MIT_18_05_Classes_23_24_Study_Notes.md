# MIT 18.05 — Introduction to Probability and Statistics
## Complete Study Notes: Classes 23 & 24
### Confidence Intervals: Three Views · Non-Normal Data · Bootstrap Methods

---

> **How to use these notes:** Every concept, example, and worked problem from the uploaded PDFs (Classes 23 and 24) is reproduced here with expanded explanations. You should never need to refer back to the original documents.

---

# PART I — CLASS 23: Confidence Intervals — Three Views

---

## Topic 1: Overview — Three Perspectives on Confidence Intervals

### 1. Concept Overview

There are three complementary ways to understand and construct confidence intervals:

1. **Standardized statistics** — Start from a known pivotal quantity (like $Z$, $T$, or $\chi^2$) and use algebra to isolate the unknown parameter in an interval.
2. **Hypothesis testing (inversion)** — A confidence interval is the set of all parameter values that would *not* be rejected as null hypotheses given the observed data.
3. **Formal definition** — The most mathematically rigorous definition, valid for any distribution.

Each perspective gives different insights. All three are consistent with each other when they apply.

> **Key Principle:** All three views agree on this: a $(1-\alpha)$ confidence interval is an interval statistic such that, in $(1-\alpha)$ fraction of repeated experiments, the interval will contain the true parameter value.

---

## Topic 2: View 1 — Confidence Intervals via Standardized Statistics

### 1. Concept Overview

This is the most computational view. The idea is simple:

1. Identify a standardized statistic (a function of data + unknown parameter) that follows a known distribution.
2. Use the known distribution to bound the statistic with probability $1-\alpha$.
3. Use algebra to isolate the unknown parameter in the middle of an interval.

### 2. z-Confidence Interval — Derivation from Standardized Mean

**Setup:** $x_1, x_2, \ldots, x_n \sim N(\mu, \sigma^2)$, with $\mu$ unknown and $\sigma$ **known**. Let $\mu_0$ denote the true (unknown) value of $\mu$.

**Step 1:** The standardized mean is:

$$Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1)$$

**Step 2:** For the standard normal critical value $z_{\alpha/2}$:

$$P\!\left(-z_{\alpha/2} < Z < z_{\alpha/2}\right) = 1 - \alpha$$

**Step 3:** Substitute the definition of $Z$:

$$P\!\left(-z_{\alpha/2} < \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} < z_{\alpha/2} \;\middle|\; \mu = \mu_0\right) = 1 - \alpha$$

**Step 4:** Multiply all parts by $\sigma/\sqrt{n}$ and rearrange to isolate $\mu_0$:

$$P\!\left(\bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} < \mu_0 < \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \;\middle|\; \mu = \mu_0\right) = 1 - \alpha$$

**Result — the $(1-\alpha)$ z-confidence interval:**

$$\boxed{\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}}$$

> **Critical note:** The probabilities are conditioned on $\mu = \mu_0$. As with all frequentist statistics, we must fix a hypothesized value of the parameter to compute probabilities. The interval is random (it depends on $\bar{x}$), not the parameter.

---

### 3. t-Confidence Interval — Derivation from Studentized Mean

**Setup:** $x_1, x_2, \ldots, x_n \sim N(\mu, \sigma^2)$, with **both** $\mu$ and $\sigma$ **unknown**.

**Step 1:** The Studentized mean follows a $t$-distribution:

$$T = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \sim t(n-1)$$

where $s^2$ is the sample variance.

**Step 2:** For the $t$-critical value $t_{\alpha/2}$:

$$P\!\left(-t_{\alpha/2} < T < t_{\alpha/2} \;\middle|\; \mu = \mu_0\right) = 1 - \alpha$$

**Step 3:** Substitute and rearrange:

$$P\!\left(\bar{x} - t_{\alpha/2} \cdot \frac{s}{\sqrt{n}} < \mu_0 < \bar{x} + t_{\alpha/2} \cdot \frac{s}{\sqrt{n}} \;\middle|\; \mu = \mu_0\right) = 1 - \alpha$$

**Result — the $(1-\alpha)$ t-confidence interval:**

$$\boxed{\bar{x} \pm t_{\alpha/2} \cdot \frac{s}{\sqrt{n}}}$$

where $t_{\alpha/2}$ has $n-1$ degrees of freedom.

> **Why t instead of z?** Because $\sigma$ is estimated by $s$, we introduce extra variability. The $t$-distribution has heavier tails than $N(0,1)$ to account for this. As $n \to \infty$, $t(n-1) \to N(0,1)$ and the two intervals converge.

---

### 4. Chi-Square Confidence Interval for Variance

**Setup:** $x_1, x_2, \ldots, x_n \sim N(\mu, \sigma^2)$, with both $\mu$ and $\sigma$ **unknown**. Let $\sigma_0$ be the true value of $\sigma$.

**Step 1:** The standardized variance follows a chi-square distribution:

$$X^2 = \frac{(n-1)s^2}{\sigma_0^2} \sim \chi^2(n-1)$$

**Step 2:** Because $\chi^2$ is **not symmetric**, we need two separate critical values. Let $c_{\alpha/2}$ and $c_{1-\alpha/2}$ be the right-tail critical values:

$$P\!\left(c_{1-\alpha/2} < X^2 < c_{\alpha/2}\right) = 1 - \alpha$$

**Step 3:** Substitute $X^2$:

$$P\!\left(c_{1-\alpha/2} < \frac{(n-1)s^2}{\sigma_0^2} < c_{\alpha/2} \;\middle|\; \sigma = \sigma_0\right) = 1 - \alpha$$

**Step 4:** Invert (this reverses inequality directions):

$$P\!\left(\frac{(n-1)s^2}{c_{\alpha/2}} < \sigma_0^2 < \frac{(n-1)s^2}{c_{1-\alpha/2}} \;\middle|\; \sigma = \sigma_0\right) = 1 - \alpha$$

**Result — the $(1-\alpha)$ $\chi^2$-confidence interval for $\sigma^2$:**

$$\boxed{\left[\frac{(n-1)s^2}{c_{\alpha/2}},\; \frac{(n-1)s^2}{c_{1-\alpha/2}}\right]}$$

> **Important:** In R, `qchisq(1 - alpha/2, df)` gives $c_{\alpha/2}$ (the *larger* right critical value), and `qchisq(alpha/2, df)` gives $c_{1-\alpha/2}$ (the *smaller* left critical value). This asymmetry means the chi-square CI is not centered on $s^2$.

---

### 5. Summary Table — Three Standardized CIs

| Interval for | Statistic | Distribution | Formula |
|---|---|---|---|
| $\mu$ ($\sigma$ known) | $(\bar{x}-\mu)/(\sigma/\sqrt{n})$ | $N(0,1)$ | $\bar{x} \pm z_{\alpha/2}\,\sigma/\sqrt{n}$ |
| $\mu$ ($\sigma$ unknown) | $(\bar{x}-\mu)/(s/\sqrt{n})$ | $t(n-1)$ | $\bar{x} \pm t_{\alpha/2}\,s/\sqrt{n}$ |
| $\sigma^2$ | $(n-1)s^2/\sigma^2$ | $\chi^2(n-1)$ | $\left[\frac{(n-1)s^2}{c_{\alpha/2}},\frac{(n-1)s^2}{c_{1-\alpha/2}}\right]$ |

---

## Topic 3: View 2 — Confidence Intervals via Hypothesis Testing (Inversion)

### 1. Concept Overview

This view answers the question: *Given data producing test statistic $x$, which values of $\theta$ would I fail to reject?*

The confidence interval is precisely the set of all such $\theta$ values. This is called **test inversion**.

> **Key Definition:** The $(1-\alpha)$ confidence interval consists of all values $\theta_0$ that are **not rejected** at significance level $\alpha$ when used as the null hypothesis $H_0: \theta = \theta_0$.

### 2. Type I CI Error

A **Type I CI error** occurs when the confidence interval does not contain the true value of $\theta$.

For a $(1-\alpha)$ confidence interval, the Type I CI error rate is $\alpha$.

This mirrors NHST: in both cases, $\alpha$ is the probability of being "wrong" when the null/true value is actually the hypothesized value.

---

### 3. Worked Example — Binomial CI via Hypothesis Inversion

#### Example 1 — Binomial(12, θ) Confidence Interval

**Problem:** Data $x$ is drawn from a $\text{Binomial}(12, \theta)$ distribution with $\theta$ unknown. Use $\alpha = 0.1$ to find the 90% confidence interval for each possible value of $x$.

**Strategy:** For each $\theta$, choose a rejection region with significance $\leq 0.1$. Then for each observed $x$, the CI contains all $\theta$ whose rejection region does NOT include $x$ (i.e., $x$ is in the non-rejection region).

**The Likelihood Table — Binomial(12, θ):**

| $\theta \backslash x$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | Sig. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | 1.00 | 0.000 |
| 0.9 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .02 | .09 | .23 | .38 | .28 | 0.026 |
| 0.8 | .00 | .00 | .00 | .00 | .00 | .00 | .02 | .05 | .13 | .24 | .28 | .21 | .07 | 0.073 |
| 0.7 | .00 | .00 | .00 | .00 | .01 | .03 | .08 | .16 | .23 | .24 | .17 | .07 | .01 | 0.052 |
| 0.6 | .00 | .00 | .00 | .01 | .04 | .10 | .18 | .23 | .21 | .14 | .06 | .02 | .00 | 0.077 |
| 0.5 | .00 | .00 | .02 | .05 | .12 | .19 | .23 | .19 | .12 | .05 | .02 | .00 | .00 | 0.092 |
| 0.4 | .00 | .02 | .06 | .14 | .21 | .23 | .18 | .10 | .04 | .01 | .00 | .00 | .00 | 0.077 |
| 0.3 | .01 | .07 | .17 | .24 | .23 | .16 | .08 | .03 | .01 | .00 | .00 | .00 | .00 | 0.052 |
| 0.2 | .07 | .21 | .28 | .24 | .13 | .05 | .02 | .00 | .00 | .00 | .00 | .00 | .00 | 0.073 |
| 0.1 | .28 | .38 | .23 | .09 | .02 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | 0.026 |
| 0.0 | 1.00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | 0.000 |

**How to build the rejection region for each $\theta$:**

For each row ($\theta$ value), identify which $x$ values to reject. The rules are:
1. Total probability in the rejection region must be $\leq \alpha = 0.1$.
2. Build the rejection region by always picking the $x$ with the **smallest probability** (most extreme data) first, until adding the next would exceed $\alpha$.

**Example — Row $\theta = 0.9$:**
- Smallest probabilities (rounded): $x = 0,1,2,3,4,5,6,7,8$ all show 0.00 to 2 decimal places.
- Adding these gives total significance $\approx 0.026 \leq 0.1$. ✓
- Adding $x = 9$ (probability 0.09) would push total to $0.026 + 0.09 = 0.116 > 0.1$. ✗
- **Rejection region for $\theta = 0.9$:** $\{x \leq 8\}$ (values 0 through 8), significance = 0.026.
- **Non-rejection region:** $\{x \in \{9, 10, 11, 12\}\}$

**Example — Row $\theta = 0.5$:**
- Smallest probabilities: $x = 0$ (0.00), $x = 1$ (0.00), $x = 11$ (0.00), $x = 12$ (0.00), then $x = 2$ (0.02), $x = 10$ (0.02)...
- The two tails add up: $0 + 0 + 0 + 0 + 0.02 + 0.02 = 0.04$; adding $x=3$ (0.05) gives 0.09 ≤ 0.1. Adding $x = 9$ (0.05) gives 0.14 > 0.1. So stop.
- **Rejection region for $\theta = 0.5$:** $\{x \leq 3\} \cup \{x \geq 9\}$ (the tails), significance = 0.092.
- **Non-rejection region:** $\{x \in \{4,5,6,7,8\}\}$

**Reading the confidence interval from the table:**

For a given observed $x$, the 90% CI contains all $\theta$ values that have $x$ in their non-rejection region — equivalently, all $\theta$ for which $x$ is *not* highlighted in the rejection region.

#### Example 2 — Reading CI for x = 8

**Problem:** Using the table above, find the 90% CI for $\theta$ when $x = 8$.

**Solution:** Look at the column $x = 8$. The non-rejected $\theta$ values (those whose non-rejection regions include $x = 8$) are $\theta \in \{0.5, 0.6, 0.7, 0.8\}$.

$$\boxed{\text{90\% CI for } \theta \text{ when } x = 8: \quad [0.5,\; 0.8]}$$

**More exact answer** (using R to check many more $\theta$ values): $[0.42, 0.85]$.

**Interpretation:** The values $\theta = 0.5$ through $\theta = 0.8$ all "explain" the data $x = 8$ in the sense that they would not be rejected as null hypotheses at the 10% significance level.

#### Example 3 — Type I CI Error Rate

**Problem:** Explain why the expected Type I CI error rate is at most 0.092.

**Solution:** A Type I CI error occurs when the confidence interval does not contain the true $\theta_{\text{true}}$. This happens exactly when the observed data $x$ falls in the rejection region for $H_0: \theta = \theta_{\text{true}}$.

The probability of this is the significance for $\theta_{\text{true}}$, which from the table is at most 0.092 (achieved at $\theta = 0.5$).

So the Type I CI error rate is at most $\max_\theta \text{significance}(\theta) = 0.092 < \alpha = 0.1$.

> **Why can't we achieve exactly $\alpha = 0.1$?** Because the binomial is a discrete distribution and probability can't be parcelled out continuously. With a continuous distribution (like Normal), we can always achieve significance exactly equal to $\alpha$.

---

### 4. Summary Notes — Hypothesis Testing View

1. **Test statistic $x$ is random** → the confidence interval (which depends on $x$) is random.
2. **For each $\theta_0$:** run a significance test at level $\alpha$ by choosing rejection regions.
3. **For observed $x$:** the CI = all $\theta_0$ values not rejected.
4. **Discrete distributions:** significance may be strictly less than $\alpha$ → CI is actually "at least $(1-\alpha)$" confidence.

---

## Topic 4: Graphical View — CI and Non-Rejection Region Duality

### 1. The $\bar{x}$-$\mu$ Plane (Class 23 Problem 2)

**Setup:** $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$, $\sigma$ known.

The key insight is that the CI and the non-rejection region are **duals** of each other, related by a simple pivot (swap of axes). This can be visualized geometrically:

**Horizontal segments** (at height $\mu$): The non-rejection region for $H_0: \mu = \mu_0$ is:

$$\left[\mu_0 - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; \mu_0 + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$

This is a horizontal interval (a range of $\bar{x}$ values) drawn at height $\mu_0$ in the $\bar{x}$-$\mu$ plane.

**Vertical segments** (at position $\bar{x}$): The $(1-\alpha)$ CI is:

$$\left[\bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$

This is a vertical interval (a range of $\mu$ values) drawn at position $\bar{x}$ in the $\bar{x}$-$\mu$ plane.

**Key geometric fact:** Both horizontal and vertical segments have the **same half-width** $z_{\alpha/2} \cdot \sigma/\sqrt{n}$. They both run between the same two diagonal guide lines:

$$\mu = \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \quad \text{and} \quad \mu = \bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**The main point:** A horizontal non-rejection segment at $\mu$ and a vertical CI segment at $\bar{x}$ **intersect** if and only if:
- $\bar{x}$ is in the non-rejection region for $H_0: \mu = \mu_0$, AND
- $\mu_0$ is in the confidence interval for $\bar{x}$.

These two statements are equivalent. The geometry makes the duality obvious.

---

### 2. The $s^2$-$\sigma^2$ Plane (Class 23 Problem 4 — Chi-Square)

**Setup:** The chi-square non-rejection region and CI exhibit the same duality, but on an asymmetric axis because the $\chi^2$ distribution is skewed.

**Non-rejection region** for $H_0: \sigma^2 = \sigma_0^2$ (horizontal, expressed as a range of $s^2$ values):

$$\frac{c_{1-\alpha/2}\, \sigma_0^2}{n-1} \leq s^2 \leq \frac{c_{\alpha/2}\, \sigma_0^2}{n-1}$$

**Confidence interval** for $\sigma^2$ (vertical, expressed as a range of $\sigma^2$ values given data $s^2$):

$$\frac{(n-1)s^2}{c_{\alpha/2}} \leq \sigma^2 \leq \frac{(n-1)s^2}{c_{1-\alpha/2}}$$

**The guide lines** are now: $s^2 = \frac{c_{1-\alpha/2} \cdot \sigma^2}{n-1}$ and $s^2 = \frac{c_{\alpha/2} \cdot \sigma^2}{n-1}$.

Pivoting swaps the formulas for the guides from "$s^2$ as a function of $\sigma^2$" to "$\sigma^2$ as a function of $s^2$" — algebraically identical to what we did for the z-interval.

---

## Topic 5: View 3 — Formal Definition

### 1. Formal Definition

> **Definition:** A $(1-\alpha)$ confidence interval for $\theta$ is an interval statistic $I_x$ (computed from data $x$) such that:
>
> $$P\!\left(I_x \text{ contains } \theta_0 \;\middle|\; \theta = \theta_0\right) = 1 - \alpha$$
>
> for **all possible** values of $\theta_0$.

### 2. Unpacking the Definition

- **The interval $I_x$ is random** (it depends on random data $x$).
- **The parameter $\theta_0$ is fixed** (it either is or isn't in the interval).
- **The probability is over repeated sampling:** if we ran the experiment many times, $(1-\alpha)$ of the resulting intervals would contain $\theta_0$.
- **This holds for every possible $\theta_0$** — the frequentist must be correct regardless of which value is actually true.

### 3. Why the Formal Definition Can Be Surprising

The formal definition allows pathological cases. For instance, it's possible to construct a "100% confidence interval" that is always $(-\infty, \infty)$ — it trivially contains any $\theta_0$. This satisfies the formal definition but is useless.

Good confidence intervals have small width while maintaining the required coverage probability. This is the real challenge.

---

## Topic 6: Frequentist CI vs. Bayesian Probability Interval

### 1. Key Distinction

| Property | Frequentist CI | Bayesian Credible Interval |
|---|---|---|
| What's random? | The interval $I_x$ | The parameter $\theta$ |
| Requires a prior? | No | Yes |
| Probability statement | $P(\text{random } I_x \ni \theta_0 \mid \theta = \theta_0)$ | $P(\theta \in I \mid \text{data})$ |
| Interpretation | 95% of such intervals contain $\theta_0$ | 95% probability $\theta$ is in $I$ |
| Depends on experimenter's intentions? | Yes (stopping rules) | No (only likelihood) |

### 2. Shared Properties

Both Bayesian and frequentist intervals:
1. Start from a model $f(x|\theta)$ with unknown parameter $\theta$.
2. Given data $x$, produce an interval $I(x)$ specifying a range of $\theta$ values.
3. Come with a number (e.g., 0.95) that is the probability of *something*.

The difference is **what is random**: the interval or the parameter.

### 3. The Common Misinterpretation

> **⚠️ The Most Common Mistake:**
>
> "There is a 95% probability that the true mean is in [45, 55]."
>
> **This is wrong from a frequentist perspective.** The interval [45, 55] is a fixed interval. The true mean $\theta_0$ is a fixed (though unknown) number. It either is or isn't in [45, 55] — there's no probability to speak of.

**Analogy to disease testing:**

$P(T^+ | D^+) = 0.95$ does NOT mean $P(D^+ | T^+) = 0.95$.

The reason: the two probabilities are computed over different **populations being sampled**:
- $P(T^+ | D^+)$: sample from all people WITH the disease.
- $P(D^+ | T^+)$: sample from all people who TESTED positive.

Similarly:
- Frequentist 95% CI: sample from all **possible confidence intervals** (over repeated experiments). 95% of those intervals contain $\theta_0$.
- Bayesian 95% interval: sample from all **possible values of $\theta$** (from the prior). 95% of those values are in the specific computed interval.

Both are valid statements, but they sample from different populations and describe different events.

### 4. When Do Bayesian and Frequentist Intervals Agree?

With a lot of data and a diffuse (non-informative) prior, the two intervals tend to be very similar numerically. The philosophical difference remains, but the practical difference shrinks.

---

## Topic 7: Confidence Intervals for Non-Normal Data (Class 23, Prep-B)

### 1. Concept Overview

So far we assumed data is normally distributed. What happens when it isn't?

Two important cases:
1. **Bernoulli/Binomial data** — estimating a proportion $\theta$ (e.g., polling).
2. **Large samples from any distribution** — Central Limit Theorem rescues us.

---

### 2. Conservative Normal CI for Bernoulli Data

#### Setup

Data $x_1, x_2, \ldots, x_n \sim \text{Bernoulli}(\theta)$.

- Each $x_i \in \{0,1\}$.
- $E[x_i] = \theta$, $\text{Var}(x_i) = \theta(1-\theta)$.
- $\bar{x}$ estimates $\theta$.
- By CLT: $\bar{x} \approx N\!\left(\theta, \frac{\theta(1-\theta)}{n}\right)$ for large $n$.

#### The Trick: Bound the Standard Deviation

**Key Fact:** For $\theta \in [0,1]$:

$$\sigma_\theta = \sqrt{\theta(1-\theta)} \leq \frac{1}{2}$$

**Proof:** The variance $\theta(1-\theta)$ is a downward parabola in $\theta$, maximized at $\theta = 1/2$, where $\sigma^2 = 1/4$. So $\sigma \leq \sqrt{1/4} = 1/2$. $\square$

#### Conservative CI Formula

Replace the unknown $\sigma_\theta$ by its maximum $\frac{1}{2}$:

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{1}{2\sqrt{n}}$$

This interval is **conservative** because $\frac{1}{2\sqrt{n}}$ overestimates the true standard error $\frac{\sigma_\theta}{\sqrt{n}}$. Overestimating the standard error makes the interval wider, which only increases coverage → the confidence level is **at least** $(1-\alpha)$.

> **Definition — Conservative CI:** An interval that achieves a confidence level *at least as large* as the nominal level. Better to be conservative (cover too much) than anti-conservative (cover too little).

#### The 95% Rule of Thumb

For $\alpha = 0.05$: $z_{0.025} \approx 2$ (not exactly 1.96, but close enough for quick calculation):

$$\bar{x} \pm \frac{1}{\sqrt{n}}$$

This is the famous **rule of thumb 95% CI for a proportion**.

#### Key Formulas

| Confidence Level | Formula |
|---|---|
| $(1-\alpha)$ conservative | $\bar{x} \pm z_{\alpha/2} / (2\sqrt{n})$ |
| 95% rule of thumb | $\bar{x} \pm 1/\sqrt{n}$ |

#### Sample Size for Desired Margin of Error

Set $ME = z_{\alpha/2} / (2\sqrt{n})$ and solve for $n$:

$$n = \left(\frac{z_{\alpha/2}}{2 \cdot ME}\right)^2$$

---

### 3. Worked Examples — Bernoulli/Polling CIs

#### Concept Question: Overnight Polling

**Problem:** Pollsters do overnight polls with margin of error $\pm 4\%$. How many people are polled?

**Solution:** Using the 95% rule of thumb:

$$ME = \frac{1}{\sqrt{n}} = 0.04$$

$$\sqrt{n} = \frac{1}{0.04} = 25 \quad \Rightarrow \quad n = 625$$

**Answer:** (e) 600–1000 people. Specifically, $n \approx 625$.

---

#### Example 1 — Polling: Candidate Preference

**Problem:** A pollster asks 196 people and finds 120 prefer Candidate A. Find the 95% conservative normal CI for $\theta$ (proportion preferring A).

**Step 1:** Compute the sample proportion:

$$\bar{x} = \frac{120}{196} \approx 0.612$$

**Step 2:** Identify parameters:

$$n = 196, \quad \alpha = 0.05, \quad z_{0.025} = 1.96$$

**Step 3:** Apply the conservative CI formula:

$$ME = \frac{z_{\alpha/2}}{2\sqrt{n}} = \frac{1.96}{2 \times 14} = \frac{1.96}{28} \approx 0.070$$

$$\boxed{I \approx 0.612 \pm 0.070 = [0.542, 0.682]}$$

**Interpretation:** With 95% confidence, between 54.2% and 68.2% of the population prefers Candidate A. A clear majority appears to support A.

---

#### Example 2 — Comparing Two Polls

**Problem:** Two firms poll voters:
1. **Fast and First** polls 40 voters, finds 22 support A.
2. **Quick but Cautious** polls 400 voters, finds 190 support A.

Find point estimates and 95% rule-of-thumb CIs.

**Poll 1 (n = 40):**

$$\bar{x}_1 = \frac{22}{40} = 0.55$$

$$ME_1 = \frac{1}{\sqrt{40}} = \frac{1}{6.32} \approx 0.158$$

$$\text{95\% CI}_1 = 0.55 \pm 0.158 = [0.392, 0.708] = 55\% \pm 16\%$$

**Poll 2 (n = 400):**

$$\bar{x}_2 = \frac{190}{400} = 0.475$$

$$ME_2 = \frac{1}{\sqrt{400}} = \frac{1}{20} = 0.05$$

$$\text{95\% CI}_2 = 0.475 \pm 0.05 = [0.425, 0.525] = 47.5\% \pm 5\%$$

**Comparison:**

| Poll | $n$ | $\hat{\theta}$ | Margin of Error | Conclusion |
|---|---|---|---|---|
| Fast and First | 40 | 55% | ±16% | Inconclusive (CI spans 39%–71%) |
| Quick but Cautious | 400 | 47.5% | ±5% | Leans toward B (CI spans 42.5%–52.5%) |

**Why is Poll 2 more accurate?** Because $ME \propto 1/\sqrt{n}$. Quadrupling the sample size halves the margin of error. Poll 2 has a margin of 5% vs. 16% for Poll 1.

**Practical note:** Even though Poll 1 shows 55% for A and Poll 2 shows 47.5%, the wider CI in Poll 1 means the data is entirely consistent with $\theta < 0.5$ (i.e., A losing). The larger poll is far more informative.

---

#### Problem 1 from Class 23 — Polling CI

**Part (a):** Sample size for margin of error 0.01 with 95% confidence:

$$\frac{1}{\sqrt{n}} = 0.01 \quad \Rightarrow \quad \boxed{n = 10{,}000}$$

**Part (b):** Sample size for margin of error 0.01 with 80% confidence:

$\alpha = 0.2$, so $z_{\alpha/2} = z_{0.1} = \text{qnorm}(0.9) = 1.2816$.

$$\frac{z_{0.1}}{2\sqrt{n}} = 0.01 \quad \Rightarrow \quad \sqrt{n} = \frac{1.2816}{0.02} = 64.08 \quad \Rightarrow \quad \boxed{n \approx 4106}$$

**Part (c):** If $n = 900$, compute 95% and 80% CIs:

$$\text{95\% CI:} \quad \bar{x} \pm \frac{1}{\sqrt{900}} = \bar{x} \pm \frac{1}{30} \approx \bar{x} \pm 0.033$$

$$\text{80\% CI:} \quad \bar{x} \pm \frac{z_{0.1}}{2\sqrt{900}} = \bar{x} \pm \frac{1.2816}{60} \approx \bar{x} \pm 0.021$$

**Note:** The 95% CI has margin 0.033, the 80% CI has margin 0.021. Higher confidence = wider interval.

---

### 4. Large Sample CI via CLT

#### Concept Overview

When data comes from **any distribution** with finite mean $\mu$ and variance $\sigma^2$, the Central Limit Theorem guarantees:

$$\frac{\bar{x} - \mu}{s/\sqrt{n}} \xrightarrow{d} N(0,1) \text{ as } n \to \infty$$

where $s$ is the sample standard deviation.

**Large sample CI (valid for any distribution, large $n$):**

$$\boxed{\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}}$$

Note: We use $z_{\alpha/2}$ (not $t_{\alpha/2}$) because for large $n$ the distributions converge.

#### How Large Must $n$ Be?

Simulation results for $\text{Exp}(1)$ data (far from normal):

| $n$ | Nominal $1-\alpha$ | Simulated Confidence |
|---|---|---|
| 20 | 0.95 | 0.905 |
| 50 | 0.95 | 0.930 |
| 100 | 0.95 | 0.938 |
| 400 | 0.95 | 0.947 |

For comparison, same table for $N(0,1)$ data:

| $n$ | Nominal $1-\alpha$ | Simulated Confidence |
|---|---|---|
| 20 | 0.95 | 0.936 |
| 50 | 0.95 | 0.944 |
| 100 | 0.95 | 0.947 |
| 400 | 0.95 | 0.949 |

**Observations:**
- For exponential data, $n \approx 100$–$400$ is needed for the nominal and true confidence levels to be close.
- For normal data with $n = 20$, there's still a gap because we used $z_{\alpha/2} = 1.96$ instead of $t_{\alpha/2}(19) \approx 2.09$.

> **Rule of thumb:** For "well-behaved" distributions (not extremely skewed), $n \geq 30$–$50$ is usually adequate. For heavily skewed data (like exponential), $n \geq 100$ is safer.

---

## Topic 8: Exact Binomial CI (Class 23 Problem 3)

### 1. Concept Overview

For small $n$, the CLT approximation is poor and we should use exact methods. The **exact binomial CI** is computed directly from the binomial likelihood table.

### 2. Worked Example — Exact Binomial CI with Binomial(8, θ)

**Problem:** Use the Binomial(8, $\theta$) probability table to:
1. Find the (two-sided) rejection region with $\alpha = 0.10$ for each $\theta$.
2. Given $x = 7$, find the 90% CI for $\theta$.
3. Given $x = 4$, find the 90% CI for $\theta$.

**Likelihood table — Binomial(8, θ):**

| $\theta \backslash x$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | .430 | .383 | .149 | .033 | .005 | .000 | .000 | .000 | .000 |
| 0.3 | .058 | .198 | .296 | .254 | .136 | .047 | .010 | .001 | .000 |
| 0.5 | .004 | .031 | .109 | .219 | .273 | .219 | .109 | .031 | .004 |
| 0.7 | .000 | .001 | .010 | .047 | .136 | .254 | .296 | .198 | .058 |
| 0.9 | .000 | .000 | .000 | .000 | .005 | .033 | .149 | .383 | .430 |

**Step 1: Build rejection regions for each $\theta$**

For each $\theta$, include in the rejection region the $x$ values with smallest probability until the next addition would exceed $\alpha = 0.1$:

- **$\theta = 0.1$:** Reject $\{5,6,7,8\}$ (all have probability ≤ 0.005). These sum to ~0.005. Adding $x=4$ (0.005) keeps us under 0.1. Actually all of $\{4,5,6,7,8\}$ sum to about 0.005. Non-rejection: $\{0,1,2,3,4\}$ contains most of the probability mass.
- **$\theta = 0.5$:** Symmetric, reject the two tails: $\{0,1\}$ (prob 0.035) and $\{7,8\}$ (prob 0.035). Total = 0.07 ≤ 0.1. Adding $x=2$ or $x=6$ would add 0.109 → over 0.1. Non-rejection: $\{2,3,4,5,6\}$.
- **$\theta = 0.9$:** Mirror of $\theta = 0.1$. Reject $\{0,1,2,3,4\}$. Non-rejection: $\{5,6,7,8\}$.

**Step 2: Read off CI from the column**

**For $x = 7$:** Look at which $\theta$ values have $x = 7$ in their non-rejection region.
- $\theta = 0.1$: $x = 7$ is in rejection region → not in CI.
- $\theta = 0.3$: $x = 7$ is in rejection region (probability 0.001, very small) → not in CI.
- $\theta = 0.5$: $x = 7$ in rejection region (tail) → not in CI.
- $\theta = 0.7$: $x = 7$ in non-rejection region (probability 0.198) → **in CI**.
- $\theta = 0.9$: $x = 7$ in non-rejection region (probability 0.383) → **in CI**.

$$\boxed{\text{90\% CI for } \theta \text{ when } x = 7: \quad [0.7,\; 0.9]}$$

**For $x = 4$:**
- $\theta = 0.1$: non-rejection (0.005, marginal). Actually in non-rejection.
- $\theta = 0.3$: non-rejection (0.136). → **in CI**.
- $\theta = 0.5$: non-rejection (0.273, center of distribution). → **in CI**.
- $\theta = 0.7$: non-rejection (0.136). → **in CI**.
- $\theta = 0.9$: rejection (0.005, near tail). Actually may still be in non-rejection.

$$\boxed{\text{90\% CI for } \theta \text{ when } x = 4: \quad [0.3,\; 0.7]}$$

**Intuition:** When $x = 7$ (many successes), high values of $\theta$ are plausible. When $x = 4$ (half successes in 8 trials), mid-range $\theta$ values are plausible.

---

## Topic 9: Common Mistakes — Confidence Intervals

| Mistake | Correct Understanding |
|---|---|
| "The 95% CI contains the true mean with 95% probability" | The CI either contains the true mean or it doesn't. The 95% is about the long-run frequency of the *procedure*, not any specific interval |
| "A wider CI is more informative" | Wider CIs are less precise; they reflect either higher confidence or more uncertainty |
| "The CI is the same as a Bayesian credible interval" | They often coincide numerically with diffuse priors but are conceptually different |
| "Using z instead of t when σ is unknown" | Always use t when σ is unknown and n is small; z only for large n or known σ |
| "Constructing CI without checking normality or sample size" | For small n with non-normal data, the CLT may not apply; use exact methods or bootstrap |
| "A 95% CI means we're 95% confident the parameter is in the interval" | Informal usage; formally, the "confidence" is about the procedure, not the specific interval |
| "More data always makes the CI more accurate" | More data narrows the CI (better precision) but doesn't fix a biased estimator |

---

# PART II — CLASS 24: Bootstrap Confidence Intervals

---

## Topic 10: Introduction to the Bootstrap

### 1. Concept Overview

The **bootstrap** (developed by Bradley Efron, 1979) is a powerful computational technique for estimating the variability of any statistic, without requiring knowledge of the underlying distribution.

The name "bootstrap" comes from the phrase "pulling oneself up by one's own bootstraps" — you use the data itself to estimate the variability of estimates computed from the same data.

**The problem the bootstrap solves:** Given data from an *unknown* distribution $F$, how do we compute a confidence interval for the mean (or median, or any other statistic)?

### 2. Intuition

Imagine you want to know how variable your sample mean $\bar{x}$ is. Ideally, you'd collect many new samples from $F$ and see how much $\bar{x}$ varies. But you can't — you only have one sample.

**The bootstrap insight:** Use the empirical distribution of your data $F^*$ as a proxy for the true distribution $F$. Resample from $F^*$ many times (sampling *with replacement* from your own data) and see how much the statistic varies across resamples.

**Why does this work?** By the Law of Large Numbers, the empirical distribution $F^*$ converges to the true distribution $F$ as $n \to \infty$. More importantly, the *variation* of the statistic around its center is well-approximated even for modest $n$.

---

## Topic 11: Sampling and the Empirical Distribution

### 1. Sampling With vs. Without Replacement

**Without replacement:** Draw $k$ items, never returning any. No duplicates possible. Used in simple random sampling.

**With replacement:** Draw an item, note it, return it, repeat. Duplicates are possible. Used in bootstrap resampling.

> **Example:** Rolling an 8-sided die repeatedly is sampling with replacement from $\{1,2,3,4,5,6,7,8\}$.

**When does it not matter?** If the sample ($n = 400$) is tiny relative to the population ($N = 300$ million), sampling with and without replacement are essentially identical. The chance of the same person being chosen twice is negligibly small.

### 2. The Empirical Distribution

**Definition:** The **empirical distribution** $F^*$ of data $x_1, \ldots, x_n$ is the discrete distribution that assigns probability $1/n$ to each observed data point.

Equivalently: write the data values on slips of paper, put them in a hat, and draw one at random.

#### Example 3 — Empirical Distribution of Die Rolls

**Data (10 rolls of an 8-sided die):** $1, 1, 2, 3, 3, 3, 3, 4, 7, 7$

**Empirical distribution $F^*$:**

| Value | 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|
| $p^*(x)$ | 2/10 | 1/10 | 4/10 | 1/10 | 2/10 |

**True distribution $F$ (uniform on $\{1,\ldots,8\}$):**

| Value | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $p(x)$ | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 | 1/8 |

$F^*$ approximates $F$. With only 10 data points, the approximation is rough (e.g., 3 appears 4 times when it should appear about 1.25 times). With 1000 data points, $F^*$ would be much closer to $F$.

**Key property:** The expected value of $F^*$ is exactly the sample mean $\bar{x}$.

---

## Topic 12: Resampling

### 1. How to Resample

Given data $x_1, \ldots, x_n$:

1. Label data points $1$ through $n$.
2. Draw $n$ random integers (with replacement) from $\{1, \ldots, n\}$.
3. The corresponding data values form a **bootstrap sample** $x_1^*, x_2^*, \ldots, x_n^*$.

Because of replacement, some original data points appear multiple times, some not at all.

#### Example 5 — Resampling the Die Data

**Data:** $1, 1, 2, 3, 3, 3, 3, 4, 7, 7$ (labeled $x_1$ through $x_{10}$)

**Draw 5 indices (with replacement):** $5, 3, 6, 6, 1$

**Resample:** $x_5 = 3$, $x_3 = 2$, $x_6 = 3$, $x_6 = 3$, $x_1 = 1$ → resample = $\{3, 2, 3, 3, 1\}$

Note: $x_6 = 3$ appears twice (drawn twice), $x_2 = 1$ doesn't appear at all.

### 2. Star Notation

| Symbol | Meaning |
|---|---|
| $x_1, \ldots, x_n$ | Original data |
| $x_1^*, \ldots, x_n^*$ | Bootstrap resample (same size $n$) |
| $\bar{x}$ | Mean of original data |
| $\bar{x}^*$ | Mean of bootstrap resample |
| $v$ | Statistic computed from original data |
| $v^*$ | Same statistic computed from resample |

---

## Topic 13: The Bootstrap Principle

### 1. Formal Statement

Given:
1. Original data $x_1, \ldots, x_n \sim F$ (unknown true distribution)
2. Statistic $u$ computed from the data
3. Empirical distribution $F^*$
4. Bootstrap resample $x_1^*, \ldots, x_n^* \sim F^*$
5. Statistic $u^*$ computed from the resample

**The Bootstrap Principle:**
1. $F^* \approx F$
2. $u^* \approx u$ (bootstrap statistic approximates the original statistic)
3. **The variation of $u$ is approximated by the variation of $u^*$**

> **The real power is in point 3:** We can approximate the sampling distribution of any statistic $u$ by computing $u^*$ across many bootstrap resamples.

### 2. Why Resamples Are the Same Size as the Original

The variation of the statistic $u$ depends on $n$. If we want to approximate this variation, we must use resamples of the same size $n$. Using smaller resamples would overestimate the variance; using larger resamples would underestimate it.

### 3. What the Bootstrap Cannot Do

**Resampling cannot improve the point estimate.** If we compute $\bar{x}^*$ for many resamples, their average will be very close to $\bar{x}$ — not closer to the true $\mu$. Bootstrapping helps us estimate the *variability* of $\bar{x}$, not improve $\bar{x}$ itself.

### 4. Justification (Intuitive)

The bootstrap works because:
- The **amount of variation** in $u$ around its center is well-approximated as long as $F^*$ is close to $F$.
- The distribution of $\delta = \bar{x} - \mu$ (variation around the center) is well-approximated by the distribution of $\delta^* = \bar{x}^* - \bar{x}$ (variation around the bootstrap center).
- The centers $\mu$ and $\bar{x}$ may differ, but the **shapes** of the distributions around their respective centers are similar.

---

## Topic 14: Empirical Bootstrap Confidence Intervals

### 1. Two Methods

There are two main approaches:

| Method | Also Called | Key Idea |
|---|---|---|
| **Percentile** | Direct percentile | Use quantiles of $\bar{x}^*$ directly as the CI |
| **Basic** | Reverse percentile | Pivot using $\delta^* = \bar{x}^* - \bar{x}$ analogous to z/t pivoting |

---

### 2. Percentile Bootstrap CI

**Algorithm:**
1. Generate $B$ bootstrap resamples: $x_1^{*(1)}, \ldots$ through $x_1^{*(B)}, \ldots$
2. Compute $\bar{x}^{*(b)}$ for each resample $b = 1, \ldots, B$.
3. The $(1-\alpha)$ percentile CI is:

$$\left[\bar{x}^*_{\alpha/2},\; \bar{x}^*_{1-\alpha/2}\right]$$

where $\bar{x}^*_q$ denotes the $q$-quantile of the bootstrap means.

**Intuition:** Treat the distribution of $\bar{x}^*$ as an approximation to the distribution of $\bar{x}$. The CI stretches from the lower $\alpha/2$ percentile to the upper $1-\alpha/2$ percentile.

---

### 3. Basic (Reverse Percentile) Bootstrap CI

**Algorithm:**
1. Generate $B$ bootstrap resamples and compute $\bar{x}^{*(b)}$ for each.
2. Compute $\delta^{*(b)} = \bar{x}^{*(b)} - \bar{x}$ for each resample.
3. Find the $\alpha/2$ and $1-\alpha/2$ quantiles of $\delta^*$: call them $\delta^*_{\alpha/2}$ and $\delta^*_{1-\alpha/2}$.
4. The CI is:

$$\left[\bar{x} - \delta^*_{1-\alpha/2},\; \bar{x} - \delta^*_{\alpha/2}\right]$$

**Derivation (the pivot):**

The bootstrap principle says: distribution of $\delta = \bar{x} - \mu$ $\approx$ distribution of $\delta^* = \bar{x}^* - \bar{x}$.

If we knew the distribution of $\delta$, we'd find quantiles $\delta_{0.1}$ and $\delta_{0.9}$ such that:

$$P(\delta_{0.1} \leq \bar{x} - \mu \leq \delta_{0.9} \mid \mu) = 0.8$$

Rearranging (pivoting):

$$P(\bar{x} - \delta_{0.9} \leq \mu \leq \bar{x} - \delta_{0.1} \mid \mu) = 0.8$$

Replacing $\delta_{0.1}$ and $\delta_{0.9}$ with their bootstrap approximations $\delta^*_{0.1}$ and $\delta^*_{0.9}$:

$$\text{80\% Basic CI} = \left[\bar{x} - \delta^*_{0.9},\; \bar{x} - \delta^*_{0.1}\right]$$

> **Key observation:** The order is reversed! $\delta^*_{0.9}$ (the larger value) appears first (subtracted), giving the lower bound. This is the "reverse" in "reverse percentile."

---

### 4. Master Worked Example — Toy Data (Example 6)

**Data:** $30, 37, 36, 43, 42, 43, 43, 46, 41, 42$ ($n = 10$)

#### Part (a) — Point Estimate

$$\bar{x} = \frac{30+37+36+43+42+43+43+46+41+42}{10} = \frac{403}{10} = 40.3$$

#### Part (b) — 20 Bootstrap Samples

Using R, 20 bootstrap resamples are generated. The column means (bootstrap means $\bar{x}^*$) are:

$$39.9,\; 42.4,\; 39.4,\; 41.9,\; 42.3,\; 39.2,\; 40.1,\; 42.5,\; 41.2,\; 41.0,$$
$$40.7,\; 40.8,\; 40.9,\; 40.2,\; 40.7,\; 39.6,\; 41.0,\; 42.6,\; 41.2,\; 41.4$$

#### Part (c) — 80% Percentile CI

**Step 1:** Sort the 20 bootstrap means:

$$39.2,\; 39.4,\; 39.6,\; 39.9,\; 40.1,\; 40.2,\; 40.7,\; 40.7,\; 40.8,\; 40.9,$$
$$41.0,\; 41.0,\; 41.2,\; 41.2,\; 41.4,\; 41.9,\; 42.3,\; 42.4,\; 42.5,\; 42.6$$

**Step 2:** For 80% CI, find the 10th and 90th percentiles.
- $B = 20$ bootstrap samples.
- 10th percentile → 2nd value in sorted list = **39.4**
- 90th percentile → 18th value in sorted list = **42.4**

$$\boxed{\text{80\% Percentile Bootstrap CI} = [39.4,\; 42.4]}$$

**Interpretation:** Based on bootstrap resampling, we are 80% confident the true mean lies between 39.4 and 42.4.

#### Part (d) — 80% Basic CI

**Step 1:** Compute $\delta^{*(b)} = \bar{x}^{*(b)} - \bar{x}$ for each resample:

Subtracting $\bar{x} = 40.3$ from each bootstrap mean:

$$-0.4,\; 2.1,\; -0.9,\; 1.6,\; 2.0,\; -1.1,\; -0.2,\; 2.2,\; 0.9,\; 0.7,$$
$$0.4,\; 0.5,\; 0.6,\; -0.1,\; 0.4,\; -0.7,\; 0.7,\; 2.3,\; 0.9,\; 1.1$$

**Step 2:** Sort the $\delta^*$ values:

$$-1.1,\; -0.9,\; -0.7,\; -0.4,\; -0.2,\; -0.1,\; 0.4,\; 0.4,\; 0.5,\; 0.6,$$
$$0.7,\; 0.7,\; 0.9,\; 0.9,\; 1.1,\; 1.6,\; 2.0,\; 2.1,\; 2.2,\; 2.3$$

**Step 3:** Find quantiles:
- $\delta^*_{0.1}$ (10th percentile) = 2nd value = **-0.9** (Note: this is in the lower tail of $\delta^*$, but recall we need $\delta^*_{0.9}$ for the lower bound of the CI)
- Wait — careful about the formula. We need:
  - The **90th percentile** of $\delta^*$ ($\delta^*_{0.9}$): 18th value = **2.1** → used for the **lower** CI bound.
  - The **10th percentile** of $\delta^*$ ($\delta^*_{0.1}$): 2nd value = **-0.9** → used for the **upper** CI bound.

**Step 4:** Apply the basic CI formula:

$$\text{Lower bound} = \bar{x} - \delta^*_{0.9} = 40.3 - 2.1 = 38.2$$

$$\text{Upper bound} = \bar{x} - \delta^*_{0.1} = 40.3 - (-0.9) = 41.2$$

$$\boxed{\text{80\% Basic Bootstrap CI} = [38.2,\; 41.2]}$$

**Comparing the two methods:**

| Method | 80% CI | Width |
|---|---|---|
| Percentile | [39.4, 42.4] | 3.0 |
| Basic | [38.2, 41.2] | 3.0 |

Both have the same width in this case (since the bootstrap distribution is approximately symmetric). They differ in location when the distribution is skewed.

---

## Topic 15: Class 24 Board Problems

### Problem 1 — Empirical Bootstrap (Hand Calculation)

**Data:** $3, 8, 1, 8, 3, 3$ ($n = 6$)

**Bootstrap samples** (each column is one bootstrap trial of $n = 6$ values):

| Column | Sample |
|---|---|
| 1 | 8,1,3,8,3,3 |
| 2 | 8,3,1,1,3,8 |
| 3 | 1,3,1,3,1,8 |
| 4 | 8,1,8,1,8,3 |
| 5 | 3,3,1,3,8,8 |
| 6 | 8,8,3,3,3,3 |
| 7 | 3,3,3,8,8,1 |
| 8 | 1,3,8,8,3,1 |

#### Part (a) — 80% Percentile CI for the Mean

**Step 1:** Compute the original sample mean:

$$\bar{x} = \frac{3+8+1+8+3+3}{6} = \frac{26}{6} = 4.33$$

**Step 2:** Compute $\bar{x}^*$ for each bootstrap sample:

| Bootstrap | Values | $\bar{x}^*$ |
|---|---|---|
| 1 | 8,1,3,8,3,3 | 26/6 = 4.33 |
| 2 | 8,3,1,1,3,8 | 24/6 = 4.00 |
| 3 | 1,3,1,3,1,8 | 17/6 = 2.83 |
| 4 | 8,1,8,1,8,3 | 29/6 = 4.83 |
| 5 | 3,3,1,3,8,8 | 26/6 = 4.33 |
| 6 | 8,8,3,3,3,3 | 28/6 = 4.67 |
| 7 | 3,3,3,8,8,1 | 26/6 = 4.33 |
| 8 | 1,3,8,8,3,1 | 24/6 = 4.00 |

**Step 3:** Sort the $\bar{x}^*$ values:

$$2.83,\; 4.00,\; 4.00,\; 4.33,\; 4.33,\; 4.33,\; 4.67,\; 4.83$$

**Step 4:** Find the 10th and 90th percentiles (interpolating):
- 10th percentile: between 1st and 2nd values → $\bar{x}^*_{0.1} \approx 3.65$
- 90th percentile: between 7th and 8th values → $\bar{x}^*_{0.9} \approx 4.72$

$$\boxed{\text{80\% Percentile Bootstrap CI for Mean} = [3.65,\; 4.72]}$$

#### Part (b) — 80% Percentile CI for the Median

**Step 1:** Compute original sample median:

Sorted data: $1, 3, 3, 3, 8, 8$. Median = average of 3rd and 4th values = $(3+3)/2 = 3$.

$$m = 3$$

**Step 2:** Compute $m^*$ (bootstrap median) for each sample:

| Bootstrap | Values | Sorted | $m^*$ |
|---|---|---|---|
| 1 | 8,1,3,8,3,3 | 1,3,3,3,8,8 | 3.0 |
| 2 | 8,3,1,1,3,8 | 1,1,3,3,8,8 | 3.0 |
| 3 | 1,3,1,3,1,8 | 1,1,1,3,3,8 | 2.0 |
| 4 | 8,1,8,1,8,3 | 1,1,3,8,8,8 | 5.5 |
| 5 | 3,3,1,3,8,8 | 1,3,3,3,8,8 | 3.0 |
| 6 | 8,8,3,3,3,3 | 3,3,3,3,8,8 | 3.0 |
| 7 | 3,3,3,8,8,1 | 1,3,3,3,8,8 | 3.0 |
| 8 | 1,3,8,8,3,1 | 1,1,3,3,8,8 | 3.0 |

**Step 3:** Sorted $m^*$ values:

$$2.0,\; 3.0,\; 3.0,\; 3.0,\; 3.0,\; 3.0,\; 3.0,\; 5.5$$

**Step 4:** 10th and 90th percentiles:
- 10th percentile: $m^*_{0.1} \approx 2.7$ (interpolating between 2.0 and 3.0)
- 90th percentile: $m^*_{0.9} \approx 3.75$ (interpolating between 3.0 and 5.5)

$$\boxed{\text{80\% Bootstrap CI for Median} = [2.7,\; 3.75]}$$

---

#### Concept Question — Which Statistic is Easiest?

**Question:** For bootstrap CIs, which is easiest to compute:
(I) Mean, (II) Median, (III) 47th percentile?

**Answer: (g) All three are equally easy.**

The bootstrap code is essentially the same for all three statistics. Only one line changes — the function computing the statistic. In R, switching from `colMeans()` to `apply(bootstrap_sample, 2, median)` to `apply(bootstrap_sample, 2, function(x) quantile(x, 0.47))` is trivial.

> **This is one of the greatest strengths of the bootstrap:** It handles statistics that have no analytical sampling distribution (like the median, trimmed mean, or 47th percentile) just as easily as it handles the mean.

---

### Problem 2 — Parametric Bootstrap

**Setup:** Data from $\text{Binomial}(8, \theta)$: results are $6, 5, 5, 5, 7, 4$ (6 trials).

#### Part (a) — Estimate $\theta$

The MLE for $\theta$ in a $\text{Binomial}(8, \theta)$ model is:

$$\hat{\theta} = \frac{\sum_{i=1}^n k_i}{n \cdot 8}$$

**Derivation:**

The log-likelihood for $n$ trials with outcomes $k_1, \ldots, k_n$:

$$\ell(\theta) = \ln(c) + \left(\sum k_i\right) \ln(\theta) + \left(\sum (8-k_i)\right) \ln(1-\theta)$$

Setting $\ell'(\theta) = 0$:

$$\frac{\sum k_i}{\theta} - \frac{\sum (8-k_i)}{1-\theta} = 0 \quad \Rightarrow \quad \hat{\theta} = \frac{\sum k_i}{n \cdot 8}$$

**Numerically:**

$$\hat{\theta} = \frac{6+5+5+5+7+4}{6 \times 8} = \frac{32}{48} = \frac{2}{3} \approx 0.667$$

#### Part (b) — R Code for Parametric Bootstrap 80% CI

```r
data = c(6, 5, 5, 5, 7, 4)
size_binom = 8
n = length(data)

# Step 1: MLE estimate of theta
theta_hat = sum(data) / (n * size_binom)  # = 2/3 ≈ 0.667

# Step 2: Generate parametric bootstrap samples
# Draw from Binomial(8, theta_hat) — NOT from resampling the data
n_boot = 100
x = rbinom(n * n_boot, size_binom, theta_hat)
bootstrap_sample = matrix(x, nrow = n, ncol = n_boot)

# Step 3: Compute bootstrap theta_star for each bootstrap sample
theta_star = colSums(bootstrap_sample) / (n * size_binom)

# Step 4: Compute bootstrap differences
delta_star = theta_star - theta_hat

# Step 5: Find quantiles of delta_star
d = quantile(delta_star, c(0.1, 0.9))

# Step 6: Compute 80% CI (basic/reverse percentile method)
ci = theta_hat - c(d[2], d[1])

sprintf("80%% confidence interval for theta: [%.3f, %.3f]", ci[1], ci[2])
```

**Key difference from empirical bootstrap:** In parametric bootstrap, bootstrap samples are drawn from $\text{Binomial}(8, \hat{\theta})$ — a parametric distribution with estimated parameters — **not** by resampling the original 6 data points.

---

## Topic 16: Parametric Bootstrap — Theory and Examples

### 1. Concept Overview

The **parametric bootstrap** assumes we know the *form* of the distribution (e.g., exponential, binomial) but not the parameters. Steps:

0. Data $x_1, \ldots, x_n \sim F(\theta)$ with $\theta$ unknown.
1. Compute $\hat{\theta}$ (e.g., MLE).
2. Draw bootstrap samples from $F(\hat{\theta})$ (the parametric model with estimated parameters).
3. For each bootstrap sample, compute $\hat{\theta}^*$.
4. Compute $\delta^* = \hat{\theta}^* - \hat{\theta}$.
5. Use the distribution of $\delta^*$ to build a CI for $\theta$.

### 2. Comparison: Empirical vs. Parametric Bootstrap

| Feature | Empirical Bootstrap | Parametric Bootstrap |
|---|---|---|
| Assumption about $F$ | None (nonparametric) | Know the family (e.g., exponential) |
| Bootstrap samples from | Resampling original data | Parametric model $F(\hat{\theta})$ |
| More efficient? | No | Yes, if model is correct |
| Robust to model misspecification? | Yes | No — if model is wrong, results may be wrong |
| Best use | Unknown distribution | Known distribution family |

---

### 3. Worked Example — Exponential Parametric Bootstrap (Example 9)

**Problem:** Data $x_1, \ldots, x_{300} \sim \text{Exp}(\lambda)$ with $\bar{x} = 2$. Estimate $\lambda$ and give a 95% parametric bootstrap CI.

**Step 1: MLE of $\lambda$**

For $\text{Exp}(\lambda)$, the MLE is $\hat{\lambda} = 1/\bar{x} = 1/2 = 0.5$.

**R Code:**

```r
n = 300
xbar = 2
lambda_hat = 1.0 / xbar   # = 0.5

n_boot = 1000

# Key: draw from Exponential(lambda_hat), not resample
x = rexp(n * n_boot, lambda_hat)
bootstrap_sample = matrix(x, nrow = n, ncol = n_boot)

# Compute bootstrap lambda estimates
lambda_star = 1.0 / colMeans(bootstrap_sample)

# Bootstrap differences
delta_star = lambda_star - lambda_hat

# 95% CI using basic method
d = quantile(delta_star, c(0.05, 0.95))
ci = lambda_hat - c(d[2], d[1])

sprintf("95%% CI for lambda: [%.3f, %.3f]", ci[1], ci[2])
```

**Interpretation:** The resulting CI estimates how uncertain we are about $\lambda$ given 300 observations with sample mean 2.

---

## Topic 17: The Old Faithful Example (Example 7 & 8)

### Example 7 — Bootstrap CI for the Median

**Data:** 272 consecutive eruption durations of Old Faithful geyser (bimodal distribution — short and long eruptions).

**Data summary:** $x_{\text{median}} = 240$ seconds.

**Bootstrap procedure:**
1. Resample 1000 bootstrap samples of size 272 from the data.
2. Compute $x^*_{\text{median}}$ for each.
3. Find the 5th and 95th percentiles: $m^*_{0.05}$ and $m^*_{0.95}$.
4. 90% percentile CI: $[m^*_{0.05}, m^*_{0.95}]$.

**Result:** 90% bootstrap CI for median eruption duration = $[230, 246]$ seconds.

**Why bootstrap is essential here:** The theoretical sampling distribution of the median is complex and depends on the density at the median. Bootstrap sidesteps all this theory entirely.

---

### Example 8 — Estimating a Probability

**Problem:** Using Old Faithful data, estimate $P(|\bar{x} - \mu| > 5 | \mu)$.

**Procedure:**
1. Data: $x_1, \ldots, x_{272}$, mean $\bar{x} = 209.27$
2. Generate 1000 bootstrap samples; compute $\bar{x}^*$ for each.
3. Compute $\delta^* = \bar{x}^* - \bar{x}$ for each.
4. Bootstrap principle: distribution of $\delta^* \approx$ distribution of $\delta = \bar{x} - \mu$.
5. Estimate: $P(|\bar{x} - \mu| > 5) \approx P(|\delta^*| > 5)$ = fraction of bootstrap $\delta^*$ values with $|\delta^*| > 5$.

**Result:** One bootstrap simulation gives $P(|\delta^*| > 5) \approx 0.230$.

**Interpretation:** There's about a 23% chance the sample mean deviates from the true mean by more than 5 seconds, given these 272 data points.

---

## Topic 18: Which Bootstrap CI Method is Better?

### Percentile vs. Basic

| Situation | Recommendation |
|---|---|
| Symmetric bootstrap distribution | Either — they're equivalent |
| Skewed bootstrap distribution | Percentile (Hesterberg's recommendation) |
| Small $n$ with continuous population | Percentile (basic can be erratic) |
| Theoretical grounding preferred | Basic (has a clean algebraic derivation) |

**Hesterberg's bottom line:** "A bootstrap percentile interval is usually a good choice — much better than a symmetric interval like $t$, that pretends there is no skewness."

**Important nuance for the median:** For small samples, the nonparametric bootstrap draws from a discrete distribution (the data), while the true population may be continuous. This can cause the bootstrap distribution of the median to not look like the true sampling distribution. Even so, the percentile interval remains "not bad" — close to the exact CI in most cases.

---

## Topic 19: R Code Reference — Bootstrap

### Empirical Bootstrap CI (Percentile and Basic)

```r
# Data
x = c(30, 37, 36, 43, 42, 43, 43, 46, 41, 42)
n = length(x)
xbar = mean(x)     # Point estimate

n_boot = 10000     # More bootstrap samples = more accurate

# Generate bootstrap samples (n x n_boot matrix)
tmp_data = sample(x, n * n_boot, replace = TRUE)
bootstrap_sample = matrix(tmp_data, nrow = n, ncol = n_boot)

# Bootstrap means
xbar_star = colMeans(bootstrap_sample)

# ---------- PERCENTILE METHOD ----------
percentile_ci = quantile(xbar_star, c(0.1, 0.9))   # 80% CI
cat("80% Percentile CI:", percentile_ci, "\n")

# ---------- BASIC METHOD ----------
delta_star = xbar_star - xbar
d = quantile(delta_star, c(0.1, 0.9))
basic_ci = xbar - c(d[2], d[1])    # Note: reversed order
cat("80% Basic CI:", basic_ci, "\n")
```

### Parametric Bootstrap CI

```r
# Parametric bootstrap for Exponential(lambda)
n = 300
xbar = 2
lambda_hat = 1.0 / xbar

n_boot = 1000
x = rexp(n * n_boot, lambda_hat)
bootstrap_sample = matrix(x, nrow = n, ncol = n_boot)

lambda_star = 1.0 / colMeans(bootstrap_sample)
delta_star = lambda_star - lambda_hat

d = quantile(delta_star, c(0.025, 0.975))   # 95% CI
ci = lambda_hat - c(d[2], d[1])
cat("95% CI for lambda:", ci, "\n")
```

---

## Topic 20: Quick Summary — Classes 23 and 24

### Class 23: Confidence Intervals — Three Views

**View 1 (Standardized Statistics):**
- z-CI ($\sigma$ known): $\bar{x} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$
- t-CI ($\sigma$ unknown): $\bar{x} \pm t_{\alpha/2} \cdot s/\sqrt{n}$, $df = n-1$
- $\chi^2$-CI for $\sigma^2$: $[(n-1)s^2/c_{\alpha/2},\; (n-1)s^2/c_{1-\alpha/2}]$

**View 2 (Hypothesis Inversion):**
- CI = all $\theta_0$ values not rejected as null hypotheses given the data.
- Type I CI error rate = $\alpha$.
- Discrete distributions achieve CI coverage $\geq 1-\alpha$ (conservative).

**View 3 (Formal):**
- $P(I_x \ni \theta_0 \mid \theta = \theta_0) = 1-\alpha$ for all $\theta_0$.
- The interval $I_x$ is random; the parameter $\theta_0$ is fixed.

**Bernoulli/Polling CIs:**
- Conservative $(1-\alpha)$ CI: $\bar{x} \pm z_{\alpha/2}/(2\sqrt{n})$
- 95% rule of thumb: $\bar{x} \pm 1/\sqrt{n}$
- Required $n$ for margin $ME$: $n = (z_{\alpha/2}/(2 \cdot ME))^2$

**Large sample CI (any distribution, large $n$, via CLT):**
- $\bar{x} \pm z_{\alpha/2} \cdot s/\sqrt{n}$
- Adequate for $n \geq 100$ for heavily skewed distributions; $n \geq 30$ for mild skewness.

**Frequentist vs. Bayesian CI:**
- Frequentist: the interval is random, the parameter is fixed.
- Bayesian: the parameter is random (from the prior), the interval is fixed.
- Neither gives $P(\theta \in I)$ in the frequentist framework.

---

### Class 24: Bootstrap Confidence Intervals

**Key Concepts:**
- **Empirical distribution $F^*$:** assigns probability $1/n$ to each observed data point.
- **Resampling:** draw $n$ values with replacement from the original data.
- **Bootstrap principle:** the variation of $u^*$ (bootstrap statistic) approximates the variation of $u$ (original statistic).
- Bootstrap cannot improve point estimates, only estimate their variability.

**Empirical Bootstrap Methods:**
- **Percentile CI:** $[\bar{x}^*_{\alpha/2}, \bar{x}^*_{1-\alpha/2}]$ (quantiles of bootstrap means)
- **Basic CI:** $[\bar{x} - \delta^*_{1-\alpha/2}, \bar{x} - \delta^*_{\alpha/2}]$ where $\delta^* = \bar{x}^* - \bar{x}$

**Parametric Bootstrap:**
- Assumes known distribution family.
- Bootstrap samples drawn from $F(\hat{\theta})$ rather than resampling.
- More efficient if model is correct; fails if model is wrong.

**The power of bootstrap:**
- Handles any statistic (mean, median, percentile, correlation, etc.) with identical code.
- No need for analytical sampling distributions.
- Only requirement: data is a representative sample from the population.

**Practical recommendations:**
- Use $B \geq 1000$ bootstrap samples (10,000 for more precision).
- Use percentile method for skewed distributions.
- Parametric bootstrap is preferred when the distribution family is known.

---

## Appendix: Complete R Function Reference

```r
# DISTRIBUTIONS
# Normal
pnorm(q, mean=0, sd=1)     # P(X <= q)
qnorm(p, mean=0, sd=1)     # q such that P(X <= q) = p
dnorm(x, mean=0, sd=1)     # pdf at x

# t-distribution
pt(q, df)                  # P(T <= q)
qt(p, df)                  # t such that P(T <= t) = p

# Chi-square
pchisq(q, df)              # P(X <= q)
qchisq(p, df)              # quantile

# Beta (for Bayesian)
pbeta(q, shape1, shape2)   # P(X <= q)
qbeta(p, shape1, shape2)   # quantile

# SAMPLING
sample(x, size, replace=TRUE)   # resample from data
rbinom(n, size, prob)           # binomial samples
rexp(n, rate)                   # exponential samples
rnorm(n, mean, sd)              # normal samples

# STATISTICS
mean(x)
median(x)
var(x)                          # sample variance (divides by n-1)
sd(x)                           # sample std deviation
quantile(x, probs)              # compute quantiles

# CRITICAL VALUES
qnorm(0.975)                    # 1.96 (95% z-critical)
qnorm(0.95)                     # 1.645 (90% z-critical)
qnorm(0.90)                     # 1.282 (80% z-critical)
qt(0.975, df=n-1)               # t-critical for 95% CI
qchisq(0.95, df=n-1)            # chi-sq right critical
qchisq(0.05, df=n-1)            # chi-sq left critical
```

---

*End of MIT 18.05 Study Notes — Classes 23 & 24.*

*Source: MIT OpenCourseWare, 18.05 Introduction to Probability and Statistics, Spring 2022. Jeremy Orloff and Jonathan Bloom.*

*These notes cover: Confidence Intervals (Three Views), Conservative CI for Bernoulli Data, Large Sample CI via CLT, Exact Binomial CI, Empirical Bootstrap (Percentile and Basic Methods), Parametric Bootstrap, and complete R code.*
