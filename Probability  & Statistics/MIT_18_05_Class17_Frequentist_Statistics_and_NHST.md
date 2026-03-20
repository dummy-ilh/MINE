# MIT 18.05 — Introduction to Probability and Statistics
## Class 17: The Frequentist School & Null Hypothesis Significance Testing I
### Complete Study Notes | Spring 2022

> **Authors:** Jeremy Orloff and Jonathan Bloom  
> **Source:** MIT OpenCourseWare — 18.05, Spring 2022  
> **Topics Covered:** Frequentist vs. Bayesian approaches, statistics and sampling distributions, NHST framework, significance level, power, Type I/II errors, critical values, p-values, z-tests

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [The Fork in the Road: Bayesian vs. Frequentist](#2-the-fork-in-the-road-bayesian-vs-frequentist)
3. [What Is a Statistic?](#3-what-is-a-statistic)
4. [Introduction to Null Hypothesis Significance Testing](#4-introduction-to-null-hypothesis-significance-testing)
5. [The Five Ingredients of NHST](#5-the-five-ingredients-of-nhst)
6. [NHST Terminology: Deep Dive](#6-nhst-terminology-deep-dive)
7. [Simple vs. Composite Hypotheses](#7-simple-vs-composite-hypotheses)
8. [Type I and Type II Errors](#8-type-i-and-type-ii-errors)
9. [Significance Level and Power](#9-significance-level-and-power)
10. [High and Low Power Tests](#10-high-and-low-power-tests)
11. [Designing a Hypothesis Test: Step-by-Step](#11-designing-a-hypothesis-test-step-by-step)
12. [Critical Values](#12-critical-values)
13. [p-Values](#13-p-values)
14. [The z-Test for Normal Hypotheses](#14-the-z-test-for-normal-hypotheses)
15. [Preview: Chi-Square and t-Tests](#15-preview-chi-square-and-t-tests)
16. [All Worked Examples — Complete Solutions](#16-all-worked-examples--complete-solutions)
17. [Concept Questions with Deep Explanations](#17-concept-questions-with-deep-explanations)
18. [Common Mistakes](#18-common-mistakes)
19. [Quick Reference Summary](#19-quick-reference-summary)

---

## 1. Learning Goals

By the end of Class 17, you should be able to:

1. **Explain** the philosophical and practical difference between the Bayesian and frequentist approaches to statistics.
2. **Define** a statistic precisely and distinguish statistics from non-statistics.
3. **Know** all the NHST vocabulary: null hypothesis, alternative hypothesis, test statistic, null distribution, rejection region, significance level, power, Type I and Type II errors, p-value, critical value.
4. **Design and run** a significance test for Bernoulli or Binomial data.
5. **Compute** a p-value for a normal (z-test) hypothesis and use it to make a decision.

---

## 2. The Fork in the Road: Bayesian vs. Frequentist

### 2.1 The Common Starting Point

Both schools of statistics start with **probability** and both know and love Bayes' theorem:

$$P(H \mid D) = \frac{P(D \mid H)\, P(H)}{P(D)}$$

When the prior $P(H)$ is known exactly, **all statisticians use Bayes' formula**. There is no controversy when the prior is known. Examples where priors are known:
- A die from a drawer with a known distribution of die types.
- A disease with a known prevalence in the population.
- A screening test of known accuracy.

The controversy begins when there is **no universally accepted prior** — different people have different prior beliefs, yet we still want to make useful inferences from data.

### 2.2 The Philosophical Split

The fundamental difference concerns the **meaning of probability**:

| Aspect | Bayesian | Frequentist |
|---|---|---|
| What is probability? | Degree of belief (subjective) | Long-run frequency of repeatable events (objective) |
| Can parameters have distributions? | Yes — a distribution reflects uncertainty about a fixed value | No — parameters are fixed constants, not random |
| Prior needed? | Yes — always requires $P(H)$ | No — avoids priors entirely |
| What is computed | Posterior $P(H \mid D)$ | Likelihood $L(H; D) = P(D \mid H)$ |
| What is valid on $\theta$? | Prior, likelihood, and posterior | Only the likelihood |

### 2.3 The Frequentist Definition of Probability

> **The Frequentist Principle:** Probability represents the long-term relative frequency of outcomes in repeatable random experiments.
>
> "A coin has probability 1/2 of heads" means: the relative frequency of heads converges to 1/2 as the number of flips goes to infinity.

**Implication:** To a frequentist, it is nonsensical to say "the probability that $\theta = 0.5$" because $\theta$ has a fixed (though unknown) value. It isn't random. There is no experiment to run repeatedly that would give a frequency distribution for $\theta$.

### 2.4 Worked Example 1 — The Bent Coin

**Problem:** A bent coin has unknown probability $\theta$ of heads.

**Bayesian view:** $\theta$ has a fixed value, but we represent our uncertainty about it with a prior pdf $f(\theta)$. After seeing data, we update to the posterior $f(\theta \mid \text{data})$.

**Frequentist view:** $\theta$ is a fixed constant. There can be no prior pdf $f(\theta)$ — it is nonsensical to put a probability distribution on a constant. However, $P(\text{heads} \mid \theta) = \theta$ is perfectly valid to both, since this describes the long-term frequency of heads if we flipped this coin many times with a fixed $\theta$.

> **Key Statement:** Bayesians put probability distributions on everything (parameters and data). Frequentists put probability distributions only on (random, repeatable, observable) data given a hypothesis. For the frequentist, **only the likelihood has meaning**.

### 2.5 The Roadmap Diagram

```
         Probability (mathematics)
                    |
         P(H|D) = P(D|H)P(H)/P(D)
        [Everyone uses Bayes when prior is known]
                    |
            ________|________
           |                 |
      Bayesian path    Frequentist path
           |                 |
    Posterior:          Likelihood only:
    P(H|D) = P(D|H)·P_prior(H)   L(H;D) = P(D|H)
          P(D)
           |                 |
    Develop prior from    Draw inferences from
    best available info   likelihood alone
```

**Practical reasons for the split:**
- **Ease of computation:** Frequentist methods are often simpler to compute (no integral for normalizing constants).
- **Philosophy:** Frequentists value objectivity — results shouldn't depend on subjective prior choices.
- **Reproducibility:** If two scientists use different priors, they get different posteriors. Frequentist methods give the same answer regardless of prior beliefs.

---

## 3. What Is a Statistic?

### 3.1 Concept Overview

In order to draw inferences from data, we need to compute things from data. A **statistic** is anything that can be computed from the data and other known (fixed, not estimated) values.

### 3.2 Working Definition

> **Definition (Statistic):** A **statistic** is anything that can be computed from data and known values. It cannot depend on any unknown parameter values.
>
> More precisely: a statistic is a rule for computing something from data, and the value of the statistic is what is computed.

**Why this matters:** A statistic is a **random variable** (because it's computed from random data). Its probability distribution is called the **sampling distribution**.

### 3.3 Types of Statistics

| Type | Definition | Example |
|---|---|---|
| **Point statistic** | A single number computed from data | Sample mean $\bar{x}$, sample max |
| **Interval statistic** | An interval computed from data | $[\min(x_i), \max(x_i)]$ |
| **Set statistic** | A set computed from data | Set of dice types consistent with the observed roll |

### 3.4 Is It a Statistic? — Decision Rule

**Ask yourself:** Can I compute this value knowing only the data $x_1, \ldots, x_n$ and any explicitly given constants?

- If YES → it is a statistic.
- If NO (because it requires knowing an unknown parameter $\theta$, $\mu$, $\sigma$, etc.) → it is NOT a statistic.

### 3.5 Key Examples from the PDF

**Setup:** Data $x_1, \ldots, x_n$ is a sample from $\text{N}(\mu, \sigma^2)$ where $\mu$ and $\sigma$ are unknown.

| Expression | Statistic? | Reason |
|---|---|---|
| Median of $x_1, \ldots, x_n$ | **Yes** | Computed from data alone |
| $[q_{0.25}, q_{0.75}]$ of $\text{N}(\mu, \sigma^2)$ | **No** | This is a property of the distribution, not of the data. It depends on unknown $\mu$, $\sigma$ |
| $z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}}$ | **No** | Requires knowing the true (unknown) $\mu$ and $\sigma$ |
| $\{x_i : |x_i - \bar{x}| < 1\}$ | **Yes** | $\bar{x}$ depends only on data; the whole expression can be computed from data |
| $z = \frac{\bar{x} - 5}{3/\sqrt{n}}$ | **Yes** | 5 and 3 are given/known constants; computed from data and knowns |
| $z = \frac{\bar{x} - \mu_0}{\sigma_0/\sqrt{n}}$ where $\mu_0, \sigma_0$ are given | **Yes** | $\mu_0, \sigma_0$ are given (known) values |

**Critical insight:** The expression $z = \frac{\bar{x} - 5}{3/\sqrt{n}}$ IS a statistic even though it looks like standardization — the 5 and the 3 are given constants, not unknown parameters. But $z = \frac{\bar{x} - \mu}{\sigma/\sqrt{n}}$ is NOT a statistic when $\mu$ and $\sigma$ are unknown parameters of the distribution.

### 3.6 Worked Example — Set Statistic (Dice)

**Setup:** Five dice: 4, 6, 8, 12, 20-sided. Pick one and roll it. The value of the roll is the data.

**Statistic:** "The set of dice types for which this roll is possible."

- If roll = 10: set statistic = $\{12, 20\}$ (only 12- and 20-sided dice can produce a 10)
- If roll = 7: set statistic = $\{8, 12, 20\}$
- If roll = 3: set statistic = $\{4, 6, 8, 12, 20\}$ (any die could produce a 3)

**Is this a statistic?** Yes — given the roll value (data), we can compute which dice could have produced it using only known facts about dice.

### 3.7 Sampling Distribution

> **Definition:** The **sampling distribution** of a statistic is the probability distribution of the statistic, computed under a given hypothesis.

**Example:** If data $x_1, \ldots, x_n \sim \text{N}(\mu, \sigma^2)$, then the sample mean $\bar{x}$ has sampling distribution:

$$\bar{x} \sim \text{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$$

This is the most important sampling distribution in frequentist statistics. It says:
- The sample mean is centered at the true mean $\mu$ (unbiased).
- Its variance shrinks by a factor of $n$ (more data → more precise estimate).

---

## 4. Introduction to Null Hypothesis Significance Testing

### 4.1 The Core Idea

**Null Hypothesis Significance Testing (NHST)** is the dominant framework for frequentist inference. The fundamental logic is:

> "If the null hypothesis were true, how likely is it that we would observe data at least as extreme as what we actually observed? If this probability is very small, we reject the null hypothesis."

This is a form of **proof by contradiction**: we assume the null hypothesis is true, then check whether the data is consistent with that assumption.

### 4.2 Motivating Examples

**Example 1 — Coin fairness:** Toss a coin 100 times.
- 85 heads → Very likely unfair (extreme under $H_0$: fair coin)
- 60 heads → Suggestive but not definitive
- 52 heads → No evidence of unfairness

NHST gives us a rigorous framework for making these judgments quantitatively.

**Example 2 — Medical treatment:** Compare a new treatment to a placebo. What level of evidence convinces us the treatment is effective? NHST provides a principled answer.

### 4.3 The Logic in Three Steps

1. **Assume** the null hypothesis $H_0$ is true.
2. **Compute** the probability of observing data at least as extreme as what we got, assuming $H_0$.
3. **Decision:** If this probability (the p-value) is small (below a threshold $\alpha$), the data is very unlikely under $H_0$, so we reject $H_0$ in favor of $H_A$.

### 4.4 Important Note: Frequentist Philosophy

In NHST, we make a decision based only on the **likelihood** $\phi(x \mid H_0)$ — the probability of the data under the null hypothesis. We never use a prior. We never compute a posterior.

> **What we do NOT say:** "The probability that $H_0$ is true is $p$."  
> **What we DO say:** "If $H_0$ were true, the probability of seeing data this extreme is $p$."

This is a subtle but crucial distinction.

---

## 5. The Five Ingredients of NHST

Every null hypothesis significance test requires exactly these five components:

### 5.1 Ingredient 1: The Null Hypothesis $H_0$

> **Definition:** The **null hypothesis** $H_0$ is the default assumption — the hypothesis we assume to be true unless we have compelling evidence against it.

**How to choose $H_0$:**
- Often the "boring," "no effect," or "status quo" hypothesis.
- Examples: "the coin is fair," "the treatment has no effect," "the defendant is innocent."
- Preferably simple (fully specifies the distribution) so we can compute exact probabilities.

**Important:** We never prove $H_0$ — we either reject it or fail to reject it.

### 5.2 Ingredient 2: The Alternative Hypothesis $H_A$

> **Definition:** The **alternative hypothesis** $H_A$ is what we accept as the best explanation if we reject $H_0$.

**How to choose $H_A$:**
- $H_A$ should be the "interesting" finding you're looking for.
- Should guide whether the rejection region is one-sided or two-sided.

### 5.3 Ingredient 3: The Test Statistic $X$

> **Definition:** The **test statistic** $X$ is a statistic computed from the data. It is a random variable because it depends on random data.

The test statistic summarizes the data into a single number that we compare against the rejection region.

**Common test statistics:**
- $X$ = number of heads in $n$ coin flips
- $\bar{X}$ = sample mean
- $Z$ = standardized sample mean
- $T$ = $t$-statistic (for unknown variance)
- $\chi^2$ = chi-square statistic (for goodness of fit)

### 5.4 Ingredient 4: The Null Distribution

> **Definition:** The **null distribution** is the probability distribution of the test statistic $X$ assuming $H_0$ is true. It is $\phi(x \mid H_0)$.

This is the crucial distribution — everything in NHST is computed relative to the null distribution.

**Example:** If $H_0$: coin is fair ($\theta = 0.5$) and $X$ = number of heads in 10 flips, then:

$$X \mid H_0 \sim \text{Binomial}(10, 0.5)$$

### 5.5 Ingredient 5: The Rejection Region

> **Definition:** The **rejection region** (also called the critical region) is the set of values of the test statistic for which we reject $H_0$ in favor of $H_A$.
>
> The **non-rejection region** (also loosely called the acceptance region) is the complement — if $X$ falls here, we do not reject $H_0$.

**Key design principle:** The rejection region should consist of values that are:
1. **Extreme under $H_0$** (unlikely given null hypothesis)
2. **More likely under $H_A$** (consistent with alternative hypothesis)

---

## 6. NHST Terminology: Deep Dive

### 6.1 The Master Example (Example 3 from PDF)

**Problem:** Test whether a coin is fair. Flip it 10 times.

**Complete NHST setup:**

1. **Null hypothesis:** $H_0$: coin is fair, i.e. $\theta = 0.5$
2. **Alternative hypothesis:** $H_A$: coin is not fair, i.e. $\theta \neq 0.5$ (two-sided)
3. **Test statistic:** $X$ = number of heads in 10 flips
4. **Null distribution:** $X \mid H_0 \sim \text{Binomial}(10, 0.5)$

**Probability table for the null distribution:**

| $x$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p(x \mid H_0)$ | 0.001 | 0.010 | 0.044 | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | 0.044 | 0.010 | 0.001 |

5. **Rejection region:** Under $H_0$, we expect about 5 heads. Values far from 5 are evidence against $H_0$. Set rejection region = $\{0, 1, 2, 8, 9, 10\}$.

**Probability of rejecting $H_0$ when it is true:**

$$P(\text{reject } H_0 \mid H_0 \text{ true}) = 0.001 + 0.010 + 0.044 + 0.044 + 0.010 + 0.001 = 0.11$$

**Notes:**
1. $H_0$ is the **cautious default** — we won't claim the coin is unfair unless we have compelling evidence.
2. The rejection region consists of outcomes that are **extreme under $H_0$** and **more likely under $H_A$**.
3. Getting $x = 3$ (in the non-rejection region) does not prove $H_0$ is true. We say: "The data does not support rejecting $H_0$."

### 6.2 The Non-Rejection Region Is NOT an "Acceptance" Region

> **Critical Distinction:** We say "do not reject $H_0$" rather than "accept $H_0$" because we can never prove the null hypothesis.
>
> "You can never prove the null hypothesis."

Failing to reject $H_0$ simply means the data is not extreme enough to be convincing evidence against it. The data might still be consistent with many other hypotheses.

---

## 7. Simple vs. Composite Hypotheses

### 7.1 Definitions

> **Definition (Simple hypothesis):** A hypothesis is **simple** if it completely specifies the probability distribution of the data.

> **Definition (Composite hypothesis):** A hypothesis is **composite** if it does not completely specify the distribution — it encompasses a range of possible distributions.

### 7.2 Examples

**Example 3 (continued):**
- $H_0$: $\theta = 0.5$ → distribution is Binomial$(10, 0.5)$ — **fully specified → simple**
- $H_A$: $\theta \neq 0.5$ → distribution is Binomial$(10, \theta)$ for some unknown $\theta \neq 0.5$ — **not fully specified → composite**

**Example 4 — Both simple:**

$$H_0: \text{data} \sim \text{N}(0, 1) \qquad H_A: \text{data} \sim \text{N}(1, 1)$$

Both fully specify a distribution → both simple.

**Example 5 — Both composite:**

$$H_0: \text{data} \sim \text{Poisson}(\lambda) \text{ (unknown } \lambda\text{)} \qquad H_A: \text{data is not Poisson}$$

Neither fully specifies a distribution → both composite.

**Example 6 — ESP test:**

$$H_0: T \sim \text{Binomial}(100, 0.25) \quad (\text{simple: no ESP})$$

$$H_A: T \sim \text{Binomial}(100, p) \text{ with } p > 0.25 \quad (\text{composite: has ESP})$$

or two-sided:

$$H_A: T \sim \text{Binomial}(100, p) \text{ with } p \neq 0.25 \quad (\text{composite: ESP or anti-ESP})$$

> **Intuition:** A simple hypothesis is like a precise claim ("it's exactly this distribution"), while a composite hypothesis is a family of claims ("it's something in this category"). Simple null hypotheses are preferred because they let us compute exact probabilities.

---

## 8. Type I and Type II Errors

### 8.1 The Error Table

No hypothesis test is perfect. There are two ways to be wrong:

|  | $H_0$ is actually true | $H_A$ is actually true |
|---|---|---|
| **We reject $H_0$** | **Type I Error** (false positive) | Correct decision ✓ |
| **We don't reject $H_0$** | Correct decision ✓ | **Type II Error** (false negative) |

**Type I Error:** We reject $H_0$ when it is actually true. (False alarm.)  
**Type II Error:** We fail to reject $H_0$ when $H_A$ is actually true. (Miss.)

### 8.2 Intuition via the Criminal Justice Analogy

$$H_0 = \text{defendant is innocent} \qquad H_A = \text{defendant is guilty}$$

- **Type I Error:** Convicting an innocent person. (False positive — we "reject innocence" when it's true.)
- **Type II Error:** Acquitting a guilty person. (False negative — we "don't reject innocence" when it's false.)

The criminal justice standard "beyond a reasonable doubt" means we demand a very small Type I error rate.

### 8.3 The Tradeoff

> **Fundamental Tension:** Reducing the Type I error rate (making it harder to reject $H_0$) increases the Type II error rate (making it easier to miss a true $H_A$), and vice versa.

You can't minimize both simultaneously. This is why we must choose a significance level $\alpha$ in advance.

---

## 9. Significance Level and Power

### 9.1 Definitions

> **Definition (Significance Level):** The **significance level** $\alpha$ of a test is the probability of a Type I error:
>
> $$\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true}) = P(\text{Type I error})$$

> **Definition (Power):** The **power** of a test is the probability of correctly rejecting $H_0$ when $H_A$ is true:
>
> $$\text{Power} = P(\text{reject } H_0 \mid H_A \text{ is true}) = 1 - P(\text{Type II error})$$

### 9.2 The Four Probabilities

| Decision | $H_0$ true | $H_A$ true |
|---|---|---|
| Reject $H_0$ | $\alpha$ (Type I error rate) | Power = $1 - P(\text{Type II})$ |
| Don't reject $H_0$ | $1 - \alpha$ | $P(\text{Type II error})$ |

**Ideal test:** Small $\alpha$ (near 0) AND large power (near 1).

### 9.3 Analogies

**Analogy 1 (Signal detection):**
- $H_0$: "nothing noteworthy is going on" (coin is fair, treatment doesn't work)
- $H_A$: "something interesting is happening" (coin is biased, treatment works)
- Power = probability of detecting something interesting when it's present
- Significance level = probability of mistakenly claiming something interesting occurred

**Analogy 2 (Medical screening):**
- $H_0$: patient is healthy
- $H_A$: patient has the disease
- Type I error = false positive (alarm a healthy patient)
- Type II error = false negative (miss a sick patient)
- Power = sensitivity = $P(\text{positive test} \mid \text{diseased})$

### 9.4 Computing Significance and Power (Example 3 Continued)

**Rejection region:** $\{0, 1, 2, 8, 9, 10\}$ (same for all three computations below)

**Three probability distributions:**

| $x$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p(x \mid \theta=0.5)$ | **0.001** | **0.010** | **0.044** | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | **0.044** | **0.010** | **0.001** |
| $p(x \mid \theta=0.6)$ | 0.000 | 0.002 | 0.011 | 0.042 | 0.111 | 0.201 | 0.251 | 0.215 | **0.121** | **0.040** | **0.006** |
| $p(x \mid \theta=0.7)$ | 0.000 | 0.000 | 0.001 | 0.009 | 0.037 | 0.103 | 0.200 | 0.267 | **0.233** | **0.121** | **0.028** |

*(Bold = values in rejection region)*

**Significance level** = sum of orange (rejection region) values in $H_0$ row:

$$\alpha = 0.001 + 0.010 + 0.044 + 0.044 + 0.010 + 0.001 = \mathbf{0.11}$$

**Power when $\theta = 0.6$** = sum of rejection region values in $\theta = 0.6$ row:

$$\text{Power}(\theta=0.6) = 0.000 + 0.002 + 0.011 + 0.121 + 0.040 + 0.006 = \mathbf{0.180}$$

**Power when $\theta = 0.7$** = sum of rejection region values in $\theta = 0.7$ row:

$$\text{Power}(\theta=0.7) = 0.000 + 0.000 + 0.001 + 0.233 + 0.121 + 0.028 = \mathbf{0.384}$$

### 9.5 Interpretation of Power Results

- Power is higher for $\theta = 0.7$ than for $\theta = 0.6$. This makes intuitive sense: a 0.7 coin deviates more from a fair coin (0.5), so it's easier to detect.
- **General principle:** Power increases as the alternative hypothesis moves farther from the null hypothesis.
- Distinguishing a fair coin from $\theta = 0.51$ would be nearly impossible with only 10 flips.

> **Key Insight for Machine Learning/AI:** Power analysis is critical for A/B testing. If your effect size is small (e.g., a 1% lift in click rate), you need a very large sample to have adequate power to detect it. Running a test with insufficient power is a waste of resources — you're likely to miss the effect even if it's real.

---

## 10. High and Low Power Tests

### 10.1 Conceptual Diagrams

**High Power Test:** The null and alternative distributions are well separated.

```
         φ(x|H_A)         φ(x|H_0)
              |               |
              ▼               ▼
     ____     |            ____
    /    \    |           /    \
___/      \___↑__________/      \___
     ←reject H₀→|←— non-reject H₀ —→
     
Null mean: 0, Alternative mean: -4 (4σ apart)
SMALL overlap → HIGH power
```

**Low Power Test:** The null and alternative distributions are close together.

```
     φ(x|H_A)    φ(x|H_0)
          |           |
          ▼           ▼
        ____        ____
       /    \      /    \
______/      \____/      \______
←reject H₀→|←— non-reject H₀ —→

Null mean: 0, Alternative mean: -0.4 (0.4σ apart)
LARGE overlap → LOW power
```

### 10.2 What Determines Power?

| Factor | Effect on Power |
|---|---|
| Effect size (distance between $H_0$ and $H_A$) | Larger effect → higher power |
| Sample size $n$ | More data → smaller variance → higher power |
| Significance level $\alpha$ | Larger $\alpha$ (more permissive) → higher power, but more Type I errors |
| Variance of the null distribution | Smaller variance → less overlap → higher power |

> **Key Insight:** We can increase power by increasing the amount of data, thereby decreasing the variance of both the null and alternative distributions. In experimental design, it is critical to determine the required sample size to achieve desired power before running the experiment.

### 10.3 Example 7 — Drug vs. Placebo

**Setup:**
- $H_0$: drug does not work better than placebo
- $H_A$: drug works better than placebo

- **Power** = probability the test concludes the drug is better, given that it truly is better.
- **Significance level** = probability the test concludes the drug is better, given that it isn't.

**Acceptable significance levels:**
- Testing chocolate cocoa content: $\alpha = 0.10$ is probably fine.
- Forensic fingerprint identification for murder trial: $\alpha = 0.10$ is definitely NOT acceptable (10% chance of falsely convicting an innocent person).

---

## 11. Designing a Hypothesis Test: Step-by-Step

### 11.1 The Five-Step Design Process

**Step 1: Choose the null hypothesis $H_0$.**

Choices: (a) $H_0$ is simple; (b) $H_0$ is the most cautious or simplest explanation (no effect, no bias, no ESP). The choice is art, not math.

**Step 2: Decide if $H_A$ is one-sided or two-sided.**

- **Two-sided:** $H_A$ is that $\theta$ differs from $\theta_0$ in either direction (e.g., $\theta \neq 0.5$). Rejection region in both tails.
- **One-sided right:** $H_A$ is $\theta > \theta_0$. Rejection region in right tail only.
- **One-sided left:** $H_A$ is $\theta < \theta_0$. Rejection region in left tail only.

**Rule:** Always choose the tail(s) that are more likely under $H_A$.

**Step 3: Choose the test statistic.**

Often obvious from context. Common choices: sample mean, sample total, $z$, $t$, $\chi^2$.

**Step 4: Choose a significance level $\alpha$ and determine the rejection region.**

Typical values: $\alpha = 0.10$, $0.05$, $0.01$.

The rejection region lives in the tail(s) of the null distribution containing probability $\leq \alpha$.

**Step 5: Determine the power(s).**

Compute $P(\text{reject } H_0 \mid \theta = \theta_A)$ for specific values of $\theta_A$ in $H_A$.

### 11.2 Rejection Regions for the Coin Example ($n = 10$ flips)

**Null:** $H_0$: $\theta = 0.5$, null distribution is Binomial$(10, 0.5)$.

**Case 1: Two-sided $H_A$ ($\theta \neq 0.5$), $\alpha = 0.05$:**

We split $\alpha = 0.05$ between both tails, putting 0.025 in each. Starting from the extreme ends:

Left tail: $p(0) + p(1) = 0.001 + 0.010 = 0.011$ (too small for $\alpha/2 = 0.025$)  
Add $p(2) = 0.044$: $0.011 + 0.044 = 0.055 > 0.025$ — don't add 2 to left tail  
Wait: $p(0) + p(1) = 0.011 \leq 0.025$ ✓, but adding $p(2) = 0.044$ gives $0.055 > 0.025$.

So left tail rejection: $\{0, 1\}$ with probability $0.011$.  
By symmetry, right tail rejection: $\{9, 10\}$ with probability $0.011$.  
Total $\alpha = 0.022 \leq 0.05$ ✓.

**Rejection region for $\alpha = 0.05$ (two-sided): $\{0, 1, 9, 10\}$, actual $\alpha = 0.022$.**

| $x$ | **0** | **1** | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p(x\|H_0)$ | **0.001** | **0.010** | 0.044 | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | 0.044 | **0.010** | **0.001** |

**Case 2: Two-sided, $\alpha = 0.01$:**

Need total probability $\leq 0.01$. Only $\{0, 10\}$ fits: $0.001 + 0.001 = 0.002 \leq 0.01$.

**Rejection region for $\alpha = 0.01$ (two-sided): $\{0, 10\}$, actual $\alpha = 0.002$.**

**Case 3: One-sided right $H_A$ ($\theta > 0.5$), $\alpha = 0.05$:**

Rejection region in right tail only. Add from right: $p(10) = 0.001$, $p(9) = 0.010$: sum = 0.011. Add $p(8) = 0.044$: sum = $0.055 > 0.05$. Don't add 8.

**Rejection region for $\alpha = 0.05$ (one-sided right): $\{9, 10\}$, actual $\alpha = 0.011$.**

**Case 4: One-sided right, $\alpha = 0.01$:**

**Rejection region: $\{10\}$, actual $\alpha = 0.001$.**

### 11.3 Why Is Exact $\alpha$ Hard to Achieve?

For discrete distributions, we can't always hit the target $\alpha$ exactly. We choose the largest rejection region whose probability is $\leq \alpha$.

For continuous distributions (like the Normal), we can achieve exact $\alpha$.

---

## 12. Critical Values

### 12.1 Definition

> **Definition:** A **critical value** is the boundary value of the test statistic that separates the rejection region from the non-rejection region.
>
> Critical values are like quantiles, except they refer to the **probability in the tail to the right**, not to the left.

**Notation:** $x_q$ denotes the critical value with probability $q$ to its **right** (and therefore $1-q$ to its left). So $x_q = q_{1-q}$ in quantile notation.

**Equivalence:** $x_q$ is the $(1-q)$-quantile: $x_q = q_{1-q}$.

### 12.2 Worked Example 9 — Standard Normal Critical Value

**Problem:** Find the 0.05 critical value for $\text{N}(0,1)$.

**Solution:** The 0.05 critical value $z_{0.05}$ has 5% probability to its right and 95% to its left. It is the 0.95 quantile.

$$z_{0.05} = q_{0.95} = \texttt{qnorm(0.95, 0, 1)} = 1.645$$

### 12.3 Worked Example 10 — Right-Sided Rejection Region

**Problem:** Test statistic $x$ has null distribution $\text{N}(100, 15^2)$. Right-sided rejection region, $\alpha = 0.05$. Find the critical value.

**Solution:** We need the value $x_{0.05}$ with 5% probability to its right (rejection region) and 95% to its left.

$$x_{0.05} = q_{0.95} = \texttt{qnorm(0.95, 100, 15)} = 124.7$$

**Rejection region:** $x > 124.7$.

**Diagram:**
```
φ(x|H₀) ~ N(100, 15²)

         ___________
        /           \     ████
_______/             \___████_______
                   100  x₀.₀₅=124.7
       ←non-reject H₀→|←reject H₀→
                        α = 5% shaded
```

### 12.4 Worked Example 11 — Left-Sided Rejection Region

**Same distribution, left-sided, $\alpha = 0.05$.**

Now we need 5% probability to the **left** (rejection region). The critical value $x_{0.95}$ has 95% to its right and 5% to its left.

$$x_{0.95} = q_{0.05} = \texttt{qnorm(0.05, 100, 15)} = 75.3$$

**Rejection region:** $x < 75.3$.

### 12.5 Worked Example 12 — Two-Sided Rejection Region

**Same distribution, two-sided, $\alpha = 0.05$.**

Split: 0.025 in each tail.

- **Left critical value:** $x_{0.975} = q_{0.025} = \texttt{qnorm(0.025, 100, 15)} = 70.6$
- **Right critical value:** $x_{0.025} = q_{0.975} = \texttt{qnorm(0.975, 100, 15)} = 129.4$

**Rejection region:** $x < 70.6$ or $x > 129.4$.

**Diagram:**
```
φ(x|H₀) ~ N(100, 15²)
          ___________
████     /           \     ████
████____/             \___████___
  x₀.₉₇₅  70.6    100  129.4  x₀.₀₂₅
← reject →|← non-reject H₀ →|← reject →
   2.5%                             2.5%
```

### 12.6 Critical Values for Standard Normal

The most commonly used critical values for $\text{N}(0,1)$:

| Test type | $\alpha$ | Critical value(s) | Rejection region |
|---|---|---|---|
| Right-sided | 0.10 | $z_{0.10} = 1.282$ | $z > 1.282$ |
| Right-sided | 0.05 | $z_{0.05} = 1.645$ | $z > 1.645$ |
| Right-sided | 0.01 | $z_{0.01} = 2.326$ | $z > 2.326$ |
| Two-sided | 0.10 | $\pm z_{0.05} = \pm 1.645$ | $|z| > 1.645$ |
| Two-sided | 0.05 | $\pm z_{0.025} = \pm 1.960$ | $|z| > 1.960$ |
| Two-sided | 0.01 | $\pm z_{0.005} = \pm 2.576$ | $|z| > 2.576$ |

---

## 13. p-Values

### 13.1 Definition

> **Definition:** The **p-value** is the probability, assuming the null hypothesis is true, of observing data at least as extreme as the actual experimental data.
>
> "At least as extreme" is defined relative to the direction(s) of the rejection region.

**Formally:**
- **Right-sided:** $p = P(X \geq x_{\text{obs}} \mid H_0)$
- **Left-sided:** $p = P(X \leq x_{\text{obs}} \mid H_0)$
- **Two-sided:** $p = P(|X - \mu_0| \geq |x_{\text{obs}} - \mu_0| \mid H_0) = 2 \cdot P(X \geq |x_{\text{obs}}| \mid H_0)$ (for symmetric distributions)

### 13.2 The p-Test

> **The p-Test (Decision Rule):**
> - If $p \leq \alpha$: **Reject $H_0$**
> - If $p > \alpha$: **Do not reject $H_0$**

### 13.3 Why the p-Test Equals the Rejection Region Test

**Key geometric insight:**

For a right-sided test, the rejection region is $\{x : x > x_q\}$ where $P(X > x_q \mid H_0) = \alpha$.

If the observed statistic $x_{\text{obs}} > x_q$ (in rejection region), then:
$$p = P(X \geq x_{\text{obs}} \mid H_0) < P(X \geq x_q \mid H_0) = \alpha$$

So $x_{\text{obs}}$ is in the rejection region **if and only if** $p < \alpha$.

**The two methods are completely equivalent:**
- "$x_{\text{obs}}$ is in the rejection region" ⟺ "$p < \alpha$"

### 13.4 Graphical Illustration

**Case 1: $x_1$ in rejection region (p < α)**

```
φ(x|H₀) = null pdf
            ___
           /   \          ████████
          /     \        █xq  x₁█
─────────/       \──────█────────█───
                  xq    x₁
         ←non-reject→  |← reject →
         
α = shaded (total right tail area from xq)
p = striped (area to right of x₁)
Since x₁ > xq: p < α → REJECT H₀
```

**Case 2: $x_1$ NOT in rejection region (p > α)**

```
φ(x|H₀) = null pdf
            ___
           /   \    ████████
          /     \  █x₁      █
─────────/  x₁   \/  xq     █──────
                  xq
         ←non-reject→|← reject →
         
p = striped (area to right of x₁)
Since x₁ < xq: p > α → DO NOT REJECT H₀
```

### 13.5 Interpretation of the p-Value

> **What the p-value IS:** The probability of seeing data this extreme (or more extreme) if $H_0$ were true.
>
> **What the p-value is NOT:**
> - NOT the probability that $H_0$ is true.
> - NOT the probability that the result is due to chance.
> - NOT a measure of effect size or practical significance.

**Common misinterpretation:** "p = 0.03 means there is a 3% probability that $H_0$ is true." This is WRONG — it's a Bayesian posterior probability, which frequentists don't compute.

**Correct interpretation:** "If $H_0$ were true, the probability of observing data this extreme is 3%."

---

## 14. The z-Test for Normal Hypotheses

### 14.1 Setup

When the test statistic is a standardized normal statistic, we call it the **z-statistic** and the test is called a **z-test**.

**The z-statistic:**

$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$$

Under $H_0$ (where $\mu = \mu_0$), the z-statistic follows $\text{N}(0,1)$.

**Why standardize?** Standardizing puts everything in units of standard deviations, making it easy to compare against universal critical values (1.645, 1.96, 2.576, etc.).

### 14.2 Worked Example 13 — MIT Student IQ (One-Sided z-Test)

**Problem:** IQ is distributed as $\text{N}(100, 15^2)$ in the general population. We suspect MIT students have higher-than-average IQ. Test 9 students; their average IQ is $\bar{x} = 112$.

**Step 1: Set up hypotheses.**

$$H_0: \text{MIT students} \sim \text{N}(100, 15^2) \quad (\text{same as general population})$$
$$H_A: \text{MIT students have mean IQ} > 100 \quad (\text{one-sided right})$$

**Step 2: Identify the null distribution for $\bar{x}$.**

Under $H_0$, with $n = 9$ students:

$$\bar{X} \mid H_0 \sim \text{N}\!\left(100, \frac{15^2}{9}\right) = \text{N}\!\left(100, \frac{225}{9}\right) = \text{N}(100, 25)$$

**Step 3: Standardize to get the z-statistic.**

$$z = \frac{\bar{x} - 100}{15/\sqrt{9}} = \frac{112 - 100}{15/3} = \frac{12}{5} = 2.4$$

Under $H_0$: $z \sim \text{N}(0,1)$.

**Step 4: Compute the p-value (right-sided).**

$$p = P(Z \geq 2.4 \mid H_0) = 1 - \Phi(2.4) = 1 - \texttt{pnorm(2.4,0,1)} = 0.00820$$

**Step 5: Decision at $\alpha = 0.05$.**

Since $p = 0.0082 \leq \alpha = 0.05$: **REJECT $H_0$**.

**Conclusion:**

> "We reject the null hypothesis in favor of the alternative hypothesis that MIT students have higher IQs on average. We have done this at significance level 0.05 with a p-value of 0.008."

**Step 6: Verify using rejection region.**

The rejection region is $z > z_{0.05} = 1.645$.  
Our $z = 2.4 > 1.645$ → in rejection region → **reject $H_0$**. ✓ (Consistent with p-value approach.)

**Diagram:**
```
φ(z|H₀) ~ N(0,1)

           ___________
          /           \        ████
─────────/             \──────███──
              0          z₀.₀₅=1.645  z=2.4
         ←────── non-reject H₀ ───→|← reject →
         
α = shaded (right of 1.645) = 0.05
p = striped (right of 2.4) = 0.008
p < α → REJECT
```

---

## 15. Preview: Chi-Square and t-Tests

### 15.1 Example 14 — Chi-Square Goodness of Fit (Preview)

**Problem:** Milk spread on 400 grid squares. Count bacteria per square.

| Amount of bacteria | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Number of squares | 56 | 104 | 80 | 62 | 42 | 27 | 9 | 9 | 5 | 3 | 2 | 1 |

**Sample mean:** $\bar{x} = 2.44$.

**Hypotheses:**
$$H_0: \text{data} \sim \text{Poisson}(2.44) \qquad H_A: \text{data is not Poisson}(2.44)$$

**Test statistic (chi-square):**

$$X^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ = observed count in cell $i$ and $E_i$ = expected count under $H_0$.

**Computation table:**

| Number per square | 0 | 1 | 2 | 3 | 4 | 5 | 6 | >6 |
|---|---|---|---|---|---|---|---|---|
| Observed | 56 | 104 | 80 | 62 | 42 | 27 | 9 | 20 |
| Expected | 34.9 | 85.1 | 103.8 | 84.4 | 51.5 | 25.1 | 10.2 | 5.0 |
| $(O-E)^2/E$ | 12.8 | 4.2 | 5.5 | 6.0 | 1.7 | 0.14 | 0.15 | **44.5** |

$$X^2 = 12.8 + 4.2 + 5.5 + 6.0 + 1.7 + 0.14 + 0.15 + 44.5 = 74.99$$

**Degrees of freedom:** 8 cells, 2 constraints (total = 400 fixed, mean = 2.44 fixed) → 6 d.f.

**p-value:** $P(\chi^2_6 > 74.99) \approx 0$ (essentially zero).

**Conclusion:** Decisively reject $H_0$. The bacteria counts do NOT follow a Poisson(2.44) distribution. The large contribution from the ">6" tail ($44.5$ out of $74.99$) indicates the fit breaks down in the tail.

### 15.2 Example 15 — Student's t-Test (Preview)

**Problem:** Compare a medical treatment (extends life) vs. placebo.

$X_1, \ldots, X_n$ = years lived after treatment  
$Y_1, \ldots, Y_m$ = years lived after placebo  
$\bar{X}, \bar{Y}$ = sample means  
$\mu_X, \mu_Y$ = unknown population means

**Hypotheses:**

$$H_0: \mu_X = \mu_Y \qquad H_A: \mu_X \neq \mu_Y$$

**Test statistic:**

$$t = \frac{\bar{X} - \bar{Y}}{s_p}$$

where $s_p$ is the pooled standard error. Under $H_0$, $t$ follows a $t$-distribution with $n + m - 2$ degrees of freedom.

**Rejection region:** Determined by threshold $t_0$ with $P(t > t_0) = \alpha$.

*(Full details will be covered in later classes.)*

---

## 16. All Worked Examples — Complete Solutions

### 16.1 Board Problem 1 — Testing Coins (Binomial NHST)

**Complete problem from in-class PDF:**

**Setup:**
- Unknown probability of heads $\theta$
- Test statistic: $x$ = number of heads in 10 tosses
- $H_0$: $\theta = 0.5$ (fair coin)
- $H_A$: $\theta \neq 0.5$ (unfair coin, two-sided)

#### Part (a) — Find the significance level for rejection region $\{0, 1, 2, 8, 9, 10\}$

| $x$ | **0** | **1** | **2** | 3 | 4 | 5 | 6 | 7 | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p(x\|H_0)$ | **0.001** | **0.010** | **0.044** | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | **0.044** | **0.010** | **0.001** |

$$\alpha = P(\text{rejection region} \mid H_0) = 0.001 + 0.010 + 0.044 + 0.044 + 0.010 + 0.001 = \boxed{0.11}$$

#### Part (b) — Find a two-sided rejection region for $\alpha = 0.05$

**Strategy:** Add values from both extremes toward center until we've accumulated approximately $0.025$ in each tail (but not exceeding it for discrete distributions).

**Left tail accumulation:**
- $x = 0$: $p = 0.001$, cumulative = 0.001
- $x = 1$: $p = 0.010$, cumulative = 0.011
- $x = 2$: $p = 0.044$, cumulative = 0.055 > 0.025 ← STOP, don't include $x=2$

Left tail rejection: $\{0, 1\}$ with probability 0.011.

**Right tail accumulation (by symmetry):**
- $x = 10$: $p = 0.001$, cumulative = 0.001
- $x = 9$: $p = 0.010$, cumulative = 0.011
- $x = 8$: $p = 0.044$, cumulative = 0.055 > 0.025 ← STOP

Right tail rejection: $\{9, 10\}$ with probability 0.011.

**Rejection region for $\alpha = 0.05$: $\{0, 1, 9, 10\}$**

| $x$ | **0** | **1** | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p(x\|H_0)$ | **0.001** | **0.010** | 0.044 | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | 0.044 | **0.010** | **0.001** |

**Actual significance:** $0.001 + 0.010 + 0.010 + 0.001 = 0.022 \leq 0.05$ ✓

---

### 16.2 Board Problem 2 — z-Statistic (Normal NHST)

**Setup:**
- $H_0$: data follows $\text{N}(5, 10^2)$
- $H_A$: data follows $\text{N}(\mu, 10^2)$ where $\mu \neq 5$ (two-sided)
- Test statistic: $z$ = standardized $\bar{x}$
- Data: $n = 64$ points with $\bar{x} = 6.25$
- Significance level: $\alpha = 0.05$

#### Part (a) — Find the rejection region

Two-sided test at $\alpha = 0.05$. The null distribution of $z$ is $\text{N}(0,1)$. Put 0.025 in each tail.

$$\text{Rejection region: } |z| > 1.96$$

Critical values: $z_{0.025} = 1.96$ and $z_{0.975} = -1.96$.

#### Part (b) — Find the z-value

Under $H_0$, $\bar{X} \sim \text{N}(5, 10^2/64) = \text{N}(5, 25/4)$, so $\text{SE} = 10/8 = 1.25$.

$$z = \frac{\bar{x} - 5}{10/\sqrt{64}} = \frac{6.25 - 5}{10/8} = \frac{1.25}{1.25} = 1$$

#### Part (c) — Decision

$z = 1$. Is it in the rejection region $|z| > 1.96$? $|1| = 1 < 1.96$ → **NOT in rejection region**.

**Decision: Do not reject $H_0$.**

#### Part (d) — Compute the p-value

Two-sided p-value:

$$p = P(|Z| > 1 \mid H_0) = 2 \cdot P(Z > 1) = 2 \times 0.1587 = \mathbf{0.317}$$

Since $p = 0.317 > 0.05 = \alpha$: **Do not reject $H_0$**. ✓

#### Part (e) — Connection between (b), (c), and (d)

The $z$-value not being in the rejection region ($|z| = 1 < 1.96$) tells us exactly the same thing as the p-value being greater than the significance level ($p = 0.317 > 0.05$): **do not reject $H_0$**.

The three approaches are equivalent:
1. Check if $z$ is in the rejection region.
2. Check if $p < \alpha$.
3. Check if $z > z_{\alpha/2}$ (for two-sided test).

**Diagram:**
```
φ(z|H₀) ~ N(0,1)

           ___________
████      /           \      ████
████_____/             \_____████
-1.96      0        z=1    1.96
←reject→|←──── non-reject H₀ ──→|←reject→

α = 0.05 (shaded, split 2.5% each tail)
p = P(|Z|>1) = 0.317 (striped, larger than α)
z=1 is in non-reject region: DON'T REJECT H₀
```

---

### 16.3 Board Problem 3 — More Coins (Asymmetric NHST)

**Setup:**
- Coin $C_1$: $P(\text{heads}) = 0.5$
- Coin $C_2$: $P(\text{heads}) = 0.6$
- One coin picked at random, flipped 8 times → 6 heads
- Significance level: $\alpha = 0.05$

**Probability tables:**

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| $p(k \mid \theta=0.5)$ | 0.004 | 0.031 | 0.109 | 0.219 | 0.273 | 0.219 | 0.109 | 0.031 | 0.004 |
| $p(k \mid \theta=0.6)$ | 0.001 | 0.008 | 0.041 | 0.124 | 0.232 | 0.279 | 0.209 | 0.090 | 0.017 |

#### Part (a): $H_0 = C_1$ ($\theta = 0.5$), $H_A = C_2$ ($\theta = 0.6$). Reject at $\alpha = 0.05$?

**Step 1:** Determine direction of rejection region.

Since $H_A$ has $\theta = 0.6 > 0.5 = H_0$, the $C_2$ coin produces more heads than $C_1$. The alternative predicts higher values of $k$. → **Right-sided rejection region**.

**Step 2:** Find rejection region from right tail of $H_0$ distribution.

Add from right: $p(8) = 0.004$; $p(7) + p(8) = 0.031 + 0.004 = 0.035 \leq 0.05$ ✓.

Add $p(6) = 0.109$: $0.035 + 0.109 = 0.144 > 0.05$ ← STOP.

**Rejection region: $\{7, 8\}$, actual $\alpha = 0.035$.**

**Step 3:** Our data: $x = 6$. Is $6 \in \{7, 8\}$? **No.**

**Decision: Do not reject $H_0 = C_1$.**

#### Part (b): $H_0 = C_2$ ($\theta = 0.6$), $H_A = C_1$ ($\theta = 0.5$). Reject at $\alpha = 0.05$?

**Step 1:** Now $H_A$ has $\theta = 0.5 < 0.6 = H_0$. The $C_1$ coin produces fewer heads. → **Left-sided rejection region**.

**Step 2:** Add from left tail of $H_0 = C_2$ distribution.

$p(0) = 0.001$; $p(0) + p(1) = 0.001 + 0.008 = 0.009$; $p(0)+p(1)+p(2) = 0.009 + 0.041 = 0.050 \leq 0.05$ ✓.

Add $p(3) = 0.124$: $0.050 + 0.124 = 0.174 > 0.05$ ← STOP.

**Rejection region: $\{0, 1, 2\}$, actual $\alpha = 0.050$.**

**Step 3:** Our data: $x = 6$. Is $6 \in \{0, 1, 2\}$? **No.**

**Decision: Do not reject $H_0 = C_2$.**

#### Part (c): Is it paradoxical that we don't reject either coin?

**No — this is NOT paradoxical.** It reflects the asymmetry of NHST:

- The null hypothesis is the **cautious choice**. We only reject $H_0$ if the data is extremely unlikely assuming $H_0$.
- Getting 6 heads out of 8 is perfectly plausible under both $\theta = 0.5$ (probability 0.109) and $\theta = 0.6$ (probability 0.209).
- Neither hypothesis makes this outcome extreme, so neither is rejected.
- The data is simply **not distinctive enough** to choose between the two coins.

> **Key Insight:** NHST is not a symmetric competition between two hypotheses. It is a one-sided test of whether the data is implausible under $H_0$. The fact that both could give us the same answer in opposite tests reflects the inherent conservatism of the framework — we err on the side of not rejecting.

---

## 17. Concept Questions with Deep Explanations

### 17.1 Concept Question 1 — What Would a Frequentist Say?

**Question:** Jaam arrives $X$ hours late to class, $X \sim \text{Uniform}(0, \theta)$, $\theta$ unknown. Jon computes the prior pdf $f(\theta)$, the likelihood $\phi(x \mid \theta)$, and the posterior pdf $f(\theta \mid x)$. Which computations would a frequentist consider valid?

**Options:** None / prior / likelihood / posterior / prior+posterior / prior+likelihood / likelihood+posterior / all three.

**Answer: 3 — likelihood only.**

**Deep Explanation:**

- **Prior $f(\theta)$:** A probability distribution over the unknown parameter $\theta$. To the frequentist, $\theta$ is a fixed constant (Jaam's true lateness distribution). Assigning a probability to it is nonsensical — there's no repeatable experiment that would generate a distribution over $\theta$.

- **Likelihood $\phi(x \mid \theta)$:** This is the probability of observing $x$ hours of lateness, given that the true parameter is $\theta$. This IS valid to the frequentist — it describes the long-run frequency of $x$ if we observed Jaam many times with a fixed $\theta$. Conditioning on $\theta$ just fixes the model parameter; it doesn't require treating $\theta$ as random.

- **Posterior $f(\theta \mid x)$:** Derived by multiplying prior × likelihood and normalizing. Since the prior is invalid to the frequentist, so is the posterior.

**Verdict:** Only the likelihood is valid from a frequentist perspective.

---

### 17.2 Concept Question 2 — Is It a Statistic?

**Setup:** Data $x_1, \ldots, x_n$ from $\text{N}(\mu, \sigma^2)$, $\mu$ and $\sigma$ unknown.

| Expression | Statistic? | Reason |
|---|---|---|
| **(a)** Median of $x_1, \ldots, x_n$ | **Yes** | Computed from data alone, no unknown parameters needed |
| **(b)** $[q_{0.25}, q_{0.75}]$ of $\text{N}(\mu, \sigma^2)$ | **No** | This interval depends on $\mu$ and $\sigma$, which are unknown. No data is used at all |
| **(c)** $(\bar{x} - \mu) / (\sigma/\sqrt{n})$ | **No** | Requires knowing unknown $\mu$ and $\sigma$ |
| **(d)** $\{x_i : |x_i - \bar{x}| < 1\}$ | **Yes** | $\bar{x}$ is computed from data; all we need is the data |
| **(e)** $z = (\bar{x} - 5) / (3/\sqrt{n})$ | **Yes** | 5 and 3 are given constants (not estimated from data); computable from data |
| **(f)** $z = (\bar{x} - \mu_0) / (\sigma_0/\sqrt{n})$, $\mu_0, \sigma_0$ given | **Yes** | $\mu_0$ and $\sigma_0$ are specified/given; computable from data and known values |

**Key distinction for (b) vs. (d):**
- (b) doesn't use the data at all — it's a property of the distribution model, which depends on unknown parameters.
- (d) uses only the data — we compute $\bar{x}$ from data and then find all data points within 1 of $\bar{x}$.

---

### 17.3 Concept Question 3 — Picture the Significance

**Question:** The null and alternative pdfs are shown. The significance level is given by the area of which region?

```
φ(x|H_A)     φ(x|H_0)
  ____    R₂ R₃  ____
 /    \   ██ ██ /    \
/      \_██████/      \____
←reject H₀→|←non-reject H₀→
     R₁        R₃  R₄
```

**Answer: $R_2 + R_3$**

**Explanation:**

The significance level is $P(\text{reject } H_0 \mid H_0)$ = the probability under the **null distribution** $\phi(x \mid H_0)$ of the **rejection region**.

- $R_2$ and $R_3$ are the areas under $\phi(x \mid H_0)$ that lie in the rejection region.
- $R_1$ is under $\phi(x \mid H_A)$, not $\phi(x \mid H_0)$, and it's in the rejection region — this contributes to **power**, not significance.
- $R_4$ is under $\phi(x \mid H_0)$ but in the **non-rejection** region.

**Significance = area under $H_0$ curve in the rejection region = $R_2 + R_3$.**

---

## 18. Common Mistakes

### 18.1 Conceptual Mistakes

| Mistake | Correction |
|---|---|
| "p = 0.03 means there's a 3% probability $H_0$ is true" | The p-value is NOT the probability that $H_0$ is true. It is $P(\text{data this extreme} \mid H_0)$. |
| "Not rejecting $H_0$ means accepting $H_0$" | Failing to reject $H_0$ only means the data isn't extreme enough to reject it. You never prove $H_0$. |
| "A smaller p-value means a bigger effect" | A small p-value means the data is unlikely under $H_0$. But with large $n$, even tiny effects give small p-values. Always report effect size separately. |
| "The significance level is fixed by convention at 0.05" | $\alpha$ is chosen based on context and the consequences of Type I vs. Type II errors. |
| Thinking the frequentist can compute $P(H_0 \mid \text{data})$ | The frequentist explicitly avoids this — it requires a prior. |

### 18.2 Statistic vs. Non-Statistic Mistakes

| Mistake | Correction |
|---|---|
| Treating $(\bar{x} - \mu)/(\sigma/\sqrt{n})$ as a statistic when $\mu, \sigma$ are unknown | This is NOT a statistic; it depends on unknown parameters. To make it a statistic, you must substitute known values for $\mu$ and $\sigma$. |
| Thinking that "computed from data" automatically means it's a statistic | It must be computed from data AND known values only. No unknown parameters allowed. |

### 18.3 Rejection Region Mistakes

| Mistake | Correction |
|---|---|
| Using the wrong tail for a one-sided test | Always check: which direction does $H_A$ predict? $H_A: \theta > \theta_0$ → right tail. $H_A: \theta < \theta_0$ → left tail. |
| For two-sided test, putting all $\alpha$ in one tail | Split $\alpha/2$ in each tail for two-sided tests. |
| Exceeding target $\alpha$ for discrete distributions | For discrete data, the actual $\alpha$ may be less than the target. Choose the rejection region with the largest probability that still doesn't exceed $\alpha$. |
| Forgetting that significance level = P(rejection region | H₀) | Significance is always computed under $H_0$, never under $H_A$. |

### 18.4 Power Mistakes

| Mistake | Correction |
|---|---|
| Thinking power is 1 - significance | Power = $P(\text{reject } H_0 \mid H_A)$, which is unrelated to significance except through the tradeoff. |
| Expecting power to be the same for all alternative values | For composite $H_A$, power is a function of the specific alternative value. Power increases as the alternative moves farther from $H_0$. |

---

## 19. Quick Reference Summary

### 19.1 Bayesian vs. Frequentist

| | Bayesian | Frequentist |
|---|---|---|
| Probability | Degree of belief | Long-run frequency |
| Parameters | Can have distributions | Fixed constants |
| Valid computations on $\theta$ | Prior, likelihood, posterior | Likelihood only |
| Inference tool | Posterior distribution | NHST, confidence intervals |

### 19.2 Statistic Definition

A statistic can be computed from **data and known (given) values only**. It cannot depend on unknown parameters.

### 19.3 NHST Recipe

1. Set up $H_0$ (null, cautious, simple if possible) and $H_A$ (alternative)
2. Choose test statistic $X$ and compute its null distribution $\phi(x \mid H_0)$
3. Choose $\alpha$ and find rejection region (in the tail(s) of the null distribution)
4. Compute test statistic from data; decide reject or not
5. Optionally: compute p-value; same decision as above

### 19.4 Key Formulas

$$\alpha = P(\text{reject } H_0 \mid H_0 \text{ true}) = P(\text{Type I error})$$

$$\text{Power} = P(\text{reject } H_0 \mid H_A \text{ true}) = 1 - P(\text{Type II error})$$

$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} \sim \text{N}(0,1) \text{ under } H_0$$

$$\text{p-value} = P(\text{data at least as extreme as observed} \mid H_0)$$

$$\text{Reject } H_0 \iff p \leq \alpha \iff \text{test statistic in rejection region}$$

### 19.5 Standard Normal Critical Values

| $\alpha$ (two-sided) | Critical value $z_{\alpha/2}$ | Rejection region |
|---|---|---|
| 0.10 | 1.645 | $|z| > 1.645$ |
| **0.05** | **1.960** | $|z| > 1.960$ |
| 0.01 | 2.576 | $|z| > 2.576$ |

### 19.6 The Three Ways to Make a Decision (All Equivalent)

| Method | Reject $H_0$ if... |
|---|---|
| Rejection region | Test statistic falls in rejection region |
| p-value | $p \leq \alpha$ |
| Critical value | $|z| > z_{\alpha/2}$ (two-sided) or $z > z_\alpha$ (one-sided right) |

### 19.7 Error Types Summary

| Error | Description | Probability | Want |
|---|---|---|---|
| Type I | Reject true $H_0$ (false positive) | $\alpha$ (significance) | Small |
| Type II | Don't reject false $H_0$ (false negative) | $\beta = 1 - \text{Power}$ | Small |
| Correct rejection | Reject false $H_0$ | Power $= 1 - \beta$ | Large |
| Correct non-rejection | Don't reject true $H_0$ | $1 - \alpha$ | Large |

---

*End of MIT 18.05 Class 17 Study Notes.*  
*Source: MIT OpenCourseWare, https://ocw.mit.edu — 18.05 Introduction to Probability and Statistics, Spring 2022.*
