# MIT 18.05 — Introduction to Probability and Statistics
## Complete Study Notes: Classes 20 & 22
### Frequentist vs. Bayesian Inference · Stopping Rules · Confidence Intervals

---

> **How to use these notes:** Every concept, example, and worked problem from the uploaded PDFs is reproduced here with expanded explanations. You should never need to refer back to the original documents. Mathematical notation uses LaTeX throughout.

---

# PART I — CLASS 20: Frequentist vs. Bayesian Inference

---

## Topic 1: Two Schools of Statistical Inference

### 1. Concept Overview

Statistics is fundamentally about drawing conclusions from data. Two major schools of thought disagree on *how* to do this:

- **Bayesian inference** treats probability as a **degree of belief**, which can be assigned to hypotheses.
- **Frequentist inference** treats probability as a **long-run frequency** of outcomes in repeated experiments. Under this view, a hypothesis is either true or false — it cannot have a "probability" of being true.

Both schools use **Bayes' formula** as their mathematical foundation, but they apply it differently.

---

### 2. Intuition

Think of a courtroom:

- A **Bayesian juror** says: *"Before seeing the evidence, I think there's a 30% chance the defendant is guilty. After seeing the evidence, I update to 85%."*
- A **Frequentist juror** says: *"I cannot assign probabilities to guilt. I can only ask: if the defendant were innocent, how likely is this evidence? If it's very unlikely, I reject innocence."*

Neither approach is universally "right" — they answer different questions. The tension between them is one of the great intellectual debates in modern statistics.

---

### 3. Formal Definitions

#### Bayes' Formula (Abstract Form)

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

#### Bayes' Formula (Statistical Form)

Replacing abstract events with hypotheses $\mathcal{H}$ and data $\mathcal{D}$:

$$\boxed{P(\mathcal{H} \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \mathcal{H})\, P(\mathcal{H})}{P(\mathcal{D})}}$$

| Term | Name | Role |
|------|------|------|
| $P(\mathcal{H})$ | **Prior** | Probability of hypothesis *before* seeing data |
| $P(\mathcal{D} \mid \mathcal{H})$ | **Likelihood** | Evidence about $\mathcal{H}$ provided by the data |
| $P(\mathcal{H} \mid \mathcal{D})$ | **Posterior** | Probability of hypothesis *after* seeing data |
| $P(\mathcal{D})$ | **Total probability of data** | Normalizing constant over all hypotheses |

---

### 4. Comparison: Bayesian vs. Frequentist

| Feature | Bayesian | Frequentist (NHST) |
|---|---|---|
| Probability of a hypothesis? | ✅ Yes — the posterior | ❌ Never |
| Requires a prior? | ✅ Yes | ❌ No |
| Depends on experimental design? | ❌ No (Likelihood Principle) | ✅ Yes |
| Computational cost | Higher (integration) | Lower |
| Era of dominance | Pre-20th century + modern ML/big data | 20th century science |
| Key output | Posterior distribution | $p$-value, confidence interval |

---

### 5. Bayesian Inference — Key Properties

**Uses probabilities for both hypotheses and data.**

- Requires choosing a **prior** $P(\mathcal{H})$ — this is subjective, which is both a strength and a weakness.
- Once the prior is chosen, all inference follows from **deductive logic** (Bayes' Theorem).
- Different investigators may choose different priors and arrive at different posteriors.
- Allows natural **sequential updating**: data can be incorporated as it arrives.
- The prior's sensitivity can be checked by trying alternative priors (**sensitivity analysis**).
- Results are easy to communicate: *"There is an 89% probability the coin is biased towards heads."*

#### Defense of Bayesian Inference

1. **Directly answers the question we care about.** When a medical test is positive, the patient wants $P(\text{sick} \mid \text{positive test})$ — not $P(\text{positive test} \mid \text{not sick})$.
2. **Logically rigorous** once a prior is chosen.
3. **Robust analysis possible** by varying priors.
4. **Avoids the stopping-rule problem** — the likelihood is all that matters, not what experiments *could have* been run.
5. **Sequential data use** — no need to pre-specify when to stop collecting data.

#### Critique of Bayesian Inference

1. **Subjective prior.** Different people produce different priors, potentially leading to different conclusions from the same data.
2. **Philosophical objection.** Hypotheses are not outcomes of repeatable experiments — a coin either is or is not fair. Assigning a probability to this is philosophically contested.

---

### 6. Frequentist (NHST) Inference — Key Properties

**Never uses or gives the probability of a hypothesis.**

- Only uses conditional distributions of data **given** a specific hypothesis (the null $H_0$).
- All inference is objective: all statisticians agree on the $p$-value.
- Demands that the **entire experiment** be specified in advance.
- The **significance level** $\alpha$ controls the Type I error rate: $P(\text{reject } H_0 \mid H_0 \text{ is true}) = \alpha$.

#### Defense of Frequentist Inference

1. **Objective.** No subjectivity — the $p$-value is the same for every statistician.
2. **Forces careful experimental design** before data collection — reduces experimenter bias.
3. **Proven track record** — over 100 years of scientific progress using frequentist methods.
4. **Explicit trade-off** between Type I and Type II errors.

#### Critique of Frequentist Inference

1. **Ad-hoc.** The notion of "data more extreme" depends on the experimental design, not just the observed data.
2. **Prone to misinterpretation.** Most people incorrectly believe that $p = 0.05$ means $P(H_0) = 0.05$.
3. **Requires pre-specification.** Any deviation from the pre-planned design invalidates the analysis.

---

### 7. The Most Important Warning: Mind Your $p$-Values

> **⚠️ Critical Misconception**
> 
> A $p$-value of 0.05 does **NOT** mean the probability that $H_0$ is true is 5%.
>
> The $p$-value is $P(\text{data this extreme or more} \mid H_0 \text{ is true})$.
>
> To get $P(H_0 \mid \text{data})$, you need Bayes' theorem and a **prior**.

#### Example — Mind Your $p$'s

**Problem:** We run a two-sample $t$-test for equal means, with $\alpha = 0.05$, and obtain a $p$-value of 0.04. What are the odds the two samples are drawn from distributions with the same mean?

**(a)** 19/1 &emsp; **(b)** 1/19 &emsp; **(c)** 1/20 &emsp; **(d)** 1/24 &emsp; **(e)** Unknown

**Answer: (e) Unknown.**

**Why?** Frequentist methods only give $P(\text{statistic} \mid H_0)$. They never give $P(H_0 \mid \text{data})$. Without a prior, we cannot compute the probability that $H_0$ is true.

---

## Topic 2: Significance Tests and p-Values

### 1. Concept Overview

In Null Hypothesis Significance Testing (NHST):

1. State a null hypothesis $H_0$ and an alternative $H_A$.
2. Choose a significance level $\alpha$ (e.g., 0.05).
3. Compute a test statistic from data.
4. Compute the $p$-value: $P(\text{test statistic this extreme or more} \mid H_0)$.
5. If $p \leq \alpha$, **reject** $H_0$. Otherwise, **fail to reject**.

### 2. Concept Question — Significance Tests

**Problem:** Three tests are run, all with $\alpha = 0.05$.
- Experiment 1: $p = 0.003$ → rejects $H_0$
- Experiment 2: $p = 0.049$ → rejects $H_0$
- Experiment 3: $p = 0.15$ → fails to reject $H_0$

**Which result has the highest probability of being correct?**

**(a)** Experiment 1 &emsp; **(b)** Experiment 2 &emsp; **(c)** Experiment 3 &emsp; **(d)** Impossible to say

**Answer: (d) Impossible to say.**

**Reasoning:** The $p$-value cannot tell you the probability that a hypothesis is correct. A smaller $p$-value means the data is less likely *if* $H_0$ is true — but that is not the same as $H_0$ being false. To compute $P(\text{result correct})$, we need a prior probability for $H_0$ and $H_A$, which the frequentist framework refuses to supply.

**Intuition:** Imagine $H_0$ is almost certainly true (strong prior). Even with $p = 0.003$, it's possible we're seeing a rare but non-impactful fluctuation. Conversely, if $H_0$ is very unlikely a priori, even $p = 0.15$ might not change much.

---

## Topic 3: Multiple Testing

### 1. Concept Overview

When many hypotheses are tested simultaneously, the probability of a **false positive** (Type I error) increases dramatically, even if each individual test is run at $\alpha = 0.05$.

### 2. Formal Framework

If $k$ independent tests are each run at significance level $\alpha$, the probability of **at least one false rejection** is:

$$P(\text{at least one false rejection}) = 1 - (1-\alpha)^k$$

For $k = 15$, $\alpha = 0.05$:

$$1 - (0.95)^{15} \approx 0.537$$

So even when $H_0$ is true everywhere, there's a 54% chance of a false positive somewhere!

### 3. Concept Question — Multiple Testing

**Problem:**

**(a)** Suppose we have 6 treatments and want to know if average recovery times are equal. If we compare two at a time, how many two-sample $t$-tests are needed?

**(b)** If we use $\alpha = 0.05$ for each of those tests, what is the probability of rejecting at least one null hypothesis (assuming all nulls are true)?

#### Part (a) Solution

Choosing 2 groups from 6 to compare:

$$\binom{6}{2} = \frac{6!}{2!\,4!} = 15 \text{ tests}$$

**Answer: (iv) 15**

#### Part (b) Solution

**Answer: (iv) Greater than 0.25**

Consider just 3 independent comparisons (group1 vs. group2, group3 vs. group4, group5 vs. group6). These are independent because they use disjoint groups.

$$P(\text{at least one rejection among 3}) = 1 - (0.95)^3 \approx 1 - 0.857 = 0.143$$

The remaining 12 comparisons are *not* independent of these, but they only *increase* the chance of a false rejection. In simulation with normal data, the actual false rejection rate across all 15 tests is approximately **0.36** — far above the nominal 0.05.

> **Key Insight:** The significance level $\alpha$ controls the Type I error rate for a **single test**. When running multiple tests without correction, the **family-wise error rate** inflates dramatically. Corrections like Bonferroni ($\alpha' = \alpha/k$) or the Benjamini-Hochberg procedure are needed.

---

## Topic 4: Stopping Rules — A Critical Frequentist vs. Bayesian Divide

### 1. Concept Overview

The **stopping rule** is the rule determining when to end data collection. In frequentist inference, the stopping rule affects the $p$-value (even when the observed data is identical). In Bayesian inference, the stopping rule is irrelevant — only the likelihood of the observed data matters.

This is one of the deepest conceptual differences between the two schools.

---

### 2. The Coin-Tossing Stopping Rule Example (Class 20 Prep)

**Setup:** Jon is testing whether his coin is biased towards heads. He tosses and reports the sequence $HHHHT$ (or equivalently $HHHHT$). Jerry forgot which experiment Jon ran:

- **Experiment 1:** Toss the coin exactly 6 times and report the number of heads.
- **Experiment 2:** Toss the coin until the first tails and report the number of heads.

Let $\theta$ = probability of heads. Hypotheses:
$$H_0: \theta = 0.5 \qquad H_A: \theta > 0.5$$

Observed data: $HHHHT$ (5 heads followed by 1 tail).

---

#### Frequentist Analysis

**Experiment 1** — Null distribution: $\text{Binomial}(6, 0.5)$

"As or more extreme" means 5 or 6 heads:

$$p = P(X \geq 5 \mid \theta = 0.5) = 1 - \text{pbinom}(4, 6, 0.5) = 0.1094$$

Since $0.1094 > 0.05$, we **fail to reject** $H_0$.

**Experiment 2** — Null distribution: $\text{Geometric}(0.5)$

"As or more extreme" means 5 or more heads before the first tails:

$$p = P(X \geq 5 \mid \theta = 0.5) = 1 - \text{pgeom}(4, 0.5) = 0.0313$$

Since $0.0313 < 0.05$, we **reject** $H_0$.

> **Startling conclusion:** The *same observed data* ($HHHHT$) leads to **opposite conclusions** depending only on which stopping rule was used!

**Why the frequentist is fine with this:** The set of possible outcomes differs:
- In Experiment 1, $THHHHH$ is as extreme as $HHHHT$.
- In Experiment 2, $THHHHH$ is impossible — the experiment would have stopped after the first $T$.

---

#### Bayesian Analysis

Jerry the Bayesian doesn't care which experiment was run. The **likelihood functions** for $HHHHT$ under the binomial and geometric models are proportional to each other:

- Binomial: $P(4\text{ heads}, 1\text{ tail}) \propto \theta^5(1-\theta)^1$
- Geometric: $P(5\text{ heads then tail}) \propto \theta^5(1-\theta)^1$

They are identical (up to a constant). Therefore, the **posterior is the same** regardless of the stopping rule.

**Prior:** $\text{Beta}(3,3)$ (relatively flat, concentrated on $[0.25, 0.75]$).

**Update:** 5 heads + 1 tail → **Posterior:** $\text{Beta}(3+5,\, 3+1) = \text{Beta}(8,4)$

**Key computations in R:**
```r
# 50% posterior probability interval
qbeta(c(0.25, 0.75), 8, 4)   # [0.58, 0.76]

# 90% posterior probability interval
qbeta(c(0.05, 0.95), 8, 4)   # [0.44, 0.86]

# Posterior probability coin is biased towards heads
1 - pbeta(0.5, 8, 4)          # 0.89
```

**Conclusion (Bayesian):** Starting from $\text{Beta}(3,3)$, the posterior probability the coin is biased towards heads is **0.89** — regardless of whether Jerry is analyzing Experiment 1 or 2.

> **Intuition:** The Bayesian says "I only observed $HHHHT$. Whether the experimenter planned to toss 6 times or until tails is information about the experimenter's intentions, not about the coin. My inference about the coin shouldn't change."

---

### 3. The Likelihood Principle

> **The Likelihood Principle:** All the evidence from data is contained in its likelihood function.

- **Consistent with Bayesian inference** — only the observed column of the likelihood table matters.
- **Inconsistent with NHST** — $p$-values depend on probabilities of *unobserved* data (the full experimental design).
- **Controversial** — frequentists dispute this principle precisely because accepting it rules out $p$-values.

---

## Topic 5: Class 20 Board Problems — Stopping Rules and Validity

### Problem 1 — Stop! (Coin Bias Test)

**Setup:** Testing whether a coin is biased towards heads. Significance level $\alpha = 0.1$.

- **Experiment 1:** Toss coin 5 times. Test statistic: number of heads $X$.
- **Experiment 2:** Toss coin until first tails. Test statistic: number of heads $X$ before first tail.

---

#### Part (a) — Test Statistic, Null Distribution, Rejection Region

**Experiment 1:**

- Test statistic: $X$ = number of heads in 5 tosses
- Null distribution: $X \sim \text{Binomial}(5, 0.5)$
- Rejection region: We need the smallest set in the right tail with probability $\leq \alpha = 0.1$

$$P(X = 5) = (0.5)^5 = 0.03125 \leq 0.1$$

**Rejection region:** $\{X = 5\}$, i.e., only the sequence $HHHHH$.

**Experiment 2:**

- Test statistic: $X$ = number of heads before first tails
- Null distribution: $X \sim \text{Geometric}(0.5)$ where $P(X = k) = (0.5)^{k+1}$
- Rejection region: We need $P(X \geq c) \leq 0.1$

$$P(X \geq 4) = (0.5)^4 = 0.0625 \leq 0.1$$
$$P(X \geq 3) = (0.5)^3 = 0.125 > 0.1$$

**Rejection region:** $\{X \geq 4\}$

Sequences producing rejection: $\{HHHHT,\, HHHHH*T,\, HHHHHH*T, \ldots\}$ (4 or more heads before the first tail).

---

#### Part (b) — Data is $HHHHT$

**Step 1: Significance tests**

For **Experiment 1** with data $HHHHT$: We have $X = 4$ heads.

"As or more extreme" means $X \geq 4$:

$$p = P(X \geq 4 \mid \theta = 0.5) = P(X=4) + P(X=5) = \binom{5}{4}(0.5)^5 + (0.5)^5 = \frac{5+1}{32} = \frac{6}{32} \approx 0.1875$$

Since $0.1875 > 0.1$, **fail to reject** $H_0$ for Experiment 1.

For **Experiment 2** with data $HHHHT$: We have $X = 4$ heads before the first tail.

"As or more extreme" means $X \geq 4$:

$$p = P(X \geq 4 \mid \theta = 0.5) = (0.5)^4 = 0.0625$$

In R: `1 - pgeom(3, 0.5) = 0.0625`

Since $0.0625 < 0.1$, **reject** $H_0$ for Experiment 2.

**Step 2: Bayesian update from flat prior $\text{Beta}(1,1)$**

Data: 4 heads, 1 tail. Using the conjugate Beta-Binomial pair:

$$\text{Prior: } \text{Beta}(1,1) \xrightarrow{\text{4 heads, 1 tail}} \text{Posterior: } \text{Beta}(1+4,\, 1+1) = \text{Beta}(5, 2)$$

**The posterior is the same for both experiments** (same likelihood).

$$P(\theta > 0.5 \mid \text{data}) = 1 - \text{pbeta}(0.5, 5, 2) = 0.89$$

**Conclusion:** After updating from the flat prior, there is approximately an **89% probability** the coin is biased towards heads — a fairly strong signal of bias, regardless of which experiment was run.

---

### Problem 2 — Stop! (Experimental Validity)

**Setup:** Three experimenters all use $\alpha = 0.05$.

#### Experiment 1 (Alessandre)
- By design: 50 trials, computed $p = 0.04$.
- Reports: $p = 0.04$, $n = 50$, declares significant.

**(a) Comment on validity:** ✅ **Valid.** This is a properly designed NHST experiment. The number of trials was fixed in advance, the test was performed once, and the result is reported honestly.

**(b) True probability of Type I error:** $= 0.05$ (by design of the test).

---

#### Experiment 2 (Sara)
- Did 50 trials: $p = 0.06$ (not significant).
- Did 50 more trials: $p = 0.04$ based on all 100 trials.
- Reports: $p = 0.04$, $n = 100$, declares significant.

**The actual experiment run:**
1. Do 50 trials.
2. If $p < 0.05$, stop and publish.
3. If not, run another 50 trials.
4. Compute $p$ again as if all 100 trials were planned from the start.

**(a) Comment on validity:** ❌ **Invalid.** The second $p$-value is computed using the **wrong null distribution**. The experiment that was actually run is a sequential procedure, but the $p$-value computation assumes a fixed-$n$ design. This violates the assumption underlying the null distribution used.

**(b) True probability of Type I error:**

If $H_0$ is true, there's already a 5% chance of false rejection at step 2. Running steps 3–4 only adds more opportunities for false rejection. So the true Type I error rate is **strictly greater than 0.05** (exact amount requires complex calculation).

---

#### Experiment 3 (Gabriel)
- Did 50 trials: $p = 0.06$ (not significant).
- Started over with a new 50 trials: $p = 0.04$.
- Reports: $p = 0.04$, $n = 50$, declares significant.

**(a) Comment on validity:** ❌ **Invalid.** Gabriel is running two separate experiments but only reporting the one that "worked." This is **$p$-hacking** — the reported result has a higher Type I error rate than $\alpha = 0.05$ because of the selection process.

**(b) True probability of Type I error:**

Use a probability tree (all probabilities computed assuming $H_0$ is true):

```
                        [.05] Reject (1st batch)
                       /
First 50 trials ──────
                       \
                        [.95] Continue
                              /
              Second 50 trials
                              \
                               [.05] Reject    [.95] Don't reject
```

$$P(\text{False rejection}) = 0.05 + 0.95 \times 0.05 = 0.05 + 0.0475 = 0.0975$$

**The true Type I error rate is approximately 9.75%, nearly double the nominal 5%.**

> **Key Takeaway:** Sequential testing without proper correction inflates Type I error. This is pervasive in published research and contributes to the **replication crisis**.

---

### Problem 3 — Chi-Square Test for Independence

**Data:**

| Education | Married once | Married multiple times | Total |
|-----------|-------------|----------------------|-------|
| College | 550 | 61 | 611 |
| No college | 681 | 144 | 825 |
| **Total** | **1231** | **205** | **1436** |

**Test:** $H_0$: Number of marriages and education level are independent. Use $\alpha = 0.01$.

#### Step-by-Step Solution

**Step 1: Estimate marginal probabilities under $H_0$**

$$P(\text{College}) = \frac{611}{1436} \approx 0.4256$$

$$P(\text{No college}) = \frac{825}{1436} \approx 0.5744$$

$$P(\text{Married once}) = \frac{1231}{1436} \approx 0.8572$$

$$P(\text{Married multiple}) = \frac{205}{1436} \approx 0.1428$$

**Step 2: Compute expected cell probabilities (under independence)**

$$p_{ij} = P(\text{row } i) \times P(\text{col } j)$$

| Education | Married once | Married multiple times |
|---|---|---|
| College | $0.4256 \times 0.8572 = 0.3648$ | $0.4256 \times 0.1428 = 0.0608$ |
| No college | $0.5744 \times 0.8572 = 0.4924$ | $0.5744 \times 0.1428 = 0.0820$ |

**Step 3: Compute expected counts** (multiply probabilities by total $n = 1436$)

| Education | Married once | Married multiple |
|---|---|---|
| College | $0.3648 \times 1436 = \mathbf{523.8}$ | $0.0608 \times 1436 = \mathbf{87.2}$ |
| No college | $0.4924 \times 1436 = \mathbf{707.2}$ | $0.0820 \times 1436 = \mathbf{117.8}$ |

**Observed vs. Expected table:**

| Cell | Observed $O$ | Expected $E$ |
|---|---|---|
| College, once | 550 | 523.8 |
| College, multiple | 61 | 87.2 |
| No college, once | 681 | 707.2 |
| No college, multiple | 144 | 117.8 |

**Step 4: Compute chi-square statistic**

$$X^2 = \sum \frac{(O - E)^2}{E}$$

$$= \frac{(550-523.8)^2}{523.8} + \frac{(61-87.2)^2}{87.2} + \frac{(681-707.2)^2}{707.2} + \frac{(144-117.8)^2}{117.8}$$

$$= \frac{(26.2)^2}{523.8} + \frac{(-26.2)^2}{87.2} + \frac{(-26.2)^2}{707.2} + \frac{(26.2)^2}{117.8}$$

$$= \frac{686.44}{523.8} + \frac{686.44}{87.2} + \frac{686.44}{707.2} + \frac{686.44}{117.8}$$

$$\approx 1.31 + 7.87 + 0.97 + 5.83 = 16.01$$

(The G-statistic version gives $G = 16.55$, very similar.)

**Step 5: Degrees of freedom**

$$df = (r-1)(c-1) = (2-1)(2-1) = 1$$

**Step 6: Compute p-value**

```r
1 - pchisq(16.01, 1)   # ≈ 0.000063
```

**Step 7: Conclusion**

Since $p \approx 0.000063 \ll 0.01 = \alpha$, we **reject $H_0$**.

**Conclusion:** There is very strong evidence that number of marriages and education level are **not independent** in this population.

**Practical interpretation:** Educated individuals are less likely to have been married multiple times — the association is statistically significant at the 1% level.

---

## Topic 6: Type I Errors and Base Rates

### 1. Concept Overview

The significance level $\alpha$ controls the **conditional** Type I error rate:

$$\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true})$$

But in practice, we want to know:

$$P(H_0 \text{ is true} \mid \text{rejected } H_0)$$

This requires **Bayes' theorem** and a **base rate** (prior probability that $H_0$ is true).

---

### 2. Discussion Examples

#### Example A: Jerry (All Treatments Ineffective)

Jerry designs treatments that are **never effective**. He uses $\alpha = 0.05$.

**(a) What percentage of his experiments result in publications?**

Since all his treatments are ineffective, $H_0$ is *always* true. The significance level guarantees:

$$P(\text{reject} \mid H_0) = 0.05$$

So approximately **5%** of his experiments will be published.

**(b) What percentage of published papers contain Type I errors?**

Since *every* treatment is ineffective, *every* rejection is a false positive.

$$P(H_0 \mid \text{rejected}) = 1 \quad \Rightarrow \quad \mathbf{100\%}$$

of his published papers contain Type I errors.

---

#### Example B: Jen (All Treatments Effective)

Jen designs treatments that are **always effective**. She uses $\alpha = 0.05$.

**(a) What percentage of experiments are published?**

This depends on the **power** of her tests ($P(\text{reject} \mid H_A \text{ true})$). Power depends on effect size, sample size, and $\alpha$:
- If treatments are only slightly more effective than placebo → power ≈ 5% → few papers published.
- If treatments are dramatically effective → power approaches 100% → almost all experiments published.

**(b) What percentage of published papers contain Type I errors?**

Since all her treatments are effective, $H_0$ is *never* true when she tests:

$$P(H_0 \mid \text{rejected}) = 0 \quad \Rightarrow \quad \mathbf{0\%}$$

None of her published papers contain Type I errors.

---

#### Example C: A Journal Publishing Significant Results

**Question:** If a journal publishes only results with $p < 0.05$, what percentage contain Type I errors?

**Answer:** Impossible to say without knowing the **base rate** of true null hypotheses.

$$P(\text{Type I error} \mid \text{published}) = P(H_0 \mid \text{rejected}) = \frac{P(\text{reject} \mid H_0) \cdot P(H_0)}{P(\text{reject})}$$

This requires $P(H_0)$ — the prior probability that a randomly chosen tested hypothesis is false. Without this, the percentage could be **anywhere from 0% to 100%**.

> **Key Insight:** Significance is $P(\text{reject} \mid H_0)$. To know how often a published result is wrong, you need the **base rate** $P(H_0)$.

---

## Topic 7: Making Decisions Under Uncertainty

### 1. Utility Functions

In statistical decision theory, actions are evaluated by a **utility function** $U$, which assigns a numeric value to each possible outcome.

**Example:** For a stock investment with gain $d$ dollars per share:
- Risk-neutral utility: $U(d) = d$
- Loss-averse utility: $U(d) = \begin{cases} -d^2 & d < 0 \\ d & d \geq 0 \end{cases}$

### 2. Decision Rules

| Framework | Decision Based On |
|---|---|
| Frequentist | $E[U \mid \mathcal{H}]$ combined with $p$-values |
| Bayesian | $E[U \mid \mathcal{H}]$ combined with the posterior |

**Key theorem:** For any decision rule, there exists a Bayesian decision rule that is at least as good (in a well-defined sense).

---

# PART II — CLASS 22: Confidence Intervals Based on Normal Data

---

## Topic 8: Interval Statistics

### 1. Concept Overview

A **statistic** is any function of the data that does not depend on unknown parameters. A **point statistic** gives a single number; an **interval statistic** gives an interval $[L, U]$ where both endpoints are computed purely from data.

### 2. Formal Definition

An **interval statistic** is a pair of point statistics $(L, U)$ such that:
- $L$ and $U$ are both computable from data alone
- Neither $L$ nor $U$ depends on unknown parameters

### 3. Examples

**Assume:** $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$, $\mu$ unknown.

| Expression | Is it a statistic? | Why |
|---|---|---|
| $[\bar{x} - 2.2,\, \bar{x} + 2.2]$ | ✅ Yes | Depends only on $\bar{x}$, which is computable |
| $\left[\bar{x} - \frac{2\sigma}{\sqrt{n}},\, \bar{x} + \frac{2\sigma}{\sqrt{n}}\right]$ | ✅ Yes, *if $\sigma$ is known* | $\sigma$ is a known value |
| $\left[\bar{x} - \frac{2\sigma}{\sqrt{n}},\, \bar{x} + \frac{2\sigma}{\sqrt{n}}\right]$ | ❌ No, *if $\sigma$ is unknown* | Contains the unknown parameter $\sigma$ |
| $\left[\bar{x} - \frac{2s}{\sqrt{n}},\, \bar{x} + \frac{2s}{\sqrt{n}}\right]$ | ✅ Yes | $s^2$ is the sample variance, computable from data |

### 4. Important Properties of Interval Statistics

1. **The interval is random** — new data generates a new interval.
2. **Frequentist-friendly** — doesn't depend on unknown parameters.
3. **Probabilities require a hypothesis** — to compute $P(\mu_0 \in [L, U])$, we must assume a value of $\mu$.

> **⚠️ Warning (Repeated Throughout MIT 18.05):**
> The confidence level is **NEVER** a probability that the unknown parameter is in the specific computed interval. The parameter is fixed — it either is or is not in the interval. The confidence level is a property of the *procedure*, not the specific outcome.

---

## Topic 9: z-Confidence Intervals for the Mean

### 1. Concept Overview

A **confidence interval** is an interval statistic designed so that, under repeated sampling, it contains the true parameter value $(1-\alpha)$ fraction of the time.

### 2. Setup

**Assume:** $x_1, x_2, \ldots, x_n \sim N(\mu, \sigma^2)$

- $\mu$ unknown
- $\sigma^2$ **known**

### 3. Formal Definition

> **Definition — z-Confidence Interval for the Mean**
>
> The $(1-\alpha)$ confidence interval for $\mu$ is:
>
> $$\left[\bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$
>
> where $z_{\alpha/2}$ is the **right critical value** satisfying $P(Z > z_{\alpha/2}) = \alpha/2$ for $Z \sim N(0,1)$.

### 4. Key Formulas

**Critical values** (memorize or look up):

| Confidence Level $(1-\alpha)$ | $\alpha$ | $\alpha/2$ | $z_{\alpha/2}$ |
|---|---|---|---|
| 90% | 0.10 | 0.05 | 1.645 |
| 95% | 0.05 | 0.025 | 1.96 |
| 99% | 0.01 | 0.005 | 2.576 |

**Margin of error:**

$$ME = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**Width of confidence interval:**

$$\text{Width} = 2 \cdot z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

---

### 5. Derivation — From Non-Rejection Regions to Confidence Intervals

The confidence interval is derived by **pivoting** the NHST non-rejection region.

**Step 1:** Under $H_0: \mu = \mu_0$, the standardized mean follows:

$$Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1)$$

**Step 2:** The non-rejection region for $Z$ at significance $\alpha$ is:

$$|Z| \leq z_{\alpha/2} \quad \Leftrightarrow \quad -z_{\alpha/2} \leq Z \leq z_{\alpha/2}$$

**Step 3:** Writing in terms of $\bar{x}$ (the non-rejection region for $\bar{x}$):

$$\mu_0 - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \leq \bar{x} \leq \mu_0 + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

So: $P\!\left(\bar{x} \in \left[\mu_0 \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right] \;\middle|\; H_0\right) = 1 - \alpha$

**Step 4 — Pivoting:** The statement "$\bar{x}$ is in the interval $[\mu_0 \pm c]$" is equivalent to "$\mu_0$ is in the interval $[\bar{x} \pm c]$" (since the interval width is symmetric).

$$\bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \leq \mu_0 \leq \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

This is the **confidence interval**, centered on $\bar{x}$ instead of $\mu_0$.

**Resulting probability statement:**

$$P\!\left(\mu_0 \in \left[\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right] \;\middle|\; H_0\right) = 1 - \alpha$$

> **Intuition for Pivoting:** If I stand at position $\mu_0$ and you're at position $\bar{x}$, the statement "you are within distance $c$ of me" is the same as "I am within distance $c$ of you." The two intervals (centered on $\mu_0$ and $\bar{x}$ respectively) always have the same width and always include/exclude each other's centers together.

---

### 6. Key Relationship: Confidence Intervals ↔ NHST

> **Fundamental Equivalence:**
>
> $\mu_0$ is in the $(1-\alpha)$ confidence interval $\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$
>
> $\Longleftrightarrow$
>
> We **do not reject** $H_0: \mu = \mu_0$ at significance level $\alpha$

This means confidence intervals and hypothesis tests give identical conclusions:
- If $\mu_0$ is *inside* the CI → fail to reject $H_0$
- If $\mu_0$ is *outside* the CI → reject $H_0$

---

### 7. Algebraic Pivoting — A Key Tool

**Pivoting** is the algebraic maneuver of reversing the roles of two quantities in an inequality.

**Lemma:** If $a$ is in the interval $[b-4, b+6]$, then $b$ is in the interval $[a-6, a+4]$.

**Proof:**

$$b - 4 \leq a \leq b + 6$$
$$\Rightarrow \quad -4 \leq a - b \leq 6$$
$$\Rightarrow \quad 4 \geq b - a \geq -6$$
$$\Rightarrow \quad a + 4 \geq b \geq a - 6 \qquad \square$$

**Symmetric case:** If $a \in [b-c, b+c]$, then $b \in [a-c, a+c]$ — used constantly in confidence interval derivations.

**Numerical examples:**
- $1.5 \in [0 - 2.3,\, 0 + 2.3]$, so $0 \in [1.5 - 2.3,\, 1.5 + 2.3] = [-0.8, 3.8]$. ✓
- $1.5 \notin [0 - 1,\, 0 + 1]$, so $0 \notin [1.5 - 1,\, 1.5 + 1] = [0.5, 2.5]$. ✓

---

### 8. How Width Changes

The width of the confidence interval is $2 z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$.

| Variable Changed | Effect on Width | Why |
|---|---|---|
| Increase $n$ | **Narrows** | More data → smaller variance of $\bar{x}$ |
| Increase confidence $(1-\alpha)$ | **Widens** | Higher confidence requires a larger net |
| Change $\mu$ | **No change** | $\mu$ doesn't appear in the width formula |
| Increase $\sigma$ | **Widens** | More underlying variability → more uncertainty |

---

### 9. Worked Examples — z-Confidence Intervals

#### Example 1 (From Problem Set)

**Data:** $4, 1, 2, 3$ drawn from $N(\mu, \sigma^2)$, $\sigma = 2$ known.

**Task:** Find a 90% z-confidence interval for $\mu$.

**Step 1: Compute sample statistics**
$$\bar{x} = \frac{4+1+2+3}{4} = \frac{10}{4} = 2.5$$
$$\frac{\sigma}{\sqrt{n}} = \frac{2}{\sqrt{4}} = \frac{2}{2} = 1$$

**Step 2: Find critical value**

For 90% CI: $\alpha = 0.10$, $\alpha/2 = 0.05$

$$z_{0.05} = \text{qnorm}(0.95) \approx 1.645$$

**Step 3: Construct interval**

$$\left[\bar{x} - z_{0.05} \cdot \frac{\sigma}{\sqrt{n}},\; \bar{x} + z_{0.05} \cdot \frac{\sigma}{\sqrt{n}}\right] = \left[2.5 - 1.645 \cdot 1,\; 2.5 + 1.645 \cdot 1\right]$$

$$\boxed{= [0.855,\; 4.145]}$$

**Interpretation:** If we repeated this experiment many times and computed a 90% CI each time, approximately 90% of those intervals would contain the true $\mu$.

---

#### Example 2 — Computational (From Class 22 Prep)

**Data:** $2.5, 5.5, 8.5, 11.5$ drawn from $N(\mu, 10^2)$, $\mu$ unknown, $\sigma = 10$ known.

**Step 1: Compute $\bar{x}$**

$$\bar{x} = \frac{2.5+5.5+8.5+11.5}{4} = \frac{28}{4} = 7.0$$

**Step 2: Standard error**

$$\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}} = \frac{10}{\sqrt{4}} = 5$$

**Step 3: Critical values**

$$z_{0.025} = \text{qnorm}(0.975) = 1.96 \quad (\text{for 95\% CI})$$
$$z_{0.10} = \text{qnorm}(0.90) = 1.28 \quad (\text{for 80\% CI})$$
$$z_{0.25} = \text{qnorm}(0.75) = 0.67 \quad (\text{for 50\% CI})$$

**Step 4: Confidence intervals**

$$95\%\text{ CI} = [7 - 1.96 \times 5,\; 7 + 1.96 \times 5] = [-2.8,\; 16.8]$$
$$80\%\text{ CI} = [7 - 1.28 \times 5,\; 7 + 1.28 \times 5] = [0.6,\; 13.4]$$
$$50\%\text{ CI} = [7 - 0.67 \times 5,\; 7 + 0.67 \times 5] = [3.65,\; 10.35]$$

**Notice:** Higher confidence → wider interval. Always.

**Part (b) — Testing $H_0: \mu = 1$**

*Method 1: Check if $\mu_0 = 1$ is in the confidence interval:*
- $95\%$ CI: $1 \in [-2.8, 16.8]$ → ✅ Fail to reject at $\alpha = 0.05$
- $80\%$ CI: $1 \in [0.6, 13.4]$ → ✅ Fail to reject at $\alpha = 0.20$
- $50\%$ CI: $1 \notin [3.65, 10.35]$ → ❌ Reject at $\alpha = 0.50$

*Method 2: Compute rejection regions centered on $\mu_0 = 1$:*

$$\alpha = 0.05: \quad (-\infty, 1 - 1.96 \times 5] \cup [1 + 1.96 \times 5, \infty) = (-\infty, -8.8] \cup [10.8, \infty)$$

Since $\bar{x} = 7 \not\in (-\infty, -8.8] \cup [10.8, \infty)$, **fail to reject**.

$$\alpha = 0.20: \quad (-\infty, 1 - 1.28 \times 5] \cup [1 + 1.28 \times 5, \infty) = (-\infty, -5.4] \cup [7.4, \infty)$$

Since $\bar{x} = 7 \not\in (-\infty, -5.4] \cup [7.4, \infty)$, **fail to reject**.

$$\alpha = 0.50: \quad (-\infty, 1 - 0.67 \times 5] \cup [1 + 0.67 \times 5, \infty) = (-\infty, -2.35] \cup [4.35, \infty)$$

Since $\bar{x} = 7 \in [4.35, \infty)$, **reject** $H_0$.

Both methods give identical conclusions. ✓

---

## Topic 10: t-Confidence Intervals for the Mean

### 1. Concept Overview

When $\sigma$ is **unknown**, we estimate it with the **sample standard deviation** $s$ and use the $t$-distribution instead of $N(0,1)$.

### 2. Key Substitutions from z to t

| z-interval | t-interval |
|---|---|
| $\sigma$ known | $\sigma$ unknown, estimated by $s$ |
| Use $\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$ | Use $s_{\bar{x}} = \frac{s}{\sqrt{n}}$ |
| Use $z_{\alpha/2}$ critical values | Use $t_{\alpha/2}$ critical values |
| $Z \sim N(0,1)$ | $T \sim t(n-1)$ |

### 3. Formal Definition

> **Definition — t-Confidence Interval for the Mean**
>
> Suppose $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$ with both $\mu$ and $\sigma$ **unknown**.
>
> The $(1-\alpha)$ confidence interval for $\mu$ is:
>
> $$\left[\bar{x} - t_{\alpha/2} \cdot \frac{s}{\sqrt{n}},\; \bar{x} + t_{\alpha/2} \cdot \frac{s}{\sqrt{n}}\right]$$
>
> where $t_{\alpha/2}$ satisfies $P(T > t_{\alpha/2}) = \alpha/2$ for $T \sim t(n-1)$, and $s^2$ is the sample variance.

### 4. Sample Variance Reminder

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

The $n-1$ in the denominator makes $s^2$ an **unbiased estimator** of $\sigma^2$.

### 5. Construction

Under $H_0: \mu = \mu_0$, the **studentized mean** follows:

$$T = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \sim t(n-1)$$

Non-rejection region: $|T| \leq t_{\alpha/2}$

Rewriting in terms of $\bar{x}$:

$$|\bar{x} - \mu_0| \leq t_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

Pivoting: $\mu_0$ is in $\left[\bar{x} \pm t_{\alpha/2} \cdot \frac{s}{\sqrt{n}}\right]$

This is the $(1-\alpha)$ t-confidence interval.

---

### 6. Worked Example — t-Confidence Interval

#### Example (From Problem Set — Part b)

**Data:** $4, 1, 2, 3$ drawn from $N(\mu, \sigma^2)$, both $\mu$ and $\sigma$ unknown.

**Task:** Find a 90% t-confidence interval for $\mu$.

**Step 1: Compute sample statistics**

$$\bar{x} = 2.5, \quad n = 4$$

$$s^2 = \frac{(4-2.5)^2 + (1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2}{4-1} = \frac{2.25 + 2.25 + 0.25 + 0.25}{3} = \frac{5}{3} \approx 1.667$$

$$s \approx 1.291, \quad \frac{s}{\sqrt{n}} = \frac{1.291}{2} \approx 0.645$$

**Step 2: Find t-critical value**

$df = n - 1 = 3$. For 90% CI: $\alpha = 0.10$, $\alpha/2 = 0.05$

$$t_{0.05}(3) = \text{qt}(0.95, 3) \approx 2.353$$

**Step 3: Construct interval**

$$\left[2.5 - 2.353 \times 0.645,\; 2.5 + 2.353 \times 0.645\right] = \left[2.5 - 1.518,\; 2.5 + 1.518\right]$$

$$\boxed{= [0.982,\; 4.018]}$$

**Compare to z-interval:** The t-interval $[0.982, 4.018]$ is **wider** than the z-interval $[0.855, 4.145]$.

> **Why wider?** Because the t-distribution has heavier tails than $N(0,1)$ (reflecting additional uncertainty from estimating $\sigma$). With small $n$, this difference is substantial. As $n \to \infty$, the t-distribution approaches $N(0,1)$ and the two intervals converge.

---

#### Example 2 — t-Confidence Interval (From Class 22 Prep)

**Data:** $2.5, 5.5, 8.5, 11.5$ from $N(\mu, \sigma^2)$, both unknown.

**Step 1:** $\bar{x} = 7.0$, $n = 4$

$$s^2 = \frac{(2.5-7)^2 + (5.5-7)^2 + (8.5-7)^2 + (11.5-7)^2}{3} = \frac{20.25 + 2.25 + 2.25 + 20.25}{3} = \frac{45}{3} = 15$$

$$s = \sqrt{15} \approx 3.873, \quad \frac{s}{\sqrt{n}} = \frac{3.873}{2} \approx 1.936$$

**Step 2: Critical values** ($df = 3$)

$$t_{0.025}(3) = \text{qt}(0.975, 3) \approx 3.182 \quad (\text{for 95\% CI})$$
$$t_{0.10}(3) = \text{qt}(0.90, 3) \approx 1.638 \quad (\text{for 80\% CI})$$
$$t_{0.25}(3) = \text{qt}(0.75, 3) \approx 0.765 \quad (\text{for 50\% CI})$$

**Step 3: Confidence intervals**

$$95\%\text{ CI} = [7 - 3.182 \times 1.936,\; 7 + 3.182 \times 1.936] \approx [0.84,\; 13.16]$$
$$80\%\text{ CI} = [7 - 1.638 \times 1.936,\; 7 + 1.638 \times 1.936] \approx [3.83,\; 10.17]$$
$$50\%\text{ CI} = [7 - 0.765 \times 1.936,\; 7 + 0.765 \times 1.936] \approx [5.52,\; 8.48]$$

---

## Topic 11: Chi-Square ($\chi^2$) Confidence Intervals for Variance

### 1. Concept Overview

To estimate the **variance** $\sigma^2$ (not just the mean), we use the chi-square distribution. The key fact is:

$$\frac{(n-1)s^2}{\sigma^2} \sim \chi^2(n-1)$$

This allows us to build a confidence interval for $\sigma^2$ by pivoting around this statistic.

### 2. Formal Definition

> **Definition — $\chi^2$ Confidence Interval for Variance**
>
> Suppose $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$ with both $\mu$ and $\sigma$ unknown.
>
> The $(1-\alpha)$ confidence interval for $\sigma^2$ is:
>
> $$\left[\frac{(n-1)s^2}{c_{\alpha/2}},\; \frac{(n-1)s^2}{c_{1-\alpha/2}}\right]$$
>
> where $c_{\alpha/2}$ is the right critical value: $P\!\left(\chi^2 > c_{\alpha/2}\right) = \alpha/2$.

### 3. Key Points About Chi-Square CIs

- **Asymmetric:** Unlike z and t intervals, this interval is **not** symmetric around $s^2$, because the $\chi^2$ distribution is not symmetric around its mean.
- The interval for $\sigma$ is found by taking the **square root** of both endpoints.
- **Note on notation:** $c_{1-\alpha/2}$ is the *left* critical value (a small number), and $c_{\alpha/2}$ is the *right* critical value (a larger number). This is why the denominators are swapped to ensure the interval has positive width.

### 4. Derivation

Under $H_0: \sigma^2 = \sigma_0^2$:

$$\frac{(n-1)s^2}{\sigma_0^2} \sim \chi^2(n-1)$$

Non-rejection region at significance $\alpha$:

$$c_{1-\alpha/2} \leq \frac{(n-1)s^2}{\sigma_0^2} \leq c_{\alpha/2}$$

Rearranging (pivoting) to isolate $\sigma_0^2$:

$$\frac{(n-1)s^2}{c_{\alpha/2}} \leq \sigma_0^2 \leq \frac{(n-1)s^2}{c_{1-\alpha/2}}$$

This is the confidence interval for $\sigma^2$.

---

### 5. Worked Example — $\chi^2$ Confidence Intervals

#### Example (From Problem Set — Parts c & d)

**Data:** $4, 1, 2, 3$ drawn from $N(\mu, \sigma^2)$, both unknown.

**Task:** Find 90% $\chi^2$ confidence intervals for $\sigma^2$ and $\sigma$.

**Step 1: Sample statistics**

$$n = 4, \quad s^2 = 1.667, \quad df = n-1 = 3$$

**Step 2: Critical values** for $\chi^2(3)$

For 90% CI: $\alpha = 0.10$, $\alpha/2 = 0.05$

$$c_{0.05}(3) = \text{qchisq}(0.95, 3) \approx 7.815 \quad \text{(right critical value, larger)}$$

$$c_{0.95}(3) = \text{qchisq}(0.05, 3) \approx 0.352 \quad \text{(left critical value, smaller)}$$

**Step 3: 90% CI for $\sigma^2$**

$$\left[\frac{(n-1)s^2}{c_{0.05}},\; \frac{(n-1)s^2}{c_{0.95}}\right] = \left[\frac{3 \times 1.667}{7.815},\; \frac{3 \times 1.667}{0.352}\right]$$

$$= \left[\frac{5.001}{7.815},\; \frac{5.001}{0.352}\right] = \left[0.640,\; 14.208\right]$$

$$\boxed{\text{90\% CI for } \sigma^2 = [0.640,\; 14.211]}$$

**Step 4: 90% CI for $\sigma$**

Take square roots of both endpoints:

$$\boxed{\text{90\% CI for } \sigma = [\sqrt{0.640},\; \sqrt{14.211}] \approx [0.800,\; 3.770]}$$

**Interpretation:** We are 90% confident the true standard deviation $\sigma$ lies between 0.80 and 3.77. This is a very wide interval — because $n = 4$ is tiny. Variance estimation requires much larger samples than mean estimation for comparable precision.

---

## Topic 12: Critical Values — Concept Question

### z Critical Values

For $Z \sim N(0,1)$, there are two common notations:

- $q_\alpha$: the **left** critical value, $P(Z \leq q_\alpha) = \alpha$
- $z_\alpha$: the **right** critical value, $P(Z > z_\alpha) = \alpha$

Note: $q_\alpha = -z_{1-\alpha}$, and $q_{1-\alpha} = z_\alpha$.

#### Concept Question 1

**Find $z_{0.025}$.**

By definition: $P(Z > z_{0.025}) = 0.025 \Rightarrow P(Z \leq z_{0.025}) = 0.975$

$$z_{0.025} = \text{qnorm}(0.975) = \mathbf{1.96}$$

**Answer: (d) 1.96**

#### Concept Question 2

**Find $-z_{0.16}$.**

We know $P(|Z| < 1) \approx 0.68$, so $P(Z > 1) \approx 0.16$, meaning $z_{0.16} \approx 1$.

$$-z_{0.16} \approx -1 \approx -0.99$$

**Answer: (b) -0.99**

---

## Topic 13: Polling — Confidence Intervals in Practice

### 1. Setup

For a poll estimating the proportion $\theta$ of people supporting candidate X:

- Let $\hat{\theta} = \bar{x}$ be the sample proportion.
- The $(1-\alpha)$ confidence interval for $\theta$ is:

$$\left[\bar{x} - \frac{z_{\alpha/2}}{2\sqrt{n}},\; \bar{x} + \frac{z_{\alpha/2}}{2\sqrt{n}}\right]$$

> **Why the $\frac{1}{2\sqrt{n}}$ form?** When $\theta$ is unknown, the worst-case standard error of $\hat{\theta}$ is $\sqrt{\theta(1-\theta)/n} \leq \frac{1}{2\sqrt{n}}$ (maximized at $\theta = 0.5$). This is the conservative maximum-margin formula.

**Margin of error:**

$$ME = \frac{z_{\alpha/2}}{2\sqrt{n}}$$

---

### 2. Worked Examples — Polling

#### Problem 3 (From Class 22 Problem Set)

**Part (a):** How many people to poll for margin of error 0.01 with 95% confidence?

For 95% CI: $z_{0.025} = 1.96 \approx 2$

$$ME = \frac{z_{\alpha/2}}{2\sqrt{n}} = 0.01$$

$$\frac{2}{2\sqrt{n}} = 0.01 \quad \Rightarrow \quad \frac{1}{\sqrt{n}} = 0.01 \quad \Rightarrow \quad \sqrt{n} = 100 \quad \Rightarrow \quad \boxed{n = 10{,}000}$$

**Part (b):** How many people for margin of error 0.01 with 80% confidence?

For 80% CI: $\alpha = 0.20$, $z_{0.10} = \text{qnorm}(0.90) = 1.2816$

$$\frac{1.2816}{2\sqrt{n}} = 0.01 \quad \Rightarrow \quad \sqrt{n} = \frac{1.2816}{0.02} = 64.08 \quad \Rightarrow \quad n = 64.08^2 \approx \boxed{4106}$$

**Note:** Lower confidence requires fewer people — trading certainty for sample size.

**Part (c):** If $n = 900$, compute the 95% and 80% confidence intervals for $\theta$.

**95% CI:**

$$ME_{95} = \frac{z_{0.025}}{2\sqrt{900}} = \frac{1.96}{2 \times 30} \approx \frac{2}{60} = \frac{1}{30} \approx 0.0333$$

$$\text{95\% CI: } \bar{x} \pm 0.0333$$

**80% CI:**

$$ME_{80} = \frac{z_{0.10}}{2\sqrt{900}} = \frac{1.2816}{60} \approx 0.0214$$

$$\text{80\% CI: } \bar{x} \pm 0.021$$

**Interpretation:** With $n = 900$ voters, the 95% confidence margin is $\pm 3.3$ percentage points; the 80% margin is $\pm 2.1$ percentage points. Real polls typically use $n \approx 1000-1500$.

---

## Topic 14: Confidence Intervals and Non-Rejection Regions — Equivalence

### Worked Example (From Problem Set — Problem 2)

**Setup:** $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$ with $\sigma$ known.

**Task:** Show that $\mu_0$ is in the $(1-\alpha)$ confidence interval $\Leftrightarrow$ $\bar{x}$ is in the non-rejection region for $H_0: \mu = \mu_0$.

**Solution:**

**Confidence interval** (centered on $\bar{x}$):
$$CI = \left[\bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$

**Non-rejection region** (centered on $\mu_0$):
$$NR = \left[\mu_0 - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; \mu_0 + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}\right]$$

Both intervals have the **same half-width** $c = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$.

Therefore:
$$\mu_0 \in CI \iff |\mu_0 - \bar{x}| \leq c \iff |\bar{x} - \mu_0| \leq c \iff \bar{x} \in NR \qquad \square$$

**Geometric intuition:** Two intervals of equal width contain each other's center simultaneously, or neither contains the other's center. They "see" each other symmetrically.

---

## Topic 15: Rule-of-Thumb Confidence Interval

For large $n$ (say $n \geq 30$), the $t$-distribution is close to $N(0,1)$, and we use the approximation:

$$t_{\alpha/2}(n-1) \approx z_{\alpha/2}$$

For 95% CI with large $n$: $z_{0.025} \approx 2$, giving the rule of thumb:

$$\bar{x} \pm 2 \cdot \frac{s}{\sqrt{n}}$$

#### Worked Example

**Given:** $n = 100$, $\bar{x} = 12$, $s = 5$.

**95% rule-of-thumb CI:**

$$12 \pm 2 \cdot \frac{5}{\sqrt{100}} = 12 \pm 2 \cdot \frac{5}{10} = 12 \pm 1 = [11, 13]$$

**Check:** The exact value $t_{0.025}(99) \approx 1.984 \approx 2$, confirming the approximation is excellent.

---

## Topic 16: Common Mistakes

### Mistakes in Hypothesis Testing / p-Values

| Mistake | Correct Understanding |
|---|---|
| "$p = 0.05$ means $P(H_0) = 0.05$" | The $p$-value is $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$ |
| "A smaller $p$-value means $H_0$ is more likely false" | Only if we know the prior — small $p$ just means rare data under $H_0$ |
| "Failing to reject $H_0$ proves it is true" | We simply didn't find enough evidence against it |
| "Rejecting $H_0$ at $\alpha = 0.05$ means a 5% chance we're wrong" | We'd need the prior $P(H_0)$ to compute that |
| "The same stopping rule doesn't matter for p-values" | It does — different stopping rules → different null distributions → different p-values |
| "Running more tests at $\alpha = 0.05$ still keeps family error at 5%" | False — multiple testing inflates the overall error rate |

### Mistakes in Confidence Intervals

| Mistake | Correct Understanding |
|---|---|
| "There is a 95% probability $\mu$ is in this specific interval" | The specific interval either contains $\mu$ or it doesn't — no probability to speak of |
| "95% of data falls in the confidence interval" | No — 95% of *intervals constructed by this procedure* would contain $\mu$ |
| "$\sigma$ unknown → use z-interval" | Wrong — use t-interval when $\sigma$ is unknown |
| "A wider interval is always better" | Wider = more confident but less precise — there's always a tradeoff |
| "A 95% CI tells us more than a $p$-value" | They are exactly equivalent for two-sided tests at $\alpha = 0.05$ |
| "The chi-square CI for $\sigma^2$ is symmetric around $s^2$" | No — the $\chi^2$ distribution is asymmetric, so the CI is asymmetric |

---

## Topic 17: Quick Reference — All Confidence Interval Formulas

### z-Confidence Interval ($\sigma$ known)

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

### t-Confidence Interval ($\sigma$ unknown)

$$\bar{x} \pm t_{\alpha/2}(n-1) \cdot \frac{s}{\sqrt{n}}$$

### $\chi^2$ Confidence Interval for $\sigma^2$

$$\left[\frac{(n-1)s^2}{c_{\alpha/2}},\; \frac{(n-1)s^2}{c_{1-\alpha/2}}\right]$$

### $\chi^2$ Confidence Interval for $\sigma$

$$\left[\sqrt{\frac{(n-1)s^2}{c_{\alpha/2}}},\; \sqrt{\frac{(n-1)s^2}{c_{1-\alpha/2}}}\right]$$

### Polling Confidence Interval (proportion $\theta$, conservative)

$$\bar{x} \pm \frac{z_{\alpha/2}}{2\sqrt{n}}$$

### Sample Size for Given Margin of Error (polling)

$$n = \left(\frac{z_{\alpha/2}}{2 \cdot ME}\right)^2$$

---

## Topic 18: Quick Summary — Everything in Bullet Points

### Part I: Frequentist vs. Bayesian

- **Bayesian** uses probabilities for hypotheses; requires a prior; gives the posterior $P(H \mid \text{data})$.
- **Frequentist** never assigns probabilities to hypotheses; uses only $P(\text{data} \mid H)$; gives $p$-values and CIs.
- **Bayes' formula** underlies both: $P(H \mid D) = P(D \mid H) P(H) / P(D)$.
- **$p$-value** = $P(\text{data this extreme or more} \mid H_0)$. It is NOT the probability $H_0$ is true.
- **Multiple testing** inflates Type I errors; $k$ independent tests at $\alpha$ gives family-wise error $1-(1-\alpha)^k$.
- **Stopping rules** affect $p$-values but NOT Bayesian posteriors (Likelihood Principle).
- **Type I error rate** = $\alpha$ for a properly designed single test. Sequential testing without correction inflates this.

### Part II: Confidence Intervals

- **Confidence interval**: an interval statistic designed so that $(1-\alpha)$ of such intervals (across repeated experiments) contain the true parameter.
- The **confidence level is not** the probability that the specific observed interval contains $\mu$.
- **z-CI**: use when $\sigma$ known. Width $= 2z_{\alpha/2}\sigma/\sqrt{n}$.
- **t-CI**: use when $\sigma$ unknown. Uses $s$ and $t(n-1)$ critical values. Wider than z-CI for small $n$.
- **$\chi^2$-CI**: for variance $\sigma^2$; asymmetric because $\chi^2$ distribution is skewed.
- **Pivoting**: the key algebraic technique connecting non-rejection regions and confidence intervals.
- **Equivalence**: $\mu_0 \in$ CI $\Leftrightarrow$ fail to reject $H_0: \mu = \mu_0$ at significance $\alpha$.
- **Width increases** with higher confidence, higher $\sigma$; decreases with larger $n$; unchanged by $\mu$.
- **Rule of thumb** for 95% CI with large $n$: $\bar{x} \pm 2s/\sqrt{n}$.
- **Polling**: $n = (z_{\alpha/2}/(2 \cdot ME))^2$ people needed for margin of error $ME$.

---

## Appendix: R Code Reference

```r
# Normal distribution
pnorm(x)              # P(Z <= x), Z ~ N(0,1)
qnorm(p)              # z such that P(Z <= z) = p
1 - pnorm(x)          # P(Z > x)

# t-distribution
qt(p, df)             # t such that P(T <= t) = p, T ~ t(df)
pt(x, df)             # P(T <= x)

# Chi-square distribution
qchisq(p, df)         # chi-sq such that P(X^2 <= c) = p
pchisq(x, df)         # P(X^2 <= x)
1 - pchisq(x, df)     # p-value for chi-square test

# Beta distribution (Bayesian updating)
pbeta(x, a, b)        # P(X <= x) for X ~ Beta(a, b)
qbeta(p, a, b)        # x such that P(X <= x) = p

# Key critical values
qnorm(0.975)          # 1.96  (95% CI z-critical)
qnorm(0.95)           # 1.645 (90% CI z-critical)
qnorm(0.90)           # 1.282 (80% CI z-critical)
qt(0.975, df=3)       # 3.182 (95% t-critical, df=3)
qchisq(0.95, df=3)    # 7.815 (chi-sq right critical, df=3)
qchisq(0.05, df=3)    # 0.352 (chi-sq left critical, df=3)
```

---

*End of MIT 18.05 Study Notes — Classes 20 & 22.*

*Source: MIT OpenCourseWare, 18.05 Introduction to Probability and Statistics, Spring 2022. Jeremy Orloff and Jonathan Bloom.*

*These notes are intended as a comprehensive study reference and expand upon the original course materials.*
