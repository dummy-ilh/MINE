# Chapter 6 (Boosted): T-tests & Z-tests — Fully Framed for A/B Testing

## 1. Intuition

By now you know the hypothesis testing logic (Ch 2), what a p-value means (Ch 3), and how to size an experiment (Ch 5). This chapter answers a narrower but frequently-tested question, and every part of it is answered **in the context of a live A/B test**: **which specific test statistic do you actually compute to decide if your experiment "won," and why does the choice matter?**

The core intuition: the choice between a t-test and a z-test isn't about the metric type (conversion rate vs. revenue) *per se* — it's about **whether you know the true population variance or have to estimate it from your sample, and how large your sample is.** In practice, at Google-scale traffic, this distinction mostly stops mattering numerically (t and z converge for large $n$), but interviewers still test whether you understand *why* it stops mattering, not just that it does.

**Why this shows up in every A/B test you'll ever run**: every experiment ends with a "ship / don't ship" decision, and that decision hinges on a test statistic crossing a threshold. Getting the *type* of test wrong doesn't just cost you elegance — it can flip your decision, especially at smaller sample sizes (holdout experiments, novel-metric tests, expensive-to-recruit populations like enterprise customers).

---

## 2. The A/B Testing Decision Table — Which Test, For Which Metric

This is the table to have memorized cold. In an A/B test, your metric type almost entirely determines your default test.

| Metric type in your A/B test | Example | Default test | Why |
|---|---|---|---|
| **Binary conversion metric** | click-through rate, signup rate, checkout completion | **Z-test (two-proportion)** | Variance $p(1-p)$ is fully determined by the mean $\hat p$ — nothing extra to estimate, CLT kicks in fast |
| **Continuous metric, large n per arm** | session duration, revenue per user, latency, at $n$ in the thousands+ | **T-test (Welch's)**, converges to z anyway | $\sigma$ unknown, estimated as $s$ — but at large $n$ the t-distribution ≈ normal, so this is mostly a formality |
| **Continuous metric, small n per arm** | enterprise/B2B experiment, expensive holdout, novel test population, $n < 30$–50 per arm | **T-test (Welch's), mandatory** | Fatter t-distribution tails materially change your critical value — using z here understates uncertainty and inflates false positives |
| **Before/after on the same users** | user's own spend pre- vs. post-launch | **Paired t-test** | Differencing out each user's baseline shrinks variance — much more powerful than treating pre/post as independent groups |
| **Heavily skewed continuous metric** | revenue per user with whales, long right tail | **Mann-Whitney U / bootstrap CI**, not t or z | Mean-based tests are unreliable when a handful of outliers dominate the sample mean |
| **Count metric (rare events)** | crashes per session, support tickets per user | Often **Poisson-based test** or bootstrap, not classic t/z | Variance-mean relationship differs from both proportions and continuous metrics — flag this if it comes up, out of scope for this chapter |

**The single sentence that answers 90% of "which test?" interview questions**: *"Is this metric a proportion (z) or does it have its own separately-estimated variance (t), and if t, is my per-arm sample size big enough that the distinction barely matters?"*

---

## 3. The Formal Distinction (Framed as: What Are You Testing in Your A/B Test?)

In an A/B test you're always comparing a **treatment arm mean** to a **control arm mean**, and asking: is the observed gap bigger than what pure randomization noise would produce?

**Z-test**: assumes the population standard deviation $\sigma$ is known, or the sample is large enough that the sample standard deviation $s$ is treated as if it equals $\sigma$ (via the Law of Large Numbers). This is the test you reach for when your A/B test's success metric is a **proportion** (converted / did-not-convert). Test statistic:

$$z = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}$$

**T-test**: used when $\sigma$ is unknown and estimated from the sample as $s$, introducing extra uncertainty that the normal distribution doesn't account for. This is the test you reach for when your A/B test's success metric is a **continuous measurement** — revenue, time-on-site, latency — where variance is a free parameter, not derived from the mean. This extra uncertainty is captured by using the **t-distribution** instead of the normal, which has fatter tails for small $n$ and converges to the normal distribution as $n \to \infty$ (specifically, as degrees of freedom → ∞).

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

Same formula structurally — the difference is entirely in **which distribution you compare the statistic to** (normal vs. t with $df$ degrees of freedom), because the t-distribution correctly accounts for the added uncertainty of estimating $\sigma$ from a finite sample.

### Why proportions (the most common A/B test metric) use z, not t

For a binomial proportion — your treatment-arm conversion rate $\hat p_1$ vs. control-arm conversion rate $\hat p_0$ — the variance $p(1-p)$ is *fully determined by the mean* $p$ itself. There's no separate variance parameter to estimate. So once you estimate $\hat{p}$ from your A/B test data, you've automatically got your variance estimate too, and by CLT for reasonably large $n$ (true of almost every consumer-scale A/B test), the sampling distribution of $\hat{p}$ is well-approximated by a normal. This is why every "is treatment's CTR higher than control's?" test you'll ever run defaults to z.

### Why continuous A/B test metrics use t

Revenue per user, session length, page-load latency — these have a true variance $\sigma^2$ that's a separate, independent parameter from the mean, and must be estimated separately as $s^2$ from your experiment's sample. Hence the extra "estimation uncertainty" the t-distribution accounts for. If you ran the exact same A/B test twice, your two estimates of $s^2$ would differ even with the same true underlying effect — that extra wobble is exactly what the t-distribution's fatter tails are compensating for.

---

## 4. 🧭 Flowchart: Picking Your Test Statistic for a Live A/B Test

```
                    START: You have treatment vs. control
                    data from your A/B test. Which test?
                                      │
                                      ▼
                Is the metric a PROPORTION (converted
                       yes/no, clicked yes/no)?
                                      │
                    ┌─────────────Yes─┴─No───────────────┐
                    ▼                                      ▼
          Use Z-TEST (two-proportion)              Metric is CONTINUOUS
          — variance = p(1-p), derived              (revenue, duration, latency)
          from the mean, nothing extra                       │
          to estimate                                        ▼
                                                Is the data PAIRED (same
                                                users, before/after)?
                                                               │
                                              ┌─────────Yes───┴───No──────┐
                                              ▼                            ▼
                                   Use PAIRED T-TEST              Independent groups —
                                   (differences out                is the metric heavily
                                    individual baseline,            SKEWED (whales,
                                    much more powerful)             long tail)?
                                                                            │
                                                          ┌───────────Yes───┴───No────┐
                                                          ▼                            ▼
                                          Consider MANN-WHITNEY U             Use WELCH'S T-TEST
                                          or BOOTSTRAP CI instead              (unequal-variance
                                          — mean-based tests unreliable        default; safe even
                                          with dominant outliers               if variances ARE equal)
                                                                                        │
                                                                                        ▼
                                                                    Is per-arm sample size LARGE
                                                                    (thousands+, typical consumer
                                                                    A/B test scale)?
                                                                                        │
                                                                      ┌───────────Yes───┴───No──┐
                                                                      ▼                          ▼
                                                          t-critical ≈ z-critical        t-critical is
                                                          (≈1.96 for α=0.05) —           NOTICEABLY larger
                                                          distinction is a               (fatter tails) —
                                                          formality at this scale        distinction MATTERS,
                                                                                          don't approximate with z
```

---

## 5. Paired vs. Unpaired (Independent) A/B Tests

- **Unpaired/independent t-test**: used when treatment and control are different, unrelated groups of users — **this is the standard A/B test setup** (randomized bucketing splits users into two separate groups).
- **Paired t-test**: used when you have before/after measurements on the *same* units — e.g., comparing each user's spending in the week before vs. the week after a feature launch, for the same set of users. This shows up in A/B testing as **within-subject / crossover designs**, or when you're doing a pre-period vs. post-period comparison rather than a randomized concurrent split. Paired tests are more powerful because they control for individual-level baseline differences — this is conceptually the ancestor of CUPED (Chapter 9), which generalizes the "control for individual baseline" idea to the regression-adjustment setting used in modern large-scale A/B testing platforms.

$$t_{paired} = \frac{\bar{d}}{s_d/\sqrt{n}}$$

where $\bar{d}$ is the mean of the individual differences and $s_d$ is the standard deviation of those differences — because you're differencing out each user's own baseline, the variance is often dramatically smaller than in the unpaired case, which is exactly why paired designs are more powerful when available (though true randomized A/B tests are usually unpaired by construction, since treatment and control are different user populations).

---

## 6. Welch's Correction — The A/B Testing Default

The standard two-sample t-test (Student's t-test) assumes **equal variances** between the two arms of your A/B test. This assumption is frequently violated in real experiments — e.g., if the treatment changes not just the mean but also the *spread* of the metric, which is extremely common with revenue metrics where a treatment might create more high-spenders in one arm than the other.

**Welch's t-test** relaxes the equal-variance assumption:

$$t_{Welch} = \frac{\bar{X}_1-\bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

(same numerator/structure as the general formula above), but uses the **Welch-Satterthwaite equation** to compute adjusted degrees of freedom that account for unequal variances and unequal sample sizes:

$$df \approx \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Practical rule for A/B testing platforms**: default to Welch's t-test rather than Student's t-test in production A/B testing, always. It's strictly more robust — when variances happen to be equal between your two arms, Welch's converges to the same answer as Student's; when they're not (a very live possibility, since treatment often changes variance, not just mean), Welch's protects you from an invalid test. Nearly all modern statistical software (including scipy's `ttest_ind` with `equal_var=False`) makes Welch's the trivial default choice, so there's essentially no cost to defaulting to it in your experimentation pipeline.

---

## 7. Worked Example — A Live A/B Test on Session Duration

You're comparing average session duration (a continuous A/B test metric, not a proportion) between treatment and control arms of a live experiment.

- Control arm: $n_0=500$, $\bar{X}_0 = 12.4$ min, $s_0 = 5.1$ min
- Treatment arm: $n_1=500$, $\bar{X}_1 = 13.1$ min, $s_1 = 7.8$ min (notice: variances look quite different — 5.1 vs 7.8 — a Welch's correction candidate, plausibly because treatment created more very-long-session power users)

**Step 1 — compute the difference and pooled-free SE:**

$$SE = \sqrt{\frac{5.1^2}{500}+\frac{7.8^2}{500}} = \sqrt{\frac{26.01}{500}+\frac{60.84}{500}} = \sqrt{0.0520+0.1217} = \sqrt{0.1737} \approx 0.4168$$

**Step 2 — t-statistic:**

$$t = \frac{13.1-12.4}{0.4168} = \frac{0.7}{0.4168} \approx 1.68$$

**Step 3 — degrees of freedom (Welch-Satterthwaite)**, approximately (skipping intermediate arithmetic): given the substantial variance difference, $df \approx 850$ (less than the naive $n_1+n_2-2=998$ you'd get from Student's t-test, reflecting the "penalty" for unequal variances).

**Step 4 — critical value**: for $df\approx850$, the t-distribution is already extremely close to normal, so $t_{critical}\approx1.96$ for $\alpha=0.05$ two-sided.

**Step 5 — the ship/no-ship decision**: since $1.68 < 1.96$, we fail to reject $H_0$ — **not statistically significant, so you would not ship this treatment** based on this metric alone, despite treatment showing a numerically higher mean session duration. This is exactly the kind of result that generates pressure to "just ship it, the number went up" — the test statistic is what protects you from shipping noise.

**Key teaching point**: notice that at $df=850$, the t-critical value (1.96ish) is essentially identical to the z-critical value (1.96) — this is the "t converges to z for large n" fact made concrete, and it's *why* large consumer A/B tests can often get away with treating t and z interchangeably. If this were $n=15$ per arm instead of 500 (e.g., a small enterprise-customer A/B test), the t-critical value would be closer to 2.05–2.15 (fatter tails), and using z instead of t would make you *overconfident* — you'd reject $H_0$ more easily than you should, inflating your false-positive rate on the exact experiments where sample size is hardest to grow.

---

## 8. Worked Example — A Live A/B Test on Conversion Rate (Z-test)

For contrast, here's the proportion case, which is what most consumer A/B tests actually run on as their primary success metric.

- Control arm: $n_0 = 10{,}000$ users, $120$ conversions → $\hat p_0 = 0.012$
- Treatment arm: $n_1 = 10{,}000$ users, $145$ conversions → $\hat p_1 = 0.0145$

**Step 1 — pooled proportion** (used under $H_0$: no difference between arms):

$$\hat p = \frac{120+145}{10{,}000+10{,}000} = \frac{265}{20{,}000} = 0.01325$$

**Step 2 — standard error under $H_0$:**

$$SE = \sqrt{\hat p(1-\hat p)\left(\frac{1}{n_0}+\frac{1}{n_1}\right)} = \sqrt{0.01325 \times 0.98675 \times \left(\frac{1}{10{,}000}+\frac{1}{10{,}000}\right)} \approx \sqrt{0.01307 \times 0.0002} \approx \sqrt{0.00000261} \approx 0.00162$$

**Step 3 — z-statistic:**

$$z = \frac{0.0145 - 0.012}{0.00162} = \frac{0.0025}{0.00162} \approx 1.54$$

**Step 4 — decision**: $1.54 < 1.96$, so again fail to reject $H_0$ at $\alpha=0.05$ — not significant, don't ship on this metric alone. Notice there was **no t-distribution, no degrees-of-freedom calculation, no Welch's correction anywhere in this example** — because the variance came directly from $\hat p(1-\hat p)$, with nothing separate to estimate. This is the structural reason proportions are simpler to test than continuous metrics, and why it's worth stating explicitly in an interview rather than just citing it as a rule.

---

## 9. Production Considerations for A/B Testing Pipelines

- **Default to Welch's t-test for continuous metrics** in your experimentation platform, unless you have strong reason to believe variances are equal — there's no real downside.
- **For proportions (the most common primary metric), use a z-test** (or equivalently a chi-square test for the 2×2 contingency table version, which gives an identical p-value for a 2-group comparison) — not a t-test, since variance is determined by the mean.
- **For heavily skewed metrics** (revenue per user, with many zeros and a long right tail from whales), neither a naive t-test nor z-test may be appropriate — consider a Mann-Whitney U test (non-parametric, tests for stochastic dominance rather than mean difference) or bootstrap confidence intervals, which don't rely on normality assumptions. Worth mentioning if asked about revenue metrics specifically, since revenue is one of the most common non-proportion primary metrics at any consumer or e-commerce company.
- **At Google-scale sample sizes** (often hundreds of thousands to millions of users per arm), t vs. z is essentially a non-issue numerically for your primary A/B test decision — but understanding *why* (CLT + large df convergence) is what interviewers are actually checking for, and it becomes very much *not* a non-issue the moment you're running a smaller-population experiment (B2B, enterprise, novel feature with a small eligible population).
- **Every metric in a multi-metric A/B test dashboard may need a different test** — your primary conversion metric gets a z-test, your secondary revenue metric gets a Welch's t-test, and a skewed guardrail metric might get a bootstrap CI. A mature experimentation platform picks the test per metric type automatically rather than applying one test to everything.

---

## 10. Interview Traps

- **Trap #1**: Saying "we use a t-test for small samples and a z-test for large samples" as if it's purely about sample size. It's actually about whether $\sigma$ is known/derivable from the mean (proportions) vs. estimated separately (continuous metrics) — sample size affects convergence, but isn't the defining distinction. In an A/B test, ask "what's the metric type?" before you ask "what's the sample size?"
- **Trap #2**: Using Student's t-test (equal-variance assumption) by default instead of Welch's, especially when the interviewer mentions or implies unequal variances between arms — a very common tell in revenue-metric A/B test scenarios.
- **Trap #3**: Using an independent/unpaired test when the A/B test data is actually paired (e.g., before/after on the same users in a crossover design), losing statistical power unnecessarily.
- **Trap #4**: Not recognizing when an A/B test's metric is so skewed that a mean-based test (t-test) is inappropriate altogether, and non-parametric alternatives should be considered.
- **Trap #5**: Applying a single test type to an entire multi-metric A/B test dashboard rather than matching the test to each metric's type individually.

---

## 11. L5-Differentiating Talking Points

- Explaining WHY proportions use z (variance determined by the mean) rather than reciting "small n → t, big n → z" shows real understanding of the underlying mechanics — and ties directly back to how your A/B testing platform should route different metric types to different tests.
- Defaulting to Welch's t-test and explaining why it's the safer choice, unprompted, signals production experience over textbook knowledge.
- Bringing up non-parametric alternatives (Mann-Whitney, bootstrap) for skewed revenue metrics shows you know the limits of the standard toolkit and have a next move ready — especially relevant since revenue is one of the most commonly-skewed metrics in real A/B tests.
- Connecting paired t-tests conceptually to CUPED as "controlling for individual baseline" bridges this chapter forward and shows you see the throughline in variance reduction across the curriculum, all the way to how modern experimentation platforms (like the one you'd build at Google) actually reduce required sample size in practice.
- Explicitly stating that a multi-metric A/B test dashboard needs per-metric test selection, not a single blanket test — this is a subtle but real production detail interviewers reward.

---

## 12. Comprehension Check
These are common A/B testing interview questions. Here's how to answer each one clearly and with the reasoning interviewers are looking for.

---

# 1. In a live A/B test, what determines whether you should use a t-test or z-test? Is it fundamentally about sample size?

**Short answer:**
No. The main factor is **what you're estimating and what assumptions you can make**, not simply sample size.

### The decision depends on:

1. **Type of metric**

   * Binary (conversion, click, signup)
   * Continuous (revenue, session duration)

2. **Sampling distribution of the estimator**

   * Is it approximately Normal?
   * Do we know its standard error?

3. **Whether population variance is known**

   * In practice, it almost never is.

---

### Binary metric

Example:

* Converted = Yes/No
* CTR
* Purchase rate

Each observation follows a Bernoulli distribution.

For sufficiently large samples,

[
\hat p \sim N\left(p,\frac{p(1-p)}{n}\right)
]

This naturally leads to a **z-test**.

---

### Continuous metric

Example:

* Revenue
* Time spent
* Session length

Population variance is unknown.

Estimate it using sample variance.

Because variance is estimated rather than known, use a **t-test**.

---

### Is sample size irrelevant?

Not entirely.

Large sample sizes make

* t-distribution
* Normal distribution

almost identical.

For large experiments,

* z-test
* t-test

produce nearly identical p-values.

But the conceptual reason is **not "large sample = z-test."**

---

# 2. Why do proportions typically use a z-test while revenue uses a t-test?

### Conversion Rate

Each user either

* converts (1)
* doesn't convert (0)

The sample mean equals the sample proportion.

Using the Central Limit Theorem,

[
\hat p \approx N\left(p,\frac{p(1-p)}{n}\right)
]

So compare two proportions with a **two-proportion z-test**.

---

### Revenue

Revenue can be

* 0
* $15
* $200
* $20,000

Variance is unknown.

Estimate variance from data.

That uncertainty leads to the **t-distribution**, hence a **t-test**.

---

**Rule of thumb**

| Metric           | Test   |
| ---------------- | ------ |
| Conversion       | z-test |
| CTR              | z-test |
| Open rate        | z-test |
| Revenue          | t-test |
| Session duration | t-test |

---

# 3. What problem does Welch's correction solve?

The standard Student t-test assumes

[
\sigma_1^2=\sigma_2^2
]

(equal population variances)

This assumption is often false.

Example

Control revenue variance

[
200
]

Treatment variance

[
600
]

Student's t-test pools these variances.

Pooling when variances differ produces incorrect standard errors.

That can inflate Type I error.

---

### Welch's t-test

Welch allows

[
\sigma_1^2\neq\sigma_2^2
]

It

* estimates each variance separately
* adjusts degrees of freedom
* produces more reliable confidence intervals

---

### Why default to Welch?

Because

* if variances are equal → almost no loss of power
* if variances differ → much safer

Modern statistical software defaults to Welch for this reason.

---

# 4. Why is a paired t-test more powerful?

Suppose each person has

Before

After

Instead of comparing two unrelated groups,

compute

[
D_i=After_i-Before_i
]

Now analyze only these differences.

---

### Why does this help?

Each person acts as their own control.

Natural variability between people disappears.

Example

Without pairing

| Person | Income |
| ------ | ------ |
| A      | 20k    |
| B      | 150k   |

Huge variability.

With pairing

| Person | Before | After |
| ------ | ------ | ----- |
| A      | 100    | 105   |
| B      | 250    | 255   |

Differences

5

5

Variance is dramatically lower.

Lower variance

↓

Higher statistical power.

---

### Why are standard A/B tests unpaired?

Randomization assigns users to

* Control **or**
* Treatment

Never both.

Each user contributes one observation.

Therefore observations are independent.

Hence an **unpaired test**.

---

# 5. Revenue has a huge right tail. Why can a standard t-test mislead?

Revenue distributions are often extremely skewed.

Example

990 users

$5–20

10 users

$50,000

Mean becomes dominated by whales.

---

### Problems

Sample mean becomes unstable.

Variance explodes.

Confidence intervals become huge.

One whale customer can change the result.

---

### Alternatives

Several approaches are common:

**1. Bootstrap (very common)**

Resample users many times.

Estimate confidence intervals empirically.

No strong Normality assumption.

---

**2. Log transform**

Analyze

[
\log(Revenue+1)
]

Reduces skewness.

---

**3. Mann–Whitney U test**

Nonparametric.

Tests distributions rather than means.

Useful when Normality assumptions are questionable.

---

**4. Winsorization**

Cap extremely large observations.

Very common in industry.

---

# 6. Using the flowchart, choose the correct test.

---

## (a) Checkout completion

Metric

Converted?

Binary

Independent groups

Large sample

↓

**Two-proportion z-test**

---

## (b) Customer support wait time

n = 20 per arm

Continuous

Small sample

Independent groups

Unknown variance

↓

**Welch's two-sample t-test**

If the data are extremely non-Normal with only 20 observations, consider a nonparametric alternative such as the Mann–Whitney U test.

---

## (c) Before/after spending on the same 200 users

Same users measured twice.

Dependent observations.

Continuous metric.

↓

**Paired t-test**

Compute

[
Difference = After - Before
]

Run a one-sample t-test on those differences.

---
