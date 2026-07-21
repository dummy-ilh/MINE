# Chapter 6.1: Beyond T and Z — The Rest of the A/B Testing Toolkit

## 1. Intuition

Chapter 6 covered the two workhorses — z for proportions, t for continuous metrics — and that pair alone answers most "which test?" questions you'll get. But real A/B tests routinely throw metrics at you that break the assumptions both tests rely on: skewed revenue distributions, rare-event counts, more than two variants, non-normal data, or situations where you can't even assume the standard formulas apply. This chapter is the "what else is in the toolbox" reference — the tests that come up when an interviewer follows "great, and what if the metric doesn't look like that?"

**The organizing question for this whole chapter**: what specific assumption of the t-test or z-test is broken by this metric, and what's the smallest, most targeted fix? Interviewers reward this framing far more than a memorized list of test names, because it shows you're diagnosing the failure mode rather than pattern-matching to a term you once heard.

---

## 2. 🧭 Master Decision Table

| Situation that breaks t/z | Test to reach for | Core idea |
|---|---|---|
| Metric is a proportion, but comparing **more than 2 groups/variants** | **Chi-square test of independence** | Extends the 2-proportion z-test logic to an $r \times c$ contingency table |
| Proportion test, but **very small sample / rare event** (expected cell counts < 5) | **Fisher's exact test** | Computes exact hypergeometric probability instead of relying on a normal/chi-square approximation |
| Continuous metric, but **heavily skewed** (revenue, whales, long tail) | **Mann-Whitney U test** (a.k.a. Wilcoxon rank-sum) | Tests whether one distribution stochastically dominates the other, using ranks instead of means — immune to outlier-driven mean distortion |
| Continuous or skewed metric, want a **CI without distributional assumptions** | **Bootstrap resampling** | Empirically builds the sampling distribution of the statistic by resampling your actual data with replacement, thousands of times |
| **Count / rare-event metric** (crashes, tickets, errors per user) | **Poisson test / negative binomial test** | Variance-mean relationship differs from both proportions and continuous metrics — Poisson assumes variance = mean, negative binomial relaxes that for "overdispersed" counts |
| **More than 2 variants** on a continuous metric simultaneously | **ANOVA (F-test)** | Tests whether *any* group mean differs, before you do pairwise comparisons — controls the overall false-positive rate across multiple arms |
| Many pairwise comparisons after ANOVA, or **many metrics/segments** tested at once | **Multiple-comparison correction** (Bonferroni, Benjamini-Hochberg/FDR) | Without correction, testing 20 metrics at $\alpha=0.05$ gives you a near-certain false positive somewhere by chance alone |
| Want to test **any statistic** (median, ratio, percentile) with no formula available | **Permutation test** | Shuffles treatment/control labels repeatedly to build a null distribution empirically — works for literally any test statistic |
| Distribution **shape** itself might differ, not just the mean | **Kolmogorov-Smirnov (K-S) test** | Tests whether two samples come from the same distribution entirely, not just whether their means differ |
| Want to **peek at results repeatedly** without inflating false-positive rate | **Sequential testing** (e.g., mSPRT, always-valid p-values) | Standard t/z p-values are only valid at a single pre-committed sample size — repeated peeking needs a different framework entirely |

---

## 3. Chi-Square Test of Independence — More Than 2 Variants, Still a Proportion

**When it shows up**: you're not running a simple A/B test, you're running an A/B/C/D test — say, 4 different onboarding flows — and your metric is still binary (converted / did not convert).

**Core idea**: build an $r \times c$ contingency table (rows = variant, columns = converted/not-converted), and test whether conversion rate is independent of variant.

$$\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in each cell and $E_i$ is the expected count under the null hypothesis of independence (computed from the row/column marginal totals).

**Relationship to the z-test**: for exactly 2 groups, a chi-square test of independence and a two-proportion z-test give you the **identical p-value** — the chi-square statistic is literally $z^2$ in that special case. Chi-square is the generalization you reach for the moment you have 3+ variants and a binary metric.

**Worked mini-example**: testing 3 checkout button colors (A, B, C) against conversion.

| Variant | Converted | Not Converted | Total |
|---|---|---|---|
| A | 120 | 4880 | 5000 |
| B | 145 | 4855 | 5000 |
| C | 110 | 4890 | 5000 |

Expected count under $H_0$ (same conversion rate everywhere) for each "Converted" cell: overall rate $= \frac{375}{15{,}000}=0.025$, so $E = 0.025 \times 5000 = 125$ per variant. Compute $\sum (O-E)^2/E$ across all 6 cells, compare to $\chi^2$ distribution with $df = (r-1)(c-1) = (3-1)(2-1)=2$. A large statistic relative to that distribution signals at least one variant differs — but **chi-square alone doesn't tell you which one**; you'd follow up with pairwise z-tests (with a multiple-comparison correction, Section 6) to localize the effect.

---

## 4. Fisher's Exact Test — When Your Numbers Are Too Small to Trust the Approximation

**When it shows up**: an A/B test on a rare event (e.g., "account gets flagged for fraud") with a small eligible population — expected cell counts under 5 in your contingency table. The chi-square and z-test both rely on a **normal approximation to the binomial**, and that approximation gets unreliable exactly when counts are small.

**Core idea**: instead of approximating, Fisher's exact test computes the *exact* probability of observing a table at least as extreme as yours, using the hypergeometric distribution — no approximation involved, so it stays valid even at very small $n$.

**Practical rule**: if any expected cell count in your 2×2 table is below ~5, prefer Fisher's exact over a z-test or chi-square test. Above that threshold, they converge and z/chi-square is computationally cheaper (Fisher's exact gets combinatorially expensive at large $n$, which is exactly when you don't need it anymore).

---

## 5. Mann-Whitney U Test — When the Mean Isn't the Right Question

**When it shows up**: revenue-per-user metrics, where 1% of users (whales) can be 100x the median user's spend. The mean is dominated by a handful of extreme values, and a t-test on the mean can be wildly noisy or misleading — a single new whale landing in one arm by chance can flip your result.

**Core idea**: rank *all* observations (treatment + control combined) from smallest to largest, then compare the **sum of ranks** between the two groups instead of comparing means directly. This tests whether one distribution is *stochastically greater* than the other (values in one group tend to rank higher), not whether the specific numeric means differ.

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2}$$

where $R_1$ is the sum of ranks in group 1. Under $H_0$ (identical distributions), $U$ has a known sampling distribution (normal approximation for larger $n$), letting you compute a p-value the same way as any other test.

**What you gain**: complete immunity to the exact magnitude of outliers — a whale spending $10,000 or $100,000 contributes the same *rank* information, so one extreme point can't single-handedly swing your result. **What you give up**: you're testing "which distribution tends to produce larger values," not "which arm has the higher mean" — if the business question is specifically about total revenue (where whales *should* count fully), Mann-Whitney answers a subtly different question than what you actually care about, and a bootstrap CI on the mean (Section 6) might be more appropriate despite the added variance.

---

## 6. Bootstrap Resampling — When You Don't Trust Any Formula

**When it shows up**: you want a confidence interval on a statistic that doesn't have a clean closed-form formula (a ratio metric, a percentile, a complex derived metric), or you're uneasy about normality assumptions entirely and want an empirical answer.

**Core idea**: resample your actual observed data, with replacement, thousands of times (typically 1,000–10,000 iterations), recompute your statistic of interest (mean difference, ratio, median, whatever) on each resample, and use the resulting distribution of statistics directly as your sampling distribution — no formula, no distributional assumption, just brute-force simulation of "what would happen if I re-ran this experiment on slightly different samples from the same underlying population."

**Procedure**:
1. From your $n_1$ treatment observations, draw a new sample of size $n_1$ *with replacement*. Do the same for control ($n_0$).
2. Compute your statistic (e.g., mean difference) on this resampled pair.
3. Repeat steps 1–2 thousands of times.
4. The 2.5th and 97.5th percentiles of the resulting distribution of statistics form your 95% confidence interval directly.

**Why it matters in production A/B testing**: bootstrap doesn't care whether your metric is a mean, a ratio (e.g., revenue per session, where both numerator and denominator are random), a percentile (p95 latency), or something totally custom — it works identically for all of them, at the cost of being more computationally expensive than a closed-form t/z test, which matters when you're running this across thousands of metrics per experiment on an internal platform.

---

## 7. Poisson & Negative Binomial Tests — Count / Rare-Event Metrics

**When it shows up**: crashes per user-session, support tickets per customer, error events per day — these are **counts of rare events over some exposure window**, not proportions (there's no natural "denominator" of possible successes the way conversion has "did/didn't convert") and not a smoothly continuous measurement either.

**Core idea**: the Poisson distribution models count data where the variance equals the mean ($\text{Var}(X) = \lambda = E[X]$) — a much more restrictive relationship than the continuous-metric case (where variance is a fully free parameter) or the proportion case (where variance is $p(1-p)$). Testing whether treatment changes the rate $\lambda$ uses this specific variance-mean relationship rather than either the z-test's or t-test's assumptions.

**The overdispersion problem**: real-world count data (crash counts, support tickets) is often **overdispersed** — actual variance exceeds the mean, frequently because some users are just inherently crash-prone or ticket-prone (heterogeneity the simple Poisson model doesn't capture). When this happens, a plain Poisson test understates your true uncertainty and can produce false positives. The fix is a **negative binomial test**, which adds an extra dispersion parameter to relax the strict variance = mean constraint.

**Interview-ready flag**: if asked about a "crashes per user" or "tickets per user" A/B test metric, immediately noting that this is count data with its own variance-mean relationship — and checking for overdispersion before defaulting to Poisson — is a stronger answer than reflexively reaching for a t-test on the raw counts.

---

## 8. ANOVA — Testing More Than 2 Variants on a Continuous Metric

**When it shows up**: an A/B/C/D/E test on a continuous metric (e.g., 5 different pricing tiers, measuring average revenue per user across all 5).

**Core idea**: before running $\binom{5}{2}=10$ separate pairwise t-tests (which would badly inflate your false-positive rate — Section 9), first ask a single global question: **is there any difference among the group means at all?** ANOVA's F-test answers exactly that, comparing between-group variance to within-group variance:

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

A large $F$ means the groups differ more from each other than individual observations differ within a group — evidence that at least one variant is different. Only if the overall F-test is significant do you proceed to pairwise comparisons (with a correction) to find out *which* variant(s) differ — this two-step "omnibus test, then pairwise" structure is the standard, disciplined way to handle a multi-arm continuous-metric test, directly analogous to how chi-square handles the multi-arm proportion case.

---

## 9. Multiple Comparisons — The Silent Killer of Multi-Metric/Multi-Arm Tests

**Why this belongs in this chapter**: it's not a new test statistic, it's a correction that applies *on top of* every test above, and skipping it is one of the most common ways a technically-correct-looking A/B test analysis is actually wrong.

**The core problem**: at $\alpha=0.05$, you accept a 5% false-positive rate *per test*. If your experimentation dashboard reports 20 metrics (or you're comparing 5 variants pairwise, or you're slicing one metric across 10 user segments), the probability that **at least one** shows a "significant" result purely by chance is:

$$P(\text{at least one false positive}) = 1-(1-0.05)^{20} \approx 0.64$$

— a coin flip you'd be very wrong to interpret as "we found something."

**Fixes**:
- **Bonferroni correction**: divide $\alpha$ by the number of tests ($\alpha/m$), the simplest and most conservative fix — controls the probability of *any* false positive (family-wise error rate), at the cost of reduced power, especially as $m$ grows large.
- **Benjamini-Hochberg (FDR control)**: a less conservative alternative that controls the *expected proportion* of false positives among your significant results, rather than the probability of even one — generally preferred in exploratory multi-metric dashboards where Bonferroni would be overly punishing.

**Production reality**: any A/B testing platform reporting dozens of metrics per experiment needs a stated multiple-comparison policy (usually: one pre-registered primary metric tested at the nominal $\alpha$, with all secondary/guardrail metrics either FDR-corrected or explicitly labeled as "directional, not confirmatory").

---

## 10. Permutation Tests — The General-Purpose Fallback

**When it shows up**: you want to test a statistic that has no known sampling distribution at all, or you want a method that makes literally zero distributional assumptions, or you simply want to sanity-check a parametric result non-parametrically.

**Core idea**: under $H_0$, treatment/control labels are meaningless — a treatment user's outcome would've looked the same if they'd been labeled control. So, shuffle the treatment/control labels randomly (keeping the data itself fixed) thousands of times, recompute your test statistic on each shuffle, and see where your *actual* observed statistic falls in that shuffled null distribution. If your real result is more extreme than 95% of the shuffled versions, you have your p-value directly, empirically, with no formula.

**Relationship to bootstrap**: bootstrap resamples *within* each arm (with replacement) to estimate a confidence interval; permutation testing reshuffles *labels across* arms (without replacement) to build a null distribution for a p-value. Both are simulation-based and distribution-free, but they answer slightly different questions — worth being precise about this distinction if asked, since conflating them is a common but avoidable interview slip.

---

## 11. Kolmogorov-Smirnov Test — When the Whole Distribution Might Have Shifted

**When it shows up**: you suspect treatment changed more than just the average — maybe it changed the *shape* of the distribution (e.g., made outcomes more bimodal, or changed variance without changing the mean at all, which a t-test would completely miss since it only compares means).

**Core idea**: the K-S test compares the two samples' empirical cumulative distribution functions (CDFs) directly, and finds the maximum vertical distance between them:

$$D = \max_x |F_1(x) - F_2(x)|$$

A large $D$ signals the two distributions differ *somewhere* — in location, spread, or shape — without committing to any specific summary statistic like the mean. This makes it a good complement to a t-test (which could show "no significant mean difference") when you suspect treatment redistributed outcomes rather than simply shifting them.

---

## 12. Sequential Testing — When You Can't Wait for a Fixed Sample Size

**When it shows up**: stakeholders want to "peek" at results daily rather than waiting for the pre-committed sample size from your power analysis (Chapter 5) — a extremely common real-world pressure that a naive t/z-test analysis handles badly.

**The core problem**: standard p-values from a t-test or z-test are only statistically valid if you look **once**, at the sample size you committed to in advance. Checking daily and stopping "as soon as it's significant" is a form of multiple comparisons in disguise (Section 9) — it inflates your false-positive rate dramatically, since you're effectively giving yourself many chances to hit $p<0.05$ by chance across all the days you peeked.

**The fix**: sequential testing frameworks (e.g., mixture sequential probability ratio tests / mSPRT, or "always-valid p-values") are specifically designed so that the significance threshold remains valid **no matter how many times or when you look**, by building the repeated-peeking risk directly into the math rather than pretending it isn't happening. This is largely out of scope for a from-scratch derivation at L5, but knowing the *name* and the *problem it solves* — and recognizing "can we just check it every day?" as the trigger for needing it — is a strong, easy-to-produce signal.

---

## 13. Production Considerations

- **Match the test to the failure mode, not to a memorized list** — every test in this chapter exists because a specific t/z assumption breaks (skew, rare events, multiple arms, count data, distribution-shape changes, repeated peeking). Naming the broken assumption is the actual interview signal, not naming the test.
- **A mature experimentation platform routes metrics to tests automatically**: proportions → z/chi-square, continuous → Welch's t (or Mann-Whitney if flagged as skewed), counts → Poisson/negative binomial, multi-arm → ANOVA/chi-square with a stated multiple-comparison policy, and any ad-hoc/custom metric → bootstrap or permutation as a safe universal fallback.
- **Multiple-comparison correction should be a platform-level policy, not an ad hoc decision per experiment** — otherwise different teams apply (or skip) it inconsistently, and the org's "significant result" rate becomes unreliable.
- **Sequential testing is a organizational, not just statistical, fix** — if your company's culture wants to peek at experiments daily, you either build a sequential-testing framework or you accept an inflated false-positive rate; there's no way to have both daily peeking and naive t/z p-values be simultaneously valid.

---

## 14. Interview Traps

1. **Reaching for a t-test on obviously skewed revenue data without flagging the skew** — mentioning Mann-Whitney or bootstrap unprompted is a strong signal; needing to be asked is a weaker one.
2. **Running many pairwise t-tests across a multi-arm experiment with no multiple-comparison correction and no omnibus test first** — this is the single most common practical mistake in real multi-arm testing.
3. **Confusing bootstrap and permutation testing** — bootstrap builds a CI by resampling within groups; permutation builds a null distribution by reshuffling labels across groups. They solve related but distinct problems.
4. **Not recognizing "can we check the dashboard every day?" as a sequential-testing trigger** — treating it as a harmless UX question rather than a statistical validity problem.
5. **Applying a chi-square/Fisher's-exact distinction incorrectly** — defaulting to chi-square even when expected cell counts are tiny, where Fisher's exact is the more defensible choice.
6. **Treating count metrics (crashes, tickets) as either a plain proportion or a plain continuous metric** — missing that they have their own variance-mean structure (and possible overdispersion) entirely.

---

## 15. L5-Differentiating Talking Points

- Framing every test in this chapter as "which assumption of the standard t/z test does this situation break?" rather than reciting a list — this is the single highest-leverage habit for this entire topic.
- Proactively distinguishing Mann-Whitney (answers "which distribution tends higher") from a mean-based test (answers "which arm has higher average") as **different questions**, not just different tools for the same question — and picking the one that matches what the business actually cares about.
- Bringing up multiple-comparison correction unprompted the moment a scenario involves 3+ variants or many metrics, rather than waiting to be asked "but what about false positives?"
- Naming sequential testing as the fix for "stakeholders want to peek daily," and being able to state *why* naive peeking breaks p-value validity, even without deriving the mSPRT math.
- Correctly identifying overdispersion as the reason a naive Poisson test can fail on real count data, and negative binomial as the fix — a detail that separates candidates who've only seen count data in a textbook from those who've dealt with it in production.

---

## 16. Comprehension Check

1. For each of the following A/B test scenarios, name the test you'd reach for and the specific t/z assumption it fixes: (a) a 5-variant pricing test on revenue, (b) a rare fraud-flag proportion test with small $n$, (c) a crashes-per-session metric, (d) a request for a CI on p95 latency.
2. Why does chi-square give the identical p-value to a two-proportion z-test when there are exactly 2 groups, and why does that stop being true with 3+ groups?
3. Explain the difference between what a Mann-Whitney U test and a t-test are each actually testing for — why might they disagree on the same dataset?
4. Walk through why testing 20 metrics at $\alpha = 0.05$ each gives you roughly a 64% chance of at least one false positive, and describe two different ways to correct for it.
5. What specifically goes wrong, statistically, if a team checks their A/B test dashboard every day and stops the test the moment a metric crosses $p<0.05$? What class of method fixes this?
6. Your count metric (support tickets per user) shows variance noticeably larger than its mean. What does that tell you, and how does it change which test you use?
7. Describe, in your own words, the mechanical difference between how a bootstrap resample is constructed and how a permutation-test resample is constructed.

---
*This chapter extends Chapter 6 (T-tests & Z-tests) with the rest of the standard A/B testing statistical toolkit: chi-square and Fisher's exact for multi-arm/rare-event proportions, Mann-Whitney U and bootstrap for skewed continuous metrics, Poisson/negative binomial for count data, ANOVA for multi-arm continuous metrics, multiple-comparison correction as a cross-cutting concern, permutation tests as a general-purpose fallback, K-S tests for distribution-shape shifts, and sequential testing for the "can we peek daily?" problem.*
*Next: Chapter 7 — Randomization Units & Interference*
