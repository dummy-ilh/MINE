# Chapter 12: Choosing the Right Test (T-Test, Z-Test, Chi-Square, Mann-Whitney)

## 1. Definition

Different metric types and sample conditions require different statistical tests to validly compare treatment vs. control:

- **Z-test:** compares means (or proportions) when the population variance is known or the sample size is large enough that the sample variance is a reliable stand-in (CLT applies cleanly).
- **T-test:** compares means when population variance is unknown and must be estimated from the sample — uses the t-distribution, which has fatter tails to account for this extra uncertainty, especially important with smaller samples.
- **Chi-square test:** compares categorical/count data — tests whether observed frequencies across categories differ from expected frequencies (e.g., conversion yes/no across arms, or SRM checks from Chapter 10).
- **Mann-Whitney U test (Wilcoxon rank-sum):** a non-parametric test comparing two independent groups *without* assuming Normality — it works on ranks rather than raw values, making it robust to outliers and skew.

## 2. Layman Explanation

Think of these as different tools for different shaped problems:

- **Z-test** is for when you have a LOT of data and are comparing averages or rates — you're confident enough in your variance estimate to use the "clean" Normal distribution.
- **T-test** is the more cautious cousin of the z-test — used when your sample is smaller, so you build in extra uncertainty (fatter tails) because you're not fully sure your variance estimate is accurate.
- **Chi-square** is for when you're not comparing averages at all, but comparing *counts in buckets* — like "how many people clicked vs. didn't click," across two groups.
- **Mann-Whitney** is your fallback when your data is too messy/skewed (like revenue with huge outlier whales) to trust a mean-based test at all — instead of comparing raw values, it compares whether one group's values tend to *rank* higher than the other's, which sidesteps the influence of extreme outliers entirely.

## 3. Formal Explanation

**Decision framework:**

| Metric type | Sample size | Test |
|---|---|---|
| Continuous, variance known or n large | Large | Z-test |
| Continuous, variance unknown | Small/moderate | T-test |
| Binary/categorical counts | Any (with sufficient expected counts) | Chi-square (or z-test on proportions, equivalent for 2x2) |
| Continuous but heavily skewed/non-Normal, or ordinal | Any, especially small | Mann-Whitney U |

**Z-test formula (two-sample, proportions):**
Z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))

where p̂ is the pooled proportion across both arms.

**T-test formula (two-sample, means, Welch's version — unequal variance):**
t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

Welch's t-test (not assuming equal variance between arms) is generally preferred over the classic Student's t-test in A/B testing, since treatment can easily change variance as well as the mean, and assuming equal variance when it's false distorts the test.

**Chi-square formula (test of independence, e.g., 2x2 contingency table):**
χ² = Σ (Observedᵢ - Expectedᵢ)² / Expectedᵢ

Same formula structure as the SRM check in Chapter 10 — chi-square is a general tool for "does this observed count distribution differ from what's expected," reused across several contexts.

**Mann-Whitney U test (conceptual mechanics):**
Ranks all observations from both groups combined, then compares the sum of ranks between groups. Under the null (no difference), ranks should be randomly interspersed between groups; a group with systematically higher ranks suggests a real difference in distribution (specifically testing whether one distribution is stochastically greater than the other, not strictly a test of means).

**Key assumption checks:**
- T/Z-tests assume approximate Normality of the *sampling distribution of the mean* (via CLT) — not of the raw data itself (see Chapter 2).
- Chi-square requires sufficient expected cell counts (rule of thumb: expected count ≥ 5 per cell) — with very rare events, use Fisher's exact test instead.
- Mann-Whitney doesn't test means directly — it tests whether one distribution tends to produce larger values than the other; interpreting it as "testing the median" is only strictly valid under an added assumption (similarly shaped distributions).

## 4. Levers — What Controls It, What Moves It

**Metric type**
- Binary/count metrics naturally point to chi-square or proportion z-tests; continuous metrics point to t/z-tests; heavily skewed continuous metrics push toward Mann-Whitney or a transformation (log) first.

**Sample size**
- Small samples with unknown variance → t-test (fatter tails, more conservative). Large samples → t and z converge and the choice matters less.

**Distributional shape / outlier sensitivity**
- Heavy skew or extreme outliers (e.g., revenue with whales) degrades the reliability of mean-based tests even with CLT's help at moderate n — Mann-Whitney (or a trimmed/winsorized mean with a t-test) becomes more robust in these cases.

**Equal vs. unequal variance between arms**
- If treatment plausibly changes variance (common — e.g., a personalization feature could increase spread of outcomes even if the mean is similar), Welch's t-test (unequal variance assumed) is the safer default over Student's classic t-test.

## 5. Worked Example

You're comparing average order value (AOV) between control (n=40) and treatment (n=42) — a fairly small sample due to limited traffic on a checkout flow test.

- Control: mean = $52.10, sd = $18.40
- Treatment: mean = $56.30, sd = $24.70

Because n is small (under ~50 per arm) and variances look meaningfully different (18.40² = 338.6 vs. 24.70² = 610.1 — treatment's variance is nearly double), Welch's t-test is the appropriate choice over a pooled-variance test or a large-sample z-test.

Welch's t-statistic:
t = (56.30 - 52.10) / √(610.1/42 + 338.6/40)
t = 4.20 / √(14.53 + 8.47)
t = 4.20 / √23.0
t = 4.20 / 4.80
t ≈ 0.875

With Welch-adjusted degrees of freedom (which would come out to roughly 75 here, computed via the Welch–Satterthwaite equation), a t-statistic of 0.875 falls well short of the ~1.99 critical value needed for significance at α=0.05 — so you'd fail to reject the null: no significant difference in AOV detected, despite the $4.20 raw gap, largely because the sample is small and treatment's variance is notably higher, both of which inflate the standard error.

## 6. Famous Q&A (Google / Apple style)

**Q: Why would you choose Welch's t-test over a standard Student's t-test for an A/B test comparing average revenue between arms?**
A: Standard Student's t-test assumes both groups have equal variance, which is often not a safe assumption in A/B testing — a treatment can easily change the *spread* of outcomes (e.g., a personalization feature might make some users spend much more while others are unaffected, increasing variance) even when the mean shift is the main thing you're testing for. Welch's t-test doesn't assume equal variances and instead adjusts the degrees of freedom based on each group's own variance, making it more robust and generally the safer default in experimentation settings where you can't guarantee equal variance a priori.

**Q: Your revenue metric has a few users spending 50x the median. Should you run a standard t-test?**
A: I'd be cautious. Even though CLT means the sampling distribution of the mean will eventually approach Normal, heavily skewed data with extreme outliers converges more slowly (Chapter 2), and a handful of whale users can dominate the observed mean difference, making the test result fragile and highly sensitive to a tiny number of data points. I'd run a Mann-Whitney U test as a robustness check (since it's based on ranks and far less sensitive to extreme values), and/or apply a transformation like log or winsorizing before running the t-test, and see if conclusions are consistent across methods before trusting the result.

**Q: What's the difference between testing "is there a difference in conversion rate" via a chi-square test versus a z-test on two proportions?**
A: For a simple 2x2 case (treatment/control × converted/not-converted), the chi-square test of independence and the two-proportion z-test are mathematically equivalent — the chi-square statistic in this specific case equals the square of the z-statistic, and they'll produce the same p-value. Chi-square becomes more useful (and necessary) when you have more than two categories or more than two arms (e.g., comparing conversion across three treatment variants at once), where a simple two-proportion z-test no longer applies directly.

**Q: A colleague argues "since our sample size is huge (millions of users), we don't need to worry about which test we pick." Do you agree?**
A: Partially — with very large samples, t-tests and z-tests converge and the classic small-sample concerns (unknown variance, non-Normality of raw data) become less important because CLT kicks in strongly. However, the choice of test still matters for a different reason at scale: with extremely large samples, even a trivially small, practically meaningless effect will show up as statistically significant, so the real question shifts from "which test" to "is this effect size actually big enough to matter" (echoing the practical-vs-statistical-significance point from Chapter 4). I'd also still watch for skew/outlier sensitivity on metrics like revenue — large n helps with Normality of the mean, but a few extreme outliers can still meaningfully shift the *point estimate* itself, independent of the test's validity.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Defaulting to Student's t-test (equal variance assumed) without checking if treatment plausibly affects variance — use Welch's by default in A/B testing
- ❌ Running a standard t-test on heavily skewed/whale-dominated revenue data without a robustness check (Mann-Whitney, log-transform, winsorizing)
- ❌ Using chi-square with very low expected cell counts (<5) — switch to Fisher's exact test instead
- ❌ Interpreting Mann-Whitney's result as strictly "the medians differ" without checking the similarly-shaped-distributions assumption
- ✅ Do: match the test to both the metric type (continuous/binary/count) AND the distributional shape (Normal-ish vs. skewed/outlier-heavy)
- ✅ Do: remember chi-square and two-proportion z-test are equivalent for a simple 2x2 case

---
*Next: Chapter 13 — Ratio Metrics & the Delta Method (why naive variance formulas fail).*
