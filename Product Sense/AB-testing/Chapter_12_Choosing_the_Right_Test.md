# Chapter 12 (Boosted): Choosing the Right Test — The Capstone Decision Chapter

## 1. Why This Chapter Exists

Chapters 6 and 6.1 built the toolkit piece by piece: z for proportions, t for continuous metrics, then chi-square, Fisher's exact, Mann-Whitney, bootstrap, Poisson, ANOVA, and the rest for everything that breaks those two defaults. This chapter is the **capstone** — the single decision process you actually run, live, in an interview or in production, to go from "here's a metric" to "here's the test" without re-deriving the whole toolkit from scratch each time.

If Chapters 6/6.1 are the reference manual, this is the muscle memory. The goal by the end of this chapter: given *any* metric description, you should be able to name the test in under 10 seconds, and — more importantly — name the specific assumption that would make you switch away from your first answer.

---

## 2. Definitions, Sharpened

- **Z-test**: compares means or proportions when the population variance is known, or the sample is large enough that the sample variance is a reliable stand-in (CLT applies cleanly). In practice, almost always used for **proportions**, because $p(1-p)$ removes the "unknown variance" problem entirely — there's nothing separate to estimate.
- **T-test**: compares means when population variance is unknown and must be estimated from the sample — uses the t-distribution, which has fatter tails to account for this extra uncertainty, especially important with smaller samples. In practice, almost always used for **continuous metrics** (revenue, duration, latency), because their variance is a free parameter, independent of the mean.
- **Chi-square test**: compares categorical/count data — tests whether observed frequencies across categories differ from expected frequencies (e.g., conversion yes/no across arms, or SRM checks from Chapter 10). The natural generalization of the proportion z-test once you have **more than 2 groups or more than 2 outcome categories**.
- **Mann-Whitney U test (Wilcoxon rank-sum)**: a non-parametric test comparing two independent groups *without* assuming Normality — it works on ranks rather than raw values, making it robust to outliers and skew. The fallback the moment a continuous metric's **shape**, not just its sample size, is the problem.

**The one-line versions, memorized**:
- Z → "proportion, or continuous metric at huge n"
- T (Welch's) → "continuous metric, variance estimated, don't assume equal variance between arms"
- Chi-square → "categorical, 3+ groups or 3+ outcome buckets"
- Mann-Whitney → "continuous, but skewed/outlier-heavy enough that the mean itself isn't trustworthy"

---

## 3. 🧭 The Live Decision Flowchart

```
                    START: An interviewer (or a dashboard) hands
                    you a metric. What test do you run?
                                      │
                                      ▼
                Is the metric a PROPORTION / binary outcome
                       (converted yes/no)?
                                      │
                    ┌─────────────Yes─┴─No────────────────┐
                    ▼                                       ▼
        How many groups/arms are               Metric is CONTINUOUS.
        being compared?                        Is it heavily SKEWED /
                    │                           outlier-dominated (whales,
        ┌──────2───┴───3+─────┐                 long tail)?
        ▼                     ▼                            │
   Z-TEST                CHI-SQUARE               ┌───Yes──┴──No──────┐
  (2-proportion)       (test of independence)      ▼                   ▼
        │                     │              MANN-WHITNEY U      Do treatment and
        ▼                     ▼              (or bootstrap/       control arms have
  Are expected cell    Are expected cell       log-transform        meaningfully
  counts ≥5 in EVERY    counts ≥5 in EVERY      as a robustness      DIFFERENT variance
  cell?                 cell?                   check)              (s₁² vs s₂²)?
        │                     │                                            │
   ┌───Yes┴─No──┐      ┌───Yes┴─No──┐                          ┌──────Yes──┴──No───────┐
   ▼             ▼      ▼            ▼                          ▼                        ▼
 proceed    Use FISHER'S  proceed   Use FISHER'S          Use WELCH'S T-TEST      Student's t-test
 as-is      EXACT TEST    as-is     EXACT TEST            (unequal-variance        would also work,
                                    (or a similar          default — always        but Welch's costs
                                    exact-test              safe to use even        nothing to use
                                    generalization)          when variances          instead, so
                                                             ARE equal)              default to it anyway
                                      │
                                      ▼
                    After ANY test above: is your sample size
                    so large that even a tiny, practically
                    meaningless effect would be "significant"?
                                      │
                                      ▼
                    Report effect SIZE and confidence interval
                    alongside the p-value — statistical
                    significance ≠ practical significance
                    (Chapter 4 callback)
```

---

## 4. Formulas, In Context

**Z-test (two-sample, proportions):**
$$z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}$$
where $\hat p$ is the pooled proportion across both arms, computed under the null hypothesis that both arms share a single true conversion rate.

**T-test (two-sample, means, Welch's version — unequal variance):**
$$t = \frac{\bar X_1 - \bar X_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

Welch's t-test (not assuming equal variance between arms) is generally preferred over the classic Student's t-test in A/B testing, since treatment can easily change variance as well as the mean, and assuming equal variance when it's false distorts the test — with a Welch-Satterthwaite adjustment to degrees of freedom:
$$df \approx \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Chi-square (test of independence, e.g., $r \times c$ contingency table):**
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

Same formula structure as the SRM check in Chapter 10 — chi-square is a general tool for "does this observed count distribution differ from what's expected," reused across several contexts. For exactly 2 groups × 2 outcomes, $\chi^2 = z^2$ — literally the same test, different name.

**Mann-Whitney U test (conceptual mechanics):**
Ranks all observations from both groups combined, then compares the sum of ranks between groups:
$$U_1 = R_1 - \frac{n_1(n_1+1)}{2}$$
Under the null (no difference), ranks should be randomly interspersed between groups; a group with systematically higher ranks suggests a real difference in distribution (specifically testing whether one distribution is stochastically greater than the other, not strictly a test of means).

**Key assumption checks, all in one place**:
- T/Z-tests assume approximate Normality of the *sampling distribution of the mean* (via CLT) — not of the raw data itself (Chapter 2 callback). This is the single most commonly-confused point in this whole topic.
- Chi-square requires sufficient expected cell counts (rule of thumb: expected count ≥ 5 per cell) — with very rare events, use Fisher's exact test instead.
- Mann-Whitney doesn't test means directly — it tests whether one distribution tends to produce larger values than the other; interpreting it as "testing the median" is only strictly valid under an added assumption (similarly shaped distributions).

---

## 5. Levers — What Controls the Choice, What Moves It

**Metric type**
Binary/count metrics naturally point to chi-square or proportion z-tests; continuous metrics point to t/z-tests; heavily skewed continuous metrics push toward Mann-Whitney or a transformation (log) first.

**Sample size**
Small samples with unknown variance → t-test (fatter tails, more conservative). Large samples → t and z converge and the choice matters less *numerically* — though see Section 8 on why it still matters *conceptually* at scale.

**Distributional shape / outlier sensitivity**
Heavy skew or extreme outliers (e.g., revenue with whales) degrades the reliability of mean-based tests even with CLT's help at moderate n — Mann-Whitney (or a trimmed/winsorized mean with a t-test) becomes more robust in these cases.

**Equal vs. unequal variance between arms**
If treatment plausibly changes variance (common — e.g., a personalization feature could increase spread of outcomes even if the mean is similar), Welch's t-test (unequal variance assumed) is the safer default over Student's classic t-test. **There is no scenario where defaulting to Welch's costs you anything** — this is worth stating explicitly, since it removes an entire decision branch from your live reasoning.

**Number of arms/groups**
2 groups → z or t directly. 3+ groups on a proportion metric → chi-square (then pairwise z-tests with correction to localize). 3+ groups on a continuous metric → ANOVA (Chapter 6.1) first, then pairwise t-tests with correction.

---

## 6. Worked Example — Small-Sample Checkout Flow Test

You're comparing average order value (AOV) between control ($n=40$) and treatment ($n=42$) — a fairly small sample due to limited traffic on a checkout flow test.

- Control: mean = \$52.10, sd = \$18.40
- Treatment: mean = \$56.30, sd = \$24.70

**Reasoning through the flowchart**: continuous metric → not obviously skewed (no whales mentioned, this is AOV not raw revenue) → check variance: $18.40^2 = 338.6$ vs. $24.70^2 = 610.1$ — treatment's variance is nearly double control's. That's a meaningfully different variance, so Welch's t-test is the right call, and it would have been the right call even if variances looked similar, since there's no cost to defaulting to it.

**Welch's t-statistic:**
$$t = \frac{56.30-52.10}{\sqrt{\frac{610.1}{42}+\frac{338.6}{40}}} = \frac{4.20}{\sqrt{14.53+8.47}} = \frac{4.20}{\sqrt{23.0}} = \frac{4.20}{4.80} \approx 0.875$$

With Welch-adjusted degrees of freedom (which would come out to roughly 75 here, computed via the Welch-Satterthwaite equation), a t-statistic of 0.875 falls well short of the ~1.99 critical value needed for significance at $\alpha=0.05$ — so you'd fail to reject the null: no significant difference in AOV detected, despite the \$4.20 raw gap, largely because the sample is small and treatment's variance is notably higher, both of which inflate the standard error.

**The decision this produces**: don't ship on this metric alone — but note the direction (treatment numerically higher) and consider whether the test is underpowered (Chapter 5 callback) rather than concluding "there's no effect."

---

## 7. Worked Example — Multi-Arm Proportion Test, Routed Through the Flowchart

Extending the reasoning to a 3-arm test: three checkout button colors (A, B, C), binary conversion outcome.

**Flowchart path**: proportion → 3+ groups → chi-square, not a two-proportion z-test.

| Variant | Converted | Not Converted | Total |
|---|---|---|---|
| A | 120 | 4880 | 5000 |
| B | 145 | 4855 | 5000 |
| C | 110 | 4890 | 5000 |

Overall rate $=\frac{375}{15{,}000}=0.025$, so expected "Converted" count per variant $=125$. All three variants' expected counts (125 and their "not converted" complements) are comfortably above the ≥5 threshold, so chi-square (not Fisher's exact) is appropriate. A significant chi-square result tells you *some* variant differs — you'd still need pairwise z-tests with a multiple-comparison correction (Bonferroni or FDR, Chapter 6.1) to find out which one, and by how much.

---

## 8. Famous Q&A (Google / Apple / OpenAI style)

**Q: Why would you choose Welch's t-test over a standard Student's t-test for an A/B test comparing average revenue between arms?**
A: Standard Student's t-test assumes both groups have equal variance, which is often not a safe assumption in A/B testing — a treatment can easily change the *spread* of outcomes (e.g., a personalization feature might make some users spend much more while others are unaffected, increasing variance) even when the mean shift is the main thing you're testing for. Welch's t-test doesn't assume equal variances and instead adjusts the degrees of freedom based on each group's own variance, making it more robust and generally the safer default in experimentation settings where you can't guarantee equal variance a priori — and since it costs nothing when variances happen to be equal, there's no reason not to default to it.

**Q: Your revenue metric has a few users spending 50x the median. Should you run a standard t-test?**
A: I'd be cautious. Even though CLT means the sampling distribution of the mean will eventually approach Normal, heavily skewed data with extreme outliers converges more slowly (Chapter 2), and a handful of whale users can dominate the observed mean difference, making the test result fragile and highly sensitive to a tiny number of data points. I'd run a Mann-Whitney U test as a robustness check (since it's based on ranks and far less sensitive to extreme values), and/or apply a transformation like log or winsorizing before running the t-test, and see if conclusions are consistent across methods before trusting the result.

**Q: What's the difference between testing "is there a difference in conversion rate" via a chi-square test versus a z-test on two proportions?**
A: For a simple 2×2 case (treatment/control × converted/not-converted), the chi-square test of independence and the two-proportion z-test are mathematically equivalent — the chi-square statistic in this specific case equals the square of the z-statistic, and they'll produce the same p-value. Chi-square becomes more useful (and necessary) when you have more than two categories or more than two arms (e.g., comparing conversion across three treatment variants at once), where a simple two-proportion z-test no longer applies directly.

**Q: A colleague argues "since our sample size is huge (millions of users), we don't need to worry about which test we pick." Do you agree?**
A: Partially — with very large samples, t-tests and z-tests converge and the classic small-sample concerns (unknown variance, non-Normality of raw data) become less important because CLT kicks in strongly. However, the choice of test still matters for a different reason at scale: with extremely large samples, even a trivially small, practically meaningless effect will show up as statistically significant, so the real question shifts from "which test" to "is this effect size actually big enough to matter" (echoing the practical-vs-statistical-significance point from Chapter 4). I'd also still watch for skew/outlier sensitivity on metrics like revenue — large n helps with Normality of the mean, but a few extreme outliers can still meaningfully shift the *point estimate* itself, independent of the test's validity.

**Q: You're asked to design the test-selection logic for an internal experimentation platform that runs thousands of experiments a week. How would you architect it?**
A: I'd build a metric-metadata layer that tags each metric with its type (proportion, continuous, count, ratio) and known distributional properties (e.g., "revenue: known to be right-skewed, flag for Mann-Whitney/bootstrap alongside the primary t-test"). The routing logic then follows this chapter's flowchart automatically: proportions with 2 arms → z-test, 3+ arms → chi-square with cell-count checks that fall back to Fisher's exact; continuous metrics get Welch's t-test by default, with an automatic skew check (e.g., a quick check of the ratio of mean to median, or an outlier-share threshold) that triggers a parallel Mann-Whitney/bootstrap robustness check. On top of all of this, I'd enforce a platform-level multiple-comparison policy (Chapter 6.1) so that teams reporting many secondary metrics aren't silently accumulating false-positive risk. The key design principle: the *person running the experiment* shouldn't have to manually pick the test — the platform should pick correctly by default and only surface a warning when something (skew, small expected cell counts, overdispersion) needs human judgment.

---

## 9. Common Mistakes / Red Flags (Quick Review)

- ❌ Defaulting to Student's t-test (equal variance assumed) without checking if treatment plausibly affects variance — use Welch's by default in A/B testing.
- ❌ Running a standard t-test on heavily skewed/whale-dominated revenue data without a robustness check (Mann-Whitney, log-transform, winsorizing).
- ❌ Using chi-square with very low expected cell counts (<5) — switch to Fisher's exact test instead.
- ❌ Interpreting Mann-Whitney's result as strictly "the medians differ" without checking the similarly-shaped-distributions assumption.
- ❌ Running $\binom{k}{2}$ pairwise tests across a multi-arm experiment with no omnibus test (chi-square/ANOVA) first and no multiple-comparison correction after.
- ❌ Treating "test chosen correctly" and "effect is practically meaningful" as the same conclusion at very large sample sizes.
- ✅ Do: match the test to both the metric type (continuous/binary/count) AND the distributional shape (Normal-ish vs. skewed/outlier-heavy).
- ✅ Do: remember chi-square and two-proportion z-test are equivalent for a simple 2×2 case.
- ✅ Do: report effect size and confidence interval alongside any p-value, especially at large n.

---

## 10. L5-Differentiating Talking Points

- Reciting the flowchart *as a decision process*, not a memorized lookup table — showing you'd arrive at the same routing logic even for a metric type not explicitly listed here.
- Stating unprompted that Welch's t-test has no downside relative to Student's, which collapses an entire branch of the decision tree and signals you've internalized the tradeoff rather than memorized "use Welch's."
- Connecting the chi-square ↔ z-test equivalence at 2×2 to *why* it stops holding at 3+ groups (chi-square's null distribution and degrees of freedom generalize; the simple two-proportion z-test formula has no natural multi-group extension).
- Bringing up multiple-comparison correction and omnibus testing (ANOVA/chi-square before pairwise) unprompted the moment a scenario has 3+ arms.
- At the "huge sample size" question, pivoting cleanly from statistical to practical significance — this is one of the most reliable ways interviewers distinguish L5 from L4 on this exact topic.
- Describing how you'd architect test-selection as a platform-level default (Q&A #5) rather than a per-experiment manual decision — shows systems thinking beyond "which formula do I plug numbers into."

---

## 11. Comprehension Check

1. Walk through the full flowchart from Section 3 for: (a) a 2-arm proportion test with expected cell counts of 3, (b) a 4-arm continuous revenue metric with clear whales, (c) a 2-arm continuous latency metric with very similar variances at $n=2$ million per arm.
2. Why is there effectively no downside to defaulting to Welch's t-test over Student's t-test in A/B testing?
3. Explain, without just stating the rule, why chi-square and the two-proportion z-test give identical p-values at exactly 2 groups but diverge as tools once you have 3+ groups.
4. A stakeholder says "the p-value is 0.001, this is a huge win." Your sample size was 5 million per arm and the absolute lift was 0.02 percentage points. How do you respond?
5. Describe how you'd design the automatic test-routing logic for an internal experimentation platform, and what metadata about each metric you'd need to make that routing correct by default.

---
*Next: Chapter 13 — Ratio Metrics & the Delta Method (why naive variance formulas fail).*
