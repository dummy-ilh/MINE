# Chapter 14: Sample Ratio Mismatch (SRM)

*(Module 3: Failure Modes & Diagnostics begins here)*

## 1. Intuition

Before you even look at your metrics, there's a much more basic question you should always ask first: **did the randomization actually work the way it was supposed to?** If you designed a 50/50 split but somehow ended up with 51,000 users in control and 49,000 in treatment, something upstream of your statistical analysis is broken — and any metric result you compute on top of that broken split is potentially meaningless, no matter how careful your t-test or CUPED adjustment was.

**Sample Ratio Mismatch (SRM)** is exactly this: a statistically significant deviation between your expected randomization ratio (e.g., 50/50) and your observed ratio (e.g., 51/49). It is, in the experimentation community, considered one of the most important diagnostic checks to run — arguably *more* important than the actual treatment effect analysis, because an SRM invalidates everything downstream.

## 2. Why SRM Happens

SRM is a symptom of a **broken or biased randomization/assignment/logging pipeline**, not random noise (a true 50/50 randomization would only rarely deviate this much by chance). Common root causes:

- **Bugs in the assignment logic**: e.g., an off-by-one error, a caching bug that assigns a stale bucket to some users, or a race condition in a high-traffic system.
- **Differential logging/exclusion**: if the treatment experience crashes for some users before a "successfully exposed" log fires, but the control experience doesn't have this failure mode, treatment users who crash are silently dropped from your denominator — this creates a *survivorship bias* look-alike where your treatment population is secretly filtered to "users who didn't crash," which correlates with your outcome metric.
- **Bot/crawler contamination differing by arm**: automated traffic sometimes gets triggers or filtered differently depending on which arm's code path it hits, distorting the ratio.
- **Redirect/loading time differences**: if the treatment experience takes measurably longer to load, users on slow connections or older devices may disproportionately bounce/timeout before being counted, systematically excluding a specific type of user from treatment but not control.

The common thread across all these causes: **something about being assigned to treatment vs. control is influencing who ends up counted in the experiment at all** — which is a violation of the core randomization guarantee from Chapter 1, and specifically undermines the claim that treatment and control groups are comparable in expectation.

## 3. Detecting SRM: The Chi-Square Goodness-of-Fit Test

Given observed counts $n_1$ (treatment) and $n_2$ (control), and an expected ratio (e.g., 50/50, so expected counts are $\frac{n_1+n_2}{2}$ each):

$$\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in arm $i$ and $E_i$ is the expected count under the intended ratio. With one degree of freedom (two groups), compare $\chi^2$ to the chi-square distribution to get a p-value.

**Practical convention (worth knowing)**: many experimentation platforms use a *very* strict threshold for SRM checks — often $p < 0.001$ rather than the usual 0.05 — specifically because SRM checks run on every single experiment, all the time, as an automated pre-check (a multiple-testing consideration, foreshadowing the next chapter), and because the cost of a false SRM alarm (needlessly distrusting a good experiment) is generally considered much lower than the cost of missing a real SRM (shipping based on a broken experiment).

## 4. Worked Example

An experiment intended a 50/50 split. Observed: 50,850 users in control, 49,150 in treatment (total $n=100{,}000$).

**Expected counts under $H_0$ (true 50/50 ratio)**: 50,000 each.

$$\chi^2 = \frac{(50850-50000)^2}{50000} + \frac{(49150-50000)^2}{50000} = \frac{850^2}{50000}+\frac{850^2}{50000} = \frac{722500}{50000}\times2 = 28.9$$

For 1 degree of freedom, a $\chi^2$ value of 28.9 corresponds to a p-value far below 0.001 (the critical $\chi^2$ value for $p=0.001$ at 1 df is about 10.83) — **this is a clear, severe SRM.** Even though the imbalance (50,850 vs 49,150, roughly a 1.7% deviation) might look small and easy to dismiss informally, with $n=100{,}000$ the statistical test has enormous power to detect even small ratio deviations, and correctly flags this as essentially impossible to arise from genuine 50/50 randomization by chance.

**What you do next**: STOP. Do not analyze or trust the treatment-effect metrics from this experiment until the root cause of the SRM is found and fixed. Common next steps: check assignment logs for the missing ~1,700 users, check whether the gap concentrates in a particular platform/browser/device segment (which often reveals the mechanism, e.g., "all missing users are on a specific old Android version that fails to fire the exposure log for the treatment experience specifically").

## 5. Production Considerations

- **SRM checks should be automated and run on every experiment, as a mandatory pre-check before any metric analysis is trusted** — not something a data scientist remembers to manually check occasionally. At Google/Meta-scale platforms, this is typically baked directly into the experimentation dashboard as a hard blocker or prominent warning banner.
- **A significant SRM doesn't just mean "throw away this experiment"** — it's a debugging signal. Segment the SRM by platform, geography, device, browser, and time to localize the root cause; this segmentation is often the single fastest way to find the actual bug.
- **SRM checks apply beyond the top-line 50/50 split** — if you have multiple treatment arms, or unequal allocation (e.g., 90/10 for a risky feature), you should still run the goodness-of-fit test against whatever your *intended* ratio was, not assume 50/50 is the only relevant baseline.
- **Even "no SRM" isn't a complete guarantee of a clean experiment** — SRM only catches *count* imbalances; it doesn't catch composition imbalances where counts match but the underlying users differ systematically for reasons other than random assignment (that requires balance checks on pre-experiment covariates, a related but distinct diagnostic).

## 6. Interview Traps

- **Trap #1**: Not knowing what SRM stands for or how to test for it — this is one of the most frequently and directly asked factual questions in A/B testing interviews, precisely because it's such a critical practical gatekeeper check.
- **Trap #2**: Treating a "small" percentage imbalance (like the 1.7% in the worked example) as obviously negligible without running the actual test — at large $n$, even small-looking imbalances can be statistically overwhelming evidence of a real problem, exactly the same "statistical vs. practical significance" theme from Chapter 2, but inverted: here, a small-looking gap IS practically important because it signals a broken pipeline, regardless of how small it looks by eye.
- **Trap #3**: Proposing to just re-run the analysis with a corrected weighting scheme instead of finding and fixing the root cause — SRM is a signal of a broken data-generating process, and no amount of downstream statistical correction reliably fixes a corrupted randomization.
- **Trap #4**: Confusing SRM (count-level mismatch) with covariate imbalance (composition-level mismatch, e.g., treatment group happens to have more mobile users despite correct counts) — these are related "is my randomization trustworthy" checks but are diagnostically distinct and are checked with different tests.

## 7. L5-Differentiating Talking Points

- Immediately stating "the first thing I'd check before trusting ANY metric result is whether there's an SRM" as your opening move on any experiment analysis question signals real practitioner instinct, not textbook knowledge — many interviewers are specifically listening for whether you volunteer this unprompted.
- Being able to compute the chi-square statistic live, as in the worked example, and correctly conclude that a "small-looking" imbalance can be a severe, statistically overwhelming SRM at large sample sizes, shows quantitative fluency plus the right intuition about when small differences matter.
- Proposing a debugging workflow (segment SRM by platform/device/geo to localize root cause) rather than just "the experiment is broken, restart it" shows you think about SRM as an actionable engineering signal, not just a stop sign.
- Distinguishing SRM (count mismatch) from covariate/composition imbalance (a related but separate diagnostic) shows precision in a space where interviewers often deliberately probe for conflation.

## 8. Comprehension Check

1. What is SRM, and why is checking for it typically considered more urgent than analyzing the actual treatment effect?
2. Compute the chi-square statistic for an experiment with 40,600 users in treatment and 39,400 in control (intended 50/50 split), and determine if this represents a significant SRM at the strict $p<0.001$ threshold.
3. Give two distinct root-cause mechanisms that could produce SRM in a real production system.
4. Why do many experimentation platforms use $p<0.001$ rather than $p<0.05$ as the SRM significance threshold?
5. Your experiment shows no SRM (counts are balanced as expected), but you're still not 100% sure randomization worked correctly. What additional diagnostic would you run, and what would it check for that SRM doesn't?

---
*Next: Chapter 15 — Novelty & Primacy Effects*
