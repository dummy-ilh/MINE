# Chapter 10: Randomization Mechanics — Hashing, Bucketing, SRM Checks

---

## 1. Intuition

Before you even look at your metrics, there's a much more basic question you should always ask first: **did the randomization actually work the way it was supposed to?** If you designed a 50/50 split but somehow ended up with 51,000 users in control and 49,000 in treatment, something upstream of your statistical analysis is broken — and any metric result you compute on top of that broken split is potentially meaningless, no matter how careful your t-test or CUPED adjustment was.

This tutorial covers two tightly linked topics:
1. **Randomization mechanics** — the engineering methods (hashing, bucketing, salting) used to actually assign users to arms in a way that's random, consistent, and independently verifiable.
2. **Sample Ratio Mismatch (SRM)** — the diagnostic check that verifies the assignment mechanism actually worked as intended, and is arguably the single most important pre-check in all of experimentation, because it can invalidate everything downstream.

### Layman analogy — hashing/bucketing
You want every user to get a "coin flip" that decides which arm they land in — but you also want the *same* user to always get the same coin-flip result every time they visit (otherwise their experience keeps flip-flopping, and your analysis can't cleanly attribute outcomes to one arm). **Hashing** solves this: instead of an actual random coin flip each time, you compute a deterministic scramble of the user's ID that *looks* random but always gives the same output for the same input. Add an experiment-specific "salt" (like a seasoning) so the same user gets an independent, unrelated bucket assignment in a different experiment — otherwise a user who happens to always land in "treatment" would end up in treatment for every single experiment ever run, a disaster for statistical validity.

### Layman analogy — SRM
SRM checking is like weighing out ingredients before baking: if a recipe calls for a 50/50 split of flour and sugar by weight, and you actually weigh what went into the bowl and find it's 60/40, something went wrong in the process — maybe the sugar container is clogged, maybe your scale is miscalibrated. You don't proceed to bake (analyze results) until you understand why the split is off, because a broken assignment process usually corrupts everything downstream, not just the ratio.

**Why SRM matters so much**: in the experimentation community it's considered one of the most important diagnostic checks to run — arguably *more* important than the actual treatment-effect analysis, because an SRM invalidates everything downstream, no matter how careful your subsequent statistics are.

---

## 2. Core Definitions

- **Randomization mechanics**: the engineering methods used to actually assign users to experiment arms in a way that is random, consistent, and independently verifiable. The core technique is **hashing**.
- **Hashing**: applying a deterministic hash function to a user identifier (plus an experiment-specific salt) to produce a pseudo-random number, which is then mapped into treatment/control **buckets**.
- **Sample Ratio Mismatch (SRM)**: a statistically significant deviation between your expected randomization ratio (e.g., 50/50) and your observed ratio (e.g., 51/49). A diagnostic check verifying that the actual observed split of users across arms matches the intended split. A significant SRM is a red flag that something in the randomization or data pipeline is broken — and it invalidates any effect estimate from that experiment until resolved.

---

## 3. Hashing-Based Bucketing — How Assignment Actually Works

$$\text{bucket(user)} = \text{hash}(\text{user\_id} + \text{experiment\_salt}) \mod N$$

Where N is the number of buckets (commonly 100 or 1000 for fine-grained traffic allocation). Buckets are then mapped to arms (e.g., buckets 0–49 → control, 50–99 → treatment for a 50/50 split).

### Properties required of the hash function
- **Deterministic**: same input always produces same output (ensures a user stays in the same arm across sessions).
- **Uniform distribution**: outputs should be evenly spread across the bucket range, so the split is actually balanced.
- **Independence across experiments**: using a different salt per experiment ensures a user's bucket assignment in Experiment A is statistically independent of their assignment in Experiment B — critical for running many concurrent experiments without confounding each other.

### Why salting matters
Reusing the same user-to-bucket mapping across all experiments means a user's treatment/control assignment would be perfectly correlated across every experiment ever run — if User X lands in "treatment" for Experiment 1, they'd also always land in "treatment" for Experiments 2, 3, 4, etc. This creates confounding between concurrent experiments: any effect you observe could be entangled with effects from other experiments the same users are simultaneously exposed to. A unique salt per experiment makes bucket assignments independent across experiments, which is what allows running many experiments concurrently without them contaminating each other.

### Supporting gradual rollouts (1% → 10% → 50%)
Using a fine-grained bucket range (e.g., 1000 buckets via `hash(user_id + salt) mod 1000`) lets you expand the treatment population by simply reassigning additional buckets to treatment (e.g., buckets 0–9 → treatment at 1%, then 0–99 → treatment at 10%), without changing which bucket any individual user falls into. Since the underlying hash-to-bucket mapping for each user never changes, everyone who was already in treatment stays in treatment as you ramp up — you're just expanding the boundary of which buckets count as "treatment," not re-randomizing anyone.

---

## 4. Why SRM Happens — Root Causes

SRM is a symptom of a **broken or biased randomization/assignment/logging pipeline**, not random noise (a true 50/50 randomization would only rarely deviate this much by chance). Common root causes, combining both the bucketing-mechanics angle and the pipeline/logging angle:

- **Bugs in the assignment logic**: e.g., an off-by-one error, a caching bug that assigns a stale bucket to some users, or a race condition in a high-traffic system.
- **Bucketing logic applied inconsistently** across platforms (web vs. mobile) or before vs. after a redirect/filter.
- **Non-uniform hash function or insufficient bucket granularity** causing clumping — a poor-quality or ad hoc hash function can silently create SRM even with technically "correct" logic, which is why standard, well-vetted hash functions (e.g., MD5, MurmurHash) are preferred.
- **Differential logging/exclusion**: if the treatment experience crashes for some users before a "successfully exposed" log fires, but the control experience doesn't have this failure mode, treatment users who crash are silently dropped from your denominator — creating a *survivorship-bias* look-alike where your treatment population is secretly filtered to "users who didn't crash," which correlates with your outcome metric. Similarly, a logging pipeline can drop events differently by arm (e.g., treatment's new code path has a subtly different logging call that fails silently more often).
- **Bot/crawler contamination differing by arm**: automated traffic sometimes gets triggered or filtered differently depending on which arm's code path it hits, distorting the ratio.
- **Redirect/loading-time differences**: if the treatment experience takes measurably longer to load, users on slow connections or older devices may disproportionately bounce/timeout before being counted, systematically excluding a specific type of user from treatment but not control.

**The common thread across all these causes**: something about being assigned to treatment vs. control is influencing who ends up counted in the experiment at all — a violation of the core randomization guarantee, and specifically undermines the claim that treatment and control groups are comparable in expectation.

---

## 5. Detecting SRM — The Chi-Square Goodness-of-Fit Test

Given observed counts $O_i$ across arms and expected counts $E_i$ under the intended ratio (e.g., for 50/50, $E_i = \frac{n_1+n_2}{2}$ each):

$$\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

With one degree of freedom (two groups), compare $\chi^2$ to the chi-square distribution to get a p-value.

**Practical convention (worth knowing)**: many experimentation platforms use a *very* strict threshold for SRM checks — often **p < 0.001** rather than the usual 0.05. Two converging reasons for this:
1. SRM checks are run at a stricter threshold than normal metric tests because even a small persistent bias is a serious pipeline problem, not a subtle effect worth debating.
2. SRM checks run on every single experiment, all the time, as an automated pre-check (a multiple-testing consideration — running thousands of checks at α=0.05 would itself generate many false alarms), and the cost of a false SRM alarm (needlessly distrusting a good experiment) is generally considered much lower than the cost of missing a real SRM (shipping based on a broken experiment).

---

## 6. Worked Examples

### Example A — Small, non-significant deviation (passes)

Suppose you're running an experiment intended as a 50/50 split, and after a week you observe:
- Control: 49,800 users
- Treatment: 50,200 users
- Total: 100,000 users

Expected under 50/50: 50,000 each.

$$\chi^2 = \frac{(49{,}800-50{,}000)^2}{50{,}000} + \frac{(50{,}200-50{,}000)^2}{50{,}000} = \frac{200^2}{50{,}000}\times 2 = 0.8+0.8 = 1.6$$

With 1 degree of freedom, the critical value at p=0.05 is 3.84. Since 1.6 < 3.84, this is **NOT** a significant SRM — this level of imbalance is consistent with normal random variation, and you can proceed with analysis.

### Example B — Moderate deviation (fails even the standard threshold)

Now suppose instead you saw Control: 48,000 / Treatment: 52,000:

$$\chi^2 = \frac{2{,}000^2}{50{,}000} + \frac{2{,}000^2}{50{,}000} = 80+80 = 160$$

160 >> 3.84 — a highly significant SRM. Before trusting any lift numbers from this experiment, you'd need to investigate: check platform-level splits (is the imbalance concentrated in iOS vs. Android?), check for bot traffic, check logging completeness per arm.

### Example C — Small-looking but severe deviation at large n (the trap)

An experiment intended a 50/50 split. Observed: 50,850 users in control, 49,150 in treatment (total n=100,000). Expected: 50,000 each.

$$\chi^2 = \frac{(50{,}850-50{,}000)^2}{50{,}000} + \frac{(49{,}150-50{,}000)^2}{50{,}000} = \frac{850^2}{50{,}000}\times 2 = 28.9$$

For 1 degree of freedom, a χ² of 28.9 corresponds to a p-value far below 0.001 (the critical χ² value for p=0.001 at 1 df is about 10.83) — **a clear, severe SRM.** Even though the imbalance (50,850 vs. 49,150, roughly a 1.7% deviation) might look small and easy to dismiss informally, at n=100,000 the statistical test has enormous power to detect even small ratio deviations, and correctly flags this as essentially impossible to arise from genuine 50/50 randomization by chance.

**What you do next**: STOP. Do not analyze or trust the treatment-effect metrics from this experiment until the root cause of the SRM is found and fixed. Common next steps: check assignment logs for the missing ~1,700 users, check whether the gap concentrates in a particular platform/browser/device segment (which often reveals the mechanism, e.g., "all missing users are on a specific old Android version that fails to fire the exposure log for the treatment experience specifically").

---

## 7. Levers — What Controls Randomization Quality & SRM Risk

**Salt choice**
- A unique salt per experiment is what guarantees independence across concurrent experiments — reusing salts (or a shared/global bucket assignment for all experiments) creates correlated assignment across experiments, undermining the validity of running them simultaneously.

**Number of buckets (N)**
- Finer granularity (e.g., 1000 buckets instead of 100) allows more precise traffic ramp control (1% → 5% → 50%) without needing to change the underlying hash logic — just remap which buckets belong to which arm.

**Hash function quality**
- A poor-quality or non-uniform hash function can silently create SRM even with technically "correct" logic — prefer standard, well-vetted hash functions (MD5, MurmurHash) over ad hoc string manipulation.

**Where in the stack bucketing happens**
- Bucketing done too late in the request pipeline (e.g., after some filtering or redirect logic that differs by platform) risks systematically excluding certain user segments from one arm — a frequent, hard-to-spot cause of SRM.

**SRM significance threshold**
- Stricter than standard metric tests (p<0.001 vs. p<0.05) by design — see Section 5.

---

## 8. Production Considerations

- **SRM checks should be automated and run on every experiment**, as a mandatory pre-check before any metric analysis is trusted — not something a data scientist remembers to manually check occasionally. At large-scale platforms, this is typically baked directly into the experimentation dashboard as a hard blocker or prominent warning banner.
- **A significant SRM doesn't just mean "throw away this experiment"** — it's a debugging signal. Segment the SRM by platform, geography, device, browser, and time to localize the root cause; this segmentation is often the single fastest way to find the actual bug.
- **SRM checks apply beyond the top-line 50/50 split** — if you have multiple treatment arms, or unequal allocation (e.g., 90/10 for a risky feature), you should still run the goodness-of-fit test against whatever your *intended* ratio was, not assume 50/50 is the only relevant baseline.
- **Even "no SRM" isn't a complete guarantee of a clean experiment** — SRM only catches *count* imbalances; it doesn't catch composition imbalances where counts match but the underlying users differ systematically for reasons other than random assignment (that requires separate balance checks on pre-experiment covariates — a related but distinct diagnostic).
- **Use fine-grained buckets (100–1000)** to support gradual rollouts without re-randomizing existing users (Section 3).

---

## 9. Interview Traps (Consolidated)

1. **Reusing the same salt across multiple experiments** → creates cross-experiment confounding.
2. **Ignoring a statistically significant SRM because the ratio "looks close enough"** → root-cause first, always; a "small-looking" imbalance (like the 1.7% in Example C) can be statistically overwhelming evidence of a real problem at large n — the same "statistical vs. practical significance" theme, but inverted: here a small-looking gap IS practically important because it signals a broken pipeline, regardless of how small it looks by eye.
3. **Bucketing at a point in the pipeline after platform-specific filtering/redirects** → risk of systematic exclusion in one arm.
4. **Using a low-quality/non-uniform hash function** → can silently create imbalance even with "correct" logic.
5. **Running SRM checks at the standard α=0.05 instead of a stricter threshold** → SRM checks warrant more sensitivity, not less, given how serious a broken pipeline is.
6. **Not knowing what SRM stands for or how to test for it** — one of the most frequently and directly asked factual questions in A/B testing interviews, precisely because it's such a critical practical gatekeeper check.
7. **Proposing to just re-run the analysis with a corrected weighting scheme instead of finding and fixing the root cause** — SRM is a signal of a broken data-generating process; no amount of downstream statistical correction reliably fixes a corrupted randomization.
8. **Confusing SRM (count-level mismatch) with covariate imbalance** (composition-level mismatch, e.g., treatment group happens to have more mobile users despite correct counts) — related "is my randomization trustworthy" checks, but diagnostically distinct, checked with different tests.

---

## 10. Common Mistakes / Red Flags — Quick Review

- ❌ Reusing the same salt across multiple experiments
- ❌ Ignoring a statistically significant SRM because the ratio "looks close enough"
- ❌ Bucketing at a point in the pipeline after platform-specific filtering/redirects
- ❌ Using a low-quality/non-uniform hash function
- ❌ Running SRM checks at the standard α=0.05 instead of a stricter threshold
- ❌ Treating "no SRM" as a full guarantee the randomization was clean (still need covariate balance checks)
- ✅ Use fine-grained buckets (100–1000) to support gradual rollouts without re-randomizing existing users
- ✅ Always run an SRM check before interpreting any experiment's primary metric result
- ✅ When SRM fires, segment by platform/device/geo/time to localize the root cause

---

## 11. Famous Interview Q&A

**Q: Your experiment shows a strong positive lift on the primary metric, but you also notice an SRM (52/48 split instead of 50/50). Should you trust the lift?**
A: No — an SRM invalidates the experiment's results until root-caused, regardless of how compelling the lift looks. A skewed split usually means something is systematically different about which users end up in each arm (not just noise), which means the two arms are no longer comparable — the observed "lift" could be entirely an artifact of whichever biased subpopulation ended up over-represented in one arm, rather than a real treatment effect. The right move is to pause interpretation, investigate the root cause (logging gaps, platform-specific bugs, bot traffic), and rerun or fix before drawing conclusions.

**Q: Why do you need a different salt for every experiment instead of reusing the same user-to-bucket mapping across all experiments?**
A: Reusing the same mapping means a user's treatment/control assignment would be perfectly correlated across every experiment ever run — if User X lands in "treatment" for Experiment 1, they'd also always land in "treatment" for Experiments 2, 3, 4, etc. This creates confounding between concurrent experiments: any effect you observe could be entangled with effects from other experiments the same users are simultaneously exposed to. A unique salt per experiment makes bucket assignments independent across experiments, which is what allows running many experiments concurrently without them contaminating each other.

**Q: An SRM check flags significance at p < 0.001, but your team argues "the actual ratio is 50.4/49.6, that's basically nothing — let's ignore it." How do you respond?**
A: I'd push back — SRM tests are typically run at a stricter threshold precisely because with large sample sizes, even a very small but *systematic* bias will trigger significance, and the concern isn't the size of the ratio but what caused it. A small, real bias (as opposed to noise) means there's a genuine, non-random reason certain users end up in one arm over another — and that same underlying mechanism could easily be correlated with the outcome metric too, silently biasing the effect estimate in ways that are hard to detect just by eyeballing the ratio. I'd want to at least identify the root cause and confirm it's unrelated to the outcome metric before dismissing it, rather than assuming a small percentage difference is automatically harmless.

**Q: You're rolling out a new feature gradually — 1% of traffic, then 10%, then 50%. How does your bucketing scheme need to support this without disrupting users already in the test?**
A: Using a fine-grained bucket range (e.g., 1000 buckets via `hash(user_id + salt) mod 1000`) lets you expand the treatment population by simply reassigning additional buckets to treatment (e.g., buckets 0-9 → treatment at 1%, then 0-99 → treatment at 10%), without changing which bucket any individual user falls into. Since the underlying hash-to-bucket mapping for each user never changes, everyone who was already in treatment stays in treatment as you ramp up — you're just expanding the boundary of which buckets count as "treatment," not re-randomizing anyone.

**Q: What is SRM, and why is checking for it typically considered more urgent than analyzing the actual treatment effect?**
A: SRM is a statistically significant deviation between the intended and observed randomization ratio. It's more urgent than the treatment-effect analysis because it's a *precondition* for that analysis being meaningful at all — if the randomization itself is broken, the two arms are no longer guaranteed comparable in expectation, so any downstream lift number (however "significant") could simply reflect whichever biased subpopulation got over-represented in one arm, not a real treatment effect. Checking SRM first is checking the foundation before trusting anything built on top of it.

**Q: Compute the chi-square statistic for an experiment with 40,600 users in treatment and 39,400 in control (intended 50/50 split), and determine if this represents a significant SRM at the strict p<0.001 threshold.**
A: Total n = 80,000; expected = 40,000 each. $\chi^2 = \frac{(40{,}600-40{,}000)^2}{40{,}000} + \frac{(39{,}400-40{,}000)^2}{40{,}000} = \frac{600^2}{40{,}000}\times2 = 9+9=18$. Since 18 > 10.83 (critical χ² at p=0.001, 1 df), this **is** a significant SRM even at the strict threshold — worth investigating before trusting any metric from this experiment.

**Q: Your experiment shows no SRM (counts are balanced as expected), but you're still not 100% sure randomization worked correctly. What additional diagnostic would you run?**
A: I'd run a **covariate/composition balance check** — comparing the distribution of pre-experiment characteristics (e.g., prior engagement level, platform, geography, account age) between the two arms. SRM only catches count-level imbalance; it's possible for counts to match exactly while the underlying users differ systematically for reasons other than random assignment (e.g., a subtle non-random reassignment bug that happens to preserve overall counts but skews composition). Balance checks catch this distinct failure mode that SRM cannot.

---

## 12. L5-Differentiating Talking Points

- Immediately stating "the first thing I'd check before trusting ANY metric result is whether there's an SRM" as your opening move on any experiment-analysis question signals real practitioner instinct, not textbook knowledge — many interviewers specifically listen for whether you volunteer this unprompted.
- Being able to compute the chi-square statistic live, and correctly conclude that a "small-looking" imbalance can be a severe, statistically overwhelming SRM at large sample sizes, shows quantitative fluency plus the right intuition about when small differences matter.
- Proposing a debugging workflow (segment SRM by platform/device/geo/time to localize root cause) rather than just "the experiment is broken, restart it" shows you think about SRM as an actionable engineering signal, not just a stop sign.
- Distinguishing SRM (count mismatch) from covariate/composition imbalance (a related but separate diagnostic) shows precision in a space where interviewers often deliberately probe for conflation.
- Explaining *why* salts and fine-grained buckets exist (not just that they exist) — cross-experiment independence, and re-randomization-free traffic ramping — shows you understand the engineering underneath the statistics, not just the formulas.

---

## 13. Comprehension Check (Self-Test)

1. What is SRM, and why is checking for it typically considered more urgent than analyzing the actual treatment effect?
2. Compute the chi-square statistic for an experiment with 40,600 users in treatment and 39,400 in control (intended 50/50 split), and determine if this represents a significant SRM at the strict p<0.001 threshold.
3. Give two distinct root-cause mechanisms that could produce SRM in a real production system.
4. Why do many experimentation platforms use p<0.001 rather than p<0.05 as the SRM significance threshold?
5. Your experiment shows no SRM (counts are balanced as expected), but you're still not 100% sure randomization worked correctly. What additional diagnostic would you run, and what would it check for that SRM doesn't?
6. Why does hashing need to be deterministic, and why does each experiment need its own salt?
7. How does fine-grained bucketing (e.g., 1000 buckets) support a gradual traffic ramp (1% → 10% → 50%) without re-randomizing users already in the experiment?
8. A team wants to "fix" a significant SRM by re-weighting the analysis to account for the imbalance instead of investigating the root cause. What's wrong with this approach?

---
*This tutorial merges two chapters — one on randomization mechanics (hashing, bucketing, salting, gradual rollouts) with a brief SRM introduction, and one that is a full deep-dive specifically on SRM (root causes, the chi-square test, a small-but-severe-imbalance worked example, and production/debugging guidance). All SRM content from both sources — including all three worked examples, the full root-cause list, and both Q&A sets — has been consolidated into unified SRM sections (4–11) alongside the hashing/bucketing mechanics that SRM checks are built on top of.*
