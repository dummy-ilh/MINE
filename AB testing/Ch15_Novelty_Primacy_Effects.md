# Chapter 15: Novelty & Primacy Effects

## 1. Intuition

An experiment that ran for exactly 7 days and showed a strong positive effect might be measuring something completely different from what you actually care about: **the effect of change itself, rather than the effect of the change being genuinely better.** Users often react to *any* change — positively (curiosity, novelty-seeking) or negatively (friction, unfamiliarity) — independent of whether the change is actually an improvement. These transient reactions fade as users adapt, and the metric you observe in week 1 can look nothing like the metric you'd observe in week 8.

The intuition split into two named, opposite-direction phenomena:

- **Novelty effect**: treatment looks *better* than it truly is early on, because users are curious/excited about the new thing, and this excitement fades over time — the true steady-state effect is smaller than what you initially observe (or could even be negative once novelty wears off).
- **Primacy effect** (sometimes used with a more specific meaning: the *cost of unfamiliarity*): treatment looks *worse* than it truly is early on, because users are confused/slowed down by an unfamiliar interface, and this friction fades as they learn/adapt — the true steady-state effect is better (or less negative) than what you initially observe.

## 2. Why This Matters for Ship Decisions

If you only run a short experiment and ship based on an inflated novelty bump, you risk two failure modes:
1. **Shipping a change that looked good but regresses to a neutral or negative long-run effect** once the novelty wears off — you've now made a worse long-term decision based on a misleadingly positive short-term signal.
2. **Killing a change that looked bad early (primacy/learning cost) but would have become genuinely good** once users adapted — you've thrown away real long-term value because you didn't wait long enough to see past the initial friction.

Both failure modes point to the same underlying fix: **you need to look at the time trend of the treatment effect, not just the pooled average over the whole experiment window**, to distinguish "this is a transient reaction" from "this is the true steady-state effect."

## 3. Detection Methods

**Method 1 — Plot the treatment effect by day/week (time-series decomposition)**: instead of reporting a single pooled effect size for the whole experiment duration, compute the treatment effect separately for each day (or week) of the experiment and plot the trend. A novelty effect shows up as a large early effect that **decays toward a smaller, stable asymptote**. A primacy/learning-cost effect shows the opposite: a negative or small early effect that **grows toward a larger, stable asymptote**.

**Method 2 — Cohort-based analysis (new users vs. existing users)**: segment by whether a user is new to the product/feature versus an existing user who's had time to adapt. If the treatment effect differs substantially between "just exposed" cohorts and "exposed for several weeks" cohorts, that's direct evidence of a novelty/primacy dynamic rather than a stable causal effect.

**Method 3 — Holdout/long-running control comparison**: keep a small, permanent holdout group (much longer duration than the standard experiment) that never receives treatment, allowing you to observe the treatment's effect over a much longer horizon on a rolling basis, well past the point where any initial novelty/primacy dynamics would have faded.

**Method 4 — Second-exposure or repeated-exposure experiments**: specifically for novelty, some teams run a "re-expose the same treatment to a fresh randomized sample after the initial novelty period has plausibly worn off in the broader population" design to isolate the steady-state effect from the initial-exposure effect.

## 4. Worked Example

A new onboarding tutorial is tested. Daily treatment effect on Day-1 retention, computed separately per day of the experiment:

| Experiment Day | Treatment Effect (pp) |
|---|---|
| Day 1-2 | +4.5pp |
| Day 3-4 | +3.1pp |
| Day 5-7 | +2.0pp |
| Day 8-10 | +1.6pp |
| Day 11-14 | +1.4pp |
| Day 15-21 | +1.3pp |

**Interpretation**: this is a textbook novelty effect signature — a large initial bump (+4.5pp) that decays and stabilizes around +1.3-1.4pp by week 3. If the experiment had only run for 4 days and been analyzed with a single pooled average, you'd have reported something like "+3.8pp lift" — nearly **3x the true steady-state effect** of ~+1.3pp. Shipping a broader rollout decision based on the 4-day pooled number would set unrealistic expectations and potentially misallocate resources based on an effect that was never going to persist.

**The correct read**: the effect does appear to be stabilizing (not decaying all the way to zero), suggesting there IS a genuine, durable improvement here — just meaningfully smaller than the early days suggested. The recommendation: extend the experiment (or trust the stabilized week-3 estimate) rather than using the inflated early numbers for planning.

## 5. Production Considerations

- **Standard experiment duration guidelines often specifically account for novelty**: many companies recommend running experiments for a minimum of 1-2 full weeks (to average over day-of-week effects, covered implicitly here) and flag results from very short experiments (under a week) as provisional/unreliable specifically due to novelty risk.
- **Novelty decay rates vary hugely by product surface**: a homepage redesign might show novelty effects that fade within days; a fundamental workflow change might take months for users to fully adapt to, meaning a "stable" reading might not even be reached within a typical 2-4 week experiment window — this is a real practical constraint worth acknowledging rather than assuming a fixed universal decay timeline.
- **Distinguish novelty/primacy from seasonality**: a day-of-week or day-of-month pattern in the treatment effect could look superficially similar to novelty decay if you're not careful — make sure your day-by-day decomposition accounts for calendar effects (e.g., compare Tuesday-to-Tuesday, not Day 1 to Day 15 if Day 1 was a weekend) before concluding it's a genuine novelty pattern.
- **The decision to extend an experiment because of a suspected novelty/primacy pattern should itself be pre-specified as a possible outcome in the experiment design doc** — deciding "let's just keep running it until the number we like appears" after the fact reintroduces the same p-hacking/peeking risk flagged in earlier chapters.

## 6. Interview Traps

- **Trap #1**: Reporting a single pooled effect size for the whole experiment without checking the day-by-day trend — this is the single most common way novelty effects go undetected and lead to bad ship decisions.
- **Trap #2**: Confusing novelty effect (looks better than true, fades down) with primacy/learning-cost effect (looks worse than true, fades up) — these are opposite-direction phenomena and mixing them up in an explanation is an easy way to lose credibility.
- **Trap #3**: Concluding "novelty effect" from a day-by-day pattern without ruling out ordinary seasonality/day-of-week confounds first.
- **Trap #4**: Not connecting the practical fix (extend experiment duration, or use a long-running holdout) back to the pre-registration principle from earlier chapters — deciding to extend post-hoc, specifically because the early numbers looked too good and you're suspicious, without a pre-specified rule for doing so, reintroduces exactly the kind of subjective, data-dependent decision-making Chapter 8 and Chapter 9 warned against.

## 7. L5-Differentiating Talking Points

- Proactively proposing a day-by-day (or week-by-week) breakdown of the treatment effect as a standard part of your analysis plan, rather than only reporting a pooled number, shows this is baked into your default workflow, not an afterthought.
- Being able to correctly name and distinguish novelty vs. primacy effects and describe their opposite time-trend signatures demonstrates precise vocabulary and understanding, not just "effects can change over time."
- Proposing a long-running holdout group as infrastructure investment (not just a one-off analysis trick) shows systems-level thinking about how a company should structure its experimentation platform to catch this class of issue routinely.
- Flagging the seasonality-confound risk unprompted, before concluding a pattern is genuinely novelty/primacy, shows rigor and prevents a common false-positive interpretation.

## 8. Comprehension Check

1. Describe the opposite time-trend signatures of a novelty effect versus a primacy/learning-cost effect.
2. In the worked example, why would reporting a single 4-day pooled effect size have been misleading, and by roughly what factor?
3. Name two detection methods for novelty/primacy effects beyond simply plotting the daily treatment effect.
4. Why is it important to rule out day-of-week seasonality before concluding a day-by-day pattern reflects a genuine novelty effect?
5. A stakeholder wants to extend a promising-looking experiment "just a bit longer to see if the effect holds up" after seeing early results. How would you distinguish this from problematic peeking/optional stopping (a concept from an earlier chapter), and what would make this a legitimate versus illegitimate decision?

---
*Next: Chapter 16 — Peeking & Sequential Testing*
