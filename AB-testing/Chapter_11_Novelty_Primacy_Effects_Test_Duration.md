# Chapter 11: Novelty/Primacy Effects & Test Duration

---

## 1. Intuition

An experiment that ran for exactly 7 days and showed a strong positive effect might be measuring something completely different from what you actually care about: **the effect of change itself, rather than the effect of the change being genuinely better.** Users often react to *any* change — positively (curiosity, novelty-seeking) or negatively (friction, unfamiliarity) — independent of whether the change is actually an improvement. These transient reactions fade as users adapt, and the metric you observe in week 1 can look nothing like the metric you'd observe in week 8.

### Layman analogy
Imagine a restaurant redesigns its menu. In the first week, regulars might order something new just because it's novel and catches their eye — sales of that new dish spike, but not because it's actually a better dish, just because it's unfamiliar and interesting. That's a **novelty effect**: a temporary bump that has nothing to do with the change being genuinely good.

Now imagine instead the restaurant rearranges the entire menu layout. Regulars who knew exactly where their favorite dish was now have to search for it, feel annoyed, and some just order the first familiar thing they recognize instead. Sales of the "better" featured item might actually look worse for the first few weeks purely because people are frustrated by the change, not because it's a bad idea. That's a **primacy effect**: a temporary underperformance driven by resistance to change, which fades once people adjust.

Test duration matters because if you only measure for 3 days, you might completely misjudge which of these you're seeing — and ship (or kill) something based on a transient reaction rather than the effect that will actually persist.

---

## 2. Definitions

- **Novelty effect**: treatment looks *better* than it truly is early on, because users are curious/excited about the new thing — they click, explore, or engage differently just because something looks different, independent of whether the change is actually better. This excitement fades over time as users habituate; the true steady-state effect is smaller than what you initially observe (or could even be negative once novelty wears off).

- **Primacy effect** (sometimes used with the more specific meaning: *the cost of unfamiliarity*): treatment looks *worse* than it truly is early on, because existing users are anchored to old habits/muscle memory and are confused, slowed down, or resistant to an unfamiliar interface — even if it's genuinely better long-term. This friction fades as they learn/adapt; the true steady-state effect is better (or less negative) than what you initially observe.

Both are threats to validity because the treatment effect measured in the first few days of a test may not represent the **steady-state effect** you actually care about for a permanent launch decision.

---

## 3. Why This Matters for Ship Decisions

If you only run a short experiment and ship (or kill) based on an unstabilized early read, you risk two opposite failure modes:

1. **Shipping a change that looked good but regresses to a neutral or negative long-run effect** once novelty wears off — you've made a worse long-term decision based on a misleadingly positive short-term signal.
2. **Killing a change that looked bad early (primacy/learning cost) but would have become genuinely good** once users adapted — you've thrown away real long-term value because you didn't wait long enough to see past the initial friction.

Both failure modes point to the same fix: **look at the time trend of the treatment effect, not just the pooled average over the whole experiment window**, to distinguish "this is a transient reaction" from "this is the true steady-state effect."

---

## 4. Detection Methods

**Method 1 — Plot the treatment effect by day/week (time-series decomposition).** Instead of reporting a single pooled effect size for the whole experiment duration, compute the treatment effect separately for each day (or week) and plot the trend.
- A **novelty effect** shows up as: a large early effect that **decays toward a smaller, stable asymptote**.
- A **primacy effect** shows up as: a negative or muted early effect that **grows toward a larger, stable asymptote**.

**Method 2 — Cohort-based analysis (new users vs. existing users).** Segment by whether a user is new to the product/feature versus an existing user who's had time to adapt. Novelty/primacy effects are fundamentally about *existing users reacting to change* — new users have no prior habits or expectations to be surprised by. So:
- If the effect differs substantially between "just exposed" cohorts and "exposed for several weeks" cohorts, that's direct evidence of a novelty/primacy dynamic rather than a stable causal effect.
- If the effect is stable for new users from day one but volatile/decaying specifically for existing users, that's strong confirmatory evidence you're looking at a genuine novelty/primacy pattern — rather than, say, a seasonal confound or an external factor that would affect both segments similarly.
- Conversely, a real regression can look identical to a primacy effect in the early days — segmentation by tenure is what lets you tell them apart, since a true regression should show up in new users too.

**Method 3 — Holdout / long-running control comparison.** Keep a small, permanent holdout group (much longer duration than the standard experiment) that never receives treatment, allowing you to observe the treatment's effect over a much longer horizon on a rolling basis — well past the point where any initial novelty/primacy dynamics would have faded.

**Method 4 — Second-exposure or repeated-exposure experiments.** Specifically for novelty, some teams run a "re-expose the same treatment to a fresh randomized sample after the initial novelty period has plausibly worn off in the broader population" design, to isolate the steady-state effect from the initial-exposure effect.

**Method 5 — Time-since-exposure ("cohort") analysis.** Rather than comparing arms by calendar day, compare arms based on **days-since-first-exposure** for each user. This normalizes for the fact that users enter the experiment on different calendar days, letting you see the true per-user adaptation curve rather than a curve muddied by staggered entry.

---

## 5. Worked Examples

### Example A — Onboarding tutorial (novelty effect)

Daily treatment effect on Day-1 retention, computed separately per day of the experiment:

| Experiment Day | Treatment Effect (pp) |
|---|---|
| Day 1-2 | +4.5pp |
| Day 3-4 | +3.1pp |
| Day 5-7 | +2.0pp |
| Day 8-10 | +1.6pp |
| Day 11-14 | +1.4pp |
| Day 15-21 | +1.3pp |

**Interpretation**: a textbook novelty effect signature — a large initial bump (+4.5pp) that decays and stabilizes around +1.3–1.4pp by week 3. If the experiment had only run for 4 days and been analyzed with a single pooled average, you'd have reported something like "+3.8pp lift" — nearly **3x the true steady-state effect** of ~+1.3pp. Shipping a broader rollout decision based on the 4-day pooled number would set unrealistic expectations and potentially misallocate resources based on an effect that was never going to persist.

**The correct read**: the effect does appear to be stabilizing (not decaying all the way to zero), suggesting there IS a genuine, durable improvement — just meaningfully smaller than the early days suggested. Recommendation: extend the experiment (or trust the stabilized week-3 estimate) rather than using the inflated early numbers for planning.

### Example B — Redesigned navigation menu (novelty effect, finer-grained)

Daily lift in "successful task completion rate" (treatment minus control):

| Day | Lift |
|---|---|
| 1 | +8.5% |
| 2 | +6.0% |
| 3 | +4.5% |
| 4 | +3.0% |
| 5 | +2.2% |
| 6 | +1.8% |
| 7 | +1.7% |
| 8 | +1.6% |
| 9 | +1.7% |
| 10 | +1.6% |

The lift decays from +8.5% down to roughly +1.6–1.7%, then flattens out around day 7-8 — another textbook novelty effect. If you'd stopped the test after 2 days and used the pooled average (~7.25%), you'd have massively overestimated the true long-run effect. The correct read: the feature does have a genuine positive steady-state effect (~1.6–1.7%), and you should report and make the ship decision based on the plateaued value, not the early inflated numbers — after confirming the curve has actually flattened (not still decaying).

*(Note both examples show the same qualitative shape: a sharp early spike decaying to a smaller-but-real plateau. A primacy effect would show the mirror image — starting negative/muted and climbing to a plateau, as in the Q&A below.)*

---

## 6. Levers — What Controls the Size/Duration of These Effects

**Magnitude of the UI/UX change**
- Larger, more visually disruptive changes (full redesigns) tend to produce stronger novelty/primacy effects than subtle changes (small copy tweaks), simply because there's more "new" for users to react to or adjust around.

**User tenure mix**
- A test population with more new users (who have no old habits to unlearn) will show weaker primacy effects than a population dominated by long-tenured power users with deeply ingrained habits.

**Test duration**
- The primary lever to let novelty/primacy effects "burn off" — but this trades off against the cost of running experiments longer (opportunity cost, more traffic tied up, slower iteration speed for the team).
- Novelty decay rates vary hugely by product surface: a homepage redesign might show novelty effects that fade within days; a fundamental workflow change might take months for users to fully adapt to, meaning a "stable" reading might not even be reached within a typical 2-4 week experiment window. This is a real practical constraint — don't assume a fixed, universal decay timeline.

**Frequency of exposure**
- Features encountered daily (e.g., a home feed layout) typically stabilize faster than features encountered rarely (e.g., an annual billing flow), simply because users get more repeated exposure per unit time to adapt.

---

## 7. Implication for Test Duration

The test needs to run long enough for the novelty/primacy curve to **plateau** before the measured average effect can be trusted as representative of the long-run steady state.

This is a **separate, additive constraint** on duration from the sample-size-driven duration calculation covered elsewhere (i.e., the power-analysis "how many users/days do I need to reach statistical significance" calculation). You may have enough *statistical power* after 3 days, but still need to run longer because the *effect itself* hasn't stabilized yet. Treat these as two independent minimums and **run for at least the longer of the two**:
- Power-based minimum: enough n to detect your MDE at your target α and power.
- Stabilization-based minimum: enough days/weeks for the novelty/primacy curve to plateau, estimated from day-over-day lift trends in the current test or historical data from similar past launches on this same surface.

**Standard duration guidelines**: many companies recommend running experiments for a minimum of 1-2 full weeks (partly to average over day-of-week effects, partly as a baseline defense against novelty risk), and flag results from very short experiments (under a week) as provisional/unreliable specifically due to novelty risk.

---

## 8. Distinguishing Novelty/Primacy from Confounds

**Seasonality/day-of-week confounds**: a day-of-week or day-of-month pattern in the treatment effect could look superficially similar to novelty decay if you're not careful. Make sure your day-by-day decomposition accounts for calendar effects (e.g., compare Tuesday-to-Tuesday, not Day 1 to Day 15 if Day 1 was a weekend) before concluding it's a genuine novelty pattern.

**Genuine regression vs. primacy effect**: a real regression can look identical to a primacy effect in the early days. Segmentation by tenure (Method 2 above) is what lets you tell them apart — a true regression should show up in new users too, whereas a pure primacy effect should be concentrated in existing users with old habits to unlearn.

**The peeking/optional-stopping trap**: the decision to extend an experiment because of a suspected novelty/primacy pattern should itself be **pre-specified as a possible outcome in the experiment design doc**. Deciding "let's just keep running it until the number we like appears" after the fact reintroduces the same p-hacking/peeking risk that governs stopping rules generally — extending a test because you're suspicious the early numbers look "too good" isn't legitimate unless the extension criterion was defined in advance.

---

## 9. Common Mistakes / Red Flags (Quick Review)

- ❌ Using the pooled average effect across the whole test window without checking the day-by-day trend first
- ❌ Stopping a test early because day-1/day-2 results look great (or terrible) without confirming stabilization
- ❌ Assuming all effects are "novelty" — a real regression can look identical to a primacy effect early on; segmentation by tenure is needed to tell them apart
- ❌ Ignoring test-duration implications when a redesign is visually large — bigger changes need more time to stabilize, not the standard default duration
- ❌ Concluding "novelty effect" from a day-by-day pattern without ruling out ordinary seasonality/day-of-week confounds first
- ❌ Extending a test post-hoc "to see if the effect holds up" without a pre-specified rule for doing so
- ✅ Plot lift by day (or by days-since-first-exposure) before trusting any pooled number
- ✅ Segment new vs. existing users to help distinguish novelty/primacy from a genuine, stable effect (or from a genuine regression)

---

## 10. Famous Interview Q&A

**Q: Your test shows a huge +10% lift on day 1, but by day 10 it's down to +1%. What's happening, and what would you report as the "true" effect?**
A: A classic novelty effect — the initial spike likely reflects users engaging with the change simply because it's new and different, not because it's genuinely better, and that curiosity-driven behavior fades as users habituate. I'd report the plateaued, steady-state value (the ~1% seen by day 10) as the more trustworthy estimate of the long-run effect, but only after confirming the curve has actually flattened rather than still trending down — I'd want a few more days of stability before locking that number in as the ship-decision basis.

**Q: A redesign shows a -5% dip in the first three days but climbs back to neutral by day 10. Should you kill the feature based on the early negative result?**
A: Not necessarily — this looks like a primacy effect, where existing users are temporarily thrown off by the change and their performance dips before they adapt. Killing the feature based on the first 3 days risks discarding something that could be neutral or even positive in steady state. I'd want to let the test run long enough to see if the metric fully recovers and stabilizes, and specifically check whether the dip is concentrated in long-tenured users (who have old habits to unlearn) versus new users (who'd show no such dip, since they have nothing to "unlearn") — that segmentation would help confirm whether this is truly a primacy effect versus a genuine regression.

**Q: How would you decide how long to run a test to account for novelty effects, rather than just using the sample-size-based duration from a power calculation?**
A: I'd treat these as two separate, additive constraints on duration: the power calculation gives the *minimum* duration needed to reach statistical significance given your MDE and traffic, while the novelty/primacy consideration gives a *separate* minimum duration needed for the treatment effect itself to stabilize. The test should run for at least the longer of the two. In practice, I'd look at day-over-day lift trends during the test (or historical data from similar past launches on this same surface) to estimate how many days it typically takes for novelty effects to plateau, and use that as a floor on duration regardless of what the power calculation alone suggests.

**Q: Why might segmenting your test results by new vs. existing users be more informative than looking at the pooled average when diagnosing novelty effects?**
A: Novelty and primacy effects are fundamentally about *existing users reacting to change* — new users have no prior habits or expectations to be surprised by. If you see a decaying/increasing effect curve concentrated specifically in existing users while new users show a flat, stable effect from day one, that's strong confirmatory evidence you're looking at a genuine novelty/primacy pattern rather than a seasonal confound or an unrelated external factor that would affect both segments similarly.

**Q: A stakeholder wants to extend a promising-looking experiment "just a bit longer to see if the effect holds up" after seeing early results. How is this different from problematic peeking/optional stopping, and what would make it legitimate vs. illegitimate?**
A: It's legitimate only if the possibility of extending — and the specific criterion for doing so (e.g., "if day-by-day lift hasn't plateaued by day 7, extend to day 14") — was **pre-specified in the experiment design doc** before seeing results. If the decision to extend is being made reactively, purely because the current number looks appealing and someone hopes it'll hold up, that's indistinguishable from optional stopping / p-hacking: you're letting the data you've already seen influence a decision that should have been fixed in advance, which inflates the true false-positive rate of your eventual conclusion.

---

## 11. L5-Differentiating Talking Points

- Proactively proposing a day-by-day (or week-by-week) breakdown of the treatment effect as a standard part of your analysis plan, rather than only reporting a pooled number, shows this is baked into your default workflow, not an afterthought.
- Being able to correctly name and distinguish novelty vs. primacy effects and describe their **opposite** time-trend signatures demonstrates precise vocabulary and understanding, not just "effects can change over time."
- Proposing a long-running holdout group as an infrastructure investment (not just a one-off analysis trick) shows systems-level thinking about how a company should structure its experimentation platform to catch this class of issue routinely.
- Flagging the seasonality-confound risk unprompted, before concluding a pattern is genuinely novelty/primacy, shows rigor and prevents a common false-positive interpretation.
- Explicitly framing test duration as **two independent constraints** (statistical power vs. effect stabilization) and taking the max of the two, rather than treating duration as a single power-calculation output, is a strong signal of practical experience.
- Connecting a post-hoc "let's extend the test" instinct back to the pre-registration/peeking principle — insisting the extension criterion be pre-specified — shows you apply statistical rigor consistently, not just when it's convenient.

---

## 12. Comprehension Check (Self-Test)

1. Describe the opposite time-trend signatures of a novelty effect versus a primacy/learning-cost effect.
2. In the onboarding-tutorial worked example, why would reporting a single 4-day pooled effect size have been misleading, and by roughly what factor?
3. Name at least three detection methods for novelty/primacy effects beyond simply plotting the daily treatment effect.
4. Why is it important to rule out day-of-week seasonality before concluding a day-by-day pattern reflects a genuine novelty effect?
5. A stakeholder wants to extend a promising-looking experiment "just a bit longer to see if the effect holds up" after seeing early results. How would you distinguish this from problematic peeking/optional stopping, and what would make this a legitimate versus illegitimate decision?
6. Why is test duration governed by two separate, additive constraints rather than just the sample-size/power calculation? Name both constraints.
7. How does segmenting by new vs. existing users help distinguish a primacy effect from a genuine regression?
8. A redesign shows a -5% dip in the first three days but recovers to neutral by day 10. What would you recommend, and what additional analysis would strengthen your confidence in that recommendation?

---
*This tutorial merges two chapters on novelty and primacy effects — one framed around definitions, detection methods, and a single onboarding-tutorial worked example; the other framed around a layman restaurant analogy, a navigation-redesign worked example, and duration-planning mechanics. No external chapter references are needed — the interaction with sample-size/power duration and with peeking/pre-registration is explained inline above.*
