# Chapter 11: Novelty/Primacy Effects & Test Duration

## 1. Definition

- **Novelty effect:** a temporary boost (or drop) in a metric caused simply by a change being *new* — users click, explore, or engage differently just because something looks different, independent of whether the change is actually better. This effect typically decays over time as users habituate.
- **Primacy effect:** the opposite pattern — a temporary *underperformance* immediately after launch because existing users are anchored to old habits/muscle memory and resist or struggle with the change, even if it's genuinely better long-term. This also typically fades as users adapt.

Both are threats to validity because they mean the treatment effect measured in the first few days of a test may not represent the *steady-state* effect you actually care about for a permanent launch decision.

## 2. Layman Explanation

Imagine a restaurant redesigns its menu. In the first week, regulars might order something new just because it's novel and catches their eye — sales of that new dish spike, but not because it's actually a better dish, just because it's unfamiliar and interesting. That's a novelty effect: a temporary bump that has nothing to do with the change being genuinely good.

Now imagine instead the restaurant rearranges the entire menu layout. Regulars who knew exactly where their favorite dish was now have to search for it, feel annoyed, and some just order the first familiar thing they recognize instead. Sales of the "better" featured item might actually look worse for the first few weeks purely because people are frustrated by the change, not because it's a bad idea. That's a primacy effect: a temporary underperformance driven by resistance to change, which fades once people adjust.

Test duration matters because if you only measure for 3 days, you might completely misjudge which of these you're seeing — and ship (or kill) something based on a transient reaction rather than the effect that will actually persist.

## 3. Formal Explanation

**Detecting novelty/primacy effects:**
The standard diagnostic is plotting the treatment effect (difference between arms) *over time* within the experiment, segmented by day or week, rather than just looking at the pooled average effect across the whole test window. 

- A novelty effect shows up as: large effect on day 1-3, decaying toward a smaller (or even negative) steady-state effect by the end of the test.
- A primacy effect shows up as: negative or muted effect early on, improving toward a larger steady-state effect as the test continues.

**New users vs. existing users as a diagnostic:**
Segmenting by user tenure (new users who have no prior habits vs. existing users with established habits) is one of the most useful decompositions — novelty/primacy effects are driven by *existing* users adjusting to change, so if the effect is stable for new users but volatire/decaying for existing users, that's strong confirmatory evidence.

**Implication for test duration:**
The test needs to run long enough for the novelty/primacy curve to plateau before the measured average effect can be trusted as representative of the long-run steady state. This is distinct from (but related to) the sample-size-driven duration calculation in Chapter 9 — you may have enough *statistical power* after 3 days, but still need to run longer because the *effect itself* hasn't stabilized yet.

**Formal approach — "cohort" or "time-since-exposure" analysis:**
Rather than comparing arms on calendar day, compare arms based on days-since-first-exposure for each user — this normalizes for the fact that users enter the experiment on different calendar days, letting you see the true adaptation curve per user.

## 4. Levers — What Controls It, What Moves It

**Magnitude of the UI/UX change**
- Larger, more visually disruptive changes (full redesigns) tend to produce stronger novelty/primacy effects than subtle changes (small copy tweaks), simply because there's more "new" for users to react to or adjust around.

**User tenure mix**
- A test population with more new users (who have no old habits to unlearn) will show weaker primacy effects than a population dominated by long-tenured power users with deeply ingrained habits.

**Test duration**
- The primary lever to let novelty/primacy effects "burn off" — but this trades off against the cost of running experiments longer (opportunity cost, more traffic tied up, slower iteration speed for the team).

**Frequency of exposure**
- Features encountered daily (e.g., a home feed layout) typically stabilize faster than features encountered rarely (e.g., an annual billing flow) simply because users get more repeated exposure per unit time to adapt.

## 5. Worked Example

Suppose you launch a redesigned navigation menu and track daily lift in "successful task completion rate" (treatment minus control), by day:

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

The lift clearly decays from +8.5% down to roughly +1.6-1.7%, then flattens out around day 7-8. This is a textbook novelty effect. If you'd stopped the test after 2 days and used the pooled average (~7.25%), you'd have massively overestimated the true long-run effect. The correct read here is: the feature does have a genuine positive steady-state effect (~1.6-1.7%), but you should report and make the ship decision based on the plateaued value, not the early inflated numbers — and you'd want to confirm the curve has actually flattened (not still decaying) before locking in that number.

## 6. Famous Q&A (Google / Apple style)

**Q: Your test shows a huge +10% lift on day 1, but by day 10 it's down to +1%. What's happening, and what would you report as the "true" effect?**
A: This pattern is a classic novelty effect — the initial spike likely reflects users engaging with the change simply because it's new and different, not because it's genuinely better, and that curiosity-driven behavior fades as users habituate. I'd report the plateaued, steady-state value (the ~1% seen by day 10) as the more trustworthy estimate of the long-run effect, but only after confirming the curve has actually flattened rather than still trending down — I'd want to see a few more days of stability before locking that number in as the ship decision basis.

**Q: A redesign shows a -5% dip in the first three days but climbs back to neutral by day 10. Should you kill the feature based on the early negative result?**
A: Not necessarily — this looks like a primacy effect, where existing users are temporarily thrown off by the change and their performance dips before they adapt. Killing the feature based on the first 3 days risks discarding something that could be neutral or even positive in steady state. I'd want to let the test run long enough to see if the metric fully recovers and stabilizes, and specifically check whether the dip is concentrated in long-tenured users (who have old habits to unlearn) versus new users (who'd show no such dip, since they have nothing to "unlearn") — that segmentation would help confirm whether this is truly a primacy effect versus a genuine regression.

**Q: How would you decide how long to run a test to account for novelty effects, rather than just using the sample-size-based duration from a power calculation?**
A: I'd treat these as two separate, additive constraints on duration: the power calculation gives you the *minimum* duration needed to reach statistical significance given your MDE and traffic, while the novelty/primacy consideration gives you a *separate* minimum duration needed for the treatment effect itself to stabilize. The test should run for at least the longer of the two. In practice, I'd look at day-over-day lift trends during the test (or historical data from similar past launches on this same surface) to estimate how many days it typically takes for novelty effects to plateau, and use that as a floor on duration regardless of what the power calculation alone suggests.

**Q: Why might segmenting your test results by new vs. existing users be more informative than looking at the pooled average when diagnosing novelty effects?**
A: Novelty and primacy effects are fundamentally about *existing users reacting to change* — new users have no prior habits or expectations to be surprised by, so if you see a decaying/increasing effect curve concentrated specifically in existing users while new users show a flat, stable effect from day one, that's strong confirmatory evidence you're looking at a genuine novelty/primacy pattern rather than, say, a seasonal confound or an unrelated external factor that would affect both segments similarly.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Using the pooled average effect across the whole test window without checking the day-by-day trend first
- ❌ Stopping a test early because day-1/day-2 results look great (or terrible) without confirming stabilization
- ❌ Assuming all effects are "novelty" — a real regression can look identical to a primacy effect early on; segmentation by tenure is needed to tell them apart
- ❌ Ignoring test duration implications when a redesign is visually large — bigger changes need more time to stabilize, not the standard default duration
- ✅ Do: plot lift by day (or by days-since-first-exposure) before trusting any pooled number
- ✅ Do: segment new vs. existing users to help distinguish novelty/primacy from a genuine, stable effect

---
*Next: Chapter 12 — Choosing the Right Test (t-test, z-test, chi-square, Mann-Whitney).*
