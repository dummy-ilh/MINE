# Chapter 16: Sequential Testing & the Peeking Problem (Always-Valid P-Values)

## 1. Definition

**The peeking problem:** repeatedly checking a test's p-value before the pre-planned sample size/duration is reached, and stopping early the moment it crosses the significance threshold, inflates the true Type I error rate far beyond the nominal α — even though each individual "look" uses a technically correct p-value formula.

**Sequential testing:** a family of statistical methods specifically designed to allow valid, repeated interim analysis of accumulating data *without* inflating the false positive rate — using techniques like alpha-spending functions or **always-valid p-values**, which remain statistically valid no matter how many times (or when) you look at them.

## 2. Layman Explanation

Imagine flipping a fair coin and checking after every single flip whether you've gotten, say, 60% or more heads so far. Early on, with only a few flips, wild swings are common — you might hit 60%+ heads purely by chance in the first 5 flips, even though the coin is perfectly fair. If you stop the moment you see that streak and declare "this coin is biased!", you're much more likely to be wrong than if you'd committed to flipping it 1,000 times and checking only at the end.

This is exactly what happens when a team checks their A/B test dashboard every single day and stops the test the moment the p-value dips below 0.05 — even though the formal p-value calculation assumes you decided your sample size in advance and only looked once. Peeking repeatedly and stopping at the first "significant" moment is like a gambler who keeps playing until they're randomly ahead, then declares themselves a winning strategy — the early "win" is often just noise, not a real signal.

Sequential testing methods are the proper, honest version of "checking as you go" — they mathematically account for the fact that you're going to peek multiple times, adjusting the bar so that your overall false-positive rate stays at the level you intended, no matter how many times you check.

## 3. Formal Explanation

**Why peeking inflates Type I error:**

A standard p-value calculation assumes a single, pre-specified sample size — the α=0.05 threshold guarantees a 5% false positive rate *for that one look*. But if you check the p-value at multiple points during data collection and stop as soon as it first crosses 0.05, you're effectively giving the null hypothesis multiple "chances" to randomly dip below the threshold. Under repeated peeking with no correction, the true Type I error rate can balloon well past 5% — sometimes to 20-30% or higher depending on how many times you peek, even though the null is true and there's no real effect.

**Formal explanation via Brownian motion / random walk intuition:**
The running p-value over time behaves like a random walk under the null hypothesis — and random walks are known to cross any fixed threshold with probability approaching 1 given enough time/looks, even with no true drift. This is why "checking until significant" is fundamentally different from "checking once at a pre-specified time."

**Always-valid p-values / sequential testing solutions:**

1. **Alpha-spending functions (e.g., O'Brien-Fleming, Pocock boundaries):** pre-specify a small number of interim analysis points and allocate a "budget" of α across them, using progressively stricter thresholds at earlier looks so the *cumulative* false-positive rate across all looks stays at the target α.
2. **Always-valid p-values (based on sequential probability ratio tests / mixture martingales):** a more modern approach (used by companies like Optimizely, and grounded in work by Johari, Koomen, Pekelis, Walsh) that allows *continuous* monitoring — you can check the p-value at literally any point, any number of times, and stop whenever you want, while the reported p-value remains statistically valid throughout. This works by constructing a test statistic that is a martingale under the null, guaranteeing the error rate stays bounded regardless of the stopping rule used.

**Key tradeoff:**
Sequential testing methods that allow valid early stopping generally require slightly more total sample size to achieve the same power as a fixed-horizon test, in exchange for the flexibility of being able to stop early when a strong effect emerges (saving time in the common case) or stop for futility (saving traffic on a test that's clearly not going anywhere).

## 4. Levers — What Controls It, What Moves It

**Number of interim looks**
- More frequent peeking without correction inflates Type I error more severely — checking daily over an 8-week test is far riskier (in terms of false-positive inflation) than checking only once at the pre-planned end.

**Pre-specification of looks (for alpha-spending methods)**
- Alpha-spending approaches require you to decide in advance how many interim looks you'll take and roughly when — deviating from this plan (adding extra unplanned looks) reintroduces the same inflation problem these methods are designed to prevent.

**Choice of method**
- Alpha-spending functions are simpler to implement but require committing to a fixed number/timing of looks upfront; always-valid p-values (martingale-based) offer more flexibility (literally continuous monitoring) but are more complex to implement correctly and are typically provided via specialized experimentation platforms rather than hand-rolled.

**Organizational culture around "peeking"**
- Teams that informally check dashboards constantly and stop tests opportunistically without any correction are the most exposed to this problem — the fix isn't "don't look at your dashboard," it's "use a method designed to allow looking safely," since telling people not to look at all is rarely realistic in practice.

## 5. Worked Example

Suppose a team runs a true null-effect experiment (no real difference between arms) and checks the p-value once per day over a 20-day test, stopping the moment p < 0.05 is observed, with no correction applied.

Simulation studies of this exact scenario (well-documented in the experimentation literature, e.g., from Optimizely's and Microsoft's published work) show that under this "peek daily, stop at first significance" behavior, the true probability of a false positive across the 20-day window is not 5% — it's commonly in the range of **20-30%+**, depending on the correlation structure of daily peeks. 

To make this concrete with a simplified illustration: if each daily peek were treated as roughly independent (a simplification, since consecutive days' p-values are actually correlated, which somewhat tempers the inflation but doesn't eliminate it), the probability of NOT seeing any false positive across 20 independent looks at 5% each would be (0.95)²⁰ ≈ 0.358, meaning the probability of AT LEAST ONE false positive would be roughly 1 - 0.358 ≈ **64%** — dramatically higher than the intended 5%. Real-world sequential p-values are correlated (not independent), which reduces this from the naive 64% figure, but published empirical studies still consistently find inflation into the 20-30%+ range for daily peeking over multi-week tests — underscoring why this is a serious, not theoretical, problem.

## 6. Famous Q&A (Google / Apple style)

**Q: Your team checks the experiment dashboard every day and plans to stop the test as soon as the p-value crosses 0.05. What's wrong with this approach, and what would you recommend instead?**
A: This inflates the true Type I error rate well beyond the intended 5% — because you're giving the null hypothesis many "chances" to randomly dip below the threshold at some point during the monitoring period, even when there's no real effect. Published studies on this exact behavior (daily peeking, stop at first significance) show the true false-positive rate can commonly land in the 20-30%+ range rather than 5%. I'd recommend either committing to a single pre-planned analysis point (no interim peeking influencing the stop decision), or better, adopting a proper sequential testing method — like an alpha-spending boundary or an always-valid p-value approach — that's specifically designed to let you monitor results as often as you want while keeping the true error rate controlled at your target level.

**Q: What's the difference between an alpha-spending approach and an "always-valid" p-value approach to sequential testing?**
A: Alpha-spending methods (like O'Brien-Fleming boundaries) require you to pre-specify a fixed number of interim analysis points in advance, and they allocate a portion of your total α budget to each look, using stricter thresholds early on so the cumulative error rate stays controlled — but you can't add extra unplanned looks without violating the guarantee. Always-valid p-values, based on constructing a martingale test statistic, are more flexible — they remain statistically valid no matter how many times or when you check, including completely unplanned, continuous monitoring, at the cost of somewhat more implementation complexity and, similar to alpha-spending, some efficiency loss (slightly larger sample size needed) compared to a single fixed-horizon test.

**Q: A stakeholder asks, "if peeking is so risky, why not just tell everyone never to look at the dashboard until the test ends?" How would you respond?**
A: In principle that would solve the statistical problem, but it's not realistic — teams need visibility into ongoing experiments for practical reasons, like catching a broken feature causing serious harm (a guardrail regression) as early as possible, or reallocating traffic if something looks badly wrong. Banning all monitoring trades one risk (inflated false positives from informal peeking) for another (failing to catch real harm quickly). The better solution is adopting a sequential testing method that's specifically built to allow safe, ongoing monitoring — giving teams the operational visibility they need without breaking the statistical guarantees.

**Q: Your platform doesn't support always-valid p-values, but leadership wants the option to stop tests early if results look strong. How would you implement this without inflating false positives?**
A: I'd use a pre-specified alpha-spending approach — for example, decide upfront that you'll look at the data at 3 fixed points (say, 25%, 50%, and 100% of planned sample size), and use a boundary function like O'Brien-Fleming that requires a very strict p-value threshold at the earliest look (e.g., p < 0.0001 at 25% completion) with progressively more lenient thresholds at later looks, such that the total false-positive budget across all 3 looks still sums to your target 5%. This gives leadership the ability to stop early on very strong, clear-cut results, while keeping the overall Type I error rate honest — the key discipline is committing to the number and timing of looks in advance and not deviating from that plan.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Checking a test's p-value repeatedly and stopping the moment it crosses 0.05, without any sequential-testing correction
- ❌ Assuming "we only peeked a couple of times, that's probably fine" — even a handful of unplanned peeks meaningfully inflates the true error rate
- ❌ Adding extra, unplanned interim looks to an alpha-spending design after the fact — this breaks the method's guarantee
- ❌ Conflating "monitoring for guardrail harm" (which is legitimate and should happen continuously) with "monitoring the primary metric to decide when to stop for a win" (which requires a proper sequential method)
- ✅ Do: use always-valid p-values or a pre-specified alpha-spending plan if early stopping flexibility is genuinely needed
- ✅ Do: separate "safety monitoring" (checking guardrails continuously is fine and encouraged) from "efficacy stopping" (deciding to end the test based on the primary metric, which needs a proper sequential method)

---
*Next: Chapter 17 — Network Effects / Interference (Cluster Randomization, Switchback Tests).*
