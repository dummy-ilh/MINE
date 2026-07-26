# Phase 17: Anomaly & Change Point Detection

This phase is shorter and more practical than Phase 16 — it builds directly on tools you already have (STL from Phase 5, residual diagnostics from Phase 6 Part 5) and asks a genuinely common production question: **how do you automatically flag "something unusual just happened" in a stream of time series data, at scale, without a human staring at every chart?**

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| control limit | a threshold; if a statistic crosses it, we flag an anomaly |
| $\bar{x}$, $s$ | the sample mean and sample standard deviation, computed from a baseline/reference period |
| $S_t$ | the CUSUM statistic at time $t$ — a running, accumulating sum, explained in section 3 |
| $\lambda$ (in EWMA context) | the EWMA smoothing parameter — deliberately the SAME symbol and SAME role as SES's $\alpha$ from Phase 5, just relabeled by convention in this specific literature |
| ESD | Extreme Studentized Deviate — a specific statistical test for outliers, explained in section 5 |
| change point | a specific moment in time where the series' underlying statistical behavior (mean, variance, trend) genuinely shifts |

---

## 2. The simplest approach: Control Charts (Shewhart charts)

**Plain English, built from a baseline period first:** take a stretch of data you believe represents "normal" behavior (a **baseline** or **reference period**), and compute its mean $\bar x$ and standard deviation $s$ (ordinary descriptive statistics — nothing new mathematically). **Then, for every NEW incoming point, flag it as anomalous if it falls outside a fixed number of standard deviations from that baseline mean** — typically $\bar x \pm 3s$ (a "3-sigma" rule, directly recalling the SAME "how many standard deviations away is unusual" logic from Phase 3's ACF confidence bands and Phase 6's prediction intervals, just now applied directly to raw observations rather than to autocorrelations or forecast errors).

**Why 3 standard deviations, specifically (the actual statistical justification, not just convention):** if the data is genuinely Normally distributed (Phase 6, Part 4's Gaussian assumption), only about 0.3% of observations should naturally fall beyond 3 standard deviations from the mean, purely by chance — **so a point crossing this threshold is either a very rare, naturally-occurring extreme value, or genuinely represents something real and different happening — the 3-sigma rule is a DELIBERATE choice balancing "catch genuine anomalies" against "don't constantly cry wolf over ordinary statistical noise," directly the same false-positive/true-positive tradeoff logic underlying the 0.05 significance-level convention used throughout Phase 4's hypothesis testing.**

**The genuine limitation, directly connecting to everything Phases 1-13 warned about:** a plain control chart assumes the data is STATIONARY (Phase 4) and has NO trend or seasonality (Phase 1) — **if you apply a plain control chart DIRECTLY to raw, seasonal data (e.g., retail sales, which are naturally much higher every December), it will falsely flag EVERY December as "anomalous," since December's normal, EXPECTED seasonal spike will exceed the fixed $\bar x\pm 3s$ band built from the whole year's average behavior — a genuine, common, avoidable mistake.**

---

## 3. The fix for trend/seasonality: apply anomaly detection to RESIDUALS, not raw data

**The core, genuinely important principle of this entire phase, stated once, clearly, since everything below is really just a variation on this single idea: NEVER run anomaly detection directly on raw data that has trend or seasonality. Instead, first REMOVE the trend and seasonality (using tools you already fully know — STL from Phase 5, section 2, or a fitted SARIMA model's residuals from Phase 6, Part 5), and THEN apply anomaly detection to what's LEFT OVER (the residuals/remainder), which SHOULD behave like stationary noise if your trend/seasonal removal was done correctly.**

**This directly, explicitly reuses Phase 6, Part 5's entire residual-diagnostic philosophy: a well-fit model's residuals should look like white noise (Phase 2) — and anomaly detection is fundamentally just asking "does THIS SPECIFIC residual look like it genuinely came from that same white-noise process, or does it look like an outlier that doesn't belong?"** **STL-residual-based anomaly detection, specifically, means: run STL (Phase 5, section 2) to split the series into trend + seasonal + remainder, then apply the SIMPLE control-chart logic from section 2 directly to the REMAINDER component only** — since the remainder, by STL's construction, should already have trend and seasonality stripped out, making the plain $\bar x\pm 3s$ (or a more robust variant, section 5) logic appropriate again.

---

## 4. CUSUM (Cumulative Sum): detecting SMALL, SUSTAINED shifts, not just single extreme points

**The genuine limitation of plain control charts (section 2), motivating this new tool:** a control chart is good at catching a single, LARGE, obvious spike — but it's genuinely POOR at detecting a SMALL, gradual, but SUSTAINED shift in the underlying level (e.g., the true mean quietly shifting from 100 to 105 and STAYING there) — **each INDIVIDUAL point after such a shift might still fall comfortably within the $\bar x\pm3s$ band (since the shift itself, 5 units, might be much smaller than $3s$), so no single point ever trips the threshold, even though something genuinely, persistently changed.**

**CUSUM's core idea, in plain English: instead of judging each point in ISOLATION, ACCUMULATE evidence of a shift over time, by keeping a RUNNING SUM of how far each point deviates from the baseline.**
$$
S_t = \max(0,\ S_{t-1} + (x_t - \bar{x}) - k)
$$
**Breaking this down, piece by piece:**
- $(x_t-\bar x)$: today's deviation from the baseline mean — same basic idea as section 2.
- $k$: a small "slack" constant (often set to about half the size of the SMALLEST shift you actually care about detecting) — plain English, "how much deviation do we consider totally normal/expected noise, before we start counting it as evidence of a genuine shift."
- $S_{t-1}$: YESTERDAY's accumulated evidence — **notice this is a RECURSIVE, self-referencing formula, directly in the same spirit as every other recursive formula throughout this course (AR(1), SES, the Kalman filter, GRU's update gate) — today's accumulated evidence builds on top of yesterday's.**
- $\max(0, \cdot)$: **a genuinely important detail — the accumulated sum is never allowed to go NEGATIVE; it RESETS to 0 whenever the running total would otherwise dip below zero.** Plain English: "we're only interested in ACCUMULATING evidence of a sustained UPWARD shift; if the data drifts back down to normal (or below), we don't want old evidence lingering around indefinitely — we reset and start accumulating fresh."

**How you actually use it: flag an alarm/anomaly whenever $S_t$ exceeds some chosen threshold $h$.** **Why does this successfully catch SMALL, SUSTAINED shifts that plain control charts miss?** Because even if EACH individual day's deviation is too small to trip a single-point threshold on its own, **those small deviations keep ACCUMULATING/ADDING UP in $S_t$ across many consecutive days (since the shift is SUSTAINED, not a one-off blip) — eventually, the ACCUMULATED total crosses the threshold $h$, even though no SINGLE day's raw deviation ever would have.** **This is a genuinely different, complementary DETECTION PHILOSOPHY from section 2's control chart: control charts catch single LARGE spikes; CUSUM catches small but PERSISTENT shifts, by design, through its accumulating structure.**

---

## 5. EWMA Control Charts: a smoothed alternative, directly reusing Phase 5's SES

**A closely related, alternative approach to CUSUM, for the exact same "catch a small but sustained shift" goal:** instead of a control chart on the RAW data (section 2), or an accumulating sum (section 4), **run an EXPONENTIALLY WEIGHTED MOVING AVERAGE — literally, mathematically, EXACTLY Phase 5, section 4's Simple Exponential Smoothing formula, just applied here for a monitoring/detection purpose instead of a forecasting purpose:**
$$
z_t = \lambda\, x_t + (1-\lambda)\, z_{t-1}
$$
**Then apply control limits (section 2's logic) to THIS SMOOTHED $z_t$ series, instead of to the raw, noisy $x_t$ directly.** **Why does smoothing help here, connecting directly back to Phase 5's original SES intuition?** Because $z_t$ is a WEIGHTED AVERAGE of RECENT history (Phase 5, section 4's geometric-decay-weighted average derivation), **a sustained shift will cause $z_t$ to GRADUALLY, STEADILY drift toward the NEW level over several periods, and CROSS the control limit once enough of the recent history reflects the new, shifted reality — while a single, one-off random spike gets heavily DAMPENED by the smoothing (since $\lambda<1$ means any single point only contributes a FRACTION of its raw size to $z_t$), making EWMA control charts naturally MORE ROBUST to isolated noise while still SENSITIVE to genuine, sustained shifts — directly the exact same "responsiveness vs. smoothness" tradeoff dial from Phase 5's original $\alpha$ discussion, now being deliberately exploited FOR anomaly detection rather than for forecasting.**

---

## 6. Seasonal Hybrid ESD (S-H-ESD): a genuinely practical, real production technique (Twitter's method)

**New term: ESD (Extreme Studentized Deviate) test.** Plain English, briefly: a formal statistical test (an extension of simple $\bar x\pm 3s$ thresholding, section 2, but MORE ROBUST) for detecting one or more outliers in a dataset, specifically designed to correctly handle the "masking" problem — **the genuine, real issue that if you have MULTIPLE outliers in your data simultaneously, they can artificially inflate the MEAN and STANDARD DEVIATION used for the test, making the outliers themselves appear LESS extreme relative to the (already-distorted) baseline, potentially causing you to MISS them.** **ESD addresses this by iteratively testing for the SINGLE most extreme point, removing it if confirmed as an outlier, RECOMPUTING the mean/standard deviation from the remaining data, and repeating — rather than computing the baseline statistics just once from data that might already be contaminated by the very outliers you're trying to detect.**

**"Seasonal Hybrid" ESD, Twitter's specific real-world extension, built by directly combining tools you already fully know:** (1) first, decompose the series using STL (Phase 5, section 2) into trend + seasonal + remainder, exactly as described in section 3 above; (2) instead of using the ordinary mean/standard deviation for the ESD test (which can be heavily distorted by even a SINGLE large outlier — recall, the ordinary mean is NOT robust to outliers, a genuinely important general statistics fact), **use the MEDIAN and MAD (Median Absolute Deviation — a more outlier-ROBUST alternative to the standard deviation, computed as the median of the absolute deviations from the median, rather than the mean of squared deviations from the mean) as more robust baseline statistics**; (3) run the ESD test (with this robust baseline) on the STL remainder component. **Plain English, the complete picture: S-H-ESD is genuinely just "STL decomposition (Phase 5) + a more outlier-robust version of the control-chart idea (section 2), applied specifically and only to the remainder component (section 3's core principle)" — a real, deployed, practical technique, and yet, once you see it laid out this way, entirely built from pieces you already have.**

---

## 7. Change Point Detection: briefly, a related but DISTINCT question

**A genuinely important distinction worth being precise about, since these two topics are easy to conflate: ANOMALY detection (sections 2-6) asks "is THIS SPECIFIC POINT unusual, relative to the recent, established normal behavior?" CHANGE POINT detection instead asks "at what specific MOMENT did the series' underlying statistical behavior ITSELF genuinely shift, such that 'normal' now means something DIFFERENT going forward?"** **A single anomalous spike (section 2-6's target) typically returns to normal afterward; a genuine change point represents a PERMANENT (or at least long-lasting) shift in the underlying process — e.g., a company changing its pricing strategy might cause a genuine, lasting change point in demand, as opposed to a single unusual promotional day causing a temporary anomalous spike.**

**PELT (Pruned Exact Linear Time) — briefly, by name and core idea:** an efficient algorithm for finding the OPTIMAL set of change points in a series, by searching over possible ways to SEGMENT the series into pieces (each piece assumed to have its own constant, internally-consistent statistical behavior) and MINIMIZING a cost function that BALANCES "how well does this segmentation fit the data" against "how many change points are we introducing" — **directly, structurally the SAME complexity-penalty philosophy as AIC/BIC from Phase 6, Part 4, section 6 (fit quality, penalized by a complexity/parameter-count term) — just applied here to choosing the NUMBER and LOCATION of segments/change points, rather than choosing an ARIMA model's $p,d,q$ orders.**

**Bayesian Online Change Point Detection — briefly, by name and core idea:** a genuinely different, real-time-oriented approach that maintains a continuously-updated PROBABILITY DISTRIBUTION over "how long has it been since the last change point" (called the **run length**), updating this belief with EACH new data point as it streams in — **directly, structurally analogous, in spirit, to the Kalman filter's predict-update loop from Phase 9 (continuously maintaining and updating a belief/estimate as new data arrives, in real time, rather than analyzing a fixed batch of historical data all at once).** You don't need the full derivation of either algorithm for interview purposes — the practical, complete takeaway: **know both names, know they solve the "WHERE did the underlying process genuinely change" question (distinct from "IS this single point unusual"), and know PELT is a batch/offline optimization approach while Bayesian Online Change Point Detection is a real-time/streaming approach — directly paralleling, respectively, the batch AIC/BIC selection philosophy versus the real-time Kalman-filter philosophy you've already fully learned in earlier phases.**

---

## 8. A small numerical illustration: CUSUM catching a sustained shift that a control chart misses

Baseline: $\bar x = 100$, and suppose (for a simple, hand-computable illustration) the "acceptable slack" $k=2$. A genuine, small, sustained shift occurs: the true level quietly shifts to 105 starting at $t=1$, and stays there. Observed data (with some small noise): $x=[104, 106, 103, 107, 105]$.

**Control chart check (section 2), suppose the control limit is $\bar x\pm 15$ (i.e., $85$ to $115$, a deliberately WIDE band for illustration, representing a genuinely large, noisy baseline standard deviation):** every single one of these 5 points (104,106,103,107,105) falls comfortably WITHIN $[85,115]$ — **a plain control chart flags NOTHING, completely missing this genuine, sustained shift, exactly the limitation described in section 4.**

**CUSUM check (section 4), starting $S_0=0$:**
$S_1 = \max(0, 0+(104-100)-2) = \max(0,2)=2$
$S_2 = \max(0,2+(106-100)-2)=\max(0,6)=6$
$S_3=\max(0,6+(103-100)-2)=\max(0,7)=7$
$S_4=\max(0,7+(107-100)-2)=\max(0,12)=12$
$S_5=\max(0,12+(105-100)-2)=\max(0,15)=15$

**Interpretation: $S_t$ climbs STEADILY: $2\to6\to7\to12\to15$ — a clear, unmistakable, ACCUMULATING trend, even though NOT ONE individual observation ever came close to breaching the (deliberately wide) $\pm15$ control-chart band.** If our CUSUM alarm threshold were, say, $h=10$, **we'd have triggered an alert at $t=4$ ($S_4=12>10$) — successfully catching a genuine, small, sustained shift that the plain control chart entirely missed, concretely demonstrating section 4's core claim with real numbers.**

---

## 9. Quick self-check questions

1. Why must you run anomaly detection on the RESIDUAL/remainder of a decomposed series, rather than directly on raw seasonal data?
   *(Answer: raw seasonal data has predictable, expected fluctuations (e.g., a December sales spike) that would trigger false-positive anomaly flags under a plain control chart, since the expected seasonal high would fall well outside a band built from the whole year's average; decomposing first (e.g., via STL) and applying anomaly detection only to the leftover remainder ensures you're only flagging genuinely unexpected deviations, not normal, predictable seasonal patterns.)*
2. Precisely why can CUSUM detect a small, sustained shift that a plain control chart would miss?
   *(Answer: CUSUM accumulates/sums deviations from baseline over time rather than judging each point in isolation; even if each individual day's deviation is too small to cross a single-point threshold, those small deviations keep adding up across many consecutive days (since the shift is sustained), eventually causing the accumulated sum to cross the CUSUM alarm threshold, even though no single day's raw value ever would have.)*
3. What specific "masking" problem does the ESD test's iterative removal procedure solve, and why does it matter?
   *(Answer: if multiple outliers exist simultaneously in the data, they can artificially inflate the baseline mean and standard deviation, making each individual outlier appear less extreme relative to that already-distorted baseline, potentially causing genuine outliers to be missed; ESD fixes this by iteratively identifying and removing the single most extreme point, recomputing the baseline statistics from the remaining (cleaner) data, and repeating.)*
4. What is the precise conceptual difference between anomaly detection and change point detection?
   *(Answer: anomaly detection asks whether a single specific point is unusual relative to established normal behavior, typically expecting the series to return to normal afterward; change point detection instead asks at what moment the series' underlying statistical behavior itself permanently or persistently shifted, such that what counts as "normal" is genuinely different going forward.)*

---

## What's next
Phase 18 moves into **Causal Inference in Time Series** — Google's own CausalImpact method (built directly on the Bayesian Structural Time Series / Kalman filter machinery from Phase 9), the synthetic control method, and a genuinely important practical warning: why ordinary A/B testing statistics can give FALSE CONFIDENCE when applied naively to autocorrelated time series data (directly connecting back to Phase 7's warning about autocorrelated regression errors, now showing up in an experimentation/A-B-testing context).

Say "next" for Phase 18, or ask for more CUSUM/anomaly-detection drilling first.
