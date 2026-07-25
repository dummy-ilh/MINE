# Chapter 18: Time Series Outliers — Seasonal-Hybrid ESD & STL-Residual Detection

## 18.1 Motivation — Why Time Series Needs Its Own Chapter

Every method so far implicitly assumes observations are (at least roughly) exchangeable — order doesn't matter. Time series data breaks this assumption completely: a value's "normalcy" depends on **trend** (is it generally rising?) and **seasonality** (is it a Monday? December? 3am?) — this is exactly the **contextual outlier** category introduced back in Ch.1 §1.3 (the December ice-cream-sales example), now given a full toolkit. Applying Ch.2's Z-score directly to raw time series values would treat a perfectly normal December sales dip as a wild outlier, simply because it's being compared against the annual mean that includes summer's peak.

## 18.2 STL Decomposition — Separating the Three Components

**STL (Seasonal-Trend decomposition using Loess)** splits a time series into three additive components:
$$
Y_t = T_t + S_t + R_t
$$
- $T_t$: the slow-moving **trend** component (extracted via iterative local regression/Loess smoothing)
- $S_t$: the repeating **seasonal** component (e.g., day-of-week or month-of-year pattern)
- $R_t$: the **remainder/residual** — whatever's left after removing trend and seasonality

**Why this matters for outlier detection:** once trend and seasonality are removed, $R_t$ should behave like a roughly stationary, structure-free series — at which point **every univariate method from Chapters 2-5 becomes valid again**, applied to $R_t$ instead of raw $Y_t$. This is the key conceptual move of this entire chapter: **time series outlier detection = decompose first, then apply the classical toolkit to what's left over.**

## 18.3 Seasonal-Hybrid ESD (S-H-ESD)

Twitter's S-H-ESD algorithm (Hochenbaum et al., 2017) combines STL decomposition with the Generalized ESD test from **Chapter 5**, with two important robustness modifications:

**Step 1 — STL decompose** the series into $T_t, S_t, R_t$.

**Step 2 — Apply Generalized ESD to the residual $R_t$**, but with a key substitution: replace the classical mean/SD in the ESD test statistic with **median and MAD** (Chapter 2's robust estimators):
$$
R_i = \frac{|R_t - \text{median}(R)|}{\text{MAD}(R)}
$$
instead of the standard Grubbs'-style $\frac{|x-\bar x|}{s}$. This substitution exists precisely because financial/operational time series residuals often still contain **multiple simultaneous outliers** even after decomposition — and Chapter 2 already established that mean/SD-based statistics can be masked by the very outliers they're trying to detect. Using median/MAD here makes the whole pipeline robust to that masking risk, compounding two separate fixes from earlier chapters (ESD's masking-resistant iterative structure from Ch.5 + MAD's robust scale from Ch.2) into one time-series-specific tool.

**Step 3 — Robust trend handling:** S-H-ESD also uses a piecewise median for long time series (splitting into windows and taking a robust central trend estimate) rather than a single global trend line, to handle trend that itself changes character over a long history — a practical detail addressing the same "don't assume one global scale fits the whole series" theme from Ch.10/Ch.15.

## 18.4 Worked Numerical (Conceptual)

Daily website traffic over 3 weeks, with a strong weekly seasonal pattern (weekends consistently ~40% lower than weekdays) and a mild upward trend. Suppose Tuesday of week 2 shows a genuine anomaly (a server outage cut traffic in half for that day).

**Raw value comparison (wrong approach):** if we naively Z-score the raw traffic numbers against the whole 3-week mean, a normal *Sunday* (already 40% below the weekday-heavy mean) might look nearly as "anomalous" as the truly broken Tuesday — because the mean is dominated by weekday values, exactly the Ch.1 contextual-outlier trap.

**STL-decomposed approach:**
- $T_t$ captures the mild upward trend (e.g., traffic climbing gradually from 10,000 to 11,500 over 3 weeks).
- $S_t$ captures the weekly pattern (e.g., $S_t \approx +1000$ on Tuesdays historically, $S_t\approx -3000$ on Sundays).
- For our anomalous Tuesday, actual traffic might be 5,000, while $T_t+S_t \approx 11,000+1000=12,000$ predicted — so:
$$
R_t = Y_t - (T_t+S_t) = 5000-12000=-7000
$$
- For a normal Sunday in the same week, actual traffic 8,500, $T_t+S_t\approx11,000-3000=8,000$:
$$
R_t = 8500-8000=+500
$$

**Applying median/MAD-based ESD to the residual series:** the broken Tuesday's residual ($-7000$) stands out enormously against the typical residual spread (residuals for normal days cluster near 0, say median$(R)\approx0$, MAD$(R)\approx300$):
$$
\frac{|-7000-0|}{300}\approx23.3
$$
Overwhelmingly flagged. Meanwhile the normal Sunday's residual ($+500$) gives:
$$
\frac{|500-0|}{300}\approx1.67
$$
Comfortably within normal range — correctly **not flagged**, exactly the opposite conclusion a naive raw-value Z-score would have risked reaching.

## 18.5 Point Anomalies vs. Contextual Anomalies vs. Collective Anomalies in Time Series (Revisiting Ch.1's Taxonomy)

- **Point anomaly in a time series:** a single timestamp's residual is extreme (the broken-Tuesday example above) — caught directly by S-H-ESD.
- **Contextual anomaly:** a value that's fine in absolute terms but wrong for its season/trend context (a normal-magnitude value landing on the wrong day) — this is exactly what STL decomposition is designed to expose, by re-centering each observation against its own expected seasonal/trend baseline before scoring.
- **Collective anomaly:** a sustained subsequence that's each-individually-unremarkable but collectively wrong (e.g., traffic flatlining at a suspiciously *constant* value for several hours, each individual value plausible but the *pattern* of zero variation is itself anomalous) — S-H-ESD's point-by-point residual testing does **not** naturally catch this; **changepoint detection** methods (e.g., CUSUM, Bayesian changepoint detection) or explicitly modeling local variance are needed instead, worth naming as a distinct follow-up tool.

## 18.6 Diagnosis: When to Use STL/S-H-ESD

| Condition | Recommendation |
|---|---|
| Strong, regular seasonality (daily/weekly/yearly patterns) | STL decomposition essential first step |
| Trend present and evolving | STL's Loess-based trend extraction handles gradual change well |
| Multiple simultaneous anomalies expected in the residual | Use median/MAD-based ESD (S-H-ESD), not classical mean/SD Grubbs' |
| Anomaly is a sustained pattern-shift, not a single-point spike | S-H-ESD alone insufficient — needs changepoint detection instead |
| Irregular/non-seasonal time series (no repeating pattern) | STL less useful — consider simpler trend-removal (differencing) then apply Ch.2-5 methods directly to the differenced series |
| Multiple independent seasonal cycles (e.g., both daily AND weekly patterns simultaneously) | Requires multi-seasonal decomposition extensions (e.g., MSTL) — plain STL only handles one seasonal period at a time |

## 18.7 Production Considerations
- STL decomposition needs to be recomputed as new data arrives (rolling window) — seasonal patterns and trend can themselves drift over time (e.g., a product's weekly usage pattern changing after a marketing campaign), so periodically refitting decomposition parameters is standard.
- Real-time/streaming anomaly detection on time series often uses a rolling window version of this pipeline (e.g., decompose the trailing N periods, score only the newest point) rather than full-history batch decomposition — a practical latency/accuracy tradeoff.
- Holiday effects and other irregular calendar events (Black Friday, national holidays) violate the clean repeating-seasonality assumption — production systems typically maintain an explicit calendar-effects adjustment layered on top of STL, rather than expecting STL alone to capture known one-off calendar irregularities.
- This whole pipeline is a standard component of production monitoring/alerting systems (e.g., detecting anomalous server metrics, sales dips, traffic drops) — often the very first anomaly-detection system a company builds, preceding more complex ML-based approaches.

## 18.8 Interview Traps
- Applying Z-score or IQR directly to raw seasonal time series data without decomposing first — the single most common mistake, producing exactly the false-positive-on-normal-troughs / false-negative-on-anomalous-peaks problem illustrated in §18.4.
- Using classical mean/SD-based Grubbs'/ESD on the residual series without considering that multiple real anomalies could still mask each other (the same Ch.2/Ch.4 masking risk) — not knowing why S-H-ESD specifically substitutes median/MAD.
- Assuming STL-residual-based detection catches every anomaly type — forgetting that sustained pattern-shifts (collective anomalies) require fundamentally different tools (changepoint detection), not just residual thresholding.
- Not accounting for irregular calendar effects (holidays) separately from regular seasonality — treating every seasonal deviation as either "normal seasonal pattern" or "anomaly" with no third category for "known, expected, one-off calendar effect."

## 18.9 L5-Differentiating Talking Points
- Framing the entire chapter as "decompose first, then hand the residual to the classical toolkit from Chapters 2-5" — this single sentence shows you understand time series outlier detection isn't a wholly separate discipline, but a preprocessing step that makes the rest of the curriculum applicable again.
- Explicitly naming the median/MAD substitution in S-H-ESD as a deliberate fix for the exact masking problem discussed in Ch.2 and Ch.4-5 — again reinforcing the cross-chapter, "each fix addresses a specific named failure mode" narrative running through the whole curriculum.
- Distinguishing collective/pattern-shift anomalies from point anomalies and correctly naming changepoint detection as the appropriate complementary tool — shows awareness of the boundary of this chapter's method, not just its mechanics.

## 18.10 Comprehension Check
1. Explain, using a concrete seasonal example, why applying Z-score directly to raw (non-decomposed) time series data can produce both false positives and false negatives.
2. Why does S-H-ESD substitute median/MAD for mean/SD inside the Generalized ESD test, specifically in the time-series context?
3. Give an example of a collective/pattern-shift time series anomaly that STL-residual-based point testing would likely miss, and name the class of method that would catch it instead.
4. Why does STL decomposition need to be periodically refit rather than computed once and reused indefinitely in a production monitoring system?

---
*Next: Chapter 19 — Treatment Strategies: Winsorizing, Capping, Transformation, and Knowing When NOT to Treat an Outlier.*
