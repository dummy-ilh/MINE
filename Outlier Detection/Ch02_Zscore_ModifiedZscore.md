# Chapter 2: Z-score & Modified Z-score (MAD-Based) Methods

## 2.1 The Z-score

**Intuition first:** imagine you want one universal way to say "how unusual is this value?" that works whether you're looking at ages (typical range 0–100), incomes (typical range in the thousands), or temperatures (typical range maybe -20 to 40). Raw numbers aren't comparable across these — "13 units away from average" means something totally different for age versus income. The Z-score fixes this by asking a scale-free question instead: **"how many standard deviations away from the mean is this point?"** Once you rescale everything into "number of standard deviations," a Z-score of 3 means the same thing (a genuinely rare value) no matter what the original units were.

**Formula, explained piece by piece:**
$$
z_i = \frac{x_i - \bar{x}}{s}
$$

- $x_i - \bar{x}$ is just "how far is this point from the average," in the original units.
- Dividing by $s$ (the standard deviation) converts that raw distance into "how many typical-sized steps away is this," where $s$ = $\sqrt{\frac{1}{n-1}\sum (x_i-\bar{x})^2}$ is itself just "the typical size of a deviation from the mean" (Chapter 1, Section 1.5).
- So $z_i$ literally reads as: *"this point is $z_i$ standard-deviation-sized steps away from typical."*

**Threshold rule:** flag $x_i$ as an outlier if $|z_i| > \tau$, commonly $\tau = 3$ — i.e., "more than 3 typical-sized steps away from the average."

### Where does the number "3" actually come from?

This isn't an arbitrary round number — it comes directly from how a normal (bell-curve) distribution behaves. If you assume your data really is bell-shaped, then the *empirical rule* tells you exactly what fraction of points should naturally fall beyond each number of standard deviations:

- $P(|Z|>1) \approx 31.7\%$ — nearly a third of all points are expected to be more than 1 SD away. Way too common to call "outliers."
- $P(|Z|>2) \approx 4.55\%$ — still roughly 1 in 22 points. Somewhat rare, but still not rare enough to be confident something's wrong.
- $P(|Z|>3) \approx 0.27\%$ — now we're down to about 1 in 370 points. Rare enough that most people are comfortable calling it "suspicious."

So choosing $\tau=3$ is really just choosing to flag the most extreme ~0.27% of data *if the data really is Gaussian* — it's a widely-used convention, not a law of nature. This is exactly the "density collapses to near-zero in the tail" argument from Chapter 1's density framework ($f(x) < \tau$): at $|z|=3$, the Gaussian density has fallen to almost nothing, so the score $1/f(x)$ has become huge.

## 2.2 Worked Numerical — Z-score

Data: `[10, 12, 12, 13, 12, 11, 14, 13, 15, 90]`

**Step 1 — mean:**
$$
\bar{x} = \frac{10+12+12+13+12+11+14+13+15+90}{10} = \frac{202}{10} = 20.2
$$
Notice already: without the 90, the other 9 values average to roughly 12.4 — so the single value 90 has already pulled the "typical" value up to 20.2. This is the mean's lack-of-robustness from Chapter 1, about to cause a problem.

**Step 2 — standard deviation (sample, dividing by $n-1$):**

Deviations from the mean (each value minus 20.2): $-10.2, -8.2, -8.2, -7.2, -8.2, -9.2, -6.2, -7.2, -5.2, 69.8$

Squared deviations: $104.04, 67.24, 67.24, 51.84, 67.24, 84.64, 38.44, 51.84, 27.04, 4872.04$

Sum of squares $= 5431.6$

$$
s = \sqrt{5431.6/9} = \sqrt{603.5} \approx 24.57
$$

Look at that last squared term: $4872.04$ out of a total sum of $5431.6$ — the single point 90 supplies about **90% of the entire variance calculation** all by itself (echoing the quadratic-sensitivity point from Chapter 1). This is exactly why $s$ has ballooned to 24.57, which is much larger than the "typical spread" of the other nine values (which cluster tightly between 10 and 15).

**Step 3 — Z-scores:**
$$
z_{90} = \frac{90-20.2}{24.57} \approx 2.84
$$
$$
z_{10} = \frac{10-20.2}{24.57} \approx -0.42
$$

**The punchline:** even though 90 is visibly, obviously an outlier — nearly **4.5× larger** than the next-highest value (15) — its own Z-score only reaches **2.84**, which falls *below* the standard $\tau=3$ cutoff! The classic Z-score test would let this outlier slip through undetected. This happens precisely because the outlier itself dragged both $\bar{x}$ and $s$ upward — it inflated the very ruler being used to measure it, shrinking its own apparent distance. This is the single most important practical failure mode of the classic Z-score, and it's an excellent numerical to have ready if an interviewer asks "when does Z-score fail?"

## 2.3 Why Z-score Breaks: The Masking Effect

**Masking** (what just happened above): an outlier inflates $s$ (the denominator), which shrinks every $z_i$ — including, ironically, the very outlier's own score — effectively hiding it from detection. The outlier "masks" itself by distorting the measuring stick used to catch it.

**Swamping** (the mirror-image problem): in skewed or heavy-tailed data, a large $s$ driven by a handful of real outliers can make *perfectly normal* points look artificially close to zero (unremarkable) when they should be flagged, or — depending on which direction the mean got dragged — can make some normal points look artificially extreme, generating false alarms on data that was actually fine.

**The shared root cause:** both masking and swamping trace back to one fact — **the Z-score uses non-robust estimators ($\bar{x}$ and $s$) to try to detect the very thing (outliers) that breaks those same estimators.** It's a bit like trying to measure how crooked a ruler is, using that same ruler. This circular weakness is exactly what motivates replacing $\bar{x}$ and $s$ with more robust alternatives — which is the entire point of the Modified Z-score.

## 2.4 Modified Z-score (MAD-based)

**Intuition:** if the problem is that $\bar{x}$ and $s$ get dragged around by the outliers they're supposed to detect, the fix is to swap in a center and a spread measure that *don't* get dragged around. From Chapter 1: the **median** has a breakdown point of ~50% (very robust), versus the mean's breakdown point of $1/n$ (fragile). We need a similarly-robust replacement for $s$ — that's the **Median Absolute Deviation (MAD)**.

**MAD formula, explained:**
$$
\text{MAD} = \text{median}\big(|x_i - \text{median}(x)|\big)
$$

Read this in two steps:
1. First find the median of the whole dataset (a robust "center").
2. Then, for every point, compute *how far it is from that center* (in absolute value — direction doesn't matter, only distance).
3. Finally, take the **median** (not the mean!) of all those distances.

Because you take a median at both stages, a single wild outlier can distort at most one of those "distance from center" values — it can't drag the *median of distances* the way it would drag an *average* of distances. MAD is, in effect, "the typical distance from typical," built entirely out of robust ingredients.

**Modified Z-score:**
$$
M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}
$$
where $\tilde{x}$ is the median. This has exactly the same shape as the ordinary Z-score — "(point − center) / spread" — just with every ingredient swapped for its robust counterpart. Flag $|M_i| > 3.5$ (Iglewicz & Hoaglin's recommended threshold — slightly higher than the classic 3, for reasons explained below).

### Where does the constant 0.6745 come from?

MAD, on its own, isn't measured on the same scale as the standard deviation $\sigma$ — for a standard normal distribution, it turns out that $\text{MAD} \approx 0.6745\,\sigma$ (MAD is naturally *smaller* than $\sigma$ for the same data, because taking a median of absolute deviations discards the stretching effect that far-out points have on $\sigma$). If you just divided by raw MAD, your "Modified Z-scores" would live on a different numeric scale than ordinary Z-scores, making the familiar "3 means rare" intuition inapplicable. Multiplying by $0.6745$ rescales MAD back up to be directly comparable to $\sigma$, so a Modified Z-score of, say, 2 means roughly the same thing as an ordinary Z-score of 2 — just computed more robustly. The recommended cutoff of 3.5 (instead of 3) exists because MAD, while far more robust, is a somewhat less *statistically efficient* estimator (it uses less of the information in the data), so a slightly wider margin is used in practice to avoid over-flagging on normal data.

## 2.5 Worked Numerical — Modified Z-score (same data)

Data: `[10, 12, 12, 13, 12, 11, 14, 13, 15, 90]`

**Step 1 — median:** sort the data → `[10,11,12,12,12,13,13,14,15,90]`. With 10 values, the median is the average of the 5th and 6th sorted values: $(12+13)/2 = 12.5$.

Notice how little the outlier affected this: the median (12.5) is very close to what you'd get from the 9 "normal" values alone, whereas the mean (20.2, from Section 2.2) was dragged noticeably upward. This is the robustness gap in action.

**Step 2 — absolute deviations from the median (12.5):**

$|10-12.5|=2.5$, $|11-12.5|=1.5$, $|12-12.5|=0.5$ (this repeats for all three 12's), $|13-12.5|=0.5$ (repeats for both 13's), $|14-12.5|=1.5$, $|15-12.5|=2.5$, $|90-12.5|=77.5$

Sorted list of these absolute deviations: `[0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 77.5]`

MAD = median of this sorted list = average of the 5th and 6th values = $(0.5+1.5)/2 = 1.0$

Notice the giant value 77.5 sits all alone at the very end of this sorted list — it barely nudges the *median* of the list at all, which is exactly the robustness property MAD is designed to have.

**Step 3 — Modified Z-score for the outlier, 90:**
$$
M_{90} = \frac{0.6745 \times (90 - 12.5)}{1.0} = 0.6745 \times 77.5 \approx 52.3
$$

**The result:** $M_{90} \approx 52.3$, wildly above the $3.5$ threshold — instantly and unambiguously flagged. Compare this to the classic Z-score's **2.84 for the same point**, which was missed entirely. This is the direct numerical proof of why MAD-based detection is strongly preferred whenever you suspect contamination might be present.

**And for a normal point**, e.g. $x=10$:
$$
M_{10} = \frac{0.6745\times(10-12.5)}{1.0} = -1.69
$$
Comfortably within bounds — correctly *not* flagged, exactly as we'd want for a genuinely typical value.

## 2.6 Diagnosis / When Each Method Applies

| Condition | Recommended method | Why |
|---|---|---|
| Data roughly Gaussian, no/few outliers suspected | Classic Z-score | Simple, fast, and there's no contamination to distort $\bar{x}$/$s$ in the first place |
| Suspected outliers/contamination present | Modified Z-score (MAD) | Robust center and spread won't be dragged around by the very points you're trying to catch |
| Small sample sizes ($n<30$) | Prefer MAD or Grubbs' test (Ch.4) | With few points, the sample SD $s$ is itself unstable and easily swung by one value |
| Multiple simultaneous outliers | MAD strongly preferred | Classic Z-score's masking effect gets worse with more outliers (more inflation of $s$); MAD degrades much more gracefully, remaining reliable until close to ~50% contamination |
| Skewed (non-symmetric) distributions | Neither is ideal | A median-based center still helps, but for skewed data consider IQR (Chapter 3) or a distribution-normalizing transformation (e.g., log transform) first |

## 2.7 Production Considerations

- MAD requires access to the median of the data, and computing a true median means either storing the full sorted array or maintaining a structure that can report it — this is inherently costlier to keep updated in a streaming system than the classic Z-score's mean/variance, which can be updated incrementally point-by-point using **Welford's algorithm** (no need to re-sort or re-store everything). For high-throughput streaming systems, exact MAD is often approximated instead using structures like **t-digest** or **KLL sketches**, which estimate quantiles (and therefore MAD-like statistics) without needing a full sort of all historical data.
- Z-score thresholds computed once during training and then hard-coded into a live serving pipeline are a classic source of **train/serve skew** if the live data distribution drifts over time — recompute periodically, or use a rolling window so the "mean" and "SD" used for scoring stay current.
- With many features, applying a Z-score independently to each feature ignores the *correlations between* features — a point can look perfectly normal on every single feature checked one at a time, and still be a genuine joint outlier once you consider the features together (recall the "10-year-old who is 6 feet tall" example from Chapter 1). This is exactly what motivates **Mahalanobis distance** (Chapter 6), which scores points using all features jointly rather than one at a time.

## 2.8 Interview Traps

- Not knowing that the classic Z-score's own outlier can effectively hide itself via variance inflation (the masking effect) — this is the single most common follow-up question after "how would you detect outliers?" and the worked example in Section 2.2 is the go-to answer.
- Treating $\tau=3$ as if it's some universal law of outlier detection, rather than a convention that only makes sense under an assumed Gaussian tail (Section 2.1).
- Forgetting that the classic Z-score assumes (at least approximate) normality — heavy-tailed or skewed data breaks the interpretation of "3 SDs = rare," because the empirical-rule percentages no longer hold.
- Not being able to say what MAD stands for, or explain the 0.6745 scaling constant, when asked to justify the Modified Z-score formula rather than just quote it.

## 2.9 L5-Differentiating Talking Points

- Explaining masking and swamping as *consequences* of using non-robust estimators (tying directly back to Chapter 1's breakdown-point discussion), rather than reciting them as two disconnected memorized terms.
- Proactively noting that even MAD isn't bulletproof: MAD's own breakdown point is also only about 50% — if *more than half* your data is actually contaminated, the "typical" value the median finds is itself corrupted, and true outliers can get missed because the corrupted majority has effectively become the new "normal." This shows you understand robustness has limits, not just that "MAD is robust" as a blanket statement.
- Mentioning production/streaming estimation challenges (t-digest, KLL sketches for approximate quantiles) unprompted — this signals systems maturity that goes beyond textbook statistics.

## 2.10 Comprehension Check — With Answers

**1. Explain in one sentence why the classic Z-score can fail to detect the very outlier that's distorting it.**

The outlier itself inflates both the mean $\bar{x}$ (pulling it toward the outlier) and, much more severely, the standard deviation $s$ (via the squared-deviation term, per Chapter 1's quadratic-sensitivity argument) — and because $s$ sits in the denominator of the Z-score formula, this inflation shrinks the computed Z-score for every point, including the outlier's own, letting it slip under the detection threshold (the masking effect, demonstrated numerically in Section 2.2, where the true outlier scored only 2.84 against a 3.0 cutoff).

**2. If MAD = 0 (happens when more than 50% of values are identical), what breaks, and what would you do instead?**

MAD sits in the denominator of the Modified Z-score formula, so $\text{MAD}=0$ makes the formula divide by zero — every non-identical point would compute to an infinite (or undefined) Modified Z-score, and the method becomes unusable as written. This happens whenever the majority of the data shares the exact same value (a common real-world case: sensor data with long stretches of a repeated reading, or heavily rounded/discretized data). Practical fixes include: (a) falling back to a small constant floor for MAD (some implementations substitute the mean absolute deviation from the median as a backup scale, or add a tiny epsilon), (b) switching to a different robust-spread estimator that doesn't collapse to zero as easily, such as the interquartile range (Chapter 3), or (c) recognizing that if the *majority* of values are identical, this might itself be a signal worth investigating directly (e.g., a stuck sensor — a collective outlier, per Chapter 1's taxonomy) rather than something to route around purely mathematically.

**3. Why is 0.6745 used specifically, and what does it assume about the underlying distribution?**

It assumes the underlying data is (at least approximately) normally distributed — under that assumption, it's a mathematical fact that $\text{MAD} \approx 0.6745\,\sigma$, meaning MAD naturally comes out smaller than the standard deviation for normally-distributed data. Multiplying MAD by $1/0.6745$ (equivalently, multiplying the numerator by 0.6745 in the formula as written) rescales it back up to be on the same numeric scale as $\sigma$, so that a Modified Z-score can be compared against roughly the same "how many SDs away" intuition used for the classic Z-score. If the data is very far from normal (e.g., strongly skewed or multimodal), this constant no longer has the same clean justification, though the Modified Z-score's robustness properties still generally hold up better than the classic version's would.

**4. Given data `[5,5,5,5,5,100]`, compute the classic Z-score and Modified Z-score for the value 100. What does the contrast reveal?**

*Classic Z-score:*
- Mean: $(5\times5 + 100)/6 = 125/6 \approx 20.83$
- Deviations from mean: five copies of $(5-20.83)=-15.83$, and one $(100-20.83)=79.17$
- Squared deviations: five copies of $15.83^2\approx250.69$ (summing to $\approx1253.47$), plus $79.17^2\approx6267.36$ → total sum of squares $\approx7520.83$
- $s = \sqrt{7520.83/5} = \sqrt{1504.17} \approx 38.78$
- $z_{100} = (100-20.83)/38.78 \approx 2.04$ — **below** the usual $\tau=3$ cutoff, so the classic Z-score would fail to flag an obviously anomalous point (100 versus a cluster of five identical 5's).

*Modified Z-score:*
- Median: with 6 values sorted as `[5,5,5,5,5,100]`, the median is the average of the 3rd and 4th values, both 5 → median $=5$
- Absolute deviations from median: five copies of $|5-5|=0$, and one $|100-5|=95$
- Sorted absolute deviations: `[0,0,0,0,0,95]` → MAD = median of these = average of 3rd and 4th = $(0+0)/2 = 0$
- **MAD = 0** — the formula divides by zero, so the Modified Z-score for 100 is undefined/infinite as written, exactly the edge case from Question 2 above.

**What the contrast reveals:** this is a sharper version of the same masking problem from Section 2.2 — the classic Z-score fails "softly" (it produces a real but too-small number, 2.04, that slips under the threshold), while the Modified Z-score fails "loudly" (it breaks outright via division by zero) *when the majority of the data is perfectly uniform*. In practice, this specific dataset shape — a large identical cluster plus one wild point — is exactly the scenario described in Question 2's answer, and would call for one of those practical fallbacks (a small epsilon floor on MAD, or switching to IQR-based detection) rather than either formula as written.

---
*Next: Chapter 3 — IQR / Tukey's Fences (boxplot method), including derivation of the 1.5× and 3× constants.*
