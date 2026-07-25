# Chapter 2: Z-score & Modified Z-score (MAD-Based) Methods

## 2.1 The Z-score

**Intuition:** measure how many standard deviations a point sits from the mean. It's a rescaling of the data so "distance from typical" is comparable across features with different units/scales.

**Formula:**
$$
z_i = \frac{x_i - \bar{x}}{s}
$$
where $\bar{x} = \frac{1}{n}\sum x_i$ and $s = \sqrt{\frac{1}{n-1}\sum (x_i-\bar{x})^2}$.

**Threshold rule:** flag $x_i$ as an outlier if $|z_i| > \tau$, commonly $\tau = 3$.

### Where the "3" comes from
Under a Gaussian, the empirical rule gives:
- $P(|Z|>1) \approx 31.7\%$
- $P(|Z|>2) \approx 4.55\%$
- $P(|Z|>3) \approx 0.27\%$

So $\tau=3$ flags roughly the most extreme 0.27% of data under normality — a convention, not a law. This is exactly the tail-density argument from Chapter 1: at $|z|=3$, $f(x)$ has collapsed to near-zero.

## 2.2 Worked Numerical — Z-score

Data: `[10, 12, 12, 13, 12, 11, 14, 13, 15, 90]`

Step 1 — mean:
$$
\bar{x} = \frac{10+12+12+13+12+11+14+13+15+90}{10} = \frac{202}{10} = 20.2
$$

Step 2 — standard deviation (sample, $n-1$):
Deviations from mean: $-10.2, -8.2, -8.2, -7.2, -8.2, -9.2, -6.2, -7.2, -5.2, 69.8$
Squared: $104.04, 67.24, 67.24, 51.84, 67.24, 84.64, 38.44, 51.84, 27.04, 4872.04$
Sum of squares $= 5431.6$
$$
s = \sqrt{5431.6/9} = \sqrt{603.5} \approx 24.57
$$

Step 3 — Z-scores:
$$
z_{90} = \frac{90-20.2}{24.57} \approx 2.84
$$
$$
z_{10} = \frac{10-20.2}{24.57} \approx -0.42
$$

**Result:** even the value 90 — nearly 4.5× the next-highest value — only reaches $z\approx2.84$, *below* the common $\tau=3$ cutoff! This happens because the single outlier itself inflated $\bar{x}$ and $s$ (Chapter 1's "mean/variance are not robust" point, now showing up as a false negative). This is the single most important practical failure of the classic Z-score, and a great numerical to have ready in an interview.

## 2.3 Why Z-score Breaks: The Masking Effect

**Masking**: outliers inflate $s$, which raises the denominator, which shrinks every $z_i$ — including the outlier's own score — hiding it.

**Swamping** (the flip side): in skewed or heavy-tailed data, a large $s$ driven by real outliers can cause *normal* points to look artificially close to zero, or conversely a shifted mean can make normal points look extreme.

Both failure modes stem from the same root cause: **Z-score uses non-robust estimators ($\bar{x}$, $s$) to detect the very thing that breaks those estimators.** This circularity motivates Modified Z-score.

## 2.4 Modified Z-score (MAD-based)

**Intuition:** replace the mean and SD with the **median** and **Median Absolute Deviation (MAD)** — both have high breakdown points (Ch.1), so a single (or even up to ~50% of) outliers can't drag them around.

**MAD formula:**
$$
\text{MAD} = \text{median}\big(|x_i - \text{median}(x)|\big)
$$

**Modified Z-score:**
$$
M_i = \frac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}
$$
where $\tilde{x}$ = median. Flag if $|M_i| > 3.5$ (Iglewicz & Hoaglin's recommended threshold).

### Where does 0.6745 come from?
For a standard normal distribution, $\text{MAD} \approx 0.6745\,\sigma$ (i.e., MAD is a scaled-down version of $\sigma$ for normal data). Dividing by MAD directly would give values on a different scale than a normal Z-score; multiplying by 0.6745 rescales MAD back onto the same scale as $\sigma$, so that $M_i$ is directly comparable to an ordinary Z-score and the same "$\approx 3$" intuition roughly applies (though the recommended cutoff is 3.5, slightly higher, because MAD is a less efficient but far more robust estimator).

## 2.5 Worked Numerical — Modified Z-score (same data)

Data: `[10, 12, 12, 13, 12, 11, 14, 13, 15, 90]`

Step 1 — median: sort → `[10,11,12,12,12,13,13,14,15,90]`. Median (avg of 5th, 6th) = $(12+13)/2 = 12.5$

Step 2 — absolute deviations from median:
$|10-12.5|=2.5$, $|11-12.5|=1.5$, $|12-12.5|=0.5$ (×3), $|13-12.5|=0.5$ (×2), $|14-12.5|=1.5$, $|15-12.5|=2.5$, $|90-12.5|=77.5$

Sorted absolute deviations: `[0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 77.5]`
MAD = median of these = avg of 5th, 6th = $(0.5+1.5)/2 = 1.0$

Step 3 — Modified Z-score for 90:
$$
M_{90} = \frac{0.6745 \times (90 - 12.5)}{1.0} = 0.6745 \times 77.5 \approx 52.3
$$

**Result:** $M_{90} \approx 52.3 \gg 3.5$ — instantly, unambiguously flagged. Compare to the classic Z-score's 2.84 (missed). This is the numerical proof of why MAD-based detection is preferred whenever contamination is suspected.

For a normal point, e.g. $x=10$:
$$
M_{10} = \frac{0.6745\times(10-12.5)}{1.0} = -1.69
$$
Well within bounds — correctly not flagged.

## 2.6 Diagnosis / When Each Method Applies

| Condition | Recommended |
|---|---|
| Data roughly Gaussian, no/few outliers suspected | Z-score fine, simple, fast |
| Suspected outliers/contamination present | Modified Z-score (MAD) |
| Small sample sizes ($n<30$) | Prefer MAD or Grubbs (Ch.4) — sample SD is unstable with small $n$ |
| Multiple simultaneous outliers | Z-score is especially unreliable here (Ch.1 masking); MAD degrades much more gracefully up to ~50% contamination |
| Skewed (non-symmetric) distributions | Neither is ideal — median-based center still helps, but consider IQR (Ch.3) or transformation first |

## 2.7 Production Considerations
- MAD requires storing the full sorted array (or a running approximate median structure) to update online — costlier than Z-score's simple running mean/variance (Welford's algorithm). For streaming systems at scale, approximate quantile structures (t-digest, KLL sketches) are used to get MAD-like behavior without full sorts.
- Z-score thresholds computed once during training and hard-coded into a serving pipeline are a classic train/serve skew source if the live distribution drifts — recompute periodically or use rolling windows.
- With many features, applying Z-score independently per feature ignores correlation between features — a value can be marginally normal on every feature yet be a joint outlier (motivates Mahalanobis distance, Ch.6).

## 2.8 Interview Traps
- Not knowing that Z-score's own outlier can hide itself via variance inflation (the masking effect) — this is the #1 follow-up question after "how would you detect outliers?"
- Using $\tau=3$ as if it's a universal law rather than a Gaussian-tail convention.
- Forgetting Z-score assumes (approximate) normality — heavy-tailed or skewed data breaks the interpretation of the threshold.
- Not knowing what MAD stands for or the 0.6745 scaling constant when asked to justify the formula.

## 2.9 L5-Differentiating Talking Points
- Explaining masking/swamping as a *consequence* of non-robust estimators, tying back to Ch.1's breakdown point discussion, rather than reciting it as a memorized fact.
- Proactively bringing up that with high contamination (many outliers), even MAD needs care — MAD's own breakdown point is ~50%, so past that point *the majority becomes the new "normal"* and true outliers get missed.
- Mentioning streaming/production estimation challenges (approximate quantiles) unprompted — shows systems maturity beyond textbook stats.

## 2.10 Comprehension Check
1. Explain in one sentence why the classic Z-score can fail to detect the very outlier that's distorting it.
2. If MAD = 0 (happens when >50% of values are identical), what breaks, and what would you do instead?
3. Why is 0.6745 used specifically, and what does it assume about the underlying distribution?
4. Given data `[5,5,5,5,5,100]`, compute the classic Z-score and Modified Z-score for the value 100. What does the contrast reveal?

---
*Next: Chapter 3 — IQR / Tukey's Fences (boxplot method), including derivation of the 1.5× and 3× constants.*
