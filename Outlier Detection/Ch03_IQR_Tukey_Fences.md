# Chapter 3: IQR / Tukey's Fences (Boxplot Method)

## 3.1 Intuition

Instead of measuring distance from the *mean* (Ch.2), measure distance from the **middle 50% of the data**. This makes the method rank-based rather than magnitude-based for its center of reference, and it doesn't require any distributional assumption like normality — a genuinely non-parametric approach.

## 3.2 Definitions

- $Q_1$ = 25th percentile (first quartile)
- $Q_3$ = 75th percentile (third quartile)
- **Interquartile Range:**
$$
IQR = Q_3 - Q_1
$$

**Tukey's fences:**
$$
\text{Lower fence} = Q_1 - k\cdot IQR
$$
$$
\text{Upper fence} = Q_3 + k\cdot IQR
$$

Standard $k=1.5$ for "outliers," and $k=3.0$ for "extreme/far outliers" — the boxplot's whiskers vs. the points shown beyond them.

## 3.3 Where Does 1.5 Come From? (the derivation interviewers love to ask)

For a **standard normal distribution**:
- $Q_1 \approx -0.6745$, $Q_3 \approx 0.6745$ (these are the same 0.6745 you saw in Ch.2's MAD constant — not a coincidence, both come from the normal's quantile function)
- $IQR = 0.6745 - (-0.6745) = 1.349$

Upper fence at $k=1.5$:
$$
Q_3 + 1.5 \times IQR = 0.6745 + 1.5(1.349) = 0.6745 + 2.0235 = 2.698
$$

So Tukey's upper fence sits at $z \approx 2.698$ standard deviations from the mean under normality — almost exactly matching the "$3\sigma$" rule of thumb from Ch.2! Tukey chose 1.5 specifically so the IQR method's flagging rate would roughly agree with the classical $3\sigma$ convention, while being far more robust (median/quartiles have breakdown points of 25–50%, vastly better than mean/SD's $1/n$).

For $k=3$ (extreme outliers):
$$
Q_3 + 3\times IQR = 0.6745 + 3(1.349) = 0.6745+4.047 = 4.72\sigma
$$
Roughly a "4.7$\sigma$" extreme-outlier line — used to distinguish "mild" outliers (shown as individual points) from "far out" ones in a boxplot (sometimes marked differently).

**Expected flagging rate under normality:** $P(Z > 2.698) \approx 0.35\%$ per tail, so about **0.7% total** flagged as mild outliers under a clean Gaussian — a useful sanity-check number: if your real flagging rate is much higher, your data likely isn't Gaussian (skewed/heavy-tailed), which is very common in real interview datasets (revenue, latency, session length).

## 3.4 Worked Numerical

Data (n=11): `[8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 60]`

**Step 1 — find quartiles.** Using the standard (linear interpolation) method:
Position of $Q_1$ = $0.25\times(n-1)+1 = 0.25\times10+1 = 3.5$ → between 3rd and 4th sorted values: $10$ and $10$ → $Q_1 = 10$
Position of $Q_3$ = $0.75\times10+1 = 8.5$ → between 8th and 9th sorted values: $13$ and $14$ → $Q_3 = 13.5$

**Step 2 — IQR:**
$$
IQR = 13.5 - 10 = 3.5
$$

**Step 3 — fences:**
$$
\text{Lower} = 10 - 1.5(3.5) = 10 - 5.25 = 4.75
$$
$$
\text{Upper} = 13.5 + 1.5(3.5) = 13.5+5.25 = 18.75
$$

**Step 4 — flag:** Any value $<4.75$ or $>18.75$. Here, $60 > 18.75$ → flagged. All other values (8–15) fall inside $[4.75, 18.75]$ → not flagged.

**Contrast with Z-score on similar data:** unlike Ch.2's example where a single extreme value inflated $s$ enough to mask itself, quartiles $Q_1, Q_3$ barely move even with 60 in the mix, because they only depend on rank position, not magnitude — 60 could be 60 or 6000 and $Q_1, Q_3$ would be identical. This is IQR's key robustness advantage over Z-score.

## 3.5 Diagnosis: When IQR Is (and Isn't) the Right Tool

| Situation | IQR appropriate? |
|---|---|
| Skewed data (income, latency, wait times) | Yes — doesn't assume symmetry like Z-score does, though fences will be asymmetric around the median which is a **feature**, not a bug |
| Small sample size (n<10) | Caution — quartile estimates are noisy with very few points |
| Multimodal data | No — a value between two modes might get flagged despite being "normal" for a distinct subgroup; segment first |
| Need a *severity* score, not just a flag | No — IQR gives a binary in/out; use Modified Z-score or LOF for a continuous outlier score |
| Discrete/heavily tied data | Caution — many identical values can make $IQR=0$ or very small, causing over-flagging (analogous to Ch.2's MAD=0 edge case) |

## 3.6 Production Considerations
- Percentile computation over big/streaming data is nontrivial exactly — approximate quantile sketches (t-digest, GK-sketch) are standard in production monitoring systems (e.g., computing p99 latency IQR bounds without storing every point).
- IQR fences computed once on a training window and then hard-coded for scoring live data will drift stale exactly like Z-score thresholds (Ch.2) — recompute periodically on a rolling window, especially for metrics with seasonality (e.g., traffic volume).
- Boxplots/IQR are extremely popular for quick EDA outlier flags in dashboards precisely because they need no distributional assumption — but remember the flag is univariate; it says nothing about multivariate joint outliers (motivates Mahalanobis, Ch.6).

## 3.7 Interview Traps
- Not being able to derive where 1.5 comes from when asked "why 1.5, why not 2?" — the normal-quantile derivation above is exactly this answer.
- Forgetting IQR fences are asymmetric when data is skewed, and treating that asymmetry as a bug rather than the expected/correct behavior.
- Confusing "outside the fences" (flagged) with "outside the whiskers as literally drawn on a boxplot" — some plotting libraries cap whiskers at the fence value or at the most extreme in-range point; know the distinction if asked to read a boxplot.
- Applying IQR per-feature on high-dimensional data and assuming that catches multivariate outliers — it doesn't (same trap as Ch.2 §2.7).

## 3.8 L5-Differentiating Talking Points
- Deriving the 1.5 constant from first principles (normal quantile function) on the spot, and connecting it back to the $\approx3\sigma$ Z-score convention — shows the two methods aren't unrelated tools but different lenses on the same underlying idea (tail density from Ch.1).
- Proactively noting IQR's *rank-based* robustness mechanism is the same underlying reason median/MAD are robust (Ch.1's breakdown-point framework) — ties the whole curriculum together instead of treating each method in isolation.
- Knowing when IQR quietly fails (heavily tied/discrete data, multimodal data) rather than presenting it as a universal default.

## 3.9 Comprehension Check
1. Derive why the "extreme outlier" fence uses $k=3$ instead of $1.5$, and what $\sigma$-equivalent that corresponds to under normality.
2. Why does IQR remain stable even if the single largest value in the dataset is replaced by an arbitrarily larger number?
3. Give a real-world example of a discrete/heavily-tied dataset where IQR fences could misbehave, and explain the mechanism.
4. For data `[1,2,2,3,3,3,4,4,5,5,50]`, compute $Q_1$, $Q_3$, IQR, and the upper fence, and state whether 50 is flagged.

---
*Next: Chapter 4 — Grubbs' Test & Dixon's Q Test (small-sample, single/known-outlier tests).*
