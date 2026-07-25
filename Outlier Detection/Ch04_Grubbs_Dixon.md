# Chapter 4: Grubbs' Test & Dixon's Q Test

## 4.1 Why We Need Formal Hypothesis Tests

Chapters 2–3 gave heuristic thresholds (3σ, 1.5×IQR) — useful, but not backed by a formal significance level. Grubbs' and Dixon's tests answer a sharper question: *"given this sample size, what's the probability this single most-extreme value could have arisen by chance from a normal distribution?"* — with an actual p-value / critical-value table, at a chosen $\alpha$. These are classical small-sample tests (originally designed for $n$ roughly 3–30) and assume the data is otherwise (approximately) normally distributed apart from the suspected outlier.

## 4.2 Grubbs' Test

**Use case:** detect a *single* outlier (either the max or the min) in an approximately normal dataset.

**Test statistic:**
$$
G = \frac{\max_i |x_i - \bar{x}|}{s}
$$
This is literally the largest Z-score in the dataset (Ch.2), but now compared against a formal critical value rather than an ad hoc "3."

**Critical value** (two-sided, significance level $\alpha$, sample size $n$):
$$
G_{crit} = \frac{n-1}{\sqrt{n}}\sqrt{\frac{t^2_{\alpha/(2n),\,n-2}}{n-2+t^2_{\alpha/(2n),\,n-2}}}
$$
where $t_{\alpha/(2n),\,n-2}$ is the critical value of the Student's t-distribution with $n-2$ degrees of freedom at significance $\alpha/(2n)$ (the $2n$ in the denominator is a Bonferroni-style correction for testing $n$ possible candidate points).

**Decision rule:** if $G > G_{crit}$, reject the null hypothesis (no outliers) — the extreme point is a statistically significant outlier.

### 4.2.1 Worked Numerical — Grubbs' Test

Data (n=8): `[9, 10, 10, 11, 9, 10, 11, 20]`

**Step 1 — mean and SD:**
$$
\bar{x} = \frac{9+10+10+11+9+10+11+20}{8} = \frac{90}{8} = 11.25
$$
Deviations: $-2.25,-1.25,-1.25,-0.25,-2.25,-1.25,-0.25,8.75$
Squares: $5.0625, 1.5625, 1.5625, 0.0625, 5.0625, 1.5625, 0.0625, 76.5625$
Sum of squares $= 91.5$
$$
s = \sqrt{91.5/7} = \sqrt{13.07} \approx 3.615
$$

**Step 2 — test statistic:**
$$
G = \frac{|20-11.25|}{3.615} = \frac{8.75}{3.615} \approx 2.42
$$

**Step 3 — critical value** at $\alpha=0.05$, $n=8$: using standard Grubbs' tables, $G_{crit}\approx 2.032$ (this value is normally looked up rather than computed live in an interview — knowing the formula's structure matters more than memorizing table values).

**Step 4 — decision:** $G = 2.42 > G_{crit} = 2.032$ → reject null → **20 is a statistically significant outlier at α=0.05.**

### 4.2.2 Iterative Grubbs (multiple outliers)
Grubbs' test as stated only removes/tests **one** point at a time. For multiple suspected outliers, apply it iteratively: test the most extreme point, remove if flagged, recompute $\bar{x}, s$ on the remaining data, and repeat. **Danger:** removing points one at a time changes $\bar{x}, s$ each round, which can cause **masking** in the opposite direction now — if two outliers are close to each other and both extreme, sequentially testing one at a time can fail to flag either (this exact problem motivates the Generalized ESD test in Chapter 5, which tests for a *known upper bound* of $k$ outliers simultaneously rather than one at a time).

## 4.3 Dixon's Q Test

**Use case:** very small samples (roughly $n = 3$ to $30$, most commonly cited for $n<10$), single suspected outlier, when you don't want to compute mean/SD at all — purely gap/range based.

**Test statistic:**
$$
Q = \frac{\text{gap}}{\text{range}} = \frac{|x_{\text{suspect}} - x_{\text{nearest neighbor}}|}{x_{\max}-x_{\min}}
$$

Intuition: how large is the gap between the suspected outlier and its nearest neighbor, *relative to the total spread of the data*? A large Q means the suspected point sits in an unusually isolated position, far from the rest of the cluster.

**Decision rule:** if $Q > Q_{crit}$ (looked up from Dixon's Q table for given $n$ and $\alpha$), flag as outlier.

### 4.3.1 Worked Numerical — Dixon's Q Test

Data (n=6, sorted): `[12, 13, 14, 14, 15, 22]`

Suspected outlier: 22 (the max).

**Step 1 — gap:** nearest neighbor to 22 is 15.
$$
\text{gap} = |22-15| = 7
$$

**Step 2 — range:**
$$
\text{range} = 22-12 = 10
$$

**Step 3 — Q statistic:**
$$
Q = 7/10 = 0.70
$$

**Step 4 — critical value:** for $n=6$ at $\alpha=0.05$, $Q_{crit} \approx 0.625$ (standard Dixon's Q table value).

**Step 5 — decision:** $Q=0.70 > Q_{crit}=0.625$ → **22 is flagged as an outlier.**

## 4.4 Grubbs vs. Dixon vs. Z-score/IQR — Diagnosis Table

| Property | Z-score/IQR (Ch 2–3) | Grubbs' | Dixon's Q |
|---|---|---|---|
| Formal significance level ($\alpha$) | No | Yes | Yes |
| Sample size regime | Any | Small–medium (this chapter typically cites up to ~n=30, though extensions exist) | Very small (n≈3–30, best known for n<10) |
| Needs mean/SD | Yes | Yes | No — purely rank/gap based |
| Assumes underlying normality | Loosely | Yes, explicitly | Yes, explicitly |
| Multiple outliers | Not natively | Only iteratively (masking risk) | Not designed for this — single point only |
| Best use case today | Quick EDA flagging | Small controlled experiments (labs, QC) needing a formal test | Chemistry/lab QC with tiny sample sizes — largely superseded in ML practice |

**Key diagnosis point for interviews:** Grubbs' and Dixon's Q both explicitly assume the *non-outlier* portion of the data is normally distributed. If that assumption is wrong (skewed data), both tests are unreliable — you'd reach for IQR (Ch.3, distribution-free) or a robust multivariate method instead.

## 4.5 Production Considerations
- Both tests are essentially never used at production ML scale — they're designed for small, carefully controlled samples (lab measurements, QC batches, small experiment datasets), not for millions of rows in a pipeline. In an ML interview, they mostly come up as **conceptual/statistical-testing knowledge checks**, not as something you'd deploy in a serving pipeline.
- Where they *do* still show up in practice: A/B test sanity checks with few data points per arm, sensor calibration QC in manufacturing, or scientific data cleaning before a small controlled study.
- Iterative Grubbs' has no natural stopping rule beyond "keep testing until nothing is flagged" — this can over-remove data if applied carelessly, especially with skewed real-world distributions misclassified as "normal."

## 4.6 Interview Traps
- Presenting Grubbs'/Dixon's as general-purpose, always-normality-assumed methods without noting they break down on skewed or heavy-tailed data (a very common interview follow-up).
- Not knowing that Grubbs' test statistic is literally just "the largest Z-score" — memorizing the formula without recognizing the connection to Ch.2 loses easy credit.
- Applying single-outlier Grubbs' naively to a case with two co-located extreme outliers and being surprised when neither is flagged (masking) — good candidates flag this risk proactively.
- Confusing $\alpha$ (test significance level) with the earlier heuristic 3σ/1.5×IQR conventions — those aren't hypothesis tests with a formal p-value, these are.

## 4.7 L5-Differentiating Talking Points
- Explicitly stating that Grubbs' is "Z-score with a rigorous critical value" ties directly back to Ch.2 — shows you see the whole curriculum as one coherent framework, not isolated formulas.
- Volunteering the masking risk of iterative Grubbs' *before* being asked, and naming Generalized ESD (Ch.5) as the principled fix for a *known* number of candidate outliers.
- Correctly scoping these two tests as "classical/small-sample statistical testing" tools rather than production ML tools — shows judgment about when a textbook method is/isn't the right one for the job at hand, which is exactly the kind of calibration L5 interviewers probe for.

## 4.8 Comprehension Check
1. Why is Grubbs' test statistic mathematically identical in structure to a Z-score, and what does the critical value add on top of it?
2. Explain the masking risk in iterative Grubbs' testing with a concrete two-outlier example.
3. Why does Dixon's Q test not require computing the mean or standard deviation at all?
4. For data `[5, 6, 6, 7, 7, 8, 20]` (n=7), compute the Dixon's Q statistic for the suspected outlier 20 (nearest neighbor is 8, range is 15).

---
*Next: Chapter 5 — Generalized ESD Test (multiple outliers, known upper bound).*
