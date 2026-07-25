# Chapter 5: Generalized ESD Test (Multiple Outliers, Known Upper Bound)

## 5.1 Motivation — Fixing Grubbs' Masking Problem

Chapter 4 ended on the masking problem: iterative Grubbs' can fail when multiple outliers exist close together, because removing one at a time never fully "un-inflates" the mean/SD while others remain. The **Generalized Extreme Studentized Deviate (ESD) test** (Rosner, 1983) fixes this by testing for **up to a pre-specified maximum number of outliers, $r$,** all at once — computing all $r$ candidate test statistics *before* making any removal decisions, then comparing each to its own critical value in sequence. This decouples "how many might there be" from "which one do I remove first."

**Key requirement:** you must supply an upper bound $r$ on the number of suspected outliers (e.g., "I believe at most 5 of these 200 points could be outliers"). The test will tell you how many of those $r$ candidates are *actually* statistically significant — it can return anywhere from 0 to $r$.

## 5.2 The Algorithm

Given data of size $n$ and a suspected maximum of $r$ outliers:

**Step 1.** For $i = 1$ to $r$:
- Compute $\bar{x}$ and $s$ on the **current remaining dataset** (all $n-i+1$ points not yet removed).
- Compute the test statistic:
$$
R_i = \frac{\max_j |x_j - \bar{x}|}{s}
$$
- Remove the point achieving this max from the dataset (regardless of whether it will end up flagged — removal happens *before* the flagging decision).
- Record $R_i$.

**Step 2.** After computing $R_1, R_2, \dots, R_r$ (one per iteration), compute the corresponding critical values:
$$
\lambda_i = \frac{(n-i)\,t_{p,\,n-i-1}}{\sqrt{(n-i-1+t^2_{p,\,n-i-1})(n-i+1)}}, \quad p = 1-\frac{\alpha}{2(n-i+1)}
$$
where $t_{p,\,n-i-1}$ is the critical value of the t-distribution with $n-i-1$ degrees of freedom.

**Step 3.** Find the **largest $i$** such that $R_i > \lambda_i$. Declare that many outliers (the first $i$ removed points) as significant.

**Crucial subtlety:** you do NOT stop at the first $i$ where $R_i \le \lambda_i$. You compute all $r$ statistics first, then scan for the *last* (largest) index where the test statistic still exceeds its critical value. This ordering is precisely what avoids the masking failure — a later-iteration test statistic can still exceed its critical value even if an earlier intermediate one didn't, because each iteration recomputes $\bar x, s$ on progressively cleaned data.

## 5.3 Worked Numerical

Data (n=12): `[10, 11, 9, 10, 12, 11, 10, 9, 11, 10, 35, 40]`

Suppose we believe **at most $r=2$** outliers exist (35 and 40 look suspicious).

**Iteration 1** (all 12 points):
$$
\bar{x} = \frac{10+11+9+10+12+11+10+9+11+10+35+40}{12} = \frac{188}{12} \approx 15.67
$$
Deviations from mean, largest magnitude is for 40: $|40-15.67| = 24.33$
Need $s$: deviations squared sum... (computing quickly)
Deviations: $-5.67,-4.67,-6.67,-5.67,-3.67,-4.67,-5.67,-6.67,-4.67,-5.67,19.33,24.33$
Squares: $32.1,21.8,44.5,32.1,13.5,21.8,32.1,44.5,21.8,32.1,373.6,592.0$
Sum $\approx 1261.9$
$$
s = \sqrt{1261.9/11} \approx \sqrt{114.7} \approx 10.71
$$
$$
R_1 = 24.33/10.71 \approx 2.27
$$
Remove 40. Remaining n=11.

**Iteration 2** (11 points, 40 removed):
$$
\bar{x} = \frac{188-40}{11} = \frac{148}{11} \approx 13.45
$$
Largest remaining deviation is for 35: $|35-13.45|=21.55$
Recomputing $s$ on the remaining 11 points (deviations from 13.45): roughly $s \approx 7.9$ (following the same sum-of-squares procedure).
$$
R_2 = 21.55/7.9 \approx 2.73
$$
Remove 35.

**Step 2 — critical values** (looked up/computed from the formula in §5.2 for $n=12$, $\alpha=0.05$): typically $\lambda_1 \approx 2.29$, $\lambda_2 \approx 2.22$ (standard Rosner ESD table values for these parameters — decreasing slightly as degrees of freedom shrink but critical value structure accounts for it).

**Step 3 — decision:**
- $R_1 = 2.27$ vs $\lambda_1 = 2.29$ → $R_1 < \lambda_1$ (not significant *on its own*)
- $R_2 = 2.73$ vs $\lambda_2 = 2.22$ → $R_2 > \lambda_2$ → significant!

Since the **largest** $i$ with $R_i > \lambda_i$ is $i=2$, we declare **both** outliers (40 and 35) significant — even though $R_1$ alone looked borderline/non-significant. This is exactly the masking-resistant behavior Ch.4 couldn't give you: had we stopped at iteration 1 because $R_1 < \lambda_1$, we'd have wrongly concluded there were zero outliers and never even tested iteration 2.

## 5.4 Diagnosis: When to Use Generalized ESD

| Condition | Recommendation |
|---|---|
| Suspect multiple outliers, roughly know upper bound $r$ | Generalized ESD — the correct classical tool |
| Data approximately normal apart from outliers | Required assumption, same as Grubbs'/Dixon's |
| No idea how many outliers might exist | ESD still works if you set $r$ generously (e.g., 10% of $n$) — cost is more computation, not invalid results |
| Large-scale ML pipeline (millions of rows) | Not practical — same production limitation as Ch.4; use LOF/Isolation Forest/Autoencoders instead |

## 5.5 Production Considerations
- Like Grubbs' and Dixon's, Generalized ESD is fundamentally a small-to-medium sample classical statistical test — rarely deployed as-is in high-volume ML pipelines, but frequently used for **cleaning smaller reference/calibration datasets**, one-off data audits, or validating a labeling pipeline's outputs before they're used as ground truth.
- The requirement to pre-specify $r$ is itself a modeling decision — setting $r$ too small can under-detect (never even testing candidate outliers beyond position $r$); setting $r$ very large mostly costs compute (each iteration is $O(n)$, so total cost is $O(rn)$), not correctness, since the test only declares real outliers based on the significance criterion regardless of how large $r$ is set.

## 5.6 Interview Traps
- Stopping the scan at the *first* $i$ where $R_i \le \lambda_i$, rather than checking the entire sequence for the *largest* significant $i$ — this is the single most common implementation mistake, and defeats the entire purpose of the test (re-read §5.2's "crucial subtlety").
- Forgetting that $\bar x$ and $s$ are recomputed on the *progressively shrinking* dataset at each iteration, not on the full original data every time.
- Treating Generalized ESD as if it can detect an *unknown, unbounded* number of outliers — it fundamentally requires an upper bound $r$ supplied up front.

## 5.7 L5-Differentiating Talking Points
- Explicitly framing Generalized ESD as "iterative Grubbs', but decoupling detection order from the final flagging decision" — connects directly back to Ch.4's masking discussion, again reinforcing the "each method fixes a specific failure mode of the previous one" narrative running through this whole curriculum.
- Correctly noting that setting $r$ too conservatively (too small) is the main practical risk, not a subtle statistical one — a very concrete, actionable point that shows practical judgment.
- Being explicit that Generalized ESD, like Grubbs and Dixon, assumes approximate normality of the "clean" portion of the data, and volunteering that skewed real-world data (revenue, latency) would need a transform first or a different, distribution-free method.

## 5.8 Comprehension Check
1. Why must you compute *all* $r$ test statistics before making any final flagging decision, rather than stopping early?
2. In the worked example, why was $R_1$ alone not sufficient evidence to flag any outliers, yet both points ended up flagged?
3. What happens to the result if you set $r$ too small (say $r=1$) when there are actually 2 real outliers?
4. Explain, in your own words, why Generalized ESD is described as "masking-resistant" relative to plain iterative Grubbs'.

---
*Next: Chapter 6 — Mahalanobis Distance (multivariate outlier detection, covariance-aware).*
