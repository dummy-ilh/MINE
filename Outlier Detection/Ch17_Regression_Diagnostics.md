# Chapter 17: Outliers in Regression — Leverage, Studentized Residuals, Cook's Distance, DFFITS/DFBETAS

## 17.1 Motivation — A Fundamentally Different Question

Every prior chapter asked "is this point unusual relative to the data's overall distribution?" Regression diagnostics ask a sharper, model-specific question: **"how much does this single point change the fitted model itself?"** A point can be a perfectly ordinary value on every feature and still have an outsized effect on the regression line (high **leverage**), or it can sit far from the fitted line despite being unremarkable in its feature values (large **residual**) — and these are genuinely different failure modes requiring different diagnostics. This chapter is essentially "Ch.1's mean/variance-sensitivity discussion (§1.5), formalized precisely for linear regression."

## 17.2 The Hat Matrix and Leverage

For linear regression $\hat{y} = X\hat\beta$ with $\hat\beta = (X^TX)^{-1}X^Ty$, the fitted values can be written as:
$$
\hat{y} = X(X^TX)^{-1}X^Ty = Hy
$$
where $H = X(X^TX)^{-1}X^T$ is the **hat matrix** (it "puts the hat on" $y$). The diagonal entries $h_{ii}$ are the **leverage** of point $i$:
$$
h_{ii} = x_i^T(X^TX)^{-1}x_i
$$

**Interpretation:** $h_{ii}$ measures how far point $i$'s **predictor values** (its $x$-position, not its $y$-value) sit from the center of the predictor space — entirely independent of whether $y_i$ fits the trend or not. A point can have high leverage (extreme $x$) yet lie perfectly on the regression line (small residual) — it's still "risky" because it has outsized *potential* influence, even if it isn't currently misbehaving.

**Rule of thumb threshold:** flag $h_{ii} > \frac{2p}{n}$ (or sometimes $\frac{3p}{n}$), where $p$ = number of parameters (including intercept), $n$ = sample size. Note $\sum_i h_{ii} = p$ always (a fixed algebraic identity), so average leverage is $p/n$ — the threshold is simply "leverage more than 2-3× the average."

## 17.3 Residuals — Standardized and Studentized

**Raw residual:** $e_i = y_i - \hat{y}_i$ — but raw residuals have unequal variance across points! Specifically:
$$
\text{Var}(e_i) = \sigma^2(1-h_{ii})
$$
High-leverage points have artificially **smaller** residual variance (the fitted line is pulled toward them, shrinking their own residual) — so comparing raw residuals directly across points is misleading; a high-leverage outlier can have a deceptively small raw residual precisely *because* it dragged the line toward itself.

**Standardized (internally studentized) residual:**
$$
r_i = \frac{e_i}{s\sqrt{1-h_{ii}}}
$$
where $s$ is the residual standard error from the *full* model (including point $i$). Corrects for the unequal-variance problem above, but $s$ itself is influenced by point $i$ if it's an outlier — same circularity issue as Ch.6-7.

**Studentized (externally studentized / deleted) residual:**
$$
t_i = \frac{e_i}{s_{(i)}\sqrt{1-h_{ii}}}
$$
where $s_{(i)}$ is the residual standard error computed with point $i$ **excluded** from the model fit. This is the direct regression analog of Ch.7's MCD fix: refit without the suspect point to get an uncontaminated scale estimate, then measure the excluded point against that clean baseline. Under standard assumptions, $t_i$ follows a **t-distribution with $n-p-1$ degrees of freedom**, giving a proper hypothesis test.

## 17.4 Cook's Distance — Combining Leverage and Residual Into One Influence Measure

**The key insight:** leverage alone tells you "could this point matter a lot?" and residual alone tells you "does this point fit poorly?" — but **influence** (how much does removing this point change the fitted coefficients?) requires **both** to be present simultaneously. A high-leverage point with a small residual (fits perfectly despite extreme $x$) has low influence; a low-leverage point with a huge residual (unusual $y$ but typical $x$) also has limited influence, since the model isn't very sensitive to typical-$x$ points. **Cook's Distance combines both ingredients:**

$$
D_i = \frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}}
$$

**Reading the formula:** the first factor ($r_i^2/p$) captures "how badly does this point fit" (residual magnitude, scaled by number of parameters); the second factor ($h_{ii}/(1-h_{ii})$) captures "how much leverage does this point have" — the product is large only when **both** are simultaneously large.

**Rule of thumb threshold:** flag $D_i > \frac{4}{n}$ (a common simple cutoff), or compare against $F_{p,\,n-p}$ distribution critical values for a more formal test.

## 17.5 DFFITS and DFBETAS — Influence on Predictions vs. Individual Coefficients

**DFFITS** (Difference in Fits): how much does the *prediction* for point $i$ itself change when point $i$ is excluded from fitting?
$$
\text{DFFITS}_i = t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}}
$$
Flag if $|\text{DFFITS}_i| > 2\sqrt{p/n}$.

**DFBETAS**: how much does a *specific regression coefficient* $\hat\beta_j$ change (in standard-error units) when point $i$ is excluded?
$$
\text{DFBETAS}_{j(i)} = \frac{\hat\beta_j - \hat\beta_{j(i)}}{\text{SE}(\hat\beta_{j(i)})}
$$
where $\hat\beta_{j(i)}$ is the coefficient estimated with point $i$ excluded. Flag if $|\text{DFBETAS}_{j(i)}| > 2/\sqrt{n}$.

**Why DFBETAS matters beyond Cook's Distance:** Cook's Distance gives one overall influence number per point, but doesn't tell you *which coefficient* is being distorted. A point might barely move the intercept but substantially swing the slope on one specific predictor — DFBETAS localizes exactly which part of the model is sensitive to that point, directly analogous to how Ch.8's Q-statistic residual vector can localize *which original feature* drove a PCA-based anomaly.

## 17.6 Worked Numerical

Simple linear regression, $n=5$: $x = [1,2,3,4,10]$, $y=[2.1,3.9,6.2,7.8,11.5]$ (point 5, $x=10$, is a suspected high-leverage point).

**Step 1 — fit the model** (least squares on all 5 points): approximate fit gives $\hat\beta_0\approx1.05$, $\hat\beta_1\approx1.02$ (illustrative; the key qualitative behavior is what matters here).

**Step 2 — leverage for point 5:** $\bar{x} = (1+2+3+4+10)/5=4.0$. Leverage in simple regression:
$$
h_{ii} = \frac{1}{n}+\frac{(x_i-\bar{x})^2}{\sum_j(x_j-\bar{x})^2}
$$
$\sum(x_j-\bar x)^2 = 9+4+1+0+36=50$
$$
h_{55} = \frac{1}{5}+\frac{(10-4)^2}{50} = 0.2+\frac{36}{50}=0.2+0.72=0.92
$$
Threshold: $2p/n = 2(2)/5=0.8$. Since $0.92>0.8$ → **point 5 flagged as high leverage.**

**Step 3 — check the residual for point 5:** using the fitted line, $\hat{y}_5 \approx 1.05+1.02(10)=11.25$, actual $y_5=11.5$ — residual is small ($e_5\approx0.25$), since the line was pulled toward this point during fitting.

**Step 4 — Cook's Distance:** because leverage is very high (0.92) but residual is small, let's compute the product structure:
$$
D_5 = \frac{r_5^2}{p}\cdot\frac{h_{55}}{1-h_{55}} \approx \frac{(0.25/s)^2}{2}\cdot\frac{0.92}{0.08}
$$
Even with a small $r_5$, the $\frac{h_{55}}{1-h_{55}} = \frac{0.92}{0.08}=11.5$ term is large enough that $D_5$ can still exceed the $4/n=0.8$ threshold, depending on the exact residual scale — this numerical illustrates the key conceptual point: **high leverage alone, even with a modest residual, can still produce meaningful influence**, because the $h_{ii}/(1-h_{ii})$ term grows explosively as $h_{ii}\to1$.

**Contrast — remove point 5 entirely and refit** on just $x=[1,2,3,4]$, $y=[2.1,3.9,6.2,7.8]$: this gives a noticeably different slope estimate (since point 5 was pulling the fitted line's slope toward itself) — this refit comparison is exactly what $\hat\beta_{j(i)}$ in the DFBETAS formula represents directly.

## 17.7 Diagnosis: Reading the Full Picture

| Leverage | Residual | Interpretation |
|---|---|---|
| Low | Low | Ordinary point, no concern |
| Low | High | Unusual $y$-value but typical $x$ — limited influence on the fitted line, but worth checking data quality |
| High | Low | Extreme $x$-value but fits the trend well — currently "well-behaved" but risky (small changes could reveal outsized influence) |
| High | High | **Genuinely influential outlier** — flagged strongly by Cook's Distance, DFFITS, and likely DFBETAS on the relevant coefficient(s) |

## 17.8 Production Considerations
- These diagnostics are computed per-observation and require refitting or algebraic shortcuts (leave-one-out formulas exist in closed form for linear regression, avoiding literal $n$ refits) — practical for datasets up to moderate size; for very large $n$, approximate/sampled influence diagnostics or robust regression methods (Huber loss, RANSAC) are used instead of exhaustively flagging every point.
- In production ML pipelines using linear/generalized linear models (common in credit scoring, pricing models), Cook's Distance-style checks are a standard part of model validation before deployment, specifically to catch training examples that could be silently distorting a small number of coefficients.
- These are batch/offline diagnostic tools, not online real-time detectors — they require the fitted model and full training set already in hand.

## 17.9 Interview Traps
- Confusing leverage (about $x$-position only) with influence (requires both leverage and residual) — a very common conflation; leverage is a *necessary but not sufficient* condition for high influence.
- Not knowing that raw residuals have unequal variance across points ($\text{Var}(e_i)=\sigma^2(1-h_{ii})$) — leading to incorrect direct comparison of raw residuals across points with different leverage.
- Forgetting that internally studentized residuals ($r_i$) still have the circularity problem (the point being tested is included in computing $s$), and not knowing externally studentized/deleted residuals ($t_i$) is the fix — the exact same "clean baseline" fix pattern as Ch.7's MCD.
- Only mentioning Cook's Distance and forgetting DFBETAS when asked "which specific coefficient is being distorted by this point" — Cook's D gives one number per point, not per-coefficient granularity.

## 17.10 L5-Differentiating Talking Points
- Explicitly stating the 2×2 leverage/residual grid (§17.7) as the conceptual foundation, with Cook's Distance as the formula that *multiplies* both signals together — showing you understand why the formula has the structure it does, not just what it computes.
- Connecting externally studentized residuals' "refit without the suspect point" logic directly back to Ch.7's MCD philosophy — reinforcing yet again that "get a clean baseline by excluding the suspect, then measure against that" is a recurring pattern across this entire curriculum (Ch.4's Grubbs formula structure, Ch.7's MCD, and now Ch.17's studentized residuals all share this exact logical move).
- Knowing when to reach for DFBETAS specifically (localizing influence to one coefficient) versus Cook's Distance (overall influence) — precise tool selection under a specific diagnostic question.

## 17.11 Comprehension Check
1. Explain why a point can have high leverage but low influence, using a concrete description (not just the formula).
2. Why do raw residuals have unequal variance across observations, and how does the standardized residual formula correct for it?
3. What is the conceptual difference between an internally studentized residual and an externally studentized (deleted) residual, and why does the distinction matter?
4. Given $h_{ii}=0.85$ and $p=3$, $n=20$, would this point be flagged as high leverage under the $2p/n$ rule of thumb? Show the computation.

---
*Next: Chapter 18 — Time Series Outliers: Seasonal-Hybrid ESD & STL-Residual-Based Detection.*
