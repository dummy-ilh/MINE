# Chapter 2 — Inference in Simple Linear Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — continuing the same 5-student dataset from Chapter 1 so every number below is traceable back to something you already computed by hand.*

Recall from Chapter 1: $\hat{\beta}_0 = 41.5$, $\hat{\beta}_1 = 7.5$, $\bar{x}=3$, $\bar{y}=64$, $S_{xx}=10$.

---

## 2.1 The Motivating Question

You fit a line and got $\hat{\beta}_1 = 7.5$. But this came from just 5 students. If you'd sampled a *different* 5 students, you'd get a slightly different slope — maybe 6.8, maybe 8.1. So the real question interviewers care about is:

**"Is your slope estimate reliable, or could this apparent relationship just be noise from a small, unlucky sample?"**

Everything in this chapter exists to answer that one question rigorously — with a number (standard error), a test (is $\beta_1$ really nonzero?), and a range (confidence interval) instead of a single fragile point estimate.

---

## 2.2 $\hat{\beta}_1$ Is a Random Variable — Its Own Sampling Distribution

This is the conceptual leap most students skip past too fast. $\hat{\beta}_1$ isn't a fixed number — it's a **statistic**, computed from random data, so it has its own probability distribution across hypothetical repeated samples.

Under the LINE assumptions from Chapter 1 (plus Normality specifically for this section), it can be shown:

$$ \hat{\beta}_1 \sim N\left(\beta_1, \ \frac{\sigma^2}{S_{xx}}\right) $$

**Plain-English reading:** the estimate is centered on the true slope (unbiased — no systematic over/under-estimation), and its spread shrinks as $S_{xx}$ (the spread of your x-values) grows. **Intuition:** the more spread out your x-values are, the more "leverage" you have to pin down the slope precisely — a dataset where everyone studied between 2.9 and 3.1 hours tells you almost nothing about the slope, no matter how many people you sample.

---

## 2.3 Estimating $\sigma^2$ — The One Missing Ingredient

The formula above needs $\sigma^2$, the true (unknown) variance of the errors. We estimate it from residuals:

$$ \hat{\sigma}^2 = MSE = \frac{SSE}{n-2} = \frac{\sum e_i^2}{n-2} $$

**Why divide by $n-2$, not $n$?** Because computing $\hat{\beta}_0$ and $\hat{\beta}_1$ each "uses up" one degree of freedom from your data — you needed the data to estimate 2 parameters before you could even compute a residual. This is the exact same logic as dividing by $n-1$ for a sample variance (which uses up 1 df estimating the mean); here we lose 2 df because we estimated 2 parameters. **This is one of the single most commonly asked "why" questions in interviews** — always answer in terms of degrees of freedom lost to parameter estimation, not "just because."

**Worked numbers (from our dataset):**

Fitted values: $\hat{y} = 49, 56.5, 64, 71.5, 79$ for $x=1,2,3,4,5$.

Residuals: $e = 1, -1.5, 1, -1.5, 1$

$$ SSE = 1^2+(-1.5)^2+1^2+(-1.5)^2+1^2 = 1+2.25+1+2.25+1 = 7.5 $$

$$ MSE = \frac{7.5}{5-2} = \frac{7.5}{3} = 2.5 $$

$$ \hat{\sigma} = s = \sqrt{2.5} \approx 1.581 $$

$s$ is called the **residual standard error** — roughly, "on average, how far off is a prediction, in the original units of y." Here: predictions are typically off by about 1.58 points.

---

## 2.4 The Standard Error of $\hat{\beta}_1$

Plugging our estimate of $\sigma^2$ into the formula from 2.2:

$$ SE(\hat{\beta}_1) = \sqrt{\frac{MSE}{S_{xx}}} = \sqrt{\frac{2.5}{10}} = \sqrt{0.25} = 0.5 $$

**Reading it:** our slope estimate of 7.5 has a "typical wobble" of about ±0.5 across hypothetical resamples. That's small relative to 7.5 — a good sign the effect is real, not noise. We formalize that intuition next.

---

## 2.5 Hypothesis Test on $\beta_1$

**The question in test form:** $H_0: \beta_1 = 0$ (no linear relationship at all) vs. $H_a: \beta_1 \neq 0$.

**Test statistic:**

$$ t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)} = \frac{7.5}{0.5} = 15 $$

with $n-2 = 3$ degrees of freedom.

**Why a t-distribution and not normal?** Because we substituted an *estimate* $\hat{\sigma}$ for the true unknown $\sigma$ — that extra uncertainty fattens the tails relative to a normal distribution, especially with few degrees of freedom. This is identical in spirit to why the one-sample t-test exists instead of just using a z-test.

**Decision:** with $t=15$ and only 3 df, this is enormous — the critical value for $\alpha=0.05$ (two-tailed, 3 df) is about $t^*=3.182$. Since $15 \gg 3.182$, we reject $H_0$ decisively: there's very strong evidence hours studied has a real, nonzero relationship with exam score. (With only 5 data points, you'd normally be cautious about strong claims — but this particular toy dataset was constructed to be almost perfectly linear, which is why the effect looks this dramatic. Real data is rarely this clean.)

---

## 2.6 Confidence Interval for $\beta_1$

$$ \hat{\beta}_1 \pm t^*_{(\alpha/2, \ n-2)} \cdot SE(\hat{\beta}_1) $$

$$ 7.5 \pm 3.182 \times 0.5 = 7.5 \pm 1.591 = (5.91, \ 9.09) $$

**Correct interpretation (the version interviewers listen for):** "If we repeated this sampling process many times and built a CI each time using this same method, about 95% of those intervals would contain the true $\beta_1$." **Incorrect interpretation to avoid saying out loud:** "There's a 95% probability the true $\beta_1$ is in this specific interval" — the true $\beta_1$ is a fixed constant, not random; the *interval* is what's random across samples, not the parameter.

---

## 2.7 The ANOVA Decomposition — Where $R^2$ Comes From

Going back to the geometric picture from Chapter 1: total variation in y splits cleanly into "variation explained by the line" and "variation left over as residual noise."

$$ \underbrace{\sum(y_i-\bar{y})^2}_{SST} = \underbrace{\sum(\hat{y}_i-\bar{y})^2}_{SSR} + \underbrace{\sum(y_i-\hat{y}_i)^2}_{SSE} $$

| Term | Name | Meaning |
|---|---|---|
| SST | Total Sum of Squares | Total variability in y, ignoring x entirely |
| SSR | Regression Sum of Squares | Variability in y *explained* by the line |
| SSE | Error Sum of Squares | Variability left unexplained (residual noise) |

This identity holds **exactly**, always, for OLS — it's a direct consequence of the orthogonality of residuals to fitted values (Chapter 1, §1.7), not a coincidence or approximation.

**Worked numbers:**

$$ SST = \sum(y_i-64)^2 = (-14)^2+(-9)^2+1^2+6^2+16^2 = 196+81+1+36+256 = 570 $$

$$ SSE = 7.5 \ \text{(computed above)} $$

$$ SSR = SST - SSE = 570 - 7.5 = 562.5 $$

---

## 2.8 $R^2$ — Coefficient of Determination

$$ R^2 = \frac{SSR}{SST} = \frac{562.5}{570} \approx 0.9868 $$

**Plain-English meaning:** about 98.7% of the variation in exam scores is explained by hours studied; the remaining 1.3% is unexplained noise. $R^2$ always lies in $[0,1]$ for simple linear regression with an intercept.

**Direct link to correlation (interview-critical):** in *simple* linear regression specifically, $R^2 = r^2$ — the coefficient of determination is exactly the square of the Pearson correlation coefficient. This is a special case that does **not** generalize to multiple regression, where $R^2$ has a different, more complex relationship to the individual predictor correlations.

**Limitation to always mention in an interview:** a high $R^2$ does not imply the model is *correct* — it says nothing about whether assumptions hold, whether there's omitted-variable bias, or whether the relationship is genuinely causal. $R^2$ also mechanically increases (never decreases) every time you add *any* predictor to a multiple regression, even a useless one — a trap we'll return to when we cover adjusted $R^2$ in Chapter 14.

---

## 2.9 Confidence Interval for the Mean Response vs. Prediction Interval for a New Observation

This is one of the most commonly confused — and most commonly interview-tested — distinctions in the entire subject.

**Two different questions:**
1. "What's my best estimate of the *average* score for **all** students who study 3 hours?" → **Confidence interval for the mean response**
2. "What's my best estimate of the score for **one specific new** student who studies 3 hours?" → **Prediction interval for a new observation**

Both are centered at the same $\hat{y}$, but the prediction interval is **always wider**, because predicting one individual's outcome carries the *additional* uncertainty of that person's own random error $\varepsilon$ — on top of the uncertainty in estimating the line itself.

**Formulas** (at a chosen $x_0$):

$$ \text{CI for mean response: } \hat{y}_0 \pm t^* \cdot s\sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{S_{xx}}} $$

$$ \text{PI for new observation: } \hat{y}_0 \pm t^* \cdot s\sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^2}{S_{xx}}} $$

Notice the PI formula is identical except for the extra "+1" inside the square root — that "+1" **is** the individual's own error variance being added in.

**Worked numbers at $x_0 = 3 = \bar{x}$ (the simplest case, since $(x_0-\bar{x})^2=0$):**

$\hat{y}_0 = 64$

CI for mean response:
$$ 64 \pm 3.182 \times 1.581\sqrt{\frac{1}{5}+0} = 64 \pm 3.182 \times 1.581 \times 0.447 = 64 \pm 2.25 = (61.75, \ 66.25) $$

PI for a new observation:
$$ 64 \pm 3.182 \times 1.581\sqrt{1+\frac{1}{5}+0} = 64 \pm 3.182 \times 1.581 \times 1.095 = 64 \pm 5.51 = (58.49, \ 69.51) $$

The prediction interval (width ≈ 11) is more than **twice as wide** as the confidence interval (width ≈ 4.5) — exactly as expected, since it must account for one more source of randomness.

*(Diagram to visualize: the fitted line with two shaded bands around it — a narrow inner band for the mean-response CI, and a wider outer band for the PI, both bands curving slightly outward at the edges of the x-range, narrowest at $x=\bar{x}$.)*

---

## 2.10 Where the Textbooks Differ

- **Kutner** builds this chapter around the **ANOVA table** as the organizing structure — SST/SSR/SSE presented first, then everything else (t-tests, CIs) derived from it. Very proof-forward.
- **Montgomery** leads with the **t-test on $\beta_1$** first as the practical entry point, introducing the ANOVA table almost as an afterthought/equivalent F-test (recall: for simple linear regression, $t^2 = F$ — the two tests are mathematically identical, just different lenses).
- **Sheather** leads with **confidence intervals computed via software output**, emphasizing reading and interpreting a regression summary table (the kind produced by `lm()` in R or `statsmodels` in Python) over hand-deriving each formula.
- All three converge on the same numbers — this chapter's structure (SE → test → CI → ANOVA → $R^2$ → mean-vs-prediction interval) is designed to touch every entry point so no single textbook's framing surprises you.

---

## 2.11 Interview Q&A

**Q: Why does $SE(\hat{\beta}_1)$ decrease as $S_{xx}$ increases?**
A: More spread in your x-values gives more "leverage" to estimate the slope precisely — imagine trying to draw a reliable line through points all clustered at nearly the same x-value versus points spread widely; the latter pins down the slope far more tightly.

**Q: Why $n-2$ degrees of freedom, not $n-1$ or $n$?**
A: Two parameters ($\beta_0, \beta_1$) were estimated from the data before residuals could even be computed — each estimated parameter costs one degree of freedom.

**Q: What's the difference between the t-test on $\beta_1$ and the overall F-test in simple linear regression?**
A: They're mathematically identical in simple linear regression — $t^2 = F$ exactly, testing the same null hypothesis. They diverge in *multiple* regression, where the F-test asks about *all* predictors jointly while individual t-tests ask about *one predictor at a time*, holding others fixed.

**Q: Why is the prediction interval wider than the confidence interval?**
A: The CI only accounts for uncertainty in estimating the *true regression line*. The PI additionally accounts for the individual random error $\varepsilon$ of the single new observation itself — an extra, irreducible source of noise that never shrinks no matter how much data you collect.

**Q: If I collect more data, does the prediction interval shrink to zero width?**
A: No — as $n \to \infty$, the CI for the mean response does shrink toward zero width, but the PI never shrinks below roughly $\pm z^* \sigma$, because individual-observation noise ($\varepsilon$) is irreducible regardless of sample size.

**Q: In simple linear regression, is $R^2$ the same as $r^2$?**
A: Yes, exactly, in simple linear regression only — this equivalence breaks down in multiple regression with more than one predictor.

---

*End of Chapter 2. Next: Chapter 3 — Matrix Formulation of Regression (vectors/matrices, normal equations in matrix form, the hat matrix, and why everything from Chapters 1–2 is a special case of the general matrix result).*
