# Chapter 2 — Inferences in Regression and Correlation Analysis
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

We continue with the same dataset from Chapter 1 (hours studied X vs. exam score Y, n=5), where we found:
$$b_0 = 56.654,\quad b_1 = 4.269,\quad SSE = 4.1154,\quad MSE = 1.3718,\quad S_{XX}=26,\quad \bar X = 5,\ \bar Y = 78$$

Chapter 1 gave us **point estimates**. Chapter 2 answers: *how confident should we be in those estimates, and how do we test hypotheses about the true β0, β1?* This is where regression stops being "just curve fitting" and becomes formal statistical inference.

---

## 2.1 Inferences Concerning β1 (the slope)

### Why we need a sampling distribution for $b_1$

**Plain English.** $b_1 = 4.269$ is a single number computed from *this* particular sample of 5 students. If we'd sampled a different 5 students, we'd get a different $b_1$ — maybe 4.1, maybe 4.5. $b_1$ is a **random variable** across repeated sampling, not a fixed truth. To say anything about the *true* unknown $\beta_1$, we need to understand how much $b_1$ bounces around from sample to sample.

**Why it exists.** Without knowing the sampling variability of $b_1$, a number like "4.269" is meaningless for inference — we can't tell if it's reliably different from, say, 0 (no effect) or is just noise.

### Deriving $\text{Var}(b_1)$

Recall $b_1 = \frac{\sum(X_i-\bar X)(Y_i - \bar Y)}{S_{XX}}$. Since the $X_i$ are treated as fixed constants (not random) in this model, and $Y_i = \beta_0+\beta_1X_i+\varepsilon_i$, we can rewrite $b_1$ as a **linear combination of the $Y_i$'s** (and hence of the $\varepsilon_i$'s):

$$
b_1 = \sum_i k_i Y_i, \quad \text{where } k_i = \frac{X_i - \bar X}{S_{XX}}
$$

(You can verify this by expanding the numerator — $\sum(X_i-\bar X)(Y_i-\bar Y) = \sum(X_i-\bar X)Y_i$ since $\sum(X_i-\bar X)\bar Y = 0$.)

Since $b_1$ is a linear combination of independent random variables $Y_i$ (each with variance $\sigma^2$, by assumption 2 from Ch.1), variance of a sum of independent terms is the sum of variances of each term scaled by its squared coefficient:

$$
\text{Var}(b_1) = \sum_i k_i^2 \text{Var}(Y_i) = \sigma^2 \sum_i k_i^2 = \sigma^2 \sum_i \frac{(X_i-\bar X)^2}{S_{XX}^2} = \frac{\sigma^2 S_{XX}}{S_{XX}^2} = \frac{\sigma^2}{S_{XX}}
$$

**Intuition for this formula — this is important and frequently tested in interviews.** $\text{Var}(b_1) = \sigma^2/S_{XX}$ tells you two things:
- More noise in the data ($\sigma^2$ larger) → less certain about the slope → larger variance. Makes sense.
- **More spread-out X values** ($S_{XX} = \sum(X_i-\bar X)^2$ larger) → **more** certain about the slope → smaller variance. This is a critical, often-missed insight: *spreading out your X values (design of experiment) reduces uncertainty in the slope estimate*, even with the same sample size and same noise level.

**Interview question:** *"If you could choose where to place your X observations before collecting data, how would you minimize the variance of your slope estimate?"*
**Ideal answer:** Push the X values as far apart as possible (maximize $S_{XX} = \sum(X_i - \bar X)^2$) — e.g., put half your observations at the lowest feasible X and half at the highest feasible X, rather than clustering them near the mean. This is a foundational idea in optimal experimental design.

### Since $\sigma^2$ is Unknown: the Standard Error and the t-distribution

We don't know $\sigma^2$; we estimate it with $MSE$ from Ch. 1. So we estimate:

$$
s^2(b_1) = \frac{MSE}{S_{XX}}, \qquad s(b_1) = \sqrt{s^2(b_1)}
$$

**Why does this force us into a t-distribution instead of a Normal distribution?** Because we're now dividing by an *estimated* standard deviation ($s(b_1)$, itself a random variable depending on the data) rather than the *true* $\sigma$. This extra source of randomness makes the distribution of $\frac{b_1-\beta_1}{s(b_1)}$ have heavier tails than Normal — precisely the t-distribution, with $n-2$ degrees of freedom (matching the df used in MSE).

$$
\frac{b_1 - \beta_1}{s(b_1)} \sim t_{(n-2)}
$$

**Confidence interval for $\beta_1$:**
$$
b_1 \pm t_{(1-\alpha/2;\,n-2)}\cdot s(b_1)
$$

**Hypothesis test** (most common: $H_0: \beta_1 = 0$ vs $H_a: \beta_1 \ne 0$ — "is there any linear association at all?"):
$$
t^* = \frac{b_1 - 0}{s(b_1)}, \quad \text{reject } H_0 \text{ if } |t^*| > t_{(1-\alpha/2;\,n-2)}
$$

### Worked Numerical Example

$$
s^2(b_1) = \frac{MSE}{S_{XX}} = \frac{1.3718}{26} = 0.05276 \quad\Rightarrow\quad s(b_1) = \sqrt{0.05276} = 0.2297
$$

For a 95% CI with $n-2 = 3$ df, $t_{(0.975; 3)} = 3.182$ (from t-table).

$$
b_1 \pm t\cdot s(b_1) = 4.269 \pm 3.182(0.2297) = 4.269 \pm 0.731 \Rightarrow (3.538,\ 5.000)
$$

**Interpretation:** We are 95% confident the true slope β1 (true effect of one more hour of study on mean exam score) lies between about 3.54 and 5.00 points.

**Test $H_0: \beta_1 = 0$:**
$$
t^* = \frac{4.269}{0.2297} = 18.59
$$
Since $18.59 \gg 3.182$, we reject $H_0$ at α=0.05 — strong evidence of a real linear relationship between hours studied and score. (With only n=5, this is a toy example — real studies need larger samples, but the mechanics are identical.)

**Common misconception:** A very large $t^*$ (like 18.59) is often misread as "the effect is huge." It actually measures *how many standard errors away from zero* the estimate is — a signal of statistical confidence, not effect size. With n=5 and low noise, even a modest slope could look "highly significant." Always report the CI (which has the actual units — points per hour) alongside or instead of the t-statistic, for practical/effect-size interpretation.

---

## 2.2 Inferences Concerning β0 (the intercept)

Following exactly the same logic, but now the variance formula is more complex because $b_0 = \bar Y - b_1\bar X$ involves both $\bar Y$ and $b_1$:

$$
\text{Var}(b_0) = \sigma^2\left[\frac{1}{n} + \frac{\bar X^2}{S_{XX}}\right]
$$

**Intuition:** two sources of uncertainty combine — uncertainty in $\bar Y$ itself ($\sigma^2/n$, ordinary sampling error of a mean) plus uncertainty propagated from the slope $b_1$ ($\bar X^2/S_{XX}$ term, since $b_0$ depends on $b_1$ scaled by $\bar X$). Notice: **if $\bar X = 0$**, this second term vanishes and $\text{Var}(b_0) = \sigma^2/n$ exactly — the intercept's uncertainty comes purely from estimating the mean. This is *why some practitioners center X (subtract $\bar X$) before fitting* — it decouples the intercept's precision from the slope's uncertainty, and it also makes $b_0$ directly interpretable as the estimated mean of Y (rather than an extrapolated, often meaningless, "Y when X=0" value).

Estimated: $s^2(b_0) = MSE\left[\frac{1}{n}+\frac{\bar X^2}{S_{XX}}\right]$, and $\frac{b_0-\beta_0}{s(b_0)}\sim t_{(n-2)}$, same mechanics as before.

**Worked example:**
$$
s^2(b_0) = 1.3718\left[\frac{1}{5}+\frac{25}{26}\right] = 1.3718(0.2+0.9615) = 1.3718(1.1615)=1.594
$$
$$
s(b_0) = \sqrt{1.594} = 1.2626
$$
95% CI: $56.654 \pm 3.182(1.2626) = 56.654\pm 4.018 \Rightarrow (52.636,\ 60.672)$.

**Interview trap:** People often skip testing/interpreting $\beta_0$ entirely because "it's just where the line crosses the axis." But in some domains (e.g., manufacturing baseline defect rate at X=0 raw material impurity), the intercept has real, important meaning. In others (e.g., X = temperature in Celsius, and 0°C is far outside your data range), it's purely a mathematical artifact and should not be over-interpreted — this is exactly the **extrapolation danger** flagged back in Chapter 1's worked example.

---

## 2.3 Considerations in Making Inferences on β0, β1

Kutner flags several practical cautions here:

1. **Do not extrapolate.** Inferences about the regression relationship only hold within the range of X actually observed in the data (in our example, 2 to 8 hours). Predicting Y at X=20 hours is extrapolation — the true relationship could be nonlinear outside the observed range, and we have zero data to check.

2. **Inferences on β0 and β1 are not independent** in general — they're often *correlated* (unless $\bar X = 0$), because both are computed from the same data and $b_0$ depends algebraically on $b_1$. This means you can't just combine separate CIs for β0 and β1 into a valid joint confidence region without adjustment (this connects forward to Chapter 4's discussion of simultaneous/joint confidence regions).

3. **Statistical significance ≠ practical significance**, as flagged above with the t* = 18.59 example.

---

## 2.4 Interval Estimation of $E\{Y_h\}$ — the Mean Response at a Given $X_h$

### Plain English

Distinct question from estimating β0, β1 individually: "What is the *mean* exam score for **all** students who study $X_h$ hours?" — not any one student's score, but the average across the (hypothetical) population of students at that X level.

**Point estimator:** $\hat Y_h = b_0 + b_1 X_h$ (plug $X_h$ into the fitted line).

### Deriving the Variance

$\hat Y_h = b_0 + b_1 X_h$ is again a linear combination of the $Y_i$'s (since both $b_0$ and $b_1$ are). Carrying through the variance algebra (Kutner shows this in detail; it uses $\text{Var}(b_0)$, $\text{Var}(b_1)$, and their covariance):

$$
\text{Var}(\hat Y_h) = \sigma^2\left[\frac{1}{n} + \frac{(X_h - \bar X)^2}{S_{XX}}\right]
$$

**Intuition — critical for interviews:** The variance is **smallest when $X_h = \bar X$** (predicting near the center of your data) and **grows the farther $X_h$ is from $\bar X$**. This is the mathematical basis for why confidence bands around a fitted regression line are "hourglass" or "bowtie" shaped — narrow in the middle of the data, flaring out at the edges. It's also a formal, quantified version of "don't trust predictions far from your data" — even *within* the observed range, precision degrades as you move away from the center, and it degrades much faster once you leave the observed range entirely (extrapolation).

**Estimated variance & CI:**
$$
s^2(\hat Y_h) = MSE\left[\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{XX}}\right], \qquad \hat Y_h \pm t_{(1-\alpha/2;n-2)}\, s(\hat Y_h)
$$

### Worked Example: mean response at $X_h = 6$ hours

$$
\hat Y_h = 56.654 + 4.269(6) = 56.654+25.615 = 82.269
$$
$$
s^2(\hat Y_h) = 1.3718\left[\frac{1}{5}+\frac{(6-5)^2}{26}\right]=1.3718(0.2+0.03846)=1.3718(0.23846)=0.3272
$$
$$
s(\hat Y_h)=0.5720
$$
95% CI: $82.269\pm3.182(0.5720)=82.269\pm1.820\Rightarrow(80.449,\ 84.089)$

**Interpretation:** We're 95% confident the *true mean* exam score for the population of all students who study 6 hours lies between about 80.4 and 84.1.

Compare this to $X_h = \bar X = 5$: $s^2(\hat Y_h)=MSE(1/5+0)=0.2744$, narrower than at $X_h=6$ — confirming the "narrowest at the center" intuition directly.

---

## 2.5 Prediction of a New Observation $Y_{h(new)}$

### The Crucial Distinction from 2.4

**Plain English.** Section 2.4 asked about the *mean* of many students at $X_h$. Now we ask: what score do we predict for **one specific new student** who studies $X_h$ hours? This new student's actual score won't be exactly the mean — it will also have their own individual error $\varepsilon$ around that mean.

**Why the interval must be wider.** A prediction interval must account for **two sources of uncertainty**, not one:
1. Uncertainty in estimating the *true mean* $E\{Y_h\}$ (same as Section 2.4).
2. The new observation's *own* random deviation from that true mean ($\varepsilon_{new}$, with variance $\sigma^2$), which is a **completely separate, additional source of randomness** — this individual student's own luck/effort/noise, uncorrelated with anything used to fit the model.

$$
\text{Var}(\text{pred. error}) = \underbrace{\sigma^2\left[\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{XX}}\right]}_{\text{uncertainty in }\hat Y_h} + \underbrace{\sigma^2}_{\text{new obs.'s own noise}} = \sigma^2\left[1+\frac{1}{n}+\frac{(X_h-\bar X)^2}{S_{XX}}\right]
$$

**This "+1" is one of the single most commonly tested concepts in ML/statistics interviews.** It is exactly the difference between a **confidence interval** (for a mean/parameter) and a **prediction interval** (for a new individual data point) — prediction intervals are always wider, and remain wide even as $n \to \infty$ (the $1/n$ and $(X_h-\bar X)^2/S_{XX}$ terms shrink toward 0, but the leading "1" never vanishes, because you can never eliminate the new observation's own inherent randomness $\sigma^2$).

**Worked example**, same $X_h=6$:
$$
s^2(pred) = MSE\left[1+\frac{1}{5}+\frac{1}{26}\right]=1.3718(1+0.2+0.03846)=1.3718(1.23846)=1.6993
$$
$$
s(pred)=1.3036
$$
95% PI: $82.269\pm3.182(1.3036)=82.269\pm4.149\Rightarrow(78.120,\ 86.418)$

Compare: the 95% **confidence** interval for the mean at X=6 was $(80.449, 84.089)$ — width 3.64. The 95% **prediction** interval for one new student at X=6 is $(78.120, 86.418)$ — width 8.30, more than double. This gap is exactly the extra $\sigma^2$ term.

**Interview question:** *"What's the difference between a confidence interval and a prediction interval in regression, and why is the prediction interval always wider?"*
**Ideal answer:** A confidence interval quantifies uncertainty about the *true mean response* at a given X — it only reflects estimation error in the fitted line, and shrinks toward zero width as sample size grows. A prediction interval quantifies uncertainty about a *single new observation*, which must additionally account for that individual's own irreducible random deviation from the mean (variance σ²) — a term that never vanishes, even with infinite data, because individual-level noise is inherent to the data-generating process, not an artifact of estimation.

**Interview trap:** In production ML systems, when people report "confidence intervals" around individual predictions (e.g., a single user's predicted lifetime value), they very often actually want **prediction intervals**, not confidence intervals — using the narrower CI formula understates uncertainty for individual-level predictions. This mislabeling is extremely common in industry and is a great thing to point out in an interview.

---

## 2.6 Confidence Band for the Entire Regression Line (Working–Hotelling)

Sections 2.4–2.5 gave intervals at *one specific* $X_h$. If you want a band that covers the **entire true regression line simultaneously**, at every X value at once, with confidence $1-\alpha$, you can't just use the pointwise $t$ multiplier — you need a larger multiplier (the Working–Hotelling $W$ statistic, based on the F-distribution) to correct for looking at infinitely many X values simultaneously:

$$
\hat Y_h \pm W\cdot s(\hat Y_h), \qquad W = \sqrt{2F_{(1-\alpha;\,2,\,n-2)}}
$$

**Intuition:** this is the same "multiple comparisons" logic that shows up everywhere in statistics (and foreshadows Chapter 4's Bonferroni procedures) — the more things you're simultaneously confident about, the wider each individual interval must be to maintain the overall confidence level.

We won't hand-compute this one in depth now — Kutner develops it more fully alongside Chapter 4's simultaneous inference material, where we'll return to it.

---

## 2.7 ANOVA Approach to Regression

### The Big Idea: Partitioning Total Variability

**Plain English.** Total variation in Y (ignoring X entirely) can be split into two pieces: the part *explained* by the linear relationship with X, and the part left over (*unexplained*, i.e., residual/error).

$$
\underbrace{\sum(Y_i-\bar Y)^2}_{SSTO} = \underbrace{\sum(\hat Y_i - \bar Y)^2}_{SSR} + \underbrace{\sum(Y_i-\hat Y_i)^2}_{SSE}
$$

- **SSTO** (Total Sum of Squares): total variability in Y around its own mean, ignoring X completely.
- **SSR** (Regression Sum of Squares): variability in Y "explained" by the fitted line — how much the fitted values $\hat Y_i$ vary around $\bar Y$.
- **SSE** (Error Sum of Squares): what's left over — the same SSE from Chapter 1.

**Why this decomposition is always exactly true (not approximate)** — it follows algebraically from writing $(Y_i-\bar Y) = (\hat Y_i - \bar Y)+(Y_i-\hat Y_i)$, squaring both sides, summing over $i$, and using the fact that the cross-term vanishes because $\sum(\hat Y_i - \bar Y)(Y_i-\hat Y_i) = 0$ (a consequence of the normal equations, similar in spirit to $\sum e_i = 0$ and $\sum X_ie_i=0$ from Chapter 1).

### Degrees of Freedom and the ANOVA Table

| Source | SS | df | MS |
|---|---|---|---|
| Regression | SSR | 1 | MSR = SSR/1 |
| Error | SSE | n-2 | MSE = SSE/(n-2) |
| Total | SSTO | n-1 | |

**Why df(Regression)=1:** with one predictor X, there's exactly one slope parameter being estimated beyond the mean, so 1 df is "spent" explaining variability with X.

### The F-test for $H_0: \beta_1 = 0$

$$
F^* = \frac{MSR}{MSE} \sim F_{(1,\,n-2)} \text{ under } H_0
$$

**Key theoretical fact (very testable in interviews): for simple linear regression, $F^* = (t^*)^2$** — the F-test and the two-sided t-test for $\beta_1=0$ are mathematically equivalent, always giving the same p-value. This is a special property of the single-predictor case (in multiple regression, F-tests and individual t-tests answer *different* questions — this distinction becomes crucial in Chapter 6+).

### Worked Example

$$
SSTO = \sum(Y_i-\bar Y)^2 = (-13)^2+(-8)^2+0^2+7^2+14^2 = 169+64+0+49+196=478
$$
$$
SSE = 4.1154 \ (\text{from Ch. 1}), \qquad SSR = SSTO - SSE = 478-4.1154=473.8846
$$

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Regression | 473.885 | 1 | 473.885 | 345.4 |
| Error | 4.115 | 3 | 1.372 | |
| Total | 478.000 | 4 | | |

Check: $t^{*2} = 18.59^2 = 345.6$ ✓ (matches $F^*=345.4$, small gap purely from rounding in $b_1, s(b_1)$).

---

## 2.9 Descriptive Measures of Linear Association: $R^2$ and $r$

$$
R^2 = \frac{SSR}{SSTO} = 1 - \frac{SSE}{SSTO}
$$

**Plain English.** $R^2$ is the **proportion of total variability in Y explained by the linear relationship with X** — ranges from 0 (X explains nothing) to 1 (X explains everything, all points on the line).

**Worked example:** $R^2 = 473.885/478 = 0.9914$ — about 99.1% of the variation in exam scores is explained by hours studied (unrealistically high for a real dataset — this is a tiny toy n=5 example; real R² values in behavioral/social data are often far lower, e.g., 0.2–0.5).

**Correlation coefficient:** $r = \pm\sqrt{R^2}$, sign matching the sign of $b_1$. Here $r = +\sqrt{0.9914}=0.9957$ (strong positive correlation).

**Common misconceptions (heavily tested in interviews):**
1. **High $R^2$ does not imply the linear model is correctly specified.** A curved relationship can still produce a deceptively high $R^2$ under a straight-line fit if the curvature is mild relative to the noise — you must check residual plots (Ch. 3), not just $R^2$.
2. **$R^2$ is not a measure of prediction accuracy on new data** — it describes in-sample fit. A model can have high in-sample $R^2$ and still generalize poorly (overfitting) — a direct bridge to bias-variance tradeoff conversations in ML interviews.
3. **$r$ (and $R^2$) measure *linear* association only.** Two variables can have a strong nonlinear relationship (e.g., a perfect parabola) and yield $r \approx 0$.
4. **Correlation ≠ causation** — even $r=0.9957$ here doesn't prove studying *causes* better scores (could be confounded by, e.g., underlying student motivation/ability driving both).

**Interview question:** *"Can a model have R² = 0.95 and still be a bad model? Give an example."*
**Ideal answer:** Yes. R² only measures the proportion of variance explained by the linear fit *in-sample*; it says nothing about whether the linearity assumption holds, whether residuals are patterned (indicating a missing nonlinear term), or whether the model generalizes to new data. For instance, fitting a straight line to data generated from a curve can still yield a high R² if the curvature is subtle relative to the noise, while a residual plot would clearly reveal a systematic U-shaped or S-shaped pattern showing the linear model is misspecified.

---

## 2.8 General Linear Test Approach (brief preview)

Kutner introduces a very general logic here that recurs throughout the rest of the book (especially model-building, Ch. 9):

1. Fit the **full model** (with the term(s) in question included), get $SSE(F)$.
2. Fit a **reduced model** (with the term(s) removed, i.e., under $H_0$), get $SSE(R)$.
3. Compare via:
$$
F^* = \frac{[SSE(R)-SSE(F)]/(df_R - df_F)}{SSE(F)/df_F}
$$

For simple linear regression, testing $H_0:\beta_1=0$: the reduced model is just $Y_i = \beta_0+\varepsilon_i$ (a flat line at $\bar Y$), giving $SSE(R) = SSTO$. Plugging in recovers exactly the same $F^*$ from the ANOVA table above — showing the ANOVA F-test is just a special case of this more general full-vs-reduced-model comparison framework. **This general linear test logic is the exact same idea behind nested model comparisons in ML** (e.g., likelihood ratio tests comparing a full neural net to an ablated version) — worth explicitly connecting in interviews.

---

## Python Implementation — From Scratch (NumPy)

```python
import numpy as np
from scipy import stats

X = np.array([2, 3, 5, 7, 8], dtype=float)
Y = np.array([65, 70, 78, 85, 92], dtype=float)
n = len(X)

X_bar, Y_bar = X.mean(), Y.mean()
Sxx = np.sum((X - X_bar) ** 2)
Sxy = np.sum((X - X_bar) * (Y - Y_bar))

b1 = Sxy / Sxx
b0 = Y_bar - b1 * X_bar
Y_hat = b0 + b1 * X
resid = Y - Y_hat

SSE = np.sum(resid ** 2)
df_e = n - 2
MSE = SSE / df_e

# --- Inference on b1 ---
se_b1 = np.sqrt(MSE / Sxx)
t_b1 = b1 / se_b1
t_crit = stats.t.ppf(0.975, df_e)
ci_b1 = (b1 - t_crit * se_b1, b1 + t_crit * se_b1)

# --- Inference on b0 ---
se_b0 = np.sqrt(MSE * (1/n + X_bar**2 / Sxx))
ci_b0 = (b0 - t_crit * se_b0, b0 + t_crit * se_b0)

# --- Mean response CI at Xh ---
Xh = 6
Yhat_h = b0 + b1 * Xh
se_mean = np.sqrt(MSE * (1/n + (Xh - X_bar)**2 / Sxx))
ci_mean = (Yhat_h - t_crit*se_mean, Yhat_h + t_crit*se_mean)

# --- Prediction interval at Xh ---
se_pred = np.sqrt(MSE * (1 + 1/n + (Xh - X_bar)**2 / Sxx))
pi_pred = (Yhat_h - t_crit*se_pred, Yhat_h + t_crit*se_pred)

# --- ANOVA / R^2 ---
SSTO = np.sum((Y - Y_bar) ** 2)
SSR = SSTO - SSE
MSR = SSR / 1
F_star = MSR / MSE
R2 = SSR / SSTO

print(f"b1={b1:.4f}, se(b1)={se_b1:.4f}, t*={t_b1:.3f}, 95% CI={ci_b1}")
print(f"b0={b0:.4f}, se(b0)={se_b0:.4f}, 95% CI={ci_b0}")
print(f"Mean response at Xh={Xh}: {Yhat_h:.3f}, 95% CI={ci_mean}")
print(f"Prediction interval at Xh={Xh}: 95% PI={pi_pred}")
print(f"SSTO={SSTO:.3f}, SSR={SSR:.3f}, SSE={SSE:.3f}, F*={F_star:.2f}, R2={R2:.4f}")
```

## Equivalent via statsmodels (recommended for inference — sklearn doesn't expose these directly)

```python
import statsmodels.api as sm
import numpy as np

X = np.array([2, 3, 5, 7, 8], dtype=float)
Y = np.array([65, 70, 78, 85, 92], dtype=float)

X_design = sm.add_constant(X)  # adds intercept column
model = sm.OLS(Y, X_design).fit()

print(model.summary())
# model.params -> [b0, b1]
# model.bse -> [se(b0), se(b1)]
# model.tvalues, model.pvalues
# model.conf_int() -> CIs for b0, b1
# model.rsquared -> R^2
# model.get_prediction(sm.add_constant([[6]])).summary_frame(alpha=0.05)
#   -> gives BOTH mean_ci_lower/upper (confidence interval)
#      AND obs_ci_lower/upper (prediction interval) in one call
```

---

## Interview Question Bank — Chapter 2

**Conceptual:**
1. Why is $b_1$ treated as a random variable, and what does its sampling distribution represent?
2. Why does spreading out your X values reduce the variance of the slope estimate?
3. What's the practical difference between a confidence interval for $E\{Y_h\}$ and a prediction interval for a new $Y_{h(new)}$?
4. Why does the prediction interval never shrink to zero width even as $n\to\infty$, while the confidence interval does?

**Derivation:**
5. Derive $\text{Var}(b_1) = \sigma^2/S_{XX}$ starting from $b_1 = \sum k_iY_i$.
6. Show why $F^* = (t^*)^2$ in simple linear regression's slope test.
7. Derive the SSTO = SSR + SSE identity and explain why the cross-term vanishes.

**ML/Statistics:**
8. Why is R² not a reliable measure of model quality on its own? Give a scenario where high R² is misleading.
9. Explain the general linear test (full vs. reduced model) approach and connect it to a concept in ML you know (e.g., likelihood ratio tests, ablation studies).
10. Why might centering X (subtracting $\bar X$) before fitting be useful practically?

**Coding:**
11. Implement, from scratch in NumPy, both the confidence interval for the mean response and the prediction interval for a new observation at a given $X_h$.
12. Using statsmodels, extract both types of interval in a single call and explain which columns correspond to which interval type.

**Traps:**
13. "t* = 18.59 for the slope means the effect is huge." — what's wrong with this statement?
14. A colleague reports a 95% "confidence interval" of ±2 points around a single predicted student's score. Is that the right interval type? Why or why not?
15. "Since residuals sum to zero and R² is high, the model assumptions are all satisfied." — what's wrong here (connects back to Ch. 1 residual properties)?

---

*This file covers Kutner Ch. 2, Sections 2.1–2.9 (inference on β0 and β1, confidence intervals for mean response, prediction intervals for new observations, confidence bands, the ANOVA decomposition and F-test, the general linear test framework, and R²/r). Chapter 3 (Diagnostics and Remedial Measures) is next — this is where we formally learn to check the four assumptions from Ch. 1 using residual plots, tests for constancy of variance, normality, and independence, plus remedial transformations when assumptions fail.*
