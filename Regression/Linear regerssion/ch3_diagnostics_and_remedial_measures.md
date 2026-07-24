# Chapter 3 — Diagnostics and Remedial Measures
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapters 1–2 assumed the model was correctly specified and its assumptions held. Chapter 3 is about **checking those assumptions** — the four from Chapter 1 (linearity/zero-mean errors, constant variance, independence, normality) — and what to do when they fail.

### A New Worked Dataset (with replicates)

To demonstrate diagnostics properly — especially the lack-of-fit test, which *requires* repeated Y observations at the same X — we need a new dataset. The Chapter 1–2 dataset had a unique X per observation and near-perfect fit, which is useless for diagnostics.

| Obs | X | Y |
|---|---|---|
| 1 | 1 | 3 |
| 2 | 1 | 5 |
| 3 | 2 | 8 |
| 4 | 2 | 6 |
| 5 | 3 | 15 |
| 6 | 3 | 13 |
| 7 | 4 | 14 |
| 8 | 4 | 20 |

Notice: two observations at each of X = 1, 2, 3, 4. This "replication" is the key structural feature that lets us separate two different sources of lack-of-fit (explained below).

**Fitting the line (same least-squares method as Ch. 1):**
$$\bar X = 2.5,\quad \bar Y = 10.5, \quad S_{XX}=10,\quad S_{XY}=46$$
$$b_1 = 46/10 = 4.6, \qquad b_0 = 10.5 - 4.6(2.5) = -1.0$$
$$\hat Y_i = -1.0 + 4.6X_i$$

**Fitted values and residuals:**

| X | Y | $\hat Y$ | $e_i = Y-\hat Y$ |
|---|---|---|---|
| 1 | 3 | 3.6 | -0.6 |
| 1 | 5 | 3.6 | 1.4 |
| 2 | 8 | 8.2 | -0.2 |
| 2 | 6 | 8.2 | -2.2 |
| 3 | 15 | 12.8 | 2.2 |
| 3 | 13 | 12.8 | 0.2 |
| 4 | 14 | 17.4 | -3.4 |
| 4 | 20 | 17.4 | 2.6 |

$$SSE = \sum e_i^2 = 0.36+1.96+0.04+4.84+4.84+0.04+11.56+6.76 = 30.4$$
$$df = n-2 = 6,\qquad MSE = 30.4/6 = 5.0667$$

We'll reuse this exact dataset throughout the chapter.

---

## 3.1 Diagnostics for the Predictor Variable

Before even fitting a model, Kutner recommends inspecting X on its own — dot plots, box plots, and sequence plots (if data were collected over time or in some order). **Why:** these catch problems that have nothing to do with the Y relationship at all — e.g., a few extreme X values that will end up having outsized *leverage* on the fit (a concept formalized fully in Chapter 10), or a sequence plot revealing the data were collected in a systematic order that could induce non-independence.

**Interview angle:** this is the statistical version of "always look at your raw data / do EDA before modeling" — a point interviewers love to hear articulated explicitly rather than assumed.

---

## 3.2 Residuals: Properties and Semistudentized Residuals

### Recap and a New Tool

We already know two algebraic facts about residuals ($\sum e_i = 0$, $\sum X_ie_i=0$, Ch. 1). Now we build a *diagnostic* tool from them: the **semistudentized residual**.

**Plain English.** A raw residual like $e_i = -3.4$ is hard to judge in isolation — is that big or small? It depends on the overall noise level in the data. We need to standardize it.

**Definition:**
$$
e_i^* = \frac{e_i}{\sqrt{MSE}}
$$

**Why "semi"-studentized, not fully studentized?** Because we're dividing by $\sqrt{MSE}$, a single overall estimate of $\sigma$, rather than by the residual's own precise standard error (which technically varies slightly by observation — this refinement, using the *hat matrix* to get each residual's exact variance, is developed fully in Chapter 10 under the name "studentized residuals" / "studentized deleted residuals"). At this stage in the book, we use the simpler, approximate version.

**Rule of thumb (Kutner):** semistudentized residuals larger than about 4 in absolute value are worth flagging as potential outliers (roughly analogous to a z-score cutoff, though residuals aren't exactly standard normal even under ideal conditions with small n).

**Worked example:**
$$
\sqrt{MSE} = \sqrt{5.0667} = 2.2509
$$

| $e_i$ | $e_i^* = e_i/2.2509$ |
|---|---|
| -0.6 | -0.267 |
| 1.4 | 0.622 |
| -0.2 | -0.089 |
| -2.2 | -0.977 |
| 2.2 | 0.977 |
| 0.2 | 0.089 |
| -3.4 | -1.510 |
| 2.6 | 1.155 |

None exceed 4 in magnitude — no observation stands out as a flagrant outlier here, even though $-3.4$ (at X=4) is the largest raw residual.

**Interview trap:** People sometimes flag the largest raw residual as "the outlier" without standardizing first. A residual of -3.4 sounds big, but relative to this dataset's overall noise level (MSE ≈ 5.07), it's only about 1.5 standard deviations — unremarkable. Always standardize before judging magnitude.

---

## 3.3 Diagnostics for Residuals: What Each Plot Reveals

Kutner walks through a series of specific residual plots, each targeting a specific assumption. This is the practical heart of the chapter — **know what each plot looks like under a violation**, since interviewers frequently show a residual plot and ask "what's wrong here?"

1. **Residuals vs. fitted values ($\hat Y_i$), or vs. X:**
 - **Random scatter around zero, no pattern** → assumptions look fine.
 - **Curved pattern (e.g., U-shape or arc)** → the true relationship is **nonlinear**; a straight line is the wrong functional form. Remedy: add a polynomial term, or transform X and/or Y (Section 3.9).
 - **"Funnel" or "megaphone" shape (spread increasing or decreasing with fitted value)** → **non-constant variance** (heteroscedasticity). Remedy: variance-stabilizing transformation (e.g., log(Y)) or weighted least squares.

2. **Residuals vs. time/sequence order (if applicable):**
 - **Cyclical or trending pattern** → **non-independent errors** (autocorrelation) — very common in time series. Remedy: covered in Ch. 12 (time series methods, e.g., adding lagged terms, or Cochrane-Orcutt procedure).

3. **Q-Q plot of residuals (or normal probability plot):**
 - Residuals should fall roughly along a straight diagonal line if normally distributed.
 - **S-shaped deviation** → heavier or lighter tails than Normal (common); **skew** → asymmetric errors. Remedy: often addressed via transformation of Y, or by relying on large-sample robustness (recall: normality is needed mainly for exact small-sample inference, less so for point estimation itself).

4. **Residuals vs. omitted variables:** if you have another candidate predictor not yet in the model, plotting residuals against it can reveal it should be included (a pattern there means the current model is missing explanatory signal) — this foreshadows multiple regression (Ch. 6+) directly.

**Worked example — our dataset's residual-vs-X plot (described in words, since we're not rendering an image):** plotting $e_i$ against $X_i$: at X=1, residuals are (-0.6, 1.4); at X=2, (-0.2,-2.2); at X=3, (2.2,0.2); at X=4, (-3.4,2.6). There's no obvious funnel or curve visible by eye with just 8 points — consistent with the formal tests below, which also fail to detect significant problems. (With so few points, formal tests have low power — a real caveat worth stating explicitly in an interview: failing to detect a violation with n=8 is weak evidence of no violation.)

---

## Tests for Constancy of Variance: the Brown-Forsythe (Modified Levene) Test

### Why We Need a Formal Test, Not Just Eyeballing Plots

**Plain English.** Residual plots are useful but subjective — different people can disagree on whether a funnel shape is "real" or just visual noise, especially with small samples. The Brown-Forsythe test formalizes the question: *does the spread of residuals differ systematically between low-X and high-X observations?*

### The Procedure

1. Split the data into two groups based on X (e.g., low half vs. high half).
2. Within each group, compute each residual's **absolute deviation from its group's median residual**:
$$
d_{ij} = |e_{ij} - \tilde e_j|
$$
(using the *median*, not the mean, is what makes this version robust to outliers/non-normality — this is the "Brown-Forsythe" modification of the older Levene test, which used group means.)
3. Run a two-sample t-test comparing $\bar d_1$ vs. $\bar d_2$ across the two groups. If variances truly differ between groups, the average absolute deviations will differ too.

$$
t_{BF}^* = \frac{\bar d_1 - \bar d_2}{s\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \qquad s^2 = \frac{\sum(d_{1j}-\bar d_1)^2+\sum(d_{2j}-\bar d_2)^2}{n-2}
$$

Compare $|t_{BF}^*|$ to $t_{(1-\alpha/2;\,n-2)}$.

### Worked Example (our 8-point dataset, split at X ≤ 2 vs. X ≥ 3)

**Group 1 (X=1,1,2,2), residuals:** $-0.6, 1.4, -0.2, -2.2$. Sorted: $-2.2,-0.6,-0.2,1.4$; median $=\frac{-0.6+(-0.2)}{2}=-0.4$.
$$
d_{1}: |-0.6-(-0.4)|=0.2,\ |1.4-(-0.4)|=1.8,\ |-0.2-(-0.4)|=0.2,\ |-2.2-(-0.4)|=1.8
$$
$$
\bar d_1 = \frac{0.2+1.8+0.2+1.8}{4}=1.0
$$

**Group 2 (X=3,3,4,4), residuals:** $2.2, 0.2, -3.4, 2.6$. Sorted: $-3.4,0.2,2.2,2.6$; median $=\frac{0.2+2.2}{2}=1.2$.
$$
d_{2}: |2.2-1.2|=1.0,\ |0.2-1.2|=1.0,\ |-3.4-1.2|=4.6,\ |2.6-1.2|=1.4
$$
$$
\bar d_2 = \frac{1.0+1.0+4.6+1.4}{4}=2.0
$$

**Pooled variance:**
$$
\sum(d_{1j}-1.0)^2 = 0.64+0.64+0.64+0.64=2.56
$$
$$
\sum(d_{2j}-2.0)^2 = 1.0+1.0+6.76+0.36=9.12
$$
$$
s^2 = \frac{2.56+9.12}{6}=1.9467,\qquad s=1.3952
$$

**Test statistic:**
$$
t_{BF}^* = \frac{1.0-2.0}{1.3952\sqrt{1/4+1/4}} = \frac{-1.0}{1.3952(0.7071)}=\frac{-1.0}{0.9866}=-1.014
$$

Critical value: $t_{(0.975;6)}=2.447$. Since $|-1.014| < 2.447$, **we fail to reject $H_0$: no significant evidence of non-constant variance.**

**Interpretation caveat:** group 2's average absolute deviation (2.0) was numerically double group 1's (1.0), which might look concerning — but with only 4 observations per group, the test has very limited power to detect real differences. This is a good, honest thing to say out loud in an interview: statistical non-significance with small n is not proof of no effect.

**Interview question:** *"Why does Brown-Forsythe use the median instead of the mean when computing deviations, and why does that matter?"*
**Ideal answer:** Using the median makes the test robust to skewness and outliers within each group — a single extreme residual would distort a mean-based deviation measure much more than a median-based one, potentially triggering false detection of heteroscedasticity that's really just being driven by one bad point. This robustness is why Brown-Forsythe is generally preferred over the original Levene test in practice.

*(Kutner also covers the **Breusch-Pagan test** as an alternative — it instead regresses squared residuals on X and tests whether that regression's slope is significantly nonzero, using a chi-square test. It's more powerful under normality but less robust to non-normal errors than Brown-Forsythe. Worth knowing both names and being able to state the core tradeoff: Breusch-Pagan assumes normality of errors more heavily; Brown-Forsythe is nonparametric/robust.)*

---

## F Test for Lack of Fit

### Why This Test Requires Replicated X Values

**Plain English.** SSE (the total residual variation) can come from two different sources: (1) genuine random noise around the *true* mean at each X (**pure error** — irreducible even for a perfectly correctly specified model), and (2) the *model being the wrong functional form* (**lack of fit** — e.g., forcing a straight line onto a curved true relationship). Normally we can't separate these two sources — but if we have **repeated Y observations at the same X value**, we can, because the spread *among replicates at the same X* can only be pure error (X is literally identical, so any difference between those Y's has nothing to do with the model's functional form).

### The Decomposition

$$
SSE = SSPE + SSLF
$$
- **SSPE** (Pure Error Sum of Squares): sum, over each distinct X level, of squared deviations of that group's Y's from their own group mean.
- **SSLF** (Lack of Fit Sum of Squares): $SSE - SSPE$ — whatever residual variation is *not* explained by pure replication noise; if the true relationship really is linear, this should be small (statistically indistinguishable from pure error, adjusted for df).

$$
F^*_{LF} = \frac{SSLF/(df_{LF})}{SSPE/(df_{PE})} = \frac{MSLF}{MSPE}
$$

where $df_{PE} = \sum_j(n_j - 1)$ (across the $c$ distinct X levels), and $df_{LF} = (n-2) - df_{PE}$.

**Under $H_0$ (linear model is correct):** $F^*_{LF} \sim F_{(df_{LF},\,df_{PE})}$. A large $F^*_{LF}$ means the lack-of-fit component is large relative to pure noise — evidence the linear model is **misspecified**.

### Worked Example

Group means by X level: X=1 → mean 4 (Y=3,5); X=2 → mean 7 (Y=8,6); X=3 → mean 14 (Y=15,13); X=4 → mean 17 (Y=14,20).

**Pure error deviations:**
- X=1: $3-4=-1,\ 5-4=1$ → squares $1,1$
- X=2: $8-7=1,\ 6-7=-1$ → squares $1,1$
- X=3: $15-14=1,\ 13-14=-1$ → squares $1,1$
- X=4: $14-17=-3,\ 20-17=3$ → squares $9,9$

$$
SSPE = 1+1+1+1+1+1+9+9 = 24
$$
$$
df_{PE} = (2-1)\times 4 = 4
$$

$$
SSLF = SSE - SSPE = 30.4 - 24 = 6.4, \qquad df_{LF} = (n-2)-df_{PE} = 6-4=2
$$
$$
MSLF = 6.4/2 = 3.2, \qquad MSPE = 24/4 = 6.0
$$
$$
F^*_{LF} = \frac{3.2}{6.0} = 0.533
$$

Critical value: $F_{(0.95;\,2,4)} = 6.94$. Since $0.533 \ll 6.94$, **we fail to reject $H_0$ — no significant evidence of lack of fit; the straight-line model is an adequate functional form for this data.**

**Interview question:** *"How would you test whether a linear model is missing important curvature, if you have replicated observations at the same X values?"*
**Ideal answer:** Decompose SSE into pure error (variation among Y's at identical X, which can only be irreducible noise) and lack-of-fit (the remainder). If SSE has a large lack-of-fit component relative to pure error (large F* in an F-test comparing MSLF to MSPE), that's direct statistical evidence the linear functional form is wrong — as opposed to residual plots, which are suggestive but subjective, this test gives a formal p-value specifically for "is the straight line the right shape."

**Important limitation to state explicitly:** this test *requires* replicated X values — in most real-world regression settings (especially with continuous predictors, e.g., recommendation system features), you rarely have exact replicates, so this exact test isn't directly usable; residual plots and comparing nested polynomial models (Section 3.9's spirit, and the general linear test from Ch. 2.8) become the practical substitutes.

---

## 3.9 Remedial Measures: Transformations

When diagnostics reveal a problem, Kutner's default remedies are **transformations** of X and/or Y (before reaching for more complex methods like weighted least squares or robust regression, covered later in the book).

### Transformations to Fix Nonlinearity (transform X)

If the residual-vs-X plot shows curvature but the variance looks roughly constant, transform **X** (this doesn't change the error variance structure, which was already fine): e.g., $X' = \sqrt{X}$, $X' = \log(X)$, $X' = 1/X$, or add $X^2$ as a second predictor (this last option becomes polynomial regression — technically already a form of *multiple* regression, foreshadowing Ch. 6+).

### Transformations to Fix Non-Constant Variance (transform Y)

If the funnel shape appears (variance increasing with fitted value, common when Y is a count or a strictly positive skewed quantity), transform **Y**: common choices are $\sqrt{Y}$ (good when variance grows roughly linearly with the mean, e.g., Poisson-like count data), $\log(Y)$ (good when variance grows roughly with the square of the mean, e.g., multiplicative/percentage-type noise — very common for financial or user-engagement metrics), or $1/Y$ (for even more extreme variance growth).

### The Box-Cox Family: Choosing a Transformation Systematically, Rather Than by Trial and Error

**Plain English.** Instead of guessing which transformation (sqrt? log? reciprocal?) works best, Box-Cox gives a formal, data-driven procedure to find the "best" power transformation of Y.

**The Box-Cox transformation family:**
$$
Y^{(\lambda)} = \begin{cases} \dfrac{Y^\lambda - 1}{\lambda} & \lambda \ne 0 \\ \ln(Y) & \lambda = 0 \end{cases}
$$

**Why this specific form?** It's constructed so the transformation is continuous in $\lambda$ — as $\lambda \to 0$, $\frac{Y^\lambda-1}{\lambda}$ mathematically approaches $\ln(Y)$ (a limit that can be shown via L'Hôpital's rule), which is why $\lambda=0$ is defined as the log case. This lets you treat $\lambda$ as a single continuous "dial": $\lambda=1$ is no transformation (a shifted identity), $\lambda=0.5$ is roughly a square-root transform, $\lambda=0$ is log, $\lambda=-1$ is roughly a reciprocal transform.

**Procedure:** for a grid of candidate $\lambda$ values, fit the regression using $Y^{(\lambda)}$ as the response, and compute the resulting SSE (or equivalently, the log-likelihood). Choose the $\lambda$ that **minimizes SSE** (maximizes likelihood) — this is itself an MLE procedure, directly connecting back to Chapter 1.11's normal-MLE framework.

**Interview trap:** People sometimes apply Box-Cox blindly to any dataset without checking that Y is strictly positive (required, since $Y^\lambda$ and $\ln Y$ are undefined/complex for $Y\le0$) — a common practical gotcha, often handled by adding a small constant shift if zeros are present.

**Interview question:** *"You notice a funnel-shaped residual plot in a Poisson-like count regression. What transformation would you try first, and why?"*
**Ideal answer:** A square-root transformation of Y is a natural first choice, since for count data following roughly a Poisson process, the variance is approximately equal to the mean — the square-root transform is the classic variance-stabilizing transform for that mean-variance relationship. More generally, one could run a Box-Cox procedure over a grid of λ to find the transformation empirically minimizing SSE, rather than guessing a single fixed form.

---

## Python Implementation — From Scratch (NumPy + SciPy)

```python
import numpy as np
from scipy import stats

X = np.array([1,1,2,2,3,3,4,4], dtype=float)
Y = np.array([3,5,8,6,15,13,14,20], dtype=float)
n = len(X)

X_bar, Y_bar = X.mean(), Y.mean()
Sxx = np.sum((X-X_bar)**2)
Sxy = np.sum((X-X_bar)*(Y-Y_bar))
b1 = Sxy/Sxx
b0 = Y_bar - b1*X_bar
Y_hat = b0 + b1*X
resid = Y - Y_hat
SSE = np.sum(resid**2)
df_e = n-2
MSE = SSE/df_e

# --- Semistudentized residuals ---
semistud = resid / np.sqrt(MSE)
print("Semistudentized residuals:", np.round(semistud, 3))

# --- Lack of fit F-test (requires replicated X) ---
unique_X = np.unique(X)
SSPE = 0
df_pe = 0
for x_val in unique_X:
    group_Y = Y[X == x_val]
    SSPE += np.sum((group_Y - group_Y.mean())**2)
    df_pe += len(group_Y) - 1

SSLF = SSE - SSPE
df_lf = df_e - df_pe
MSLF = SSLF/df_lf
MSPE = SSPE/df_pe
F_lof = MSLF/MSPE
F_crit = stats.f.ppf(0.95, df_lf, df_pe)
print(f"SSPE={SSPE}, SSLF={SSLF}, F*_LF={F_lof:.3f}, F_crit={F_crit:.3f}")

# --- Brown-Forsythe test ---
mask1 = X <= 2
mask2 = X >= 3
d1 = np.abs(resid[mask1] - np.median(resid[mask1]))
d2 = np.abs(resid[mask2] - np.median(resid[mask2]))
n1, n2 = len(d1), len(d2)
pooled_var = (np.sum((d1-d1.mean())**2) + np.sum((d2-d2.mean())**2)) / (n1+n2-2)
s = np.sqrt(pooled_var)
t_bf = (d1.mean() - d2.mean()) / (s*np.sqrt(1/n1+1/n2))
t_crit = stats.t.ppf(0.975, n1+n2-2)
print(f"t*_BF={t_bf:.3f}, t_crit={t_crit:.3f}")

# --- Box-Cox (using scipy, requires Y > 0) ---
Y_bc, best_lambda = stats.boxcox(Y)
print(f"Best Box-Cox lambda: {best_lambda:.3f}")
```

---

## Interview Question Bank — Chapter 3

**Conceptual:**
1. Name the four core regression assumptions and the specific residual plot used to check each.
2. What does a "funnel"-shaped residual plot indicate, and what's the standard remedy?
3. Why do semistudentized residuals give a more meaningful sense of "how large" a residual is than the raw residual?

**Derivation:**
4. Derive why $SSE = SSPE + SSLF$, and explain intuitively why replicated X values are required to separate them.
5. Explain why the Box-Cox transformation is defined as $\ln(Y)$ specifically at $\lambda=0$ (connect to the limit of $(Y^\lambda-1)/\lambda$).

**ML/Statistics:**
6. Compare Brown-Forsythe and Breusch-Pagan tests for heteroscedasticity — when would you prefer one over the other?
7. In a real ML feature pipeline, you rarely have exact replicated X values. How would you adapt the "lack of fit" idea to test if a linear model is missing curvature?
8. Why is finding a high R² not sufficient evidence that a linear model is correctly specified — connect this to the lack-of-fit test specifically.

**Coding:**
9. Implement the pure error / lack-of-fit decomposition from scratch given a dataset with replicated X values.
10. Implement the Brown-Forsythe test from scratch (median-based, two-group).

**Traps:**
11. "The residual at X=4 is -3.4, clearly the biggest outlier in the dataset." What's the correct way to evaluate this claim?
12. "We failed to reject H0 in the Brown-Forsythe test, so we've proven the variance is constant." What's wrong with this conclusion, especially with small n?
13. Someone applies Box-Cox directly to a variable that includes zero values and gets an error. What's going on, and how would you fix it?

---

*This file covers Kutner Ch. 3 — diagnostics for the predictor variable, residual properties and semistudentized residuals, the full catalog of residual-plot diagnoses (nonlinearity, heteroscedasticity, non-independence, non-normality), the Brown-Forsythe test (worked by hand), the F-test for lack of fit using pure error decomposition (worked by hand), and remedial transformations including Box-Cox. Chapter 4 (Simultaneous Inference and Other Topics) is next — Bonferroni joint confidence intervals, regression through the origin, and effects of measurement error in X.*
