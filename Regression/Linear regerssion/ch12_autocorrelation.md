# Chapter 12 — Autocorrelation in Time Series Data
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

All prior chapters assumed Chapter 1's third error assumption: $\varepsilon_i$ and $\varepsilon_j$ are uncorrelated for $i\ne j$. This assumption is routinely violated whenever data are collected **sequentially over time** — today's error often resembles yesterday's. This chapter formalizes that problem and its remedies.

### A New Worked Dataset (Time Series)

A simple trend Y (e.g., monthly sales) tracked over 8 consecutive time periods, $X_t=t$:

| t | Y |
|---|---|
| 1 | 14.0 |
| 2 | 15.5 |
| 3 | 17.0 |
| 4 | 17.0 |
| 5 | 18.5 |
| 6 | 21.0 |
| 7 | 25.0 |
| 8 | 27.5 |

(Behind the scenes: true relationship $Y_t=10+2t+\varepsilon_t$, with errors built in *runs* — several consecutive positive deviations, then several consecutive negative ones, then positive again — precisely the pattern positive autocorrelation produces, as opposed to errors that bounce randomly between signs.)

---

## 12.1 Problems Caused by Autocorrelated Errors

### Why OLS Still "Works" But Its Standard Errors Lie to You

**Plain English.** If errors are autocorrelated (today's error predictable from yesterday's), the least-squares coefficient estimates $b_0, b_1$ remain **unbiased** — the derivation of unbiasedness (Chapter 1–2) never actually required the errors to be uncorrelated, only that they average to zero. **What breaks is everything built on top of the variance formulas**: $s^2(b_1)=MSE/S_{XX}$ silently assumes independent errors, and when that assumption fails, this formula is **wrong** — typically **too small** under positive autocorrelation (the most common real-world case), meaning your confidence intervals are falsely narrow and your t-tests falsely significant.

**Why positive autocorrelation shrinks the apparent standard error, intuitively.** With positively autocorrelated errors, consecutive residuals tend to reinforce each other rather than independently canceling out — effectively, your $n$ observations carry **less independent information** than $n$ truly independent observations would. The classical formulas, unaware of this, report a standard error as if you had the full information of $n$ independent draws — overstating your actual precision.

**Interview trap, very commonly missed:** people sometimes think autocorrelation is "only a time-series problem," irrelevant to typical ML feature engineering. But it shows up constantly wherever there's implicit sequential structure the model doesn't know about — e.g., server logs ordered by time, user sessions with within-session dependence, or even A/B test data collected across days with a day-of-week effect not included as a feature. Any residual pattern correlated with an implicit ordering is autocorrelation, whether or not "time series" is in the problem's name.

---

## 12.2 The First-Order Autoregressive Error Model (AR(1))

### Formalizing "Today's Error Resembles Yesterday's"

$$
Y_t = \beta_0+\beta_1X_t+\varepsilon_t, \qquad \varepsilon_t = \rho\,\varepsilon_{t-1}+u_t
$$

**What each new piece means:**
- $\rho$ (rho): the **autocorrelation coefficient**, $-1<\rho<1$. $\rho>0$ means positive autocorrelation (errors tend to persist in sign/direction); $\rho<0$ means errors tend to oscillate; $\rho=0$ recovers the ordinary independent-errors model exactly.
- $u_t$: a fresh, independent "innovation" at each time step — the genuinely new, unpredictable information at time $t$, assumed to satisfy the usual Chapter 1 assumptions ($E[u_t]=0$, constant variance $\sigma_u^2$, uncorrelated across $t$).

**Why this specific structure (rather than some other correlation pattern)?** AR(1) says the error's dependence on the past decays geometrically: $\varepsilon_t$ depends directly on $\varepsilon_{t-1}$, which depends on $\varepsilon_{t-2}$, and so on — so the correlation between errors two steps apart is $\rho^2$, three steps apart is $\rho^3$, etc. This is a parsimonious, single-parameter way to capture "today resembles yesterday, yesterday resembles the day before, with the resemblance fading over time" — a very common real-world pattern, though certainly not the only possible one (Kutner notes higher-order AR models exist for more complex dependence structures, briefly).

---

## 12.3 Detecting Autocorrelation: the Durbin-Watson Test

### The Test Statistic

$$
D = \frac{\sum_{t=2}^n(e_t-e_{t-1})^2}{\sum_{t=1}^n e_t^2}
$$

**Why this specific ratio detects autocorrelation.** If consecutive residuals are similar (positive autocorrelation), their differences $e_t-e_{t-1}$ tend to be **small**, making the numerator small relative to the denominator — so $D$ comes out **close to 0**. If residuals bounce independently, differences are typically as large as the residuals themselves, and $D$ centers around **2**. If residuals alternate sign (negative autocorrelation), differences are *larger* than the residuals, pushing $D$ **toward 4**. **$D$ ranges from 0 to 4, with 2 indicating no autocorrelation** — this range and midpoint are worth memorizing directly, since interviewers often just ask "what does a Durbin-Watson statistic of [some number] tell you" without further context.

### Worked Example: Fitting the Trend and Computing D

**OLS fit** (standard Chapter 1 method, $\bar X=4.5$, $\bar Y=19.4375$, $S_{XX}=42$, $S_{XY}=77.75$):
$$
b_1=\frac{77.75}{42}=1.851, \qquad b_0=19.4375-1.851(4.5)=11.107
$$
$$
\hat Y_t = 11.107+1.851t
$$

**Notice already: the true slope was 2.0, and OLS gives 1.851** — a symptom (not a guarantee) that something beyond pure random noise is influencing the fit.

**Residuals:**

| t | Y | $\hat Y$ | $e_t$ |
|---|---|---|---|
| 1 | 14.0 | 12.958 | **1.042** |
| 2 | 15.5 | 14.809 | **0.691** |
| 3 | 17.0 | 16.661 | **0.339** |
| 4 | 17.0 | 18.512 | **-1.512** |
| 5 | 18.5 | 20.363 | **-1.863** |
| 6 | 21.0 | 22.214 | **-1.214** |
| 7 | 25.0 | 24.065 | **0.935** |
| 8 | 27.5 | 25.917 | **1.583** |

**Look at the sign pattern: + + + − − − + +.** Three straight positives, three straight negatives, two more positives — long runs of the same sign, exactly the visual signature of positive autocorrelation (compare to what pure independent noise would look like: signs flipping unpredictably almost every step).

**Computing D:**
$$
\sum(e_t-e_{t-1})^2 = (-0.351)^2+(-0.352)^2+(-1.851)^2+(-0.351)^2+(0.649)^2+(2.149)^2+(0.648)^2
$$
$$
=0.123+0.124+3.426+0.123+0.421+4.618+0.420=9.256
$$
$$
\sum e_t^2 = 1.086+0.478+0.115+2.286+3.471+1.474+0.874+2.506=12.289
$$
$$
D = \frac{9.256}{12.289}=0.753
$$

**Comparing to critical values** (for $n=8$, one predictor, $\alpha=0.05$: $d_L\approx0.763$, $d_U\approx1.332$, standard published table values): since $D=0.753 < d_L=0.763$, **we're in the "reject $H_0$" region — conclude significant positive autocorrelation.** (If $D$ had landed between $d_L$ and $d_U$, the test would be formally inconclusive — a known, often-criticized limitation of the classical Durbin-Watson test, one reason modern practice often supplements or replaces it with other diagnostics like the Breusch-Godfrey test, which doesn't have this inconclusive zone.)

**Interview question:** *"You compute a Durbin-Watson statistic of 0.75 on your regression residuals. What does that tell you, and what should you NOT conclude from it?"*
**Ideal answer:** A value that low (well below the neutral value of 2, and below the lower critical bound) is strong evidence of **positive** autocorrelation in the residuals — consecutive errors are correlated, likely because of an omitted time-related pattern (trend, seasonality, or genuine serial dependence in the underlying process). What you should *not* conclude is that the coefficient estimates themselves are wrong — they remain unbiased — the real danger is that the *standard errors and significance tests* built on the assumption of independent errors are no longer trustworthy, typically overstating your actual precision.

### Estimating $\rho$ from the Residuals

$$
\hat\rho = \frac{\sum_{t=2}^n e_te_{t-1}}{\sum_{t=1}^n e_t^2}
$$

**Worked example:**
$$
\sum e_te_{t-1} = 0.720+0.234-0.513+2.817+2.262-1.135+1.480=5.866
$$
$$
\hat\rho = \frac{5.866}{12.289}=0.477
$$

A moderately strong estimated autocorrelation of about 0.48 — consistent with the clearly-too-low Durbin-Watson statistic above (a common rough approximation, $D\approx2(1-\hat\rho)$, gives $2(1-0.477)=1.05$; the approximation is asymptotic and doesn't match our small-$n=8$ exact calculation closely, but it captures the right qualitative direction and magnitude).

---

## 12.4 Remedial Measures for Autocorrelation

### The Cochrane-Orcutt Procedure

**Plain English.** If we know (or can estimate) $\rho$, we can construct **transformed variables** that remove the autocorrelation, then apply ordinary least squares to the transformed data — restoring OLS's efficiency and valid standard errors.

$$
Y_t' = Y_t-\hat\rho\,Y_{t-1}, \qquad X_t' = X_t-\hat\rho\,X_{t-1} \qquad (t=2,\ldots,n)
$$

**Why this transformation removes the autocorrelation.** Substituting the AR(1) structure into the model and rearranging shows that regressing $Y_t'$ on $X_t'$ is mathematically equivalent to fitting the **original** model but with the autocorrelated part of the error stripped out, leaving only the well-behaved innovation term $u_t$ — exactly parallel to how Chapter 11's weighted least squares divided out the heteroscedastic part of the variance. (One observation is unavoidably lost, since $Y_1', X_1'$ require a nonexistent $Y_0, X_0$.)

### Worked Example: Applying Cochrane-Orcutt with $\hat\rho=0.477$

Transformed values (constant spacing in $X_t'$ appears automatically, since $X_t'=X_t-0.477X_{t-1}$ and $X_t$ increments by exactly 1 each period):

| t | $X_t'$ | $Y_t'$ |
|---|---|---|
| 2 | 1.523 | 8.818 |
| 3 | 2.045 | 9.602 |
| 4 | 2.568 | 8.886 |
| 5 | 3.091 | 10.386 |
| 6 | 3.614 | 12.170 |
| 7 | 4.136 | 14.977 |
| 8 | 4.659 | 15.568 |

**Fitting OLS on these 7 transformed points** ($\bar X'=3.091$, $\bar Y'=11.487$, using the same Chapter 1 method):
$$
S'_{XX}=7.650, \qquad S'_{XY}=17.921
$$
$$
b_1' = \frac{17.921}{7.650}=2.343, \qquad b_0' = 11.487-2.343(3.091)=4.246
$$

**Transforming the intercept back to the original scale** (the slope needs no adjustment; only the intercept does, since the transformation shifted the effective "origin"):
$$
b_0 = \frac{b_0'}{1-\hat\rho} = \frac{4.246}{1-0.477}=\frac{4.246}{0.523}=8.119
$$

**Cochrane-Orcutt corrected fit:** $\hat Y=8.119+2.343X$ — compare to plain OLS's $11.107+1.851X$, against the true generating values $\beta_0=10, \beta_1=2.0$.

**An honest note worth stating explicitly (good interview instinct):** in this particular tiny sample ($n=8$, only 7 points survive the transformation), the corrected slope (2.343) doesn't land dramatically closer to the true 2.0 than plain OLS's 1.851 did — both are within a similar distance, just on opposite sides. **This is expected, not a failure of the method**: Cochrane-Orcutt's efficiency advantage is a repeated-sampling / asymptotic property (valid standard errors, lower variance *on average across many hypothetical samples*), not a guarantee that any single small sample's point estimate will land closer to the truth. Overclaiming "the corrected method always gives a better answer in this one dataset" is a subtle but real statistical misunderstanding worth avoiding.

**Interview question:** *"Does correcting for autocorrelation (e.g., via Cochrane-Orcutt) guarantee your coefficient estimates get closer to the true values?"*
**Ideal answer:** No — the correction's real benefit is restoring **valid standard errors and improved efficiency in expectation** (lower variance across repeated samples), not a guarantee that any particular sample's point estimate improves. With small samples, sampling variability can easily make an uncorrected estimate look closer to the truth in one specific dataset purely by chance. The value of the correction is in the long-run reliability of your inference (correct confidence intervals, correct Type I error rates), not in guaranteeing improvement every single time.

### Alternative: First Differencing (the $\rho=1$ Special Case)

**Plain English.** If autocorrelation is severe enough that $\rho\approx1$ (errors essentially behave like a random walk, barely reverting at all), the Cochrane-Orcutt transformation simplifies to just **differencing** the data directly: $\Delta Y_t = Y_t-Y_{t-1}$, $\Delta X_t=X_t-X_{t-1}$, then regress $\Delta Y$ on $\Delta X$ (typically without an intercept, since differencing removes any constant term entirely). This is computationally simpler and requires no estimate of $\rho$ at all — you're just assuming it's exactly 1 rather than estimating it. **This is exactly the same "first differencing" used throughout modern time-series/ML practice** to remove trends before modeling (e.g., as a preprocessing step before fitting ARIMA models).

### Hildreth-Lu Procedure (Briefly)

Rather than estimating $\hat\rho$ once from OLS residuals (as Cochrane-Orcutt does), the **Hildreth-Lu procedure** performs a **grid search** over candidate $\rho$ values (e.g., trying $\rho=0.1, 0.2, \ldots, 0.9$), refitting the transformed regression at each candidate, and choosing whichever $\rho$ minimizes the resulting SSE. **Why this can outperform Cochrane-Orcutt:** it doesn't rely on a single, potentially noisy one-shot estimate of $\rho$ from the original (autocorrelation-contaminated) residuals — it directly searches for the best-fitting value. This is conceptually a direct ancestor of hyperparameter grid search in modern ML — the exact same idea (try a grid of candidate values for a nuisance parameter, pick whichever minimizes an error criterion) applied to $\rho$ instead of, say, a regularization strength.

---

## 12.5 Forecasting with Autocorrelated Errors (Brief)

When errors are autocorrelated, a forecast for time $n+1$ can be improved by using the *known* correlation with the most recent observed error: since $\varepsilon_{n+1}=\rho\varepsilon_n+u_{n+1}$, and $E[u_{n+1}]=0$, the best forecast anticipates that some of the most recent residual will "carry over":
$$
\hat Y_{n+1} = b_0+b_1X_{n+1}+\hat\rho\, e_n
$$
**Why this improves on a naive forecast that ignores $e_n$ entirely:** if the last observed residual was, say, strongly positive, AR(1) structure says the *next* one is expected to still be somewhat positive (scaled by $\hat\rho$) — ignoring this known structure and just using $b_0+b_1X_{n+1}$ throws away real, exploitable predictive information sitting in the most recent residual.

---

## Python Implementation — From Scratch and statsmodels

```python
import numpy as np

t = np.arange(1, 9, dtype=float)
Y = np.array([14.0,15.5,17.0,17.0,18.5,21.0,25.0,27.5])
n = len(t)

# --- OLS fit ---
t_bar, Y_bar = t.mean(), Y.mean()
Sxx = np.sum((t-t_bar)**2)
Sxy = np.sum((t-t_bar)*(Y-Y_bar))
b1 = Sxy/Sxx
b0 = Y_bar - b1*t_bar
resid = Y - (b0 + b1*t)
print(f"OLS: b0={b0:.3f}, b1={b1:.3f}")

# --- Durbin-Watson statistic ---
diff = np.diff(resid)
DW = np.sum(diff**2) / np.sum(resid**2)
print(f"Durbin-Watson D = {DW:.4f}")

# --- Estimate rho from residuals ---
rho_hat = np.sum(resid[1:]*resid[:-1]) / np.sum(resid**2)
print(f"rho_hat = {rho_hat:.4f}")

# --- Cochrane-Orcutt transformation ---
Y_prime = Y[1:] - rho_hat*Y[:-1]
X_prime = t[1:] - rho_hat*t[:-1]
Xp_bar, Yp_bar = X_prime.mean(), Y_prime.mean()
Sxx_p = np.sum((X_prime-Xp_bar)**2)
Sxy_p = np.sum((X_prime-Xp_bar)*(Y_prime-Yp_bar))
b1_co = Sxy_p / Sxx_p
b0_co_transformed = Yp_bar - b1_co*Xp_bar
b0_co = b0_co_transformed / (1 - rho_hat)
print(f"Cochrane-Orcutt: b0={b0_co:.3f}, b1={b1_co:.3f}")
```

```python
# statsmodels: Durbin-Watson is built into OLS results; GLSAR implements Cochrane-Orcutt-style iteration
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

X_sm = sm.add_constant(t)
model = sm.OLS(Y, X_sm).fit()
print("Durbin-Watson (statsmodels):", durbin_watson(model.resid))

# GLSAR: iteratively estimates rho and refits (generalizes Cochrane-Orcutt)
glsar_model = sm.GLSAR(Y, X_sm, rho=1)
glsar_results = glsar_model.iterative_fit(maxiter=10)
print("GLSAR coefficients:", glsar_results.params)
print("Estimated rho:", glsar_model.rho)
```

---

## Interview Question Bank — Chapter 12

**Conceptual:**
1. Why do OLS coefficient estimates remain unbiased under autocorrelated errors, while the standard errors become invalid?
2. What does a Durbin-Watson statistic near 0, near 2, and near 4 each indicate?
3. Why is autocorrelation not purely a "time series" concern — where else might it silently show up in ML pipelines?

**Derivation:**
4. Explain, conceptually, why the Cochrane-Orcutt transformation $Y_t-\hat\rho Y_{t-1}$ removes the AR(1) structure from the error term.
5. Derive why positive autocorrelation tends to make the classical $s^2(b_1)=MSE/S_{XX}$ formula understate the true variance of $b_1$.

**ML/Statistics:**
6. Compare the Cochrane-Orcutt and Hildreth-Lu procedures — what's the key methodological difference, and which is more computationally expensive?
7. Connect first-differencing for autocorrelation to a technique you'd use in modern time-series forecasting (e.g., ARIMA preprocessing).
8. Why doesn't correcting for autocorrelation guarantee a "more accurate" point estimate in any single small dataset, even though it improves inference in general?

**Coding:**
9. Implement the Durbin-Watson statistic and an estimate of ρ from OLS residuals, from scratch in NumPy.
10. Implement the Cochrane-Orcutt procedure from scratch, including the intercept back-transformation.

**Traps:**
11. "Autocorrelated errors bias my coefficient estimates." — correct this misconception precisely.
12. "My Durbin-Watson statistic is 1.9, so there's definitely no autocorrelation of any kind in my data." — what important limitation of the test does this overlook (hint: what specific structure does DW test for)?
13. Someone applies Cochrane-Orcutt and finds their corrected estimate is actually farther from a known true value than plain OLS in one specific dataset. Does this mean the correction failed?

---

*This file covers Kutner Ch. 12 — the consequences of autocorrelated errors for standard error validity (not bias), the AR(1) error model, the Durbin-Watson test worked by hand (with a residual pattern showing clear positive-autocorrelation runs), and remedial measures including Cochrane-Orcutt (worked in full), first differencing, and Hildreth-Lu. This is the last of the "classical assumption-violation" chapters — Chapter 13 (nonlinear regression / neural networks) and Chapter 14 (logistic, Poisson, and generalized linear models) are next, with Chapter 14 in particular being extremely high-value for ML interviews given how directly it connects to classification and count-data modeling.*
