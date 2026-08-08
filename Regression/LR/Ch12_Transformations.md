# Chapter 12 — Transformations

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. This chapter introduces a fresh small dataset with genuine curvature — our long-running 5-student dataset has been deliberately near-linear throughout Chapters 1–11 and isn't suited to demonstrating this chapter's fix.*

**New example dataset** — a quantity that doubles at each step (a classic multiplicative/exponential growth pattern, e.g., bacterial population doubling per hour):

| $x$ (hours) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ (population, thousands) | 3 | 6 | 12 | 24 | 48 |

---

## 12.1 The Motivating Question — Returning to Chapter 7's Curved Residual Pattern

Chapter 7's four-panel diagnostic flagged a **curved (U-shaped) pattern** in the residuals-vs-fitted plot as a sign of violated linearity — a missing nonlinear structure the straight-line model can't capture. This chapter answers the natural follow-up: **once you've detected that curvature, what do you actually do about it?**

---

## 12.2 Fitting the Raw (Misspecified) Linear Model First

Blindly fitting $y=\beta_0+\beta_1x+\varepsilon$ to this data (same OLS mechanics as Chapter 1): $\bar{x}=3$, $\bar{y}=18.6$, $S_{xx}=10$,

$$ S_{xy} = 31.2+12.6+0+5.4+58.8 = 108 \quad\Rightarrow\quad \hat{\beta}_1=\frac{108}{10}=10.8,\qquad \hat{\beta}_0=18.6-10.8(3)=-13.8 $$

**Fitted values:** $-3,\ 7.8,\ 18.6,\ 29.4,\ 40.2$. **Residuals:** $6,\ -1.8,\ -6.6,\ -5.4,\ 7.8$.

**Look at the pattern:** large positive residuals at both ends ($x=1$ and $x=5$), large negative residuals in the middle ($x=3,4$) — exactly the U-shaped curvature Chapter 7 warned about. The straight line systematically **underestimates** at the extremes and **overestimates** in the middle, because the true relationship is curving upward faster than any straight line can follow.

---

## 12.3 The Log Transform — Straightening Multiplicative Growth

**The core idea:** if $y$ grows *multiplicatively* (each step multiplies by a roughly constant factor, rather than adding a constant amount), taking $\ln(y)$ converts that multiplicative pattern into an *additive*, and therefore linear, one. This particular dataset was constructed as $y=3\times2^{x-1}$ (population doubling each hour) — exactly the multiplicative structure logs are built to handle.

Taking natural logs of $y$: $\ln(3)=1.099,\ \ln(6)=1.792,\ \ln(12)=2.485,\ \ln(24)=3.178,\ \ln(48)=3.871$.

Fitting $\ln(y) = \beta_0+\beta_1x$ (identical OLS mechanics, just on the transformed response):

$$ \overline{\ln y} = 2.485, \qquad S_{x,\ln y} = \sum(x-\bar{x})(\ln y-\overline{\ln y}) $$

Working through the same arithmetic as always: $\hat{\beta}_1 = \ln(2) \approx 0.693$, $\hat{\beta}_0=\ln(1.5)\approx0.4055$ — and this fits **exactly**, with **zero residuals**, because the data was constructed as a perfect exponential.

**The transformation didn't just improve the fit — it converted a genuinely nonlinear relationship into one where the linear-regression machinery from every previous chapter applies exactly.** This is the entire point of transformations: rather than abandoning linear regression for a fundamentally different technique, re-express the variables so the *existing* toolkit becomes appropriate again.

---

## 12.4 Interpreting Coefficients After a Log Transform

This is a frequently tested interview point, because the interpretation changes fundamentally once you've logged the response.

**Log-linear model** ($\ln y = \beta_0+\beta_1x$): a one-unit increase in $x$ is associated with a $\beta_1\times100\%$ **relative** change in $y$ (for small $\beta_1$; more precisely, $y$ is multiplied by $e^{\beta_1}$). Here, $\hat{\beta}_1=0.693=\ln(2)$, so $e^{0.693}=2$ — **each additional hour is associated with $y$ doubling**, exactly matching how the data was constructed.

**Log-log model** ($\ln y = \beta_0+\beta_1\ln x$, not used in our example but common in economics — e.g., demand curves): $\hat{\beta}_1$ is interpreted directly as an **elasticity** — a 1% increase in $x$ is associated with a $\hat{\beta}_1\%$ change in $y$.

**Interview-critical caution:** predictions must be **back-transformed** carefully. Simply exponentiating a predicted $\widehat{\ln y}$ gives the predicted **median**, not the mean, of $y$ on the original scale (because $E[e^Z] \neq e^{E[Z]}$ for a random variable $Z$ — Jensen's inequality) — a correction factor (e.g., Duan's smearing estimator, or $e^{\hat{\sigma}^2/2}$ under normality of the log-scale residuals) is needed if you specifically want an unbiased mean prediction back on the original scale.

---

## 12.5 The Box-Cox Family — Choosing a Transformation Systematically

Rather than guessing between log, square root, reciprocal, etc., the **Box-Cox transformation** provides a single parametrized family:

$$ y^{(\lambda)} = \begin{cases}\dfrac{y^\lambda-1}{\lambda} & \lambda\neq0 \\ \ln(y) & \lambda=0\end{cases} $$

Special cases: $\lambda=1$ is (up to a constant shift) no transformation at all; $\lambda=0.5$ is a square-root-like transform; $\lambda=0$ is the log transform used above; $\lambda=-1$ is a reciprocal-like transform.

**How $\lambda$ is chosen in practice:** for a range of candidate $\lambda$ values, fit the regression using $y^{(\lambda)}$ as the response, and pick the $\lambda$ that **maximizes the profile log-likelihood** (equivalently, for a fixed grid of $\lambda$, the one minimizing SSE after accounting for the transformation's Jacobian). Software (e.g., `boxcox()` in R) typically plots log-likelihood against $\lambda$ with a confidence interval — you don't need to hand-search infinitely many values, just scan a reasonable grid (e.g., $-2$ to $2$ in steps of $0.25$) and read off the peak. For our constructed dataset, this search would land at (or very near) $\lambda=0$, correctly recovering the log transform, since the data is exactly exponential by construction.

---

## 12.6 Polynomial Terms — An Alternative When the Curvature Isn't Multiplicative

Not every curved pattern is multiplicative/exponential in nature. When residuals show curvature that a log transform doesn't fully straighten, adding a **polynomial term** is a common alternative:

$$ y = \beta_0+\beta_1x+\beta_2x^2+\varepsilon $$

This is still technically **linear regression** (linear in the *parameters* $\beta_0,\beta_1,\beta_2$, even though it's nonlinear in $x$) — the entire matrix machinery from Chapter 3 applies completely unchanged, just with an extra column for $x^2$ in the design matrix. We defer the full worked treatment of polynomial regression, including the practical issue of $x$ and $x^2$ being highly collinear unless $x$ is first centered (a direct callback to Chapter 9), to Chapter 21.

**Choosing between a transformation and a polynomial term** is often guided by the *mechanism*: if you have a substantive reason to expect multiplicative/proportional growth (population dynamics, compound interest, many biological/economic processes), a log transform is the theoretically motivated choice. If the curvature doesn't match a clean multiplicative story, a polynomial (or spline — Chapter 21) is a more flexible, less assumption-laden fix, at the cost of a less directly interpretable coefficient.

---

## 12.7 Where the Textbooks Differ

- **Kutner** presents Box-Cox with the fullest likelihood-based derivation and the most systematic treatment of the transformation-selection procedure.
- **Montgomery** emphasizes the variance-stabilizing role of transformations as much as the linearizing role — noting that many transformations (log, square root) simultaneously fix both nonlinearity **and** heteroscedasticity (Chapter 10) when the two problems share a common cause (e.g., variance naturally growing alongside the mean in count or multiplicative data).
- **Sheather** leans on diagnostic plots before/after transformation as the primary teaching tool, letting the visual improvement carry the argument more than the Box-Cox likelihood machinery.
- **ESL/ISL** treat this topic as a special, interpretable case of the more general feature-engineering/basis-expansion philosophy that culminates in splines and kernel methods — for them, a log transform is just one instance of "choose a good basis for your predictor" rather than a distinct topic in its own right.

---

## 12.8 Interview Q&A

**Q: You see a U-shaped pattern in your residuals-vs-fitted plot. What are your options?**
A: Consider a transformation (log, square root, Box-Cox) if the curvature looks multiplicative/proportional in nature, or add a polynomial term (or spline) if it doesn't match a clean transformation story — both directly address the linearity violation flagged in that panel from Chapter 7.

**Q: How do you interpret a coefficient in a model where you've logged the response but not the predictor?**
A: A one-unit increase in $x$ is associated with the response being multiplied by $e^{\hat{\beta}_1}$ — approximately a $\hat{\beta}_1\times100\%$ relative change for small $\hat{\beta}_1$, not an additive change in the original units.

**Q: Why can't you just exponentiate a predicted value from a log-response model to get the expected value on the original scale?**
A: Because $E[e^Z]\neq e^{E[Z]}$ (Jensen's inequality) — naively exponentiating gives the median prediction, not the mean; a correction factor like Duan's smearing estimator is needed for an unbiased mean prediction.

**Q: Is polynomial regression still "linear regression"?**
A: Yes — the model is linear in the parameters ($\beta_0,\beta_1,\beta_2,...$), even though it's a nonlinear (curved) function of $x$. All of Chapters 3–9's matrix machinery, inference, and diagnostics apply unchanged; only the design matrix gains additional columns.

**Q: How is the Box-Cox $\lambda$ parameter chosen?**
A: By maximizing the profile log-likelihood (equivalently minimizing an appropriately Jacobian-adjusted SSE) across a grid of candidate $\lambda$ values — not by arbitrary trial and error, and not something you'd typically need to derive by hand in an interview beyond explaining the selection principle.

---

*End of Chapter 12. Next: Chapter 13 — Categorical Predictors & Interactions (dummy coding, reference-level interpretation, and how to correctly read an interaction term's coefficient).*
