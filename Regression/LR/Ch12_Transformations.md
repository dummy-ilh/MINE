# Chapter 12 — Transformations

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations. This chapter introduces a fresh small dataset with genuine curvature — our long-running 5-student dataset has been deliberately near-linear throughout Chapters 1–11 and isn't suited to demonstrating this chapter's fix.*

**New example dataset** — a quantity that doubles at each step (a classic multiplicative/exponential growth pattern, e.g., bacterial population doubling per hour):

| $x$ (hours) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ (population, thousands) | 3 | 6 | 12 | 24 | 48 |

---

## 12.1 The Motivating Question — Returning to Chapter 7's Curved Residual Pattern

Chapter 7's four-panel diagnostic flagged a **curved (U-shaped) pattern** in the residuals-vs-fitted plot as a sign of violated linearity — a missing nonlinear structure the straight-line model can't capture. This chapter answers the natural follow-up: **once you've detected that curvature, what do you actually do about it?**

**Plain-language framing before anything else:** picture a population of bacteria that doubles every hour. Hour 1: 3,000. Hour 2: 6,000. Hour 3: 12,000. Hour 5: 48,000. If you try to draw a *straight* line through those points, you're doomed from the start — the growth is accelerating, and no straight line can bend to keep up. This chapter is about a clever trick: instead of giving up on straight lines, you can sometimes *reshape the data itself* so that a straight line becomes the right tool again. It's the mathematical equivalent of "if a photo looks warped, don't throw out your ruler — straighten the photo first."

---

## 12.2 Fitting the Raw (Misspecified) Linear Model First

Blindly fitting $y=\beta_0+\beta_1x+\varepsilon$ to this data (same OLS mechanics as Chapter 1): $\bar{x}=3$, $\bar{y}=18.6$, $S_{xx}=10$,

$$ S_{xy} = 31.2+12.6+0+5.4+58.8 = 108 \quad\Rightarrow\quad \hat{\beta}_1=\frac{108}{10}=10.8,\qquad \hat{\beta}_0=18.6-10.8(3)=-13.8 $$

**Fitted values:** $-3,\ 7.8,\ 18.6,\ 29.4,\ 40.2$. **Residuals:** $6,\ -1.8,\ -6.6,\ -5.4,\ 7.8$.

**Look at the pattern:** large positive residuals at both ends ($x=1$ and $x=5$), large negative residuals in the middle ($x=3,4$) — exactly the U-shaped curvature Chapter 7 warned about. The straight line systematically **underestimates** at the extremes and **overestimates** in the middle, because the true relationship is curving upward faster than any straight line can follow.

**Walking through what this means in plain words:** notice the straight-line fit even predicts a *negative* population at hour 1 (-3, which is nonsensical for an actual bacteria count) — a strong tell that a straight line is fundamentally the wrong shape here, not just a slightly-off fit. The pattern in the residuals (big miss, smaller miss, smaller miss, bigger miss, biggest miss) is the model's way of "confessing" that it's trying to force a curve into a straight shape and failing predictably at both ends.

---

## 12.3 The Log Transform — Straightening Multiplicative Growth

**The core idea:** if $y$ grows *multiplicatively* (each step multiplies by a roughly constant factor, rather than adding a constant amount), taking $\ln(y)$ converts that multiplicative pattern into an *additive*, and therefore linear, one. This particular dataset was constructed as $y=3\times2^{x-1}$ (population doubling each hour) — exactly the multiplicative structure logs are built to handle.

**Plain-English version of the core trick, before the log rules:** logarithms have a special property — they turn *multiplication* into *addition*. "Doubling every hour" is a multiplication story ($\times2$ each step). But $\ln(\text{doubling}) = \ln(2)$, a fixed number you *add* each step instead of multiplying by. So on the log scale, "doubling every hour" simply becomes "add the same fixed amount every hour" — and "add the same fixed amount every step" is *exactly* what a straight line describes. That's the whole trick: logs convert a multiplicative growth story into an additive one, and additive stories are what straight lines are built to capture.

Taking natural logs of $y$: $\ln(3)=1.099,\ \ln(6)=1.792,\ \ln(12)=2.485,\ \ln(24)=3.178,\ \ln(48)=3.871$.

Fitting $\ln(y) = \beta_0+\beta_1x$ (identical OLS mechanics, just on the transformed response):

$$ \overline{\ln y} = 2.485, \qquad S_{x,\ln y} = \sum(x-\bar{x})(\ln y-\overline{\ln y}) $$

Working through the same arithmetic as always: $\hat{\beta}_1 = \ln(2) \approx 0.693$, $\hat{\beta}_0=\ln(1.5)\approx0.4055$ — and this fits **exactly**, with **zero residuals**, because the data was constructed as a perfect exponential.

**The transformation didn't just improve the fit — it converted a genuinely nonlinear relationship into one where the linear-regression machinery from every previous chapter applies exactly.** This is the entire point of transformations: rather than abandoning linear regression for a fundamentally different technique, re-express the variables so the *existing* toolkit becomes appropriate again.

**In plain words, why "zero residuals" happened here:** we deliberately built this example data as a perfect doubling pattern, so once we look at it through the "log lens," the underlying straight-line relationship is exact — no noise at all. Real data will almost never fit this perfectly, but the mechanics are identical: the log transform is doing genuine, honest work here, not a trick specific to this toy example.

---

## 12.4 Interpreting Coefficients After a Log Transform

This is a frequently tested interview point, because the interpretation changes fundamentally once you've logged the response.

**Log-linear model** ($\ln y = \beta_0+\beta_1x$): a one-unit increase in $x$ is associated with a $\beta_1\times100\%$ **relative** change in $y$ (for small $\beta_1$; more precisely, $y$ is multiplied by $e^{\beta_1}$). Here, $\hat{\beta}_1=0.693=\ln(2)$, so $e^{0.693}=2$ — **each additional hour is associated with $y$ doubling**, exactly matching how the data was constructed.

**Why this coefficient-reading rule makes sense, in plain words:** once you've logged $y$, the coefficient $\hat{\beta}_1$ no longer means "add this many units of $y$ per unit of $x$" (the normal, un-logged reading) — it means "multiply $y$ by $e^{\hat\beta_1}$ per unit of $x$." Here $e^{0.693}\approx2$, which is just the mathematical way of confirming what we already knew by construction: the population doubles every hour. This is exactly why log-linear models are so popular for growth-rate stories — the coefficient directly tells you "percent change per unit of $x$," a number people intuitively understand (like "5% growth per year").

**Log-log model** ($\ln y = \beta_0+\beta_1\ln x$, not used in our example but common in economics — e.g., demand curves): $\hat{\beta}_1$ is interpreted directly as an **elasticity** — a 1% increase in $x$ is associated with a $\hat{\beta}_1\%$ change in $y$.

**In plain words:** if both $x$ and $y$ are logged, the coefficient becomes a clean "percent-for-percent" story — for example, if $\hat\beta_1=-0.5$ in a demand curve, that means "a 1% price increase is associated with about a 0.5% drop in quantity demanded." This "percent change causes percent change" framing is exactly what economists mean by elasticity.

**Interview-critical caution:** predictions must be **back-transformed** carefully. Simply exponentiating a predicted $\widehat{\ln y}$ gives the predicted **median**, not the mean, of $y$ on the original scale (because $E[e^Z] \neq e^{E[Z]}$ for a random variable $Z$ — Jensen's inequality) — a correction factor (e.g., Duan's smearing estimator, or $e^{\hat{\sigma}^2/2}$ under normality of the log-scale residuals) is needed if you specifically want an unbiased mean prediction back on the original scale.

**Plain-language version of this subtle trap:** it's tempting to think "I'll just predict on the log scale, then undo the log with $e^{(\cdot)}$, and I'm done." That instinct is *almost* right, but not quite: undoing the log this simple way actually gives you the *typical middle value* (the median), not the *average* (the mean) — and those two aren't the same once you're back on the original scale, because logging and un-logging aren't perfectly symmetric when there's randomness involved. If your goal really is the average prediction, you need a small correction factor on top of the simple undo-the-log step. This is a genuinely common mistake, which is exactly why interviewers like asking about it.

---

## 12.5 The Box-Cox Family — Choosing a Transformation Systematically

Rather than guessing between log, square root, reciprocal, etc., the **Box-Cox transformation** provides a single parametrized family:

$$ y^{(\lambda)} = \begin{cases}\dfrac{y^\lambda-1}{\lambda} & \lambda\neq0 \\ \ln(y) & \lambda=0\end{cases} $$

Special cases: $\lambda=1$ is (up to a constant shift) no transformation at all; $\lambda=0.5$ is a square-root-like transform; $\lambda=0$ is the log transform used above; $\lambda=-1$ is a reciprocal-like transform.

**Plain-English framing before the formula:** instead of manually trying "maybe log will work... maybe square root... maybe something else," Box-Cox packages *every* common transformation into one adjustable dial, controlled by a single number $\lambda$. Turn the dial to $\lambda=0$ and you get the log transform. Turn it to $\lambda=0.5$ and you get something like a square root. Turn it to $\lambda=1$ and you're back to the untransformed original data. Rather than guessing which specific transform to try, you can let the data itself tell you where on this dial to land.

**How $\lambda$ is chosen in practice:** for a range of candidate $\lambda$ values, fit the regression using $y^{(\lambda)}$ as the response, and pick the $\lambda$ that **maximizes the profile log-likelihood** (equivalently, for a fixed grid of $\lambda$, the one minimizing SSE after accounting for the transformation's Jacobian). Software (e.g., `boxcox()` in R) typically plots log-likelihood against $\lambda$ with a confidence interval — you don't need to hand-search infinitely many values, just scan a reasonable grid (e.g., $-2$ to $2$ in steps of $0.25$) and read off the peak. For our constructed dataset, this search would land at (or very near) $\lambda=0$, correctly recovering the log transform, since the data is exactly exponential by construction.

**In plain words, how the "search" actually works:** the software tries a bunch of dial settings (say, $\lambda=-2,-1.75,-1.5,...,2$), refits the regression at each setting, and checks "how well does the model fit at this setting." It then simply picks whichever setting fit best. For our doubling-population data, that search would correctly discover that $\lambda=0$ (the log transform) fits best — the data was built as a pure exponential, so it makes sense the automatic search rediscovers exactly the transform we already reasoned our way to by hand.

---

## 12.6 Polynomial Terms — An Alternative When the Curvature Isn't Multiplicative

Not every curved pattern is multiplicative/exponential in nature. When residuals show curvature that a log transform doesn't fully straighten, adding a **polynomial term** is a common alternative:

$$ y = \beta_0+\beta_1x+\beta_2x^2+\varepsilon $$

This is still technically **linear regression** (linear in the *parameters* $\beta_0,\beta_1,\beta_2$, even though it's nonlinear in $x$) — the entire matrix machinery from Chapter 3 applies completely unchanged, just with an extra column for $x^2$ in the design matrix. We defer the full worked treatment of polynomial regression, including the practical issue of $x$ and $x^2$ being highly collinear unless $x$ is first centered (a direct callback to Chapter 9), to Chapter 21.

**Plain-English framing of the "still linear" idea, which trips people up:** it sounds contradictory to call $y=\beta_0+\beta_1x+\beta_2x^2$ "linear regression" when there's clearly a squared term making a curve. The resolution: "linear" in this context refers to the *coefficients* $\beta_0,\beta_1,\beta_2$, not to the shape of the curve you can draw. As far as the math is concerned, $x^2$ is just another input column, no different in kind from $x$ itself — the model is still a straight-line combination of its inputs, it just so happens that one of those inputs is $x$ squared instead of plain $x$. This is why all the Chapter 3 machinery (matrices, hat matrix, everything) works completely unchanged.

**Choosing between a transformation and a polynomial term** is often guided by the *mechanism*: if you have a substantive reason to expect multiplicative/proportional growth (population dynamics, compound interest, many biological/economic processes), a log transform is the theoretically motivated choice. If the curvature doesn't match a clean multiplicative story, a polynomial (or spline — Chapter 21) is a more flexible, less assumption-laden fix, at the cost of a less directly interpretable coefficient.

**Simplest possible decision rule:** if you can tell a believable "this grows by a percentage each step" story (bacteria, compound interest, viral spread), reach for a log transform first — it gives you clean, interpretable coefficients. If the curve doesn't fit any tidy story like that, a polynomial term is a more flexible fallback — it'll bend to match almost any curve shape, but the coefficient itself won't have a clean, real-world meaning the way "doubling every hour" does.

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
*(Simple version: if the growth "feels like a percentage story," try a log; otherwise, let a polynomial term bend to fit the curve.)*

**Q: How do you interpret a coefficient in a model where you've logged the response but not the predictor?**
A: A one-unit increase in $x$ is associated with the response being multiplied by $e^{\hat{\beta}_1}$ — approximately a $\hat{\beta}_1\times100\%$ relative change for small $\hat{\beta}_1$, not an additive change in the original units.
*(Simple version: the coefficient tells you a percentage change, not an amount change.)*

**Q: Why can't you just exponentiate a predicted value from a log-response model to get the expected value on the original scale?**
A: Because $E[e^Z]\neq e^{E[Z]}$ (Jensen's inequality) — naively exponentiating gives the median prediction, not the mean; a correction factor like Duan's smearing estimator is needed for an unbiased mean prediction.
*(Simple version: undoing the log the simple way gives you the "typical middle" value, not the true average — you need one extra correction step to get the real average back.)*

**Q: Is polynomial regression still "linear regression"?**
A: Yes — the model is linear in the parameters ($\beta_0,\beta_1,\beta_2,...$), even though it's a nonlinear (curved) function of $x$. All of Chapters 3–9's matrix machinery, inference, and diagnostics apply unchanged; only the design matrix gains additional columns.
*(Simple version: "linear" describes the coefficients, not the shape of the curve you end up drawing.)*

**Q: How is the Box-Cox $\lambda$ parameter chosen?**
A: By maximizing the profile log-likelihood (equivalently minimizing an appropriately Jacobian-adjusted SSE) across a grid of candidate $\lambda$ values — not by arbitrary trial and error, and not something you'd typically need to derive by hand in an interview beyond explaining the selection principle.
*(Simple version: try a bunch of dial settings, keep whichever one fits the data best.)*

---

*End of Chapter 12. Next: Chapter 13 — Categorical Predictors & Interactions (dummy coding, reference-level interpretation, and how to correctly read an interaction term's coefficient).*
