# Chapter 22 — From Linear to Logistic (The GLM Bridge)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. This chapter is intentionally a bridge, not a full treatment — for the complete logistic regression interview curriculum (odds ratios, deviance, full worked MLE, classification metrics), see the separate logistic regression curriculum in this study plan. Here, the goal is narrower and specific: show precisely how everything built in Chapters 1–21 generalizes into the GLM framework, with logistic regression as the running example.*

**New example dataset** — a binary outcome (pass/fail) with some genuine overlap (not perfectly separable, which matters — see §22.5):

| $x$ (hours) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ (pass=1) | 0 | 1 | 0 | 1 | 1 |

---

## 22.1 The Motivating Question — Why Not Just Use OLS on a 0/1 Outcome?

Nothing stops you from literally fitting $\hat{y}=\beta_0+\beta_1x$ with $y\in\{0,1\}$ — this is called the **linear probability model**, and it's occasionally used, but it has three specific, well-known problems, each traceable to an assumption from earlier chapters:

1. **Predictions can fall outside $[0,1]$** — a fitted "probability" of $1.3$ or $-0.2$ is nonsensical, and nothing in the OLS machinery prevents it.
2. **Heteroscedasticity is guaranteed, not just possible** — for a Bernoulli outcome, $\text{Var}(y_i)=p_i(1-p_i)$, which mechanically depends on the mean itself. This directly violates the constant-variance assumption from Chapter 1's LINE framework, and it's not a modeling accident to be diagnosed away (Chapter 10) — it's baked into the nature of a binary outcome.
3. **The true relationship between $x$ and $P(y=1)$ is essentially never linear across its full range** — probability must flatten out near 0 and 1 (you can't keep decreasing below 0% or increasing above 100% no matter how extreme $x$ gets), producing an inherently S-shaped, not straight-line, relationship.

**The GLM framework fixes all three simultaneously**, via a single, elegant generalization.

---

## 22.2 The Generalized Linear Model — Three Components

Every GLM (including ordinary linear regression as a special case) is built from three pieces:

1. **Random component**: the assumed distribution of $y$ given $x$ — Gaussian for ordinary linear regression, **Binomial** for logistic regression, Poisson for count data, etc.
2. **Systematic component**: the linear predictor $\eta=\mathbf{x}^T\boldsymbol{\beta}$ — this is the **only** part that stays literally identical to every prior chapter; it's still just $\beta_0+\beta_1x_1+...$
3. **Link function** $g(\cdot)$: connects the mean of $y$ to the linear predictor: $g(\mu)=\eta$, i.e., $\mu=g^{-1}(\eta)$.

| Model | Random component | Link function $g(\mu)$ | Inverse link |
|---|---|---|---|
| Ordinary linear regression | Gaussian | Identity: $g(\mu)=\mu$ | $\mu=\eta$ |
| Logistic regression | Binomial | Logit: $g(\mu)=\ln\left(\frac{\mu}{1-\mu}\right)$ | $\mu=\frac{1}{1+e^{-\eta}}$ |
| Poisson regression | Poisson | Log: $g(\mu)=\ln(\mu)$ | $\mu=e^\eta$ |

**Ordinary linear regression is simply the special case where the link is the identity function and the random component is Gaussian** — every single chapter of this curriculum has, in GLM terms, been working within one specific corner of this broader framework the entire time.

---

## 22.3 Why the Logit Link, Specifically

The **logit** (log-odds) function, $\ln\left(\frac{p}{1-p}\right)$, maps probabilities in $(0,1)$ onto the entire real line $(-\infty,\infty)$ — exactly matching the range of an unconstrained linear predictor $\eta=\beta_0+\beta_1x$. This single property directly solves problem 1 from §22.1: no matter what value $\eta$ takes, transforming it back through the inverse logit (the **sigmoid function**, $\mu=1/(1+e^{-\eta})$) is *guaranteed* to land in $(0,1)$ — an out-of-range predicted probability becomes structurally impossible, not just unlikely.

**Coefficient interpretation**, directly parallel to Chapter 12's log-linear interpretation: $\hat{\beta}_1$ is the change in **log-odds** per one-unit increase in $x$; $e^{\hat{\beta}_1}$ is the multiplicative change in the **odds** — this is the same "exponentiate to interpret" pattern from Chapter 12, §12.4, just applied to odds rather than to the response itself.

---

## 22.4 Fitting a GLM — Maximum Likelihood via IRLS, a Direct Callback to Chapters 10 and 19

There's no closed-form solution for $\hat{\boldsymbol{\beta}}$ under logistic regression (unlike OLS's clean $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$) — it's fit by **maximum likelihood**, and the standard algorithm turns out to be **exactly** the Iteratively Reweighted Least Squares (IRLS) procedure first introduced in Chapter 20 for robust regression, and structurally identical to the WLS/GLS machinery from Chapters 10 and 19. At each iteration:

1. Compute current fitted probabilities $\hat{p}_i = 1/(1+e^{-\hat{\eta}_i})$ from the current coefficient estimates.
2. Compute **weights** $w_i = \hat{p}_i(1-\hat{p}_i)$ — the Bernoulli variance at the current fitted probability (directly reflecting problem 2 from §22.1: this weighting explicitly accounts for the mean-dependent variance that plain OLS would ignore).
3. Compute a **working response** $z_i = \hat{\eta}_i + \dfrac{y_i-\hat{p}_i}{\hat{p}_i(1-\hat{p}_i)}$ — a linearized stand-in for the response on the scale of the linear predictor.
4. Run ordinary **WLS** (Chapter 10's exact machinery) of $z$ on $x$ using weights $w$, to get updated $\hat{\boldsymbol{\beta}}$.
5. Repeat until $\hat{\boldsymbol{\beta}}$ stops changing.

**Worked example — one IRLS iteration**, starting from $\hat{\beta}_0=\hat{\beta}_1=0$ (so $\hat{\eta}_i=0$ for every point, giving $\hat{p}_i=0.5$ for every point):

**Weights:** $w_i = 0.5(1-0.5)=0.25$ for all 5 points (equal, since all starting probabilities are 0.5).

**Working responses:** $z_i = 0 + \dfrac{y_i-0.5}{0.25} = 4(y_i-0.5)$. With $y=0,1,0,1,1$: $z=-2,\ 2,\ -2,\ 2,\ 2$.

Since all weights are equal, this first-iteration WLS step is just ordinary OLS of $z$ on $x$ (Chapter 1's exact mechanics): $\bar{x}=3$, $\bar{z}=0.4$, $S_{xx}=10$:

$$ S_{xz} = \sum(x-\bar{x})(z-\bar{z}) = 4.8-1.6+0+1.6+3.2 = 8.0 $$

$$ \hat{\beta}_{1,new} = 8.0/10 = 0.8, \qquad \hat{\beta}_{0,new} = 0.4-0.8(3) = -2.0 $$

**After one IRLS iteration:** $\hat{\beta}_0\approx-2.0,\ \hat{\beta}_1\approx0.8$. Software would continue this process — recomputing $\hat{p}_i$ from these new coefficients, updating weights and working responses, and refitting — for several more iterations until convergence (typically only a handful of iterations for well-behaved data). **The core insight worth carrying forward from this entire curriculum:** logistic regression fitting isn't a fundamentally different algorithm from everything built in Chapters 1–21 — it's **the same WLS machinery from Chapter 10, applied repeatedly**, with the twist that the weights themselves depend on the current parameter estimates (since Bernoulli variance depends on the mean), requiring iteration rather than Chapter 10's single-shot WLS where weights were assumed known in advance.

---

## 22.5 A Danger Specific to This Setting — Perfect Separation

If the classes were **perfectly** separable by $x$ (e.g., every pass had $x\geq3$ and every fail had $x<3$, with no overlap at all), the MLE for $\hat{\beta}_1$ would **not converge to a finite value** — the likelihood keeps increasing as $\hat{\beta}_1\to\infty}$, pushing predicted probabilities toward exactly 0 or 1 for every point without limit. This is why the example dataset above was deliberately constructed with overlap ($x=2$ passes despite being lower than the failing $x=3$) — a genuinely important practical warning: **near-perfect separation in a real logistic regression dataset is a red flag for numerically unstable, enormous, and essentially meaningless coefficient estimates**, not evidence of an unusually strong effect.

---

## 22.6 Where the Textbooks Differ

- **Kutner** doesn't cover GLMs at all — it's scoped entirely to (Gaussian) linear regression, making this chapter's bridge content the point where this synthesized curriculum extends beyond Kutner's coverage.
- **Montgomery** covers GLMs briefly, mainly logistic and Poisson regression, with a practical/applied framing similar to its linear regression treatment.
- **ESL/ISL** cover logistic regression thoroughly as a core topic in its own right (not just a bridge chapter), including its relationship to linear discriminant analysis and other classification methods — a fuller treatment than this bridge chapter attempts.
- **Sheather** doesn't extend into GLMs either, remaining focused on linear regression specifically.
- **This chapter's IRLS-as-generalized-WLS framing** is drawn primarily from the statistical (rather than ML) GLM literature (e.g., McCullagh & Nelder's classic GLM text, not one of this curriculum's core four sources, but the standard reference for this specific connection) — included here because it's the cleanest possible bridge back to Chapters 10 and 19's material.

---

## 22.7 Interview Q&A

**Q: Why can't you just use ordinary linear regression on a 0/1 outcome?**
A: Predictions aren't constrained to $[0,1]$, the variance is mechanically heteroscedastic since Bernoulli variance depends on the mean ($p(1-p)$), and the true relationship between $x$ and probability is inherently S-shaped rather than linear across its full range.

**Q: What are the three components of a GLM?**
A: A random component (the assumed distribution family, e.g., Binomial for logistic regression), a systematic component (the linear predictor $\eta=\mathbf{x}^T\boldsymbol{\beta}$, unchanged from ordinary linear regression), and a link function connecting the mean of the response to the linear predictor.

**Q: How does the algorithm used to fit logistic regression relate to weighted least squares?**
A: Logistic regression is fit via IRLS (Iteratively Reweighted Least Squares) — at each step, it computes weights from the current fitted probabilities ($p_i(1-p_i)$) and a linearized "working response," then runs ordinary WLS (Chapter 10) on that working response. It's the exact same WLS machinery, just iterated because the appropriate weights depend on parameters that are themselves being estimated.

**Q: What is "perfect separation," and why is it dangerous in logistic regression?**
A: When a predictor (or combination of predictors) perfectly distinguishes the two outcome classes with no overlap, the maximum likelihood estimate for the corresponding coefficient diverges to infinity rather than converging to a finite value — producing numerically unstable, uninterpretable results rather than evidence of an unusually strong true effect.

**Q: How do you interpret a logistic regression coefficient?**
A: $\hat{\beta}_1$ is the change in log-odds of the outcome per one-unit increase in $x$; exponentiating it, $e^{\hat{\beta}_1}$, gives the multiplicative change in the odds — directly parallel to the log-linear-model interpretation from Chapter 12.

---

*End of Chapter 22. Next: Chapter 23 — Assumption Violations in Practice (integrated case studies combining multiple diagnostics from Chapters 7–11 on a single dataset, since real data rarely presents just one clean violation at a time).*
